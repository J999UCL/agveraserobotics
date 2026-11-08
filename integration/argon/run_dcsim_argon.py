"""Run the team's Argon agent (unchanged) against the dcsim robot.

Copy dcsim_adapter.py and this file into robot/sandbox/ of the team repo, start the simulator,
then from robot/sandbox/:

  DCSIM=localhost:5555 DCSIM_REPO=/path/to/AgentVerseBob2 python run_dcsim_argon.py --only clean_failure
  ... --brain modal            # optional: Argon on Modal (Gemini key stays in the Modal secret)
  ... --brain gemini           # default: Argon called from here (needs GEMINI_API_KEY)
  ... --brain none             # local autopilot only: escalations end in the safe fallback
  ... --brain gemini --model gemini-3.5-flash-lite   # a smaller model than the agent's default
  ... --mode argon             # Argon decides every step (default: local_first)

Traces: runs/dcsim/<mode>_<scenario>.json, camera frames in runs/dcsim/<mode>_<scenario>/
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from agent import Budget, run  # noqa: E402
from dcsim_adapter import DcsimRack  # noqa: E402
from policy import run_local_first  # noqa: E402
from scenarios import SCENARIOS, grade  # noqa: E402

OUT = os.path.join(HERE, "runs", "dcsim")


class NoBrain:
    def decide(self, prompt, images):
        raise RuntimeError("no Argon brain configured (--brain none)")


def make_brain(kind: str):
    if kind == "none":
        return NoBrain(), contextlib.nullcontext()
    if kind == "gemini":
        from agent import ArgonBrain
        return ArgonBrain(), contextlib.nullcontext()
    from modal_app import app, argon_decide

    class RemoteArgonBrain:
        def decide(self, prompt, images):
            r = argon_decide.remote(prompt, images, "LOW")
            return r["decision"], r["raw"]

    return RemoteArgonBrain(), app.run()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="")
    p.add_argument("--mode", default="local_first", choices=["local_first", "argon"])
    p.add_argument("--brain", default="gemini", choices=["gemini", "modal", "none"])
    p.add_argument("--server", default="R1-05-U20")
    p.add_argument("--model", default="", help="with --brain gemini: use this model instead of agent.MODEL")
    a = p.parse_args()
    if a.model:
        if a.brain != "gemini":
            p.error("--model needs --brain gemini (the Modal app has its own model setting)")
        import agent
        agent.MODEL = a.model
    names = a.only.split(",") if a.only else list(SCENARIOS)
    runner = run_local_first if a.mode == "local_first" else run
    brain, ctx = make_brain(a.brain)
    rows = []
    with ctx:
        for name in names:
            print(f"\n=== {name} ({a.mode}, dcsim robot, brain={a.brain}) ===", flush=True)
            t0 = time.time()
            w = DcsimRack(SCENARIOS[name][0](), server=a.server,
                          frames_dir=os.path.join(OUT, f"{a.mode}_{name}"), verbose=True)
            t = runner(w, brain, Budget())
            g = grade(name, w)
            sim_status = w.c.get_status()
            w.close()
            res = {"scenario": name, "mode": a.mode, "sim": "dcsim", "brain": a.brain, "grade": g,
                   "model_calls": t.model_calls, "local_decisions": t.local_decisions,
                   "escalations": t.escalations, "model_seconds": round(t.model_seconds, 1),
                   "fallback": t.fallback, "steps": t.steps, "final_world": w.snapshot(),
                   "sim_time_s": sim_status["time"], "wall_s": round(time.time() - t0, 1)}
            os.makedirs(OUT, exist_ok=True)
            with open(os.path.join(OUT, f"{a.mode}_{name}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, default=str)
            failed = [c for c, ok in g["checks"].items() if not ok]
            rows.append((name, "PASS" if g["passed"] else "FAIL", t.model_calls, t.local_decisions,
                         ", ".join(failed) or (t.fallback or ""), res["wall_s"]))
            print(f"=== {name}: {rows[-1][1]} {failed or ''} ({res['wall_s']} s)", flush=True)
    print(f"\nmode={a.mode} sim=dcsim brain={a.brain}")
    for r in rows:
        print(f"{r[0]:<18}{r[1]:<6} argon={r[2]:<3} local={r[3]:<3} {r[5]:>6}s  {r[4][:90]}")


if __name__ == "__main__":
    main()
