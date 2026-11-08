"""Run the agent against the team's MuJoCo scene, with Argon called on Modal.

The sim (physics, renders, local perception, local autopilot) runs on this machine;
only Argon's decisions go to Modal, where the Gemini key lives.

  uv run --with trimesh==5.2.0 python robot/sandbox/run_mujoco_argon.py --mode argon --only clean_failure
  uv run --with trimesh==5.2.0 python robot/sandbox/run_mujoco_argon.py            # local_first, all

Traces: robot/sandbox/runs/mujoco/<mode>_<scenario>.json, frames in .../<mode>_<scenario>/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from agent import Budget, run  # noqa: E402
from modal_app import app, argon_decide  # noqa: E402
from mujoco_adapter import MujocoRack  # noqa: E402
from policy import run_local_first  # noqa: E402
from scenarios import SCENARIOS, grade  # noqa: E402

OUT = os.path.join(HERE, "runs", "mujoco")


class RemoteArgonBrain:
    def __init__(self, thinking="LOW"):
        self.thinking = thinking

    def decide(self, prompt, images):
        r = argon_decide.remote(prompt, images, self.thinking)
        return r["decision"], r["raw"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="")
    p.add_argument("--mode", default="local_first", choices=["local_first", "argon"])
    a = p.parse_args()
    names = a.only.split(",") if a.only else list(SCENARIOS)
    runner = run_local_first if a.mode == "local_first" else run
    rows = []
    with app.run():
        for name in names:
            print(f"\n=== {name} ({a.mode}, MuJoCo) ===", flush=True)
            w = MujocoRack(SCENARIOS[name][0](), frames_dir=os.path.join(OUT, f"{a.mode}_{name}"))
            t = runner(w, RemoteArgonBrain(), Budget())
            g = grade(name, w)
            w.close()
            res = {"scenario": name, "mode": a.mode, "sim": "mujoco", "grade": g,
                   "model_calls": t.model_calls, "local_decisions": t.local_decisions,
                   "escalations": t.escalations, "model_seconds": round(t.model_seconds, 1),
                   "fallback": t.fallback, "steps": t.steps, "final_world": w.snapshot()}
            os.makedirs(OUT, exist_ok=True)
            with open(os.path.join(OUT, f"{a.mode}_{name}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            failed = [c for c, ok in g["checks"].items() if not ok]
            rows.append((name, "PASS" if g["passed"] else "FAIL", t.model_calls,
                         t.local_decisions, ", ".join(failed) or (t.fallback or "")))
            print(f"=== {name}: {rows[-1][1]} {failed or ''}", flush=True)
    print(f"\nmode={a.mode} sim=mujoco")
    for r in rows:
        print(f"{r[0]:<18}{r[1]:<6} argon={r[2]:<3} local={r[3]:<3} {r[4]}")


if __name__ == "__main__":
    main()
