"""Run the toy-rack scenarios against Gemini 4 Argon on Modal (CPU only).

  uv run modal run robot/sandbox/modal_app.py                          # local-first, all scenarios
  uv run modal run robot/sandbox/modal_app.py --mode argon             # Argon decides everything
  uv run modal run robot/sandbox/modal_app.py --only clean_failure --trials 3

Traces land in robot/sandbox/runs/<mode>_<scenario>_t<k>.json (+ final camera png).
"""

from __future__ import annotations

import base64
import json
import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("google-genai==3.0.0", "pillow")
    .add_local_dir(HERE, "/root/sandbox", ignore=["runs", "__pycache__"])
)
app = modal.App("rack-reasoning-sandbox", image=image)


@app.function(secrets=[modal.Secret.from_name("gemini")], timeout=1200)
def run_scenario(name: str, thinking: str = "LOW", mode: str = "local_first") -> dict:
    sys.path.insert(0, "/root/sandbox")
    from agent import ArgonBrain, Budget, run
    from policy import run_local_first
    from scenarios import SCENARIOS, grade

    print(f"=== {name} ({mode}) ===", flush=True)
    w = SCENARIOS[name][0]()
    runner = run_local_first if mode == "local_first" else run
    trace = runner(w, ArgonBrain(thinking), Budget())
    result = grade(name, w)
    print(f"=== {name}: {'PASS' if result['passed'] else 'FAIL'} {result['checks']}", flush=True)
    return {
        "scenario": name,
        "mode": mode,
        "grade": result,
        "model_calls": trace.model_calls,
        "local_decisions": trace.local_decisions,
        "escalations": trace.escalations,
        "model_seconds": round(trace.model_seconds, 1),
        "fallback": trace.fallback,
        "steps": trace.steps,
        "final_world": w.snapshot(),
        "final_camera_png_b64": base64.b64encode(w.camera_png()).decode(),
    }


@app.function(secrets=[modal.Secret.from_name("gemini")], timeout=300)
def argon_decide(prompt: str, images: list[bytes], thinking: str = "LOW") -> dict:
    """One Argon decision, for a sim running elsewhere (e.g. MuJoCo on the laptop).
    The Gemini key never leaves Modal."""
    sys.path.insert(0, "/root/sandbox")
    from agent import ArgonBrain
    decision, raw = ArgonBrain(thinking).decide(prompt, images)
    return {"decision": decision, "raw": raw}


@app.local_entrypoint()
def main(only: str = "", thinking: str = "LOW", trials: int = 1, mode: str = "local_first"):
    """--only a,b subset; --trials N repeat; --mode local_first|argon."""
    sys.path.insert(0, HERE)
    from scenarios import SCENARIOS

    names = only.split(",") if only else list(SCENARIOS)
    jobs = [n for n in names for _ in range(trials)]
    out_dir = os.path.join(HERE, "runs")
    os.makedirs(out_dir, exist_ok=True)
    stats: dict[str, list] = {n: [] for n in names}
    seen: dict[str, int] = {}
    for name, res in zip(jobs, run_scenario.map(jobs, kwargs={"thinking": thinking, "mode": mode},
                                                 return_exceptions=True)):
        k = seen[name] = seen.get(name, 0) + 1
        if isinstance(res, Exception):
            stats[name].append(("ERROR", 0, 0, 0.0, str(res)[:120]))
            continue
        png = base64.b64decode(res.pop("final_camera_png_b64"))
        stem = os.path.join(out_dir, f"{mode}_{name}_t{k}")
        with open(stem + "_final.png", "wb") as f:
            f.write(png)
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        failed = [c for c, ok in res["grade"]["checks"].items() if not ok]
        stats[name].append(("PASS" if res["grade"]["passed"] else "FAIL", res["model_calls"],
                            res["local_decisions"], res["model_seconds"],
                            ", ".join(failed) or (res["fallback"] or "")))

    print(f"\nmode={mode}")
    print(f"{'SCENARIO':<18}{'PASS':<7}{'ARGON':<7}{'LOCAL':<7}{'ARGON s':<9}FAILED CHECKS")
    total = passed = calls_all = local_all = 0
    for n, rs in stats.items():
        k = max(len(rs), 1)
        p = sum(r[0] == "PASS" for r in rs)
        total += len(rs); passed += p
        calls_all += sum(r[1] for r in rs); local_all += sum(r[2] for r in rs)
        notes = "; ".join(sorted({r[4] for r in rs if r[0] != "PASS" and r[4]}))
        print(f"{n:<18}{f'{p}/{len(rs)}':<7}{sum(r[1] for r in rs) / k:<7.1f}"
              f"{sum(r[2] for r in rs) / k:<7.1f}{sum(r[3] for r in rs) / k:<9.0f}{notes}")
    share = local_all / max(local_all + calls_all, 1)
    print(f"\nTOTAL {passed}/{total}   decisions made locally: {share:.0%}")
