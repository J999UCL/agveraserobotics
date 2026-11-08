"""End-to-end environment test (in-process, headless): a full work shift.

    python scripts/sim_test.py

Fleet health -> drive to each faulty server -> pull tray -> swap -> push in -> go home ->
restock -> jam test. Saves frames to outputs/sim_test/.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image  # noqa: E402

from dcsim.sim import DataCenterSim  # noqa: E402
from dcsim.skills import Skills  # noqa: E402

out = Path(__file__).resolve().parents[1] / "outputs" / "sim_test"
out.mkdir(parents=True, exist_ok=True)
sim = DataCenterSim()
sk = Skills(sim)
t0 = time.time()


def shot(name):
    Image.fromarray(sim.render("follow", 1280, 720)).save(out / f"{name}_follow.png")
    Image.fromarray(sim.render("wrist")).save(out / f"{name}_wrist.png")


def show(label, r):
    keep = {k: v for k, v in r.items() if k != "status"}
    print(f"[{time.time() - t0:5.1f}s wall | sim {sim.time:6.1f}s] {label}: {json.dumps(keep, default=str)[:300]}", flush=True)


sim.step(10)
fleet = sim.fleet_health()
print("fleet:", [(f["server"], f["overall"], f["problem_slots"]) for f in fleet])
shot("start")
for f in fleet:
    if not f["problem_slots"]:
        continue
    sid = f["server"]
    show(f"go_to {sid}", sk.go_to(sid))
    show("pull_tray_out", sk.pull_tray_out())
    show("view_tray", sk.view_tray())
    shot(f"{sid}_tray")
    for slot in f["problem_slots"]:
        show(f"swap {slot}", sk.swap(slot))
    sim.step(200)
    h = sim.server_health(sid)
    print(f"  {sid} after: {h['overall']}, temps {h['cpu_temps_c']}")
    show("push_tray_in", sk.push_tray_in())
print("fleet after:", [(f["server"], f["overall"]) for f in sim.fleet_health()])
print("onboard before restock:", sim.storage("onboard"))
show("go_home", sk.go_home())
show("restock", sk.restock())
shot("restocked")
show("go_to R1-02-U10", sk.go_to("R1-02-U10"))
show("pull_tray_out", sk.pull_tray_out())
sim.set_jam("R1-02-U10", "A6")
show("pick A6 (jammed)", sk.pick("A6"))
show("push_tray_in", sk.push_tray_in())
shot("end")
print("bin:", [p.id for p in sim.parts.values() if p.where == "bin"])
print(f"done in {time.time() - t0:.1f}s wall, {sim.time:.1f}s sim")
