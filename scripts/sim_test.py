"""End-to-end environment test: pull the tray, diagnose, swap a DIMM and a fan, jam abort.

    python scripts/sim_test.py            (headless, saves frames to outputs/sim_test/)
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
    Image.fromarray(sim.render("overview", 1280, 720)).save(out / f"{name}_overview.png")
    Image.fromarray(sim.render("wrist")).save(out / f"{name}_wrist.png")


def show(label, r):
    keep = {k: v for k, v in r.items() if k != "status"}
    print(f"[{time.time() - t0:5.1f}s wall | sim {sim.time:6.1f}s] {label}: {json.dumps(keep, default=str)}", flush=True)


sim.step(25)
h = sim.server_health()
print("health:", h["overall"], "| bad parts:", [d for d in h["dimms"] + h["fans"] if d["status"] not in ("ok",)])
shot("start")
show("pull_tray_out", sk.pull_tray_out())
show("view_tray", sk.view_tray())
shot("tray_view")
show("swap A3 (failed DIMM)", sk.swap("A3"))
show("swap F2 (stalled fan)", sk.swap("F2"))
sim.step(200)  # let POST finish
h = sim.server_health()
print("health after swaps:", h["overall"], "| A3:", h["dimms"][2], "| F2:", h["fans"][1], "| temps", h["cpu_temps_c"])
shot("after_swaps")
sim.set_jam("A6")
show("pick A6 (jammed)", sk.pick("A6"))
print("bin contents:", [p.id for p in sim.parts.values() if p.where == "bin"])
print("cart inventory:", sim.cart_inventory())
print("last events:", sim.events[-6:])
show("push_tray_in", sk.push_tray_in())
shot("end")
print(f"done in {time.time() - t0:.1f}s wall, {sim.time:.1f}s sim")
