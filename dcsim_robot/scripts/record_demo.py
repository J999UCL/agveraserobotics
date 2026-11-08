"""Record a scripted maintenance shift to MP4 (environment showcase, no AI involved).

    python scripts/record_demo.py                    # outputs/demo.mp4 at 4x speed
    python scripts/record_demo.py --speed 2 --out outputs/demo_2x.mp4

Story: work orders -> drive (a technician steps into the aisle; the robot stops and waits)
-> pull tray -> inspect -> swap failed RAM -> next server -> swap failed fan -> a stuck part
is aborted safely -> home -> restock.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcsim.recorder import Recorder  # noqa: E402
from dcsim.sim import DataCenterSim  # noqa: E402
from dcsim.skills import Skills  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--speed", type=float, default=4.0)
p.add_argument("--out", default="outputs/demo.mp4")
args = p.parse_args()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
sim = DataCenterSim()
sk = Skills(sim)
rec = Recorder(sim, args.out, speed=args.speed)
t0 = time.time()


def do(caption: str, fn, *a):
    rec.caption(caption)
    r = fn(*a)
    print(f"[sim {sim.time:6.1f}s] {caption}: ok={r.get('ok')} {r.get('error') or ''}", flush=True)
    return r


orders = [w for w in sim.work_orders if w["status"] == "open"]
rec.caption(f"{len(orders)} open work orders: " + ", ".join(f"{w['server']} {w['slot']}" for w in orders))
sim.step(75)

first = orders[0]
sim.spawn_technician("block", x=1.4, delay=9.0, duration=4.0)
do(f"Driving to {first['server']} (technician in the aisle)", sk.go_to, first["server"])
do("Pulling the server tray out", sk.pull_tray_out)
do("Inspecting: amber LED marks the failed part", sk.view_tray)
sim.step(40)
do(f"Hot-swapping {first['component']} in slot {first['slot']}", sk.swap, first["slot"])
rec.caption("Server verifying the new part...")
sim.step(175)
do("Sliding the tray back in", sk.push_tray_in)

second = orders[1]
do(f"Driving to {second['server']}", sk.go_to, second["server"])
do("Pulling the server tray out", sk.pull_tray_out)
do("Inspecting", sk.view_tray)
sim.step(40)
sim.set_jam(second["server"], "A6")
do("A part is stuck: watching extraction force", sk.pick, "A6")
do(f"Hot-swapping {second['component']} in slot {second['slot']}", sk.swap, second["slot"])
rec.caption("Server verifying the new part...")
sim.step(175)
do("Sliding the tray back in", sk.push_tray_in)
do("Returning to the dock", sk.go_home)
do("Restocking spares from the depot", sk.restock)
rec.caption("All work orders resolved")
sim.step(100)
rec.close()
print(f"wrote {args.out}: {rec.frames} frames, {sim.time:.0f}s sim in {time.time() - t0:.0f}s wall")
print("work orders:", [(w["id"], w["status"]) for w in sim.work_orders])
