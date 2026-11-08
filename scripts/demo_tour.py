"""Guided tour of the environment for showing the team (sim server must be running).

    python scripts/demo_tour.py [host] [port] [--fast]

NOTE: this drives the arm using ground-truth bay positions (debug_bay_pose) so it can
show the physics and triggers. The real pipeline must find the bay with perception.
The obstacle and jam reactions here are a simple reference for the controller.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hotswap_client import SimClient  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
host = args[0] if args else "localhost"
port = int(args[1]) if len(args) > 1 else 5555
FAST = "--fast" in sys.argv
FINGER_REACH = 0.1034
ABORT_N = 25.0

sim = SimClient(host, port)


def say(msg: str, pause: float = 1.5) -> None:
    print(f"\n>>> {msg}", flush=True)
    if not FAST:
        time.sleep(pause)


def grasp_targets(cab: int, bay: int):
    tab = np.array(sim.debug_bay_pose(cab, bay)["tab_center"])
    hand = np.array([tab[0] - 0.015 + 0.012 - FINGER_REACH, tab[1], tab[2]])
    return hand - [0.10, 0, 0], hand


def corridor_blocked(f) -> bool:
    """Toy obstacle check: something much closer than the rack in the middle of the image."""
    h, w = f.depth.shape
    c = f.depth[h // 4: 3 * h // 4, 2 * w // 5: 3 * w // 5]
    c = c[c > 0]
    return c.size > 0 and np.percentile(c, 5) < 0.6 * np.median(c)


sim.reset()
sim.step(30)
say("Monitoring system reports a failed drive:", 0.5)
for alert in sim.get_telemetry():
    print("   ", alert)
cab, bay = 1, 6

say("Cart tours the aisle: cabinet 0 ...")
sim.go_to_cabinet(0)
say("... cabinet 2 ...")
sim.go_to_cabinet(2)
say(f"... and back to cabinet {cab}, where the failed drive is.")
sim.go_to_cabinet(cab)

say("Arm moves to the inspection pose. Wrist camera sees all 12 bays (amber LED = faulty).")
sim.view_rack()
f = sim.get_wrist_frames()
print(f"    wrist frame {f.rgb.shape[1]}x{f.rgb.shape[0]}, depth {f.depth[f.depth > 0].min():.2f}-{f.depth.max():.2f} m")

# ---------------------------------------------------------------- obstacle
say("Approaching bay 6. A loose cable will drop into the path halfway...")
pre, grasp = grasp_targets(cab, bay)
sim.open_gripper()
start = np.array(sim.hand_pose()["position"])
dropped = False
done_drop = False  # drop the cable only once
i, n = 0, 150
while i < n:
    a = (i + 1) / n
    sim.ik_step((start + (pre - start) * a).tolist(), steps=1)
    if i == n // 3 and not done_drop:
        sim.drop_obstacle()
        dropped = done_drop = True
        sim.step(30)
    if dropped and i % 5 == 0 and corridor_blocked(sim.get_wrist_frames()):
        say("OBSTACLE in the depth corridor -> stop, back off 5 cm, wait.", 0.5)
        sim.move_relative(dx=-0.05)
        sim.step(60)
        say("Technician clears the cable. Path clear -> resume.")
        sim.clear_obstacle()
        sim.step(10)
        start = np.array(sim.hand_pose()["position"])
        i, n = 0, 90
        dropped = False
        continue
    i += 1

# ---------------------------------------------------------------- healthy pull
say("Gripping the pull tab and extracting along a straight line.")
sim.move_hand_to(grasp.tolist(), max_speed=0.05)
print("    grasp:", sim.close_gripper())
peak, out = 0.0, 0.0
start = np.array(sim.hand_pose()["position"])
for k in range(1, 61):
    sim.ik_step((start - [0.18 * k / 60, 0, 0]).tolist(), steps=6)
    force = sim.get_pull_force()
    peak = max(peak, force)
    out = sim.caddy_extraction(cab, bay)
    if k % 10 == 0:
        print(f"    pulled {out * 100:4.1f} cm   force {force:5.1f} N")
print("    status:", sim.caddy_status(cab, bay), f"peak {peak:.1f} N")
say("Drive extracted. Returning to the ready pose.")
sim.stow_arm()
sim.open_gripper()

# ---------------------------------------------------------------- jam
jb = 5
say(f"Now bay {jb} is mechanically JAMMED. Watch the force-based abort at {ABORT_N:.0f} N.")
sim.set_jam(cab, jb, True)
pre, grasp = grasp_targets(cab, jb)
sim.move_hand_to(pre.tolist())
sim.move_hand_to(grasp.tolist(), max_speed=0.05)
sim.close_gripper()
start = np.array(sim.hand_pose()["position"])
for k in range(1, 121):
    sim.ik_step((start - [0.03 * k / 120, 0, 0]).tolist(), steps=1)
    force = sim.get_pull_force()
    if force > ABORT_N:
        # React FIRST (the sim keeps running in real time), talk afterwards.
        # 1) stop pulling: move the target slightly back toward the rack to relieve tension
        hold = np.array(sim.hand_pose()["position"]) + [0.004, 0, 0]
        sim.ik_step(hold.tolist(), steps=10)
        # 2) only then open the fingers
        sim.open_gripper()
        say(f"SAFE ABORT: pull force {force:.1f} N > {ABORT_N:.0f} N. Released before the connector broke.", 0.5)
        sim.move_relative(dx=-0.08)
        break
print("    connector broken?", sim.connector_broken(), "| telemetry:", sim.get_telemetry()[-1])
sim.stow_arm()
say("Tour finished. sim.reset() puts everything back.", 0)
