"""Exercise every environment call and save evidence frames to outputs/env_test/.

    python scripts/env_test.py --headless       (with Isaac Sim's Python)

This is a TEST of the environment, not the real controller: it uses ground-truth bay
positions (debug_bay_pose), which the real perception pipeline must not do.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from hotswap.scene import FINGER_REACH, HotSwapScene  # noqa: E402

out = ROOT / "outputs" / "env_test"
out.mkdir(parents=True, exist_ok=True)
results: dict = {}


def save(name: str) -> None:
    f = scene.wait_for_frames()
    if f is not None:
        Image.fromarray(f.rgb).save(out / f"{name}_wrist.png")
    o = scene.get_overview_rgb()
    if o is not None:
        Image.fromarray(o).save(out / f"{name}_overview.png")


def log(key, value) -> None:
    results[key] = value
    print(f"[env_test] {key}: {value}", flush=True)


def grasp_pose(cab: int, bay: int, depth_into_tab: float = 0.012):
    tab = np.array(scene.debug_bay_pose(cab, bay)["tab_center"])
    tab_front = tab[0] - scene.cfg.caddy.tab_depth / 2
    hand = np.array([tab_front + depth_into_tab - FINGER_REACH, tab[1], tab[2]])
    return hand - np.array([0.08, 0, 0]), hand


scene = HotSwapScene(simulation_app)
scene.build()
scene.play()
scene.step(20)
log("start_cabinet", scene.current_cabinet())

# --- cart -------------------------------------------------------------------
scene.go_to_cabinet(0)
log("after_go_to_0", {"cabinet": scene.current_cabinet(), "base_y": round(scene.base_y, 3)})
save("cabinet0")
scene.go_to_cabinet(1)
log("after_go_to_1", {"cabinet": scene.current_cabinet(), "hand": np.round(scene.hand_pose()[0], 3).tolist()})

# --- view + telemetry -------------------------------------------------------
log("view_rack", scene.view_rack())
save("view_rack")
log("telemetry", scene.get_telemetry())

# --- normal extraction of the faulty bay (cabinet 1, bay 6) -----------------
cab, bay = scene.cfg.faulty
pre, grasp = grasp_pose(cab, bay)
scene.open_gripper()
log("pregrasp_reached", scene.move_hand_to(pre))
log("grasp_reached", scene.move_hand_to(grasp, max_speed=0.05))
scene.close_gripper()
log("grasp", {"width": round(scene.gripper_width(), 4), "is_grasping": scene.is_grasping()})
peak = 0.0
target = grasp - np.array([0.18, 0, 0])
start = scene.hand_pose()[0]
for i in range(1, 361):  # 6 s pull at 3 cm/s
    a = min(1.0, i / 360)
    scene.ik_step(start + (target - start) * a)
    scene.step()
    peak = max(peak, scene.get_pull_force())
log("normal_pull", {**scene.caddy_status(cab, bay), "peak_force_n": round(peak, 1),
                    "still_grasping": scene.is_grasping()})
save("extracted")

# --- jammed caddy (cabinet 1, bay 5): force must climb fast ------------------
scene.open_gripper()
scene.stow_arm()
jb = 5
scene.set_jam(cab, jb, True)
pre, grasp = grasp_pose(cab, jb)
scene.move_hand_to(pre)
scene.move_hand_to(grasp, max_speed=0.05)
scene.close_gripper()
trace = []
start = scene.hand_pose()[0]
for i in range(1, 121):  # try to pull 3 cm over 2 s, ignoring force (the controller must not!)
    scene.ik_step(start - np.array([0.03 * min(1.0, i / 120), 0, 0]))
    scene.step()
    if i % 10 == 0:
        trace.append(round(scene.get_pull_force(), 1))
log("jam_force_trace_n", trace)
log("jam_status", {**scene.caddy_status(cab, jb), "connector_broken": scene.connector_broken()})
scene.open_gripper()
scene.stow_arm()

# --- obstacle ---------------------------------------------------------------
scene.view_rack()
pre, _ = grasp_pose(cab, 2)
scene.move_hand_to(pre)
where = scene.drop_obstacle()
scene.step(60)
f = scene.wait_for_frames()
h, w = f.depth.shape
center = f.depth[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
log("obstacle", {"dropped_at": where, "min_center_depth_m": round(float(np.nanmin(center)), 3)})
save("obstacle")
scene.clear_obstacle()
scene.step(5)
f = scene.wait_for_frames()
center = f.depth[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
log("obstacle_cleared_min_depth_m", round(float(np.nanmin(center)), 3))

# --- reset ------------------------------------------------------------------
log("reset", scene.reset(seed=3))
scene.step(30)
log("after_reset", {"cabinet": scene.current_cabinet(), "bay6": scene.caddy_status(1, 6),
                    "telemetry": scene.get_telemetry()})
save("after_reset")

(out / "results.json").write_text(json.dumps(results, indent=2, default=str))
print(f"[env_test] done, saved to {out}")
simulation_app.close()
