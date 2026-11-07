"""Phase 1: cart + Franka + rack + wrist RGB-D camera.

Run with Isaac Sim's bundled Python (NOT a normal venv):

    C:\\isaacsim\\python.bat scripts\\phase1_scene.py              # GUI, stays open
    C:\\isaacsim\\python.bat scripts\\phase1_scene.py --headless   # capture frames and exit

Moves the arm to a scan pose facing the storage server, then saves wrist RGB, depth,
the overview shot and camera intrinsics to outputs/phase1/. It also checks that
unprojecting the faulty bay's pixel lands back on the bay (the perception contract).
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

from hotswap.scene import HotSwapScene, base_to_pixel, pixel_to_base  # noqa: E402

out = ROOT / "outputs" / "phase1"
out.mkdir(parents=True, exist_ok=True)

scene = HotSwapScene(simulation_app)
scene.build()
scene.play()
scene.step(30)

server_z = scene.bays[(1, 1)].face_center[2]
# put the camera (not the hand) level with the bays, ~0.4 m in front of them
scan_pos = np.array([0.30, 0.0, server_z]) - scene.camera_offset_world() * np.array([0, 0, 1])
reached = scene.move_hand_to(scan_pos)
print(f"[phase1] scan pose reached={reached} hand={np.round(scene.hand_pose()[0], 3)}")
scene.step(30)  # let the renderer settle

frames = None
for _ in range(120):
    frames = scene.get_wrist_frames()
    if frames is not None and frames.rgb.any():
        break
    scene.step()
if frames is None:
    print("[phase1] ERROR: no wrist frames received")
    simulation_app.close()
    sys.exit(1)

Image.fromarray(frames.rgb).save(out / "wrist_rgb.png")
d = np.where(np.isfinite(frames.depth), frames.depth, 0)
d_vis = (255 * (1 - np.clip(d / 1.0, 0, 1))).astype(np.uint8)  # near = bright, 0-1 m
Image.fromarray(d_vis).save(out / "wrist_depth.png")
np.save(out / "wrist_depth.npy", frames.depth)
overview = scene.get_overview_rgb()
if overview is not None:
    Image.fromarray(overview).save(out / "overview.png")

# perception contract check: faulty tab -> pixel -> unproject -> should land on the tab
bay = scene.bays[scene.cfg.faulty_bay]
u, v, z = base_to_pixel(bay.tab_center, frames)
ui, vi = int(round(u)), int(round(v))
h, w = frames.depth.shape
check = {}
if 0 <= ui < w and 0 <= vi < h:
    p = pixel_to_base(ui, vi, float(frames.depth[vi, ui]), frames)
    check = {"pixel": [ui, vi], "expected_depth": round(z, 4), "measured_depth": round(float(frames.depth[vi, ui]), 4),
             "unprojected": np.round(p, 4).tolist(), "true_tab_center": np.round(bay.tab_center, 4).tolist(),
             "error_m": round(float(np.linalg.norm(p - bay.tab_center)), 4)}
print(f"[phase1] unprojection check: {check}")

info = {
    "K": frames.K.tolist(),
    "resolution_hw": [h, w],
    "cam_pos": frames.cam_pos.round(4).tolist(),
    "cam_rot": frames.cam_rot.round(4).tolist(),
    "faulty_bay": list(scene.cfg.faulty_bay),
    "unprojection_check": check,
    "depth_range_m": [float(np.nanmin(d[d > 0])) if (d > 0).any() else None, float(np.nanmax(d))],
}
(out / "camera_info.json").write_text(json.dumps(info, indent=2))
print(f"[phase1] saved frames to {out}")

if args.headless:
    simulation_app.close()
else:
    print("[phase1] GUI open. Close the window to exit.")
    while simulation_app.is_running():
        scene.ik_step(scan_pos)
        scene.step()
    simulation_app.close()
