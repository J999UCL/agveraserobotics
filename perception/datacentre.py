"""Bridge between perception and the team's MuJoCo data-centre scene (sim/datacentre.xml).

The scene file is not modified. At load time this adds, via MjSpec:
  * one controllable status LED per drive carrier (the scene's own LEDs are one shared mesh per
    server, so they can't show a fault on a single drive), drawn over the existing LED positions
  * a wrist camera on the Panda hand, with api.md's intrinsics (fx = fy = 366.5 at 640x480)
  * optionally a dropped cable in front of a server, to exercise obstacle detection

Frames follow api.md: positions in the robot base frame (Panda link0: +X towards the racks, +Z up).
Bay numbering per server: 2 rows x 4 carriers, left to right, top to bottom (as seen by the robot).

    m, d = load()
    set_led(m, rack=2, slot=6, bay=5, colour="amber")
    cap = capture(m, d, *scan_pose(m, d, rack=2, slot=6))      # or capture_wrist(m, d) once the arm moves
    scan = make_perception(m, d).scan(cap, target_xyz=server_target(m, d, 2, 6))
    bay_views(scan)      # agent-friendly: [{"bay", "led", "temp_c", "obstructed"}, ...]
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .geometry import USD_TO_CV, make_T
from .hotswap import HotSwapPerception, Scan
from .types import Capture

SIM_XML = Path(__file__).resolve().parents[1] / "sim" / "datacentre.xml"
RACKS, SLOTS, ROWS, COLS = 4, 20, 2, 4
W, H, FX = 640, 480, 366.5
FOVY = 2 * math.degrees(math.atan((H / 2) / FX))
LED_RGBA = {"green": (0.1, 1.0, 0.25, 1), "amber": (1.0, 0.6, 0.0, 1), "red": (1.0, 0.06, 0.06, 1),
            "blue": (0.1, 0.35, 1.0, 1), "off": (0.04, 0.04, 0.04, 1)}
# Drive-carrier geometry, tray-local (from sim/server_hardware.py). Server fronts face -Y.
LED_SIZE = (0.0018, 0.0012, 0.0022)
TAB_OFFSET_TRAY = np.array([0.082, 0.001, -0.008])   # blue latch front centre minus LED centre (CAD)
HANDLE_CLEARANCE = 0.07                              # front handles stick out ~5 cm: obstacles must be further


def ident(rack: int, slot: int) -> str:
    return f"r{rack:02}_u{slot * 2 + 1:02}"


def _carrier(bay: int) -> tuple[float, float]:
    r, c = divmod(bay, COLS)
    return -0.164 + c * 0.109, 0.022 + (1 - r) * 0.035   # x, z of the carrier centre (row 0 = top)


def led_local(bay: int) -> np.ndarray:
    x, z = _carrier(bay)
    return np.array([x - 0.043, -0.4055, z + 0.008])


def grab_local(bay: int) -> np.ndarray:
    x, z = _carrier(bay)
    return np.array([x + 0.039, -0.4045, z])


def load(status_leds: bool = True, wrist_camera: bool = True, cable=None):
    """cable: optional ((x0, y0, z0), (x1, y1, z1)) world endpoints of a dropped cable."""
    spec = mujoco.MjSpec.from_file(str(SIM_XML))
    if status_leds:
        spec.add_material(name="status_led", emission=1.0)
        for rack in range(1, RACKS + 1):
            for slot in range(SLOTS):
                body = spec.body(ident(rack, slot))
                for bay in range(ROWS * COLS):
                    body.add_geom(name=f"{ident(rack, slot)}_led{bay}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                  size=list(LED_SIZE), pos=led_local(bay).tolist(), rgba=list(LED_RGBA["green"]),
                                  material="status_led", contype=0, conaffinity=0, group=2)
    if wrist_camera:
        # Hand +Z points along the fingers; the camera looks the same way, 6 cm to the side.
        spec.body("hand").add_camera(name="wrist_cam", pos=[0.06, 0.0, 0.05], quat=[0, 1, 0, 0], fovy=FOVY)
    if cable is not None:
        (a, b) = cable
        spec.worldbody.add_geom(name="dropped_cable", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.007, 0, 0],
                                fromto=[*a, *b], rgba=[0.05, 0.05, 0.05, 1], contype=0, conaffinity=0)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def set_led(m: mujoco.MjModel, rack: int, slot: int, bay: int, colour: str) -> None:
    m.geom_rgba[m.geom(f"{ident(rack, slot)}_led{bay}").id] = LED_RGBA[colour]


# ---------------------------------------------------------------- frames
def T_world_base(d: mujoco.MjData) -> np.ndarray:
    b = d.body("link0")
    return make_T(b.xmat.reshape(3, 3).copy(), b.xpos.copy())


def to_base(d: mujoco.MjData, p_world) -> np.ndarray:
    T = np.linalg.inv(T_world_base(d))
    return T[:3, :3] @ np.asarray(p_world, float) + T[:3, 3]


def _tray_point(d: mujoco.MjData, rack: int, slot: int, local) -> np.ndarray:
    b = d.body(ident(rack, slot))
    return b.xmat.reshape(3, 3) @ np.asarray(local, float) + b.xpos


def server_target(m, d, rack: int, slot: int) -> np.ndarray:
    """Centre of a server's LED grid in the base frame. This is map knowledge (where the server is),
    not where the fault is, so it's allowed in the live pipeline."""
    centre = np.mean([led_local(b) for b in range(ROWS * COLS)], axis=0)
    return to_base(d, _tray_point(d, rack, slot, centre))


def scan_pose(m, d, rack: int, slot: int, standoff: float = 0.30):
    """World (camera position, look-at point) square to a server, like api.md's view_server().
    For testing until the arm can be driven there."""
    target = _tray_point(d, rack, slot, np.mean([led_local(b) for b in range(ROWS * COLS)], axis=0))
    out = d.body(ident(rack, slot)).xmat.reshape(3, 3) @ np.array([0, -1.0, 0])  # out of the server front
    return target + standoff * out, target


def tab_offset_base(d: mujoco.MjData, rack: int = 2, slot: int = 0) -> np.ndarray:
    R = np.linalg.inv(T_world_base(d))[:3, :3] @ d.body(ident(rack, slot)).xmat.reshape(3, 3)
    return R @ TAB_OFFSET_TRAY


def make_perception(m, d, **kw) -> HotSwapPerception:
    kw.setdefault("tab_offset", tab_offset_base(d))
    kw.setdefault("obstacle_margin", HANDLE_CLEARANCE)
    return HotSwapPerception(rows=ROWS, cols=COLS, **kw)


# ---------------------------------------------------------------- rendering
def _render(m, d, cam, w=W, h=H):
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0  # the scene keeps collision proxies in group 3
    with mujoco.Renderer(m, h, w) as r:
        r.update_scene(d, camera=cam, scene_option=opt)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(d, camera=cam, scene_option=opt)
        depth = r.render().copy()
    return rgb, depth.astype(np.float32)


def _K(fovy_deg: float, w=W, h=H) -> np.ndarray:
    f = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def capture(m, d, cam_pos_world, target_world, t: float = 0.0) -> Capture:
    """Render from a free camera pose (world frame) and return a base-frame Capture."""
    pos, tgt = np.asarray(cam_pos_world, float), np.asarray(target_world, float)
    fwd = (tgt - pos) / np.linalg.norm(tgt - pos)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = pos + fwd
    cam.distance = 1.0
    cam.azimuth = math.degrees(math.atan2(fwd[1], fwd[0]))
    cam.elevation = math.degrees(math.asin(np.clip(fwd[2], -1, 1)))
    old = m.vis.global_.fovy
    m.vis.global_.fovy = FOVY
    try:
        rgb, depth = _render(m, d, cam)
    finally:
        m.vis.global_.fovy = old
    x = np.cross(fwd, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T_world_cam = make_T(np.column_stack([x, np.cross(fwd, x), fwd]), pos)  # OpenCV axes
    return Capture(rgb, depth, _K(FOVY), np.linalg.inv(T_world_base(d)) @ T_world_cam, t=t)


def capture_wrist(m, d, t: float = 0.0) -> Capture:
    """Render the Panda's wrist camera (what api.md's get_wrist_frames() returns)."""
    cid = m.camera("wrist_cam").id
    rgb, depth = _render(m, d, cid)
    T_world_cam = make_T(d.cam_xmat[cid].reshape(3, 3).copy(), d.cam_xpos[cid].copy()) @ USD_TO_CV
    return Capture(rgb, depth, _K(m.cam_fovy[cid]), np.linalg.inv(T_world_base(d)) @ T_world_cam, t=t)


# ---------------------------------------------------------------- for the agent
def bay_views(scan: Scan, n_bays: int = ROWS * COLS) -> list[dict]:
    """Same shape as the agent's BayView (bay, led, temp_c, obstructed), plus grab point and reasons."""
    out = []
    for i in range(n_bays):
        b = scan.bay(i)
        if b is None:
            out.append({"bay": i, "led": "unknown", "temp_c": None, "obstructed": False,
                        "grab_xyz_base": None, "reasons": ["not seen"]})
            continue
        led = {"hidden": "unknown", "blue": "identify"}.get(b.led, b.led)
        out.append({"bay": i, "led": led, "temp_c": None, "obstructed": scan.path_blocked(b),
                    "grab_xyz_base": b.tab_xyz_base, "reasons": b.reasons})
    return out
