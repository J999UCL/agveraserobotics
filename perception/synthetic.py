"""A stand-in server rack rendered with MuJoCo, so perception can be tested before Isaac Sim is ready.

Robot base = world origin, +x forward. The rack face is at x = RACK_X, facing the robot.
Bays: ROWS x COLS drive caddies, each with an AprilTag asset label and a status LED.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .geometry import USD_TO_CV, make_T
from .tags import CELLS_TAG, CELLS_TEXTURE, tag_texture
from .thermal import synth_thermal
from .types import Capture

ASSETS = Path(__file__).parent / "_assets"
RACK_X = 0.80
ROWS, COLS = 4, 3
BAY_DY, BAY_DZ = 0.13, 0.055
RACK_Z0 = 0.80                # height of the bottom row
FACE = (0.11, 0.040)          # caddy front face (w, h)
LABEL = 0.030                 # printed label width; black tag = LABEL * 8/10
TAG_SIZE = LABEL * CELLS_TAG / CELLS_TEXTURE
LED_DY = 0.042                # LED offset from the label centre (to the right, seen from the robot)
LED_RGB = {"green": (0.1, 1.0, 0.2), "amber": (1.0, 0.6, 0.0), "red": (1.0, 0.05, 0.05)}


@dataclass
class Bay:
    row: int
    col: int
    tag_id: int
    comp_id: str
    present: bool = True
    led: str = "green"
    temp_c: float = 38.0

    @property
    def tag_centre(self) -> np.ndarray:  # robot base frame
        y = (self.col - (COLS - 1) / 2) * BAY_DY
        return np.array([RACK_X - 0.0025, y, RACK_Z0 + self.row * BAY_DZ])


@dataclass
class RackScene:
    bays: list[Bay]
    model: mujoco.MjModel = field(repr=False)
    data: mujoco.MjData = field(repr=False)
    body_temp: dict = field(default_factory=dict)  # body id -> temperature

    def inventory(self) -> dict:
        return {b.tag_id: {"id": b.comp_id, "type": "drive", "face": list(FACE),
                           "grasp_offset": [0.0, 0.0, 0.02], "temp_limit": 55} for b in self.bays}

    def expected(self) -> dict:
        return {b.comp_id: b.tag_centre.tolist() for b in self.bays}


def default_bays() -> list[Bay]:
    bays, tag = [], 0
    for r in range(ROWS):
        for c in range(COLS):
            bays.append(Bay(r, c, tag, f"C03-U12-D{tag}", temp_c=float(34 + (tag * 7) % 9)))
            tag += 1
    bays[4].temp_c = 71.0           # overheating drive
    bays[7].led = "red"             # failed drive
    bays[2].led = "amber"           # predictive-failure warning
    bays[10].present = False        # empty bay: drive is missing
    return bays


def build(bays: list[Bay] | None = None) -> RackScene:
    bays = bays or default_bays()
    assets, bodies = [], []
    q_face = f"{math.cos(-math.pi / 4)} 0 {math.sin(-math.pi / 4)} 0"  # turns +z to face -x (the robot)
    for b in bays:
        p = b.tag_centre
        if not b.present:
            continue
        tex = tag_texture(b.tag_id, ASSETS)
        assets.append(f'<texture name="t{b.tag_id}" type="2d" file="{tex}"/>'
                      f'<material name="m{b.tag_id}" texture="t{b.tag_id}" specular="0" shininess="0"/>')
        led = LED_RGB[b.led]
        bodies.append(
            f'<body name="bay{b.tag_id}" pos="{RACK_X + 0.08} {p[1]} {p[2]}">'
            f'<geom type="box" size="0.08 {FACE[0] / 2} {FACE[1] / 2}" rgba=".18 .19 .22 1"/>'
            f'<geom type="box" pos="-0.0815 0 0" quat="{q_face}" size="{LABEL / 2} {LABEL / 2} 0.001" material="m{b.tag_id}"/>'
            f'<geom type="box" pos="-0.0815 {-LED_DY} 0" size="0.0015 0.003 0.003" '
            f'material="led_{b.led}" rgba="{led[0]} {led[1]} {led[2]} 1"/></body>')
    xml = f"""
<mujoco>
  <visual><global offwidth="1280" offheight="960"/><headlight ambient=".45 .45 .45" diffuse=".5 .5 .5"/></visual>
  <asset>
    {''.join(assets)}
    <material name="led_green" emission="1"/><material name="led_amber" emission="1"/><material name="led_red" emission="1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.5" dir="0.3 0 -1" diffuse=".6 .6 .6" castshadow="false"/>
    <geom type="plane" size="3 3 .1" rgba=".35 .36 .38 1"/>
    <!-- rack: back plate (seen through empty bays), sides, top -->
    <geom type="box" pos="{RACK_X + 0.20} 0 {RACK_Z0 + 0.08}" size="0.01 0.26 0.16" rgba=".08 .08 .09 1"/>
    <geom type="box" pos="{RACK_X + 0.08} -0.25 {RACK_Z0 + 0.08}" size="0.12 0.01 0.16" rgba=".25 .26 .3 1"/>
    <geom type="box" pos="{RACK_X + 0.08} 0.25 {RACK_Z0 + 0.08}" size="0.12 0.01 0.16" rgba=".25 .26 .3 1"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    temps = {model.body(f"bay{b.tag_id}").id: b.temp_c for b in bays if b.present}
    return RackScene(bays, model, data, temps)


def look_at(pos, target) -> np.ndarray:
    """Camera pose (OpenCV axes) at `pos` looking at `target`, world z up."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y_cv = np.cross(f, x)  # down
    return make_T(np.column_stack([x, y_cv, f]), pos)


def render(scene: RackScene, cam_pos, target, width: int = 640, height: int = 480, fovy: float = 60.0,
           ambient_c: float = 24.0, seed: int = 0) -> Capture:
    """Render RGB, depth and a synthetic thermal image from a free camera pose."""
    m, d = scene.model, scene.data
    T = look_at(cam_pos, target)
    T_gl = T @ USD_TO_CV  # back to MuJoCo/OpenGL camera axes
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    fwd = -T_gl[:3, 2]
    cam.lookat[:] = np.asarray(cam_pos) + fwd
    cam.distance = 1.0
    cam.azimuth = math.degrees(math.atan2(fwd[1], fwd[0]))
    cam.elevation = math.degrees(math.asin(np.clip(fwd[2], -1, 1)))
    m.vis.global_.fovy = fovy
    with mujoco.Renderer(m, height, width) as r:
        r.update_scene(d, camera=cam)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(d, camera=cam)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=cam)
        seg = r.render()[..., 0].copy()
        r.disable_segmentation_rendering()
    body_seg = np.full(seg.shape, -1, np.int32)
    valid = seg >= 0
    body_seg[valid] = m.geom_bodyid[seg[valid]]
    thermal = synth_thermal(body_seg, scene.body_temp, ambient=ambient_c, seed=seed)
    f_px = (height / 2) / math.tan(math.radians(fovy) / 2)
    K = np.array([[f_px, 0, width / 2], [0, f_px, height / 2], [0, 0, 1.0]])
    return Capture(rgb=rgb, depth=depth.astype(np.float32), K=K, T_base_cam=T, thermal=thermal)
