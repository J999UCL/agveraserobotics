"""MuJoCo stand-in for the HotSwapScene server front, for testing perception before Isaac Sim is up.

Base frame like api.md: +X towards the rack, +Z up, +Y to the robot's left.
One 12-bay server (3 rows x 4 columns), bay numbers left to right, top to bottom.
Each caddy has a status LED (to the robot's left of centre) and a pull tab that sticks out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

FACE_X = 0.60                  # caddy front faces
CENTRE_Z = 0.60
ROWS, COLS = 3, 4
PITCH_Y, PITCH_Z = 0.105, 0.029
CADDY = (0.095, 0.024)         # face width (y), height (z)
LED_OFF = (0.035, 0.006)       # LED from caddy centre: +y (robot's left), +z
TAB_OFF = (-0.010, -0.004)     # tab centre from caddy centre
TAB_OUT = 0.012                # tab sticks out this far
FOVY_ISAAC = 2 * math.degrees(math.atan(240 / 366.5))  # matches api.md intrinsics (fx = 366.5 at 640x480)
LED_RGB = {"green": (0.1, 1.0, 0.2), "amber": (1.0, 0.6, 0.0), "blue": (0.15, 0.4, 1.0), "red": (1.0, 0.05, 0.05)}


def bay_centre(bay: int) -> np.ndarray:
    r, c = divmod(bay, COLS)
    y = ((COLS - 1) / 2 - c) * PITCH_Y           # column 0 is on the robot's left (+y)
    z = CENTRE_Z + ((ROWS - 1) / 2 - r) * PITCH_Z  # row 0 is the top
    return np.array([FACE_X, y, z])


def led_pos(bay: int) -> np.ndarray:
    return bay_centre(bay) + [-0.0015, LED_OFF[0], LED_OFF[1]]


def tab_pos(bay: int) -> np.ndarray:
    return bay_centre(bay) + [-TAB_OUT / 2, TAB_OFF[0], TAB_OFF[1]]


@dataclass
class HotSwapStandIn:
    leds: dict
    obstacle: bool
    model: mujoco.MjModel = field(repr=False)
    data: mujoco.MjData = field(repr=False)
    body_temp: dict = field(default_factory=dict)
    cable: tuple = ((0.47, 0.065, 0.50), (0.47, 0.045, 0.71))


def build(leds: dict | None = None, obstacle: bool = False) -> HotSwapStandIn:
    leds = leds or {b: "green" for b in range(ROWS * COLS)}
    geoms = []
    for b in range(ROWS * COLS):
        c = bay_centre(b)
        geoms.append(f'<geom type="box" pos="{c[0] + 0.05} {c[1]} {c[2]}" size="0.05 {CADDY[0] / 2} {CADDY[1] / 2}" rgba=".2 .21 .24 1"/>')
        t = tab_pos(b)
        geoms.append(f'<geom type="box" pos="{t[0]} {t[1]} {t[2]}" size="{TAB_OUT / 2} 0.02 0.004" rgba=".55 .56 .6 1"/>')
        colour = leds.get(b, "green")
        if colour != "off":
            p, rgb = led_pos(b), LED_RGB[colour]
            geoms.append(f'<geom type="box" pos="{p[0]} {p[1]} {p[2]}" size="0.0015 0.0025 0.0025" '
                         f'material="led" rgba="{rgb[0]} {rgb[1]} {rgb[2]} 1"/>')
    cable = ""
    if obstacle:
        (a, b) = HotSwapStandIn.__dataclass_fields__["cable"].default
        cable = f'<geom type="capsule" fromto="{" ".join(map(str, a))} {" ".join(map(str, b))}" size="0.007" rgba=".1 .1 .1 1"/>'
    xml = f"""
<mujoco>
  <visual><global offwidth="1280" offheight="960"/><headlight ambient=".45 .45 .45" diffuse=".5 .5 .5"/></visual>
  <asset><material name="led" emission="1"/></asset>
  <worldbody>
    <light pos="0 0 2" dir="0.4 0 -1" diffuse=".6 .6 .6" castshadow="false"/>
    <geom type="plane" size="3 3 .1" rgba=".35 .36 .38 1"/>
    <!-- server chassis (recessed between caddies), cabinet behind -->
    <geom type="box" pos="{FACE_X + 0.25} 0 {CENTRE_Z}" size="0.25 0.24 0.048" rgba=".12 .12 .14 1"/>
    <geom type="box" pos="{FACE_X + 0.006 + 0.005} 0 {CENTRE_Z}" size="0.005 0.24 0.048" rgba=".15 .15 .17 1"/>
    <geom type="box" pos="{FACE_X + 0.4} 0 {CENTRE_Z}" size="0.02 0.4 0.6" rgba=".08 .08 .09 1"/>
    {''.join(geoms)}
    {cable}
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return HotSwapStandIn(leds, obstacle, model, data)
