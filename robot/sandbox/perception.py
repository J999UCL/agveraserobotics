"""Local (on-robot) perception: reads the camera and thermal images directly.

No model, no network, milliseconds. This is the "edge" half of perception:
it turns pixels into per-bay LED colour, temperature and obstruction flags.
Argon only gets the images when local perception can't settle a question.

Colours are classified by hue and saturation, not by distance to exact RGB values,
so a dimmer or brighter image still reads correctly; a colour that matches no LED
state reads "unknown" instead of being forced into the nearest class. The MuJoCo
adapter (mujoco_adapter.py) does the same over real renders.
"""

from __future__ import annotations

import colorsys
import io
from dataclasses import dataclass

from PIL import Image

from world import N_BAYS

_EMPTY_V = 0.06       # below this brightness a grey pixel is the empty (black) bay


@dataclass
class BayView:
    bay: int
    led: str          # green | amber | red | off | none (empty bay) | unknown
    temp_c: float     # estimated from thermal colour
    obstructed: bool


def _x(i: int) -> int:
    return 20 + i * 106


def _hsv(rgb):
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    return h * 360, s, v


def classify_led(rgb) -> str:
    h, s, v = _hsv(rgb)
    if s < 0.35:                         # grey: unlit LED, or the empty bay behind it
        return "none" if v < _EMPTY_V else "off"
    if 80 <= h <= 170:
        return "green"
    if 25 <= h < 70:
        return "amber"
    if h < 15 or h > 340:
        return "red"
    return "unknown"


def _is_obstruction(rgb) -> bool:
    h, s, v = _hsv(rgb)                  # yellow marker, at any brightness
    return 45 <= h <= 62 and s > 0.8 and v > 0.3


def read_thermal(thermal_png: bytes) -> list[float]:
    th = Image.open(io.BytesIO(thermal_png)).convert("RGB")
    return [round(30 + 30 * th.getpixel((_x(i) + 60, 80))[0] / 255, 1) for i in range(N_BAYS)]


def read_rack(camera_png: bytes, thermal_png: bytes) -> list[BayView]:
    cam = Image.open(io.BytesIO(camera_png)).convert("RGB")
    th = Image.open(io.BytesIO(thermal_png)).convert("RGB")
    out = []
    for i in range(N_BAYS):
        led = classify_led(cam.getpixel((_x(i) + 47, 179)))
        r = th.getpixel((_x(i) + 60, 80))[0]
        temp = 30 + 30 * r / 255
        obstructed = _is_obstruction(cam.getpixel((_x(i) + 80, 214)))
        out.append(BayView(i, led, round(temp, 1), obstructed))
    return out
