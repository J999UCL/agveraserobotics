"""Status-LED detection: small, bright, saturated blobs, classified by hue."""
from __future__ import annotations

import cv2
import numpy as np

# OpenCV hue is 0..180
# Blue LEDs often render/photograph as cyan (the green channel saturates), so "blue" starts at 86.
HUES = {"red": [(0, 7), (172, 180)], "amber": [(8, 33)], "green": [(40, 85)], "blue": [(86, 130)]}


def _colour(h: float) -> str | None:
    for name, ranges in HUES.items():
        if any(lo <= h <= hi for lo, hi in ranges):
            return name
    return None


def find_leds(rgb: np.ndarray, min_area: int = 3, max_area: int = 600, glow_v: int = 240) -> list[tuple[str, float, float]]:
    """Returns [(colour, u, v)]. One pass: bright+saturated mask -> blobs -> median hue per blob.
    A lit LED saturates the sensor (V close to 255); shiny coloured plastic catching the light
    (e.g. blue drive latches) peaks lower, so blobs must reach `glow_v` somewhere."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 2] >= 200) & (hsv[..., 1] >= 120)).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if not (min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area):
            continue
        x, y, w, h = (stats[i, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        sel = labels[y:y + h, x:x + w] == i
        if hsv[y:y + h, x:x + w, 2][sel].max() < glow_v:
            continue  # reflection, not a light
        hue = hsv[y:y + h, x:x + w, 0][sel].astype(np.int32)
        # red wraps around 180, so measure it on a shifted scale
        med = float(np.median((hue + 90) % 180) - 90) % 180 if np.mean((hue < 10) | (hue > 170)) > 0.5 else float(np.median(hue))
        c = _colour(med)
        if c:
            out.append((c, float(cents[i][0]), float(cents[i][1])))
    return out
