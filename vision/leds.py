"""Status-LED detection: small, bright, saturated blobs, classified by hue."""
from __future__ import annotations

import cv2
import numpy as np

# OpenCV hue is 0..180
# Blue LEDs often render/photograph as cyan (the green channel saturates), so "blue" starts at 86.
HUES = {"red": [(0, 7), (172, 180)], "amber": [(8, 33)], "green": [(40, 85)], "blue": [(86, 130)]}


def _colour(h: float, hues: dict | None = None) -> str | None:
    for name, ranges in (hues or HUES).items():
        if any(lo <= h <= hi for lo, hi in ranges):
            return name
    return None


def find_leds(rgb: np.ndarray, min_area: int = 3, max_area: int = 600, glow_v: int = 250,
              glow_px: int = 2, hues: dict | None = None) -> list[tuple[str, float, float]]:
    """Returns [(colour, u, v)]. Bright+saturated mask -> blobs -> per-blob circular-mean hue,
    all vectorised (a rack front can show hundreds of LEDs).
    A lit LED saturates the sensor (V = 255 over several pixels); shiny coloured plastic catching
    the light (e.g. blue drive latches) peaks lower, so a blob needs `glow_px` pixels >= `glow_v`.
    hues: colour table to classify with (default HUES); scenes with extra LED colours pass their own."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 2] >= 200) & (hsv[..., 1] >= 120)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    lab = labels[mask]
    hue = hsv[..., 0][mask].astype(np.float64)
    val = hsv[..., 2][mask]
    glow = np.bincount(lab, weights=(val >= glow_v).astype(np.float64), minlength=n)
    ang = hue * (np.pi / 90.0)  # OpenCV hue 0..180 -> 0..2pi (red wraps around)
    c = np.bincount(lab, weights=np.cos(ang), minlength=n)
    s = np.bincount(lab, weights=np.sin(ang), minlength=n)
    mean_h = (np.degrees(np.arctan2(s, c)) % 360.0) / 2.0
    area = stats[:, cv2.CC_STAT_AREA]
    ok = (area >= min_area) & (area <= max_area) & (glow >= glow_px)
    ok[0] = False
    out = []
    for i in np.nonzero(ok)[0]:
        col = _colour(float(mean_h[i]), hues)
        if col:
            out.append((col, float(cents[i][0]), float(cents[i][1])))
    return out
