"""Thermal camera: reading temperatures, and faking a thermal image in simulation.

Isaac Sim has no thermal camera, so the simulator renders one from its semantic segmentation:
give every component a temperature, paint those into the image, then blur (heat spreads) and
add sensor noise. Say so in the write-up.
"""
from __future__ import annotations

import cv2
import numpy as np


def synth_thermal(segmentation: np.ndarray, temps: dict, ambient: float = 24.0, blur_px: float = 3.0,
                  noise_c: float = 0.4, seed: int | None = None) -> np.ndarray:
    """segmentation: H x W integer ids; temps: {id: degrees C}. Unlisted ids get `ambient`."""
    img = np.full(segmentation.shape, ambient, np.float32)
    for sid, t in temps.items():
        img[segmentation == sid] = t
    if blur_px > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_px)
    rng = np.random.default_rng(seed)
    return img + rng.normal(0.0, noise_c, img.shape).astype(np.float32)


def region_temp(thermal: np.ndarray, bbox, pct: float = 95.0) -> float | None:
    """Hot-spot temperature inside a pixel box (95th percentile ignores single noisy pixels)."""
    x, y, w, h = (int(v) for v in bbox)
    patch = thermal[max(y, 0):y + h, max(x, 0):x + w]
    return float(np.percentile(patch, pct)) if patch.size else None


def colourise(thermal: np.ndarray, lo: float = 20.0, hi: float = 80.0) -> np.ndarray:
    """Thermal image -> RGB heat map for display."""
    norm = np.clip((thermal - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
