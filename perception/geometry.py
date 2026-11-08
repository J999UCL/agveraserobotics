"""Pixels <-> 3D. All camera maths uses OpenCV axes (+x right, +y down, +z forward)."""
from __future__ import annotations

import numpy as np

# USD (Isaac Sim) and MuJoCo cameras look down their -Z axis with +Y up.
# Right-multiplying their camera pose by this gives the OpenCV-axes pose.
USD_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def make_T(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def cv_pose_from_usd(T_usd: np.ndarray) -> np.ndarray:
    """Convert a camera pose expressed with USD/OpenGL camera axes into OpenCV camera axes."""
    return T_usd @ USD_TO_CV


def backproject(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    """Pixel (u, v) at depth z (along the optical axis) -> point in the camera frame."""
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def transform(T: np.ndarray, p) -> np.ndarray:
    return T[:3, :3] @ np.asarray(p, dtype=float) + T[:3, 3]


def project(p_base, K: np.ndarray, T_base_cam: np.ndarray):
    """Robot-base point -> (u, v, depth). depth <= 0 means behind the camera."""
    T_cam_base = np.linalg.inv(T_base_cam)
    pc = transform(T_cam_base, p_base)
    if pc[2] <= 1e-6:
        return None, None, float(pc[2])
    return float(K[0, 0] * pc[0] / pc[2] + K[0, 2]), float(K[1, 1] * pc[1] / pc[2] + K[1, 2]), float(pc[2])


def depth_at(depth: np.ndarray, u: float, v: float, r: int = 3) -> float | None:
    """Median valid depth in a small window around (u, v)."""
    h, w = depth.shape
    x0, x1 = max(int(u) - r, 0), min(int(u) + r + 1, w)
    y0, y1 = max(int(v) - r, 0), min(int(v) + r + 1, h)
    patch = depth[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(vals)) if vals.size else None
