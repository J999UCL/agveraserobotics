"""Glue between HotSwapScene.get_wrist_frames() (see api.md) and perception.

    frames = scene.get_wrist_frames()
    cap = capture_from_wrist_frames(frames, pixel_to_base=pixel_to_base)   # axes auto-detected
    scan = HotSwapPerception().scan(cap)

The only unknown in api.md is which axis convention `cam_rot` uses (Isaac/USD cameras look
down -Z with +Y up; OpenCV/ROS look down +Z with +Y down). Pass the scene's own
`pixel_to_base` and this checks both conventions against it and picks the one that agrees.
"""
from __future__ import annotations

import numpy as np

from .geometry import USD_TO_CV, backproject, make_T, transform
from .types import Capture


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _pose(frames) -> np.ndarray:
    rot = np.asarray(frames.cam_rot, float)
    R = quat_wxyz_to_R(rot) if rot.shape == (4,) else rot.reshape(3, 3)
    return make_T(R, np.asarray(frames.cam_pos, float))


def detect_axes(frames, pixel_to_base) -> str:
    """'opencv' or 'usd': whichever makes our maths agree with the scene's pixel_to_base()."""
    depth = np.asarray(frames.depth)
    vs, us = np.nonzero(np.isfinite(depth) & (depth > 0))
    if len(us) == 0:
        raise ValueError("no valid depth pixels to check the camera convention against")
    idx = np.linspace(0, len(us) - 1, 9).astype(int)
    T = _pose(frames)
    K = np.asarray(frames.K, float)
    err = {}
    for name, Tc in (("opencv", T), ("usd", T @ USD_TO_CV)):
        e = 0.0
        for i in idx:
            u, v = float(us[i]), float(vs[i])
            z = float(depth[vs[i], us[i]])
            theirs = np.asarray(pixel_to_base(u, v, z, frames), float)
            e += float(np.linalg.norm(transform(Tc, backproject(u, v, z, K)) - theirs))
        err[name] = e / len(idx)
    best = min(err, key=err.get)
    if err[best] > 0.01:
        raise ValueError(f"neither camera convention matches pixel_to_base (errors {err}); "
                         "check whether pixel coordinates are pixel centres and depth is along the optical axis")
    return best


def capture_from_wrist_frames(frames, cam_axes: str = "auto", pixel_to_base=None, thermal=None, t: float = 0.0) -> Capture:
    """frames: object with .rgb .depth .K .cam_pos .cam_rot (3x3 matrix or wxyz quaternion), robot base frame."""
    T = _pose(frames)
    if cam_axes == "auto":
        cam_axes = detect_axes(frames, pixel_to_base) if pixel_to_base is not None else "usd"
    if cam_axes == "usd":
        T = T @ USD_TO_CV
    elif cam_axes != "opencv":
        raise ValueError("cam_axes must be 'auto', 'opencv' or 'usd'")
    rgb = np.asarray(frames.rgb)[..., :3].astype(np.uint8)
    depth = np.asarray(frames.depth, np.float32)
    return Capture(rgb=rgb, depth=depth, K=np.asarray(frames.K, float), T_base_cam=T, thermal=thermal, t=t)
