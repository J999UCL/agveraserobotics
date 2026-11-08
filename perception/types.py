"""Data types shared between the simulator, perception and the agent.

Conventions (agreed API):
  * metres, degrees Celsius
  * every 3D position is in the ROBOT BASE frame
  * camera frame follows OpenCV: +x right, +y down, +z forward (out of the lens)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Capture:
    """One camera frame, as returned by the simulator's capture()."""
    rgb: np.ndarray                  # H x W x 3, uint8
    depth: np.ndarray                # H x W, metres along the optical axis (Isaac: distance_to_image_plane)
    K: np.ndarray                    # 3 x 3 intrinsics
    T_base_cam: np.ndarray           # 4 x 4 pose of the camera (OpenCV axes) in the robot base frame
    thermal: np.ndarray | None = None  # H x W, degrees C, same pixel grid as rgb (optional)
    t: float = 0.0                   # timestamp, seconds


@dataclass
class Detection:
    """One component the camera found."""
    id: str                          # e.g. "C03-U12-D2" (from the inventory), or "tag17" if unknown
    type: str                        # drive | psu | fan | switch | ... | unknown
    tag_id: int | None
    bbox: tuple                      # x, y, w, h in pixels
    xyz_base: list                   # centre of the asset tag, robot base frame
    grasp_xyz_base: list             # where the gripper should go (tag centre + inventory offset)
    normal_base: list                # direction the front face points (towards the viewer)
    temp_c: float | None = None      # hottest reading on the component (95th percentile)
    led: str | None = None           # green | amber | red | blue | None
    status: str = "ok"               # ok | warning | hot | failed | missing
    confidence: float = 1.0          # 0..1: how much to trust this detection
    reasons: list = field(default_factory=list)  # human-readable evidence for status / confidence

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = [int(v) for v in self.bbox]
        return d
