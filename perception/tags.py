"""AprilTag asset labels (tag36h11): generate printable/texture images and detect them."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DICT = cv2.aruco.DICT_APRILTAG_36h11
# Texture layout: the black tag (8x8 cells incl. its border) plus a 1-cell white margin = 10 cells.
CELLS_TAG, CELLS_TEXTURE = 8, 10


def tag_texture(tag_id: int, out_dir: str | Path, cell_px: int = 20) -> str:
    """PNG for a component's asset label. If the whole PNG is printed L metres wide,
    pass tag_size = L * 8 / 10 to the detector (the black square's side)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tag36h11_{tag_id:03d}.png"
    if not path.exists():
        img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICT), tag_id, CELLS_TAG * cell_px)
        img = cv2.copyMakeBorder(img, cell_px, cell_px, cell_px, cell_px, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(str(path), img)
    return str(path)


class TagDetector:
    def __init__(self):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)

    def detect(self, rgb: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Returns [(tag_id, corners 4x2 float32)] with corners TL, TR, BR, BL."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None:
            return []
        return [(int(i), c.reshape(4, 2).astype(np.float32)) for c, i in zip(corners, ids.flatten())]


def tag_pose(corners: np.ndarray, tag_size: float, K: np.ndarray):
    """Tag pose in the camera frame -> (R 3x3, t 3) or None.
    IPPE_SQUARE is exact but returns NaN for a perfectly head-on view, so fall back to ITERATIVE."""
    s = tag_size / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    for flag in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_ITERATIVE):
        ok, rvec, tvec = cv2.solvePnP(obj, corners, K, None, flags=flag)
        if ok and np.all(np.isfinite(tvec)) and np.all(np.isfinite(rvec)):
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec.reshape(3)
    return None
