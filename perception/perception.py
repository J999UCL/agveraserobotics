"""detect(capture) -> components with 3D positions (robot base frame), temperature, LED and status.

Evidence used, all on-device (OpenCV + numpy, no network):
  * AprilTag asset labels  -> identity (via inventory) and precise 3D pose
  * depth image            -> cross-checks the tag pose; finds empty bays
  * thermal image          -> hot components
  * status LEDs            -> amber = warning, red = failed
Every status and confidence comes with human-readable `reasons`, so the agent's decisions can
be traced back to what the camera saw.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .geometry import backproject, depth_at, project, transform
from .leds import find_leds
from .tags import TagDetector, tag_pose
from .thermal import colourise, region_temp
from .types import Capture, Detection

DEFAULT_LIMITS_C = {"drive": 55.0, "psu": 70.0, "fan": 60.0, "switch": 65.0, "server": 75.0, "unknown": 60.0}
STATUS_RANK = {"ok": 0, "warning": 1, "hot": 2, "failed": 3, "missing": 3}
def _diagonal_centre(c: np.ndarray) -> tuple[float, float]:
    """Image of the tag's true centre = where its diagonals cross (corner mean is biased in perspective)."""
    p1, p3, p2, p4 = c[0], c[2], c[1], c[3]
    d1, d2 = p3 - p1, p4 - p2
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        m = c.mean(axis=0)
        return float(m[0]), float(m[1])
    s = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    p = p1 + s * d1
    return float(p[0]), float(p[1])


STATUS_RGB = {"ok": (60, 220, 90), "warning": (255, 190, 40), "hot": (255, 90, 30),
              "failed": (255, 40, 40), "missing": (200, 60, 255)}


class Perception:
    """
    inventory: {tag_id: {"id": "C03-U12-D2", "type": "drive",
                         "face": [w, h],            # front-face size in metres (default 3x2 tag sizes)
                         "face_offset": [dx, dy],   # face centre relative to tag centre (x right, y up)
                         "grasp_offset": [x, y, z], # gripper target relative to tag centre (z = out of the face)
                         "temp_limit": 55}}
    tag_size: side of the black AprilTag square in metres.
    """

    def __init__(self, inventory: dict | None = None, tag_size: float = 0.024, temp_limits: dict | None = None):
        self.inventory = {int(k): v for k, v in (inventory or {}).items()}
        self.tag_size = tag_size
        self.limits = {**DEFAULT_LIMITS_C, **(temp_limits or {})}
        self.tags = TagDetector()
        self.last_latency_ms = 0.0

    @classmethod
    def from_json(cls, path: str | Path, **kw) -> "Perception":
        data = json.loads(Path(path).read_text())
        return cls(inventory=data.get("inventory", {}), tag_size=data.get("tag_size", 0.024), **kw)

    # ------------------------------------------------------------------ main entry point
    def detect(self, cap: Capture) -> list[Detection]:
        t0 = time.perf_counter()
        leds = find_leds(cap.rgb)
        R_bc, dets = cap.T_base_cam[:3, :3], []
        for tag_id, corners in self.tags.detect(cap.rgb):
            pose = tag_pose(corners, self.tag_size, cap.K)
            if pose is None:
                continue
            R_ct, t_ct = pose
            info = self.inventory.get(tag_id, {})
            ctype = info.get("type", "unknown")
            reasons, conf = [], 1.0

            # How well can we see the tag?
            side_px = float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)))
            conf *= float(np.clip((side_px - 10.0) / 20.0, 0.2, 1.0))
            if side_px < 20:
                reasons.append(f"tag only {side_px:.0f}px wide, move closer to confirm")
            # Which way the part faces: a plane fit on the depth image is far steadier than the
            # orientation of a tag that's only ~20 px wide; fall back to the tag if depth is poor.
            normal_c = self._depth_normal(cap, corners)
            if normal_c is None:
                normal_c = R_ct[:, 2]
            view = -t_ct / np.linalg.norm(t_ct)
            angle = float(np.degrees(np.arccos(np.clip(normal_c @ view, -1, 1))))
            if angle > 55:
                conf *= 0.6
                reasons.append(f"viewing the label at {angle:.0f} deg, oblique")
            u, v = _diagonal_centre(corners)
            z = depth_at(cap.depth, u, v)
            if z is None:
                conf *= 0.8
                reasons.append("no depth reading at the label")
            elif abs(z - t_ct[2]) > 0.03:
                conf *= 0.5
                reasons.append(f"label pose and depth disagree by {abs(z - t_ct[2]) * 100:.1f} cm")
            else:
                # Depth is more precise than a small tag's PnP range; keep PnP for orientation.
                t_ct = backproject(u, v, z, cap.K)
            if tag_id not in self.inventory:
                conf *= 0.7
                reasons.append(f"tag {tag_id} is not in the inventory")

            # 3D: tag centre, grasp point, face region. Parts are mounted upright, so the face frame
            # is built from the label's normal + gravity, whichever way round the label was stuck on.
            F = self._face_frame(normal_c, cap)
            p_base = transform(cap.T_base_cam, t_ct)
            g_off = np.asarray(info.get("grasp_offset", [0.0, 0.0, 0.0]), float)
            g_base = transform(cap.T_base_cam, t_ct + F @ g_off)
            bbox = self._face_bbox(info, F, t_ct, cap)

            # Status evidence (temperature only from pixels on this part's own front face)
            temp = self._face_temp(cap, bbox, t_ct, normal_c) if cap.thermal is not None else None
            led = self._led_for(leds, bbox)
            limit = float(info.get("temp_limit", self.limits.get(ctype, self.limits["unknown"])))
            status = "ok"
            if led == "amber":
                status = "warning"
                reasons.append("amber fault LED is on")
            if temp is not None and temp >= limit:
                status = "hot"
                reasons.append(f"thermal {temp:.1f} C is over its {limit:.0f} C limit")
            if led == "red":
                status = "failed"
                reasons.append("red fault LED is on")

            dets.append(Detection(
                id=info.get("id", f"tag{tag_id}"), type=ctype, tag_id=tag_id, bbox=bbox,
                xyz_base=[round(float(x), 4) for x in p_base],
                grasp_xyz_base=[round(float(x), 4) for x in g_base],
                normal_base=[round(float(x), 4) for x in R_bc @ normal_c],
                temp_c=None if temp is None else round(temp, 1), led=led, status=status,
                confidence=round(conf, 2), reasons=reasons))
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return sorted(dets, key=lambda d: d.id)

    def find_missing(self, cap: Capture, expected: dict, detections: list[Detection],
                     margin_px: int = 12) -> list[Detection]:
        """Check components that should be in view but weren't detected.
        expected: {component_id: [x, y, z]} = where its tag centre is when installed (base frame).
        Returns Detections with status "missing" (bay is empty) or "warning" (something is there
        but its label can't be read). Components blocked from view are skipped."""
        seen = {d.id for d in detections}
        h, w = cap.depth.shape
        out = []
        for cid, p in expected.items():
            if cid in seen:
                continue
            u, v, z_exp = project(p, cap.K, cap.T_base_cam)
            if u is None or not (margin_px <= u < w - margin_px and margin_px <= v < h - margin_px):
                continue
            z_meas = depth_at(cap.depth, u, v, r=4)
            if z_meas is None:
                continue
            box = (int(u) - 10, int(v) - 10, 20, 20)
            if z_meas > z_exp + 0.02:
                out.append(Detection(cid, "unknown", None, box, list(map(float, p)), list(map(float, p)), [0, 0, 0],
                                     status="missing", confidence=0.9,
                                     reasons=[f"bay is empty: surface is {(z_meas - z_exp) * 100:.1f} cm deeper than an installed part"]))
            elif abs(z_meas - z_exp) <= 0.015:
                out.append(Detection(cid, "unknown", None, box, list(map(float, p)), list(map(float, p)), [0, 0, 0],
                                     status="warning", confidence=0.4,
                                     reasons=["something is in the bay but its label can't be read"]))
        return out

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _depth_normal(cap: Capture, corners: np.ndarray, min_pts: int = 40) -> np.ndarray | None:
        """Fit a plane to the depth pixels on the label; return its normal (camera frame) facing the camera."""
        h, w = cap.depth.shape
        x0, y0 = np.floor(corners.min(axis=0)).astype(int)
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
        if x1 <= x0 or y1 <= y0:
            return None
        vs, us = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        z = cap.depth[y0:y1 + 1, x0:x1 + 1]
        ok = np.isfinite(z) & (z > 0)
        if ok.sum() < min_pts:
            return None
        z, us, vs = z[ok], us[ok] + 0.5, vs[ok] + 0.5
        K = cap.K
        P = np.column_stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z])
        c = P.mean(axis=0)
        Q = P - c
        evals, evecs = np.linalg.eigh(Q.T @ Q)  # 3x3: cheap and deterministic
        if evals[0] > 0.06 * evals[1]:  # not flat enough to trust
            return None
        n = evecs[:, 0]
        return -n if n @ c > 0 else n

    @staticmethod
    def _face_temp(cap: Capture, bbox, p_c: np.ndarray, n_c: np.ndarray, tol: float = 0.015) -> float | None:
        """95th-percentile temperature of the pixels that lie on the face plane, so neighbouring
        parts seen at an angle (or the rack behind) don't leak into the reading."""
        x, y, w, h = bbox
        # inner 70% of the face: edges pick up heat bleeding from neighbours
        x, y, w, h = x + int(0.15 * w), y + int(0.15 * h), max(int(0.7 * w), 1), max(int(0.7 * h), 1)
        z = cap.depth[y:y + h, x:x + w]
        vs, us = np.mgrid[y:y + h, x:x + w]
        K = cap.K
        P = np.stack([(us + 0.5 - K[0, 2]) * z / K[0, 0], (vs + 0.5 - K[1, 2]) * z / K[1, 1], z], axis=-1)
        on_face = np.isfinite(z) & (np.abs((P - p_c) @ n_c) < tol)
        vals = cap.thermal[y:y + h, x:x + w][on_face]
        if vals.size < 10:
            return region_temp(cap.thermal, bbox)
        return float(np.median(vals))  # a faulty part is hot across its face; median ignores bleed

    @staticmethod
    def _face_frame(normal_c: np.ndarray, cap: Capture) -> np.ndarray:
        """Columns: face x (right, as seen from the robot), y (up), z (out of the face), camera frame."""
        up_c = cap.T_base_cam[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        x = np.cross(up_c, normal_c)
        if np.linalg.norm(x) < 1e-6:  # looking straight down at a horizontal label
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(normal_c, x)
        return np.column_stack([x, y, normal_c])

    def _face_bbox(self, info: dict, F: np.ndarray, t_ct, cap: Capture):
        fw, fh = info.get("face", [3 * self.tag_size, 2 * self.tag_size])
        dx, dy = info.get("face_offset", [0.0, 0.0])
        pts = []
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                pc = t_ct + F @ np.array([dx + sx * fw, dy + sy * fh, 0.0])
                pts.append([cap.K[0, 0] * pc[0] / pc[2] + cap.K[0, 2], cap.K[1, 1] * pc[1] / pc[2] + cap.K[1, 2]])
        pts = np.array(pts)
        x0, y0 = np.floor(pts.min(axis=0)).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
        h, w = cap.rgb.shape[:2]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
        return (int(x0), int(y0), int(max(x1 - x0, 1)), int(max(y1 - y0, 1)))

    @staticmethod
    def _led_for(leds, bbox, pad: int = 3) -> str | None:
        x, y, w, h = bbox
        inside = [(c, u, v) for c, u, v in leds if x - pad <= u <= x + w + pad and y - pad <= v <= y + h + pad]
        if not inside:
            return None
        worst = {"red": 3, "amber": 2, "blue": 1, "green": 0}
        return max(inside, key=lambda l: worst[l[0]])[0]

    # ------------------------------------------------------------------ for the agent / display
    @staticmethod
    def summary(dets: list[Detection]) -> str:
        """Compact text for the agent / LLM prompt."""
        lines = []
        for d in dets:
            t = f" {d.temp_c:.0f}C" if d.temp_c is not None else ""
            why = f" ({'; '.join(d.reasons)})" if d.reasons else ""
            lines.append(f"{d.id} [{d.type}] {d.status.upper()}{t} at "
                         f"({d.xyz_base[0]:.3f}, {d.xyz_base[1]:.3f}, {d.xyz_base[2]:.3f}) conf {d.confidence:.2f}{why}")
        return "\n".join(lines)

    @staticmethod
    def annotate(cap: Capture, dets: list[Detection]) -> np.ndarray:
        """RGB (and thermal, if present) with boxes coloured by status."""
        img = cap.rgb.copy()
        for d in dets:
            x, y, w, h = d.bbox
            col = STATUS_RGB[d.status]
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            label = f"{d.id} {d.status}" + (f" {d.temp_c:.0f}C" if d.temp_c is not None else "")
            cv2.putText(img, label, (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        if cap.thermal is None:
            return img
        return np.concatenate([img, colourise(cap.thermal)], axis=1)
