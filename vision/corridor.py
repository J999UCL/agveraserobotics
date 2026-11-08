"""Is the gripper's approach path clear? Simulator-independent: needs only an RGB-D frame with its
camera pose (vision.types.Capture), where the gripper is, and where it's going.

    check = check_path(cap, start_xyz=tool_now, goal_xyz=grab_point)
    check.status      # "clear" | "blocked" | "unknown"
    check.blocked_at  # metres along the path to the first obstruction
    check.reasons     # why, for the log / the agent

The path is a cylinder (radius = gripper half-width + margin) from the gripper to just short of the
goal. Two questions are answered from depth:
  1. Is anything INSIDE the cylinder? -> blocked
  2. Can the camera actually SEE the cylinder? Sample points along it; a sample counts as verified
     free when the camera's ray passes beyond it. If too little of the path is verified (out of view,
     or hidden behind something), the answer is "unknown": slow down and look again, don't assume.

ApproachMonitor runs this every frame during a motion and turns it into continue / slow / stop, so
the robot corrects itself mid-execution when something drops into its way.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .types import Capture


@dataclass
class PathCheck:
    status: str                       # clear | blocked | unknown
    length_m: float                   # length of the checked path (start -> goal minus margins)
    blocked_at: float | None = None   # distance from the start to the first obstruction (m)
    nearest_xyz: list | None = None   # first obstruction point (base frame)
    clearance_m: float | None = None  # closest obstruction point to the path's centre line (m)
    obstacle_px: int = 0              # obstruction pixels inside the path
    verified: float = 0.0             # fraction of the path the camera could see and confirm free
    bbox: tuple | None = None         # obstruction bounding box in pixels
    latency_ms: float = 0.0
    reasons: list = field(default_factory=list)
    rim_uv: list = field(default_factory=list)  # projected path outline, for drawing

    @property
    def clear(self) -> bool:
        return self.status == "clear"


def check_path(cap: Capture, start_xyz, goal_xyz, radius: float = 0.05, goal_margin: float = 0.025,
               start_margin: float = 0.02, gripper_exclude_m: float = 0.03, ignore_mask: np.ndarray | None = None,
               min_obstacle_px: int = 25, min_verified: float = 0.7, step: int = 2, n_axial: int = 24,
               n_around: int = 8) -> PathCheck:
    """radius: gripper half-width plus safety margin. goal_margin: stop checking this far before the
    goal (the part being grabbed is supposed to be there). ignore_mask: H x W bool, True where the
    robot's own fingers appear in a wrist camera (they sit in a fixed spot of the image). Prefer it to
    big distance margins: start_margin / gripper_exclude_m blind the first few cm of the path, and
    something that drops in right in front of the gripper must still be seen.
    min_obstacle_px is in full-resolution pixels."""
    t0 = time.perf_counter()
    a, b = np.asarray(start_xyz, float), np.asarray(goal_xyz, float)
    full = float(np.linalg.norm(b - a))
    if full < 1e-6:
        return PathCheck("clear", 0.0, verified=1.0, reasons=["already at the goal"])
    u = (b - a) / full
    L = full - goal_margin

    # 1) Anything inside the cylinder? Back-project the depth image (every `step`-th pixel).
    h, w = cap.depth.shape
    K = cap.K
    vs, us = np.mgrid[0:h:step, 0:w:step].astype(np.float32)
    z = cap.depth[::step, ::step].astype(np.float32)
    valid = np.isfinite(z) & (z > gripper_exclude_m)
    if ignore_mask is not None:
        valid &= ~np.asarray(ignore_mask, bool)[::step, ::step]
    z = np.where(valid, z, 0)
    pc = np.stack([(us + 0.5 - K[0, 2]) * z / K[0, 0], (vs + 0.5 - K[1, 2]) * z / K[1, 1], z], -1)
    R, tt = cap.T_base_cam[:3, :3].astype(np.float32), cap.T_base_cam[:3, 3].astype(np.float32)
    P = pc @ R.T + tt
    rel = P - a.astype(np.float32)
    along = rel @ u.astype(np.float32)
    radial = np.linalg.norm(rel - along[..., None] * u.astype(np.float32), axis=-1)
    inside = valid & (along > start_margin) & (along < L) & (radial < radius)
    inside = cv2.morphologyEx(inside.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    n_in = int(inside.sum()) * step * step

    # 2) Can the camera see the cylinder? Sample its axis and rim.
    T_cam_base = np.linalg.inv(cap.T_base_cam)
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    ts = np.linspace(max(start_margin, 0.0), L, n_axial)
    angs = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    samples = [a + t * u for t in ts] + [a + t * u + radius * 0.8 * (np.cos(g) * e1 + np.sin(g) * e2)
                                          for t in ts for g in angs]
    S = np.array(samples)
    Sc = S @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    in_front = Sc[:, 2] > gripper_exclude_m
    uu = np.where(in_front, K[0, 0] * Sc[:, 0] / np.maximum(Sc[:, 2], 1e-6) + K[0, 2], -1)
    vv = np.where(in_front, K[1, 1] * Sc[:, 1] / np.maximum(Sc[:, 2], 1e-6) + K[1, 2], -1)
    in_img = in_front & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    seen = np.zeros(len(S), bool)
    idx = np.nonzero(in_img)[0]
    if len(idx):
        zi = cap.depth[vv[idx].astype(int), uu[idx].astype(int)]
        zi = np.where(np.isfinite(zi), zi, np.inf)  # nothing hit = the ray went past the sample
        seen[idx] = zi > Sc[idx, 2] + 0.01            # the ray passes beyond the sample: free there
    verified = float(seen.mean())
    rim = [(float(x), float(y)) for x, y, ok in zip(uu[n_axial:], vv[n_axial:], in_img[n_axial:]) if ok]

    out = PathCheck("clear", round(L, 4), verified=round(verified, 2), rim_uv=rim)
    if n_in >= min_obstacle_px:
        al = along[inside]
        k = int(np.argmin(al))
        pts = P[inside]
        ys, xs = np.nonzero(inside)
        out.status = "blocked"
        out.blocked_at = round(float(al[k]), 4)
        out.nearest_xyz = [round(float(v), 4) for v in pts[k]]
        out.clearance_m = round(float(radial[inside].min()), 4)
        out.obstacle_px = n_in
        out.bbox = (int(xs.min()) * step, int(ys.min()) * step,
                    int(xs.max() - xs.min() + 1) * step, int(ys.max() - ys.min() + 1) * step)
        out.reasons.append(f"something is in the path {out.blocked_at * 100:.0f} cm ahead "
                           f"({out.clearance_m * 100:.1f} cm from its centre line)")
    elif verified < min_verified:
        out.status = "unknown"
        why = "out of the camera's view" if in_img.mean() < min_verified else "hidden behind something"
        out.reasons.append(f"only {verified * 100:.0f}% of the path could be checked ({why}): slow down and look again")
    else:
        out.reasons.append(f"path clear ({verified * 100:.0f}% of it checked)")
    out.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return out


class ApproachMonitor:
    """Call update() every frame while the arm moves; it re-checks the path from where the gripper is
    now and says continue / slow / stop. `confirm` frames in a row are needed before resuming after a
    stop, so a flickering reading doesn't make the arm lurch."""

    def __init__(self, goal_xyz, confirm: int = 2, **check_kw):
        self.goal = np.asarray(goal_xyz, float)
        self.confirm, self.kw = confirm, check_kw
        self.stopped = False
        self._clear_run = 0
        self.history: list[tuple[str, PathCheck]] = []

    def update(self, cap: Capture, tool_xyz) -> tuple[str, PathCheck]:
        chk = check_path(cap, tool_xyz, self.goal, **self.kw)
        if chk.status == "blocked":
            self.stopped, self._clear_run = True, 0
            action = "stop"
        elif chk.status == "unknown":
            self._clear_run = 0
            action = "stop" if self.stopped else "slow"
        else:
            self._clear_run += 1
            if self.stopped and self._clear_run < self.confirm:
                action = "stop"
            else:
                self.stopped = False
                action = "continue"
        self.history.append((action, chk))
        return action, chk


def annotate_path(img: np.ndarray, chk: PathCheck) -> np.ndarray:
    """Draw the path outline (green clear / red blocked / amber unknown) and the obstruction."""
    out = img.copy()
    col = {"clear": (60, 220, 90), "blocked": (255, 40, 40), "unknown": (255, 190, 40)}[chk.status]
    for x, y in chk.rim_uv:
        cv2.circle(out, (int(x), int(y)), 2, col, -1)
    if chk.bbox:
        x, y, w, h = chk.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 40, 40), 2)
    label = {"clear": "PATH CLEAR", "blocked": f"PATH BLOCKED {chk.blocked_at * 100:.0f} cm ahead" if chk.blocked_at else "PATH BLOCKED",
             "unknown": "PATH UNKNOWN: slow down, look again"}[chk.status]
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return out
