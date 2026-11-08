"""Perception for the HotSwapScene (see api.md): find every drive bay from its status LED,
number the bays like a real server (left to right, top to bottom), say which one is faulty, give
the 3D point to grab, and spot anything that has fallen between the gripper and the rack.

The scene never reveals the faulty bay's coordinates; this module finds them in the camera image.
Robot base frame: +X towards the rack, +Z up, +Y to the robot's left. Runs on-device (numpy + OpenCV).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .geometry import depth_at, project
from .leds import find_leds
from .types import Capture

LED_STATUS = {"green": "ok", "amber": "fault", "blue": "identify", "red": "fault", "off": "unknown"}


@dataclass
class BayObs:
    bay: int | None             # 0..rows*cols-1, left to right, top to bottom; None if the grid didn't match
    row: int | None
    col: int | None
    led: str                    # green | amber | blue | red | off | hidden (something in front of it)
    status: str                 # ok | fault | identify | unknown
    led_xyz_base: list
    tab_xyz_base: list          # where to grab (LED position + calibrated offset)
    uv: tuple                   # pixel of the LED
    confidence: float
    reasons: list = field(default_factory=list)


@dataclass
class Obstacle:
    xyz_base: list              # centroid
    nearest_xyz_base: list      # the point closest to the robot
    clearance_m: float          # how far it sticks out in front of the rack face
    size_m: list                # width (along the rack face) and height
    bbox: tuple
    pixels: int
    span_s: list = field(default_factory=list)   # [min, max] along the rack face ("right"), base frame
    span_z: list = field(default_factory=list)   # [min, max] height


@dataclass
class Scan:
    bays: list[BayObs]
    obstacles: list[Obstacle]
    rack_normal: list | None    # rack face plane: points p with n.p + d = 0, n pointing at the robot
    rack_d: float | None
    latency_ms: float
    notes: list = field(default_factory=list)
    rack_right: list | None = None  # unit vector along the rack face, left to right (base frame)

    def bay(self, i: int) -> BayObs | None:
        return next((b for b in self.bays if b.bay == i), None)

    def path_blocked(self, b: BayObs, corridor: float = 0.08, corridor_z: float | None = None) -> bool:
        """Is anything in the gripper's approach corridor in front of this bay (or hiding its LED)?
        corridor / corridor_z: half-width / half-height of the gripper's approach path."""
        if b.led == "hidden":
            return True
        if not self.rack_right or not self.obstacles:
            return False
        cz = corridor if corridor_z is None else corridor_z
        p = np.asarray(b.tab_xyz_base)
        s = float(p @ np.asarray(self.rack_right))
        return any(o.span_s[0] - corridor <= s <= o.span_s[1] + corridor
                   and o.span_z[0] - cz <= p[2] <= o.span_z[1] + cz for o in self.obstacles)

    def faults(self) -> list[BayObs]:
        return [b for b in self.bays if b.status == "fault"]

    def summary(self) -> str:
        """Compact text for the controller log / supervisor LLM."""
        lines = [f"rack face: {'found' if self.rack_normal else 'NOT found'}; "
                 f"{len(self.bays)} bays; {len(self.obstacles)} obstacle(s); {self.latency_ms:.0f} ms"]
        for b in self.bays:
            if b.status != "ok":
                p = b.tab_xyz_base
                lines.append(f"bay {b.bay}: {b.status.upper()} ({b.led} LED) grab at "
                             f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) conf {b.confidence:.2f}"
                             + (f" [{'; '.join(b.reasons)}]" if b.reasons else ""))
        for o in self.obstacles:
            p = o.nearest_xyz_base
            lines.append(f"OBSTACLE {o.clearance_m * 100:.0f} cm in front of the rack, nearest point "
                         f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), {o.size_m[0] * 100:.0f}x{o.size_m[1] * 100:.0f} cm")
        lines += self.notes
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


def base_points(cap: Capture, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Every `step`-th pixel -> robot base frame. Returns (h x w x 3 points, h x w valid mask)."""
    h, w = cap.depth.shape
    K = cap.K
    vs, us = np.mgrid[0:h:step, 0:w:step].astype(np.float32)
    z = cap.depth[::step, ::step].astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    z = np.where(valid, z, 0)
    pc = np.stack([(us + 0.5 - K[0, 2]) * z / K[0, 0], (vs + 0.5 - K[1, 2]) * z / K[1, 1], z], axis=-1)
    R, t = cap.T_base_cam[:3, :3].astype(np.float32), cap.T_base_cam[:3, 3].astype(np.float32)
    return pc @ R.T + t, valid


def _best_grid(groups: list[np.ndarray], coord: np.ndarray, want: int, descending: bool = False) -> list[np.ndarray]:
    """From candidate rows/columns, choose `want` of them: most LEDs, most even spacing."""
    if len(groups) <= want:
        return groups
    from itertools import combinations
    centres = [float(coord[g].mean()) for g in groups]
    best, best_score = None, -np.inf
    for combo in combinations(range(len(groups)), want):
        count = sum(len(groups[i]) for i in combo)
        c = sorted(centres[i] for i in combo)
        gaps = np.diff(c)
        irregular = float(gaps.std() / gaps.mean()) if want > 2 and gaps.mean() > 0 else 0.0
        score = count * (1.0 - irregular)
        if score > best_score:
            best, best_score = combo, score
    chosen = [groups[i] for i in best]
    return sorted(chosen, key=lambda g: -coord[g].mean() if descending else coord[g].mean())


def _clusters(values: np.ndarray, gap: float) -> list[np.ndarray]:
    """1D clustering: indices grouped where sorted neighbours are closer than `gap`."""
    if len(values) == 0:
        return []
    order = np.argsort(values)
    groups, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if values[b] - values[a] > gap:
            groups.append(np.array(cur))
            cur = []
        cur.append(b)
    groups.append(np.array(cur))
    return groups


class HotSwapPerception:
    def __init__(self, rows: int = 3, cols: int = 4, tab_offset=(0.0, 0.0, 0.0),
                 obstacle_margin: float = 0.03, gripper_exclude_m: float = 0.10, min_obstacle_px: int = 60,
                 seed: int = 0, step: int = 2, window_z: float = 0.05, window_s: float = 0.25):
        """tab_offset: from a bay's LED to the point the gripper should grab, base frame (calibrate once
        with calibrate_tab_offset(); it's a property of the caddy design, not of where the fault is).
        obstacle_margin: how far in front of the drive faces something must stick out to count as an
        obstacle (set it above any fixed parts, e.g. server handles).
        window_z / window_s: half-height / half-width around target_xyz that belongs to one server."""
        self.window_z, self.window_s = window_z, window_s
        self.rows, self.cols = rows, cols
        self.tab_offset = np.asarray(tab_offset, float)
        self.margin = obstacle_margin
        self.gripper_exclude = gripper_exclude_m
        self.min_obstacle_px = min_obstacle_px  # in full-resolution pixels
        self.rng = np.random.default_rng(seed)
        self.step = step

    # ------------------------------------------------------------------ main entry point
    def scan(self, cap: Capture, target_xyz=None) -> Scan:
        """target_xyz: centre of the server's drive area in the base frame (map knowledge, allowed by
        api.md). When given, only that server's LEDs are used, so servers above/below don't confuse
        the bay numbering. Without it, every LED on the rack face is used."""
        return self.scan_many(cap, [target_xyz])[0]

    def scan_many(self, cap: Capture, targets: list) -> list[Scan]:
        """One frame, several servers: the rack face and obstacles are found once, then each target
        server's bays are read. Returns one Scan per target (obstacles shared)."""
        t0 = time.perf_counter()
        notes: list[str] = []
        P, valid = base_points(cap, self.step)  # half resolution is plenty for planes and obstacles
        valid &= cap.depth[::self.step, ::self.step] > self.gripper_exclude  # the gripper's own fingers aren't obstacles
        cam = cap.T_base_cam[:3, 3]
        # "Towards the rack" and "to the right", both horizontal, from the camera's own pose, so this
        # works whichever way the robot base happens to face.
        up = np.array([0.0, 0.0, 1.0])
        f = cap.T_base_cam[:3, 2]
        f_h = f - (f @ up) * up
        if np.linalg.norm(f_h) < 0.3:
            notes.append("camera is looking too steeply up/down to see a rack face")
            return [Scan([], [], None, None, (time.perf_counter() - t0) * 1000, list(notes)) for _ in targets]
        f_h /= np.linalg.norm(f_h)
        r_h = np.cross(f_h, up)

        leds_all = self._led_points(cap)
        plane = self._rack_plane(P[valid], cam, f_h, r_h, np.array([l[3] for l in leds_all]) if leds_all else None)
        if plane is None:
            notes.append("couldn't find the rack face: move so the server fills more of the view")
            return [Scan([], [], None, None, (time.perf_counter() - t0) * 1000, list(notes)) for _ in targets]
        n, d = plane
        dist = P @ n + d  # >0 = in front of the rack face (towards the robot)
        # From here on use the rack's own axes (orthonormal, horizontal): "in" and "right" along the face.
        f_h = -n - (-n @ up) * up
        f_h /= np.linalg.norm(f_h)
        r_h = np.cross(f_h, up)
        per_target = []
        for target in targets:
            tnotes = list(notes)
            bays, led_pts = self._bays(cap, n, d, tnotes, f_h, r_h, target, leds_all)
            per_target.append((bays, led_pts, tnotes))
        all_leds = [p for _, pts, _ in per_target for p in pts]
        if all_leds:  # watch for obstacles over the drive areas we're looking at
            pts = np.array(all_leds)
            sz = np.column_stack([pts @ r_h, pts[:, 2]])
            pad = np.array([0.10, 0.06])  # wider sideways: latches/handles sit beside the LEDs
            lo, hi = sz.min(axis=0) - pad, sz.max(axis=0) + pad
        else:
            on_face = valid & (np.abs(dist) < 0.012)
            sz = np.column_stack([P[on_face] @ r_h, P[on_face][:, 2]])
            lo, hi = np.percentile(sz, 1, axis=0), np.percentile(sz, 99, axis=0)
        obstacles = self._obstacles(cap, P, valid, dist, lo, hi, r_h)
        ms = (time.perf_counter() - t0) * 1000
        nl, rl = [round(float(v), 4) for v in n], [round(float(v), 5) for v in r_h]
        return [Scan(bays, obstacles, nl, round(float(d), 4), ms, tnotes, rl) for bays, _, tnotes in per_target]

    # ------------------------------------------------------------------ rack face
    def _led_points(self, cap: Capture) -> list:
        """Every glowing LED in the frame with its 3D position (base frame): [(colour, u, v, xyz)]."""
        out = []
        R, t = cap.T_base_cam[:3, :3], cap.T_base_cam[:3, 3]
        for colour, u, v in find_leds(cap.rgb):
            z = depth_at(cap.depth, u, v, r=1)
            if z is None:
                continue
            pc = np.array([(u - cap.K[0, 2]) * z / cap.K[0, 0], (v - cap.K[1, 2]) * z / cap.K[1, 1], z])
            out.append((colour, u, v, R @ pc + t))
        return out

    def _rack_plane(self, pts: np.ndarray, cam: np.ndarray, f_h: np.ndarray, r_h: np.ndarray,
                    led_pts: np.ndarray | None = None, bin_m: float = 0.004, min_frac: float = 0.04,
                    tol: float = 0.008):
        """The cart stops square to the cabinet, so flat surfaces facing the robot show up as sharp
        peaks in the histogram of distance along the viewing direction. Candidates: every strong,
        wide peak. Choose the one the status LEDs sit on (server handles in front and cabinet walls
        behind can be bigger); without LEDs, the nearest. Refined with a least-squares plane."""
        cands = []
        if len(pts) < 500:
            return None
        x = pts @ f_h
        lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
        if hi - lo < bin_m:
            hi = lo + bin_m * 4
        hist, edges = np.histogram(x, bins=max(int((hi - lo) / bin_m), 4), range=(lo, hi))
        smooth = np.convolve(hist, [1, 1, 1], mode="same")
        strong = np.nonzero(smooth >= min_frac * len(x))[0]
        i, tried = 0, set()
        for s in strong:  # nearest first
            if s in tried:
                continue
            i = s
            while i + 1 < len(smooth) and smooth[i + 1] >= smooth[i]:  # climb to the top of that peak
                i += 1
            tried.update(range(s, i + 1))
            x0 = 0.5 * (edges[i] + edges[i + 1])
            inl = pts[np.abs(x - x0) < tol]
            if len(inl) < 200:
                continue
            # A server front is wide and flat; a cable or tool close to the camera is not.
            s = inl @ r_h
            wy = np.percentile(s, 95) - np.percentile(s, 5)
            wz = np.percentile(inl[:, 2], 95) - np.percentile(inl[:, 2], 5)
            if wy < 0.12 or wz < 0.03:
                continue
            if len(inl) > 20000:
                inl = inl[self.rng.choice(len(inl), 20000, replace=False)]
            c = inl.mean(axis=0)
            _, evecs = np.linalg.eigh((inl - c).T @ (inl - c))
            n = evecs[:, 0]
            # refine: tabs, LEDs and recessed chassis near the face pull the first fit off; keep tight inliers
            for t2 in (0.004, 0.0025):
                tight = inl[np.abs((inl - c) @ n) < t2]
                if len(tight) < 100:
                    break
                c = tight.mean(axis=0)
                _, evecs = np.linalg.eigh((tight - c).T @ (tight - c))
                n = evecs[:, 0]
            if abs(n @ f_h) < 0.9:  # not facing the camera
                continue
            if n @ (cam - c) < 0:
                n = -n
            if led_pts is None:
                return n, float(-n @ c)  # nearest strong peak
            cands.append((int((np.abs(led_pts @ n - n @ c) < 0.025).sum()), -len(cands), n, float(-n @ c)))
        if not cands:
            return None
        _, _, n, d = max(cands, key=lambda k: (k[0], k[1]))  # most LEDs on it, then nearest
        return n, d

    # ------------------------------------------------------------------ bays from LEDs
    def _bays(self, cap: Capture, n, d, notes, f_h, r_h, target_xyz=None, leds_all=None):
        leds = []
        t = None if target_xyz is None else np.asarray(target_xyz, float)
        for colour, u, v, p in (leds_all if leds_all is not None else self._led_points(cap)):
            if abs(p @ n + d) > 0.025:  # LEDs sit on the rack face
                continue
            if t is not None and (abs(p[2] - t[2]) > self.window_z or abs((p - t) @ r_h) > self.window_s):
                continue  # another server's LED
            leds.append((colour, u, v, p))
        if not leds:
            notes.append("no status LEDs visible" + (" on the target server" if t is not None else ""))
            return [], np.zeros((0, 3))
        pts = np.array([l[3] for l in leds])
        s = pts @ r_h
        row_groups = sorted(_clusters(pts[:, 2], 0.012), key=lambda g: -pts[g, 2].mean())   # top first
        col_groups = sorted(_clusters(s, 0.025), key=lambda g: s[g].mean())                 # left to right
        # More rows/columns than the server has: keep the evenly spaced subset with the most LEDs,
        # treat the rest as stray blobs (reflections, other lights) instead of breaking the numbering.
        row_groups = _best_grid(row_groups, pts[:, 2], self.rows, descending=True)
        col_groups = _best_grid(col_groups, s, self.cols)
        keep = set(int(i) for g in row_groups for i in g) & set(int(i) for g in col_groups for i in g)
        if len(keep) < len(leds):
            notes.append(f"ignored {len(leds) - len(keep)} light(s) that aren't on this server's bay grid")
        grid_ok = len(row_groups) == self.rows and len(col_groups) == self.cols
        if not grid_ok:
            notes.append(f"LED grid is {len(row_groups)}x{len(col_groups)}, expected {self.rows}x{self.cols}: "
                         "bay numbers unknown, move to see the whole server")
        row_of = {int(i): r for r, g in enumerate(row_groups) for i in g}
        col_of = {int(i): c for c, g in enumerate(col_groups) for i in g}
        row_z = [float(pts[g, 2].mean()) for g in row_groups]
        col_s = [float(s[g].mean()) for g in col_groups]

        out, seen = [], set()
        for i, (colour, u, v, p) in enumerate(leds):
            if i not in keep:
                continue
            r, c = (row_of[i], col_of[i]) if grid_ok else (None, None)
            bay = r * self.cols + c if grid_ok else None
            area_conf = 1.0
            reasons = []
            if not grid_ok:
                area_conf *= 0.5
                reasons.append("bay number uncertain")
            if colour in ("amber", "red"):
                reasons.append(f"{colour} LED: fault")
            out.append(BayObs(bay, r, c, colour, LED_STATUS.get(colour, "unknown"),
                              [round(float(x), 4) for x in p], [round(float(x), 4) for x in p + self.tab_offset],
                              (round(u, 1), round(v, 1)), round(area_conf, 2), reasons))
            seen.add(bay)
        if grid_ok:  # bays whose LED is off: position from the grid, flagged unknown
            # Depth of the LED plane as a function of (s, z), fitted to the LEDs we did see.
            kp = pts[sorted(keep)]
            depth = kp @ f_h
            A = np.column_stack([np.ones(len(kp)), kp @ r_h, kp[:, 2]])
            coef = np.linalg.lstsq(A, depth, rcond=None)[0] if len(kp) >= 3 else np.array([depth.mean(), 0, 0])
            up = np.array([0.0, 0.0, 1.0])
            for r in range(self.rows):
                for c in range(self.cols):
                    b = r * self.cols + c
                    if b in seen:
                        continue
                    dz = coef[0] + coef[1] * col_s[c] + coef[2] * row_z[r]
                    p = col_s[c] * r_h + row_z[r] * up + dz * f_h
                    u, v, z_exp = project(p, cap.K, cap.T_base_cam)
                    z_meas = depth_at(cap.depth, u, v, r=2) if u is not None else None
                    if z_meas is not None and z_meas < z_exp - 0.03:
                        led, why = "hidden", (f"LED hidden: something is {(z_exp - z_meas) * 100:.0f} cm in front of it, "
                                              "clear the view and look again")
                    else:
                        led, why = "off", "LED is off: empty bay, powered-down drive, or LED fault"
                    out.append(BayObs(b, r, c, led, "unknown", [round(float(x), 4) for x in p],
                                      [round(float(x), 4) for x in p + self.tab_offset],
                                      (round(u or 0, 1), round(v or 0, 1)), 0.5, [why]))
        return sorted(out, key=lambda b: (b.bay is None, b.bay if b.bay is not None else 0)), pts[sorted(keep)]

    # ------------------------------------------------------------------ obstacles
    def _obstacles(self, cap, P, valid, dist, lo, hi, r_h) -> list[Obstacle]:
        S = P @ r_h
        in_front = (valid & (dist > self.margin) & (dist < 0.6)
                    & (S > lo[0]) & (S < hi[0]) & (P[..., 2] > lo[1]) & (P[..., 2] < hi[1]))
        mask = cv2.morphologyEx(in_front.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        s = self.step
        out = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] * s * s < self.min_obstacle_px:
                continue
            sel = lab == i
            pts, dd = P[sel], dist[sel]
            k = int(np.argmax(dd))
            x, y, w, h = (int(stats[i, j]) * s for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            out.append(Obstacle(
                xyz_base=[round(float(v), 4) for v in pts.mean(axis=0)],
                nearest_xyz_base=[round(float(v), 4) for v in pts[k]],
                clearance_m=round(float(dd.max()), 4),
                size_m=[round(float(np.ptp(pts @ r_h)), 4), round(float(np.ptp(pts[:, 2])), 4)],
                bbox=(x, y, w, h), pixels=int(stats[i, cv2.CC_STAT_AREA]) * s * s,
                span_s=[round(float((pts @ r_h).min()), 4), round(float((pts @ r_h).max()), 4)],
                span_z=[round(float(pts[:, 2].min()), 4), round(float(pts[:, 2].max()), 4)]))
        return sorted(out, key=lambda o: -o.clearance_m)

    # ------------------------------------------------------------------ display
    @staticmethod
    def annotate(cap: Capture, scan: Scan) -> np.ndarray:
        img = cap.rgb.copy()
        col = {"ok": (60, 220, 90), "fault": (255, 160, 0), "identify": (80, 160, 255), "unknown": (170, 170, 170)}
        for b in scan.bays:
            u, v = (int(x) for x in b.uv)
            c = col[b.status]
            cv2.circle(img, (u, v), 9, c, 2)
            cv2.putText(img, f"{b.bay if b.bay is not None else '?'}", (u + 10, v + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
            if b.status == "fault":
                cv2.putText(img, "FAULT", (u - 20, v - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
        for o in scan.obstacles:
            x, y, w, h = o.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 40, 40), 2)
            cv2.putText(img, f"OBSTACLE {o.clearance_m * 100:.0f}cm out", (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, f"perception {scan.latency_ms:.0f} ms", (8, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img


def calibrate_tab_offset(scan: Scan, debug_tab_xyz: dict) -> np.ndarray:
    """One-off calibration (debug API allowed for this): average offset from each bay's LED to its
    true tab centre, e.g. debug_tab_xyz = {bay: scene.debug_bay_pose(cab, bay).tab_centre}."""
    offs = [np.asarray(debug_tab_xyz[b.bay]) - np.asarray(b.led_xyz_base)
            for b in scan.bays if b.bay in debug_tab_xyz and b.led != "off"]
    return np.mean(offs, axis=0) if offs else np.zeros(3)
