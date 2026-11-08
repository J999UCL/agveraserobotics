"""Tray perception for the dcsim robot: the wrist camera looks straight down into an open server tray.

    scan = TrayPerception().scan(cap, slot_leds, grasp_points=None)
    scan.slots["A3"].led      # green | amber | red | lime | blue | off | hidden
    scan.slots["A3"].status   # ok | fault | verifying | identify | off | unknown
    scan.obstacles            # anything lying across the open tray (from depth)
    scan.summary()            # short text for the log / the agent

Inputs: one wrist frame (capture_rgbd("wrist") dict, or a vision Capture) and get_slot_leds(server),
the world position of each slot's LED (map knowledge, like a rack diagram). Optionally
get_slot_grasp_points(server), which adds a per-slot "is the gripper's way down clear?" check.

How each slot is read:
  * glowing blobs from leds.find_leds() are matched to the nearest projected slot LED; blobs that
    aren't near a known slot (other lights in view) are ignored
  * a slot with no blob: if depth shows something in front of where its LED should be, it's
    "hidden" (e.g. a taller RAM stick, or a cable); otherwise the pixel there is dark ("off": empty
    slot or unpowered) or lit but dim (classified by colour, lower confidence)
  * lime (a new part settling in, ~3 s) is "verifying", never a fault
Obstacles (needs grasp points): each slot's way down is checked in depth (corridor.check_path);
slots blocked by the same thing are grouped into one obstacle.
Everything runs on-device (numpy + OpenCV).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .corridor import check_path, prepare_points
from .geometry import USD_TO_CV, make_T, project
from .leds import _colour, find_leds
from .types import Capture

STATUS = {"green": "ok", "amber": "fault", "red": "fault", "lime": "verifying", "blue": "identify",
          "off": "off", "hidden": "unknown"}
# dcsim LED colours, measured from its renders: amber ~25, lime (new part settling in) ~30, green ~67
TRAY_HUES = {"red": [(0, 7), (172, 180)], "amber": [(8, 27)], "lime": [(28, 39)], "green": [(40, 85)], "blue": [(86, 130)]}
DRAW = {"ok": (60, 220, 90), "fault": (255, 160, 0), "verifying": (200, 255, 60), "identify": (80, 160, 255),
        "off": (150, 150, 150), "unknown": (255, 255, 255)}


@dataclass
class SlotObs:
    slot: str
    led: str
    status: str
    confidence: float
    reasons: list = field(default_factory=list)
    uv: tuple | None = None              # where the LED is in the image
    blocked: bool = False                # gripper's way down to it is obstructed (if checked)


@dataclass
class TrayObstacle:
    xyz: list                            # centroid, world frame
    top_z: float                         # highest point
    above_parts_m: float                 # how far above the part(s) it lies
    size_m: list                         # spread of the blocking points in x and y
    bbox: tuple                          # pixels
    over_slots: list                     # slots it lies over


@dataclass
class TrayScan:
    slots: dict
    obstacles: list
    latency_ms: float
    notes: list = field(default_factory=list)

    def faults(self) -> list[SlotObs]:
        return [s for s in self.slots.values() if s.status == "fault"]

    def led(self, slot: str) -> str:
        return self.slots[slot].led if slot in self.slots else "hidden"

    def summary(self) -> str:
        n_ok = sum(s.status == "ok" for s in self.slots.values())
        lines = [f"tray: {len(self.slots)} slots read ({n_ok} ok), {len(self.obstacles)} obstacle(s), "
                 f"{self.latency_ms:.0f} ms"]
        for s in self.slots.values():
            if s.status != "ok" or s.blocked:
                lines.append(f"{s.slot}: {s.status.upper()} ({s.led} LED, conf {s.confidence:.2f})"
                             + (" BLOCKED" if s.blocked else "") + (f" [{'; '.join(s.reasons)}]" if s.reasons else ""))
        for o in self.obstacles:
            lines.append(f"OBSTACLE across the tray over {', '.join(o.over_slots) or 'no slot'}: "
                         f"up to {o.above_parts_m * 100:.0f} cm above the parts")
        return "\n".join(lines + self.notes)

    def to_dict(self) -> dict:
        return asdict(self)


def capture_from_dcsim(cap: dict) -> Capture:
    """capture_rgbd("wrist") -> Capture in the world frame (dcsim cameras use MuJoCo axes: -Z forward)."""
    T = make_T(np.asarray(cap["cam_rot"], float), np.asarray(cap["cam_pos"], float)) @ USD_TO_CV
    return Capture(np.asarray(cap["rgb"])[..., :3].astype(np.uint8), np.asarray(cap["depth"], np.float32),
                   np.asarray(cap["K"], float), T)


class TrayPerception:
    def __init__(self, match_px: float = 14.0, depth_tol: float = 0.012, dark_v: int = 70,
                 part_margin: float = 0.02, footprint_pad: float = 0.04, min_obstacle_px: int = 40,
                 path_radius: float = 0.03, approach_from: float = 0.30, hues: dict | None = None,
                 step: int = 4):
        """match_px: max distance from a projected slot LED for a blob to count as that LED (RAM LEDs
        are ~35 px apart). dark_v: below this brightness an LED is off. part_margin: how far above the
        tallest part something must stand to count as an obstacle."""
        self.match_px, self.depth_tol, self.dark_v = match_px, depth_tol, dark_v
        self.part_margin, self.footprint_pad, self.min_obstacle_px = part_margin, footprint_pad, min_obstacle_px
        self.path_radius, self.approach_from = path_radius, approach_from
        self.hues = hues or TRAY_HUES
        self.step = step  # depth subsampling for the path checks (a cable is ~20 px wide from the view pose)

    # ------------------------------------------------------------------ main entry point
    def scan(self, cap, slot_leds: dict, grasp_points: dict | None = None) -> TrayScan:
        t0 = time.perf_counter()
        frame = cap if isinstance(cap, Capture) else capture_from_dcsim(cap)
        h, w = frame.depth.shape
        notes: list[str] = []

        proj = {}
        for slot, p in slot_leds.items():
            u, v, z = project(p, frame.K, frame.T_base_cam)
            if u is not None and 0 <= u < w and 0 <= v < h:
                proj[slot] = (u, v, z)
        missing = [s for s in slot_leds if s not in proj]
        if missing:
            notes.append(f"not in view: {', '.join(missing)}")

        # match glowing blobs to slots (nearest projected LED, within match_px); ignore the rest
        blobs = find_leds(frame.rgb, hues=self.hues)
        best: dict[str, tuple[float, str, float, float]] = {}
        stray = 0
        for colour, u, v in blobs:
            d, slot = min(((np.hypot(u - pu, v - pv), s) for s, (pu, pv, _) in proj.items()), default=(np.inf, None))
            if slot is None or d > self.match_px:
                stray += 1
                continue
            if slot not in best or d < best[slot][0]:
                best[slot] = (d, colour, u, v)
        if stray:
            notes.append(f"ignored {stray} light(s) that aren't slot LEDs")

        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        slots = {}
        for slot, (pu, pv, z_exp) in proj.items():
            if slot in best:
                _, colour, u, v = best[slot]
                slots[slot] = SlotObs(slot, colour, STATUS[colour], 1.0, self._led_reason(colour), (round(u, 1), round(v, 1)))
            else:
                slots[slot] = self._read_dark(slot, pu, pv, z_exp, frame, hsv)

        # Obstacles: check each slot's own way down (the tray's fixed walls and brackets stand taller
        # than a fallen cable would, so one "higher than the parts" threshold can't separate them).
        obstacles: list[TrayObstacle] = []
        if grasp_points:
            hits = {}
            pts = prepare_points(frame, step=self.step)  # back-project once, shared by every slot's check
            for slot, g in grasp_points.items():
                if slot not in slots:
                    continue
                g = np.asarray(g, float)
                chk = check_path(frame, g + [0, 0, self.approach_from], g, radius=self.path_radius, points=pts, step=self.step)
                if chk.status == "blocked":
                    slots[slot].blocked = True
                    above = chk.nearest_xyz[2] - g[2]
                    slots[slot].reasons.append(f"something is lying over it, {above * 100:.0f} cm above the part")
                    hits[slot] = (chk, above)
            obstacles = self._group(hits)
        else:
            notes.append("no grasp points given: obstacles over the tray not checked")
        return TrayScan(slots, obstacles, (time.perf_counter() - t0) * 1000, notes)

    @staticmethod
    def _group(hits: dict) -> list[TrayObstacle]:
        """Slots blocked by the same thing (overlapping image boxes, or blocking points within 15 cm of
        each other: one long cable can lie over several slots) -> one obstacle each."""
        groups: list[list[str]] = []
        for slot, (chk, _) in hits.items():
            x, y, w, h = chk.bbox
            p = np.asarray(chk.nearest_xyz)
            for g in groups:
                if any((x < hits[o][0].bbox[0] + hits[o][0].bbox[2] and hits[o][0].bbox[0] < x + w
                        and y < hits[o][0].bbox[1] + hits[o][0].bbox[3] and hits[o][0].bbox[1] < y + h)
                       or np.linalg.norm(p - np.asarray(hits[o][0].nearest_xyz)) < 0.15 for o in g):
                    g.append(slot)
                    break
            else:
                groups.append([slot])
        out = []
        for g in groups:
            boxes = np.array([hits[s][0].bbox for s in g])
            x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
            x1, y1 = (boxes[:, 0] + boxes[:, 2]).max(), (boxes[:, 1] + boxes[:, 3]).max()
            pts = np.array([hits[s][0].nearest_xyz for s in g])
            out.append(TrayObstacle([round(float(c), 4) for c in pts.mean(axis=0)], round(float(pts[:, 2].max()), 4),
                                    round(float(max(hits[s][1] for s in g)), 4),
                                    [round(float(np.ptp(pts[:, 0])), 4), round(float(np.ptp(pts[:, 1])), 4)],
                                    (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), sorted(g)))
        return out

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _led_reason(colour: str) -> list:
        return {"amber": ["amber LED: fault"], "red": ["red LED: fault"], "lime": ["lime LED: new part verifying"],
                "blue": ["blue LED: identify"]}.get(colour, [])

    def _read_dark(self, slot, u, v, z_exp, frame, hsv) -> SlotObs:
        """No glowing blob at this slot: hidden, off, or a dim LED."""
        h, w = frame.depth.shape
        r = 3
        x0, x1, y0, y1 = max(int(u) - r, 0), min(int(u) + r + 1, w), max(int(v) - r, 0), min(int(v) + r + 1, h)
        dep = frame.depth[y0:y1, x0:x1]
        on_led = np.isfinite(dep) & (np.abs(dep - z_exp) < self.depth_tol)
        in_front = np.isfinite(dep) & (dep < z_exp - self.depth_tol)
        uv = (round(u, 1), round(v, 1))
        if on_led.sum() < 3 and in_front.mean() > 0.5:
            return SlotObs(slot, "hidden", "unknown", 0.4,
                           [f"LED hidden: something is {(z_exp - float(np.median(dep[in_front]))) * 100:.0f} cm in front of it"], uv)
        px = hsv[y0:y1, x0:x1][on_led] if on_led.any() else hsv[y0:y1, x0:x1].reshape(-1, 3)
        k = int(np.argmax(px[:, 2].astype(int) + px[:, 1].astype(int)))
        hh, ss, vv = (int(c) for c in px[k])
        if vv < self.dark_v or ss < 80:
            return SlotObs(slot, "off", "off", 0.9, ["LED off: empty slot, or no power"], uv)
        colour = _colour(hh, self.hues) or "off"
        return SlotObs(slot, colour, STATUS[colour], 0.6, self._led_reason(colour) + ["dim LED: read by colour only"], uv)

    # ------------------------------------------------------------------ display
    @staticmethod
    def annotate(rgb: np.ndarray, scan: TrayScan) -> np.ndarray:
        img = np.ascontiguousarray(rgb[..., :3]).copy()
        for s in scan.slots.values():
            if s.uv is None:
                continue
            u, v = (int(c) for c in s.uv)
            col = DRAW[s.status]
            cv2.circle(img, (u, v), 9, (255, 40, 40) if s.blocked else col, 2)
            cv2.putText(img, s.slot, (u - 9, v - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        for o in scan.obstacles:
            x, y, w, h = o.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 40, 40), 2)
            cv2.putText(img, "OBSTACLE", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 40, 40), 1, cv2.LINE_AA)
        return img
