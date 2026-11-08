"""Plug the agent into the team's MuJoCo data-centre scene (sim/).

MujocoRack is a drop-in replacement for World: same methods, same return
strings, so agent.py / scenarios.py work unchanged.

Mapping (prototype):
  * our 8 "bays"      -> 8 servers in rack R04, slots 6..13 (U13..U27)
  * pull_caddy        -> extend that server's tray under the rail service force;
                         JAMMED is detected PHYSICALLY (tray fails to travel)
  * insert_caddy      -> close the tray
  * status LEDs       -> shown on each server's drive LEDs (perception/datacentre.py adds one
                         controllable LED per drive carrier at load time; sim/ is not modified)
  * obstructions      -> a physical loose cable hung in front of the bay's drive latches
  * perceive()        -> on-robot camera perception (perception/): LEDs found in the image and
                         placed in 3D with depth, obstacles found from depth. Nothing is read from
                         the simulator's ground truth except "caddy removed", which the robot knows
                         because it pulled it itself.
  * camera_png        -> MuJoCo render of rack R04's front, annotated with what perception found
  * thermal_png       -> synthetic (MuJoCo has no thermal camera), from server state
  * SMART / RAID      -> logical server state inherited from World (not physics)

Not covered by the sim yet: the arm is parked (no motion control), and drives are
not individually removable, so the "swap" is a tray-level service action.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import os
import sys

import cv2
import mujoco
import numpy as np
from PIL import Image

from world import World

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import run as simrun  # noqa: E402  (team's scene helpers: load, drive, settled)


def _rack_perception(module: str):
    """Load the repo-root perception package under another name: this folder already has a
    perception.py (local pixel-reading helpers), so a plain `import perception` would clash."""
    name = "rack_perception"
    if name not in sys.modules:
        pkg = os.path.join(ROOT, "perception")
        spec = importlib.util.spec_from_file_location(name, os.path.join(pkg, "__init__.py"),
                                                      submodule_search_locations=[pkg])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(f"{name}.{module}")


dc = _rack_perception("datacentre")
project = _rack_perception("geometry").project

RACK = 4
FIRST_SLOT = 6
OPEN, CLOSED = 0.70, 0.0
SERVICE_STEPS = 1500      # 3 s of sim time per tray move
VIEW_W, VIEW_H = 1280, 960
VIEW_STANDOFF = 1.1       # further back than a fully extended tray (0.72 m), so it never blocks the view
AGENT_IMAGE = (960, 720)  # size of the annotated image handed to the agent
LED_BGR = {"green": (60, 220, 90), "amber": (255, 170, 0), "red": (255, 50, 50), "blue": (80, 160, 255),
           "off": (150, 150, 150), "unknown": (200, 200, 200), "none": (120, 120, 120)}


class MujocoRack(World):
    def __init__(self, base: World, frames_dir: str | None = None):
        super().__init__(**{k: getattr(base, k) for k in base.__dataclass_fields__})
        self.m, self.d = dc.load(cable_mocap=True)
        self.targets: dict[str, float] = {}
        self.frames_dir = frames_dir
        self.frame_no = 0
        slots = [self._slot(i) for i in range(len(self.bays))]
        self.view = dc.rack_view_pose(self.m, self.d, RACK, slots, standoff=VIEW_STANDOFF)
        # Where each server is (map knowledge, trays closed); where its fault is comes from the camera.
        self.server_targets = [dc.server_target(self.m, self.d, RACK, s) for s in slots]
        self.vision = dc.make_perception(self.m, self.d, step=4)
        self.last_scans = None
        self._cache = None

    # ---------------------------------------------------------------- mapping
    def _slot(self, bay: int) -> int:
        return FIRST_SLOT + bay

    def _ident(self, bay: int) -> str:
        return f"r{RACK:02}_u{(FIRST_SLOT + bay) * 2 + 1:02}"

    def _joint(self, bay: int) -> str:
        return self._ident(bay) + "_slide"

    def tray_pos(self, bay: int) -> float:
        return float(self.d.joint(self._joint(bay)).qpos[0])

    def _move_tray(self, bay: int, target: float) -> bool:
        self.targets[self._joint(bay)] = target
        for _ in range(SERVICE_STEPS):
            simrun.drive(self.m, self.d, self.targets)
        return simrun.settled(self.m, self.d, {self._joint(bay): target}, 0.02)

    def _sync_scene(self):
        """Put the world's state into the physical scene: drive LEDs and the loose cable."""
        for i, b in enumerate(self.bays):
            dc.show_server_status(self.m, RACK, self._slot(i), b.led)
        if self.obstacle_at is None:
            dc.park_cable(self.m, self.d)
        else:
            dc.place_cable(self.m, self.d, RACK, self._slot(self.obstacle_at))
        mujoco.mj_forward(self.m, self.d)

    def _save_frame(self, png: bytes, tag: str):
        if self.frames_dir:
            os.makedirs(self.frames_dir, exist_ok=True)
            self.frame_no += 1
            with open(os.path.join(self.frames_dir, f"{self.frame_no:03d}_{tag}.png"), "wb") as f:
                f.write(png)

    # ------------------------------------------------------------------ views
    def _look(self):
        """Render the rack camera (RGB + depth) and run on-robot perception on it."""
        self._sync_scene()
        key = (round(float(self.d.time), 6), tuple(b.led for b in self.bays), self.obstacle_at,
               tuple(round(self.tray_pos(i), 3) for i in range(len(self.bays))))
        if self._cache is None or self._cache[0] != key:
            cap = dc.capture(self.m, self.d, *self.view, width=VIEW_W, height=VIEW_H)
            self._cache = (key, cap, self.vision.scan_many(cap, self.server_targets))
        return self._cache[1], self._cache[2]

    def camera_png(self) -> bytes:
        cap, scans = self._look()
        img = Image.fromarray(self._annotate(cap, scans)).resize(AGENT_IMAGE, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        self._save_frame(png, "camera")
        return png

    def _annotate(self, cap, scans) -> np.ndarray:
        """Draw what perception concluded (not ground truth): per-bay status, blocked bays,
        obstacles it found, plus the arm's own position (proprioception)."""
        img = cap.rgb.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, f"rack R04 front camera | labels = on-robot perception "
                         f"({scans[0].latency_ms:.0f} ms, no ground truth)",
                    (8, 21), font, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        for i, sc in enumerate(scans):
            v = dc.server_view(sc)
            led = "none" if self.bays[i].caddy == "removed" else v["led"]
            col = LED_BGR.get(led, (255, 255, 255))
            for b in sc.bays:
                cv2.circle(img, (int(b.uv[0]), int(b.uv[1])), 7, LED_BGR.get(b.led, col), 1, cv2.LINE_AA)
            u, vv, z = project(self.server_targets[i], cap.K, cap.T_base_cam)
            if u is None:
                continue
            x, y = 8, int(vv) + 6
            label = f"bay {i}: {led.upper()}" + ("  BLOCKED" if v["obstructed"] else "")
            if self.arm_at == str(i):
                label += "  <ARM"
            (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
            cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 5), (0, 0, 0), -1)
            cv2.putText(img, label, (x, y), font, 0.55, col, 1, cv2.LINE_AA)
        for o in scans[0].obstacles:
            x, y, w, h = o.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 40, 40), 2)
            cv2.putText(img, f"OBSTRUCTION {o.clearance_m * 100:.0f} cm out", (x, max(44, y - 6)),
                        font, 0.55, (255, 60, 60), 2, cv2.LINE_AA)
        return img

    def perceive(self):
        """On-robot perception of the camera render: LED colour per bay from the image, obstruction
        from depth. The only non-camera input: a caddy the robot itself pulled reads 'none'."""
        from perception import BayView, read_thermal   # this folder's helpers (BayView type, thermal)
        self.camera_png()                              # also saves the annotated frame
        _, scans = self._look()
        self.last_scans = scans
        temps = read_thermal(self.thermal_png())
        views = []
        for i, sc in enumerate(scans):
            v = dc.server_view(sc)
            led = "none" if self.bays[i].caddy == "removed" else v["led"]
            views.append(BayView(i, led, temps[i], v["obstructed"]))
        return views

    # thermal_png, telemetry, rack_summary: inherited (logical server state)

    # ---------------------------------------------------------------- actions
    def pull_caddy(self) -> str:
        if self.stopped or self.arm_at == "home" or self.holding:
            return super().pull_caddy()       # same refusal / error strings
        i = int(self.arm_at)
        if self.bays[i].caddy in ("removed", "jammed"):
            return super().pull_caddy()       # empty bay / latched jam: no physics needed
        jammed = i in self.jam_bays
        dof = int(self.m.joint(self._joint(i)).dofadr[0])
        saved = float(self.m.dof_frictionloss[dof])
        if jammed:
            self.m.dof_frictionloss[dof] = 400.0    # seized rail: exceeds the 100 N service force
        ok = self._move_tray(i, OPEN)
        moved = self.tray_pos(i)
        self.m.dof_frictionloss[dof] = saved
        if not ok:
            self._move_tray(i, CLOSED)          # back off, like the force reflex
            self.bays[i].caddy = "jammed"
            self.events.append(f"force reflex: tray jam at bay {i} (moved {moved * 1000:.0f} mm)")
            return (f"JAMMED: tray at bay {i} travelled only {moved * 1000:.0f} mm of "
                    f"{OPEN * 1000:.0f} mm under the 100N service force; backed off.")
        self.jam_bays.discard(i)
        return super().pull_caddy() + f" (tray extended {moved * 1000:.0f} mm)"

    def insert_caddy(self) -> str:
        res = super().insert_caddy()
        if res.startswith("OK"):
            i = int(self.arm_at)
            if not self._move_tray(i, CLOSED):
                return res + f"; WARNING tray did not fully close ({self.tray_pos(i) * 1000:.0f} mm)"
        return res

    def wait(self) -> str:
        for _ in range(200):
            simrun.drive(self.m, self.d, self.targets)
        return super().wait()

    def close(self):
        self._cache = None
