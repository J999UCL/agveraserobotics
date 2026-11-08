"""Plug the agent into the team's MuJoCo data-centre scene (sim/).

MujocoRack is a drop-in replacement for World: same methods, same return
strings, so agent.py / scenarios.py work unchanged.

Mapping (prototype):
  * our 8 "bays"      -> 8 servers in rack R04, slots 6..13 (U13..U27)
  * pull_caddy        -> extend that server's tray under the rail service force;
                         JAMMED is detected PHYSICALLY (tray fails to travel)
  * insert_caddy      -> close the tray
  * LEDs              -> the server's two front handles are recoloured per status
  * camera_png        -> MuJoCo render of rack R04's front
  * thermal_png       -> synthetic (MuJoCo has no thermal camera), from server state
  * SMART / RAID      -> logical server state inherited from World (not physics)

Not covered by the sim yet: the arm is parked (no motion control), and drives are
not individually removable, so the "swap" is a tray-level service action.
"""

from __future__ import annotations

import io
import os
import sys

import mujoco
import numpy as np
from PIL import Image

from world import LED_RGB, World

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import run as simrun  # noqa: E402  (team's scene helpers: load, drive, settled)

RACK = 4
FIRST_SLOT = 6
OPEN, CLOSED = 0.70, 0.0
SERVICE_STEPS = 1500      # 3 s of sim time per tray move


def _hue_class(rgb) -> str | None:
    """Status colour of a lit pixel by hue; None for grey/dark rack pixels."""
    import colorsys
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    if s < 0.45 or v < 0.25:
        return None
    h *= 360
    if 80 <= h <= 170:
        return "green"
    if 25 <= h < 70:
        return "amber"
    if h < 15 or h > 340:
        return "red"
    return None


class MujocoRack(World):
    def __init__(self, base: World, frames_dir: str | None = None):
        super().__init__(**{k: getattr(base, k) for k in base.__dataclass_fields__})
        self.m, self.d = simrun.load()
        self.targets: dict[str, float] = {}
        self.frames_dir = frames_dir
        self.frame_no = 0
        self.renderer = mujoco.Renderer(self.m, height=480, width=640)
        self.cam = mujoco.MjvCamera()
        mid = (self.d.body(self._ident(0)).xpos + self.d.body(self._ident(7)).xpos) / 2
        self.cam.lookat[:] = [mid[0], -0.5, mid[2]]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.2, 90.0, -5.0
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0

    # ---------------------------------------------------------------- mapping
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

    def _sync_leds(self):
        for i, b in enumerate(self.bays):
            rgba = [c / 255 for c in LED_RGB[b.led]] + [1.0]
            for side in (-1, 1):
                gid = self.m.geom(f"{self._ident(i)}_handle_{side}").id
                self.m.geom_matid[gid] = -1
                self.m.geom_rgba[gid] = rgba

    def _save_frame(self, png: bytes, tag: str):
        if self.frames_dir:
            os.makedirs(self.frames_dir, exist_ok=True)
            self.frame_no += 1
            with open(os.path.join(self.frames_dir, f"{self.frame_no:03d}_{tag}.png"), "wb") as f:
                f.write(png)

    # ------------------------------------------------------------------ views
    def camera_png(self) -> bytes:
        self._sync_leds()
        mujoco.mj_forward(self.m, self.d)
        self.renderer.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        img = Image.fromarray(self.renderer.render())
        self._overlay(img)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        self._save_frame(png, "camera")
        return png

    def _overlay(self, img: Image.Image):
        """Bay labels, arm position and obstruction markers, placed from known geometry
        (like an AR overlay). Kept clear of the handles that local perception samples."""
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 330, 16], fill=(0, 0, 0))
        d.text((4, 3), "rack R04 front camera  (labels = overlay)", fill=(230, 230, 230))
        for i in range(len(self.bays)):
            ul, v, _ = self._project(self.d.geom_xpos[self.m.geom(f"{self._ident(i)}_handle_-1").id])
            ur, _, _ = self._project(self.d.geom_xpos[self.m.geom(f"{self._ident(i)}_handle_1").id])
            y = int(v) - 6
            d.rectangle([int(ul) - 52, y - 1, int(ul) - 10, y + 11], fill=(0, 0, 0))
            d.text((int(ul) - 50, y), f"bay {i}", fill=(255, 255, 255))
            x = int(ur) + 10
            if self.arm_at == str(i):
                d.rectangle([x, y - 1, x + 40, y + 11], fill=(30, 80, 200))
                d.text((x + 3, y), "<ARM", fill=(255, 255, 255))
                x += 46
            if self.obstacle_at == i:
                d.rectangle([x, y - 1, x + 82, y + 11], fill=(250, 220, 0))
                d.text((x + 3, y), "OBSTRUCTION", fill=(0, 0, 0))

    def _project(self, p) -> tuple[float, float, float]:
        """World point -> (u, v) pixel and depth for self.cam (free camera)."""
        az, el = np.radians(self.cam.azimuth), np.radians(self.cam.elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        pos = np.array(self.cam.lookat) - self.cam.distance * fwd
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        v = np.asarray(p) - pos
        z = v @ fwd
        h, w = self.renderer.height, self.renderer.width
        f = (h / 2) / np.tan(np.radians(self.m.vis.global_.fovy) / 2)
        return w / 2 + f * (v @ right) / z, h / 2 - f * (v @ up) / z, z

    def perceive(self):
        """Local perception on the MuJoCo render: LED = colour at each projected handle."""
        from perception import BayView, read_thermal
        img = Image.open(io.BytesIO(self.camera_png())).convert("RGB")
        temps = read_thermal(self.thermal_png())
        views = []
        for i in range(len(self.bays)):
            votes: dict[str, int] = {}
            for side in (-1, 1):
                gid = self.m.geom(f"{self._ident(i)}_handle_{side}").id
                u, v, z = self._project(self.d.geom_xpos[gid])
                if z <= 0:
                    continue
                for du in range(-2, 3):
                    for dv in range(-8, 9):   # handle is ~17 px tall; next server is 23 px away
                        x, y = int(round(u)) + du, int(round(v)) + dv
                        if 0 <= x < img.width and 0 <= y < img.height:
                            c = _hue_class(img.getpixel((x, y)))
                            if c:
                                votes[c] = votes.get(c, 0) + 1
            led = max(votes, key=votes.get) if votes else "off"
            if self.bays[i].caddy == "removed":
                led = "none"
            views.append(BayView(i, led, temps[i], self.obstacle_at == i))
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
        self.renderer.close()
