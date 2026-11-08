"""Plug the team's Argon agent (robot/sandbox in J999UCL/agveraserobotics) into the dcsim robot.

DcsimRack is a drop-in replacement for World, like mujoco_adapter.MujocoRack: same methods, same
return strings, so agent.py, policy.py and scenarios.py run unchanged. Copy this file next to
them (robot/sandbox/) and start the simulator (python scripts/run_server.py, or on Modal).

Mapping
  * the agent's 8 "bays"   -> RAM slots A1-A8 of one server (default R1-05-U20)
  * move_to(bay)           -> the robot DRIVES to that server if it isn't there (lidar-guarded),
                              slides the tray out and points the wrist camera at the slot
  * pull_caddy             -> pick the RAM stick. A jam is PHYSICAL: the pull force climbs until
                              the force reflex aborts at 25 N (below the 40 N socket limit)
  * stow_old               -> drop it in the robot's returns bin
  * pick_spare             -> take a spare from the robot's onboard cassette. If the cassette is
                              empty but spares are left, the robot drives to the depot, restocks
                              and comes back
  * insert_caddy           -> install the spare in the empty slot
  * obstacle               -> a technician blocking the aisle; the lidar stops the robot and the
                              move is BLOCKED. wait() lets the person walk away
  * LEDs                   -> the slot LEDs in the scene show what the World says (including wrong
                              LEDs), and perceive() reads them back from the wrist-camera RGB-D
  * telemetry / RAID / thermal -> logical state inherited from World (as in MujocoRack)

The agent's own words (drives, caddies, SMART, RAID) are unchanged; read "caddy" as "RAM stick".
"""

from __future__ import annotations

import colorsys
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from world import World

# dcsim_client lives at the root of the dcsim repo (AgentVerseBob2). Set DCSIM_REPO to it, or keep
# this file inside that repo (integration/argon/).
_repo = os.environ.get("DCSIM_REPO") or str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)
from dcsim_client import SimClient  # noqa: E402

SERVER = "R1-05-U20"
SPARE_SLOTS = ("CD1", "CD2", "CD3", "CD4")
WAIT_TICKS = 150                  # one agent "wait" = 3 s of sim time (the 60 s is logical)
LED_DRAW = {"green": (60, 220, 90), "amber": (255, 170, 20), "red": (255, 60, 60), "off": (150, 150, 150),
            "blue": (90, 140, 255), "none": (150, 150, 150), "unknown": (255, 255, 255)}


def _addr() -> tuple[str, int]:
    host, _, port = os.environ.get("DCSIM", "localhost:5555").rpartition(":")
    return host or "localhost", int(port)


def classify_led(rgb) -> str:
    """Status colour of a lit LED pixel by hue (so lighting changes don't matter)."""
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    if v < 0.35 or s < 0.35:
        return "off"
    h *= 360
    if 75 <= h <= 170:
        return "green"
    if 20 <= h < 60:
        return "amber"
    if h < 15 or h > 340:
        return "red"
    if 190 <= h <= 260:
        return "blue"
    return "unknown"


class DcsimRack(World):
    def __init__(self, base: World, server: str = SERVER, client: SimClient | None = None,
                 frames_dir: str | None = None, verbose: bool = False):
        super().__init__(**{k: getattr(base, k) for k in base.__dataclass_fields__})
        self.server = server
        self.c = client or SimClient(*_addr())
        self.frames_dir = frames_dir
        self.frame_no = 0
        self.verbose = verbose
        self.blocked_bay: int | None = None      # bay whose approach the lidar last stopped
        self.technician = False
        self._led_sent: dict[str, str | None] = {}
        self._scan = None                       # last perception result (when the tray is out of view)
        self.c.reset(faults=[])                 # healthy data centre; this World holds the incident
        for i in self.jam_bays:
            self.c.set_jam(self._slot(i), server, True)
        self._sync_leds()
        self._at_server()                       # robot starts on patrol at the server, tray out
        self._view()
        self._scan = self._read_leds()
        if self.obstacle_at is not None:        # called away after the patrol look; aisle then blocks
            self._log("obstacle scenario: robot returns to the dock, a technician then blocks the aisle")
            self.c.push_tray_in()
            self.c.go_home()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _slot(bay: int) -> str:
        return f"A{bay + 1}"

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"       [dcsim] {msg}", flush=True)

    def _status(self) -> dict:
        return self.c.get_status()

    def _here(self) -> bool:
        return self._status()["at_server"] == self.server

    def _tray_out(self) -> bool:
        return self._status()["trays"][self.server] > 0.2

    def _sync_leds(self) -> None:
        for i, b in enumerate(self.bays):
            want = None if b.caddy == "removed" else b.led
            if self._led_sent.get(self._slot(i), "x") != want:
                self.c.set_led(self.server, self._slot(i), want)
                self._led_sent[self._slot(i)] = want

    def _at_server(self) -> str | None:
        """Drive to the server (if needed) and open its tray. Returns an error string or None."""
        if not self._here():
            r = self.c.go_to(self.server)
            if not r["ok"]:
                return r.get("error", "go_to failed")
        if not self._tray_out():
            r = self.c.pull_tray_out()
            if not r["ok"]:
                return r.get("error", "pull_tray_out failed")
        return None

    def _view(self) -> bool:
        if not (self._here() and self._tray_out()) or self._status()["holding"]:
            return False
        return bool(self.c.view_tray()["ok"])

    def _read_leds(self, cap: dict | None = None):
        """LED colour per bay from the wrist RGB-D frame: project each slot LED (map knowledge)
        into the image, check the depth agrees (not hidden), classify the pixel colour."""
        cap = cap or self.c.capture_rgbd("wrist")
        rgb, depth, K = cap["rgb"], cap["depth"], cap["K"]
        pos, R = cap["cam_pos"], cap["cam_rot"]
        leds = self.c.get_slot_leds(self.server)
        h, w = depth.shape
        out = {}
        for i in range(len(self.bays)):
            p = R.T @ (np.asarray(leds[self._slot(i)]) - pos)  # MuJoCo camera: looks down -Z, +Y up
            if p[2] >= -0.05:
                out[i] = ("unknown", None)
                continue
            u = K[0, 2] + K[0, 0] * p[0] / -p[2]
            v = K[1, 2] - K[1, 1] * p[1] / -p[2]
            ui, vi = int(round(u)), int(round(v))
            r = 6                                      # the LED is ~10 px across at the view height
            if not (r <= ui < w - r and r <= vi < h - r):
                out[i] = ("unknown", None)
                continue
            # keep only pixels at the LED's own depth: parts of the LED can sit behind a RAM stick
            win_d = depth[vi - r:vi + r + 1, ui - r:ui + r + 1]
            win_c = rgb[vi - r:vi + r + 1, ui - r:ui + r + 1].astype(int)
            on_led = np.abs(win_d - (-p[2])) < 0.012
            if on_led.sum() < 4:
                out[i] = ("unknown", (u, v))           # hidden: something is in front of it
                continue
            px = win_c[on_led]
            px = px[np.argmax(px.max(axis=1) + (px.max(axis=1) - px.min(axis=1)))]  # brightest, most saturated
            out[i] = (classify_led(px), (u, v))
        return {"leds": out, "rgb": rgb}

    # ------------------------------------------------------------------ views
    def camera_png(self) -> bytes:
        self._sync_leds()
        if self._view():
            scan = self._scan = self._read_leds()
            img = Image.fromarray(scan["rgb"])
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
            d.text((6, 5), f"{self.server} tray, wrist camera | labels = on-robot LED reading", fill=(235, 235, 235))
            for i, (led, uv) in scan["leds"].items():
                if self.bays[i].caddy == "removed":
                    led = "none"
                if uv is None:
                    continue
                u, v = uv
                col = LED_DRAW.get(led, (255, 255, 255))
                d.ellipse([u - 8, v - 8, u + 8, v + 8], outline=col, width=2)
                d.text((u - 4, v - 24), str(i), fill=col)
            key = "  ".join(f"bay{i}={'none' if self.bays[i].caddy == 'removed' else led}"
                            + ("<ARM" if self.arm_at == str(i) else "") for i, (led, _) in scan["leds"].items())
            d.rectangle([0, img.height - 20, img.width, img.height], fill=(0, 0, 0))
            d.text((6, img.height - 16), key, fill=(235, 235, 235))
        else:
            img = Image.fromarray(self.c.capture_image("follow"))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, img.width, 26], fill=(0, 0, 0))
            d.text((8, 7), f"follow camera: robot is not at {self.server} with the tray open", fill=(235, 235, 235))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        if self.frames_dir:
            os.makedirs(self.frames_dir, exist_ok=True)
            self.frame_no += 1
            with open(os.path.join(self.frames_dir, f"{self.frame_no:03d}_camera.png"), "wb") as f:
                f.write(png)
        return png

    def perceive(self):
        """On-robot perception: LED per bay from the wrist camera (the last look if the tray is not
        in view right now), thermal from the thermal image, obstructed = the lidar stopped the
        robot on its way to that bay. The only non-camera input: a stick the robot itself removed."""
        from perception import BayView, read_thermal
        self._sync_leds()
        if self._view():
            self._scan = self._read_leds()
        temps = read_thermal(self.thermal_png())
        views = []
        for i, b in enumerate(self.bays):
            led = "none" if b.caddy == "removed" else self._scan["leds"][i][0]
            views.append(BayView(i, led, temps[i], self.blocked_bay == i))
        return views

    # ---------------------------------------------------------------- actions
    def move_to(self, bay: int) -> str:
        if not 0 <= bay < len(self.bays) or self.stopped:
            return super().move_to(bay)
        if self.obstacle_at is not None and not self._here() and not self.technician:
            st = self._status()["base"]
            rack_x = (int(self.server.split("-")[1]) - 1) * 0.61
            self.c.spawn_technician("block", x=round((st["x"] + rack_x) / 2, 2), delay=0.0, duration=3600.0)
            self.technician = True
        if self.c.get_status()["holding"]:
            return "ERROR: put the held part down first"
        err = self._at_server()
        if err:
            if "blocked" in err:
                self.blocked_bay = bay
                self.events.append(f"proximity reflex: path to bay {bay} blocked")
                return (f"BLOCKED: lidar stopped the robot in the aisle on the way to bay {bay} "
                        f"({self.server}); it waited 20 s and the obstruction did not clear. Robot halted in place.")
            return f"ERROR: could not reach {self.server}: {err}"
        self.blocked_bay = None
        self.arm_at = str(bay)
        self._view()
        return f"OK: arm at bay {bay} (RAM slot {self._slot(bay)} of {self.server}, tray open)"

    def wait(self) -> str:
        res = super().wait()                     # obstruction / RAID counters
        if self.technician and self.obstacle_at is None:
            self.c.clear_technician()
            self.technician = False
        self.c.step(WAIT_TICKS)
        return res

    def pull_caddy(self) -> str:
        if self.stopped or self.arm_at == "home" or self.holding:
            return super().pull_caddy()
        i = int(self.arm_at)
        if self.bays[i].caddy in ("removed", "jammed"):
            return super().pull_caddy()
        r = self.c.pick(self._slot(i))
        if not r["ok"]:
            if "stuck" in r.get("error", ""):
                self.bays[i].caddy = "jammed"
                self.jam_bays.discard(i)
                self.events.append(f"force reflex: jam at bay {i}")
                return (f"JAMMED: pull force reached {r.get('peak_force_n')} N at bay {i}; the force reflex "
                        "aborted at 25 N (socket damage at 40 N) and the arm backed off. Stick still seated.")
            return f"ERROR: pull failed: {r.get('error')}"
        self.jam_bays.discard(i)
        res = super().pull_caddy()
        return res + f" (RAM stick lifted out, peak force {r.get('peak_force_n', 0):.0f} N)"

    def stow_old(self) -> str:
        if self.stopped or self.holding != "old":
            return super().stow_old()
        r = self.c.discard()
        if not r["ok"]:
            return f"ERROR: could not drop it in the returns bin: {r.get('error')}"
        return super().stow_old()

    def pick_spare(self) -> str:
        if self.stopped or self.holding or self.spares <= 0:
            return super().pick_spare()
        note = ""
        onboard = self.c.get_onboard_inventory()
        src = next((s for s in SPARE_SLOTS if onboard.get(s)), None)
        if src is None:                          # cassette empty: fetch more from the depot
            note = " (cassette was empty: the robot drove to the depot, restocked and came back)"
            for step in (self.c.push_tray_in, self.c.go_home, self.c.restock):
                r = step()
                if not r["ok"]:
                    return f"ERROR: restock trip failed at {step.__name__}: {r.get('error')}"
            err = self._at_server()
            if err:
                return f"ERROR: restocked but could not get back to {self.server}: {err}"
            onboard = self.c.get_onboard_inventory()
            src = next((s for s in SPARE_SLOTS if onboard.get(s)), None)
            if src is None:
                return "ERROR: no spare RAM on the robot or at the depot"
        r = self.c.pick(src)
        if not r["ok"]:
            return f"ERROR: could not take the spare from {src}: {r.get('error')}"
        self._spare_from = src
        return super().pick_spare() + note

    def return_spare(self) -> str:
        if self.holding not in ("new", "new_doa"):
            return super().return_spare()
        onboard = self.c.get_onboard_inventory()
        dest = getattr(self, "_spare_from", None)
        if dest is None or onboard.get(dest):
            dest = next((s for s in SPARE_SLOTS if not onboard.get(s)), None)
        r = self.c.place(dest) if dest else {"ok": False, "error": "no free cassette slot"}
        if not r["ok"]:
            return f"ERROR: could not put the spare back: {r.get('error')}"
        return super().return_spare()

    def insert_caddy(self) -> str:
        if self.stopped or self.holding not in ("new", "new_doa") or self.arm_at == "home":
            return super().insert_caddy()
        i = int(self.arm_at)
        if self.bays[i].caddy != "removed":
            return super().insert_caddy()
        r = self.c.place(self._slot(i))
        if not r["ok"]:
            return f"ERROR: insert failed: {r.get('error')}"
        res = super().insert_caddy()
        self._sync_leds()
        return res

    def safe_stop(self, reason: str) -> str:
        try:
            self.c.call("stop")
        except Exception:  # noqa: BLE001 - nothing is moving between agent steps
            pass
        return super().safe_stop(reason)

    def close(self) -> None:
        """End of the incident: tray back in, robot back to the dock (if nothing is held)."""
        try:
            if not self._status()["holding"]:
                if self._here() and self._tray_out():
                    self.c.push_tray_in()
                self.c.go_home()
        finally:
            self.c.close()
