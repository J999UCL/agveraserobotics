"""Client for the data-center sim server. Needs only numpy + pillow (no MuJoCo).

    from dcsim_client import SimClient
    sim = SimClient("localhost", 5555)
    sim.get_fleet_health()                    # which servers have problems, and where
    sim.go_to("R1-05-U20")                    # drive there, face the rack, lift to height
    sim.pull_tray_out()
    img = sim.capture_image("wrist")          # numpy RGB
    sim.swap("A3")                            # pull the failed DIMM, bin it, install an onboard spare
    sim.push_tray_in(); sim.go_home(); sim.restock()

Slots are short names at the current server (A3, F2); onboard spares are CD1/CF1.
Skills block until done. A second SimClient can still call stop(), get_status(),
get_pull_force(), capture_image() while one runs.
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import socket
from typing import Any

import numpy as np


class SimError(RuntimeError):
    pass


def _img(b64: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


class SimClient:
    def __init__(self, host: str = "localhost", port: int = 5555, timeout: float = 300.0) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buf = b""
        self._ids = itertools.count()

    def call(self, cmd: str, **args: Any) -> Any:
        rid = str(next(self._ids))
        self.sock.sendall((json.dumps({"id": rid, "cmd": cmd, "args": args}) + "\n").encode())
        while True:
            while b"\n" not in self._buf:
                chunk = self.sock.recv(1 << 20)
                if not chunk:
                    raise SimError("connection closed by simulator")
                self._buf += chunk
            line, self._buf = self._buf.split(b"\n", 1)
            msg = json.loads(line)
            if msg.get("id") == rid:
                if msg.get("error"):
                    raise SimError(msg["error"])
                return msg.get("result")

    def close(self) -> None:
        self.sock.close()

    # diagnostics ----------------------------------------------------------
    def get_fleet_health(self) -> list[dict]: return self.call("get_fleet_health")
    def get_server_health(self, server: str | None = None) -> dict: return self.call("get_server_health", server=server)
    def get_event_log(self, server: str) -> list[dict]: return self.call("get_event_log", server=server)
    def get_work_orders(self, status: str | None = None) -> list[dict]: return self.call("get_work_orders", status=status)
    def get_events(self, since: float = 0.0) -> list[dict]: return self.call("get_events", since=since)
    def get_installed(self, server: str | None = None) -> dict: return self.call("get_installed", server=server)
    def get_onboard_inventory(self) -> dict: return self.call("get_onboard_inventory")
    def get_depot_inventory(self) -> dict: return self.call("get_depot_inventory")
    def list_servers(self) -> list[dict]: return self.call("list_servers")

    # status ---------------------------------------------------------------
    def ping(self) -> str: return self.call("ping")
    def get_status(self) -> dict: return self.call("get_status")
    def get_pull_force(self) -> dict: return self.call("get_pull_force")
    def stop(self) -> dict: return self.call("stop")
    def step(self, n: int = 1) -> dict: return self.call("step", n=n)

    # navigation -------------------------------------------------------------
    def go_to(self, server: str) -> dict: return self.call("go_to", server=server)
    def go_home(self) -> dict: return self.call("go_home")
    def restock(self) -> dict: return self.call("restock")

    # skills (at the current server) ------------------------------------------
    def pull_tray_out(self) -> dict: return self.call("pull_tray_out")
    def push_tray_in(self) -> dict: return self.call("push_tray_in")
    def view_tray(self) -> dict: return self.call("view_tray")
    def pick(self, slot: str) -> dict: return self.call("pick", slot=slot)
    def place(self, slot: str) -> dict: return self.call("place", slot=slot)
    def discard(self) -> dict: return self.call("discard")
    def swap(self, slot: str) -> dict: return self.call("swap", slot=slot)
    def home(self) -> dict: return self.call("home")

    # low level --------------------------------------------------------------
    def move_tcp(self, x: float, y: float, z: float, orientation: str = "down", speed: float = 0.2) -> dict:
        """Gripper tip to (x, y, z) in the ROBOT frame (relative to the arm base, +X forward, +Z up)."""
        return self.call("move_tcp", x=x, y=y, z=z, orientation=orientation, speed=speed)

    def open_gripper(self) -> dict: return self.call("open_gripper")
    def close_gripper(self) -> dict: return self.call("close_gripper")

    # cameras ----------------------------------------------------------------
    def capture_image(self, camera: str = "wrist") -> np.ndarray:
        """camera: "wrist" (640x480, on the gripper), "follow" (behind the robot), "overview", "aisle_end"."""
        return _img(self.call("capture_image", camera=camera)["jpeg_b64"])

    def capture_rgbd(self, camera: str = "wrist") -> dict:
        r = self.call("capture_image", camera=camera, depth=True)
        h, w = r["height"], r["width"]
        depth = np.frombuffer(base64.b64decode(r["depth_f16_b64"]), dtype=np.float16).reshape(h, w).astype(np.float32)
        return {"rgb": _img(r["jpeg_b64"]), "depth": depth, "K": np.array(r["K"]),
                "cam_pos": np.array(r["cam_pos"]), "cam_rot": np.array(r["cam_rot"])}

    def pixel_to_world(self, u: float, v: float, depth: float, camera: str = "wrist") -> list[float]:
        return self.call("pixel_to_world", u=u, v=v, depth=depth, camera=camera)

    # demo controls ----------------------------------------------------------
    def set_jam(self, slot: str, server: str | None = None, on: bool = True) -> dict:
        return self.call("set_jam", slot=slot, server=server, on=on)

    def inject_fault(self, server: str, slot: str, fault: str) -> dict:
        return self.call("inject_fault", server=server, slot=slot, fault=fault)

    def spawn_technician(self, mode: str = "cross", x: float | None = None, delay: float = 0.0, duration: float = 4.0) -> dict:
        """Demo: a technician walks into the aisle ("cross" once, or "block" = stand there for duration s)."""
        return self.call("spawn_technician", mode=mode, x=x, delay=delay, duration=duration)

    def get_lidar_scan(self) -> dict: return self.call("get_lidar_scan")
    def get_nav_log(self) -> list[dict]: return self.call("get_nav_log")

    def set_led(self, server: str, slot: str, state: str | None = None) -> dict:
        """Demo: force an LED (green/amber/red/off/blue) regardless of health; None = automatic."""
        return self.call("set_led", server=server, slot=slot, state=state)

    def get_slot_leds(self, server: str) -> dict:
        """World xyz of each slot's status LED on a server (A1-A8, F1-F5)."""
        return self.call("get_slot_leds", server=server)

    def clear_technician(self) -> dict: return self.call("clear_technician")

    def get_slot_grasp_points(self, server: str) -> dict:
        """World xyz the gripper closes on for each slot (A1-A8, F1-F5): map knowledge for perception."""
        return self.call("get_slot_grasp_points", server=server)

    def drop_cable(self, server: str, slot: str, height: float = 0.07) -> dict:
        """Demo: a loose cable lands across the open tray along `slot`, `height` m above the part.
        Visual only: the wrist camera has to catch it."""
        return self.call("drop_cable", server=server, slot=slot, height=height)

    def clear_cable(self) -> dict: return self.call("clear_cable")

    def reset(self, seed: int | None = None, faults: list | None = None) -> dict:
        """faults: [[server, slot, fault], ...]; [] = a healthy data centre."""
        return self.call("reset", seed=seed, faults=faults)
