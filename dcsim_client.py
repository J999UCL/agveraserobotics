"""Client for the data-center sim server. Needs only numpy + pillow (no MuJoCo).

    from dcsim_client import SimClient
    sim = SimClient("localhost", 5555)
    health = sim.get_server_health()          # what's broken, per DIMM / fan
    sim.pull_tray_out()
    img = sim.capture_image("wrist")          # numpy RGB
    sim.swap("A3")                            # pull failed DIMM, bin it, install a spare

Positions are in the robot base frame (+X toward the rack, +Y left, +Z up, meters).
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
    def get_server_health(self) -> dict: return self.call("get_server_health")
    def get_event_log(self) -> list[dict]: return self.call("get_event_log")
    def get_events(self, since: float = 0.0) -> list[dict]: return self.call("get_events", since=since)
    def get_installed(self) -> dict: return self.call("get_installed")
    def get_cart_inventory(self) -> dict: return self.call("get_cart_inventory")
    def list_slots(self) -> dict: return self.call("list_slots")

    # status ---------------------------------------------------------------
    def ping(self) -> str: return self.call("ping")
    def get_status(self) -> dict: return self.call("get_status")
    def get_pull_force(self) -> dict: return self.call("get_pull_force")
    def stop(self) -> dict: return self.call("stop")
    def step(self, n: int = 1) -> dict: return self.call("step", n=n)

    # skills -----------------------------------------------------------------
    def pull_tray_out(self) -> dict: return self.call("pull_tray_out")
    def push_tray_in(self) -> dict: return self.call("push_tray_in")
    def view_tray(self) -> dict: return self.call("view_tray")
    def pick(self, slot: str) -> dict: return self.call("pick", slot=slot)
    def place(self, slot: str) -> dict: return self.call("place", slot=slot)
    def discard(self) -> dict: return self.call("discard")
    def swap(self, slot: str) -> dict: return self.call("swap", slot=slot)
    def home(self) -> dict: return self.call("home")

    # low level --------------------------------------------------------------
    def move_tcp(self, x: float, y: float, z: float, orientation: str = "fingers_y", speed: float = 0.2) -> dict:
        return self.call("move_tcp", x=x, y=y, z=z, orientation=orientation, speed=speed)

    def open_gripper(self) -> dict: return self.call("open_gripper")
    def close_gripper(self) -> dict: return self.call("close_gripper")

    # cameras ----------------------------------------------------------------
    def capture_image(self, camera: str = "wrist") -> np.ndarray:
        """camera: "wrist" (640x480, on the gripper), "overview" or "tray_top" (1280x720)."""
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
    def set_jam(self, slot: str, on: bool = True) -> dict: return self.call("set_jam", slot=slot, on=on)
    def inject_fault(self, slot: str, fault: str) -> dict: return self.call("inject_fault", slot=slot, fault=fault)
    def reset(self, seed: int | None = None) -> dict: return self.call("reset", seed=seed)
