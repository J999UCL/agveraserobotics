"""Client for the hot-swap simulation server. No Isaac Sim needed.

    pip install numpy pillow
    from hotswap_client import SimClient

    sim = SimClient("localhost", 5555)        # or the host/port Modal prints
    sim.go_to_cabinet(1)
    sim.view_rack()
    f = sim.get_wrist_frames()                # f.rgb (H,W,3) uint8, f.depth (H,W) meters, f.K
    xyz = sim.get_robot_coords_at(u, v)       # pixel -> robot frame
    sim.move_hand_to([0.45, -0.06, 0.64])
    sim.close_gripper()
    sim.get_pull_force()

Every position is in the ROBOT BASE frame: +X toward the rack, +Y left, +Z up (meters).
Blocking calls (moves) return when done. While one runs, a SECOND SimClient can still
call quick commands like stop(), get_pull_force() or capture_frame().
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import socket
from dataclasses import dataclass
from typing import Any

import numpy as np


class SimError(RuntimeError):
    pass


@dataclass
class Frames:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32 meters (0 = nothing hit)
    K: np.ndarray  # 3x3 intrinsics, OpenCV convention
    cam_pos: np.ndarray  # camera position, robot frame
    cam_rot: np.ndarray  # USD camera axes in robot frame (camera looks down its -Z)

    def pixel_to_robot(self, u: float, v: float) -> np.ndarray | None:
        d = float(self.depth[int(v), int(u)])
        if d <= 0:
            return None
        x = (u - self.K[0, 2]) / self.K[0, 0] * d
        y = (v - self.K[1, 2]) / self.K[1, 1] * d
        return self.cam_rot @ np.array([x, -y, -d]) + self.cam_pos


def _decode_jpeg(b64: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


class SimClient:
    def __init__(self, host: str = "localhost", port: int = 5555, timeout: float = 120.0) -> None:
        self.addr = (host, port)
        self.sock = socket.create_connection(self.addr, timeout=timeout)
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

    # --- status ----------------------------------------------------------
    def ping(self) -> str: return self.call("ping")
    def get_status(self) -> dict: return self.call("get_status")
    def hand_pose(self) -> dict: return self.call("hand_pose")
    def step(self, n: int = 1) -> dict: return self.call("step", n=n)

    # --- cart ------------------------------------------------------------
    def go_to_cabinet(self, cabinet: int) -> dict: return self.call("go_to_cabinet", cabinet=cabinet)
    def current_cabinet(self) -> int | None: return self.call("current_cabinet")

    # --- arm -------------------------------------------------------------
    def move_hand_to(self, position, orientation=None, max_speed: float = 0.25) -> dict:
        return self.call("move_hand_to", position=list(position), orientation=orientation, max_speed=max_speed)

    def move_relative(self, dx=0.0, dy=0.0, dz=0.0, max_speed: float = 0.25) -> dict:
        return self.call("move_relative", dx=dx, dy=dy, dz=dz, max_speed=max_speed)

    def ik_step(self, position, orientation=None, steps: int = 1) -> dict:
        """Small step toward a pose, then advance `steps` frames. For your own reactive loop."""
        return self.call("ik_step", position=list(position), orientation=orientation, steps=steps)

    def move_joints(self, joints, duration: float = 1.5) -> dict: return self.call("move_joints", joints=list(joints), duration=duration)
    def view_rack(self, standoff: float = 0.30) -> dict: return self.call("view_rack", standoff=standoff)
    def stow_arm(self) -> dict: return self.call("stow_arm")
    def stop(self) -> dict: return self.call("stop")

    # --- gripper ---------------------------------------------------------
    def open_gripper(self) -> dict: return self.call("open_gripper")
    def close_gripper(self) -> dict: return self.call("close_gripper")
    def gripper_move(self, width: float) -> dict: return self.call("gripper_move", width=width)
    def gripper_width(self) -> float: return self.call("gripper_width")
    def is_grasping(self) -> bool: return self.call("is_grasping")

    # --- camera ----------------------------------------------------------
    def capture_frame(self) -> np.ndarray:
        return _decode_jpeg(self.call("capture_frame")["jpeg_b64"])

    def get_wrist_frames(self) -> Frames:
        r = self.call("get_wrist_frames")
        depth = np.frombuffer(base64.b64decode(r["depth_f16_b64"]), dtype=np.float16).reshape(r["shape"]).astype(np.float32)
        return Frames(_decode_jpeg(r["rgb_jpeg_b64"]), depth, np.array(r["K"]), np.array(r["cam_pos"]), np.array(r["cam_rot"]))

    def get_depth_at(self, u: int, v: int) -> float | None: return self.call("get_depth_at", u=u, v=v)["depth_m"]

    def get_robot_coords_at(self, u: int, v: int) -> list[float] | None:
        return self.call("get_robot_coords_at", u=u, v=v)["position"]

    def get_camera_info(self) -> dict: return self.call("get_camera_info")

    def get_overview(self) -> np.ndarray | None:
        r = self.call("get_overview")
        return None if r is None else _decode_jpeg(r["jpeg_b64"])

    # --- caddies ---------------------------------------------------------
    def get_pull_force(self) -> float: return self.call("get_pull_force")
    def caddy_extraction(self, cabinet: int, bay: int) -> float: return self.call("caddy_extraction", cabinet=cabinet, bay=bay)
    def caddy_status(self, cabinet: int, bay: int) -> dict: return self.call("caddy_status", cabinet=cabinet, bay=bay)

    # --- demo triggers ---------------------------------------------------
    def set_fault(self, cabinet: int, bay: int) -> dict: return self.call("set_fault", cabinet=cabinet, bay=bay)
    def set_led(self, cabinet: int, bay: int, state: str) -> dict: return self.call("set_led", cabinet=cabinet, bay=bay, state=state)
    def get_telemetry(self) -> list[dict]: return self.call("get_telemetry")
    def set_jam(self, cabinet: int, bay: int, on: bool = True) -> dict: return self.call("set_jam", cabinet=cabinet, bay=bay, on=on)
    def connector_broken(self) -> bool: return self.call("connector_broken")
    def drop_obstacle(self, position=None, delay: float = 0.0) -> dict: return self.call("drop_obstacle", position=position, delay=delay)
    def clear_obstacle(self) -> dict: return self.call("clear_obstacle")
    def reset(self, seed: int | None = None) -> dict: return self.call("reset", seed=seed)

    # --- debug (ground truth, tests only) ---------------------------------
    def debug_bay_pose(self, cabinet: int, bay: int) -> dict: return self.call("debug_bay_pose", cabinet=cabinet, bay=bay)
