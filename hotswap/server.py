"""TCP command server that lives inside the simulation loop.

Protocol: newline-delimited JSON over TCP.
    request : {"id": "abc", "cmd": "move_hand_to", "args": {"position": [0.4, 0, 0.64]}}
    response: {"id": "abc", "result": ..., "error": null}

The server is polled from a scene step hook, so it keeps answering even while a
blocking motion is running. While busy it only accepts "quick" commands (status,
sensors, stop, triggers); anything else gets {"error": "busy"}. That is what lets a
controller send `stop` or watch `get_pull_force` mid-pull.

Plain TCP on purpose: the same server works on localhost, across the LAN, or through
a Modal tunnel.
"""

from __future__ import annotations

import base64
import io
import json
import selectors
import socket
import traceback
from typing import Any, Callable

import numpy as np
from PIL import Image

from .scene import HotSwapScene, Stopped, base_to_pixel, pixel_to_base

QUICK = {
    "ping", "get_status", "hand_pose", "joint_positions", "gripper_width", "is_grasping",
    "get_pull_force", "caddy_status", "caddy_extraction", "get_telemetry", "connector_broken",
    "current_cabinet", "stop", "set_jam", "drop_obstacle", "clear_obstacle", "set_led", "set_fault",
    "capture_frame", "get_wrist_frames", "get_depth_at", "get_robot_coords_at", "get_camera_info",
    "get_overview", "sim_time",
}


def _jpeg(rgb: np.ndarray, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _np(a) -> Any:
    if isinstance(a, np.ndarray):
        return a.tolist()
    if isinstance(a, (np.floating, np.integer, np.bool_)):
        return a.item()
    return a


class SimServer:
    def __init__(self, scene: HotSwapScene, host: str = "0.0.0.0", port: int = 5555) -> None:
        self.scene = scene
        self.sel = selectors.DefaultSelector()
        self.lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lsock.bind((host, port))
        self.lsock.listen()
        self.lsock.setblocking(False)
        self.sel.register(self.lsock, selectors.EVENT_READ, data=None)
        self.busy = False
        self._polling = False
        self.handlers: dict[str, Callable[..., Any]] = self._build_handlers()
        scene.step_hooks.append(self.poll)
        print(f"[sim-server] listening on {host}:{port}", flush=True)

    # ------------------------------------------------------------ networking
    def poll(self) -> None:
        if self._polling:  # a handler stepped the sim; don't recurse into the same poll
            return
        self._polling = True
        try:
            for key, _ in self.sel.select(timeout=0):
                if key.data is not None and key.fileobj.fileno() == -1:
                    continue  # closed by a nested poll while this event list was pending
                if key.data is None:
                    conn, addr = self.lsock.accept()
                    conn.setblocking(False)
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.sel.register(conn, selectors.EVENT_READ, data={"buf": b"", "addr": addr})
                    print(f"[sim-server] client connected {addr}", flush=True)
                else:
                    self._read(key.fileobj, key.data)
        finally:
            self._polling = False

    def _read(self, conn: socket.socket, data: dict) -> None:
        if conn.fileno() == -1:
            return
        try:
            chunk = conn.recv(1 << 16)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            chunk = b""
        if not chunk:
            self._drop(conn)
            return
        data["buf"] += chunk
        while b"\n" in data["buf"]:
            line, data["buf"] = data["buf"].split(b"\n", 1)
            if line.strip():
                self._handle(conn, line)

    def _drop(self, conn: socket.socket) -> None:
        try:
            self.sel.unregister(conn)
        except (KeyError, ValueError):
            pass
        conn.close()

    def _send(self, conn: socket.socket, msg: dict) -> None:
        if conn.fileno() == -1:
            return  # client went away while its command ran
        payload = (json.dumps(msg, default=_np) + "\n").encode()
        try:
            conn.setblocking(True)
            conn.sendall(payload)
            conn.setblocking(False)
        except OSError:
            self._drop(conn)

    def _handle(self, conn: socket.socket, line: bytes) -> None:
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            self._send(conn, {"id": None, "result": None, "error": f"bad json: {e}"})
            return
        rid, cmd, args = req.get("id"), req.get("cmd"), req.get("args") or {}
        fn = self.handlers.get(cmd)
        if fn is None:
            self._send(conn, {"id": rid, "result": None, "error": f"unknown command '{cmd}'"})
            return
        if self.busy and cmd not in QUICK:
            self._send(conn, {"id": rid, "result": None, "error": "busy: a motion is running (send 'stop' to abort)"})
            return
        quick = cmd in QUICK
        if not quick:
            self.busy = True
            self.scene._stop = False  # a stale stop must not cancel this new motion
        # Release the poll guard so `stop` etc. are still received while this command runs.
        was_polling, self._polling = self._polling, False
        try:
            result, error = fn(**args), None
        except Stopped:
            result, error = {"stopped": True}, None
        except Exception as e:  # report, keep serving
            result, error = None, f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            self._polling = was_polling
            if not quick:
                self.busy = False
        self._send(conn, {"id": rid, "result": result, "error": error})

    # ------------------------------------------------------------- commands
    def _build_handlers(self) -> dict[str, Callable[..., Any]]:
        s = self.scene

        def frames_or_raise():
            f = s.wait_for_frames()
            if f is None:
                raise RuntimeError("no camera frame yet")
            return f

        def get_status():
            pos, quat = s.hand_pose()
            return {"cabinet": s.current_cabinet(), "hand_position": pos.round(4), "hand_quat_wxyz": quat.round(4),
                    "fingertip_position": s.fingertip_position().round(4), "joints": s.joint_positions().round(4),
                    "gripper_width": round(s.gripper_width(), 4), "is_grasping": s.is_grasping(),
                    "pull_force_n": round(s.get_pull_force(), 2), "connector_broken": s.connector_broken(),
                    "sim_time": round(s.sim_time(), 3)}

        def capture_frame(quality: int = 85):
            f = frames_or_raise()
            return {"jpeg_b64": _jpeg(f.rgb, quality), "width": f.rgb.shape[1], "height": f.rgb.shape[0]}

        def get_wrist_frames(quality: int = 90):
            f = frames_or_raise()
            d = np.where(np.isfinite(f.depth), f.depth, 0).astype(np.float16)
            return {"rgb_jpeg_b64": _jpeg(f.rgb, quality), "depth_f16_b64": base64.b64encode(d.tobytes()).decode(),
                    "shape": list(f.depth.shape), "K": f.K, "cam_pos": f.cam_pos, "cam_rot": f.cam_rot}

        def get_depth_at(u: int, v: int):
            f = frames_or_raise()
            d = float(f.depth[int(v), int(u)])
            return {"depth_m": d if np.isfinite(d) else None}

        def get_robot_coords_at(u: int, v: int):
            f = frames_or_raise()
            d = float(f.depth[int(v), int(u)])
            if not np.isfinite(d):
                return {"position": None, "error": "no depth at that pixel"}
            return {"position": pixel_to_base(u, v, d, f).round(4), "depth_m": round(d, 4)}

        def project(position):
            f = frames_or_raise()
            u, v, d = base_to_pixel(position, f)
            return {"u": round(u, 1), "v": round(v, 1), "depth_m": round(d, 4)}

        def get_camera_info():
            h, w = s.cfg.camera.resolution
            return {"width": w, "height": h, "K": s.intrinsics(), "depth": "distance to image plane, meters",
                    "frame": "robot base: +X to rack, +Y left, +Z up"}

        def get_overview(quality: int = 80):
            o = s.get_overview_rgb()
            return None if o is None else {"jpeg_b64": _jpeg(o, quality)}

        def move_hand_to(position, orientation=None, max_speed: float = 0.25):
            kw = {"max_speed": max_speed}
            if orientation is not None:
                kw["orientation"] = np.asarray(orientation)
            return {"reached": s.move_hand_to(position, **kw), "hand_position": s.hand_pose()[0].round(4)}

        def move_relative(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, max_speed: float = 0.25):
            p = s.hand_pose()[0] + np.array([dx, dy, dz])
            return move_hand_to(p.tolist(), max_speed=max_speed)

        def ik_step(position, orientation=None, steps: int = 1):
            err = 0.0
            for _ in range(int(steps)):
                err = s.ik_step(position, orientation if orientation is not None else s_forward())
                s.step()
            return {"error_m": round(err, 4)}

        def s_forward():
            from .robot import HAND_FORWARD_QUAT
            return HAND_FORWARD_QUAT

        def step(n: int = 1):
            s.step(int(n))
            return {"sim_time": round(s.sim_time(), 3)}

        def gripper_move(width: float):
            s.gripper_move(width)
            return {"width": round(s.gripper_width(), 4)}

        def gripper_grasp(width: float = 0.05, force: float = 70.0, speed: float = 0.1):
            s.close_gripper()
            w = s.gripper_width()
            ok = s.is_grasping() and abs(w - width) < 0.01
            return {"grasped": ok, "width": round(w, 4), "note": "force/speed accepted for compatibility"}

        h: dict[str, Callable[..., Any]] = {
            "ping": lambda: "pong",
            "get_status": get_status,
            "hand_pose": lambda: dict(zip(("position", "quat_wxyz"), [a.round(4) for a in s.hand_pose()])),
            "joint_positions": lambda: s.joint_positions().round(4),
            "sim_time": lambda: s.sim_time(),
            "step": step,
            # cart
            "go_to_cabinet": lambda cabinet: {"ok": s.go_to_cabinet(int(cabinet)), "cabinet": s.current_cabinet()},
            "current_cabinet": lambda: s.current_cabinet(),
            # arm
            "move_hand_to": move_hand_to,
            "move_relative": move_relative,
            "ik_step": ik_step,
            "move_joints": lambda joints, duration=1.5: {"reached": s.move_joints(joints[:7], duration)},
            "view_rack": lambda standoff=0.30: {"reached": s.view_rack(standoff)},
            "stow_arm": lambda: {"reached": s.stow_arm()},
            "stop": lambda: (s.stop() if self.busy else None, {"stopped": self.busy})[1],
            # gripper
            "gripper_move": gripper_move,
            "gripper_grasp": gripper_grasp,
            "open_gripper": lambda: gripper_move(0.08),
            "close_gripper": lambda: (s.close_gripper(), {"width": round(s.gripper_width(), 4), "is_grasping": s.is_grasping()})[1],
            "gripper_width": lambda: round(s.gripper_width(), 4),
            "is_grasping": lambda: s.is_grasping(),
            # camera
            "capture_frame": capture_frame,
            "get_wrist_frames": get_wrist_frames,
            "get_depth_at": get_depth_at,
            "get_robot_coords_at": get_robot_coords_at,
            "project_to_pixel": project,
            "get_camera_info": get_camera_info,
            "get_overview": get_overview,
            # caddies
            "get_pull_force": lambda: round(s.get_pull_force(), 2),
            "caddy_extraction": lambda cabinet, bay: round(s.caddy_extraction(int(cabinet), int(bay)), 4),
            "caddy_status": lambda cabinet, bay: s.caddy_status(int(cabinet), int(bay)),
            # triggers
            "set_fault": lambda cabinet, bay: (s.set_fault(int(cabinet), int(bay)), {"ok": True})[1],
            "set_led": lambda cabinet, bay, state: (s.set_led(int(cabinet), int(bay), state), {"ok": True})[1],
            "get_telemetry": lambda: s.get_telemetry(),
            "set_jam": lambda cabinet, bay, on=True: (s.set_jam(int(cabinet), int(bay), bool(on)), {"ok": True})[1],
            "connector_broken": lambda: s.connector_broken(),
            "drop_obstacle": lambda position=None, delay=0.0: {"position": s.drop_obstacle(position, float(delay))},
            "clear_obstacle": lambda: (s.clear_obstacle(), {"ok": True})[1],
            "reset": lambda seed=None: s.reset(seed=seed),
            # debug (ground truth: tests only)
            "debug_bay_pose": lambda cabinet, bay: s.debug_bay_pose(int(cabinet), int(bay)),
        }
        return h
