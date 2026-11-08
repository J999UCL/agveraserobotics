"""TCP JSON-lines server that runs inside the simulation loop.

    request : {"id": "1", "cmd": "swap", "args": {"slot": "A3"}}
    response: {"id": "1", "result": {...}, "error": null}

Polled from a step hook, so it keeps answering during blocking skills. While a skill
runs, only QUICK commands (status, health, images, stop, demo triggers) are accepted;
anything else gets "busy". Plain TCP, so it works on localhost, a LAN or a Modal tunnel.
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

from .kinematics import QUAT_DOWN_FINGERS_X, QUAT_DOWN_FINGERS_Y
from .sim import DataCenterSim, Stopped
from .skills import Skills

QUICK = {"ping", "get_status", "get_server_health", "get_event_log", "get_events", "get_cart_inventory",
         "get_installed", "capture_image", "get_camera_info", "pixel_to_world", "get_pull_force", "stop",
         "set_jam", "inject_fault", "list_slots"}


def _jpeg(rgb: np.ndarray, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _np(a):
    if isinstance(a, np.ndarray):
        return a.tolist()
    if isinstance(a, (np.floating, np.integer, np.bool_)):
        return a.item()
    raise TypeError(type(a))


class SimServer:
    def __init__(self, sim: DataCenterSim, host: str = "0.0.0.0", port: int = 5555) -> None:
        self.sim = sim
        self.skills = Skills(sim)
        self.sel = selectors.DefaultSelector()
        self.lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lsock.bind((host, port))
        self.lsock.listen()
        self.lsock.setblocking(False)
        self.sel.register(self.lsock, selectors.EVENT_READ, data=None)
        self.busy = False
        self._polling = False
        self.handlers = self._handlers()
        sim.step_hooks.append(self.poll)
        print(f"[sim-server] listening on {host}:{port}", flush=True)

    # ------------------------------------------------------------ networking
    def poll(self) -> None:
        if self._polling:
            return
        self._polling = True
        try:
            for key, _ in self.sel.select(timeout=0):
                if key.data is None:
                    conn, addr = self.lsock.accept()
                    conn.setblocking(False)
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.sel.register(conn, selectors.EVENT_READ, data={"buf": b""})
                    print(f"[sim-server] client connected {addr}", flush=True)
                elif key.fileobj.fileno() != -1:
                    self._read(key.fileobj, key.data)
        finally:
            self._polling = False

    def _drop(self, conn) -> None:
        try:
            self.sel.unregister(conn)
        except (KeyError, ValueError):
            pass
        conn.close()

    def _read(self, conn, data) -> None:
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

    def _send(self, conn, msg: dict) -> None:
        if conn.fileno() == -1:
            return
        try:
            conn.setblocking(True)
            conn.sendall((json.dumps(msg, default=_np) + "\n").encode())
            conn.setblocking(False)
        except OSError:
            self._drop(conn)

    def _handle(self, conn, line: bytes) -> None:
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
        quick = cmd in QUICK
        if self.busy and not quick:
            self._send(conn, {"id": rid, "result": None, "error": "busy: a motion is running (send 'stop' to abort)"})
            return
        if not quick:
            self.busy = True
            self.sim._stop = False
        was, self._polling = self._polling, False  # keep receiving (e.g. stop) while this runs
        try:
            result, error = fn(**args), None
        except Stopped:
            result, error = {"ok": False, "error": "stopped"}, None
        except Exception as e:
            result, error = None, f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            self._polling = was
            if not quick:
                self.busy = False
        self._send(conn, {"id": rid, "result": result, "error": error})

    # ------------------------------------------------------------- commands
    def _handlers(self) -> dict[str, Callable[..., Any]]:
        s, k = self.sim, self.skills
        quats = {"fingers_y": QUAT_DOWN_FINGERS_Y, "fingers_x": QUAT_DOWN_FINGERS_X}

        def capture_image(camera: str = "wrist", depth: bool = False, quality: int = 85):
            big = camera in ("overview", "tray_top")
            w, h = (1280, 720) if big else (None, None)
            rgb = s.render(camera, w, h)
            out = {"camera": camera, "jpeg_b64": _jpeg(rgb, quality), "width": rgb.shape[1], "height": rgb.shape[0]}
            if depth:
                d = s.render(camera, w, h, depth=True).astype(np.float16)
                out["depth_f16_b64"] = base64.b64encode(d.tobytes()).decode()
                out["K"] = s.camera_intrinsics(camera, w, h)
                pos, R = s.camera_pose(camera)
                out["cam_pos"], out["cam_rot"] = pos, R
            return out

        def move_tcp(x: float, y: float, z: float, orientation: str = "fingers_y", speed: float = 0.2):
            ok = s.move_tcp([x, y, z], quats[orientation], speed=speed)
            return {"ok": ok, "status": s.status()}

        def step(n: int = 1):
            s.step(int(n))
            return {"time": round(s.time, 3)}

        return {
            "ping": lambda: "pong",
            "get_status": s.status,
            "get_server_health": s.server_health,
            "get_event_log": s.event_log,
            "get_events": lambda since: [e for e in s.events if e["t"] >= float(since)],
            "get_installed": s.installed,
            "get_cart_inventory": s.cart_inventory,
            "list_slots": lambda: {n: {"kind": d.kind, "where": d.where} for n, d in s.layout.slots.items()},
            "get_pull_force": lambda: {"force_n": round(s.pull_force(), 2), "peak_n": round(s.peak_force(), 2)},
            "capture_image": capture_image,
            "get_camera_info": lambda camera="wrist": {"K": s.camera_intrinsics(camera), "pose": s.camera_pose(camera)},
            "pixel_to_world": lambda u, v, depth, camera="wrist": s.pixel_to_world(u, v, depth, camera).round(4),
            "step": step,
            "stop": lambda: (s.stop() if self.busy else None, {"stopped": self.busy})[1],
            # skills
            "pull_tray_out": k.pull_tray_out,
            "push_tray_in": k.push_tray_in,
            "view_tray": k.view_tray,
            "pick": k.pick,
            "place": k.place,
            "discard": k.discard,
            "swap": k.swap,
            "home": lambda: (s.home(), {"ok": True, "status": s.status()})[1],
            # low level
            "move_tcp": move_tcp,
            "move_joints": lambda joints, duration=2.0: (s.move_joints(joints, duration), {"ok": True})[1],
            "open_gripper": s.open_gripper,
            "close_gripper": s.close_gripper,
            # demo / test controls
            "set_jam": lambda slot, on=True: (s.set_jam(slot, on), {"ok": True})[1],
            "inject_fault": lambda slot, fault: (s.health.inject(s.installed()[slot], fault), {"ok": True})[1],
            "reset": lambda seed=None: s.reset(seed=seed),
        }
