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

QUICK = {"ping", "get_status", "get_fleet_health", "get_server_health", "get_event_log", "get_work_orders",
         "get_events", "get_installed", "get_onboard_inventory", "get_depot_inventory", "list_servers",
         "capture_image", "get_camera_info", "pixel_to_world", "get_pull_force", "stop", "set_jam", "inject_fault",
         "get_lidar_scan", "get_nav_log", "spawn_technician"}


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
        except Stopped as e:
            result, error = {"ok": False, "error": str(e) or "stopped"}, None
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
        quats = {"down": QUAT_DOWN_FINGERS_Y, "down_fingers_x": QUAT_DOWN_FINGERS_X}
        big_cams = ("follow", "overview", "aisle_end")

        def capture_image(camera: str = "wrist", depth: bool = False, quality: int = 85):
            w, h = (1280, 720) if camera in big_cams else (None, None)
            rgb = s.render(camera, w, h)
            out = {"camera": camera, "jpeg_b64": _jpeg(rgb, quality), "width": rgb.shape[1], "height": rgb.shape[0]}
            if depth:
                d = s.render(camera, w, h, depth=True).astype(np.float16)
                out["depth_f16_b64"] = base64.b64encode(d.tobytes()).decode()
                out["K"] = s.camera_intrinsics(camera, w, h)
                pos, R = s.camera_pose(camera)
                out["cam_pos"], out["cam_rot"] = pos, R
            return out

        def move_tcp(x: float, y: float, z: float, orientation: str = "down", speed: float = 0.2):
            """x, y, z in the robot frame (relative to the arm's base, +X forward)."""
            ok = s.move_tcp(s.to_world([x, y, z]), s.robot_quat(quats[orientation]), speed=speed)
            return {"ok": ok, "status": s.status()}

        def step(n: int = 1):
            s.step(int(n))
            return {"time": round(s.time, 3)}

        def server_health(server: str | None = None):
            server = server or s.active_server
            if server is None:
                raise ValueError("give a server id (or go_to one first)")
            return s.server_health(server)

        def set_jam(slot: str, server: str | None = None, on: bool = True):
            s.set_jam(server or s.active_server, slot, on)
            return {"ok": True}

        def inject_fault(server: str, slot: str, fault: str):
            s.inject_fault(server, slot, fault)
            return {"ok": True, "work_order": s.work_orders[-1]}

        return {
            "ping": lambda: "pong",
            "get_status": s.status,
            # diagnostics
            "get_fleet_health": s.fleet_health,
            "get_server_health": server_health,
            "get_event_log": lambda server: s.health[server].sel,
            "get_work_orders": lambda status=None: [w for w in s.work_orders if status is None or w["status"] == status],
            "get_events": lambda since=0.0: [e for e in s.events if e["t"] >= float(since)],
            "get_installed": lambda server=None: s.installed(server or s.active_server),
            "get_onboard_inventory": lambda: s.storage("onboard"),
            "get_depot_inventory": lambda: s.storage("depot"),
            "list_servers": lambda: [{"server": sid, "rack": f"R{sv.row}-{sv.rack:02d}", "u": sv.u}
                                     for sid, sv in s.layout.servers.items()],
            "get_pull_force": lambda: {"force_n": round(s.pull_force(), 2), "peak_n": round(s.peak_force(), 2)},
            "get_lidar_scan": s.lidar_scan,
            "get_nav_log": lambda: s.nav_log,
            # cameras
            "capture_image": capture_image,
            "get_camera_info": lambda camera="wrist": {"K": s.camera_intrinsics(camera), "pose": s.camera_pose(camera)},
            "pixel_to_world": lambda u, v, depth, camera="wrist": s.pixel_to_world(u, v, depth, camera).round(4),
            "step": step,
            "stop": lambda: (s.stop() if self.busy else None, {"stopped": self.busy})[1],
            # skills
            "go_to": k.go_to,
            "go_home": k.go_home,
            "restock": k.restock,
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
            "set_jam": set_jam,
            "inject_fault": inject_fault,
            "spawn_technician": lambda mode="cross", x=None, delay=0.0, duration=4.0: s.spawn_technician(mode, x, delay, duration),
            "reset": lambda seed=None: s.reset(seed=seed),
        }
