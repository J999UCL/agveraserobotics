"""isaac-mcp: MCP tools for the simulated hot-swap robot.

Tool names and arguments follow panda-mcp (github.com/ratsbane/panda-mcp, MIT) where an
equivalent exists, so prompts/plans written for that project carry over. Everything is
forwarded to the simulation server (scripts/run_sim_server.py, locally or on Modal).

    pip install "mcp<2" numpy pillow
    HOTSWAP_SIM=host:port python -m isaac_mcp.server          (default localhost:5555)

Differences from the real-robot panda-mcp:
  * The default gripper orientation points INTO the rack (+X), not down at a table.
  * pick_at / place_at (table-top) are replaced by rack tools (view_rack, pull caddies).
  * Extra environment tools: go_to_cabinet, get_pull_force, set_jam, drop_obstacle, ...
"""

from __future__ import annotations

import base64
import math
import os
import sys
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hotswap_client import SimClient  # noqa: E402

mcp = FastMCP("isaac-mcp")
_lock = threading.Lock()
_client: SimClient | None = None
_quick: SimClient | None = None  # second connection so stop() works during a move


def _addr() -> tuple[str, int]:
    host, _, port = os.environ.get("HOTSWAP_SIM", "localhost:5555").rpartition(":")
    return host or "localhost", int(port)


def sim() -> SimClient:
    global _client
    with _lock:
        if _client is None:
            _client = SimClient(*_addr())
        return _client


def quick() -> SimClient:
    global _quick
    with _lock:
        if _quick is None:
            _quick = SimClient(*_addr())
        return _quick


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """Extrinsic XYZ (R = Rz(yaw) Ry(pitch) Rx(roll)) -> wxyz."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


# ---------------------------------------------------------------- panda-mcp style
@mcp.tool()
def connect() -> dict:
    """Connect to the simulated robot. Safe to call repeatedly."""
    return {"connected": sim().ping() == "pong", "sim": "%s:%d" % _addr()}


@mcp.tool()
def get_status() -> dict:
    """Robot state: joints, gripper pose (robot frame), gripper width, grasp, pull force, cabinet."""
    return quick().get_status()


@mcp.tool()
def move_cartesian(x: float, y: float, z: float, roll: float | None = None, pitch: float | None = None,
                   yaw: float | None = None, confirmed: bool = True) -> dict:
    """Move the gripper (panda_hand) to x, y, z meters in the robot frame (+X toward the rack,
    +Y left, +Z up). Without roll/pitch/yaw the gripper points into the rack. Reachable band
    with that orientation: z 0.55-0.65, x 0.30-0.52, |y| < 0.16."""
    q = None if None in (roll, pitch, yaw) else _rpy_to_quat(roll, pitch, yaw)
    return sim().move_hand_to([x, y, z], orientation=q)


@mcp.tool()
def move_relative(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, confirmed: bool = True) -> dict:
    """Move the gripper by a delta in meters (robot frame)."""
    return sim().move_relative(dx, dy, dz)


@mcp.tool()
def move_cartesian_sequence(waypoints: list[dict], confirmed: bool = True) -> dict:
    """Move through waypoints [{x, y, z}, ...] in order. Stops at the first unreachable one."""
    out = []
    for wp in waypoints:
        r = sim().move_hand_to([wp["x"], wp["y"], wp["z"]])
        out.append(r)
        if not r.get("reached"):
            break
    return {"results": out}


@mcp.tool()
def move_joints(joints: list[float], confirmed: bool = True) -> dict:
    """Move to a 7-joint configuration (radians)."""
    return sim().move_joints(joints)


@mcp.tool()
def gripper_move(width: float) -> dict:
    """Open/close the gripper to a width in meters (0 = closed, 0.08 = open)."""
    return sim().gripper_move(width)


@mcp.tool()
def gripper_grasp(width: float = 0.05, force: float = 70.0, speed: float = 0.1) -> dict:
    """Close on an object of the expected width (pull tabs are 0.05 m wide). Returns grasped=True
    if the fingers stopped within 1 cm of that width."""
    return sim().call("gripper_grasp", width=width, force=force, speed=speed)


@mcp.tool()
def stop() -> dict:
    """Immediately stop the current motion and hold position."""
    return quick().stop()


# ------------------------------------------------------------------- camera / depth
@mcp.tool()
def capture_frame() -> Image:
    """Wrist camera RGB image (640x480 JPEG)."""
    r = quick().call("capture_frame")
    return Image(data=base64.b64decode(r["jpeg_b64"]), format="jpeg")


@mcp.tool()
def get_overview_image() -> Image:
    """Third-person view of the aisle (1280x720 JPEG)."""
    r = quick().call("get_overview")
    return Image(data=base64.b64decode(r["jpeg_b64"]), format="jpeg")


@mcp.tool()
def get_camera_info() -> dict:
    """Wrist camera resolution and intrinsics K."""
    return quick().get_camera_info()


@mcp.tool()
def get_depth_at(x: int, y: int) -> dict:
    """Depth in meters at wrist-image pixel (x = column, y = row)."""
    return {"depth_m": quick().get_depth_at(x, y)}


@mcp.tool()
def get_robot_coords_at(x: int, y: int) -> dict:
    """Robot-frame xyz of the surface seen at wrist-image pixel (x = column, y = row)."""
    return {"position": quick().get_robot_coords_at(x, y)}


# -------------------------------------------------------------- rack environment
@mcp.tool()
def go_to_cabinet(cabinet: int) -> dict:
    """Stow the arm and drive the cart to cabinet 0, 1 or 2 (left to right)."""
    return sim().go_to_cabinet(cabinet)


@mcp.tool()
def view_rack() -> dict:
    """Move so the wrist camera looks square at the storage server's 12 drive bays."""
    return sim().view_rack()


@mcp.tool()
def stow_arm() -> dict:
    """Return the arm to its safe ready pose."""
    return sim().stow_arm()


@mcp.tool()
def get_pull_force() -> dict:
    """Force (N) resisting the caddy being pulled. Healthy caddies peak ~15 N; jammed ones keep climbing."""
    return {"force_n": quick().get_pull_force()}


@mcp.tool()
def caddy_status(cabinet: int, bay: int) -> dict:
    """Extraction distance, force, jammed/released/connector_broken for one caddy."""
    return quick().caddy_status(cabinet, bay)


@mcp.tool()
def get_telemetry() -> list[dict]:
    """Data-center monitoring alerts (which cabinet/bay failed) and physical events."""
    return quick().get_telemetry()


@mcp.tool()
def set_jam(cabinet: int, bay: int, on: bool = True) -> dict:
    """DEMO: make a caddy stuck so pulling it builds force."""
    return quick().set_jam(cabinet, bay, on)


@mcp.tool()
def drop_obstacle() -> dict:
    """DEMO: drop a loose cable between the gripper and the rack."""
    return quick().drop_obstacle()


@mcp.tool()
def clear_obstacle() -> dict:
    """DEMO: remove the dropped cable."""
    return quick().clear_obstacle()


@mcp.tool()
def reset(seed: int | None = None) -> dict:
    """Reset the whole scene. A seed picks a random faulty bay."""
    return sim().reset(seed)


if __name__ == "__main__":
    mcp.run()
