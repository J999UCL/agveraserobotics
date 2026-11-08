"""MCP server exposing the data-center maintenance robot (MuJoCo sim) as agent tools.

    pip install -r requirements-client.txt
    DCSIM=host:port python mcp_server.py           (default localhost:5555)

The robot is a mobile manipulator in a data-center aisle (rows R1 and R2, racks 01-06) with
4 serviceable 2U servers (ids like "R1-05-U20"). Each server tray holds 8 RAM sticks
(slots A1-A8) and 5 fan modules (F1-F5). The robot carries spares on board (CD1-CD4 RAM,
CF1-CF2 fans) and a returns bin; its dock is at the spares depot.

Typical agent loop:
  get_work_orders / get_fleet_health -> pick a server -> go_to(server) -> pull_tray_out ->
  view_tray + capture_image("wrist") to confirm the amber LED -> swap(slot) ->
  get_server_health to verify -> push_tray_in -> next server ... -> go_home -> restock.
Safety reflexes (force abort on stuck parts, stopping for obstacles) run inside the robot;
tool results report them.
"""

from __future__ import annotations

import base64
import os
import sys
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dcsim_client import SimClient  # noqa: E402

mcp = FastMCP("datacenter-robot", instructions=__doc__)
_lock = threading.Lock()
_main: SimClient | None = None
_quick: SimClient | None = None


def _addr() -> tuple[str, int]:
    host, _, port = os.environ.get("DCSIM", "localhost:5555").rpartition(":")
    return host or "localhost", int(port)


def main_client() -> SimClient:
    global _main
    with _lock:
        if _main is None:
            _main = SimClient(*_addr())
        return _main


def quick_client() -> SimClient:
    """Second connection so status/stop/images work while a skill is running."""
    global _quick
    with _lock:
        if _quick is None:
            _quick = SimClient(*_addr())
        return _quick


# ----------------------------------------------------------------- diagnostics
@mcp.tool()
def get_work_orders(status: str | None = None) -> list[dict]:
    """Maintenance tickets opened by monitoring: server, rack, U, slot, component, fault, severity,
    status (open / in_progress / resolved). Filter with status="open"."""
    return quick_client().get_work_orders(status)


@mcp.tool()
def get_fleet_health() -> list[dict]:
    """One line per serviceable server: id, rack, U, overall health, problem slots, CPU temps."""
    return quick_client().get_fleet_health()


@mcp.tool()
def get_server_health(server: str) -> dict:
    """Full BMC-style report for one server: per-DIMM (A1-A8: ECC counts, status), per-fan
    (F1-F5: rpm, status), CPU temperatures, usable memory, tray position."""
    return quick_client().get_server_health(server)


@mcp.tool()
def get_event_log(server: str) -> list[dict]:
    """A server's system event log: faults, parts removed/inserted, verification, aborted pulls."""
    return quick_client().get_event_log(server)


@mcp.tool()
def get_status() -> dict:
    """Robot state: base position/heading/lift, current server, what it holds, onboard spares, trays."""
    return quick_client().get_status()


@mcp.tool()
def get_inventory() -> dict:
    """Spare parts on the robot (CD*/CF*) and on the depot shelf (DD*/DF*)."""
    c = quick_client()
    return {"onboard": c.get_onboard_inventory(), "depot": c.get_depot_inventory()}


@mcp.tool()
def capture_image(camera: str = "wrist") -> Image:
    """Camera image. "wrist": on the gripper (after view_tray it looks down on the fans and RAM;
    faulty parts show an amber LED). "follow": behind the robot. "overview", "aisle_end": room views."""
    r = quick_client().call("capture_image", camera=camera)
    return Image(data=base64.b64decode(r["jpeg_b64"]), format="jpeg")


# ------------------------------------------------------------------ navigation
@mcp.tool()
def go_to(server: str) -> dict:
    """Drive to a server's rack (e.g. "R1-05-U20"), face it and raise the arm to its height."""
    return main_client().go_to(server)


@mcp.tool()
def go_home() -> dict:
    """Drive back to the dock at the spares depot."""
    return main_client().go_home()


@mcp.tool()
def restock() -> dict:
    """At the dock: refill the robot's empty spare slots from the depot shelf."""
    return main_client().restock()


# --------------------------------------------------------------------- skills
@mcp.tool()
def pull_tray_out() -> dict:
    """Slide the current server's tray out of the rack to the service position."""
    return main_client().pull_tray_out()


@mcp.tool()
def push_tray_in() -> dict:
    """Slide the current server's tray back into the rack."""
    return main_client().push_tray_in()


@mcp.tool()
def view_tray() -> dict:
    """Hold the wrist camera above the open tray (then call capture_image("wrist"))."""
    return main_client().view_tray()


@mcp.tool()
def swap(slot: str) -> dict:
    """Hot-swap the part in a slot of the current server (A1-A8 RAM, F1-F5 fans): pull it, drop it in
    the returns bin, install a matching onboard spare. Aborts safely if the part is stuck."""
    return main_client().swap(slot)


@mcp.tool()
def pick(slot: str) -> dict:
    """Grasp and lift the part in a slot: A1-A8/F1-F5 (current server), CD1-CF2 (onboard), DD1-DF3 (depot)."""
    return main_client().pick(slot)


@mcp.tool()
def place(slot: str) -> dict:
    """Install the held part into an empty, compatible slot."""
    return main_client().place(slot)


@mcp.tool()
def discard() -> dict:
    """Drop the held part into the robot's returns bin."""
    return main_client().discard()


@mcp.tool()
def home() -> dict:
    """Move the arm to its ready pose."""
    return main_client().home()


@mcp.tool()
def stop() -> dict:
    """Immediately stop the current motion."""
    return quick_client().stop()


# ---------------------------------------------------------------- demo controls
@mcp.tool()
def inject_fault(server: str, slot: str, fault: str) -> dict:
    """DEMO: make a part fail (opens a work order). fault: dimm_uncorrectable | dimm_correctable |
    fan_failed | fan_degraded."""
    return quick_client().inject_fault(server, slot, fault)


@mcp.tool()
def set_jam(server: str, slot: str, on: bool = True) -> dict:
    """DEMO: make a part mechanically stuck (pulling it builds force until the robot aborts)."""
    return quick_client().set_jam(slot, server, on)


@mcp.tool()
def spawn_technician(mode: str = "block", delay: float = 3.0, duration: float = 5.0) -> dict:
    """DEMO: a technician walks into the aisle ahead of the robot ("cross" or "block"); the robot's
    lidar makes it stop and wait during go_to/go_home."""
    return quick_client().spawn_technician(mode, None, delay, duration)


@mcp.tool()
def get_lidar_scan() -> dict:
    """The robot's front lidar: 37 ranges (m) from -90 to +90 degrees; None = nothing within 4 m."""
    return quick_client().get_lidar_scan()


@mcp.tool()
def reset(seed: int | None = None) -> dict:
    """DEMO: reset the whole data center. A seed randomizes which parts are faulty."""
    return main_client().reset(seed)


if __name__ == "__main__":
    mcp.run()
