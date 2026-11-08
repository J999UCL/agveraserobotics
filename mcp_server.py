"""MCP server exposing the data-center robot sim as agent tools.

    pip install -r requirements-client.txt
    DCSIM=host:port python mcp_server.py           (default localhost:5555)

Typical agent loop: get_server_health -> decide what is faulty -> pull_tray_out ->
capture_image("wrist") to confirm the amber LED -> swap(slot) -> get_server_health to
verify -> push_tray_in. Lower-level tools (pick/place/discard/move_tcp) exist for
custom sequences, e.g. removing a part without replacing it.
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
def get_server_health() -> dict:
    """BMC-style health report: overall status, per-DIMM (slots A1-A8: ECC counts, status) and
    per-fan (slots F1-F5: rpm, status) readings, CPU temperatures, usable memory, tray position."""
    return quick_client().get_server_health()


@mcp.tool()
def get_event_log() -> list[dict]:
    """The server's system event log (faults, part removed/inserted, verification results)."""
    return quick_client().get_event_log()


@mcp.tool()
def get_cart_inventory() -> dict:
    """Spare parts on the robot's cart: slots CD1-CD4 (DIMMs) and CF1-CF2 (fans) -> part id or null."""
    return quick_client().get_cart_inventory()


@mcp.tool()
def get_status() -> dict:
    """Robot state: gripper position, what it is holding, pull force, tray extension."""
    return quick_client().get_status()


@mcp.tool()
def capture_image(camera: str = "wrist") -> Image:
    """Camera image. "wrist": on the gripper (after view_tray it looks down on the fans and RAM; faulty
    parts have an amber LED). "overview": third-person view. "tray_top": fixed top-down view of the tray."""
    r = quick_client().call("capture_image", camera=camera)
    return Image(data=base64.b64decode(r["jpeg_b64"]), format="jpeg")


# --------------------------------------------------------------------- skills
@mcp.tool()
def pull_tray_out() -> dict:
    """Grab the server tray's handle and slide it out on its rails to the service position."""
    return main_client().pull_tray_out()


@mcp.tool()
def push_tray_in() -> dict:
    """Slide the tray back into the rack and return the arm home."""
    return main_client().push_tray_in()


@mcp.tool()
def view_tray() -> dict:
    """Position the wrist camera above the open tray (then call capture_image("wrist"))."""
    return main_client().view_tray()


@mcp.tool()
def swap(slot: str) -> dict:
    """Full hot-swap of the part in a tray slot (A1-A8 RAM, F1-F5 fans): pull it, drop it in the
    used-parts bin, take a matching spare from the cart, install it. Aborts safely if the part is stuck."""
    return main_client().swap(slot)


@mcp.tool()
def pick(slot: str) -> dict:
    """Grasp and lift the part in a tray slot (A1-A8, F1-F5) or cart slot (CD1-CD4, CF1-CF2).
    Watches the extraction force and aborts if the part is stuck."""
    return main_client().pick(slot)


@mcp.tool()
def place(slot: str) -> dict:
    """Install the held part into an empty, compatible slot."""
    return main_client().place(slot)


@mcp.tool()
def discard() -> dict:
    """Drop the held part into the used-parts bin on the cart."""
    return main_client().discard()


@mcp.tool()
def home() -> dict:
    """Move the arm to its ready pose."""
    return main_client().home()


@mcp.tool()
def stop() -> dict:
    """Immediately stop the current motion."""
    return quick_client().stop()


# ------------------------------------------------------------------ low level
@mcp.tool()
def move_tcp(x: float, y: float, z: float, orientation: str = "fingers_y") -> dict:
    """Move the gripper tip to x, y, z (meters, robot frame: +X toward the rack, +Y left, +Z up),
    pointing down. orientation: "fingers_y" (fingers close along Y) or "fingers_x"."""
    return main_client().move_tcp(x, y, z, orientation)


@mcp.tool()
def open_gripper() -> dict:
    """Open the gripper (drops whatever it holds; seats it if it is lined up with a slot)."""
    return main_client().open_gripper()


@mcp.tool()
def close_gripper() -> dict:
    """Close the gripper; reports which part (if any) it is now holding."""
    return main_client().close_gripper()


# ---------------------------------------------------------------- demo controls
@mcp.tool()
def inject_fault(slot: str, fault: str) -> dict:
    """DEMO: make the part in `slot` fail. fault: dimm_uncorrectable | dimm_correctable | fan_failed | fan_degraded."""
    return quick_client().inject_fault(slot, fault)


@mcp.tool()
def set_jam(slot: str, on: bool = True) -> dict:
    """DEMO: make the part in `slot` mechanically stuck (pulling it builds force until the robot aborts)."""
    return quick_client().set_jam(slot, on)


@mcp.tool()
def reset(seed: int | None = None) -> dict:
    """DEMO: reset the scene. Default faults are A3 (RAM) and F2 (fan); a seed randomizes them."""
    return main_client().reset(seed)


if __name__ == "__main__":
    mcp.run()
