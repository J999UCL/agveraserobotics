"""All dimensions (meters) in the ROBOT BASE frame = MuJoCo world frame.

  origin : Franka base (link0), on top of the service cart
  +X     : toward the server rack     +Y : robot's left     +Z : up
  floor  : z = -CART_HEIGHT

Reach note: the Franka grasping straight down is comfortable ~0.35-0.70 m from its base,
so the tray travel and its internal layout are chosen to keep the handle, fan wall and
RAM bank inside that band when the tray is pulled out to its service position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

U = 0.04445  # one rack unit

CART_HEIGHT = 0.40


@dataclass(frozen=True)
class RackCfg:
    front_x: float = 0.70
    width: float = 0.60
    opening: float = 0.45
    depth: float = 1.00
    units: int = 42
    plinth: float = 0.10


@dataclass(frozen=True)
class TrayCfg:
    """The serviceable 2U server. Local frame: origin at front-bottom-center, +X into the rack."""

    width: float = 0.44
    depth: float = 0.62
    height: float = 0.088
    bottom_z: float = 0.03  # world z of the tray floor (~0.43 m above the real floor)
    travel: float = 0.32  # how far it slides out on its rails to the service position
    board_z: float = 0.006  # top of the motherboard
    # handle: a vertical pull tab sticking out of the front bezel. Gripped from the FRONT,
    # gripper tilted 75 deg down, fingers closing on its sides (along Y). The hand stays in
    # front of the rack the whole time.
    handle_x: float = -0.032  # tab center (tray frame)
    handle_z: float = 0.068  # near the top of the bezel: keeps the whole pull inside the arm's reach
    handle_len: float = 0.055  # how far it sticks out (X)
    handle_height: float = 0.036
    handle_thick: float = 0.012  # Y
    # fan wall
    fan_x: float = 0.12
    fan_size: tuple[float, float, float] = (0.05, 0.07, 0.06)  # x, y, z
    fan_ys: tuple[float, ...] = (0.168, 0.084, 0.0, -0.084, -0.168)
    # RAM bank (abstracted: thicker and wider spaced than real DIMMs so fingers fit)
    dimm_x: float = 0.235  # center
    dimm_size: tuple[float, float, float] = (0.13, 0.008, 0.035)
    dimm_ys: tuple[float, ...] = (0.175, 0.125, 0.075, 0.025, -0.025, -0.075, -0.125, -0.175)
    socket_h: float = 0.008
    # CPUs / PSU (static)
    cpu_x: float = 0.40
    cpu_ys: tuple[float, ...] = (0.11, -0.11)


@dataclass(frozen=True)
class CartCfg:
    deck_x: tuple[float, float] = (-0.50, 0.28)
    deck_halfwidth: float = 0.30
    wing_x: tuple[float, float] = (0.05, 0.45)  # side wings for spares + bin
    wing_halfwidth: float = 0.70
    # spares shelf on the robot's right (-Y), used-parts bin on the left (+Y)
    spare_dimm_x: float = 0.34
    spare_dimm_ys: tuple[float, ...] = (-0.40, -0.45, -0.50, -0.55)
    spare_fan_x: float = 0.22
    spare_fan_ys: tuple[float, ...] = (-0.44, -0.54)
    bin_center: tuple[float, float] = (0.27, 0.50)
    bin_size: tuple[float, float, float] = (0.26, 0.24, 0.12)


@dataclass(frozen=True)
class CameraCfg:
    width: int = 640
    height: int = 480
    wrist_fovy: float = 70.0
    # on the hand, offset sideways from the fingers, looking where the fingers point
    wrist_pos: tuple[float, float, float] = (0.06, 0.0, 0.03)


@dataclass(frozen=True)
class SimCfg:
    rack: RackCfg = field(default_factory=RackCfg)
    tray: TrayCfg = field(default_factory=TrayCfg)
    cart: CartCfg = field(default_factory=CartCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    timestep: float = 0.002
    control_dt: float = 0.02  # arm targets are updated at 50 Hz
    # component seating physics
    seat_release_n: float = 12.0  # pull force that unseats a healthy part
    jam_break_n: float = 40.0  # pulling a jammed part harder than this damages the socket
    capture_pos_tol: float = 0.012  # m, how close a part must be to a slot to seat
    capture_yaw_tol_deg: float = 10.0
    grasp_tol: float = 0.02  # m, TCP to part grasp point for the gripper to "take" it
