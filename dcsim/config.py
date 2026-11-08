"""Scene configuration. Units: meters, radians. World frame: floor at z=0.

Layout: one cold aisle along +X. Row R1's rack fronts face the aisle from +Y, row R2's from -Y.
Each rack has its own frame: origin at the front-bottom-center, +X into the rack, +Y to the
left when facing it. Servers, trays and slots are defined in rack/tray frames, so the
manipulation geometry (tested in the single-rack version) is identical at every rack:
the Franka base stands SERVICE_STANDOFF in front of the rack, facing it, and the lift puts
the tray floor TRAY_ABOVE_BASE above the Franka base.
"""

from __future__ import annotations

from dataclasses import dataclass, field

U = 0.04445  # one rack unit
SERVICE_STANDOFF = 0.70  # Franka base -> rack front, when servicing
TRAY_ABOVE_BASE = 0.03  # tray floor height above the Franka base when servicing


@dataclass(frozen=True)
class RoomCfg:
    racks_per_row: int = 6
    rack_pitch: float = 0.61
    row_x0: float = 0.0  # x of the first rack's center
    aisle_halfwidth: float = 0.90  # cold aisle: rack fronts at y = +/- this
    room_x: tuple[float, float] = (-2.6, 5.2)
    room_y: tuple[float, float] = (-3.4, 3.4)
    ceiling: float = 3.0


@dataclass(frozen=True)
class RackCfg:
    width: float = 0.60
    opening: float = 0.45
    depth: float = 1.00
    units: int = 42
    plinth: float = 0.10


@dataclass(frozen=True)
class TrayCfg:
    """The serviceable 2U server. Tray frame: origin at front-bottom-center, +X into the rack."""

    width: float = 0.44
    depth: float = 0.62
    height: float = 0.088
    travel: float = 0.32
    board_z: float = 0.006
    # pull tab, gripped from the front (gripper tilted 75 deg down, fingers on its sides)
    handle_x: float = -0.032
    handle_z: float = 0.068
    handle_len: float = 0.055
    handle_height: float = 0.036
    handle_thick: float = 0.012
    fan_x: float = 0.12
    fan_size: tuple[float, float, float] = (0.05, 0.07, 0.06)
    fan_ys: tuple[float, ...] = (0.168, 0.084, 0.0, -0.084, -0.168)
    dimm_x: float = 0.235
    dimm_size: tuple[float, float, float] = (0.13, 0.008, 0.035)
    dimm_ys: tuple[float, ...] = (0.175, 0.125, 0.075, 0.025, -0.025, -0.075, -0.125, -0.175)
    socket_h: float = 0.008
    cpu_x: float = 0.40
    cpu_ys: tuple[float, ...] = (0.11, -0.11)


@dataclass(frozen=True)
class RobotCfg:
    """Mobile base + lift. Onboard storage positions are in the Franka base (link0) frame."""

    lift_range: tuple[float, float] = (0.35, 1.25)  # Franka base height above the floor
    chassis_size: tuple[float, float, float] = (0.62, 0.50, 0.28)  # x, y, z
    chassis_offset_x: float = -0.12  # chassis center behind the Franka base
    drive_speed: float = 0.6  # m/s
    turn_speed: float = 1.2  # rad/s
    lift_speed: float = 0.25  # m/s
    # onboard spares cassette (right side, -Y) and returns bin (left side, +Y), on the lift carriage
    spare_dimm_x: float = 0.12
    spare_dimm_ys: tuple[float, ...] = (-0.32, -0.37, -0.42, -0.47)
    spare_fan_x: float = -0.07
    spare_fan_ys: tuple[float, ...] = (-0.34, -0.44)
    bin_center: tuple[float, float] = (0.02, 0.39)
    bin_size: tuple[float, float, float] = (0.30, 0.15, 0.10)


@dataclass(frozen=True)
class ServerSpec:
    id: str  # e.g. "R1-05-U20"
    row: int  # 1 or 2
    rack: int  # 1..6
    u: int  # bottom rack unit of the 2U server


@dataclass(frozen=True)
class CameraCfg:
    width: int = 640
    height: int = 480
    wrist_fovy: float = 70.0
    wrist_pos: tuple[float, float, float] = (0.06, 0.0, 0.03)


@dataclass(frozen=True)
class SimCfg:
    room: RoomCfg = field(default_factory=RoomCfg)
    rack: RackCfg = field(default_factory=RackCfg)
    tray: TrayCfg = field(default_factory=TrayCfg)
    robot: RobotCfg = field(default_factory=RobotCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    servers: tuple[ServerSpec, ...] = (
        ServerSpec("R1-02-U10", 1, 2, 10),
        ServerSpec("R1-05-U20", 1, 5, 20),
        ServerSpec("R2-03-U14", 2, 3, 14),
        ServerSpec("R2-06-U24", 2, 6, 24),
    )
    timestep: float = 0.002
    control_dt: float = 0.02
    seat_release_n: float = 12.0
    jam_break_n: float = 40.0
    capture_pos_tol: float = 0.012
    capture_yaw_tol_deg: float = 10.0
    grasp_tol: float = 0.02

    def u_bottom(self, u: int) -> float:
        return self.rack.plinth + (u - 1) * U
