"""All scene dimensions in one place (meters, robot base frame).

Frame convention (shared with perception + control):
  * Origin = Franka base (panda_link0), sitting on top of the cart.
  * +X points from the robot toward the rack, +Y to the robot's left, +Z up.
  * The floor is at z = -CART.height.

So anything perception unprojects into "world" coordinates is already in the robot
base frame. No extra transform needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RACK_UNIT = 0.04445  # 1U = 1.75 in


@dataclass(frozen=True)
class CartCfg:
    height: float = 0.35  # floor -> top of cart = robot base height
    length: float = 0.70  # along X
    width: float = 0.55  # along Y
    center_x: float = -0.12  # cart extends behind the robot
    wheel_radius: float = 0.06


@dataclass(frozen=True)
class RackCfg:
    front_x: float = 0.62  # X of the rack's front posts (robot base -> rack face)
    units: int = 42
    outer_width: float = 0.60
    opening_width: float = 0.450  # 19-inch rail opening
    depth: float = 1.00
    plinth: float = 0.10  # floor -> bottom of U1
    post: float = 0.04  # post cross-section


@dataclass(frozen=True)
class ServerCfg:
    """The 2U storage server holding the hot-swap drive caddies."""

    # U20 puts the bays at base z ~= 0.61-0.67: the band where a forward-pointing Franka hand
    # reaches every bay with margin on all joint limits (probed). ~1 m above the floor.
    u_position: int = 20  # bottom U of the server (U1 = lowest)
    u_height: int = 2
    width: float = 0.430
    depth: float = 0.70
    recess: float = 0.015  # front bezel sits this far behind the rack posts
    bay_cols: int = 4
    bay_rows: int = 3


@dataclass(frozen=True)
class CaddyCfg:
    """3.5-inch drive caddy, face as seen from the front."""

    face_width: float = 0.098
    face_height: float = 0.026
    depth: float = 0.16
    # pull tab the gripper pinches from the sides (fingers close along Y)
    tab_width: float = 0.050  # < 0.08 gripper opening, leaves ~25 mm finger clearance each side
    tab_height: float = 0.018
    tab_depth: float = 0.030  # how far it sticks out of the caddy face
    led_size: float = 0.006


@dataclass(frozen=True)
class CameraCfg:
    resolution: tuple[int, int] = (480, 640)  # (height, width)
    focal_length_mm: float = 12.0  # ~82 deg horizontal FOV, sees the whole 2U face from ~0.3 m
    horizontal_aperture_mm: float = 20.955
    near: float = 0.02
    far: float = 10.0
    # mounted on panda_hand, offset along hand +X (above the gripper when the hand
    # points at the rack) and set back from the fingertips so fingers don't fill the frame
    hand_offset: tuple[float, float, float] = (0.065, 0.0, 0.045)


@dataclass(frozen=True)
class SceneCfg:
    cart: CartCfg = field(default_factory=CartCfg)
    rack: RackCfg = field(default_factory=RackCfg)
    server: ServerCfg = field(default_factory=ServerCfg)
    caddy: CaddyCfg = field(default_factory=CaddyCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    faulty_bay: tuple[int, int] = (1, 2)  # (row, col), row 0 = top
    physics_dt: float = 1.0 / 60.0

    @property
    def floor_z(self) -> float:
        return -self.cart.height

    def u_bottom_z(self, u: int) -> float:
        """Base-frame Z of the bottom edge of rack unit `u` (1-indexed)."""
        return self.floor_z + self.rack.plinth + (u - 1) * RACK_UNIT
