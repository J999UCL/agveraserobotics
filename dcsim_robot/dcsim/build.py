"""Build the data-center MuJoCo model (MjSpec) around Menagerie's Franka Panda.

Robot: the Franka's link0 gets four driven joints (base x, y, yaw + lift z), so the arm,
the lift carriage and the onboard storage all ride on them. The wheeled chassis is a
kinematic (mocap) body that the sim keeps under the arm every tick.

Collision layers (contype / conaffinity):
  robot + scenery     : 1 / 1
  visual-only         : 0 / 0
  tray pull tab       : 1 / 0   (only the gripper touches it)
  components          : 2 / 1   (touch the gripper, robot deck, bins, floor; not trays/each other)

Every component has three welds, toggled at runtime:
  weld_slot_<id>  : seated in its server tray slot (body2 = that tray)
  weld_hand_<id>  : held by the gripper (body2 = hand)
  weld_store_<id> : parked in onboard storage or on the depot shelf (body2 = link0 or world)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .config import U, SimCfg

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "franka_emika_panda"

BOX, CYL, PLANE = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_PLANE
SLIDE, HINGE = mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE

FRAME = [0.06, 0.06, 0.07, 1]
PANEL = [0.11, 0.11, 0.12, 1]
CHASSIS = [0.55, 0.57, 0.60, 1]
BOARD = [0.05, 0.30, 0.15, 1]
DIMM = [0.08, 0.20, 0.10, 1]
DIMM_CHIP = [0.03, 0.03, 0.03, 1]
SOCKET = [0.02, 0.02, 0.02, 1]
FAN = [0.12, 0.12, 0.13, 1]
FAN_TAB = [0.10, 0.30, 0.75, 1]
HEATSINK = [0.70, 0.72, 0.75, 1]
ROBOT_WHITE = [0.90, 0.91, 0.92, 1]
ROBOT_GREY = [0.22, 0.23, 0.25, 1]

DEPOT_SHELF_Z = 0.80
LIDAR_RAYS = 37  # across the front 180 degrees (5 deg apart)
LIDAR_Z = 0.20
LIDAR_RANGE = 4.0


@dataclass
class SlotDef:
    name: str  # tray: "A1".."A8", "F1".."F5"; onboard: "CD1".., "CF1"..; depot: "DD1".., "DF1"..
    kind: str  # "dimm" | "fan"
    where: str  # "tray" | "onboard" | "depot"
    server: str | None  # server id for tray slots
    pos: np.ndarray  # in the parent frame (tray, link0 or depot)


@dataclass
class ComponentDef:
    id: str
    kind: str
    slot_key: str  # initial slot key ("R1-05-U20:A3", "onboard:CD1", "depot:DD2")


@dataclass
class ServerDef:
    id: str
    index: int
    row: int
    rack: int
    u: int
    tray_body: str
    slide_joint: str
    handle_eq: str
    rack_pos: np.ndarray  # world, rack front-bottom-center
    rack_yaw: float
    tray_bottom: float  # height of the tray floor above the floor


@dataclass
class Layout:
    slots: dict[str, SlotDef]  # key "server:slot" | "onboard:slot" | "depot:slot"
    components: dict[str, ComponentDef]
    servers: dict[str, ServerDef]
    racks: dict[str, tuple[np.ndarray, float]]  # "R1-05" -> (front-center pos, yaw)
    depot: tuple[np.ndarray, float]


def yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rack_pose(cfg: SimCfg, row: int, rack: int) -> tuple[np.ndarray, float]:
    """Front-bottom-center of a rack and its yaw (rack +X points into the rack)."""
    x = cfg.room.row_x0 + (rack - 1) * cfg.room.rack_pitch
    if row == 1:
        return np.array([x, cfg.room.aisle_halfwidth, 0.0]), np.pi / 2
    return np.array([x, -cfg.room.aisle_halfwidth, 0.0]), -np.pi / 2


def depot_pose(cfg: SimCfg) -> tuple[np.ndarray, float]:
    """Spares depot at the aisle's start; its +X points away from the aisle (like a rack)."""
    return np.array([cfg.room.row_x0 - 1.65, 0.0, 0.0]), np.pi


def _box(body, name, center, half, rgba=None, contype=1, conaffinity=1, material=None, group=0, mass=None):
    kw = dict(name=name, type=BOX, size=list(half), pos=list(center), contype=contype, conaffinity=conaffinity, group=group)
    if material:
        kw["material"] = material  # no rgba: an explicit geom rgba would override the material
    else:
        kw["rgba"] = list(rgba)
    if mass is not None:
        kw["mass"] = mass
    return body.add_geom(**kw)


def _weld(spec, name, b1, b2, active, anchor):
    eq = spec.add_equality(name=name, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                           name1=b1, name2=b2, active=active, solref=[0.01, 1.0])
    # weld data: anchor on body2 (3), point on body1 (3), relative quat (4), torquescale (1).
    # Zero quat = the compiler takes the relative pose from the initial placement.
    data = np.zeros(len(eq.data))
    data[0:3] = anchor
    data[10] = 1.0
    eq.data = data


# =========================================================================== build
def build_spec(cfg: SimCfg | None = None) -> tuple[mujoco.MjSpec, Layout]:
    cfg = cfg or SimCfg()
    spec = mujoco.MjSpec.from_file(str(ASSETS / "panda.xml"))
    for key in list(spec.keys):
        spec.delete(key)
    spec.option.timestep = cfg.timestep
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.ambient = [0.30, 0.30, 0.32]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.stat.extent = 4.0
    for name, rgba, em in [("led_ok", [0.1, 1.0, 0.2, 1], 1.0), ("led_fault", [1.0, 0.45, 0.0, 1], 1.0),
                           ("led_identify", [0.2, 0.4, 1.0, 1], 1.0), ("led_off", [0.05, 0.05, 0.05, 1], 0.0),
                           ("led_rebuild", [0.8, 1.0, 0.1, 1], 0.8), ("led_red", [1.0, 0.06, 0.06, 1], 1.0), ("light_panel", [1, 1, 0.97, 1], 1.0),
                           ("accent_glow", [0.1, 0.65, 1.0, 1], 0.9)]:
        spec.add_material(name=name, rgba=rgba, emission=em)

    slots: dict[str, SlotDef] = {}
    servers: dict[str, ServerDef] = {}
    racks: dict[str, tuple[np.ndarray, float]] = {}
    _build_room(spec, cfg)
    _build_robot(spec, cfg, slots)
    by_rack = {(s.row, s.rack): s for s in cfg.servers}
    for row in (1, 2):
        for i in range(1, cfg.room.racks_per_row + 1):
            pos, yaw = rack_pose(cfg, row, i)
            racks[f"R{row}-{i:02d}"] = (pos, yaw)
            sp = by_rack.get((row, i))
            body = _build_rack(spec, cfg, row, i, pos, yaw, sp)
            if sp is not None:
                servers[sp.id] = _build_tray(spec, cfg, body, sp, len(servers), pos, yaw, slots)
    depot = depot_pose(cfg)
    _build_depot(spec, cfg, depot, slots)
    _build_technician(spec)
    components = _build_components(spec, cfg, slots, servers, depot)
    return spec, Layout(slots, components, servers, racks, depot)


def build_model(cfg: SimCfg | None = None) -> tuple[mujoco.MjModel, Layout, mujoco.MjSpec]:
    spec, layout = build_spec(cfg)
    return spec.compile(), layout, spec


# =========================================================================== room
def _build_room(spec, cfg: SimCfg) -> None:
    r = cfg.room
    wb = spec.worldbody
    cx, cy = sum(r.room_x) / 2, sum(r.room_y) / 2
    hx, hy = (r.room_x[1] - r.room_x[0]) / 2, (r.room_y[1] - r.room_y[0]) / 2
    wb.add_geom(name="floor", type=PLANE, size=[hx, hy, 0.1], pos=[cx, cy, 0], rgba=[0.62, 0.63, 0.65, 1])
    seam = [0.45, 0.46, 0.48, 1]
    for k in range(int(2 * hx / 0.6) + 1):  # raised-floor tile seams (0.6 m grid)
        wb.add_geom(name=f"seam_x{k}", type=BOX, size=[0.004, hy, 0.0005], pos=[r.room_x[0] + k * 0.6, cy, 0.0006],
                    rgba=seam, contype=0, conaffinity=0)
    for k in range(int(2 * hy / 0.6) + 1):
        wb.add_geom(name=f"seam_y{k}", type=BOX, size=[hx, 0.004, 0.0005], pos=[cx, r.room_y[0] + k * 0.6, 0.0006],
                    rgba=seam, contype=0, conaffinity=0)
    x0 = r.row_x0 - r.rack_pitch / 2
    for k in range(r.racks_per_row):  # perforated tiles down the cold aisle
        for j in (-1, 1):
            wb.add_geom(name=f"perf_{k}_{j}", type=BOX, size=[0.28, 0.28, 0.0008],
                        pos=[x0 + 0.3 + k * 0.61, j * 0.3, 0.0012], rgba=[0.38, 0.40, 0.43, 1], contype=0, conaffinity=0)
            for g in range(5):
                wb.add_geom(name=f"perf_{k}_{j}_slot{g}", type=BOX, size=[0.24, 0.006, 0.0005],
                            pos=[x0 + 0.3 + k * 0.61, j * 0.3 - 0.2 + g * 0.1, 0.0022], rgba=[0.22, 0.23, 0.25, 1],
                            contype=0, conaffinity=0)
    wall = [0.80, 0.81, 0.83, 1]
    wb.add_geom(name="wall_back", type=BOX, size=[0.05, hy, r.ceiling / 2], pos=[r.room_x[1], cy, r.ceiling / 2],
                rgba=wall, contype=0, conaffinity=0)
    wb.add_geom(name="wall_side", type=BOX, size=[hx, 0.05, r.ceiling / 2], pos=[cx, r.room_y[1], r.ceiling / 2],
                rgba=wall, contype=0, conaffinity=0)
    for k in range(5):
        for j, y in enumerate((0.0, 2.2, -2.2)):
            wb.add_geom(name=f"ceil_light_{k}_{j}", type=BOX, size=[0.6, 0.15, 0.01],
                        pos=[r.room_x[0] + 0.9 + k * 1.6, y, r.ceiling - 0.02], material="light_panel", contype=0, conaffinity=0)
    row_mid = r.row_x0 + (r.racks_per_row - 1) * r.rack_pitch / 2
    for side, y in (("r1", 1.40), ("r2", -1.40)):
        wb.add_geom(name=f"cable_tray_{side}", type=BOX, size=[(r.racks_per_row * r.rack_pitch) / 2 + 0.3, 0.15, 0.02],
                    pos=[row_mid, y, 2.35], rgba=[0.75, 0.62, 0.15, 1], contype=0, conaffinity=0)
        for k in range(4):
            wb.add_geom(name=f"cable_tray_{side}_hanger{k}", type=BOX, size=[0.01, 0.01, 0.31],
                        pos=[r.row_x0 - 0.2 + k * 1.2, y, 2.68], rgba=[0.3, 0.3, 0.32, 1], contype=0, conaffinity=0)
    end_x = r.row_x0 + (r.racks_per_row - 0.5) * r.rack_pitch + 0.9
    for j, y in enumerate((1.6, -1.6)):  # cooling units at the end of each row
        wb.add_geom(name=f"crac_{j}", type=BOX, size=[0.45, 0.9, 1.0], pos=[end_x + 0.6, y, 1.0],
                    rgba=[0.86, 0.87, 0.88, 1])
        for k in range(6):
            wb.add_geom(name=f"crac_{j}_grille{k}", type=BOX, size=[0.002, 0.7, 0.02],
                        pos=[end_x + 0.149, y, 1.3 + k * 0.08], rgba=[0.3, 0.3, 0.32, 1], contype=0, conaffinity=0)
        wb.add_geom(name=f"crac_{j}_panel", type=BOX, size=[0.002, 0.12, 0.06], pos=[end_x + 0.149, y, 0.8],
                    material="accent_glow", contype=0, conaffinity=0)
    wb.add_light(name="key", pos=[1.5, -1.5, 2.9], dir=[0, 0.3, -1], diffuse=[0.55, 0.55, 0.55], castshadow=True)
    wb.add_light(name="fill", pos=[-1.5, 1.5, 2.9], dir=[0.4, -0.3, -1], diffuse=[0.3, 0.3, 0.32], castshadow=False)
    wb.add_light(name="aisle", pos=[3.5, 0, 2.9], dir=[-0.3, 0, -1], diffuse=[0.25, 0.25, 0.27], castshadow=False)
    wb.add_camera(name="overview", pos=[-2.4, -2.3, 2.6], xyaxes=[0.6, -0.8, 0, 0.4, 0.3, 0.87], fovy=60)
    wb.add_camera(name="aisle_end", pos=[4.0, 0.0, 2.0], xyaxes=[0, 1, 0, -0.35, 0, 0.94], fovy=65)


# =========================================================================== racks
def _build_rack(spec, cfg: SimCfg, row: int, i: int, pos, yaw, serviceable):
    r = cfg.rack
    body = spec.worldbody.add_body(name=f"rack_R{row}_{i:02d}", pos=list(pos), quat=yaw_quat(yaw))
    p = f"R{row}{i:02d}_"
    h = r.plinth + r.units * U + 0.05
    post_y = r.opening / 2 + 0.02
    for nm, x in (("f", 0.02), ("r", r.depth - 0.02)):
        for side, sy in (("l", 1), ("r", -1)):
            _box(body, f"{p}post_{nm}{side}", [x, sy * post_y, h / 2], [0.02, 0.02, h / 2], FRAME)
    for side, sy in (("l", 1), ("r", -1)):
        _box(body, f"{p}side_{side}", [r.depth / 2, sy * (r.width / 2 - 0.005), h / 2], [r.depth / 2, 0.005, h / 2], PANEL)
    _box(body, f"{p}top", [r.depth / 2, 0, h - 0.01], [r.depth / 2, r.width / 2, 0.01], PANEL)
    _box(body, f"{p}back", [r.depth - 0.005, 0, h / 2], [0.005, r.width / 2, h / 2], PANEL)
    _box(body, f"{p}plinth", [r.depth / 2, 0, r.plinth / 2], [r.depth / 2, r.width / 2, r.plinth / 2], FRAME)
    # header: row-coloured ID bar + one dot per rack number
    _box(body, f"{p}header", [-0.002, 0, h - 0.06], [0.003, r.width / 2, 0.05], [0.16, 0.16, 0.18, 1], 0, 0)
    _box(body, f"{p}idbar", [-0.006, 0, h - 0.06], [0.002, 0.16, 0.018],
         [0.95, 0.75, 0.1, 1] if row == 1 else [0.2, 0.6, 0.95, 1], 0, 0)
    for k in range(i):
        _box(body, f"{p}iddot{k}", [-0.009, 0.13 - k * 0.05, h - 0.06], [0.002, 0.012, 0.012], [0.05, 0.05, 0.05, 1], 0, 0)
    gap = None
    if serviceable is not None:
        lo = cfg.u_bottom(serviceable.u) - 0.004
        gap = (lo, lo + cfg.tray.height + 0.008)
    rng = np.random.default_rng(row * 10 + i)
    z, k = r.plinth, 0
    top = r.plinth + r.units * U
    while z + U <= top:
        hu = int(rng.choice([1, 1, 2, 2, 2, 4]))
        z1 = z + hu * U
        if z1 > top:
            hu, z1 = 1, z + U
        if gap and not (z1 <= gap[0] or z >= gap[1]):
            z = gap[1] + 0.002
            continue
        zc = (z + z1) / 2
        if rng.random() < 0.12:
            _box(body, f"{p}blank{k}", [0.012, 0, zc], [0.004, r.opening / 2 + 0.015, hu * U / 2 - 0.001], [0.07, 0.07, 0.08, 1], 0, 0)
        else:
            _box(body, f"{p}srv{k}", [0.36, 0, zc], [0.33, 0.215, hu * U / 2 - 0.002], [0.17, 0.17, 0.19, 1], 0, 0)
            _box(body, f"{p}srv{k}_bezel", [0.028, 0, zc], [0.003, 0.215, hu * U / 2 - 0.003], [0.24, 0.24, 0.26, 1], 0, 0)
            _box(body, f"{p}srv{k}_led", [0.024, 0.18, zc], [0.002, 0.004, 0.004], material="led_ok", contype=0, conaffinity=0)
            if hu >= 2:
                _box(body, f"{p}srv{k}_stripe", [0.024, -0.05, zc], [0.002, 0.11, hu * U * 0.12], [0.36, 0.37, 0.4, 1], 0, 0)
        z, k = z1, k + 1
    return body


# =========================================================================== trays
def _build_tray(spec, cfg: SimCfg, rack_body, sp, index: int, rack_pos, rack_yaw, slots) -> ServerDef:
    t = cfg.tray
    pfx = f"s{index}_"
    bottom = cfg.u_bottom(sp.u)
    tray = rack_body.add_body(name=f"tray_{index}", pos=[0, 0, bottom])
    tray.add_joint(name=f"{pfx}slide", type=SLIDE, axis=[1, 0, 0], range=[-t.travel, 0.0], damping=60.0, armature=1.0)
    tray.add_geom(name=f"{pfx}mass", type=BOX, size=[0.05, 0.05, 0.01], pos=[t.depth / 2, 0, 0.01],
                  mass=6.0, contype=0, conaffinity=0, group=4)
    W, D, H = t.width / 2, t.depth, t.height

    def v(name, c, h, rgba=None, mat=None):
        return _box(tray, f"{pfx}{name}", c, h, rgba, 0, 0, material=mat)

    v("floor", [D / 2, 0, 0.002], [D / 2, W, 0.002], CHASSIS)
    v("board", [D / 2 + 0.02, 0, 0.004], [D / 2 - 0.04, W - 0.01, 0.002], BOARD)
    # tray walls: contype 4 / conaffinity 1 -> the robot (contype 1) hits them, parts (contype 2) don't
    for side, sy in (("l", 1), ("r", -1)):
        _box(tray, f"{pfx}wall_{side}", [D / 2, sy * (W - 0.002), H / 2], [D / 2, 0.002, H / 2], CHASSIS, 4, 1)
    _box(tray, f"{pfx}wall_back", [D - 0.002, 0, H / 2], [0.002, W, H / 2], CHASSIS, 4, 1)
    v("bezel", [0.004, 0, H / 2], [0.004, W, H / 2], [0.18, 0.18, 0.2, 1])
    v("status_led", [-0.001, 0.19, H - 0.012], [0.002, 0.005, 0.005], mat="led_ok")
    v("id_strip", [-0.001, -0.17, H - 0.012], [0.001, 0.03, 0.006], [0.95, 0.95, 0.95, 1])
    for i, y in enumerate((-0.15, -0.05, 0.05, 0.15)):
        v(f"drive_{i}", [0.04, y, H / 2], [0.035, 0.045, 0.03], [0.12, 0.12, 0.14, 1])
        v(f"drive_{i}_led", [0.004, y + 0.035, H / 2 + 0.022], [0.002, 0.003, 0.003], mat="led_ok")
    _box(tray, f"{pfx}handle", [t.handle_x, 0, t.handle_z], [t.handle_len / 2, t.handle_thick / 2, t.handle_height / 2],
         [0.85, 0.15, 0.1, 1], 1, 0)
    v("handle_base", [-0.002, 0, t.handle_z], [0.003, 0.03, t.handle_height / 2 + 0.006], [0.6, 0.1, 0.08, 1])
    for i, y in enumerate(t.cpu_ys):
        v(f"cpu_{i}", [t.cpu_x, y, t.board_z + 0.03], [0.045, 0.045, 0.03], HEATSINK)
        for k in range(6):
            v(f"cpu_{i}_fin{k}", [t.cpu_x, y - 0.0375 + k * 0.015, t.board_z + 0.065], [0.045, 0.002, 0.005], HEATSINK)
    v("psu", [0.55, -0.12, t.board_z + 0.035], [0.06, 0.08, 0.035], [0.2, 0.2, 0.22, 1])
    for i, y in enumerate(t.fan_ys):
        name = f"F{i + 1}"
        slots[f"{sp.id}:{name}"] = SlotDef(name, "fan", "tray", sp.id, np.array([t.fan_x, y, t.board_z + t.fan_size[2] / 2]))
        v(f"fan_bay_{name}", [t.fan_x, y, t.board_z + 0.001], [t.fan_size[0] / 2 + 0.004, t.fan_size[1] / 2 + 0.004, 0.001], SOCKET)
    for i, y in enumerate(t.dimm_ys):
        name = f"A{i + 1}"
        zc = t.board_z + t.socket_h + t.dimm_size[2] / 2
        slots[f"{sp.id}:{name}"] = SlotDef(name, "dimm", "tray", sp.id, np.array([t.dimm_x, y, zc]))
        v(f"socket_{name}", [t.dimm_x, y, t.board_z + t.socket_h / 2], [t.dimm_size[0] / 2 + 0.005, 0.006, t.socket_h / 2], SOCKET)
        v(f"led_{name}", [t.dimm_x + t.dimm_size[0] / 2 + 0.014, y, t.board_z + 0.003], [0.007, 0.007, 0.003], mat="led_off")
    eq = spec.add_equality(name=f"{pfx}grip_handle", type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_BODY,
                           name1="hand", name2=f"tray_{index}", active=False, solref=[0.02, 1.0])
    eq.data = np.zeros(len(eq.data))
    # rail latch: holds the tray where it is unless the gripper has its pull tab (set at runtime)
    lock = spec.add_equality(name=f"{pfx}rail_lock", type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                             name1=f"{pfx}slide", active=True, solref=[0.01, 1.0])
    lock.data = np.zeros(len(lock.data))
    return ServerDef(sp.id, index, sp.row, sp.rack, sp.u, f"tray_{index}", f"{pfx}slide", f"{pfx}grip_handle",
                     np.asarray(rack_pos, float), float(rack_yaw), bottom)


# =========================================================================== robot
def _build_robot(spec, cfg: SimCfg, slots) -> None:
    rc, t = cfg.robot, cfg.tray
    link0 = spec.body("link0")
    for name, jt, axis, rng in (("base_x", SLIDE, [1, 0, 0], None), ("base_y", SLIDE, [0, 1, 0], None),
                                ("base_yaw", HINGE, [0, 0, 1], None), ("lift", SLIDE, [0, 0, 1], list(rc.lift_range))):
        # explicit limits: joints under link0 otherwise inherit the Franka's +/-2.9 default range
        kw = dict(name=name, type=jt, axis=axis, armature=5.0, damping=0.0)
        if rng:
            kw["range"] = rng
            kw["limited"] = mujoco.mjtLimited.mjLIMITED_TRUE
        else:
            kw["limited"] = mujoco.mjtLimited.mjLIMITED_FALSE
        link0.add_joint(**kw)
    for name, kp, kv, fr in (("base_x", 8000, 1500, 3000), ("base_y", 8000, 1500, 3000),
                             ("base_yaw", 4000, 600, 1500), ("lift", 200000, 15000, 8000)):
        a = spec.add_actuator(name=f"act_{name}", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = kp
        a.biasprm[0:3] = [0, -kp, -kv]
        a.forcelimited = True
        a.forcerange = [-fr, fr]
    # lift carriage: deck, side pods (spares cassette right, returns bin left), inner lift column
    _box(link0, "carriage_deck", [0.0, 0, -0.025], [0.23, 0.50, 0.02], ROBOT_GREY)
    _box(link0, "carriage_trim", [0.234, 0, -0.025], [0.004, 0.50, 0.021], material="accent_glow", contype=0, conaffinity=0)
    _box(link0, "lift_inner", [-0.10, 0, -0.62], [0.06, 0.09, 0.60], [0.75, 0.76, 0.78, 1], 0, 0)
    for i, y in enumerate(rc.spare_dimm_ys):
        name = f"CD{i + 1}"
        slots[f"onboard:{name}"] = SlotDef(name, "dimm", "onboard", None, np.array([rc.spare_dimm_x, y, 0.008 + t.dimm_size[2] / 2]))
        _box(link0, f"holder_{name}", [rc.spare_dimm_x, y, 0.004], [t.dimm_size[0] / 2 + 0.005, 0.008, 0.004], SOCKET, 0, 0)
    for i, y in enumerate(rc.spare_fan_ys):
        name = f"CF{i + 1}"
        slots[f"onboard:{name}"] = SlotDef(name, "fan", "onboard", None, np.array([rc.spare_fan_x, y, t.fan_size[2] / 2 + 0.002]))
        _box(link0, f"pad_{name}", [rc.spare_fan_x, y, 0.001], [0.035, 0.045, 0.001], SOCKET, 0, 0)
    _box(link0, "spares_rail", [0.03, -0.505, 0.03], [0.20, 0.005, 0.03], ROBOT_WHITE)
    bx, by = rc.bin_center
    sx, sy, sz = rc.bin_size
    col = [0.18, 0.20, 0.23, 1]
    _box(link0, "bin_floor", [bx, by, 0.003], [sx / 2, sy / 2, 0.003], col)
    for nm, ctr, half in [("bin_xn", [bx - sx / 2, by, sz / 2], [0.004, sy / 2, sz / 2]),
                          ("bin_xp", [bx + sx / 2, by, sz / 2], [0.004, sy / 2, sz / 2]),
                          ("bin_yn", [bx, by - sy / 2, sz / 2], [sx / 2, 0.004, sz / 2]),
                          ("bin_yp", [bx, by + sy / 2, sz / 2], [sx / 2, 0.004, sz / 2])]:
        _box(link0, nm, ctr, half, col)
    _box(link0, "bin_label", [bx, by + sy / 2 + 0.005, sz * 0.6], [0.08, 0.001, 0.02], [0.95, 0.75, 0.1, 1], 0, 0)
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.004, 0, 0], rgba=[1, 0, 0, 0.5], group=5)
    hand.add_camera(name="wrist", pos=list(cfg.camera.wrist_pos), quat=[0, 0.70710678, -0.70710678, 0],
                    fovy=cfg.camera.wrist_fovy)
    # wheeled chassis (mecanum): kinematic body the sim keeps under the arm
    ch = spec.worldbody.add_body(name="chassis", mocap=True, pos=[rc.chassis_offset_x, 0, 0])
    cx, cy, cz = rc.chassis_size
    _box(ch, "chassis_body", [0, 0, cz / 2 + 0.05], [cx / 2, cy / 2, cz / 2 - 0.02], ROBOT_WHITE)
    _box(ch, "chassis_skirt", [0, 0, 0.06], [cx / 2 - 0.01, cy / 2 - 0.01, 0.035], ROBOT_GREY)
    _box(ch, "chassis_lightstrip", [0, 0, cz + 0.02], [cx / 2 + 0.002, cy / 2 + 0.002, 0.008], material="accent_glow",
         contype=0, conaffinity=0)
    _box(ch, "lift_sleeve", [0.02, 0, cz + 0.20], [0.08, 0.11, 0.18], ROBOT_WHITE, 0, 0)
    _box(ch, "lidar_puck_base", [cx / 2 - 0.05, 0, cz + 0.04], [0.035, 0.035, 0.01], ROBOT_GREY, 0, 0)
    ch.add_geom(name="lidar_puck", type=CYL, size=[0.03, 0.02, 0], pos=[cx / 2 - 0.05, 0, cz + 0.07],
                rgba=[0.05, 0.05, 0.05, 1], contype=0, conaffinity=0)
    _box(ch, "bumper_front", [cx / 2 + 0.01, 0, 0.09], [0.012, cy / 2, 0.03], [0.1, 0.1, 0.1, 1])
    _box(ch, "bumper_back", [-cx / 2 - 0.01, 0, 0.09], [0.012, cy / 2, 0.03], [0.1, 0.1, 0.1, 1])
    for i, (sx_, sy_) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        wx, wy = sx_ * (cx / 2 - 0.11), sy_ * (cy / 2 + 0.005)
        ch.add_geom(name=f"wheel_{i}", type=CYL, size=[0.065, 0.03, 0], pos=[wx, wy, 0.065],
                    quat=[0.7071, 0.7071, 0, 0], rgba=[0.08, 0.08, 0.09, 1], contype=0, conaffinity=0)
        for k in range(4):  # mecanum roller stripes
            a = k * np.pi / 4
            ch.add_geom(name=f"wheel_{i}_roller{k}", type=BOX, size=[0.06, 0.004, 0.012], pos=[wx, wy + sy_ * 0.032, 0.065],
                        quat=[float(np.cos(a / 2)), 0, 0, float(np.sin(a / 2))], rgba=[0.35, 0.36, 0.38, 1],
                        contype=0, conaffinity=0)
    # follows the robot's position (not its heading): behind and above it, looking along +X
    ch.add_camera(name="follow", mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM, pos=[-1.25, -0.55, 2.35],
                  xyaxes=[0.4, -0.917, 0, 0.71, 0.31, 0.63], fovy=62)
    # 2D lidar: a fan of rangefinder rays across the front 180 degrees, 20 cm above the floor
    for k in range(LIDAR_RAYS):
        a = -np.pi / 2 + np.pi * k / (LIDAR_RAYS - 1)
        qz = [np.cos(a / 2), 0, 0, np.sin(a / 2)]
        qy = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]  # turn the site's +Z (ray direction) to +X
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array(qz), np.array(qy))
        ch.add_site(name=f"lidar_{k:02d}", pos=[cx / 2 + 0.025, 0, LIDAR_Z], quat=list(q), size=[0.005, 0, 0], group=5)
        sens = spec.add_sensor(name=f"lidar_{k:02d}", type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
                               objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"lidar_{k:02d}", cutoff=LIDAR_RANGE)
        sens.intprm[0] = 1  # MuJoCo 4 rangefinder data spec: 1 = distance only


# =========================================================================== depot
def _build_depot(spec, cfg: SimCfg, depot, slots) -> None:
    pos, yaw = depot
    t = cfg.tray
    body = spec.worldbody.add_body(name="depot", pos=list(pos), quat=yaw_quat(yaw))
    z = DEPOT_SHELF_Z
    light = [0.85, 0.86, 0.88, 1]
    _box(body, "depot_cabinet", [0.30, 0, z / 2], [0.30, 0.55, z / 2], light)
    _box(body, "depot_shelf", [-0.10, 0, z - 0.01], [0.30, 0.50, 0.01], ROBOT_GREY)
    _box(body, "depot_back", [0.58, 0, z + 0.45], [0.02, 0.55, 0.45], light)
    _box(body, "depot_sign", [0.555, 0, z + 0.75], [0.004, 0.25, 0.06], material="accent_glow", contype=0, conaffinity=0)
    _box(body, "dock_pad", [-0.70 - cfg.robot.chassis_offset_x, 0, 0.002], [0.40, 0.33, 0.002], [0.95, 0.75, 0.1, 1], 0, 0)
    for i, y in enumerate((0.15, 0.10, 0.05, 0.0, -0.05, -0.10)):
        name = f"DD{i + 1}"
        slots[f"depot:{name}"] = SlotDef(name, "dimm", "depot", None, np.array([-0.22, y, z + 0.008 + t.dimm_size[2] / 2]))
        _box(body, f"depot_holder_{name}", [-0.22, y, z + 0.004], [t.dimm_size[0] / 2 + 0.005, 0.008, 0.004], SOCKET, 0, 0)
    for i, y in enumerate((0.12, 0.0, -0.12)):
        name = f"DF{i + 1}"
        slots[f"depot:{name}"] = SlotDef(name, "fan", "depot", None, np.array([-0.08, y, z + t.fan_size[2] / 2 + 0.002]))
        _box(body, f"depot_pad_{name}", [-0.08, y, z + 0.001], [0.035, 0.045, 0.001], SOCKET, 0, 0)


# =========================================================================== people
def _build_technician(spec) -> None:
    """A walking technician (moving obstacle), driven by the sim. Parked far away until used."""
    b = spec.worldbody.add_body(name="technician", mocap=True, pos=[0, 0, -10])
    skin, shirt, pants = [0.85, 0.68, 0.55, 1], [0.15, 0.35, 0.65, 1], [0.15, 0.15, 0.18, 1]
    for side, y in (("l", 0.09), ("r", -0.09)):
        b.add_geom(name=f"tech_leg_{side}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.06, 0.40, 0],
                   pos=[0, y, 0.46], rgba=pants, contype=0, conaffinity=0)
        b.add_geom(name=f"tech_arm_{side}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.045, 0.27, 0],
                   pos=[0, y * 2.3, 1.18], rgba=shirt, contype=0, conaffinity=0)
    b.add_geom(name="tech_torso", type=BOX, size=[0.12, 0.2, 0.28], pos=[0, 0, 1.22], rgba=shirt, contype=0, conaffinity=0)
    b.add_geom(name="tech_vest", type=BOX, size=[0.125, 0.205, 0.08], pos=[0, 0, 1.30], rgba=[0.95, 0.75, 0.1, 1],
               contype=0, conaffinity=0)
    b.add_geom(name="tech_head", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.11, 0, 0], pos=[0, 0, 1.66], rgba=skin,
               contype=0, conaffinity=0)
    b.add_geom(name="tech_helmet", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.115, 0, 0], pos=[0, 0, 1.70],
               rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0)


# =========================================================================== parts
def _build_components(spec, cfg: SimCfg, slots, servers, depot) -> dict[str, ComponentDef]:
    t = cfg.tray
    comps: dict[str, ComponentDef] = {}
    plan: list[str] = []
    for sid in servers:
        plan += [f"{sid}:A{i + 1}" for i in range(len(t.dimm_ys))] + [f"{sid}:F{i + 1}" for i in range(len(t.fan_ys))]
    plan += ["onboard:CD1", "onboard:CD2", "onboard:CF1", "depot:DD1", "depot:DD2", "depot:DD3", "depot:DF1", "depot:DF2"]
    dpos, dyaw = depot
    counters = {"dimm": 0, "fan": 0}
    for key in plan:
        s = slots[key]
        counters[s.kind] += 1
        cid = f"{s.kind}_{counters[s.kind]:02d}"
        if s.where == "tray":
            sv = servers[s.server]
            world = sv.rack_pos + rot_z(sv.rack_yaw) @ (np.array([0, 0, sv.tray_bottom]) + s.pos)
            quat = yaw_quat(sv.rack_yaw)
        elif s.where == "depot":
            world = dpos + rot_z(dyaw) @ s.pos
            quat = yaw_quat(dyaw)
        else:  # onboard: relative to link0 at its compile-time pose (world origin)
            world = s.pos.copy()
            quat = [1.0, 0, 0, 0]
        body = spec.worldbody.add_body(name=cid, pos=list(world), quat=quat)
        body.add_freejoint(name=f"{cid}_free")
        if s.kind == "dimm":
            sx, sy, sz = t.dimm_size
            body.add_geom(name=f"{cid}_pcb", type=BOX, size=[sx / 2, sy / 2, sz / 2], rgba=DIMM, mass=0.05,
                          contype=2, conaffinity=1, friction=[1.5, 0.02, 0.002], condim=4)
            for k in range(4):
                for side in (-1, 1):
                    body.add_geom(name=f"{cid}_chip{k}{'ab'[side > 0]}", type=BOX, size=[0.012, 0.0015, 0.008],
                                  pos=[-0.045 + k * 0.03, side * (sy / 2 + 0.0015), -0.004], rgba=DIMM_CHIP,
                                  contype=0, conaffinity=0, group=1)
            grasp_z = sz / 2 - 0.009
        else:
            fx, fy, fz = t.fan_size
            body.add_geom(name=f"{cid}_body", type=BOX, size=[fx / 2, fy / 2, fz / 2], rgba=FAN, mass=0.15,
                          contype=2, conaffinity=1, friction=[1.5, 0.02, 0.002], condim=4)
            body.add_geom(name=f"{cid}_rotor", type=CYL, size=[0.026, 0.002, 0], pos=[-fx / 2 - 0.001, 0, 0],
                          quat=[0.7071, 0, 0.7071, 0], rgba=[0.3, 0.3, 0.32, 1], contype=0, conaffinity=0, group=1)
            body.add_geom(name=f"{cid}_tab", type=BOX, size=[0.02, 0.006, 0.012], pos=[0, 0, fz / 2 + 0.012],
                          rgba=FAN_TAB, mass=0.01, contype=2, conaffinity=1, friction=[1.5, 0.02, 0.002], condim=4)
            body.add_geom(name=f"{cid}_led", type=BOX, size=[0.004, 0.004, 0.002], pos=[fx / 2 - 0.008, fy / 2 - 0.008, fz / 2 + 0.002],
                          material="led_ok", contype=0, conaffinity=0)
            grasp_z = fz / 2 + 0.012
        body.add_site(name=f"{cid}_grasp", pos=[0, 0, grasp_z], size=[0.003, 0, 0], rgba=[0, 0.5, 1, 0.4], group=5)
        # slot weld: always to the part's own server tray (only tray parts start active; the sim
        # re-targets runtime welds by rewriting their data when a part moves between servers)
        tray_body = servers[s.server].tray_body if s.where == "tray" else next(iter(servers.values())).tray_body
        _weld(spec, f"weld_slot_{cid}", cid, tray_body, s.where == "tray", s.pos if s.where == "tray" else np.zeros(3))
        _weld(spec, f"weld_hand_{cid}", cid, "hand", False, np.zeros(3))
        if s.where == "onboard":
            _weld(spec, f"weld_store_{cid}", cid, "link0", True, s.pos)
        elif s.where == "depot":
            _weld(spec, f"weld_store_{cid}", cid, "world", True, world)
        else:
            _weld(spec, f"weld_store_{cid}", cid, "link0", False, np.zeros(3))
        comps[cid] = ComponentDef(cid, s.kind, key)
    return comps
