"""Build the MuJoCo model programmatically (MjSpec) around Menagerie's Franka Panda.

Collision layers (contype / conaffinity):
  robot + static scenery : 1 / 1
  tray visuals           : 0 / 0   (the tray is a kinematic-ish drawer on a slide joint)
  tray handle            : 1 / 0   (only the gripper touches it)
  components (RAM, fans) : 2 / 1   (touch the gripper, cart, bin, floor; not the tray or
                                     each other, so packed parts never jam in physics)

Every component is a free body with three weld constraints, toggled at runtime:
  weld_slot_<id> : seated in a tray slot      (body2 = tray)
  weld_hand_<id> : held by the gripper         (body2 = hand)
  weld_cart_<id> : parked on the cart shelf    (body2 = world)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .config import CART_HEIGHT, U, SimCfg

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "franka_emika_panda"

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
PLANE = mujoco.mjtGeom.mjGEOM_PLANE

# colors
FRAME = [0.06, 0.06, 0.07, 1]
PANEL = [0.10, 0.10, 0.11, 1]
CHASSIS = [0.55, 0.57, 0.60, 1]
BOARD = [0.05, 0.30, 0.15, 1]
DIMM = [0.08, 0.20, 0.10, 1]
DIMM_CHIP = [0.03, 0.03, 0.03, 1]
SOCKET = [0.02, 0.02, 0.02, 1]
FAN = [0.12, 0.12, 0.13, 1]
FAN_TAB = [0.10, 0.30, 0.75, 1]
HEATSINK = [0.70, 0.72, 0.75, 1]
CART = [0.80, 0.45, 0.05, 1]
CART_DARK = [0.10, 0.10, 0.11, 1]
FLOOR = [0.52, 0.53, 0.55, 1]


@dataclass
class SlotDef:
    name: str  # "A1".."A8" (RAM), "F1".."F5" (fans), "CD1".. / "CF1".. (cart spares)
    kind: str  # "dimm" | "fan"
    where: str  # "tray" | "cart"
    pos: np.ndarray  # in tray frame (tray slots) or world frame (cart slots)


@dataclass
class ComponentDef:
    id: str  # body name, e.g. "dimm_03" / "fan_02"
    kind: str
    slot: str  # initial slot
    spare: bool


@dataclass
class Layout:
    slots: dict[str, SlotDef]
    components: dict[str, ComponentDef]


def _mat(spec: mujoco.MjSpec, name: str, rgba, emission: float = 0.0) -> None:
    spec.add_material(name=name, rgba=rgba, emission=emission)


def _box(body, name, center, half, rgba, contype=1, conaffinity=1, material=None, group=0):
    kw = dict(name=name, type=BOX, size=list(half), pos=list(center),
              contype=contype, conaffinity=conaffinity, group=group)
    if material:
        kw["material"] = material  # no rgba: an explicit geom rgba would override the material
    else:
        kw["rgba"] = list(rgba)
    return body.add_geom(**kw)


def build_spec(cfg: SimCfg | None = None) -> tuple[mujoco.MjSpec, Layout]:
    cfg = cfg or SimCfg()
    spec = mujoco.MjSpec.from_file(str(ASSETS / "panda.xml"))
    for key in list(spec.keys):
        spec.delete(key)
    spec.option.timestep = cfg.timestep
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]

    for name, rgba, em in [("led_ok", [0.1, 1.0, 0.2, 1], 1.0), ("led_fault", [1.0, 0.45, 0.0, 1], 1.0),
                           ("led_identify", [0.2, 0.4, 1.0, 1], 1.0), ("led_off", [0.05, 0.05, 0.05, 1], 0.0),
                           ("led_rebuild", [0.8, 1.0, 0.1, 1], 0.8)]:
        _mat(spec, name, rgba, em)

    wb = spec.worldbody
    wb.add_light(name="sun", pos=[0.5, 0, 3.0], dir=[-0.2, 0.1, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    wb.add_light(name="fill", pos=[-1.5, -1.0, 2.0], dir=[0.6, 0.4, -1], diffuse=[0.3, 0.3, 0.3], castshadow=False)
    fz = -CART_HEIGHT
    wb.add_geom(name="floor", type=PLANE, size=[6, 6, 0.1], pos=[0, 0, fz], rgba=FLOOR)

    # TCP site between the fingertips (panda hand z points along the fingers)
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.004, 0, 0], rgba=[1, 0, 0, 0.5], group=5)
    # wrist camera: looks along hand +Z (MuJoCo cameras look down their own -Z)
    # image up = toward the rack (+X) when the gripper points down with fingers along Y
    hand.add_camera(name="wrist", pos=list(cfg.camera.wrist_pos), quat=[0, 0.70710678, -0.70710678, 0],
                    fovy=cfg.camera.wrist_fovy)
    wb.add_camera(name="overview", pos=[-0.9, -1.3, 0.9], xyaxes=[0.82, -0.57, 0, 0.25, 0.36, 0.9], fovy=55)
    wb.add_camera(name="tray_top", pos=[0.50, 0, 1.30], xyaxes=[0, -1, 0, 1, 0, 0], fovy=35)

    _build_rack(wb, cfg)
    _build_cart(wb, cfg)
    slots: dict[str, SlotDef] = {}
    tray = _build_tray(spec, wb, cfg, slots)
    _build_cart_slots(wb, cfg, slots)
    components = _build_components(spec, wb, cfg, slots, tray)
    return spec, Layout(slots, components)


# ------------------------------------------------------------------------ scenery
def _build_rack(wb, cfg: SimCfg) -> None:
    r, t = cfg.rack, cfg.tray
    fz = -CART_HEIGHT
    h = r.plinth + r.units * U + 0.05
    x0, x1 = r.front_x, r.front_x + r.depth
    cx, zc = (x0 + x1) / 2, fz + h / 2
    rack = wb.add_body(name="rack")
    post_y = r.opening / 2 + 0.02
    for nm, x in (("f", x0 + 0.02), ("r", x1 - 0.02)):
        for side, sy in (("l", 1), ("r", -1)):
            _box(rack, f"post_{nm}{side}", [x, sy * post_y, zc], [0.02, 0.02, h / 2], FRAME)
    for side, sy in (("l", 1), ("r", -1)):
        _box(rack, f"side_{side}", [cx, sy * (r.width / 2 - 0.005), zc], [r.depth / 2, 0.005, h / 2], PANEL)
    _box(rack, "top", [cx, 0, fz + h - 0.01], [r.depth / 2, r.width / 2, 0.01], PANEL)
    _box(rack, "back", [x1 - 0.005, 0, zc], [0.005, r.width / 2, h / 2], PANEL)
    # filler servers above and below the serviceable tray's slot (visual)
    slot_lo, slot_hi = t.bottom_z - 0.004, t.bottom_z + t.height + 0.004
    z = fz + r.plinth
    k = 0
    rng = np.random.default_rng(3)
    while z + U <= fz + r.plinth + r.units * U:
        hu = int(rng.choice([1, 2, 2, 1]))
        z1 = z + hu * U
        if not (z1 <= slot_lo or z >= slot_hi):
            z = slot_hi + 0.002
            continue
        if z1 > fz + r.plinth + r.units * U:
            break
        zc_s = (z + z1) / 2
        _box(rack, f"srv_{k}", [x0 + 0.35, 0, zc_s], [0.33, 0.215, hu * U / 2 - 0.002], [0.16, 0.16, 0.18, 1], 0, 0)
        _box(rack, f"srv_{k}_bezel", [x0 + 0.017, 0, zc_s], [0.003, 0.215, hu * U / 2 - 0.004], [0.22, 0.22, 0.24, 1], 0, 0)
        _box(rack, f"srv_{k}_led", [x0 + 0.013, 0.18, zc_s], [0.002, 0.004, 0.004], [0, 1, 0.2, 1], 0, 0, material="led_ok")
        z, k = z1, k + 1
    # rails for the tray
    for side, sy in (("l", 1), ("r", -1)):
        _box(rack, f"rail_{side}", [x0 + 0.5, sy * (t.width / 2 + 0.008), t.bottom_z + 0.02],
             [0.48, 0.006, 0.012], [0.4, 0.4, 0.42, 1], 0, 0)


def _build_cart(wb, cfg: SimCfg) -> None:
    c = cfg.cart
    fz = -CART_HEIGHT
    cart = wb.add_body(name="cart")
    dx = (c.deck_x[1] - c.deck_x[0]) / 2
    cxd = (c.deck_x[1] + c.deck_x[0]) / 2
    _box(cart, "cart_deck", [cxd, 0, -0.015], [dx, c.deck_halfwidth, 0.015], CART_DARK)
    _box(cart, "cart_body", [cxd, 0, fz / 2 + 0.05], [dx - 0.02, c.deck_halfwidth - 0.02, -fz / 2 - 0.08], CART)
    wx = (c.wing_x[1] - c.wing_x[0]) / 2
    cxw = (c.wing_x[1] + c.wing_x[0]) / 2
    for side, sy in (("r", -1), ("l", 1)):
        yc = sy * (c.deck_halfwidth + c.wing_halfwidth) / 2
        hw = (c.wing_halfwidth - c.deck_halfwidth) / 2
        _box(cart, f"wing_{side}", [cxw, yc, -0.015], [wx, hw, 0.015], CART_DARK)
        _box(cart, f"wing_{side}_leg", [cxw, yc, fz / 2], [0.02, 0.02, -fz / 2 - 0.02], CART_DARK)
    for i, (x, y) in enumerate([(c.deck_x[0] + 0.05, 0.25), (c.deck_x[0] + 0.05, -0.25), (c.deck_x[1] - 0.05, 0.25),
                                (c.deck_x[1] - 0.05, -0.25)]):
        cart.add_geom(name=f"wheel_{i}", type=CYL, size=[0.05, 0.02, 0], pos=[x, y, fz + 0.05],
                      quat=[0.7071, 0.7071, 0, 0], rgba=CART_DARK)
    _box(cart, "cart_handle", [c.deck_x[0] - 0.03, 0, 0.30], [0.015, 0.25, 0.015], CART_DARK)
    for side, sy in (("l", 1), ("r", -1)):
        _box(cart, f"cart_handle_post_{side}", [c.deck_x[0] - 0.03, sy * 0.25, 0.15], [0.015, 0.015, 0.15], CART_DARK)
    # used-parts bin (left wing)
    bx, by = c.bin_center
    sx, sy_, sz = c.bin_size
    binb = wb.add_body(name="bin")
    col = [0.15, 0.35, 0.65, 1]
    _box(binb, "bin_floor", [bx, by, 0.005], [sx / 2, sy_ / 2, 0.005], col)
    for nm, ctr, half in [("bin_xn", [bx - sx / 2, by, sz / 2], [0.005, sy_ / 2, sz / 2]),
                          ("bin_xp", [bx + sx / 2, by, sz / 2], [0.005, sy_ / 2, sz / 2]),
                          ("bin_yn", [bx, by - sy_ / 2, sz / 2], [sx / 2, 0.005, sz / 2]),
                          ("bin_yp", [bx, by + sy_ / 2, sz / 2], [sx / 2, 0.005, sz / 2])]:
        _box(binb, nm, ctr, half, col)
    _box(binb, "bin_label", [bx - sx / 2 - 0.006, by, sz * 0.6], [0.001, 0.08, 0.025], [0.95, 0.8, 0.1, 1], 0, 0)


# --------------------------------------------------------------------------- tray
def _build_tray(spec, wb, cfg: SimCfg, slots: dict[str, SlotDef]):
    t = cfg.tray
    tray = wb.add_body(name="tray", pos=[cfg.rack.front_x, 0, t.bottom_z])
    tray.add_joint(name="tray_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0],
                   range=[-t.travel, 0.0], damping=60.0, armature=1.0)
    tray.add_geom(name="tray_mass", type=BOX, size=[0.05, 0.05, 0.01], pos=[t.depth / 2, 0, 0.01],
                  mass=6.0, contype=0, conaffinity=0, group=4)
    W, D, H = t.width / 2, t.depth, t.height

    def v(name, c, h, rgba, mat=None):  # visual-only tray geom
        return _box(tray, f"tray_{name}", c, h, rgba, 0, 0, material=mat)

    v("floor", [D / 2, 0, 0.002], [D / 2, W, 0.002], CHASSIS)
    v("board", [D / 2 + 0.02, 0, 0.004], [D / 2 - 0.04, W - 0.01, 0.002], BOARD)
    for side, sy in (("l", 1), ("r", -1)):
        v(f"wall_{side}", [D / 2, sy * (W - 0.002), H / 2], [D / 2, 0.002, H / 2], CHASSIS)
    v("wall_back", [D - 0.002, 0, H / 2], [0.002, W, H / 2], CHASSIS)
    v("bezel", [0.004, 0, H / 2], [0.004, W, H / 2], [0.18, 0.18, 0.2, 1])
    v("status_led", [-0.001, 0.19, H - 0.012], [0.002, 0.005, 0.005], [0, 1, 0.2, 1], "led_ok")
    # front drive bay (decorative)
    for i, y in enumerate((-0.15, -0.05, 0.05, 0.15)):
        v(f"drive_{i}", [0.04, y, H / 2], [0.035, 0.045, 0.03], [0.12, 0.12, 0.14, 1])
        v(f"drive_{i}_led", [0.004, y + 0.035, H / 2 + 0.022], [0.002, 0.003, 0.003], [0, 1, 0.2, 1], "led_ok")
    # handle (gripped top-down, fingers along X); only the gripper collides with it
    hx, hz = t.handle_x, t.handle_z
    _box(tray, "tray_handle", [hx, 0, hz], [t.handle_thick / 2, t.handle_len / 2, t.handle_thick / 2],
         [0.85, 0.15, 0.1, 1], 1, 0)
    for side, sy in (("l", 1), ("r", -1)):
        v(f"handle_arm_{side}", [hx / 2, sy * (t.handle_len / 2 - 0.007), hz], [-hx / 2, 0.007, 0.007], [0.85, 0.15, 0.1, 1])
    # CPUs + PSU (static)
    for i, y in enumerate(t.cpu_ys):
        v(f"cpu_{i}", [t.cpu_x, y, t.board_z + 0.03], [0.045, 0.045, 0.03], HEATSINK)
        for k in range(6):
            v(f"cpu_{i}_fin{k}", [t.cpu_x, y - 0.0375 + k * 0.015, t.board_z + 0.065], [0.045, 0.002, 0.005], HEATSINK)
    v("psu", [0.55, -0.12, t.board_z + 0.035], [0.06, 0.08, 0.035], [0.2, 0.2, 0.22, 1])
    # fan wall slots
    for i, y in enumerate(t.fan_ys):
        name = f"F{i + 1}"
        slots[name] = SlotDef(name, "fan", "tray", np.array([t.fan_x, y, t.board_z + t.fan_size[2] / 2]))
        v(f"fan_bay_{name}", [t.fan_x, y, t.board_z + 0.001], [t.fan_size[0] / 2 + 0.004, t.fan_size[1] / 2 + 0.004, 0.001],
          SOCKET)
    # RAM sockets + per-slot fault LEDs on the board ("light path diagnostics")
    for i, y in enumerate(t.dimm_ys):
        name = f"A{i + 1}"
        zc = t.board_z + t.socket_h + t.dimm_size[2] / 2
        slots[name] = SlotDef(name, "dimm", "tray", np.array([t.dimm_x, y, zc]))
        v(f"socket_{name}", [t.dimm_x, y, t.board_z + t.socket_h / 2], [t.dimm_size[0] / 2 + 0.005, 0.006, t.socket_h / 2], SOCKET)
        v(f"led_{name}", [t.dimm_x + t.dimm_size[0] / 2 + 0.014, y, t.board_z + 0.003], [0.007, 0.007, 0.003],
          [0, 1, 0.2, 1], "led_off")
    return tray


def _build_cart_slots(wb, cfg: SimCfg, slots: dict[str, SlotDef]) -> None:
    c, t = cfg.cart, cfg.tray
    holder = wb.add_body(name="spares_shelf")
    for i, y in enumerate(c.spare_dimm_ys):
        name = f"CD{i + 1}"
        zc = 0.008 + t.dimm_size[2] / 2
        slots[name] = SlotDef(name, "dimm", "cart", np.array([c.spare_dimm_x, y, zc]))
        _box(holder, f"holder_{name}", [c.spare_dimm_x, y, 0.004], [t.dimm_size[0] / 2 + 0.005, 0.008, 0.004], SOCKET, 0, 0)
    for i, y in enumerate(c.spare_fan_ys):
        name = f"CF{i + 1}"
        slots[name] = SlotDef(name, "fan", "cart", np.array([c.spare_fan_x, y, t.fan_size[2] / 2 + 0.002]))
        _box(holder, f"pad_{name}", [c.spare_fan_x, y, 0.001], [0.035, 0.045, 0.001], SOCKET, 0, 0)
    _box(holder, "shelf_label", [c.wing_x[0] + 0.01, -0.47, 0.03], [0.001, 0.09, 0.025], [0.1, 0.7, 0.3, 1], 0, 0)


# --------------------------------------------------------------------- components
def _build_components(spec, wb, cfg: SimCfg, slots: dict[str, SlotDef], tray) -> dict[str, ComponentDef]:
    t = cfg.tray
    comps: dict[str, ComponentDef] = {}
    plan = [(f"A{i + 1}", False) for i in range(len(t.dimm_ys))] + [("CD1", True), ("CD2", True)]
    plan += [(f"F{i + 1}", False) for i in range(len(t.fan_ys))] + [("CF1", True)]
    counters = {"dimm": 0, "fan": 0}
    tray_origin = np.array([cfg.rack.front_x, 0, t.bottom_z])
    for slot_name, spare in plan:
        s = slots[slot_name]
        counters[s.kind] += 1
        cid = f"{s.kind}_{counters[s.kind]:02d}"
        world = s.pos + (tray_origin if s.where == "tray" else 0)
        body = wb.add_body(name=cid, pos=list(world))
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
            fx, fy, fz_ = t.fan_size
            body.add_geom(name=f"{cid}_body", type=BOX, size=[fx / 2, fy / 2, fz_ / 2], rgba=FAN, mass=0.15,
                          contype=2, conaffinity=1, friction=[1.5, 0.02, 0.002], condim=4)
            body.add_geom(name=f"{cid}_rotor", type=CYL, size=[0.026, 0.002, 0], pos=[-fx / 2 - 0.001, 0, 0],
                          quat=[0.7071, 0, 0.7071, 0], rgba=[0.3, 0.3, 0.32, 1], contype=0, conaffinity=0, group=1)
            body.add_geom(name=f"{cid}_tab", type=BOX, size=[0.02, 0.006, 0.012], pos=[0, 0, fz_ / 2 + 0.012],
                          rgba=FAN_TAB, mass=0.01, contype=2, conaffinity=1, friction=[1.5, 0.02, 0.002], condim=4)
            body.add_geom(name=f"{cid}_led", type=BOX, size=[0.004, 0.004, 0.002], pos=[fx / 2 - 0.008, fy / 2 - 0.008, fz_ / 2 + 0.002],
                          material="led_ok", contype=0, conaffinity=0)
            grasp_z = fz_ / 2 + 0.012
        body.add_site(name=f"{cid}_grasp", pos=[0, 0, grasp_z], size=[0.003, 0, 0], rgba=[0, 0.5, 1, 0.4], group=5)
        for kind, other in (("slot", "tray"), ("hand", "hand"), ("cart", "world")):
            active = (kind == "slot" and s.where == "tray") or (kind == "cart" and s.where == "cart")
            eq = spec.add_equality(name=f"weld_{kind}_{cid}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                   name1=cid, name2=other, active=active, solref=[0.01, 1.0])
            # MjSpec's default eq data is laid out for joint polynomials (anchor would be 0,1,0).
            # Weld layout: anchor(3), relpose pos(3), relpose quat(4), torquescale(1).
            # A zero quat tells the compiler to take the relative pose from the initial placement.
            # The anchor is where the weld acts, in body2's frame: put it at the part itself
            # (slot position), otherwise small rotation errors become a visible sag.
            data = np.zeros(len(eq.data))
            if kind == "slot" or kind == "cart":
                data[0:3] = s.pos if (kind == "slot") == (s.where == "tray") else 0.0
            data[10] = 1.0
            eq.data = data
        comps[cid] = ComponentDef(cid, s.kind, slot_name, spare)
    return comps


def build_model(cfg: SimCfg | None = None) -> tuple[mujoco.MjModel, Layout, mujoco.MjSpec]:
    spec, layout = build_spec(cfg)
    model = spec.compile()
    return model, layout, spec
