"""Procedural aisle of 42U cabinets, each with a 12-bay storage server, filler servers,
and room dressing.

Each drive caddy is its own Xform (body + pull tab + LED) sitting at identity, so
physics.py can make it a rigid body on a slide joint whose position = displacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RACK_UNIT, SceneCfg
from .geometry import BoxBatch, MaterialLibrary, ensure_xform

# colors (linear RGB)
FRAME = (0.05, 0.05, 0.06)
PANEL = (0.09, 0.09, 0.10)
CHASSIS = (0.20, 0.21, 0.23)
BEZEL = (0.13, 0.13, 0.15)
BEZEL_LIGHT = (0.30, 0.31, 0.33)
CADDY = (0.10, 0.10, 0.11)
TAB = (0.05, 0.25, 0.65)  # blue latch, like many real caddies
BLANK = (0.06, 0.06, 0.07)
SWITCH = (0.12, 0.13, 0.14)
PORT = (0.01, 0.01, 0.01)
FLOOR = (0.55, 0.56, 0.58)
FLOOR_TILE = (0.45, 0.47, 0.50)

LED_STATES = ("ok", "fault", "identify", "off")


@dataclass
class BayInfo:
    cabinet: int
    bay: int
    row: int
    col: int
    caddy_path: str  # rigid-body Xform (identity pose when seated)
    body_path: str
    tab_path: str
    led_path: str
    tab_center: np.ndarray  # WORLD frame, center of the pull tab (grasp point)
    face_center: np.ndarray  # WORLD frame, center of the caddy's front face


def cabinet_root(cabinet: int) -> str:
    return f"/World/Cabinet{cabinet}"


def make_materials() -> MaterialLibrary:
    mats = MaterialLibrary()
    mats.add("led_ok", (0.02, 0.6, 0.05), emissive=(0.0, 4.0, 0.3))
    mats.add("led_fault", (0.9, 0.35, 0.0), emissive=(14.0, 3.0, 0.0))  # amber, bright
    mats.add("led_identify", (0.05, 0.2, 0.9), emissive=(0.2, 1.2, 8.0))
    mats.add("led_off", (0.03, 0.03, 0.03))
    mats.add("led_status", (0.02, 0.6, 0.05), emissive=(0.0, 2.0, 0.2))
    mats.add("metal_dark", FRAME, roughness=0.4, metallic=0.6)
    return mats


def build_room(cfg: SceneCfg, batch: BoxBatch) -> None:
    fz = cfg.floor_z
    batch.add("/World/Room/floor", (0.5, 0.0, fz - 0.05), (8.0, 8.0, 0.1), FLOOR)
    for i in range(-4, 4):
        for j in range(-5, 5):
            if (i + j) % 2 == 0:
                batch.add(f"/World/Room/tile_{i + 4}_{j + 5}", (i * 0.6 - 0.3, j * 0.6 + 0.3, fz + 0.0005),
                          (0.59, 0.59, 0.001), FLOOR_TILE, collide=False)


def build_end_rack(cfg: SceneCfg, batch: BoxBatch, name: str, y: float) -> None:
    """Closed-door rack at the ends of the aisle: one box plus status LEDs. Set dressing."""
    r = cfg.rack
    h = r.plinth + r.units * RACK_UNIT + 0.05
    cx = r.front_x + r.depth / 2
    batch.add(f"/World/{name}/body", (cx, y, cfg.floor_z + h / 2), (r.depth, r.outer_width, h), PANEL)
    rng = np.random.default_rng(sum(map(ord, name)))
    for k in range(10):
        z = cfg.floor_z + r.plinth + rng.uniform(0.1, 1.8)
        batch.add(f"/World/{name}/led_{k}", (r.front_x - 0.001, y + rng.uniform(-0.2, 0.2), z),
                  (0.002, 0.006, 0.006), (0.0, 0.6, 0.1), collide=False, material="led_status")


def build_rack(cfg: SceneCfg, batch: BoxBatch, cabinet: int) -> None:
    r = cfg.rack
    root = cabinet_root(cabinet)
    y0 = cfg.cabinet_y(cabinet)
    fz = cfg.floor_z
    height = r.plinth + r.units * RACK_UNIT + 0.05
    x0, x1 = r.front_x, r.front_x + r.depth
    cx = (x0 + x1) / 2
    zc = fz + height / 2
    post_y = r.opening_width / 2 + r.post / 2
    ensure_xform(root)

    for name, x in (("front", x0 + r.post / 2), ("rear", x1 - r.post / 2)):
        for side, sy in (("l", 1), ("r", -1)):
            batch.add(f"{root}/post_{name}_{side}", (x, y0 + sy * post_y, zc), (r.post, r.post, height),
                      FRAME, material="metal_dark")
    side_y = r.outer_width / 2 - 0.005
    for side, sy in (("l", 1), ("r", -1)):
        batch.add(f"{root}/side_{side}", (cx, y0 + sy * side_y, zc), (r.depth, 0.01, height), PANEL)
    batch.add(f"{root}/top", (cx, y0, fz + height - 0.01), (r.depth, r.outer_width, 0.02), PANEL)
    batch.add(f"{root}/plinth", (cx, y0, fz + r.plinth / 2), (r.depth, r.outer_width, r.plinth), FRAME)
    batch.add(f"{root}/back", (x1 - 0.005, y0, zc), (0.01, r.outer_width, height), PANEL)
    for u in range(1, r.units + 1, 5):
        z = cfg.u_bottom_z(u) + RACK_UNIT / 2
        batch.add(f"{root}/umark_{u}", (x0 - 0.0005, y0 + post_y, z), (0.001, 0.02, 0.003), BEZEL_LIGHT, collide=False)
    # cabinet number plate on the top of the frame (white = readable in the overview shot)
    batch.add(f"{root}/label", (x0 - 0.001, y0, fz + height - 0.06), (0.002, 0.12, 0.05), (0.85, 0.85, 0.85), collide=False)


# (bottom U, height, kind). Everything except U20-21 (the storage server).
_BASE_LAYOUT = [(42, 1, "switch"), (41, 1, "blank"), (39, 2, "server"), (37, 2, "server"),
                (36, 1, "blank"), (35, 1, "server"), (34, 1, "server"), (33, 1, "server"),
                (31, 2, "server"), (29, 2, "blank"), (27, 2, "server"), (25, 2, "server"),
                (24, 1, "server"), (22, 2, "server"), (18, 2, "server"), (17, 1, "blank"),
                (15, 2, "server"), (13, 2, "server"), (12, 1, "blank"), (10, 2, "server"),
                (8, 2, "server"), (5, 3, "blank"), (3, 2, "server"), (1, 2, "blank")]


def build_filler_units(cfg: SceneCfg, batch: BoxBatch, cabinet: int) -> None:
    """Generic servers, blanking panels and a top-of-rack switch around the storage server."""
    s, r = cfg.server, cfg.rack
    root = cabinet_root(cabinet)
    y0 = cfg.cabinet_y(cabinet)
    target = set(range(s.u_position, s.u_position + s.u_height))
    rng = np.random.default_rng(cabinet + 7)
    x_face = r.front_x + s.recess
    for u, h, kind in _BASE_LAYOUT:
        assert not (set(range(u, u + h)) & target), f"filler at U{u} overlaps the storage server"
        if kind == "server" and rng.random() < 0.2:  # vary cabinets a little
            kind = "blank"
        z0 = cfg.u_bottom_z(u)
        hz = h * RACK_UNIT - 0.002
        zc = z0 + h * RACK_UNIT / 2
        base = f"{root}/U{u:02d}_{kind}"
        if kind == "blank":
            batch.add(base, (r.front_x + 0.003, y0, zc), (0.004, r.opening_width + 0.03, hz), BLANK)
            continue
        depth = 0.30 if kind == "switch" else s.depth
        batch.add(f"{base}/body", (x_face + depth / 2, y0, zc), (depth, s.width, hz), CHASSIS)
        batch.add(f"{base}/bezel", (x_face - 0.003, y0, zc), (0.006, s.width, hz),
                  SWITCH if kind == "switch" else BEZEL, collide=False)
        if kind == "switch":
            for k in range(24):
                row, col = divmod(k, 12)
                y = y0 - 0.17 + col * 0.026
                zz = zc + (row - 0.5) * 0.016
                batch.add(f"{base}/port_{k}", (x_face - 0.0065, y, zz), (0.002, 0.014, 0.011), PORT, collide=False)
                if k % 3 == 0:
                    batch.add(f"{base}/portled_{k}", (x_face - 0.0068, y + 0.009, zz + 0.007),
                              (0.002, 0.003, 0.002), (0, 0.6, 0.1), collide=False, material="led_status")
        else:
            batch.add(f"{base}/stripe", (x_face - 0.0065, y0 - 0.05, zc), (0.002, 0.24, hz * 0.35), BEZEL_LIGHT, collide=False)
            batch.add(f"{base}/led", (x_face - 0.0068, y0 + 0.19, zc), (0.002, 0.005, 0.005), (0, 0.6, 0.1),
                      collide=False, material="led_status")


def build_storage_server(cfg: SceneCfg, batch: BoxBatch, cabinet: int) -> list[BayInfo]:
    """The 2U, 12-bay server. The chassis is a hollow shell so caddies can slide out."""
    s, c, r = cfg.server, cfg.caddy, cfg.rack
    y0 = cfg.cabinet_y(cabinet)
    base = f"{cabinet_root(cabinet)}/StorageServer"
    ensure_xform(base)
    ensure_xform(f"{base}/Shell")
    ensure_xform(f"{base}/Caddies")
    shell = f"{base}/Shell"
    wall = 0.003
    h = s.u_height * RACK_UNIT - 0.002
    z0 = cfg.u_bottom_z(s.u_position) + 0.001
    zc = z0 + h / 2
    x_face = r.front_x + s.recess
    xc = x_face + s.depth / 2

    batch.add(f"{shell}/top", (xc, y0, z0 + h - wall / 2), (s.depth, s.width, wall), CHASSIS)
    batch.add(f"{shell}/bottom", (xc, y0, z0 + wall / 2), (s.depth, s.width, wall), CHASSIS)
    for side, sy in (("l", 1), ("r", -1)):
        batch.add(f"{shell}/side_{side}", (xc, y0 + sy * (s.width / 2 - wall / 2), zc), (s.depth, wall, h), CHASSIS)
        batch.add(f"{shell}/ear_{side}", (x_face - 0.002, y0 + sy * (s.width / 2 + 0.008), zc), (0.004, 0.016, h), BEZEL)
    backplane_x = x_face + c.depth + 0.004
    batch.add(f"{shell}/backplane", (backplane_x + 0.002, y0, zc), (0.004, s.width - 2 * wall, h - 2 * wall), (0.05, 0.25, 0.1))
    internals_d = s.depth - (backplane_x - x_face) - 0.01
    batch.add(f"{shell}/internals", (backplane_x + 0.01 + internals_d / 2, y0, zc),
              (internals_d, s.width - 2 * wall, h - 2 * wall), CHASSIS)

    inner_w = s.width - 2 * wall
    inner_h = h - 2 * wall
    pitch_y = inner_w / s.bay_cols
    pitch_z = inner_h / s.bay_rows
    for k in range(1, s.bay_cols):
        y = y0 + inner_w / 2 - k * pitch_y
        batch.add(f"{shell}/sep_{k}", (x_face + c.depth / 2, y, zc), (c.depth, pitch_y - c.face_width - 0.003, inner_h), CHASSIS)

    bays = []
    for row in range(s.bay_rows):  # row 0 = top
        for col in range(s.bay_cols):  # col 0 = robot's left (+Y)
            bay = row * s.bay_cols + col
            y = y0 + inner_w / 2 - (col + 0.5) * pitch_y
            z = z0 + wall + inner_h - (row + 0.5) * pitch_z
            p = f"{base}/Caddies/bay{bay:02d}"
            ensure_xform(p)
            body = batch.add(f"{p}/body", (x_face + c.depth / 2, y, z), (c.depth, c.face_width, c.face_height), CADDY)
            tab_c = np.array([x_face - c.tab_depth / 2, y - 0.012, z])
            tab = batch.add(f"{p}/tab", tuple(tab_c), (c.tab_depth, c.tab_width, c.tab_height), TAB)
            led = batch.add(f"{p}/led", (x_face - 0.0015, y + c.face_width / 2 - 0.010, z + c.face_height / 2 - 0.006),
                            (0.003, c.led_size, c.led_size), (0.0, 0.6, 0.1), collide=False, material="led_ok")
            bays.append(BayInfo(cabinet, bay, row, col, p, body, tab, led, tab_c, np.array([x_face, y, z])))
    return bays
