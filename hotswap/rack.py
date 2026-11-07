"""Procedural 42U rack, the 12-bay storage server, filler servers, and room dressing.

Phase 1: everything is static geometry. The caddies are separate prims (body, pull
tab, LED) so phase 2 can turn each caddy into a rigid body on a prismatic joint
without changing the layout.
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
    row: int
    col: int
    body_path: str
    tab_path: str
    led_path: str
    tab_center: np.ndarray  # base frame, center of the pull tab (grasp point)
    face_center: np.ndarray  # base frame, center of the caddy's front face


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
    # raised-floor tiles: a grid of slightly darker tiles in the aisle (visual only)
    for i in range(-4, 4):
        for j in range(-4, 4):
            if (i + j) % 2 == 0:
                batch.add(
                    f"/World/Room/tile_{i + 4}_{j + 4}",
                    (i * 0.6 + 0.3 - 0.6, j * 0.6 + 0.3, fz + 0.0005),
                    (0.59, 0.59, 0.001),
                    FLOOR_TILE,
                    collide=False,
                )


def build_neighbor_rack(cfg: SceneCfg, batch: BoxBatch, name: str, y: float) -> None:
    """Closed-door neighbour rack: one box plus a few status LEDs. Pure set dressing."""
    r = cfg.rack
    h = r.plinth + r.units * RACK_UNIT + 0.05
    cx = r.front_x + r.depth / 2
    batch.add(f"/World/{name}/body", (cx, y, cfg.floor_z + h / 2), (r.depth, r.outer_width, h), PANEL)
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    for k in range(10):
        z = cfg.floor_z + r.plinth + rng.uniform(0.1, 1.8)
        batch.add(
            f"/World/{name}/led_{k}",
            (r.front_x - 0.001, y + rng.uniform(-0.2, 0.2), z),
            (0.002, 0.006, 0.006),
            (0.0, 0.6, 0.1),
            collide=False,
            material="led_status",
        )


def build_rack(cfg: SceneCfg, batch: BoxBatch) -> None:
    r = cfg.rack
    fz = cfg.floor_z
    height = r.plinth + r.units * RACK_UNIT + 0.05
    x0, x1 = r.front_x, r.front_x + r.depth
    cx = (x0 + x1) / 2
    zc = fz + height / 2
    post_y = r.opening_width / 2 + r.post / 2
    ensure_xform("/World/Rack")

    # 4 posts
    for name, x in (("front", x0 + r.post / 2), ("rear", x1 - r.post / 2)):
        for side, sy in (("l", 1), ("r", -1)):
            batch.add(f"/World/Rack/post_{name}_{side}", (x, sy * post_y, zc), (r.post, r.post, height), FRAME, material="metal_dark")
    # side panels, top, plinth, back
    side_y = r.outer_width / 2 - 0.005
    for side, sy in (("l", 1), ("r", -1)):
        batch.add(f"/World/Rack/side_{side}", (cx, sy * side_y, zc), (r.depth, 0.01, height), PANEL)
    batch.add("/World/Rack/top", (cx, 0, fz + height - 0.01), (r.depth, r.outer_width, 0.02), PANEL)
    batch.add("/World/Rack/plinth", (cx, 0, fz + r.plinth / 2), (r.depth, r.outer_width, r.plinth), FRAME)
    batch.add("/World/Rack/back", (x1 - 0.005, 0, zc), (0.01, r.outer_width, height), PANEL)
    # U markings on the left front post (small light ticks every 5U), visual only
    for u in range(1, r.units + 1, 5):
        z = cfg.u_bottom_z(u) + RACK_UNIT / 2
        batch.add(f"/World/Rack/umark_{u}", (x0 - 0.0005, post_y, z), (0.001, 0.02, 0.003), BEZEL_LIGHT, collide=False)


def build_filler_units(cfg: SceneCfg, batch: BoxBatch) -> None:
    """Generic servers, blanking panels and a top-of-rack switch around the target server."""
    s = cfg.server
    target = set(range(s.u_position, s.u_position + s.u_height))
    # (bottom U, height, kind) -- hand-written layout so it looks like a real rack
    layout = [(42, 1, "switch"), (41, 1, "blank"), (39, 2, "server"), (37, 2, "server"),
              (36, 1, "blank"), (35, 1, "server"), (34, 1, "server"), (33, 1, "server"),
              (31, 2, "server"), (29, 2, "blank"), (27, 2, "server"), (25, 2, "server"),
              (24, 1, "server"), (22, 2, "server"), (18, 2, "server"), (17, 1, "blank"),
              (15, 2, "server"), (13, 2, "server"), (12, 1, "blank"), (10, 2, "server"),
              (8, 2, "server"), (5, 3, "blank"), (3, 2, "server"), (1, 2, "blank")]
    r = cfg.rack
    x_face = r.front_x + s.recess
    for u, h, kind in layout:
        assert not (set(range(u, u + h)) & target), f"filler at U{u} overlaps the target server"
        z0 = cfg.u_bottom_z(u)
        hz = h * RACK_UNIT - 0.002
        zc = z0 + h * RACK_UNIT / 2
        base = f"/World/Rack/U{u:02d}_{kind}"
        if kind == "blank":
            batch.add(base, (r.front_x + 0.003, 0, zc), (0.004, r.opening_width + 0.03, hz), BLANK)
            continue
        depth = 0.30 if kind == "switch" else s.depth
        batch.add(f"{base}/body", (x_face + depth / 2, 0, zc), (depth, s.width, hz), CHASSIS)
        batch.add(f"{base}/bezel", (x_face - 0.003, 0, zc), (0.006, s.width, hz), SWITCH if kind == "switch" else BEZEL, collide=False)
        if kind == "switch":
            for k in range(24):  # two rows of 12 ports
                row, col = divmod(k, 12)
                y = -0.17 + col * 0.026
                batch.add(f"{base}/port_{k}", (x_face - 0.0065, y, zc + (row - 0.5) * 0.016),
                          (0.002, 0.014, 0.011), PORT, collide=False)
                if k % 3 == 0:
                    batch.add(f"{base}/portled_{k}", (x_face - 0.0068, y + 0.009, zc + (row - 0.5) * 0.016 + 0.007),
                              (0.002, 0.003, 0.002), (0, 0.6, 0.1), collide=False, material="led_status")
        else:
            batch.add(f"{base}/stripe", (x_face - 0.0065, -0.05, zc), (0.002, 0.24, hz * 0.35), BEZEL_LIGHT, collide=False)
            batch.add(f"{base}/led", (x_face - 0.0068, 0.19, zc), (0.002, 0.005, 0.005), (0, 0.6, 0.1),
                      collide=False, material="led_status")


def build_storage_server(cfg: SceneCfg, batch: BoxBatch) -> dict[tuple[int, int], BayInfo]:
    """The 2U, 12-bay server. Chassis is a hollow shell so caddies can later slide out."""
    s, c, r = cfg.server, cfg.caddy, cfg.rack
    base = "/World/Rack/StorageServer"
    ensure_xform(base)
    ensure_xform(f"{base}/Caddies")
    wall = 0.003
    h = s.u_height * RACK_UNIT - 0.002
    z0 = cfg.u_bottom_z(s.u_position) + 0.001
    zc = z0 + h / 2
    x_face = r.front_x + s.recess
    xc = x_face + s.depth / 2

    # shell: top, bottom, sides; backplane right behind the drives; internals behind that
    batch.add(f"{base}/top", (xc, 0, z0 + h - wall / 2), (s.depth, s.width, wall), CHASSIS)
    batch.add(f"{base}/bottom", (xc, 0, z0 + wall / 2), (s.depth, s.width, wall), CHASSIS)
    for side, sy in (("l", 1), ("r", -1)):
        batch.add(f"{base}/side_{side}", (xc, sy * (s.width / 2 - wall / 2), zc), (s.depth, wall, h), CHASSIS)
        # rack ears holding the server to the front posts
        batch.add(f"{base}/ear_{side}", (x_face - 0.002, sy * (s.width / 2 + 0.008), zc), (0.004, 0.016, h), BEZEL)
    backplane_x = x_face + c.depth + 0.004
    batch.add(f"{base}/backplane", (backplane_x + 0.002, 0, zc), (0.004, s.width - 2 * wall, h - 2 * wall), (0.05, 0.25, 0.1))
    internals_d = s.depth - (backplane_x - x_face) - 0.01
    batch.add(f"{base}/internals", (backplane_x + 0.01 + internals_d / 2, 0, zc), (internals_d, s.width - 2 * wall, h - 2 * wall), CHASSIS)

    inner_w = s.width - 2 * wall
    inner_h = h - 2 * wall
    pitch_y = inner_w / s.bay_cols
    pitch_z = inner_h / s.bay_rows
    # vertical separators between columns
    for k in range(1, s.bay_cols):
        y = inner_w / 2 - k * pitch_y
        batch.add(f"{base}/sep_{k}", (x_face + c.depth / 2, y, zc), (c.depth, pitch_y - c.face_width - 0.003, inner_h), CHASSIS)

    bays: dict[tuple[int, int], BayInfo] = {}
    for row in range(s.bay_rows):  # row 0 = top
        for col in range(s.bay_cols):  # col 0 = robot's left (+Y)
            y = inner_w / 2 - (col + 0.5) * pitch_y
            z = z0 + wall + inner_h - (row + 0.5) * pitch_z
            p = f"{base}/Caddies/bay_r{row}_c{col}"
            body = batch.add(f"{p}/body", (x_face + c.depth / 2, y, z), (c.depth, c.face_width, c.face_height), CADDY)
            tab_c = np.array([x_face - c.tab_depth / 2, y - 0.012, z])
            tab = batch.add(f"{p}/tab", tuple(tab_c), (c.tab_depth, c.tab_width, c.tab_height), TAB)
            led = batch.add(
                f"{p}/led",
                (x_face - 0.0015, y + c.face_width / 2 - 0.010, z + c.face_height / 2 - 0.006),
                (0.003, c.led_size, c.led_size),
                (0.0, 0.6, 0.1),
                collide=False,
                material="led_ok",
            )
            bays[(row, col)] = BayInfo(row, col, body, tab, led, tab_c, np.array([x_face, y, z]))
    return bays
