"""Perception on the team's real MuJoCo data-centre scene (sim/datacentre.xml).

    python -m pytest -q tests/test_datacentre.py
"""
from __future__ import annotations

import numpy as np
import pytest

from vision import datacentre as dc

pytestmark = pytest.mark.skipif(not dc.SIM_XML.exists(), reason="sim/datacentre.xml not found")

CASES = [(2, 6, 5), (2, 12, 0), (3, 3, 7), (1, 15, 2)]   # rack, slot (2U server), faulty bay


@pytest.fixture(scope="module")
def scene():
    return dc.load()


def truth_led(d, rack, slot, bay):
    return dc.to_base(d, dc._tray_point(d, rack, slot, dc.led_local(bay)))


def truth_grab(d, rack, slot, bay):
    return dc.to_base(d, dc._tray_point(d, rack, slot, dc.grab_local(bay)))


@pytest.mark.parametrize("rack,slot,fault", CASES)
def test_finds_the_faulty_drive(scene, rack, slot, fault):
    m, d = scene
    dc.set_led(m, rack, slot, fault, "amber")
    try:
        cap = dc.capture(m, d, *dc.scan_pose(m, d, rack, slot))
        scan = dc.make_perception(m, d).scan(cap, target_xyz=dc.server_target(m, d, rack, slot))
    finally:
        dc.set_led(m, rack, slot, fault, "green")
    assert len(scan.bays) == 8, scan.summary()
    assert [b.bay for b in scan.faults()] == [fault]
    assert all(b.status == "ok" for b in scan.bays if b.bay != fault)
    for b in scan.bays:
        err = np.linalg.norm(np.array(b.led_xyz_base) - truth_led(d, rack, slot, b.bay))
        assert err < 0.005, f"bay {b.bay}: {err * 1000:.1f} mm"
    grab_err = np.linalg.norm(np.array(scan.faults()[0].tab_xyz_base) - truth_grab(d, rack, slot, fault))
    assert grab_err < 0.005, f"grab point off by {grab_err * 1000:.1f} mm"
    assert scan.obstacles == [], "server handles must not count as obstacles"


def test_dropped_cable_is_an_obstacle(scene):
    m0, d0 = scene
    pos, tgt = dc.scan_pose(m0, d0, 2, 6)
    out = (pos - tgt) / np.linalg.norm(pos - tgt)
    a = tgt + 0.15 * out + np.array([0.04, 0, -0.16])
    b = tgt + 0.15 * out + np.array([0.06, 0, 0.16])
    m, d = dc.load(cable=(a, b))
    dc.set_led(m, 2, 6, 5, "amber")
    cap = dc.capture(m, d, pos, tgt)
    scan = dc.make_perception(m, d).scan(cap, target_xyz=dc.server_target(m, d, 2, 6))
    assert len(scan.obstacles) == 1, scan.summary()
    assert scan.obstacles[0].clearance_m == pytest.approx(0.15 + 0.007 + 0.0045, abs=0.02)
    views = dc.bay_views(scan)
    assert any(v["obstructed"] for v in views)          # bays behind the cable are flagged
    assert [v["bay"] for v in views if v["led"] == "amber"] == [5]


def test_bay_views_match_agent_format(scene):
    m, d = scene
    dc.set_led(m, 2, 6, 3, "amber")
    try:
        cap = dc.capture(m, d, *dc.scan_pose(m, d, 2, 6))
        views = dc.bay_views(dc.make_perception(m, d).scan(cap, target_xyz=dc.server_target(m, d, 2, 6)))
    finally:
        dc.set_led(m, 2, 6, 3, "green")
    assert [v["bay"] for v in views] == list(range(8))
    assert {"bay", "led", "temp_c", "obstructed"} <= set(views[0])
    assert views[3]["led"] == "amber" and views[0]["led"] == "green"
    assert not any(v["obstructed"] for v in views)


def test_wrist_camera_renders(scene):
    m, d = scene
    cap = dc.capture_wrist(m, d)
    assert cap.rgb.shape == (480, 640, 3)
    assert np.isfinite(cap.depth).mean() > 0.5
    assert cap.K[0, 0] == pytest.approx(366.5, abs=0.5)


def test_within_edge_budget(scene):
    m, d = scene
    cap = dc.capture(m, d, *dc.scan_pose(m, d, 2, 6))
    per = dc.make_perception(m, d)
    tgt = dc.server_target(m, d, 2, 6)
    per.scan(cap, target_xyz=tgt)
    assert min(per.scan(cap, target_xyz=tgt).latency_ms for _ in range(5)) < 60
