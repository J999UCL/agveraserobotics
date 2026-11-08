"""Approach-path check: clear / blocked / unknown, and the continue-stop-continue monitor.

    python -m pytest -q vision/tests/test_corridor.py
"""
from __future__ import annotations

import numpy as np
import pytest

from vision import datacentre as dc
from vision.corridor import ApproachMonitor, check_path
from vision.synthetic import render
from vision.synthetic_hotswap import CENTRE_Z, FACE_X, FOVY_ISAAC, build, tab_pos

LEDS = {b: "green" for b in range(12)}
CAM, TGT = [FACE_X - 0.30, 0.0, CENTRE_Z], [FACE_X, 0.0, CENTRE_Z]
BACK = np.array([-1.0, 0.0, 0.0])  # away from the rack (stand-in base frame: +X towards the rack)


@pytest.fixture(scope="module")
def clear_cap():
    return render(build(LEDS), CAM, TGT, fovy=FOVY_ISAAC)


@pytest.fixture(scope="module")
def cable_cap():
    return render(build(LEDS, obstacle=True), CAM, TGT, fovy=FOVY_ISAAC)


def test_clear_path(clear_cap):
    g = tab_pos(6)
    c = check_path(clear_cap, g + 0.15 * BACK, g)
    assert c.status == "clear" and c.verified >= 0.9, c.reasons


def test_cable_blocks_the_path_behind_it(cable_cap):
    g = tab_pos(5)                      # the cable hangs in front of this bay's latch
    c = check_path(cable_cap, g + 0.25 * BACK, g)
    assert c.status == "blocked", c.reasons
    assert c.blocked_at == pytest.approx(0.47 - 0.007 - (g[0] - 0.25), abs=0.02)
    assert c.reasons and "cm ahead" in c.reasons[0]


def test_cable_does_not_block_a_path_beside_it(cable_cap):
    g = tab_pos(6)
    assert check_path(cable_cap, g + 0.15 * BACK, g).status == "clear"


def test_path_out_of_view_is_unknown_not_clear(clear_cap):
    c = check_path(clear_cap, [0.30, 0.80, 0.60], [0.60, 0.90, 0.60])
    assert c.status == "unknown" and "view" in c.reasons[0]


def test_path_hidden_behind_something_is_unknown(clear_cap):
    # a path inside the server chassis, behind the drive faces: the camera can't see it
    c = check_path(clear_cap, [FACE_X + 0.02, 0.0, CENTRE_Z], [FACE_X + 0.15, 0.0, CENTRE_Z])
    assert c.status == "unknown" and "hidden" in c.reasons[0]


def test_fast_enough(cable_cap):
    g = tab_pos(5)
    check_path(cable_cap, g + 0.25 * BACK, g)
    assert min(check_path(cable_cap, g + 0.25 * BACK, g).latency_ms for _ in range(5)) < 60


# ---------------------------------------------------------------- real scene, camera on the gripper
@pytest.mark.skipif(not dc.SIM_XML.exists(), reason="sim/datacentre.xml not found")
def test_monitor_stops_when_a_cable_drops_in_and_resumes_after():
    m, d = dc.load(cable_mocap=True)
    # bay 2: a middle column (the rightmost latches sit right beside the server handle, which the
    # check rightly reports as being in a 10 cm-wide gripper's way: see the next test)
    rack, slot, bay = 2, 6, 2
    goal_w = dc._tray_point(d, rack, slot, dc.grab_local(bay))
    out_w = d.body(dc.ident(rack, slot)).xmat.reshape(3, 3) @ np.array([0, -1.0, 0])
    tool_w = goal_w + 0.28 * out_w                              # gripper 28 cm in front of the latch

    def frame():  # wrist camera: rides on the gripper, looking at the goal
        return dc.capture(m, d, tool_w + np.array([0, 0, 0.03]), goal_w)

    mon = ApproachMonitor(dc.to_base(d, goal_w), confirm=2)
    tool = dc.to_base(d, tool_w)
    actions = [mon.update(frame(), tool)[0]]
    dc.place_cable(m, d, rack, slot, side_x=dc.grab_local(bay)[0] + 0.01)  # something drops into the path
    act, chk = mon.update(frame(), tool)
    actions.append(act)
    dc.park_cable(m, d)                                         # cleared
    actions += [mon.update(frame(), tool)[0], mon.update(frame(), tool)[0]]
    assert actions == ["continue", "stop", "stop", "continue"], [(a, c.reasons) for a, c in mon.history]
    assert chk.status == "blocked" and chk.blocked_at == pytest.approx(0.28 - 0.15, abs=0.03)


@pytest.mark.skipif(not dc.SIM_XML.exists(), reason="sim/datacentre.xml not found")
def test_server_handle_counts_for_a_wide_gripper_but_not_a_narrow_one():
    m, d = dc.load()
    goal_w = dc._tray_point(d, 2, 6, dc.grab_local(3))           # rightmost latch, next to the handle
    out_w = d.body(dc.ident(2, 6)).xmat.reshape(3, 3) @ np.array([0, -1.0, 0])
    tool_w = goal_w + 0.28 * out_w
    cap = dc.capture(m, d, tool_w + np.array([0, 0, 0.03]), goal_w)
    goal, tool = dc.to_base(d, goal_w), dc.to_base(d, tool_w)
    assert check_path(cap, tool, goal, radius=0.05).status == "blocked"   # handle is 2.3 cm to the side
    assert check_path(cap, tool, goal, radius=0.015).status == "clear"


@pytest.mark.skipif(not dc.SIM_XML.exists(), reason="sim/datacentre.xml not found")
def test_something_right_in_front_of_the_gripper_is_still_seen():
    # regression: a big "ignore the first few cm" margin once let a cable 5 cm ahead read as clear
    m, d = dc.load(cable_mocap=True)
    rack, slot, bay = 2, 6, 2
    goal_w = dc._tray_point(d, rack, slot, dc.grab_local(bay))
    out_w = d.body(dc.ident(rack, slot)).xmat.reshape(3, 3) @ np.array([0, -1.0, 0])
    tool_w = goal_w + 0.20 * out_w                                   # cable hangs 15 cm out: 5 cm ahead
    dc.place_cable(m, d, rack, slot, side_x=dc.grab_local(bay)[0] + 0.01)
    cap = dc.capture(m, d, tool_w + np.array([0, 0, 0.03]), goal_w)
    c = check_path(cap, dc.to_base(d, tool_w), dc.to_base(d, goal_w))
    assert c.status == "blocked" and c.blocked_at < 0.08, c.reasons
