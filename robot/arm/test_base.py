"""Rail + lift positioning tests (arm layer, step 1).

Run with the sim environment:
  .venv-mujoco/Scripts/python -m pytest robot/arm -q      (Windows)
  .venv-mujoco/bin/python -m pytest robot/arm -q          (macOS / Linux)
"""

import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from base import TUCK, Base, load  # noqa: E402

# Far corners and the middle: bottom-left, top-right, middle, back to the bottom.
ROUTE = [(1, 1), (4, 39), (2, 21), (3, 1)]
REACH = (0.30, 0.85)   # m from the shoulder; the Panda reaches ~0.855 m


def every_server():
    return [(r, u) for r in range(1, 5) for u in range(1, 40, 2)]


def test_every_server_is_reachable_and_clear():
    """Kinematics only: for all 80 servers the target fits on the rail and lift, both the
    closed and the fully extended handle are within reach, and a fully extended tray does
    not touch the robot standing beside it."""
    m, d = load()
    base = Base(m, d)
    for rack, u in every_server():
        t = base.target(rack, u)
        assert base.rail_lo <= t.rail <= base.rail_hi and base.lift_lo <= t.lift <= base.lift_hi
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.joint("rail_x").qpos = t.rail
        d.joint("lift_z").qpos = t.lift
        d.joint(f"r{rack:02}_u{u:02}_slide").qpos = 0.70
        mujoco.mj_forward(m, d)
        shoulder = d.body("link2").xpos
        for handle in (t.handle, t.handle - [0, 0.70, 0]):
            assert REACH[0] <= np.linalg.norm(handle - shoulder) <= REACH[1], (rack, u)
        assert base.external_contact() is None, (rack, u, base.external_contact())


def drive_route():
    m, d = load()
    base = Base(m, d)
    results = []
    for rack, u in ROUTE:
        results.append(base.go_to(rack, u))
        t = base.target(rack, u)
        arm_base = d.body("link0").xpos
        expect = base.base0 + [t.rail, 0, t.lift]
        assert np.max(np.abs(arm_base - expect)) < 0.002, (rack, u, arm_base, expect)
    return results, d.qpos.copy(), d.time


def test_go_to_reaches_each_target_and_is_deterministic():
    first, qpos1, t1 = drive_route()
    assert all(r.startswith("OK") for r in first), first
    second, qpos2, t2 = drive_route()
    assert first == second
    assert t1 == t2
    assert np.array_equal(qpos1, qpos2), "same commands must give bit-identical motion"


def test_arm_is_folded_while_travelling():
    m, d = load()
    base = Base(m, d)
    d.ctrl[base.arm_act] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]   # unfolded
    for _ in range(1000):
        mujoco.mj_step(m, d)
    assert base.go_to(2, 11).startswith("OK")
    q = np.array([d.joint(f"joint{i}").qpos[0] for i in range(1, 8)])
    assert np.max(np.abs(q - TUCK)) < 0.02


def test_refuses_with_open_tray_or_held_part():
    m, d = load()
    base = Base(m, d)
    d.joint("r03_u09_slide").qpos = 0.30
    mujoco.mj_forward(m, d)
    rail_before = base.rail_q
    res = base.go_to(1, 1)
    assert res.startswith("REFUSED") and "r03_u09" in res
    assert base.rail_q == rail_before
    d.joint("r03_u09_slide").qpos = 0.0
    base.holding = "drive"
    assert base.go_to(1, 1).startswith("REFUSED")


def test_contact_reflex_stops_the_base():
    """Even with the open-tray rule switched off, touching a tray stops the base in place."""
    m, d = load()
    base = Base(m, d)
    base.refusal = lambda: None
    assert base.go_to(4, 1).startswith("OK")          # lift fully down, beside rack 4
    d.joint("r03_u01_slide").qpos = 0.70               # low tray pulled out into the aisle
    mujoco.mj_forward(m, d)
    res = base.go_to(1, 1)                             # path runs through that tray
    assert res.startswith("BLOCKED") and "r03_u01" in res, res
    stopped = base.rail_q
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert abs(base.rail_q - stopped) < 0.02, "base must hold where the reflex stopped it"
    assert base.rail_q > 0.32, "stopped before passing the tray"


@pytest.mark.parametrize("rack,u", [(0, 1), (5, 1), (1, 0), (1, 41)])
def test_rejects_unknown_positions(rack, u):
    m, d = load()
    assert Base(m, d).go_to(rack, u).startswith("ERROR")
