"""Tray open/close tests (arm layer, step 3). The full suite takes a few minutes.

  .venv-mujoco/Scripts/python -m pytest robot/arm -q
"""

import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from arm import Arm  # noqa: E402
from base import Base, load  # noqa: E402
from skills import APPROACH, FACING_RACK, FINGER_GAP, OPEN, RETREAT, Y, TrayService  # noqa: E402


def service():
    m, d = load()
    base = Base(m, d)
    arm = Arm(base)
    return m, d, base, arm, TrayService(base, arm)


def test_every_tray_has_a_collision_free_open_and_close():
    """Kinematics only, all 80 servers: from the base's line-up there is one arm posture for
    the whole opening sequence and one for the closing sequence, and neither collides with
    anything (the tray moving along, the rack, or the robot's own carriage)."""
    m, d, base, arm, svc = service()

    def feasible(name, start, chain, tray_at):
        for seed in arm.seeds(start):
            q = arm.solve_ik(start, FACING_RACK, seed)
            if q is None or svc._sweep(name, [q], [tray_at(0, 0.0)]):
                continue
            qk, pk = q, start
            for i, nxt in enumerate(chain):
                path = arm.plan_line(qk, pk, nxt, FACING_RACK)
                if path is None or svc._sweep(name, path, [tray_at(i, (k + 1) / len(path))
                                                           for k in range(len(path))]):
                    break
                qk, pk = path[-1], nxt
            else:
                return True
        return False

    for rack in range(1, 5):
        for u in range(1, 40, 2):
            name = f"r{rack:02}_u{u:02}"
            t = base.target(rack, u)
            mujoco.mj_resetDataKeyframe(m, d, 0)
            d.joint("rail_x").qpos, d.joint("lift_z").qpos = t.rail, t.lift
            d.joint("finger_joint1").qpos = d.joint("finger_joint2").qpos = FINGER_GAP  # as the skill
            mujoco.mj_forward(m, d)
            h = d.geom(f"{name}_handle_1").xpos.copy()
            assert feasible(name, h - APPROACH * Y, [h, h - OPEN * Y, h - (OPEN + RETREAT) * Y],
                            lambda i, s: (0.0, OPEN * s, OPEN)[i]), (name, "open")
            d.joint(f"{name}_slide").qpos = OPEN
            mujoco.mj_forward(m, d)
            he = d.geom(f"{name}_handle_1").xpos.copy()
            assert feasible(name, he - RETREAT * Y, [he, he + OPEN * Y, h - APPROACH * Y],
                            lambda i, s: (OPEN, OPEN * (1 - s), 0.0)[i]), (name, "close")


def open_and_close(rack, u):
    m, d, base, arm, svc = service()
    name = f"r{rack:02}_u{u:02}_slide"
    opened = svc.open_tray(rack, u)
    open_q = float(d.joint(name).qpos[0])
    refused = base.go_to(1, 1)                      # base must not move with a tray open
    closed = svc.close_tray(rack, u)
    return (opened, refused, closed), open_q, float(d.joint(name).qpos[0]), d.qpos.copy(), d.time, base, arm


def test_open_then_close_and_repeat_runs_are_bit_identical():
    (opened, refused, closed), open_q, closed_q, qpos1, t1, base, arm = open_and_close(2, 21)
    assert opened.startswith("OK") and abs(open_q - OPEN) < 0.015, opened
    assert refused.startswith("REFUSED"), refused
    assert closed.startswith("OK") and closed_q < 0.003, closed
    assert base.holding is None
    assert np.max(np.abs(arm.q() - np.array([0, -1.0, 0, -3.0, 0, 2.1, 0.7853]))) < 0.02  # folded
    results2, _, _, qpos2, t2, _, _ = open_and_close(2, 21)
    assert results2 == (opened, refused, closed)
    assert t1 == t2 and np.array_equal(qpos1, qpos2), "same commands must give bit-identical motion"


def test_lowest_tray_opens_and_closes():
    """U1 handles are 16 cm off the floor: the hand passes close to the robot's own carriage."""
    (opened, _, closed), open_q, closed_q, *_ = open_and_close(4, 1)
    assert opened.startswith("OK") and abs(open_q - OPEN) < 0.015, opened
    assert closed.startswith("OK") and closed_q < 0.003, closed


def test_jammed_tray_is_reported_and_left_closed():
    m, d, base, arm, svc = service()
    dof = m.joint("r03_u15_slide").dofadr[0]
    m.dof_frictionloss[dof] = 400.0                 # seized rail
    res = svc.open_tray(3, 15)
    assert res.startswith("JAMMED"), res
    assert d.joint("r03_u15_slide").qpos[0] < 0.003
    assert base.holding is None
    assert base.go_to(1, 1).startswith("OK"), "nothing left open or held, so the base may move"


def test_close_refused_when_base_is_elsewhere():
    m, d, base, arm, svc = service()
    d.joint("r02_u21_slide").qpos = OPEN
    mujoco.mj_forward(m, d)
    assert svc.close_tray(2, 21).startswith("REFUSED")


def test_already_open_or_closed_is_a_no_op():
    m, d, base, arm, svc = service()
    assert svc.close_tray(2, 21) == "OK: tray R02 U21 already closed"
    d.joint("r02_u21_slide").qpos = OPEN
    mujoco.mj_forward(m, d)
    assert svc.open_tray(2, 21).startswith("OK: tray R02 U21 already open")
