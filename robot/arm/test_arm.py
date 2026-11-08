"""Arm motion and force reading tests (arm layer, step 2).

  .venv-mujoco/Scripts/python -m pytest robot/arm -q
"""

import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from arm import Arm, rotation  # noqa: E402
from base import Base, load  # noqa: E402

FACING_RACK = rotation([0, 1, 0], [1, 0, 0])


def setup(rack=2, u=21):
    m, d = load()
    base = Base(m, d)
    arm = Arm(base)
    assert base.go_to(rack, u).startswith("OK")
    arm.gripper("closed")
    return m, d, base, arm, d.geom(f"r{rack:02}_u{u:02}_handle_1").xpos.copy()


def test_free_and_straight_moves_are_accurate():
    m, d, base, arm, h = setup()
    res = arm.move_free(h - [0, 0.10, 0], FACING_RACK)
    assert res.startswith("OK"), res
    p, r = arm.tcp()
    assert np.linalg.norm(p - (h - [0, 0.10, 0])) < 0.002
    # Straight line: sample the tool path and check it never strays from the line.
    start, goal = p.copy(), h - [0, 0.03, 0]
    worst = [0.0]
    step = base.step

    def tracking_step():
        step()
        q, _ = arm.tcp()
        s = np.clip((q - start) @ (goal - start) / np.sum((goal - start) ** 2), 0, 1)
        worst[0] = max(worst[0], np.linalg.norm(q - (start + s * (goal - start))))
    base.step = tracking_step
    res = arm.move_linear(goal, FACING_RACK)
    assert res.startswith("OK"), res
    assert np.linalg.norm(arm.tcp()[0] - goal) < 0.002
    assert worst[0] < 0.003, f"strayed {worst[0] * 1000:.1f} mm from the straight line"


def test_touch_stops_on_push_back_and_force_reading_matches():
    m, d, base, arm, h = setup()
    arm.move_free(h - [0, 0.10, 0], FACING_RACK)
    arm.move_linear(h - [0, 0.03, 0], FACING_RACK)
    res = arm.guarded_move([0, 1, 0], 0.05, max_force=5.0)
    assert res.startswith("CONTACT"), res
    peak = float(res.split(" at ")[1].split(" N")[0])
    assert 5.0 < peak < 8.0, res                       # small overshoot only
    est, truth = arm.force(), arm.contact_force_truth()
    assert np.linalg.norm(truth) > 0.1, "should still be touching"
    assert np.linalg.norm(est - truth) < 0.05 + 0.02 * np.linalg.norm(truth)
    assert abs(d.joint("r02_u21_slide").qpos[0]) < 1e-3, "touch must not move the tray"
    back = arm.guarded_move([0, -1, 0], 0.10, max_force=5.0, speed=0.05)
    assert back.startswith("OK"), back
    assert np.linalg.norm(arm.contact_force_truth()) == 0


def test_force_reading_is_zero_in_free_space():
    m, d, base, arm, h = setup()
    arm.move_free(h - [0, 0.10, 0], FACING_RACK)
    assert np.linalg.norm(arm.force()) < 0.05


def test_moving_contact_is_refused_in_a_free_move():
    m, d, base, arm, h = setup()
    arm.move_free(h - [0, 0.10, 0], FACING_RACK)
    res = arm.move_linear(h + [0, 0.02, 0], FACING_RACK)   # straight through the handle
    assert res.startswith("BLOCKED"), res


def test_unreachable_pose_is_an_error():
    m, d, base, arm, h = setup()
    assert arm.move_free(h + [0, 0, 1.5], FACING_RACK) == "ERROR: pose out of reach"


def run_sequence():
    m, d, base, arm, h = setup()
    out = [arm.move_free(h - [0, 0.10, 0], FACING_RACK),
           arm.move_linear(h - [0, 0.03, 0], FACING_RACK),
           arm.guarded_move([0, 1, 0], 0.05, max_force=5.0),
           arm.guarded_move([0, -1, 0], 0.10, max_force=5.0, speed=0.05)]
    return out, d.qpos.copy(), d.time


def test_repeat_runs_are_bit_identical():
    a, qa, ta = run_sequence()
    b, qb, tb = run_sequence()
    assert a == b and ta == tb
    assert np.array_equal(qa, qb)


@pytest.mark.parametrize("rack,u", [(1, 1), (4, 39)])
def test_touch_works_at_the_extremes(rack, u):
    """Lowest server on the left rack, highest on the right rack."""
    m, d, base, arm, h = setup(rack, u)
    assert arm.move_free(h - [0, 0.10, 0], FACING_RACK).startswith("OK")
    assert arm.move_linear(h - [0, 0.03, 0], FACING_RACK).startswith("OK")
    assert arm.guarded_move([0, 1, 0], 0.05, max_force=5.0).startswith("CONTACT")
