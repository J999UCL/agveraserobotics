"""Plug-in test: run the unchanged decision loop against the team's MuJoCo scene.

Uses scripted decisions (no Argon), so it checks the interface, the physics hooks
and the grading, not model quality. Needs trimesh (imported by sim/run.py):
  uv run --with trimesh==5.2.0 pytest robot/sandbox/test_mujoco_plugin.py -q -s
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("trimesh")

from agent import run  # noqa: E402
from mujoco_adapter import MujocoRack  # noqa: E402
from scenarios import SCENARIOS, grade  # noqa: E402
from test_loop import GOOD_SWAP_5, D, ScriptedBrain  # noqa: E402

FRAMES = os.path.join(os.path.dirname(__file__), "runs", "mujoco")


def rack(name):
    return MujocoRack(SCENARIOS[name][0](), frames_dir=os.path.join(FRAMES, name))


def test_clean_swap_through_mujoco():
    w = rack("clean_failure")
    t = run(w, ScriptedBrain(GOOD_SWAP_5), verbose=False)
    g = grade("clean_failure", w)
    assert g["passed"], g
    assert w.tray_pos(5) < 0.03          # tray physically closed again
    assert t.steps[3]["result"].startswith("OK") and "extended" in t.steps[3]["result"]
    w.close()


def test_jam_detected_by_physics():
    w = rack("jammed_caddy")
    brain = ScriptedBrain([D("look"), D("read_telemetry", bay=3), D("move_to", bay=3),
                           D("pull_caddy"), D("flag", bay=3, reason="jam"), D("safe_stop", reason="jam")])
    t = run(w, brain, verbose=False)
    assert t.steps[3]["result"].startswith("JAMMED")
    assert w.tray_pos(3) < 0.03 and w.holding is None
    assert grade("jammed_caddy", w)["passed"]
    w.close()


def test_camera_is_a_real_render_with_status_leds():
    w = rack("clean_failure")
    png = w.camera_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 50_000
    w.close()


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_local_perception_reads_status_leds_from_render(name):
    w = MujocoRack(SCENARIOS[name][0]())
    assert [v.led for v in w.perceive()] == [b.led for b in w.bays]
    w.close()


def test_local_first_autopilot_on_mujoco():
    from policy import run_local_first
    from test_policy import NeverBrain
    w = rack("clean_failure")
    t = run_local_first(w, NeverBrain(), verbose=False)
    assert grade("clean_failure", w)["passed"] and t.model_calls == 0
    assert w.tray_pos(5) < 0.03
    w.close()
