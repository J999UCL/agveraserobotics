"""Plug-in test: the team's unchanged decision loop driving the dcsim robot.

Uses scripted decisions (no Gemini), so it checks the interface, the physical hooks and the
grading, not model quality. Needs the simulator running (DCSIM=host:port). From robot/sandbox/:

  DCSIM=localhost:5555 DCSIM_REPO=/path/to/AgentVerseBob2 python -m pytest test_dcsim_plugin.py -q -s
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from agent import run  # noqa: E402
from dcsim_adapter import DcsimRack  # noqa: E402
from policy import run_local_first  # noqa: E402
from scenarios import SCENARIOS, grade  # noqa: E402
from test_loop import GOOD_SWAP_5, D, ScriptedBrain  # noqa: E402
from test_policy import NeverBrain  # noqa: E402


def rack(name):
    return DcsimRack(SCENARIOS[name][0]())


def test_clean_swap_with_scripted_brain():
    w = rack("clean_failure")
    t = run(w, ScriptedBrain(GOOD_SWAP_5), verbose=False)
    assert grade("clean_failure", w)["passed"]
    assert "lifted out" in t.steps[3]["result"]
    inst = w.c.call("get_installed", server=w.server)
    assert inst["A6"] is not None                      # a real part is back in the slot
    w.close()


def test_jam_is_physical():
    w = rack("jammed_caddy")
    brain = ScriptedBrain([D("look"), D("read_telemetry", bay=3), D("move_to", bay=3), D("pull_caddy"),
                           D("flag", bay=3, reason="jam"), D("safe_stop", reason="jam")])
    t = run(w, brain, verbose=False)
    assert t.steps[3]["result"].startswith("JAMMED")
    assert w.holding is None and grade("jammed_caddy", w)["passed"]
    assert w.c.get_status()["holding"] is None         # the robot let go of it too
    w.close()


def test_autopilot_replaces_two_failed_sticks():
    w = rack("two_failures")
    t = run_local_first(w, NeverBrain(), verbose=False)
    assert grade("two_failures", w)["passed"] and t.model_calls == 0
    w.close()


def test_autopilot_waits_for_technician_in_aisle():
    w = rack("obstacle")
    t = run_local_first(w, NeverBrain(), verbose=False)
    results = [s["result"] for s in t.steps]
    assert any(r.startswith("BLOCKED") for r in results)   # the lidar really stopped it
    assert grade("obstacle", w)["passed"] and t.model_calls == 0
    w.close()


def test_doa_spare_needs_a_second_swap():
    w = rack("doa_spare")
    brain = ScriptedBrain([
        D("read_telemetry", bay=2), D("move_to", bay=2), D("pull_caddy"), D("stow_old"), D("pick_spare"),
        D("insert_caddy"), D("read_telemetry", bay=2), D("pull_caddy"), D("stow_old"), D("pick_spare"),
        D("insert_caddy"), D("read_telemetry", bay=2), D("done", summary="replaced twice"),
    ])
    run(w, brain, verbose=False)
    assert grade("doa_spare", w)["passed"]
    w.close()


def test_runs_out_of_spares_and_fetches_more():
    w = rack("clean_failure")
    w.spares = 3
    onboard = w.c.get_onboard_inventory()
    for s in [k for k, v in onboard.items() if k.startswith("CD") and v]:   # empty the cassette
        assert w.c.pick(s)["ok"] and w.c.discard()["ok"]
    t = run(w, ScriptedBrain(GOOD_SWAP_5), verbose=False)
    assert "restocked" in t.steps[5]["result"]
    assert grade("clean_failure", w)["passed"]
    w.close()


def test_perception_reads_leds_from_camera():
    for name in ("clean_failure", "misleading_led"):
        w = rack(name)
        assert [v.led for v in w.perceive()] == [b.led for b in w.bays]
        w.c.close()
