"""Local-first autopilot tests (no model). Run: uv run pytest robot/sandbox -q"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from policy import run_local_first  # noqa: E402
from scenarios import SCENARIOS, grade  # noqa: E402
from test_loop import D, ScriptedBrain  # noqa: E402


class NeverBrain:
    def decide(self, prompt, images):
        raise AssertionError("Argon should not be needed for this incident")


def world(name):
    return SCENARIOS[name][0]()


@pytest.mark.parametrize("name", ["clean_failure", "two_failures", "raid_rebuild",
                                  "obstacle", "no_spares", "same_group_double"])
def test_clear_cut_incidents_are_fully_local(name):
    w = world(name)
    t = run_local_first(w, NeverBrain(), verbose=False)
    assert t.model_calls == 0 and not t.escalations and t.fallback is None
    assert grade(name, w)["passed"], grade(name, w)


def test_raid_rebuild_waits_before_pulling():
    w = world("raid_rebuild")
    t = run_local_first(w, NeverBrain(), verbose=False)
    acts = [s["decision"]["action"] for s in t.steps]
    assert acts.index("wait") < acts.index("pull_caddy")


@pytest.mark.parametrize("name, expect", [
    ("misleading_led", "bay 2: drive health is OK"),
    ("false_alarm_heat", "bay 4: drive health is OK"),
    ("jammed_caddy", "bay 3: pull_caddy returned: JAMMED"),
    ("doa_spare", "bay 2: verification failed"),
])
def test_ambiguous_or_surprising_incidents_escalate(name, expect):
    w = world(name)
    t = run_local_first(w, ScriptedBrain([D("safe_stop", reason="test")]), verbose=False)
    assert any(e.startswith(expect) for e in t.escalations), t.escalations
    assert t.model_calls >= 1


def test_misleading_led_swaps_real_fault_locally_and_escalates_only_led():
    w = world("misleading_led")
    brain = ScriptedBrain([D("flag", bay=2, reason="LED fault, drive healthy"),
                           D("done", summary="x")])
    t = run_local_first(w, brain, verbose=False)
    assert grade("misleading_led", w)["passed"]
    assert t.escalations == [e for e in t.escalations if e.startswith("bay 2")]
    assert t.model_calls == 2 and t.local_decisions >= 6


def test_escalation_hands_over_context_and_unlocks_pull():
    """DOA: Argon gets the local history and can pull without re-reading telemetry."""
    w = world("doa_spare")
    brain = ScriptedBrain([D("pull_caddy"), D("stow_old"), D("pick_spare"), D("insert_caddy"),
                           D("read_telemetry", bay=2), D("done", summary="x")])
    t = run_local_first(w, brain, verbose=False)
    assert "LOCAL AUTOPILOT -> ARGON: bay 2" in brain.prompts[0]
    assert "[local] insert_caddy" in brain.prompts[0]
    assert grade("doa_spare", w)["passed"]
    assert not any(s["result"].startswith("POLICY") for s in t.steps)


def test_escalation_respects_budget():
    from agent import Budget
    w = world("false_alarm_heat")
    t = run_local_first(w, ScriptedBrain([D("wait")] * 10), Budget(max_model_calls=2), verbose=False)
    assert t.model_calls == 2 and w.stopped and "budget" in t.fallback


def test_argon_offline_during_escalation_fails_safe():
    class Dead:
        def decide(self, *a):
            raise ConnectionError("offline")
    w = world("jammed_caddy")
    t = run_local_first(w, Dead(), verbose=False)
    assert w.stopped and w.flags and "unreachable" in t.fallback


def test_same_group_double_replaces_dead_drive_first_then_waits_for_rebuild():
    w = world("same_group_double")
    t = run_local_first(w, NeverBrain(), verbose=False)
    pulls = [s for s in t.steps if s["decision"]["action"] == "pull_caddy"]
    acts = [s["decision"]["action"] for s in t.steps]
    assert len(pulls) == 2 and acts.index("wait") > acts.index("pull_caddy")
    assert grade("same_group_double", w)["passed"]


@pytest.mark.parametrize("name", ["smart_false_alarm", "smart_silent"])
def test_smart_disagreement_escalates_and_flag_passes(name):
    w = world(name)
    t = run_local_first(w, ScriptedBrain([D("flag", bay=3, reason="SMART disagrees with LED/heat"),
                                          D("done", summary="flagged")]), verbose=False)
    assert t.escalations and t.escalations[0].startswith("bay 3")
    assert grade(name, w)["passed"], grade(name, w)


def test_guard_refuses_pulling_survivor_of_degraded_group():
    w = world("same_group_double")
    run_local_first(w, ScriptedBrain([]), verbose=False)  # sanity: local path is safe
    assert not [e for e in w.events if e.startswith("CRITICAL")]
    from agent import guard
    w2 = world("same_group_double")
    w2.move_to(1)
    assert "already down" in guard(w2, D("pull_caddy", conf=0.99), {"telemetry_read": {1}})
