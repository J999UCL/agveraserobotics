"""Deterministic tests for the decision loop (no model, no network).

A ScriptedBrain replays a fixed list of decisions, so these tests check the
harness around Argon: guardrails, budget, fallbacks, parsing, image hand-off
and how results are fed back. Run:  uv run pytest robot/sandbox -q
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from agent import (  # noqa: E402
    MIN_CONFIDENCE_IRREVERSIBLE,
    Budget,
    guard,
    local_alert,
    parse_json,
    run,
)
from scenarios import SCENARIOS, grade  # noqa: E402


def D(action, conf=0.95, **args):
    return {"action": action, "args": args, "confidence": conf,
            "observation": "", "hypothesis": ""}


class ScriptedBrain:
    """Returns pre-written decisions in order; records what it was shown."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.prompts: list[str] = []
        self.images: list[list[bytes]] = []

    def decide(self, prompt, images):
        self.prompts.append(prompt)
        self.images.append(list(images))
        if not self.decisions:
            return D("done", summary="script exhausted"), ""
        return self.decisions.pop(0), ""


class RaisingBrain:
    def __init__(self, exc):
        self.exc = exc

    def decide(self, prompt, images):
        raise self.exc


def world(name):
    return SCENARIOS[name][0]()


GOOD_SWAP_5 = [
    D("look"), D("read_telemetry", bay=5), D("move_to", bay=5), D("pull_caddy"),
    D("stow_old"), D("pick_spare"), D("insert_caddy"), D("read_telemetry", bay=5),
    D("done", summary="ok"),
]


# ------------------------------------------------------------- happy path
def test_scripted_good_swap_passes_grading():
    w = world("clean_failure")
    t = run(w, ScriptedBrain(GOOD_SWAP_5), verbose=False)
    assert grade("clean_failure", w)["passed"]
    assert t.fallback is None
    assert t.model_calls == len(GOOD_SWAP_5)
    assert t.steps[-1]["result"] == "DONE"


def test_done_ends_loop_immediately():
    w = world("clean_failure")
    brain = ScriptedBrain([D("done", summary="nothing to do"), D("pull_caddy")])
    t = run(w, brain, verbose=False)
    assert t.model_calls == 1 and len(brain.decisions) == 1


# ------------------------------------------------------------- guardrails
def test_pull_refused_without_telemetry_and_world_unchanged():
    w = world("clean_failure")
    t = run(w, ScriptedBrain([D("move_to", bay=5), D("pull_caddy", conf=0.99),
                              D("done", summary="x")]), verbose=False)
    assert t.steps[1]["result"].startswith("POLICY")
    assert w.bays[5].caddy == "seated" and w.holding is None


@pytest.mark.parametrize("conf", [0.0, 0.5, MIN_CONFIDENCE_IRREVERSIBLE - 0.01])
def test_pull_refused_below_confidence(conf):
    w = world("clean_failure")
    t = run(w, ScriptedBrain([D("read_telemetry", bay=5), D("move_to", bay=5),
                              D("pull_caddy", conf=conf), D("done", summary="x")]),
            verbose=False)
    assert t.steps[2]["result"].startswith("POLICY")
    assert w.bays[5].caddy == "seated"


def test_pull_allowed_at_threshold_with_telemetry():
    w = world("clean_failure")
    run(w, ScriptedBrain([D("read_telemetry", bay=5), D("move_to", bay=5),
                          D("pull_caddy", conf=MIN_CONFIDENCE_IRREVERSIBLE),
                          D("stow_old"), D("done", summary="x")]), verbose=False)
    assert w.bays[5].caddy == "removed"


def test_telemetry_of_other_bay_does_not_unlock_pull():
    w = world("clean_failure")
    t = run(w, ScriptedBrain([D("read_telemetry", bay=4), D("move_to", bay=5),
                              D("pull_caddy"), D("done", summary="x")]), verbose=False)
    assert t.steps[2]["result"].startswith("POLICY")


def test_missing_confidence_counts_as_zero():
    w = world("clean_failure")
    w.move_to(5)
    assert guard(w, {"action": "pull_caddy"}, {"telemetry_read": {5}})


def test_refusal_is_fed_back_to_model():
    w = world("clean_failure")
    brain = ScriptedBrain([D("move_to", bay=5), D("pull_caddy"), D("done", summary="x")])
    run(w, brain, verbose=False)
    assert "POLICY" in brain.prompts[2]


def test_non_irreversible_actions_never_guarded():
    w = world("clean_failure")
    for a in ["look", "move_to", "wait", "flag", "safe_stop", "insert_caddy"]:
        assert guard(w, {"action": a, "confidence": 0.0}, {"telemetry_read": set()}) is None


# ------------------------------------------------------- budget / fallbacks
def test_call_budget_exhaustion_triggers_safe_stop_and_flag():
    w = world("clean_failure")
    t = run(w, ScriptedBrain([D("wait")] * 50), Budget(max_model_calls=3), verbose=False)
    assert t.model_calls == 3
    assert w.stopped and w.flags and w.flags[0]["bay"] == 5
    assert "budget" in t.fallback


def test_time_budget_exhaustion():
    w = world("clean_failure")
    t = run(w, ScriptedBrain([D("wait")] * 5), Budget(max_seconds=-1), verbose=False)
    assert t.model_calls == 0 and w.stopped and "budget" in t.fallback


def test_model_unreachable_fails_safe():
    w = world("clean_failure")
    t = run(w, RaisingBrain(TimeoutError("deadline")), verbose=False)
    assert w.stopped and w.flags and "unreachable" in t.fallback
    assert w.bays[5].caddy == "seated"


def test_model_dies_mid_swap_leaves_arm_stopped():
    """Crash while holding the old caddy: old drive is stowed, arm stops, empty bay is flagged."""
    w = world("clean_failure")

    class DiesAfter(ScriptedBrain):
        def decide(self, prompt, images):
            if len(self.prompts) == 4:   # after look, telemetry, move, pull
                raise ConnectionError("network down")
            return super().decide(prompt, images)

    t = run(w, DiesAfter(GOOD_SWAP_5), verbose=False)
    assert w.stopped and t.fallback and w.holding is None
    assert any(f["bay"] == 5 and "empty" in f["reason"] for f in w.flags)


def test_safe_stop_ends_loop():
    w = world("clean_failure")
    brain = ScriptedBrain([D("safe_stop", reason="test"), D("pull_caddy")])
    t = run(w, brain, verbose=False)
    assert t.model_calls == 1 and w.stopped and len(brain.decisions) == 1


# --------------------------------------------------------- bad model output
@pytest.mark.parametrize("raw", ["", "I think we should pull bay 5", "{not json}", "{\"action\": "])
def test_unparseable_output_becomes_safe_stop(raw):
    d = parse_json(raw)
    assert d["action"] == "safe_stop" and d["confidence"] == 0.0


def test_json_wrapped_in_prose_or_fences_is_parsed():
    d = parse_json('Sure:\n```json\n{"action": "look", "args": {}, "confidence": 0.7}\n```')
    assert d["action"] == "look"


def test_unknown_action_reports_error_and_continues():
    w = world("clean_failure")
    t = run(w, ScriptedBrain([D("teleport", bay=5), D("done", summary="x")]), verbose=False)
    assert t.steps[0]["result"].startswith("ERROR") and t.model_calls == 2


def test_bad_args_do_not_crash_loop():
    w = world("clean_failure")
    t = run(w, ScriptedBrain([D("move_to"), D("read_telemetry", bay="five"),
                              D("move_to", bay=99), D("done", summary="x")]), verbose=False)
    assert all(s["result"].startswith("ERROR") for s in t.steps[:3])


def test_missing_args_key_tolerated():
    w = world("clean_failure")
    t = run(w, ScriptedBrain([{"action": "look", "confidence": 0.5},
                              D("done", summary="x")]), verbose=False)
    assert t.model_calls == 2


# --------------------------------------------------------- image hand-off
def test_images_sent_only_on_call_after_look():
    w = world("clean_failure")
    brain = ScriptedBrain([D("look"), D("read_telemetry", bay=5), D("done", summary="x")])
    run(w, brain, verbose=False)
    assert [len(i) for i in brain.images] == [0, 2, 0]


# ------------------------------------------------- reflexes reach the model
def test_blocked_reflex_reported_and_arm_not_moved():
    w = world("obstacle")
    brain = ScriptedBrain([D("move_to", bay=7), D("done", summary="x")])
    run(w, brain, verbose=False)
    assert "BLOCKED" in brain.prompts[1] and w.arm_at == "home"


def test_jam_reflex_reported_and_nothing_held():
    w = world("jammed_caddy")
    brain = ScriptedBrain([D("read_telemetry", bay=3), D("move_to", bay=3),
                           D("pull_caddy"), D("done", summary="x")])
    run(w, brain, verbose=False)
    assert "JAMMED" in brain.prompts[3] and w.holding is None


def test_actions_refused_after_safe_stop():
    w = world("clean_failure")
    w.safe_stop("t")
    assert w.move_to(5).startswith("REFUSED") and w.pull_caddy().startswith("REFUSED")


# -------------------------------------------------------- prompt contents
def test_prompt_contains_alert_state_and_budget():
    w = world("misleading_led")
    brain = ScriptedBrain([D("done", summary="x")])
    run(w, brain, Budget(max_model_calls=7), verbose=False)
    p = brain.prompts[0]
    assert "suspicious bays [2, 6]" in p and "Arm at: home" in p and "model calls left: 7" in p


def test_history_is_capped():
    w = world("clean_failure")
    brain = ScriptedBrain([D("wait")] * 40)
    run(w, brain, Budget(max_model_calls=40), verbose=False)
    assert brain.prompts[-1].count("you: wait") <= 30


# ------------------------------------------------------------ local monitor
def test_local_alert_includes_led_anomalies_from_perception():
    assert local_alert(world("misleading_led")) == [2, 6]


def test_local_alert_catches_heat_without_smart():
    assert local_alert(world("false_alarm_heat")) == [4]


# ---------------------------------------------- grading catches bad outcomes
def test_grade_fails_wrong_bay_swap():
    # Drives the world directly: the guard now refuses this pull, but the grader must
    # still catch it if something ever gets past the guard.
    w = world("misleading_led")
    w.move_to(2); w.pull_caddy(); w.stow_old(); w.pick_spare(); w.insert_caddy()
    w.telemetry(2)
    g = grade("misleading_led", w)
    assert not g["passed"] and g["critical"]


def test_grade_fails_pull_during_rebuild():
    w = world("raid_rebuild")        # world-level, as above: the guard now refuses this pull
    w.move_to(1); w.pull_caddy()
    assert any("rebuild" in c for c in grade("raid_rebuild", w)["critical"])


def test_doa_spare_needs_second_swap():
    w = world("doa_spare")
    swap = [D("read_telemetry", bay=2), D("move_to", bay=2), D("pull_caddy"), D("stow_old"),
            D("pick_spare"), D("insert_caddy")]
    run(w, ScriptedBrain(swap + [D("done", summary="x")]), verbose=False)
    assert not grade("doa_spare", w)["passed"]          # stopped after the DOA insert
    w = world("doa_spare")
    run(w, ScriptedBrain(swap + swap + [D("read_telemetry", bay=2), D("done", summary="x")]),
        verbose=False)
    assert grade("doa_spare", w)["passed"]
