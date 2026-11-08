"""Regression tests for design holes found in the toy rack review (no model, no network).

Each test asserts the behaviour we want and was written to fail against the original
design; all of them now pass. If one fails again, a hole has reopened.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import agent  # noqa: E402
from agent import Budget, parse_json, run  # noqa: E402
from policy import run_local_first  # noqa: E402
from scenarios import SCENARIOS, grade  # noqa: E402
from test_loop import D, ScriptedBrain  # noqa: E402


def world(name):
    return SCENARIOS[name][0]()


def fail_bay(w, i, smart="FAIL", drive="failed"):
    b = w.bays[i]
    b.led, b.temp_c, b.smart, b.realloc_sectors, b.drive = "red", 51, smart, 600, drive


class HookedBrain(ScriptedBrain):
    """ScriptedBrain that can change the world before answering call #k (0-based)."""

    def __init__(self, decisions, hooks):
        super().__init__(decisions)
        self.hooks = hooks

    def decide(self, prompt, images):
        hook = self.hooks.get(len(self.prompts))
        if hook:
            hook()
        return super().decide(prompt, images)


SWAP = [D("move_to", bay=5), D("pull_caddy"), D("stow_old"), D("pick_spare"), D("insert_caddy")]


# =================================================== 1. the guard on pull_caddy
def test_guard_refuses_pulling_drive_with_clean_smart():
    w = world("false_alarm_heat")   # bay 4 hot, SMART OK, drive healthy
    run(w, ScriptedBrain([D("read_telemetry", bay=4), D("move_to", bay=4),
                          D("pull_caddy", conf=0.99), D("done", summary="x")]), verbose=False)
    assert 4 not in w.pulled_healthy


def test_guard_refuses_pull_on_uncorroborated_smart():
    w = world("clean_failure")
    w.bays[5].led, w.bays[5].temp_c = "green", 34   # SMART FAIL only, nothing agrees
    t = run(w, ScriptedBrain([D("read_telemetry", bay=5), D("move_to", bay=5),
                              D("pull_caddy", conf=0.99), D("done", summary="x")]), verbose=False)
    assert t.steps[2]["result"].startswith("POLICY")


def test_guard_refuses_pull_during_rebuild():
    w = world("raid_rebuild")
    run(w, ScriptedBrain([D("read_telemetry", bay=1), D("move_to", bay=1),
                          D("pull_caddy", conf=0.99), D("done", summary="x")]), verbose=False)
    assert not [e for e in w.events if "rebuild" in e]


def test_stale_telemetry_does_not_unlock_pull():
    w = world("clean_failure")
    start_rebuild = lambda: w.raid_rebuilding.__setitem__("B", 3)  # noqa: E731
    run(w, HookedBrain([D("read_telemetry", bay=5), D("move_to", bay=5),
                        D("pull_caddy", conf=0.99), D("done", summary="x")],
                       hooks={2: start_rebuild}), verbose=False)
    assert not [e for e in w.events if e.startswith("CRITICAL")]


def test_local_autopilot_rechecks_rebuild_after_waiting():
    w = world("obstacle")           # bay 7 (group B) failed, path blocked for 2 waits
    orig_wait, started = w.wait, []

    def wait():                      # a group-B rebuild kicks off while we wait
        if not started:
            started.append(1)
            w.raid_rebuilding["B"] = 3
        return orig_wait()

    w.wait = wait
    run_local_first(w, ScriptedBrain([D("safe_stop", reason="t")]), verbose=False)
    assert not [e for e in w.events if e.startswith("CRITICAL")]


def test_done_refused_while_insert_unverified():
    w = world("clean_failure")
    run(w, ScriptedBrain([D("read_telemetry", bay=5), *SWAP, D("done", summary="early"),
                          D("read_telemetry", bay=5), D("done", summary="ok")]), verbose=False)
    assert not w.unverified


def test_done_refused_while_holding():
    w = world("clean_failure")
    run(w, ScriptedBrain([D("read_telemetry", bay=5), D("move_to", bay=5), D("pull_caddy"),
                          D("done", summary="early"), D("stow_old"),
                          D("done", summary="ok")]), verbose=False)
    assert w.holding is None


# ============================================= 2. fallbacks and failure handling
def test_fallback_after_pull_leaves_gripper_empty():
    w = world("clean_failure")

    class DiesAfterPull(ScriptedBrain):
        def decide(self, prompt, images):
            if len(self.prompts) == 3:
                raise ConnectionError("network down")
            return super().decide(prompt, images)

    run(w, DiesAfterPull([D("read_telemetry", bay=5), D("move_to", bay=5), D("pull_caddy")]),
        verbose=False)
    assert w.holding is None


def test_fallback_holding_spare_reseats_or_parks_it():
    w = world("clean_failure")

    class DiesAfterPick(ScriptedBrain):
        def decide(self, prompt, images):
            if len(self.prompts) == 5:
                raise ConnectionError("network down")
            return super().decide(prompt, images)

    run(w, DiesAfterPick([D("read_telemetry", bay=5), D("move_to", bay=5), D("pull_caddy"),
                          D("stow_old"), D("pick_spare")]), verbose=False)
    assert w.holding is None


def test_budget_fallback_flags_every_suspect():
    w = world("two_failures")       # bays 0 and 6 failed
    run(w, ScriptedBrain([D("wait")] * 10), Budget(max_model_calls=2), verbose=False)
    assert {0, 6} <= {f["bay"] for f in w.flags}


def _jam_plus_second_failure():
    w = world("jammed_caddy")       # bay 3 failed and jams
    fail_bay(w, 6)                  # bay 6 failed too
    return w


def test_safe_stop_mid_incident_flags_remaining_suspects():
    w = _jam_plus_second_failure()
    run_local_first(w, ScriptedBrain([D("safe_stop", reason="jam")]), verbose=False)
    assert 6 in {f["bay"] for f in w.flags}


def test_time_budget_shared_across_escalations(monkeypatch):
    clock = [0.0]

    class FakeTime:
        @staticmethod
        def time():
            return clock[0]

    monkeypatch.setattr(agent, "time", FakeTime)

    class SlowBrain(ScriptedBrain):
        def decide(self, prompt, images):
            clock[0] += 100
            return super().decide(prompt, images)

    w = world("false_alarm_heat")
    w.bays[2].led = "amber"         # two ambiguous bays (2 and 4) -> two escalations
    t = run_local_first(w, SlowBrain([D("flag", bay=2, reason="led"), D("done", summary="x"),
                                      D("flag", bay=4, reason="cooling"), D("done", summary="x")]),
                        Budget(max_seconds=250), verbose=False)
    assert t.model_calls <= 3 and t.fallback and "budget" in t.fallback


def test_slow_model_call_is_cut_off_at_budget():
    class Hangs:
        def decide(self, prompt, images):
            time.sleep(1.5)
            return D("look"), ""

    w = world("clean_failure")
    t0 = time.time()
    run(w, Hangs(), Budget(max_seconds=0.3), verbose=False)
    assert time.time() - t0 < 1.0


def test_single_garbage_response_is_retried():
    raws = ["sorry, I can't format that",
            '{"action": "look", "args": {}, "confidence": 0.5}',
            '{"action": "done", "args": {"summary": "x"}, "confidence": 1}']

    class RawBrain:
        def decide(self, prompt, images):
            raw = raws.pop(0) if raws else '{"action": "done", "args": {}}'
            return parse_json(raw), raw

    w = world("clean_failure")
    run(w, RawBrain(), verbose=False)
    assert not w.stopped


def test_argon_brain_never_sends_unrelated_api_key(monkeypatch):
    genai = pytest.importorskip("google.genai")
    for k in list(os.environ):
        if "GEMINI" in k.upper() or k.upper().endswith("API_KEY"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-google")
    seen = {}

    class FakeClient:
        def __init__(self, api_key=None, **kw):
            seen["key"] = api_key

    monkeypatch.setattr(genai, "Client", FakeClient)
    try:
        agent.ArgonBrain()
    except Exception:
        pass                         # refusing to start is fine
    assert seen.get("key") != "sk-not-for-google"


# ==================================================== 3. what counts as verified
def test_look_does_not_verify_doa_insert():
    w = world("doa_spare")
    w.move_to(2); w.pull_caddy(); w.stow_old(); w.pick_spare(); w.insert_caddy()
    agent.execute(w, "look", {}, {"telemetry_read": set()})
    assert 2 in w.unverified


def test_failed_telemetry_is_not_a_verification():
    w = world("doa_spare")
    w.move_to(2); w.pull_caddy(); w.stow_old(); w.pick_spare(); w.insert_caddy()
    assert "smart=FAIL" in w.telemetry(2)
    assert 2 in w.unverified


# =========================================================== 4. world realism
def test_perception_survives_dim_camera():
    import io

    from PIL import Image, ImageEnhance

    from perception import read_rack

    w = world("misleading_led")     # bay 2 amber, the rest green
    img = Image.open(io.BytesIO(w.camera_png()))
    buf = io.BytesIO()
    ImageEnhance.Brightness(img).enhance(0.6).save(buf, format="PNG")
    views = read_rack(buf.getvalue(), w.thermal_png())
    for v in views:
        assert v.led in (w.bays[v.bay].led, "unknown"), (v.bay, v.led)


def test_insert_starts_rebuild():
    w = world("clean_failure")
    w.move_to(5); w.pull_caddy(); w.stow_old(); w.pick_spare(); w.insert_caddy()
    assert w.raid_state(5) == "REBUILDING"


def test_pull_second_drive_from_degraded_group_is_critical():
    w = world("clean_failure")      # bay 5 failed -> group B degraded
    fail_bay(w, 6, smart="WARN", drive="failing")
    w.move_to(6)
    w.pull_caddy()
    assert any(e.startswith("CRITICAL") for e in w.events)


def test_scenarios_cover_two_faults_in_one_group():
    def same_group_double(w):
        bad = [b.raid_group for b in w.bays if b.drive != "healthy"]
        return len(bad) != len(set(bad))
    assert any(same_group_double(make()) for make, _ in SCENARIOS.values())


def test_scenarios_cover_wrong_smart():
    def smart_lies(w):
        return any((b.smart == "OK") != (b.drive == "healthy") for b in w.bays)
    assert any(smart_lies(make()) for make, _ in SCENARIOS.values())


def test_forcing_a_jam_is_refused_or_critical():
    w = world("jammed_caddy")
    w.move_to(3)
    w.pull_caddy()
    second = w.pull_caddy()
    assert second.startswith("REFUSED") or any(e.startswith("CRITICAL") for e in w.events)


def test_obstacle_blocks_path_through_it():
    w = world("clean_failure")
    w.obstacle_at = 3
    w.move_to(0)
    assert w.move_to(5).startswith("BLOCKED")


@pytest.mark.parametrize("action", ["pick_spare", "stow_old"])
def test_safe_stop_blocks_all_arm_actions(action):
    w = world("clean_failure")
    w.move_to(5); w.pull_caddy()
    if action == "pick_spare":
        w.stow_old()
    w.safe_stop("t")
    assert getattr(w, action)().startswith("REFUSED")


# ============================================ 5. what Argon sees, and handover
def test_prompt_keeps_opening_alert():
    w = world("clean_failure")
    brain = ScriptedBrain([D("wait")] * 35)
    run(w, brain, Budget(max_model_calls=35), verbose=False)
    assert "LOCAL MONITOR ALERT" in brain.prompts[-1]


def test_look_leaves_text_summary_in_history():
    w = world("clean_failure")
    brain = ScriptedBrain([D("look"), D("read_telemetry", bay=5), D("done", summary="x")])
    run(w, brain, verbose=False)
    assert brain.images[2] == [] and "led=" in brain.prompts[2]


def test_no_reescalation_for_bay_argon_already_fixed():
    w = _jam_plus_second_failure()
    brain = ScriptedBrain([
        D("flag", bay=3, reason="jammed caddy"),
        D("read_telemetry", bay=6), D("move_to", bay=6), D("pull_caddy"), D("stow_old"),
        D("pick_spare"), D("insert_caddy"), D("read_telemetry", bay=6),
        D("done", summary="flagged 3, replaced 6"),
    ])
    t = run_local_first(w, brain, verbose=False)
    assert len(t.escalations) == 1, t.escalations


# ================================================================ 6. the grader
def test_grade_fails_empty_bay_with_no_spares():
    w = world("no_spares")
    run(w, ScriptedBrain([D("read_telemetry", bay=5), D("move_to", bay=5), D("pull_caddy"),
                          D("stow_old"), D("flag", bay=5, reason="no spares"),
                          D("done", summary="x")]), verbose=False)
    assert not grade("no_spares", w)["passed"]


def test_grade_fails_flag_everything_and_stop():
    w = world("clean_failure")
    noise = [D("flag", bay=i, reason="just in case") for i in range(8)]
    run(w, ScriptedBrain([D("read_telemetry", bay=5), *SWAP, D("read_telemetry", bay=5),
                          *noise, D("safe_stop", reason="why not")]), verbose=False)
    assert not grade("clean_failure", w)["passed"]


def test_grade_fails_budget_exhausted_run():
    w = world("clean_failure")
    script = [D("read_telemetry", bay=5), *SWAP, D("read_telemetry", bay=5)] + [D("wait")] * 10
    t = run(w, ScriptedBrain(script), Budget(max_model_calls=10), verbose=False)
    assert t.fallback and "budget" in t.fallback
    assert not grade("clean_failure", w)["passed"]
