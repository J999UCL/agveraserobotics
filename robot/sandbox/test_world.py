"""World sanity checks, no model. Run: uv run python robot/sandbox/test_world.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from agent import Budget, guard, local_alert, run  # noqa: E402
from scenarios import SCENARIOS, grade  # noqa: E402


def swap(w, bay):
    assert w.move_to(bay).startswith("OK"), w.move_to(bay)
    assert w.pull_caddy().startswith("OK")
    w.stow_old(); w.pick_spare()
    assert w.insert_caddy().startswith("OK")
    w.telemetry(bay)  # verify the new drive


w = SCENARIOS["clean_failure"][0]()
assert local_alert(w) == [5]
swap(w, 5)
assert grade("clean_failure", w)["passed"]

w = SCENARIOS["misleading_led"][0]()
assert local_alert(w) == [2, 6]       # LED anomaly (2) seen by perception, real fault (6)
swap(w, 2)                            # wrong bay -> must fail grading
assert not grade("misleading_led", w)["passed"]

w = SCENARIOS["jammed_caddy"][0]()
w.move_to(3)
assert w.pull_caddy().startswith("JAMMED")

w = SCENARIOS["obstacle"][0]()
assert w.move_to(7).startswith("BLOCKED")
w.wait(); assert w.move_to(7).startswith("BLOCKED")
w.wait(); assert w.move_to(7).startswith("OK")

w = SCENARIOS["raid_rebuild"][0]()
assert "REBUILDING" in w.telemetry(1)
swap(w, 1)
assert not grade("raid_rebuild", w)["passed"]   # pulled during rebuild

w = SCENARIOS["clean_failure"][0]()
w.move_to(5)
assert guard(w, {"action": "pull_caddy", "confidence": 0.95}, {"telemetry_read": set()})
assert guard(w, {"action": "pull_caddy", "confidence": 0.5}, {"telemetry_read": {5}})
assert guard(w, {"action": "pull_caddy", "confidence": 0.9}, {"telemetry_read": {5}}) is None


class Dead:
    def decide(self, *a):
        raise ConnectionError("offline")


w = SCENARIOS["clean_failure"][0]()
t = run(w, Dead(), Budget(), verbose=False)
assert w.stopped and w.flags and "unreachable" in t.fallback

os.makedirs(os.path.join(os.path.dirname(__file__), "runs"), exist_ok=True)
w = SCENARIOS["misleading_led"][0]()
open(os.path.join(os.path.dirname(__file__), "runs", "_sample_camera.png"), "wb").write(w.camera_png())
open(os.path.join(os.path.dirname(__file__), "runs", "_sample_thermal.png"), "wb").write(w.thermal_png())
print("world tests passed")
