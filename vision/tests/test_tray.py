"""vision/tray.py on Emanak's simulator (dcsim_robot/), run in-process.

    python -m pytest -q vision/tests/test_tray.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

DCSIM = Path(__file__).resolve().parents[2] / "dcsim_robot"
pytestmark = pytest.mark.skipif(not (DCSIM / "dcsim" / "sim.py").is_file(), reason="dcsim_robot/ not in this checkout")

SERVER = "R1-05-U20"


@pytest.fixture(scope="module")
def sim():
    sys.path.insert(0, str(DCSIM))
    from dcsim.sim import DataCenterSim
    from dcsim.skills import Skills
    s = DataCenterSim()
    sk = Skills(s)
    s.reset(faults=[])
    assert sk.go_to(SERVER)["ok"] and sk.pull_tray_out()["ok"] and sk.view_tray()["ok"]
    s.skills = sk
    yield s
    s.clear_cable()


def frame(s):
    pos, rot = s.camera_pose("wrist")
    return {"rgb": s.render("wrist"), "depth": s.render("wrist", depth=True), "K": s.camera_intrinsics("wrist"),
            "cam_pos": pos, "cam_rot": rot}


def scan(s, **kw):
    from vision.tray import TrayPerception
    return TrayPerception(**kw).scan(frame(s), s.slot_leds(SERVER), s.slot_grasp_points(SERVER))


def test_reads_every_led_colour(sim):
    import mujoco
    want = {"A1": "green", "A2": "off", "A3": "amber", "A5": "red", "A6": "green", "A7": "blue", "A8": "green"}
    for slot, st in want.items():
        sim.set_led(SERVER, slot, st)
    sim.step(1)
    sim._set_led(sim._led_geom(SERVER, "A4"), "rebuild")   # a new part settling in: lime
    mujoco.mj_forward(sim.m, sim.d)
    sc = scan(sim)
    got = {k: v.led for k, v in sc.slots.items()}
    assert {k: got[k] for k in want} == want
    assert got["A4"] == "lime" and sc.slots["A4"].status == "verifying"      # never a fault
    assert all(got[f"F{i}"] == "green" for i in range(1, 6))
    assert [s.slot for s in sc.faults()] == ["A3", "A5"]
    for slot in want:
        sim.set_led(SERVER, slot, None)


def test_ignores_lights_that_are_not_slot_leds(sim):
    sc = scan(sim)
    assert set(sc.slots) == {f"A{i}" for i in range(1, 9)} | {f"F{i}" for i in range(1, 6)}
    assert any("aren't slot LEDs" in n for n in sc.notes)


def test_clean_tray_has_no_obstacles(sim):
    sc = scan(sim)
    assert sc.obstacles == [] and not any(s.blocked for s in sc.slots.values()), sc.summary()


def test_cable_over_a_slot_is_one_obstacle(sim):
    sim.drop_cable(SERVER, "A6")
    try:
        sc = scan(sim)
    finally:
        sim.clear_cable()
    assert len(sc.obstacles) == 1, sc.summary()
    assert "A6" in sc.obstacles[0].over_slots
    blocked = {k for k, v in sc.slots.items() if v.blocked}
    assert "A6" in blocked and not blocked & {"A1", "A2", "A3", "A4", "A5", "A7", "A8"}
    assert 0.04 < sc.obstacles[0].above_parts_m < 0.12


def test_empty_slot_reads_off(sim):
    sk = sim.skills
    sim.set_led(SERVER, "A2", None)
    assert sk.pick("A2")["ok"] and sk.discard()["ok"] and sk.view_tray()["ok"]
    sc = scan(sim)
    assert sc.slots["A2"].led == "off" and sc.slots["A2"].status == "off"


def test_fast_enough(sim):
    from vision.tray import TrayPerception
    tp = TrayPerception()
    f, leds, gp = frame(sim), sim.slot_leds(SERVER), sim.slot_grasp_points(SERVER)
    tp.scan(f, leds, gp)
    assert min(tp.scan(f, leds, gp).latency_ms for _ in range(5)) < 60
