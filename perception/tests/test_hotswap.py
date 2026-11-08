"""HotSwap perception tests on the MuJoCo stand-in server (ground truth known exactly).

    python -m pytest -q
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from perception.adapter import capture_from_wrist_frames, detect_axes
from perception.geometry import USD_TO_CV, backproject, transform
from perception.hotswap import HotSwapPerception, calibrate_tab_offset
from perception.synthetic import render
from perception.synthetic_hotswap import CENTRE_Z, FACE_X, FOVY_ISAAC, build, led_pos, tab_pos

LEDS = {b: "green" for b in range(12)} | {6: "amber", 9: "blue", 3: "off"}
TARGET = [FACE_X, 0.0, CENTRE_Z]
VIEWS = {
    "scan_pose": [FACE_X - 0.30, 0.0, CENTRE_Z],             # view_server(standoff=0.30)
    "offset": [FACE_X - 0.32, 0.04, CENTRE_Z + 0.03],
}


def snap(scene, pos):
    return render(scene, pos, TARGET, fovy=FOVY_ISAAC)


@pytest.fixture(scope="module")
def clear():
    return build(LEDS, obstacle=False)


@pytest.fixture(scope="module")
def cluttered():
    return build(LEDS, obstacle=True)


@pytest.fixture(scope="module")
def tab_offset(clear):
    # calibrate once from one view (debug API is allowed for this), test on others
    scan = HotSwapPerception().scan(snap(clear, [FACE_X - 0.29, -0.03, CENTRE_Z - 0.02]))
    return calibrate_tab_offset(scan, {b: tab_pos(b) for b in range(12)})


@pytest.mark.parametrize("view", list(VIEWS))
def test_bay_numbers_and_positions(clear, view):
    scan = HotSwapPerception().scan(snap(clear, VIEWS[view]))
    lit = [b for b in scan.bays if b.led != "off"]
    assert len(lit) == 11
    for b in lit:
        assert b.bay is not None
        err = np.linalg.norm(np.array(b.led_xyz_base) - led_pos(b.bay))
        assert err < 0.01, f"bay {b.bay}: LED position off by {err * 100:.1f} cm (wrong bay number?)"


@pytest.mark.parametrize("view", list(VIEWS))
def test_fault_identify_and_off(clear, view):
    scan = HotSwapPerception().scan(snap(clear, VIEWS[view]))
    assert [b.bay for b in scan.faults()] == [6]
    assert scan.bay(9).status == "identify"
    off = scan.bay(3)
    assert off is not None and off.led == "off" and off.status == "unknown"
    assert np.linalg.norm(np.array(off.led_xyz_base) - led_pos(3)) < 0.015


@pytest.mark.parametrize("view", list(VIEWS))
def test_grab_point_within_1cm(clear, tab_offset, view):
    scan = HotSwapPerception(tab_offset=tab_offset).scan(snap(clear, VIEWS[view]))
    fault = scan.faults()[0]
    err = np.linalg.norm(np.array(fault.tab_xyz_base) - tab_pos(6))
    assert err < 0.01, f"grab point off by {err * 100:.1f} cm"


@pytest.mark.parametrize("view", list(VIEWS))
def test_no_false_obstacles(clear, view):
    scan = HotSwapPerception().scan(snap(clear, VIEWS[view]))
    assert scan.obstacles == [], scan.summary()


@pytest.mark.parametrize("view", list(VIEWS))
def test_finds_dropped_cable(cluttered, view):
    scan = HotSwapPerception().scan(snap(cluttered, VIEWS[view]))
    assert len(scan.obstacles) == 1, scan.summary()
    o = scan.obstacles[0]
    assert o.clearance_m == pytest.approx(FACE_X - 0.47 + 0.007, abs=0.02)  # cable front surface
    assert 0.03 < o.nearest_xyz_base[1] < 0.08
    # the fault is still found with the cable in the way
    assert [b.bay for b in scan.faults()] == [6]
    # bays behind the cable are reported as hidden, not as dead LEDs; the real dark bay stays "off"
    assert scan.bay(3).led == "off"
    assert all(b.led in ("hidden",) for b in scan.bays if b.led not in ("green", "amber", "blue", "off"))


def test_within_edge_budget(cluttered):
    per = HotSwapPerception()
    cap = snap(cluttered, VIEWS["scan_pose"])
    per.scan(cap)
    times = [per.scan(cap).latency_ms for _ in range(5)]
    assert min(times) < 60  # fastest run = compute cost without background load


# ---------------------------------------------------------------- adapter for api.md WristFrames
def _fake_frames(axes: str):
    cap = snap(build(LEDS), VIEWS["scan_pose"])  # OpenCV-axes pose
    T_scene = cap.T_base_cam if axes == "opencv" else cap.T_base_cam @ np.linalg.inv(USD_TO_CV)
    frames = SimpleNamespace(rgb=cap.rgb, depth=cap.depth, K=cap.K, cam_pos=T_scene[:3, 3], cam_rot=T_scene[:3, :3])

    def pixel_to_base(u, v, depth, fr):  # what the scene's own helper computes
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = fr.cam_rot, fr.cam_pos
        if axes == "usd":
            T = T @ USD_TO_CV
        return transform(T, backproject(u, v, depth, fr.K))
    return cap, frames, pixel_to_base


@pytest.mark.parametrize("axes", ["opencv", "usd"])
def test_adapter_detects_camera_convention(axes):
    cap, frames, p2b = _fake_frames(axes)
    assert detect_axes(frames, p2b) == axes
    got = capture_from_wrist_frames(frames, pixel_to_base=p2b)
    assert np.allclose(got.T_base_cam, cap.T_base_cam, atol=1e-6)
    assert [b.bay for b in HotSwapPerception().scan(got).faults()] == [6]
