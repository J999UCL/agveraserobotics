"""Perception tests on the stand-in MuJoCo rack (ground truth known exactly).

    python -m pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest

from perception import Perception
from perception.geometry import backproject, cv_pose_from_usd, project, transform
from perception.synthetic import RACK_X, RACK_Z0, build, render

CENTRE = [RACK_X, 0.0, RACK_Z0 + 0.08]
VIEWS = {
    "head_on": [RACK_X - 0.45, 0.0, RACK_Z0 + 0.08],
    "close": [RACK_X - 0.30, 0.05, RACK_Z0 + 0.10],
    "oblique": [RACK_X - 0.42, -0.22, RACK_Z0 + 0.22],
}


@pytest.fixture(scope="module")
def scene():
    return build()


@pytest.fixture(scope="module")
def results(scene):
    per = Perception(inventory=scene.inventory())
    out = {}
    for name, pos in VIEWS.items():
        cap = render(scene, pos, CENTRE)
        per.detect(cap)  # warm-up: first call pays one-off OpenCV start-up costs
        times = []
        for _ in range(5):
            dets = per.detect(cap)
            times.append(per.last_latency_ms)
        out[name] = (cap, dets, dets + per.find_missing(cap, scene.expected(), dets), float(min(times)))
    return out


def truth(scene):
    return {b.comp_id: b for b in scene.bays}


@pytest.mark.parametrize("view", list(VIEWS))
def test_finds_every_installed_drive(scene, results, view):
    _, dets, _, _ = results[view]
    installed = {b.comp_id for b in scene.bays if b.present}
    assert {d.id for d in dets} == installed


@pytest.mark.parametrize("view", list(VIEWS))
def test_positions_within_1cm(scene, results, view):
    _, dets, _, _ = results[view]
    gt = truth(scene)
    errs = [np.linalg.norm(np.array(d.xyz_base) - gt[d.id].tag_centre) for d in dets]
    assert max(errs) < 0.01, f"worst position error {max(errs) * 100:.2f} cm"


@pytest.mark.parametrize("view", list(VIEWS))
def test_status_flags(scene, results, view):
    _, _, allds, _ = results[view]
    status = {d.id: d.status for d in allds}
    for b in scene.bays:
        if not b.present:
            want = "missing"
        elif b.led == "red":
            want = "failed"
        elif b.temp_c >= 55:
            want = "hot"
        elif b.led == "amber":
            want = "warning"
        else:
            want = "ok"
        assert status.get(b.comp_id) == want, f"{b.comp_id}: got {status.get(b.comp_id)}, want {want}"


@pytest.mark.parametrize("view", list(VIEWS))
def test_temperatures_within_3C(scene, results, view):
    # neighbours' heat must not leak into a reading, even at an angle
    _, dets, _, _ = results[view]
    gt = truth(scene)
    for d in dets:
        assert abs(d.temp_c - gt[d.id].temp_c) < 3.0, f"{d.id}: read {d.temp_c} C, true {gt[d.id].temp_c} C"


@pytest.mark.parametrize("view", list(VIEWS))
def test_every_flag_has_a_reason(results, view):
    _, _, allds, _ = results[view]
    for d in allds:
        if d.status != "ok":
            assert d.reasons, f"{d.id} is {d.status} with no reason"


def test_normals_face_the_robot(results):
    _, dets, _, _ = results["head_on"]
    for d in dets:
        assert d.normal_base[0] < -0.9  # rack faces -x, towards the robot


def test_grasp_point_in_front_of_label(scene, results):
    _, dets, _, _ = results["head_on"]
    gt = truth(scene)
    for d in dets:
        assert d.grasp_xyz_base[0] == pytest.approx(gt[d.id].tag_centre[0] - 0.02, abs=0.01)


def test_within_edge_budget(results):
    # 640x480 frame, fastest of 5 (= compute cost without background load). Laptop must be well under the 100 ms robot budget,
    # leaving headroom for a slower onboard computer.
    assert max(r[3] for r in results.values()) < 60


def test_geometry_roundtrip():
    rng = np.random.default_rng(0)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]])
    from perception.synthetic import look_at
    T = look_at([0.1, 0.2, 1.0], [0.8, 0.0, 0.9])
    for _ in range(50):
        p = np.array([0.8, 0, 0.9]) + rng.normal(0, 0.1, 3)
        u, v, z = project(p, K, T)
        back = transform(T, backproject(u, v, z, K))
        assert np.allclose(back, p, atol=1e-9)


def test_usd_conversion_flips_view_axis():
    T_usd = np.eye(4)  # USD camera at origin looking down its -Z
    T_cv = cv_pose_from_usd(T_usd)
    assert np.allclose(T_cv[:3, 2], [0, 0, -1])  # OpenCV +Z (forward) = USD -Z
    assert np.allclose(T_cv[:3, 1], [0, -1, 0])  # OpenCV +Y (down) = USD -Y
