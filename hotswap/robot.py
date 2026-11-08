"""Franka on a maintenance cart, with a wrist RGB-D camera and a third-person camera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SceneCfg
from .geometry import BoxBatch, ensure_xform, look_at_quat

ROBOT_PATH = "/World/robot"
HAND_PATH = f"{ROBOT_PATH}/panda_hand"
WRIST_CAM_PATH = f"{HAND_PATH}/wrist_cam"
OVERVIEW_CAM_PATH = "/World/overview_cam"

# Hand orientation (wxyz) with the gripper pointing at the rack: hand +Z = world +X,
# hand +X = world up, fingers close along world Y. This is the orientation the Franka
# naturally reaches from READY_Q (found by probing forward kinematics).
HAND_FORWARD_QUAT = np.array([0.0, 0.70710678, 0.0, 0.70710678])

# "Ready" joint pose: hand pointing at the rack at roughly shoulder height.
# With q1=q3=q5=0 the hand pitch is -q2 + q4 + q6; pi/2 points it forward.
READY_Q = [0.0, -0.61, 0.0, -2.59, 0.0, 3.55, 0.785, 0.04, 0.04]  # hand at ~(0.38, 0, 0.65)

# Wrist camera orientation in the hand frame. Looks along hand +Z (same way as the
# fingers), image-up = hand +X (= world up when the hand points at the rack).
WRIST_CAM_QUAT = np.array([0.0, 0.70710678, 0.70710678, 0.0])

CART_BODY = (0.75, 0.42, 0.05)  # safety orange
CART_DARK = (0.08, 0.08, 0.09)


def build_cart(cfg: SceneCfg, batch: BoxBatch) -> None:
    c = cfg.cart
    fz = cfg.floor_z
    ensure_xform("/World/Cart")  # parent Xform so the whole cart can glide along the aisle
    body_h = c.height - c.wheel_radius * 2 - 0.02
    batch.add("/World/Cart/body", (c.center_x, 0, fz + 2 * c.wheel_radius + body_h / 2),
              (c.length, c.width, body_h), CART_BODY)
    batch.add("/World/Cart/top", (c.center_x, 0, -0.011), (c.length + 0.02, c.width + 0.02, 0.02), CART_DARK)
    batch.add("/World/Cart/handle", (c.center_x - c.length / 2 - 0.03, 0, 0.35), (0.03, c.width * 0.8, 0.03), CART_DARK)
    for side, sy in (("l", 1), ("r", -1)):
        batch.add(f"/World/Cart/handle_post_{side}", (c.center_x - c.length / 2 - 0.03, sy * c.width * 0.4, 0.17),
                  (0.03, 0.03, 0.36), CART_DARK)
    for i, (dx, dy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        batch.add(f"/World/Cart/wheel_{i}",
                  (c.center_x + dx * (c.length / 2 - 0.08), dy * (c.width / 2 - 0.06), fz + c.wheel_radius),
                  (2 * c.wheel_radius, 0.04, 2 * c.wheel_radius), CART_DARK)


@dataclass
class Cameras:
    wrist: object  # CameraSensor
    overview: object  # CameraSensor
    wrist_cam: object  # RtxCamera


def build_robot():
    """Spawn the Franka at the origin (on top of the cart)."""
    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.robot.experimental.manipulators.examples", True
    )
    from isaacsim.robot.experimental.manipulators.examples.franka.franka import Franka

    return Franka(robot_path=ROBOT_PATH, create_robot=True)


def build_cameras(cfg: SceneCfg) -> Cameras:
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

    cc = cfg.camera
    h, w = cc.resolution

    wrist = RtxCamera(WRIST_CAM_PATH, translations=np.array(cc.hand_offset), orientations=WRIST_CAM_QUAT)
    wrist.camera.set_focal_lengths(cc.focal_length_mm)
    wrist.camera.set_apertures(cc.horizontal_aperture_mm, cc.horizontal_aperture_mm * h / w)
    wrist.camera.set_clipping_ranges(cc.near, cc.far)
    wrist_sensor = CameraSensor(wrist, resolution=cc.resolution, annotators=["rgb", "distance_to_image_plane"])

    eye, target = np.array([-1.5, -2.3, 1.25]), np.array([0.55, 0.0, 0.35])
    over = RtxCamera(OVERVIEW_CAM_PATH, positions=eye, orientations=look_at_quat(eye, target))
    over.camera.set_focal_lengths(18.0)
    over.camera.set_apertures(20.955, 20.955 * 720 / 1280)
    over_sensor = CameraSensor(over, resolution=(720, 1280), annotators=["rgb"])
    return Cameras(wrist=wrist_sensor, overview=over_sensor, wrist_cam=wrist)
