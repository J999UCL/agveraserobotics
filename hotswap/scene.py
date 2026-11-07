"""Public scene API that perception and control code build on.

Usage (inside Isaac Sim's python.bat, AFTER creating SimulationApp):

    scene = HotSwapScene(simulation_app)
    scene.build()
    scene.play()
    while ...:
        scene.step()
        frames = scene.get_wrist_frames()   # rgb, depth, K, camera pose
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.objects import DistantLight, DomeLight
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.simulation_manager import SimulationManager

from .config import SceneCfg
from .geometry import BoxBatch, quat_to_mat
from .rack import (
    LED_STATES,
    BayInfo,
    build_filler_units,
    build_neighbor_rack,
    build_rack,
    build_room,
    build_storage_server,
    make_materials,
)
from .robot import HAND_FORWARD_QUAT, READY_Q, WRIST_CAM_PATH, build_cameras, build_cart, build_robot


@dataclass
class WristFrames:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, meters along the optical axis (inf = nothing hit)
    K: np.ndarray  # 3x3 pinhole intrinsics (OpenCV convention: u right, v down)
    cam_pos: np.ndarray  # (3,) camera position, robot base frame
    cam_rot: np.ndarray  # (3, 3) USD camera axes in base frame (camera looks down its -Z)


class HotSwapScene:
    def __init__(self, simulation_app, cfg: SceneCfg | None = None, device: str = "cpu") -> None:
        self.app = simulation_app
        self.cfg = cfg or SceneCfg()
        self.device = device
        self.robot = None
        self.cameras = None
        self.bays: dict[tuple[int, int], BayInfo] = {}
        self.materials = None
        self.led_states: dict[tuple[int, int], str] = {}

    # ------------------------------------------------------------------ build
    def build(self) -> None:
        cfg = self.cfg
        SimulationManager.setup_simulation(dt=cfg.physics_dt, device=self.device)

        DomeLight("/World/Lights/dome").set_intensities(450)
        DistantLight("/World/Lights/ceiling").set_intensities(1500)

        self.materials = make_materials()
        batch = BoxBatch()
        build_room(cfg, batch)
        for name, dy in (("RackLeft", 1), ("RackRight", -1)):
            build_neighbor_rack(cfg, batch, name, dy * (cfg.rack.outer_width + 0.01))
        build_rack(cfg, batch)
        build_filler_units(cfg, batch)
        self.bays = build_storage_server(cfg, batch)
        build_cart(cfg, batch)
        batch.build(self.materials)

        self.robot = build_robot()
        self.cameras = build_cameras(cfg)

        for bay in self.bays:
            self.set_led(bay, "fault" if bay == cfg.faulty_bay else "ok")

    def play(self) -> None:
        self.app.update()
        app_utils.play()
        self.app.update()
        self.reset_arm()

    def reset_arm(self) -> None:
        """Snap the arm to READY_Q (hand pointing at the rack), gripper open."""
        q = np.array([READY_Q], dtype=np.float32)
        self.robot.set_dof_positions(q)
        self.robot.set_dof_position_targets(q)
        self.step(2)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.app.update()

    # ------------------------------------------------------------- disturbances
    def set_led(self, bay: tuple[int, int], state: str) -> None:
        """state: 'ok' (green), 'fault' (amber), 'identify' (blue), 'off'."""
        if state not in LED_STATES:
            raise ValueError(f"LED state must be one of {LED_STATES}")
        self.materials.bind(self.bays[bay].led_path, f"led_{state}")
        self.led_states[bay] = state

    # --------------------------------------------------------------- sensors
    def get_wrist_frames(self) -> WristFrames | None:
        """Latest wrist RGB-D frame, or None until the renderer has produced one."""
        sensor = self.cameras.wrist
        rgb, _ = sensor.get_data("rgb")
        depth, _ = sensor.get_data("distance_to_image_plane")
        if rgb is None or depth is None:
            return None
        rgb = rgb.numpy()[..., :3].astype(np.uint8)
        depth = depth.numpy().astype(np.float32).reshape(rgb.shape[:2])
        pos, quat = XformPrim(WRIST_CAM_PATH).get_world_poses()
        return WristFrames(rgb, depth, self.intrinsics(), pos.numpy()[0], quat_to_mat(quat.numpy()[0]))

    def get_overview_rgb(self) -> np.ndarray | None:
        rgb, _ = self.cameras.overview.get_data("rgb")
        return None if rgb is None else rgb.numpy()[..., :3].astype(np.uint8)

    def intrinsics(self) -> np.ndarray:
        cc = self.cfg.camera
        h, w = cc.resolution
        fx = cc.focal_length_mm * w / cc.horizontal_aperture_mm
        return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])

    # ------------------------------------------------------------- robot help
    def camera_offset_world(self, hand_quat=HAND_FORWARD_QUAT) -> np.ndarray:
        """Where the wrist camera sits relative to panda_hand, in base frame, for a hand orientation."""
        return quat_to_mat(hand_quat) @ np.asarray(self.cfg.camera.hand_offset)

    def hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = self.robot.end_effector_link.get_world_poses()
        return pos.numpy()[0], quat.numpy()[0]

    def ik_step(self, position, orientation=HAND_FORWARD_QUAT) -> float:
        """One differential-IK step of panda_hand toward the pose. Returns position error (m)."""
        self.robot.set_end_effector_pose(position=np.asarray(position), orientation=np.asarray(orientation))
        return float(np.linalg.norm(self.hand_pose()[0] - np.asarray(position)))

    def move_hand_to(self, position, orientation=HAND_FORWARD_QUAT, tol=0.005, max_steps=600) -> bool:
        """Blocking helper for scripts/tests. The real controller should step its own FSM."""
        for _ in range(max_steps):
            err = self.ik_step(position, orientation)
            self.step()
            if err < tol:
                return True
        return False


def pixel_to_base(u: float, v: float, depth: float, frames: WristFrames) -> np.ndarray:
    """Unproject pixel (u, v) with depth (distance to image plane) to robot base frame."""
    K = frames.K
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    p_cam_usd = np.array([x, -y, -depth])  # OpenCV (x right, y down, z fwd) -> USD camera axes
    return frames.cam_rot @ p_cam_usd + frames.cam_pos


def base_to_pixel(p: np.ndarray, frames: WristFrames) -> tuple[float, float, float]:
    """Project a base-frame point into the wrist image. Returns (u, v, depth)."""
    q = frames.cam_rot.T @ (np.asarray(p) - frames.cam_pos)
    depth = -q[2]
    K = frames.K
    return K[0, 0] * q[0] / depth + K[0, 2], -K[1, 1] * q[1] / depth + K[1, 2], depth
