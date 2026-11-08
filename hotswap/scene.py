"""Public environment API (see docs/api.md).

Usage (inside Isaac Sim's Python, AFTER creating SimulationApp):

    scene = HotSwapScene(simulation_app)
    scene.build()
    scene.play()
    scene.go_to_cabinet(1)
    scene.view_rack()
    frames = scene.get_wrist_frames()

Frames: every position this class takes or returns is in the ROBOT BASE frame
(origin = Franka base, +X toward the rack, +Y left, +Z up). The cart only moves along
world Y, so world = base + (0, base_y, 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.objects import DistantLight, DomeLight
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.simulation_manager import SimulationManager

from .config import SceneCfg
from .geometry import BoxBatch, quat_to_mat
from .obstacle import CableObstacle
from .physics import CaddySystem
from .rack import (
    LED_STATES,
    BayInfo,
    build_end_rack,
    build_filler_units,
    build_rack,
    build_room,
    build_storage_server,
    cabinet_root,
    make_materials,
)
from .robot import HAND_FORWARD_QUAT, READY_Q, WRIST_CAM_PATH, build_cameras, build_cart, build_robot

FINGER_REACH = 0.1034  # panda_hand origin -> fingertip center, along hand +Z


@dataclass
class WristFrames:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, meters along the optical axis (inf = nothing hit)
    K: np.ndarray  # 3x3 pinhole intrinsics (OpenCV convention: u right, v down)
    cam_pos: np.ndarray  # (3,) camera position, robot base frame
    cam_rot: np.ndarray  # (3, 3) USD camera axes in base frame (camera looks down its -Z)


class Stopped(Exception):
    """Raised inside blocking motions when stop() was called."""


class HotSwapScene:
    def __init__(self, simulation_app, cfg: SceneCfg | None = None, device: str = "cpu") -> None:
        self.app = simulation_app
        self.cfg = cfg or SceneCfg()
        self.device = device
        self.robot = None
        self.cameras = None
        self.materials = None
        self.caddies: CaddySystem | None = None
        self.obstacle: CableObstacle | None = None
        self.bays: dict[tuple[int, int], BayInfo] = {}
        self.base_y = self.cfg.cabinet_y(self.cfg.aisle.start_cabinet)
        self._cabinet: int | None = self.cfg.aisle.start_cabinet
        self._alerts: list[dict] = []
        self._grip_closed = False
        self._stop = False
        self._t = 0.0
        self.step_hooks: list[Callable[[], None]] = []  # e.g. the TCP server polls here

    # ================================================================ build
    def build(self) -> None:
        cfg = self.cfg
        SimulationManager.setup_simulation(dt=cfg.physics_dt, device=self.device)
        DomeLight("/World/Lights/dome").set_intensities(450)
        DistantLight("/World/Lights/ceiling").set_intensities(1500)

        self.materials = make_materials()
        batch = BoxBatch()
        build_room(cfg, batch)
        n = cfg.aisle.n_cabinets
        build_end_rack(cfg, batch, "EndRackLeft", cfg.cabinet_y(-1))
        build_end_rack(cfg, batch, "EndRackRight", cfg.cabinet_y(n))
        bays: list[BayInfo] = []
        for c in range(n):
            build_rack(cfg, batch, c)
            build_filler_units(cfg, batch, c)
            bays += build_storage_server(cfg, batch, c)
        build_cart(cfg, batch)
        batch.build(self.materials)
        self.bays = {(b.cabinet, b.bay): b for b in bays}

        self.caddies = CaddySystem(bays, cfg.caddy_physics)
        self.caddies.setup([f"{cabinet_root(c)}/StorageServer/Shell" for c in range(n)])
        self.obstacle = CableObstacle()
        self.robot = build_robot()
        self.cameras = build_cameras(cfg)
        self._cart = XformPrim("/World/Cart", reset_xform_op_properties=True)
        self._place_base(self.base_y)
        self._reset_leds()

    def play(self) -> None:
        self.app.update()
        app_utils.play()
        self.app.update()
        self.reset_arm()

    def step(self, n: int = 1) -> None:
        dt = self.cfg.physics_dt
        for _ in range(n):
            self._t += dt
            self.obstacle.update(dt)
            self.app.update()
            self.caddies.update()
            for hook in self.step_hooks:
                hook()

    def sim_time(self) -> float:
        return self._t

    def reset(self, faulty: tuple[int, int] | None = None, seed: int | None = None) -> dict:
        """Everything back to the start. `seed` picks a random faulty bay."""
        if seed is not None:
            rng = np.random.default_rng(seed)
            faulty = (int(rng.integers(self.cfg.aisle.n_cabinets)), int(rng.integers(self.cfg.n_bays)))
        self.obstacle.clear()
        self.caddies.reset()
        self._alerts.clear()
        self._reset_leds(faulty)
        self._place_base(self.cfg.cabinet_y(self.cfg.aisle.start_cabinet))
        self._cabinet = self.cfg.aisle.start_cabinet
        self.reset_arm()
        return {"faulty": list(faulty or self.cfg.faulty)}

    # ============================================================ frames
    def to_world(self, p) -> np.ndarray:
        return np.asarray(p, dtype=float) + np.array([0.0, self.base_y, 0.0])

    def to_base(self, p) -> np.ndarray:
        return np.asarray(p, dtype=float) - np.array([0.0, self.base_y, 0.0])

    # ============================================================ cart
    def _place_base(self, y: float) -> None:
        self.base_y = float(y)
        self.robot.set_world_poses(positions=np.array([[0.0, y, 0.0]]))
        self._cart.set_world_poses(positions=np.array([[0.0, y, 0.0]]))

    def current_cabinet(self) -> int | None:
        return self._cabinet

    def go_to_cabinet(self, cabinet: int) -> bool:
        """Stow the arm, then glide the cart to `cabinet`. Blocking."""
        if not 0 <= cabinet < self.cfg.aisle.n_cabinets:
            raise ValueError(f"cabinet must be 0..{self.cfg.aisle.n_cabinets - 1}")
        self.stow_arm()
        y0, y1 = self.base_y, self.cfg.cabinet_y(cabinet)
        if abs(y1 - y0) < 1e-4:
            self._cabinet = cabinet
            return True
        self._cabinet = None
        hops = abs(y1 - y0) / self.cfg.cabinet_pitch
        steps = max(1, int(self.cfg.aisle.move_duration * hops / self.cfg.physics_dt))
        q = self.joint_positions()
        for i in range(1, steps + 1):
            a = i / steps
            a = a * a * (3 - 2 * a)  # smoothstep
            self._place_base(y0 + (y1 - y0) * a)
            self.robot.set_dof_position_targets(np.array([q], dtype=np.float32))
            self._step_checked()
        self._cabinet = cabinet
        self.step(10)
        return True

    # ============================================================ arm
    def joint_positions(self) -> np.ndarray:
        return self.robot.get_dof_positions().numpy()[0]

    def hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = self.robot.end_effector_link.get_world_poses()
        return self.to_base(pos.numpy()[0]), quat.numpy()[0]

    def fingertip_position(self) -> np.ndarray:
        pos, quat = self.hand_pose()
        return pos + quat_to_mat(quat) @ np.array([0, 0, FINGER_REACH])

    def ik_step(self, position, orientation=HAND_FORWARD_QUAT) -> float:
        """One differential-IK step of panda_hand toward the pose. Returns position error (m)."""
        self.robot.set_end_effector_pose(position=self.to_world(position), orientation=np.asarray(orientation))
        return float(np.linalg.norm(self.hand_pose()[0] - np.asarray(position)))

    def move_hand_to(self, position, orientation=HAND_FORWARD_QUAT, tol=0.004, max_steps=600,
                     max_speed: float = 0.25) -> bool:
        """Blocking straight-line move of panda_hand (speed-limited, m/s). Returns True if reached."""
        target = np.asarray(position, dtype=float)
        start = self.hand_pose()[0]
        dist = float(np.linalg.norm(target - start))
        n_interp = max(1, int(dist / (max_speed * self.cfg.physics_dt)))
        for i in range(max_steps):
            a = min(1.0, (i + 1) / n_interp)
            err = self.ik_step(start + (target - start) * a, orientation)
            self._step_checked()
            if a >= 1.0 and np.linalg.norm(self.hand_pose()[0] - target) < tol:
                return True
        return False

    def move_joints(self, q7, duration: float = 1.5) -> bool:
        q0 = self.joint_positions()
        q1 = q0.copy()
        q1[:7] = np.asarray(q7, dtype=float)
        steps = max(1, int(duration / self.cfg.physics_dt))
        for i in range(1, steps + 1):
            a = i / steps
            a = a * a * (3 - 2 * a)
            self.robot.set_dof_position_targets(np.array([q0 + (q1 - q0) * a], dtype=np.float32))
            self._step_checked()
        self.step(15)
        return bool(np.abs(self.joint_positions()[:7] - q1[:7]).max() < 0.02)

    def reset_arm(self) -> None:
        """Snap (teleport) the arm to READY_Q, gripper open."""
        q = np.array([READY_Q], dtype=np.float32)
        self.robot.set_dof_positions(q)
        self.robot.set_dof_position_targets(q)
        self._grip_closed = False
        self.step(2)

    def stow_arm(self) -> bool:
        """Smoothly return to the ready pose (hand pointing at the rack, clear of it)."""
        return self.move_joints(READY_Q[:7])

    def view_rack(self, standoff: float = 0.30) -> bool:
        """Camera square in front of the storage server, ~standoff m in front of the hand."""
        server_z = self._server_center_z()
        cam_dz = (quat_to_mat(HAND_FORWARD_QUAT) @ np.asarray(self.cfg.camera.hand_offset))[2]
        return self.move_hand_to([standoff, 0.0, server_z - cam_dz])

    def _server_center_z(self) -> float:
        zs = [b.face_center[2] for (c, _), b in self.bays.items() if c == 0]
        return float(np.mean(zs))

    # ------------------------------------------------------------ gripper
    def gripper_move(self, width: float) -> None:
        w = float(np.clip(width, 0.0, 0.08)) / 2
        self.robot.set_dof_position_targets(np.array([[w, w]], dtype=np.float32), dof_indices=[7, 8])
        self._grip_closed = width < 0.07
        self.step(30)

    def open_gripper(self) -> None:
        self.gripper_move(0.08)

    def close_gripper(self) -> None:
        self.gripper_move(0.0)

    def gripper_width(self) -> float:
        return float(self.joint_positions()[7:9].sum())

    def is_grasping(self) -> bool:
        """Fingers commanded closed but stopped on something."""
        return self._grip_closed and self.gripper_width() > 0.004

    # ------------------------------------------------------------ stop
    def stop(self) -> None:
        """Abort the current blocking motion and hold position."""
        self._stop = True
        q = self.joint_positions()
        self.robot.set_dof_position_targets(np.array([q], dtype=np.float32))

    def _step_checked(self) -> None:
        if self._stop:
            self._stop = False
            raise Stopped("motion stopped")
        self.step()

    # ============================================================ sensing
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
        return WristFrames(rgb, depth, self.intrinsics(), self.to_base(pos.numpy()[0]), quat_to_mat(quat.numpy()[0]))

    def wait_for_frames(self, max_steps: int = 120) -> WristFrames | None:
        for _ in range(max_steps):
            f = self.get_wrist_frames()
            if f is not None and f.rgb.any():
                return f
            self.step()
        return None

    def get_overview_rgb(self) -> np.ndarray | None:
        rgb, _ = self.cameras.overview.get_data("rgb")
        return None if rgb is None else rgb.numpy()[..., :3].astype(np.uint8)

    def intrinsics(self) -> np.ndarray:
        cc = self.cfg.camera
        h, w = cc.resolution
        fx = cc.focal_length_mm * w / cc.horizontal_aperture_mm
        return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])

    # ============================================================ caddies
    def get_pull_force(self) -> float:
        """Newtons resisting extraction on whichever caddy is being pulled (0 if none)."""
        return self.caddies.max_force()

    def caddy_extraction(self, cabinet: int, bay: int) -> float:
        """Meters a caddy has slid out (0 = seated). Released caddies report >= 0.15."""
        s = self.caddies.state(cabinet, bay)
        return max(0.0, -s.x) if not s.released else max(self.cfg.caddy_physics.release_at, -s.x)

    def caddy_status(self, cabinet: int, bay: int) -> dict:
        s = self.caddies.state(cabinet, bay)
        return {"extraction_m": round(self.caddy_extraction(cabinet, bay), 4), "force_n": round(s.force, 2),
                "jammed": s.jammed, "released": s.released, "connector_broken": s.broken}

    # ============================================================ triggers
    def set_led(self, cabinet: int, bay: int, state: str) -> None:
        if state not in LED_STATES:
            raise ValueError(f"LED state must be one of {LED_STATES}")
        self.materials.bind(self.bays[(cabinet, bay)].led_path, f"led_{state}")

    def _reset_leds(self, faulty: tuple[int, int] | None = None) -> None:
        for (c, b) in self.bays:
            self.set_led(c, b, "ok")
        self.set_fault(*(faulty or self.cfg.faulty))

    def set_fault(self, cabinet: int, bay: int) -> None:
        self.set_led(cabinet, bay, "fault")
        self._alerts.append({"cabinet": cabinet, "bay": bay, "server": "storage-01",
                             "status": "drive_failure", "detail": "SMART predictive failure, drive offline"})

    def get_telemetry(self) -> list[dict]:
        """Monitoring alerts (logical address only, no coordinates) + physical events."""
        return list(self._alerts) + list(self.caddies.events)

    def set_jam(self, cabinet: int, bay: int, on: bool = True) -> None:
        self.caddies.set_jam(cabinet, bay, on)

    def connector_broken(self) -> bool:
        return any(s.broken for s in self.caddies.states)

    def drop_obstacle(self, position=None, delay: float = 0.0) -> list[float]:
        """Drop a loose cable into the gripper's path (or at `position`, base frame). Returns where."""
        if position is None:
            tip = self.fingertip_position()
            hand = self.hand_pose()[0]
            rack_face = self.cfg.rack.front_x - 0.03  # just in front of the pull tabs
            x = tip[0] + 0.5 * max(0.0, rack_face - tip[0])  # halfway between fingers and rack
            position = [x, hand[1], hand[2] + 0.05]
        self.obstacle.drop(self.to_world(position), delay)
        return [round(float(v), 3) for v in position]

    def clear_obstacle(self) -> None:
        self.obstacle.clear()

    # ============================================================ debug
    def debug_bay_pose(self, cabinet: int, bay: int) -> dict:
        """Ground truth for testing ONLY. The real pipeline must find bays with the camera."""
        b = self.bays[(cabinet, bay)]
        shift = np.array([self.caddies.state(cabinet, bay).x, 0, 0])
        return {"tab_center": self.to_base(b.tab_center + shift).round(4).tolist(),
                "face_center": self.to_base(b.face_center + shift).round(4).tolist(),
                "row": b.row, "col": b.col}


def pixel_to_base(u: float, v: float, depth: float, frames: WristFrames) -> np.ndarray:
    """Unproject pixel (u, v) with depth (distance to image plane) to robot base frame."""
    K = frames.K
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return frames.cam_rot @ np.array([x, -y, -depth]) + frames.cam_pos


def base_to_pixel(p, frames: WristFrames) -> tuple[float, float, float]:
    """Project a base-frame point into the wrist image. Returns (u, v, depth)."""
    q = frames.cam_rot.T @ (np.asarray(p) - frames.cam_pos)
    depth = -q[2]
    K = frames.K
    return K[0, 0] * q[0] / depth + K[0, 2], -K[1, 1] * q[1] / depth + K[1, 2], depth


__all__ = ["HotSwapScene", "WristFrames", "Stopped", "pixel_to_base", "base_to_pixel"]
