"""Damped-least-squares IK for the Franka TCP (site "tcp", between the fingertips)."""

from __future__ import annotations

import mujoco
import numpy as np

ARM_DOF = 7
# Nominal posture the solver drifts toward in the null space (elbow up, wrist down).
Q_NOMINAL = np.array([0.0, 0.1, 0.0, -2.2, 0.0, 2.3, 0.785])

# Gripper pointing straight down. "fingers_y": fingers close along world Y (RAM, fan tabs).
# "fingers_x": fingers close along world X (the tray handle).
QUAT_DOWN_FINGERS_Y = np.array([0.0, 0.0, 1.0, 0.0])
QUAT_DOWN_FINGERS_X = np.array([0.0, 0.70710678, 0.70710678, 0.0])


def _quat_from_axes(z_hand, y_hand) -> np.ndarray:
    z = np.asarray(z_hand, float) / np.linalg.norm(z_hand)
    y = np.asarray(y_hand, float)
    y = y - z * np.dot(y, z)
    y /= np.linalg.norm(y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([np.cross(y, z), y, z]).flatten())
    return q


# Gripper pointing at the rack, tilted 75 deg down, fingers closing along Y: used for the
# tray's pull tab. (Pointing level at the rack is out of the Franka's reach at tray height.)
FRONT_PITCH_DEG = 75.0  # steepest tilt that covers the whole 32 cm pull (60-65 run out of reach near the base)
FRONT_APPROACH = np.array([np.cos(np.radians(FRONT_PITCH_DEG)), 0.0, -np.sin(np.radians(FRONT_PITCH_DEG))])
QUAT_FRONT = _quat_from_axes(FRONT_APPROACH, [0, 1, 0])


def quat_err(q_des: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """Rotation vector taking q_cur to q_des (world frame)."""
    neg = np.zeros(4)
    mujoco.mju_negQuat(neg, q_cur)
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, q_des, neg)
    if dq[0] < 0:
        dq = -dq
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, dq, 1.0)
    return v


class IK:
    def __init__(self, model: mujoco.MjModel, site: str = "tcp") -> None:
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site(site).id
        self.lo = model.jnt_range[:ARM_DOF, 0].copy()
        self.hi = model.jnt_range[:ARM_DOF, 1].copy()
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)  # needed by mj_jacSite
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.d.site_xmat[self.site])
        return self.d.site_xpos[self.site].copy(), quat

    def solve(self, pos, quat, q_init, iters: int = 300, tol_p: float = 5e-4, tol_r: float = 5e-3,
              damping: float = 0.03, null_gain: float = 0.05) -> tuple[np.ndarray, bool, float]:
        q = np.clip(np.asarray(q_init, dtype=float)[:ARM_DOF].copy(), self.lo, self.hi)
        pos, quat = np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
        err_p = np.inf
        for _ in range(iters):
            p, qc = self.fk(q)
            err_p_v = pos - p
            err_r_v = quat_err(quat, qc)
            err_p = float(np.linalg.norm(err_p_v))
            if err_p < tol_p and np.linalg.norm(err_r_v) < tol_r:
                return q, True, err_p
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp[:, :ARM_DOF], self.jacr[:, :ARM_DOF]])
            e = np.concatenate([err_p_v, err_r_v])
            JJt = J @ J.T + damping**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            # null-space pull toward a comfortable posture
            N = np.eye(ARM_DOF) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (null_gain * (Q_NOMINAL - q))
            step = np.linalg.norm(dq)
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, self.lo + 1e-3, self.hi - 1e-3)
        p, qc = self.fk(q)
        err_p = float(np.linalg.norm(pos - p))
        ok = err_p < 2e-3 and np.linalg.norm(quat_err(quat, qc)) < 2e-2
        return q, ok, err_p
