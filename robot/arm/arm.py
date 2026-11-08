"""Arm motion and force reading (arm layer, step 2).

Motion
  * solve_ik(pos, rot)      joint angles that put the tool point at a pose (damped least
                            squares from a fixed seed, so the same target gives the same answer)
  * move_free(pos, rot)     joint-space move to that pose; the whole path is checked for
                            collisions before the arm moves, and a contact while moving stops it
  * move_linear(pos, rot)   straight-line tool move (re-solved every control step)
  * guarded_move(dir, ...)  straight-line move that stops as soon as the sensed contact force
                            along the motion passes a limit: "touch until it pushes back"
  * gripper(open/closed)

Joint position servos get a gravity feed-forward, so the arm does not sag off its targets.

Force reading
  The Panda measures torque in every joint. The torque that is not explained by the arm's own
  motors, gravity, friction and acceleration is the external load, and mapping it back through
  the Jacobian gives the force at the tool point:
      tau_ext = M(q) qacc - (tau_motor + tau_passive - tau_bias)
      F_tool  = (J J^T)^-1 J tau_ext
  In the sim these terms are exact; a real arm adds sensor noise and would filter them.

Results are OK / CONTACT / BLOCKED / ERROR strings like the rest of the robot stack.
"""

from __future__ import annotations

import mujoco
import numpy as np

from base import ARM_ACTUATORS, ARM_JOINTS, Base, _min_jerk

TCP_OFFSET = 0.1034          # tool point: between the fingertips, along the hand's z axis
LINEAR_SPEED = 0.10          # m/s peak for straight-line moves, tune later
TOUCH_SPEED = 0.01           # m/s for guarded moves; at 0.02 the impact overshoots a 5 N limit to ~19 N
JOINT_SPEED = 1.0            # rad/s peak for joint-space moves
POS_TOL = 0.002              # m
ROT_TOL = np.radians(1.0)
LIMIT_MARGIN = 0.05          # rad kept clear of every joint limit
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
HAND_BODIES = ("hand", "left_finger", "right_finger")


def rotation(z_axis, y_axis) -> np.ndarray:
    """Tool orientation from where the fingers point (z) and the finger-closing axis (y)."""
    z = np.asarray(z_axis, float); z /= np.linalg.norm(z)
    y = np.asarray(y_axis, float); y = y - z * (y @ z); y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def _rot_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """World-frame axis * angle taking `current` to `target`."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (target @ current.T).flatten())
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


class Arm:
    def __init__(self, base: Base):
        self.base = base
        self.m, self.d = m, d = base.m, base.d
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.dofs = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.act = np.array([m.actuator(a).id for a in ARM_ACTUATORS])
        self.kp = m.actuator_gainprm[self.act, 0].copy()
        lo, hi = m.jnt_range[[m.joint(j).id for j in ARM_JOINTS]].T
        self.lo, self.hi = lo + LIMIT_MARGIN, hi - LIMIT_MARGIN
        self.grip_act = m.actuator("actuator8").id
        self.hand = m.body("hand").id
        self.hand_bodies = {m.body(b).id for b in HAND_BODIES}
        self.scratch = mujoco.MjData(m)      # for IK and path checks; never stepped

    # ------------------------------------------------------------ kinematics
    def q(self) -> np.ndarray:
        return self.d.qpos[self.qadr].copy()

    def tcp(self, d: mujoco.MjData | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = d or self.d
        rot = d.xmat[self.hand].reshape(3, 3).copy()
        return d.xpos[self.hand] + rot[:, 2] * TCP_OFFSET, rot

    def _jacobian(self, d: mujoco.MjData) -> np.ndarray:
        """6x7 tool-point Jacobian (linear rows first) for the arm joints only."""
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        point, _ = self.tcp(d)
        mujoco.mj_jac(self.m, d, jp, jr, point, self.hand)
        return np.vstack([jp, jr])[:, self.dofs]

    def _set_scratch(self, q: np.ndarray):
        self.scratch.qpos[:] = self.d.qpos
        self.scratch.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.scratch)
        mujoco.mj_comPos(self.m, self.scratch)

    def solve_ik(self, pos, rot, seed=None, iters=300) -> np.ndarray | None:
        """Joint angles for tool pose (pos, rot), or None if it is out of reach."""
        q = np.clip(self.q() if seed is None else np.asarray(seed, float), self.lo, self.hi)
        rest = q.copy()
        for _ in range(iters):
            self._set_scratch(q)
            p, r = self.tcp(self.scratch)
            err = np.concatenate([pos - p, _rot_error(rot, r)])
            if np.linalg.norm(err[:3]) < POS_TOL / 4 and np.linalg.norm(err[3:]) < ROT_TOL / 4:
                return q
            J = self._jacobian(self.scratch)
            JJt = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            null = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            dq += null @ (0.1 * (rest - q))           # stay near the seed posture
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return None

    def path_collision(self, path: list[np.ndarray]) -> str | None:
        """Kinematic sweep: first robot-vs-world contact along a joint path, or None."""
        for q in path:
            self._set_scratch(q)
            mujoco.mj_collision(self.m, self.scratch)
            hit = self.base.external_contact(self.scratch)
            if hit:
                return hit
        return None

    # --------------------------------------------------------------- control
    def _command(self, q_target: np.ndarray):
        """Position servo target plus gravity/Coriolis feed-forward at the current state."""
        self.d.ctrl[self.act] = q_target + self.d.qfrc_bias[self.dofs] / self.kp

    def _step(self, q_target: np.ndarray):
        self._command(q_target)
        self.base.step()

    def _settle(self, q_target: np.ndarray, seconds: float = 1.0) -> bool:
        for _ in range(self.base._steps(seconds)):
            if np.max(np.abs(self.q() - q_target)) < 1e-3 and np.max(np.abs(self.d.qvel[self.dofs])) < 0.01:
                return True
            self._step(q_target)
        return False

    def _halt(self, msg: str) -> str:
        self._command(self.q())
        return msg

    def gripper(self, state) -> str:
        """'open', 'closed', or a finger opening in metres per finger (0-0.04)."""
        width = {"open": 0.04, "closed": None}.get(state, state)
        if width is None:
            self.d.ctrl[self.grip_act] = GRIP_CLOSED      # squeezes: aims past shut
        else:
            # Servo target = (gain*ctrl + bias0) / kp, so pick ctrl for the requested opening.
            gain = self.m.actuator_gainprm[self.grip_act, 0]
            b0, kp = self.m.actuator_biasprm[self.grip_act, 0], -self.m.actuator_biasprm[self.grip_act, 1]
            self.d.ctrl[self.grip_act] = float(np.clip((kp * np.clip(width, 0, 0.04) - b0) / gain, 0, GRIP_OPEN))
        q = self.q()
        for _ in range(self.base._steps(0.6)):
            self._step(q)
        return f"OK: gripper {state if isinstance(state, str) else f'{width * 1000:.0f} mm per finger'}"

    def finger_width(self) -> float:
        return float(self.d.joint("finger_joint1").qpos[0])

    def move_joints(self, q_goal: np.ndarray, check: bool = True) -> str:
        start = self.q()
        dist = float(np.max(np.abs(q_goal - start)))
        n = self.base._steps(max(1.875 * dist / JOINT_SPEED, 0.05))
        path = [start + (q_goal - start) * _min_jerk(k / n) for k in range(1, n + 1)]
        if check:
            hit = self.path_collision(path[::10] + [q_goal])
            if hit:
                return f"BLOCKED: planned path would collide ({hit}); arm not moved."
        for qk in path:
            self._step(qk)
            hit = self.base.external_contact()
            if hit:
                return self._halt(f"BLOCKED: arm stopped on contact ({hit}).")
        if not self._settle(q_goal):
            return self._halt("BLOCKED: arm did not settle on its target.")
        return "OK: arm moved"

    def seeds(self, pos) -> list[np.ndarray]:
        """Fixed list of starting postures facing `pos` (same order every time)."""
        base_rot = self.d.xmat[self.m.body("link0").id].reshape(3, 3)
        local = base_rot.T @ (np.asarray(pos) - self.d.xpos[self.m.body("link0").id])
        yaw = float(np.arctan2(local[1], local[0]))
        return [np.array([yaw, j2, 0.0, j4, 0.0, j6, j7])
                for j2 in (0.0, 0.5) for j4 in (-1.6, -2.2) for j6 in (1.6, 2.2)
                for j7 in (0.785, -0.785, 2.356)]

    def _margin(self, q: np.ndarray) -> float:
        """Distance to the nearest joint limit, as a fraction of that joint's range."""
        return float(np.min(np.minimum(q - self.lo, self.hi - q) / (self.hi - self.lo)))

    def plan_line(self, q0, p_from, p_to, rot, step: float = 0.01) -> list[np.ndarray] | None:
        """Joint path for a straight tool line, solved in small steps from q0 so it stays on
        one arm posture. None if any point is out of reach or the posture would jump."""
        p_from, p_to = np.asarray(p_from, float), np.asarray(p_to, float)
        n = max(1, int(np.ceil(np.linalg.norm(p_to - p_from) / step)))
        path, q = [], np.asarray(q0, float)
        for k in range(1, n + 1):
            q_next = self.solve_ik(p_from + (p_to - p_from) * k / n, rot, seed=q, iters=100)
            if q_next is None or np.max(np.abs(q_next - q)) > 0.3:
                return None
            path.append(q_next)
            q = q_next
        return path

    def best_ik(self, pos, rot, then=()) -> np.ndarray | None:
        """Solve from every seed; keep the solution furthest from any joint limit.

        `then` lists the straight-line moves that will follow, as tool positions in order
        (orientation unchanged). A seed only counts if all of them can be done from it
        without leaving its posture, and its margin is the worst along the whole chain."""
        pos = np.asarray(pos, float)
        best, best_margin = None, -1.0
        for seed in self.seeds(pos):
            q = self.solve_ik(pos, rot, seed)
            if q is None or self._margin(q) <= best_margin + 1e-9:
                continue
            margin, ok, qk, pk = self._margin(q), True, q, pos
            for nxt in then:
                path = self.plan_line(qk, pk, nxt, rot)
                if path is None:
                    ok = False
                    break
                margin = min(margin, min(self._margin(x) for x in path))
                qk, pk = path[-1], np.asarray(nxt, float)
            if ok and margin > best_margin + 1e-9:
                best, best_margin = q, margin
        return best

    def move_free(self, pos, rot, seed=None, then=()) -> str:
        pos = np.asarray(pos, float)
        q = self.best_ik(pos, rot, then) if seed is None else self.solve_ik(pos, rot, seed)
        if q is None:
            return "ERROR: pose out of reach"
        res = self.move_joints(q)
        return res if not res.startswith("OK") else self._report("OK: at pose", pos, rot)

    def _report(self, head: str, pos, rot) -> str:
        p, r = self.tcp()
        de = np.linalg.norm(p - pos) * 1000
        ang = np.degrees(np.linalg.norm(_rot_error(rot, r)))
        return f"{head} (off by {de:.1f} mm, {ang:.2f} deg)"

    def _linear(self, pos, rot, speed, stop=None):
        """Shared straight-line driver. `stop(k)` may end the move early with a message."""
        p0, r0 = self.tcp()
        pos = np.asarray(pos, float)
        dist = float(np.linalg.norm(pos - p0))
        n = self.base._steps(max(1.875 * dist / speed, 0.05))
        rot_axis = _rot_error(rot, r0)
        q = self.q()
        for k in range(1, n + 1):
            s = _min_jerk(k / n)
            rk = np.zeros(9)
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, rot_axis / (np.linalg.norm(rot_axis) or 1.0),
                                      np.linalg.norm(rot_axis) * s)
            mujoco.mju_quat2Mat(rk, quat)
            q_next = self.solve_ik(p0 + (pos - p0) * s, rk.reshape(3, 3) @ r0, seed=q, iters=30)
            if q_next is None:
                return self._halt("ERROR: straight line leaves the reachable space; arm stopped.")
            q = q_next
            self._step(q)
            if stop:
                msg = stop(k)
                if msg:
                    return self._halt(msg)
        self._settle(q)
        return None

    def move_linear(self, pos, rot, speed: float = LINEAR_SPEED) -> str:
        def contact(_):
            hit = self.base.external_contact()
            return f"BLOCKED: arm stopped on contact ({hit})." if hit else None
        msg = self._linear(pos, rot, speed, contact)
        return msg or self._report("OK: straight move done", pos, rot)

    def guarded_move(self, direction, distance: float, max_force: float,
                     speed: float = TOUCH_SPEED, also_stop=None) -> str:
        """Move the tool straight along `direction` up to `distance`, stopping when the
        surface pushes back harder than `max_force` (N), or when `also_stop()` returns a
        message (e.g. a held part slipping)."""
        u = np.asarray(direction, float); u /= np.linalg.norm(u)
        p0, r0 = self.tcp()

        def pushed(_):
            f = -self.force() @ u               # resistance along the motion
            if f > max_force:
                moved = (self.tcp()[0] - p0) @ u
                return f"CONTACT: stopped after {moved * 1000:.1f} mm at {f:.1f} N"
            return also_stop() if also_stop else None
        msg = self._linear(p0 + u * distance, r0, speed, pushed)
        return msg or f"OK: no contact within {distance * 1000:.0f} mm"

    # ----------------------------------------------------------------- force
    def joint_torque_external(self) -> np.ndarray:
        """External torque on the 7 arm joints, from motor torque and the arm's own dynamics."""
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        Mqacc = np.zeros(m.nv)
        mujoco.mj_mulM(m, d, Mqacc, d.qacc)
        internal = d.qfrc_actuator + d.qfrc_passive - d.qfrc_bias
        return (Mqacc - internal)[self.dofs]

    def wrench(self) -> np.ndarray:
        """External [force(3), torque(3)] at the tool point, world frame, acting on the robot."""
        J = self._jacobian(self.d)
        return np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), J @ self.joint_torque_external())

    def force(self) -> np.ndarray:
        return self.wrench()[:3]

    def contact_force_truth(self) -> np.ndarray:
        """Sum of the simulator's contact forces on the hand and fingers (for tests only)."""
        m, d = self.m, self.d
        total = np.zeros(3)
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in self.hand_bodies) == (b2 in self.hand_bodies):
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            world = c.frame.reshape(3, 3).T @ f6[:3]    # force of geom1 on geom2
            total += world if b2 in self.hand_bodies else -world
        return total
