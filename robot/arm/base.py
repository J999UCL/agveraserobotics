"""Rail + lift positioning for the service arm (arm layer, step 1).

The arm rides a carriage on a rail along the aisle (left/right) and a lift on that
carriage (up/down). go_to(rack, u) lines the arm up with one server:

  * rail target  = the server's rack x + SIDE_OFFSET, so the arm stands beside the tray
                   and a fully extended tray slides past it instead of into it
  * lift target  = arm shoulder SHOULDER_ABOVE over the tray's handles (clamped to the stroke),
                   so the arm reaches down: from there one arm posture can pull the tray
                   all the way out (level with the handle it only manages ~24 cm)

Targets come from the loaded model's geometry, the motion is a fixed-speed minimum-jerk
profile, and nothing is random: the same calls on the same scene give the same motion.

Safety rules, checked before the base moves:
  * nothing held, and every tray closed (an open tray is in the carriage's path)
  * the arm is folded into its travel pose first
While moving, any contact between the robot and the world stops the base in place
(a reflex, reported as BLOCKED). Results are strings in the style of the rest of
the robot stack: OK / REFUSED / BLOCKED.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[2] / "sim" / "datacentre.xml"

N_RACKS = 4
SLOTS_PER_RACK = 20
SIDE_OFFSET = 0.50     # arm base stands this far to the +x side of the tray centre
SHOULDER = 0.333       # Panda shoulder (joint 2) height above its base
SHOULDER_ABOVE = 0.20  # shoulder this far above the handle
RAIL_SPEED = 0.5       # m/s peak, tune later
LIFT_SPEED = 0.3       # m/s peak, tune later
ARM_SPEED = 1.0        # rad/s peak while folding into the travel pose
TOLERANCE = 0.002      # m, base counts as arrived within this
SETTLE_S = 1.5         # extra time allowed to settle before reporting BLOCKED
TRAY_OPEN = 0.005      # m, a tray further out than this counts as open
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATORS = [f"actuator{i}" for i in range(1, 8)]
TUCK = np.array([0.0, -1.0, 0.0, -3.0, 0.0, 2.1, 0.7853])  # folded travel pose


def tray_name(rack: int, u: int) -> str:
    """rack 1-4, u = any U the 2U server occupies (1-40): 'r04_u21'."""
    if not 1 <= rack <= N_RACKS:
        raise ValueError(f"no rack {rack} (valid 1-{N_RACKS})")
    if not 1 <= u <= 2 * SLOTS_PER_RACK:
        raise ValueError(f"no U{u} (valid 1-{2 * SLOTS_PER_RACK})")
    return f"r{rack:02}_u{(u - 1) // 2 * 2 + 1:02}"


@dataclass(frozen=True)
class Target:
    rail: float        # rail_x joint position (m)
    lift: float        # lift_z joint position (m)
    handle: np.ndarray # world position of the closed tray's handle midpoint


def _min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


class Base:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData,
                 step: Callable[[], None] | None = None):
        self.m, self.d = m, d
        self.step = step or (lambda: mujoco.mj_step(m, d))
        self.holding: str | None = None   # set by grasping skills (later steps)
        self.rail_act = m.actuator("rail_x").id
        self.lift_act = m.actuator("lift_z").id
        self.arm_act = [m.actuator(a).id for a in ARM_ACTUATORS]
        self.rail_lo, self.rail_hi = m.jnt_range[m.joint("rail_x").id]
        self.lift_lo, self.lift_hi = m.jnt_range[m.joint("lift_z").id]
        # Arm base world position when rail and lift are at zero (fixed by the scene).
        mujoco.mj_kinematics(m, d)
        self.base0 = d.body("link0").xpos - [self.rail_q, 0.0, self.lift_q]
        cart = m.body("service_cart").id
        self.robot_bodies = {b for b in range(m.nbody) if self._under(b, cart)}
        self.carriage = {m.body(n).id for n in ("service_cart", "service_lift", "robot_mount")}
        self.reaching = {m.body(n).id for n in ("link2", "link3", "link4", "link5", "link6",
                                                 "link7", "hand", "left_finger", "right_finger")}
        self.trays = [m.joint(j).name for j in range(m.njnt)
                      if m.joint(j).name.endswith("_slide")]

    # ------------------------------------------------------------ geometry
    def _under(self, body: int, root: int) -> bool:
        while body and body != root:
            body = int(self.m.body_parentid[body])
        return body == root

    @property
    def rail_q(self) -> float:
        return float(self.d.joint("rail_x").qpos[0])

    @property
    def lift_q(self) -> float:
        return float(self.d.joint("lift_z").qpos[0])

    def target(self, rack: int, u: int) -> Target:
        """Rail and lift positions that line the arm up with server (rack, u)."""
        name = tray_name(rack, u)
        m = self.m
        rack_body = m.body(f"rack_{rack:02}").id
        tray = m.body(name).id
        # Racks are fixed world bodies, so their model positions are world positions.
        tray_origin = m.body_pos[rack_body] + m.body_pos[tray]
        left = m.geom_pos[m.geom(f"{name}_handle_-1").id]
        right = m.geom_pos[m.geom(f"{name}_handle_1").id]
        handle = tray_origin + (left + right) / 2
        rail = float(np.clip(tray_origin[0] + SIDE_OFFSET - self.base0[0], self.rail_lo, self.rail_hi))
        lift = float(np.clip(handle[2] + SHOULDER_ABOVE - SHOULDER - self.base0[2], self.lift_lo, self.lift_hi))
        return Target(rail, lift, handle)

    # -------------------------------------------------------------- checks
    def open_trays(self) -> list[str]:
        return [j for j in self.trays if self.d.joint(j).qpos[0] > TRAY_OPEN]

    def contact_problem(self, d: mujoco.MjData, i: int) -> str | None:
        """Contact i as a problem: the robot touching the world, or the reaching part of
        the arm touching the robot's own carriage or lift. None if it is neither."""
        c = d.contact[i]
        b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
        if (b1 in self.robot_bodies) != (b2 in self.robot_bodies):
            inside, other = (b1, c.geom2) if b1 in self.robot_bodies else (b2, c.geom1)
            return f"{self.m.body(inside).name} touched {self.m.geom(other).name or self.m.body(int(self.m.geom_bodyid[other])).name}"
        for a, b, g in ((b1, b2, c.geom2), (b2, b1, c.geom1)):
            if a in self.reaching and b in self.carriage:
                return f"{self.m.body(a).name} touched the robot's own {self.m.geom(g).name or self.m.body(b).name}"
        return None

    def external_contact(self, d: mujoco.MjData | None = None) -> str | None:
        """First contact between the robot and the world, or the arm and its own carriage."""
        d = d or self.d
        for i in range(d.ncon):
            problem = self.contact_problem(d, i)
            if problem:
                return problem
        return None

    def refusal(self) -> str | None:
        if self.holding:
            return f"REFUSED: holding {self.holding}; the base only moves with an empty gripper."
        opened = self.open_trays()
        if opened:
            mm = self.d.joint(opened[0]).qpos[0] * 1000
            return (f"REFUSED: tray {opened[0].removesuffix('_slide')} is open ({mm:.0f} mm); "
                    "close it before moving the base.")
        return None

    # -------------------------------------------------------------- motion
    def _arm_q(self) -> np.ndarray:
        return np.array([self.d.joint(j).qpos[0] for j in ARM_JOINTS])

    def tuck(self) -> str:
        """Fold the arm into the travel pose (base stays put)."""
        start = self._arm_q()
        dist = float(np.max(np.abs(TUCK - start)))
        if dist < 0.01:
            return "OK: arm folded"
        steps = self._steps(1.875 * dist / ARM_SPEED)
        for k in range(1, steps + 1):
            self.d.ctrl[self.arm_act] = start + (TUCK - start) * _min_jerk(k / steps)
            self.step()
            hit = self.external_contact()
            if hit:
                self.d.ctrl[self.arm_act] = self._arm_q()
                return f"BLOCKED: arm stopped while folding ({hit})."
        for _ in range(self._steps(SETTLE_S)):
            if np.max(np.abs(self._arm_q() - TUCK)) < 0.02:
                return "OK: arm folded"
            self.step()
        return f"BLOCKED: arm did not reach its travel pose (error {np.max(np.abs(self._arm_q() - TUCK)):.3f} rad)."

    def go_to(self, rack: int, u: int) -> str:
        try:
            t = self.target(rack, u)
        except ValueError as e:
            return f"ERROR: {e}"
        where = f"R{rack:02} U{u}"
        refusal = self.refusal()
        if refusal:
            return refusal
        folded = self.tuck()
        if not folded.startswith("OK"):
            return folded
        r0, l0 = float(self.d.ctrl[self.rail_act]), float(self.d.ctrl[self.lift_act])
        dr, dl = t.rail - r0, t.lift - l0
        steps = self._steps(1.875 * max(abs(dr) / RAIL_SPEED, abs(dl) / LIFT_SPEED))
        for k in range(1, steps + 1):
            s = _min_jerk(k / steps)
            self.d.ctrl[self.rail_act] = r0 + dr * s
            self.d.ctrl[self.lift_act] = l0 + dl * s
            self.step()
            hit = self.external_contact()
            if hit:
                return self._halt(f"BLOCKED: base stopped on the way to {where} ({hit}).")
        for _ in range(self._steps(SETTLE_S)):
            if self._arrived(t):
                return (f"OK: arm at {where} (rail {t.rail:.3f} m, lift {t.lift:.3f} m)")
            self.step()
        return self._halt(f"BLOCKED: base did not settle at {where} "
                          f"(off by {self._error(t) * 1000:.1f} mm).")

    def _halt(self, msg: str) -> str:
        """Reflex: hold the base where it is now."""
        self.d.ctrl[self.rail_act] = self.rail_q
        self.d.ctrl[self.lift_act] = self.lift_q
        return msg

    def _error(self, t: Target) -> float:
        return max(abs(self.rail_q - t.rail), abs(self.lift_q - t.lift))

    def _arrived(self, t: Target) -> bool:
        v = max(abs(self.d.joint("rail_x").qvel[0]), abs(self.d.joint("lift_z").qvel[0]))
        return self._error(t) < TOLERANCE and v < 0.005

    def _steps(self, seconds: float) -> int:
        return max(1, int(np.ceil(seconds / self.m.opt.timestep)))


def load() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """The data-centre scene at its home keyframe (no trimesh needed)."""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d
