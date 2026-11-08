"""Tray skills: open and close a server tray with the arm (arm layer, step 3).

open_tray(rack, u)
  line the base up -> open the fingers to 26 mm (the handle is 14 mm, the rack wall 45 mm away)
  -> pick ONE arm posture that can do the whole sequence (reach, grip, pull 70 cm, back off),
     checked for collisions with the tray moving along -> reach 10 cm in front of the right
     handle -> straight in -> grip -> pull straight out, stopping if the tray resists more
     than PULL_LIMIT (a jam: push it back closed, let go, report JAMMED) -> let go -> back off
close_tray(rack, u)
  the mirror image: grip the extended handle, push in until the tray hits its closed stop
  (the push-back spike), let go, back off, fold the arm.

Every step is a fixed rule on the current scene state; nothing is random or learned.
Results are OK / JAMMED / BLOCKED / REFUSED / ERROR strings like the rest of the robot stack.
"""

from __future__ import annotations

import mujoco
import numpy as np

from arm import Arm, rotation
from base import TRAY_OPEN, Base, tray_name

FACING_RACK = rotation([0, 1, 0], [1, 0, 0])   # fingers point into the rack, close sideways
OPEN = 0.70             # m the tray is pulled out (rail travel is 0.72)
FINGER_GAP = 0.013      # m per finger when straddling the handle
APPROACH = 0.10         # m in front of the handle before going straight in
RETREAT = 0.04          # m backed off the handle after letting go
PULL_SPEED = 0.10       # m/s peak while moving a tray
PULL_LIMIT = 30.0       # N resistance that counts as a jam (a free tray needs ~5 N)
PUSH_LIMIT = 30.0       # N push-back that means the tray has reached its closed stop
HELD = (0.0007, 0.012)  # finger joint reading (m) with the 14 mm handle gripped (~1.4 mm;
                        # the finger shells are thicker than the pads), and both fingers touching it
MAX_SLIP = 0.010        # m the hand may move relative to the handle while pulling
Y = np.array([0.0, 1.0, 0.0])


class TrayService:
    def __init__(self, base: Base, arm: Arm):
        self.base, self.arm = base, arm
        self.m, self.d = base.m, base.d

    # ---------------------------------------------------------------- state
    def _tray_q(self, name: str) -> float:
        return float(self.d.joint(f"{name}_slide").qpos[0])

    def _handle(self, name: str) -> np.ndarray:
        return self.d.geom(f"{name}_handle_1").xpos.copy()

    def _at(self, rack: int, u: int) -> bool:
        t = self.base.target(rack, u)
        return abs(self.base.rail_q - t.rail) < 0.003 and abs(self.base.lift_q - t.lift) < 0.003

    def _sweep(self, name: str, path: list[np.ndarray], tray: list[float]) -> str | None:
        """Collision check of an arm path with the tray at the matching positions. The hand
        touching this tray's handle is expected; anything else is a collision."""
        m, s = self.m, self.arm.scratch
        tray_body = m.body(name).id
        jadr = m.jnt_qposadr[m.joint(f"{name}_slide").id]
        for q, tq in zip(path, tray):
            s.qpos[:] = self.d.qpos
            s.qpos[self.arm.qadr] = q
            s.qpos[jadr] = tq
            mujoco.mj_kinematics(m, s)
            mujoco.mj_collision(m, s)
            for i in range(s.ncon):
                c = s.contact[i]
                pair = {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])}
                if tray_body in pair and pair & self.arm.hand_bodies:
                    continue                      # the hand on this tray's handle is expected
                problem = self.base.contact_problem(s, i)
                if problem:
                    return problem
        return None

    def _plan(self, name: str, start, chain, tray_at) -> tuple[np.ndarray | None, str | None]:
        """Arm posture for `start` that can follow the straight lines in `chain`, and a
        collision check of the whole sequence. tray_at(i, s) gives the tray position on
        segment i at fraction s."""
        q0 = self.arm.best_ik(start, FACING_RACK, then=chain)
        if q0 is None:
            return None, "ERROR: no single arm posture can do this whole sequence"
        q, p = q0, np.asarray(start, float)
        for i, nxt in enumerate(chain):
            path = self.arm.plan_line(q, p, nxt, FACING_RACK)
            tray = [tray_at(i, (k + 1) / len(path)) for k in range(len(path))]
            hit = self._sweep(name, path, tray)
            if hit:
                return None, f"BLOCKED: planned motion would collide ({hit}); arm not moved"
            q, p = path[-1], np.asarray(nxt, float)
        return q0, None

    def _fingers_on(self, geom: str) -> set[str]:
        """Which fingers are touching the given geom (from contact sensing)."""
        m, d, g = self.m, self.d, self.m.geom(geom).id
        on = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if g in (c.geom1, c.geom2):
                other = int(m.geom_bodyid[c.geom2 if c.geom1 == g else c.geom1])
                if m.body(other).name in ("left_finger", "right_finger"):
                    on.add(m.body(other).name)
        return on

    def _grip(self, name: str) -> str | None:
        self.arm.gripper("closed")
        w = self.arm.finger_width()
        if not (HELD[0] < w < HELD[1] and len(self._fingers_on(f"{name}_handle_1")) == 2):
            self.arm.gripper(FINGER_GAP)
            return f"ERROR: missed the handle (fingers closed to {w * 1000:.1f} mm)"
        self.base.holding = f"{name} handle"
        return None

    def _let_go_and_back_off(self) -> str:
        self.arm.gripper(FINGER_GAP)
        self.base.holding = None
        return self.arm.guarded_move(-Y, RETREAT, max_force=5.0, speed=0.05)

    # --------------------------------------------------------------- skills
    def open_tray(self, rack: int, u: int) -> str:
        try:
            name = tray_name(rack, u)
        except ValueError as e:
            return f"ERROR: {e}"
        where = f"R{rack:02} U{u}"
        if self._tray_q(name) > TRAY_OPEN:
            return f"OK: tray {where} already open ({self._tray_q(name) * 1000:.0f} mm)"
        res = self.base.go_to(rack, u)
        if not res.startswith("OK"):
            return res
        self.arm.gripper(FINGER_GAP)
        h = self._handle(name)
        pre, out, back = h - APPROACH * Y, h - OPEN * Y, h - (OPEN + RETREAT) * Y
        q, err = self._plan(name, pre, [h, out, back],
                            lambda i, s: (0.0, OPEN * s, OPEN)[i])
        if err:
            return err
        for step in (lambda: self.arm.move_joints(q), lambda: self.arm.move_linear(h, FACING_RACK)):
            res = step()
            if not res.startswith("OK"):
                return res
        err = self._grip(name)
        if err:
            return err
        hand0, tray0 = self.arm.tcp()[0], self._tray_q(name)

        def slip() -> float:
            return abs((hand0 - self.arm.tcp()[0]) @ Y - (self._tray_q(name) - tray0))

        def slipping():
            return f"SLIP: handle slid {slip() * 1000:.0f} mm in the grip" if slip() > MAX_SLIP else None

        res = self.arm.guarded_move(-Y, OPEN, max_force=PULL_LIMIT, speed=PULL_SPEED,
                                    also_stop=slipping)
        moved = self._tray_q(name) - tray0
        if res.startswith(("CONTACT", "SLIP")):
            # The tray resists more than the pull (or the grip) allows: a jam. Put it back
            # the way it was before letting go.
            why = (f"resisted more than {PULL_LIMIT:.0f} N" if res.startswith("CONTACT")
                   else "held harder than the grip")
            self.arm.guarded_move(Y, moved + 0.01, max_force=PUSH_LIMIT, speed=0.05)
            self._let_go_and_back_off()
            self.base.tuck()
            return (f"JAMMED: tray {where} {why} after {moved * 1000:.0f} mm; pushed back to "
                    f"{self._tray_q(name) * 1000:.0f} mm and let go.")
        if not res.startswith("OK"):
            return res
        slip_mm = slip() * 1000
        self._let_go_and_back_off()
        tq = self._tray_q(name)
        if abs(tq - OPEN) > 0.015:
            return f"ERROR: tray {where} ended at {tq * 1000:.0f} mm, expected {OPEN * 1000:.0f} mm."
        return f"OK: tray {where} open ({tq * 1000:.0f} mm, slip {slip_mm:.1f} mm)"

    def close_tray(self, rack: int, u: int) -> str:
        try:
            name = tray_name(rack, u)
        except ValueError as e:
            return f"ERROR: {e}"
        where = f"R{rack:02} U{u}"
        tq = self._tray_q(name)
        if tq < TRAY_OPEN:
            return f"OK: tray {where} already closed"
        if not self._at(rack, u):
            return (f"REFUSED: base is not lined up with {where}, and it cannot move while "
                    "that tray is open.")
        self.arm.gripper(FINGER_GAP)
        h = self._handle(name)
        closed = h + tq * Y
        pre, after = h - RETREAT * Y, closed - APPROACH * Y
        q, err = self._plan(name, pre, [h, closed, after],
                            lambda i, s: (tq, tq * (1 - s), 0.0)[i])
        if err:
            return err
        for step in (lambda: self.arm.move_joints(q), lambda: self.arm.move_linear(h, FACING_RACK)):
            res = step()
            if not res.startswith("OK"):
                return res
        err = self._grip(name)
        if err:
            return err
        # Push in until the closed stop pushes back.
        res = self.arm.guarded_move(Y, tq + 0.02, max_force=PUSH_LIMIT, speed=PULL_SPEED)
        self.arm.gripper(FINGER_GAP)
        self.base.holding = None
        left = self._tray_q(name)
        if left > 0.003:
            return f"BLOCKED: tray {where} stopped at {left * 1000:.0f} mm while closing ({res})."
        back = self.arm.guarded_move(-Y, APPROACH, max_force=5.0, speed=0.05)
        if not back.startswith("OK"):
            return back
        folded = self.base.tuck()
        return f"OK: tray {where} closed ({left * 1000:.1f} mm); {folded[4:]}"
