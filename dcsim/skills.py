"""Reference manipulation skills built on DataCenterSim.

These use the server's documented slot layout (where the DIMM sockets and fan bays are,
like a service manual) - NOT which part is faulty. Deciding what to swap is the agents'
job, from get_server_health() / the event log / the LEDs in the camera image.

Every skill returns {"ok": bool, ...} and never raises for ordinary failures, so an
agent can read the result and decide what to do next.
"""

from __future__ import annotations

import numpy as np

from .kinematics import FRONT_APPROACH, QUAT_FRONT
from .kinematics import QUAT_DOWN_FINGERS_Y as FY
from .sim import DataCenterSim, Stopped

SAFE_Z = 0.30  # transit height above the cart/tray
ABORT_FORCE_N = 25.0  # stop pulling a part above this (healthy parts unseat at ~12 N)


class Skills:
    def __init__(self, sim: DataCenterSim) -> None:
        self.sim = sim

    # ------------------------------------------------------------ helpers
    def _transit(self, xy, z=SAFE_Z, quat=FY) -> bool:
        s = self.sim
        p, _ = s.tcp_pose()
        if p[2] < z - 0.01:
            if not s.move_tcp([p[0], p[1], z], s.tcp_pose()[1], speed=0.2):
                return False
        return s.move_tcp([xy[0], xy[1], z], quat, speed=0.35)

    def _result(self, ok: bool, **kw) -> dict:
        return {"ok": ok, **kw, "status": self.sim.status()}

    # ------------------------------------------------------------ tray
    def _grip_tab(self) -> np.ndarray:
        """Approach the pull tab from the front (gripper tilted down at the rack) and close on it."""
        s = self.sim
        h = s.tray_handle_position()
        pre = h - 0.09 * FRONT_APPROACH  # back along the approach direction: in front of and above the tab
        s.open_gripper()
        self._transit(pre[:2], quat=QUAT_FRONT)
        s.move_tcp(pre, QUAT_FRONT, speed=0.15)
        s.move_tcp(h, QUAT_FRONT, speed=0.06)
        s.close_gripper()
        return h

    def _let_go_of_tab(self) -> None:
        s = self.sim
        s.open_gripper()
        s.move_tcp(s.tcp_pose()[0] - 0.09 * FRONT_APPROACH, QUAT_FRONT, speed=0.12)

    def pull_tray_out(self) -> dict:
        s = self.sim
        if s.tray_extension() > s.cfg.tray.travel - 0.01:
            return self._result(True, note="tray already in service position")
        try:
            h = self._grip_tab()
            s.move_tcp(h - [s.cfg.tray.travel + 0.01, 0, 0], QUAT_FRONT, speed=0.10, settle=0.5)
            self._let_go_of_tab()
        except Stopped:
            return self._result(False, error="stopped")
        ext = s.tray_extension()
        return self._result(ext > s.cfg.tray.travel - 0.02, tray_extension_m=round(ext, 3))

    def push_tray_in(self) -> dict:
        s = self.sim
        if s.holding():
            return self._result(False, error="put the part down first")
        try:
            h = self._grip_tab()
            s.move_tcp(h + [s.tray_extension() + 0.01, 0, 0], QUAT_FRONT, speed=0.10, settle=0.5)
            self._let_go_of_tab()
            s.home()
        except Stopped:
            return self._result(False, error="stopped")
        return self._result(s.tray_extension() < 0.02, tray_extension_m=round(s.tray_extension(), 3))

    def view_tray(self, height: float = 0.46) -> dict:
        """Wrist camera looks straight down at the fan wall + RAM bank."""
        s = self.sim
        x = s.d.xpos[s.tray_body][0] + (s.cfg.tray.fan_x + s.cfg.tray.dimm_x) / 2
        try:
            ok = self._transit([x - 0.01, 0.0], z=height)
        except Stopped:
            return self._result(False, error="stopped")
        return self._result(ok)

    # ------------------------------------------------------------ parts
    def pick(self, slot: str) -> dict:
        """Grasp the part in `slot` (tray "A1".."A8"/"F1".."F5" or cart "CD1".."CF2") and lift it."""
        s = self.sim
        if s.holding():
            return self._result(False, error=f"already holding {s.holding()}")
        if slot not in s.layout.slots:
            return self._result(False, error=f"unknown slot {slot}")
        pid = s.installed().get(slot) or s.cart_inventory().get(slot)
        if pid is None:
            return self._result(False, error=f"slot {slot} is empty")
        if s.layout.slots[slot].where == "tray" and s.tray_extension() < s.cfg.tray.travel - 0.02:
            return self._result(False, error="pull the tray out first")
        g = s.slot_grasp_point(slot)
        try:
            s.open_gripper()
            self._transit(g[:2])
            s.move_tcp(g + [0, 0, 0.06], FY, speed=0.15)
            s.move_tcp(g, FY, speed=0.05)
            s.peak_force(reset=True)
            r = s.close_gripper()
            if r["holding"] != pid:
                s.open_gripper()
                s.move_tcp(g + [0, 0, 0.08], FY, speed=0.15)
                return self._result(False, error="grasp missed", gripper=r)
            # one smooth 12 cm lift, checking the extraction force every 20 ms
            lifted = s.move_tcp(g + [0, 0, 0.12], FY, speed=0.04, settle=0.1,
                                guard=lambda: "force" if s.pull_force() > ABORT_FORCE_N else None)
            peak = s.peak_force()
            if not lifted and s.guard_reason == "force":
                f = s.pull_force()
                s.move_tcp(g + [0, 0, 0.002], FY, speed=0.02, settle=0.1)  # relieve tension first
                s.open_gripper()
                s.move_tcp(g + [0, 0, 0.08], FY, speed=0.15)
                s.health.log("warning", f"{slot}: extraction aborted at {f:.1f} N (part stuck)", slot=slot)
                return self._result(False, error="part stuck: extraction aborted to protect the socket",
                                    peak_force_n=round(peak, 1), slot=slot)
        except Stopped:
            return self._result(False, error="stopped")
        held = s.holding()
        return self._result(held == pid and s.parts[pid].where == "hand", part=pid, peak_force_n=round(peak, 1))

    def place(self, slot: str) -> dict:
        """Lower the held part into an empty `slot` and let go."""
        s = self.sim
        pid = s.holding()
        if pid is None:
            return self._result(False, error="not holding anything")
        sd = s.layout.slots.get(slot)
        if sd is None or sd.kind != s.parts[pid].kind:
            return self._result(False, error=f"{pid} does not fit slot {slot}")
        if (s.installed().get(slot) or s.cart_inventory().get(slot)) is not None:
            return self._result(False, error=f"slot {slot} is occupied")
        g = s.slot_grasp_point(slot)
        try:
            self._transit(g[:2])
            s.move_tcp(g + [0, 0, 0.05], FY, speed=0.15)
            s.move_tcp(g + [0, 0, 0.001], FY, speed=0.03, settle=0.3)
            s.open_gripper()
            s.move_tcp(g + [0, 0, 0.08], FY, speed=0.15)
        except Stopped:
            return self._result(False, error="stopped")
        p = s.parts[pid]
        return self._result(p.where == "slot" and p.slot == slot, part=pid, where=p.where, slot=p.slot)

    def discard(self) -> dict:
        """Drop the held part into the used-parts bin."""
        s = self.sim
        pid = s.holding()
        if pid is None:
            return self._result(False, error="not holding anything")
        try:
            self._transit(s.cfg.cart.bin_center, z=0.28)
            s.open_gripper()
            for _ in range(40):
                s.step()
        except Stopped:
            return self._result(False, error="stopped")
        return self._result(s.parts[pid].where == "bin", part=pid, where=s.parts[pid].where)

    def find_spare(self, kind: str) -> str | None:
        for slot, pid in self.sim.cart_inventory().items():
            if pid is not None and self.sim.parts[pid].kind == kind:
                return slot
        return None

    def swap(self, slot: str) -> dict:
        """Full hot-swap: pull the part in `slot`, bin it, fetch a spare, install it."""
        s = self.sim
        steps = []
        kind = s.layout.slots[slot].kind
        spare = self.find_spare(kind)
        if spare is None:
            return self._result(False, error=f"no spare {kind} on the cart")
        for name, fn in (("pick", lambda: self.pick(slot)), ("discard", self.discard),
                         ("pick_spare", lambda: self.pick(spare)), ("place", lambda: self.place(slot))):
            r = fn()
            steps.append({"step": name, "ok": r["ok"], **{k: v for k, v in r.items() if k in ("error", "part", "peak_force_n")}})
            if not r["ok"]:
                return self._result(False, steps=steps, error=f"{name} failed: {r.get('error')}")
        s.home()
        return self._result(True, steps=steps)


__all__ = ["Skills", "np"]
