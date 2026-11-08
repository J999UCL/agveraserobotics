"""Robot skills built on DataCenterSim: navigation, tray handling, part swaps, restocking.

These use the data center's known layout (where racks/servers are, where DIMM sockets and fan
bays sit in a tray, like a floor plan and a service manual) - NOT which part is faulty.
Deciding what to fix is the agents' job (fleet health, server health, event log, camera).

Every skill returns {"ok": bool, ...}; ordinary failures come back as an "error" string the
agent can reason about instead of an exception.
"""

from __future__ import annotations

import numpy as np

from .kinematics import FRONT_APPROACH, QUAT_DOWN_FINGERS_Y, QUAT_FRONT
from .sim import DataCenterSim, Stopped, rot_z

SAFE_Z = 0.30  # transit height above the Franka base
ABORT_FORCE_N = 25.0  # stop pulling a part above this (healthy parts unseat at ~12 N)


class Skills:
    def __init__(self, sim: DataCenterSim) -> None:
        self.sim = sim

    # ------------------------------------------------------------ helpers
    def _down(self) -> np.ndarray:
        return self.sim.robot_quat(QUAT_DOWN_FINGERS_Y)

    def _front(self) -> np.ndarray:
        return self.sim.robot_quat(QUAT_FRONT)

    def _heading(self) -> np.ndarray:
        return rot_z(float(self.sim.base_pose()[2]))

    def _transit(self, target_world, quat=None, z_rel: float = SAFE_Z) -> bool:
        """Lift to a safe height above the robot, then move over `target_world` (x, y)."""
        s = self.sim
        quat = self._down() if quat is None else quat
        p, q = s.tcp_pose()
        pr = s.to_robot(p)
        if pr[2] < z_rel - 0.01 and not s.move_tcp(s.to_world([pr[0], pr[1], z_rel]), q, speed=0.2):
            return False
        tr = s.to_robot(np.asarray(target_world, float))
        return s.move_tcp(s.to_world([tr[0], tr[1], z_rel]), quat, speed=0.35)

    def _result(self, ok: bool, **kw) -> dict:
        return {"ok": ok, **kw, "status": self.sim.status()}

    def _need_server(self) -> str | None:
        return self.sim.active_server

    # ------------------------------------------------------------ navigation
    def go_to(self, server: str) -> dict:
        """Drive to a server's rack, face it, and raise the lift to its height."""
        s = self.sim
        if server not in s.layout.servers:
            return self._result(False, error=f"unknown server {server}; known: {list(s.layout.servers)}")
        if s.holding():
            return self._result(False, error="put the held part down before driving")
        out = [sid for sid in s.layout.servers if s.tray_extension(sid) > 0.02]
        if out:
            return self._result(False, error=f"tray of {out[0]} is still out; push it in first")
        try:
            t0 = s.time
            s.stow()
            route = s.route_to(s.service_pose(server))
            ok = s.drive_route(route)
            if ok:
                s.unstow()
        except Stopped as e:
            return self._result(False, error=str(e) or "stopped")
        s.active_server = server if ok else None
        extra = {} if ok else {"error": "aisle blocked: obstacle did not clear, robot stopped and is waiting in the aisle"}
        return self._result(ok, server=server, drive_time_s=round(s.time - t0, 1), nav_events=list(s.nav_log), **extra)

    def go_home(self) -> dict:
        """Drive back to the dock at the spares depot."""
        s = self.sim
        if s.holding():
            return self._result(False, error="put the held part down before driving")
        out = [sid for sid in s.layout.servers if s.tray_extension(sid) > 0.02]
        if out:
            return self._result(False, error=f"tray of {out[0]} is still out; push it in first")
        try:
            t0 = s.time
            s.stow()
            ok = s.drive_route(s.route_to(s.dock_pose()))
            if ok:
                s.unstow()
        except Stopped as e:
            return self._result(False, error=str(e) or "stopped")
        s.active_server = None
        extra = {} if ok else {"error": "aisle blocked: obstacle did not clear"}
        return self._result(ok, at="dock" if ok else "aisle", drive_time_s=round(s.time - t0, 1),
                            nav_events=list(s.nav_log), **extra)

    # ------------------------------------------------------------ tray
    def _grip_tab(self, server: str) -> np.ndarray:
        s = self.sim
        h = s.tray_handle_position(server)
        approach = self._heading() @ FRONT_APPROACH
        pre = h - 0.09 * approach
        s.open_gripper()
        self._transit(pre, quat=self._front())
        s.move_tcp(pre, self._front(), speed=0.15)
        s.move_tcp(h, self._front(), speed=0.06)
        s.close_gripper()
        return h

    def _let_go_of_tab(self) -> None:
        s = self.sim
        s.open_gripper()
        s.move_tcp(s.tcp_pose()[0] - 0.09 * (self._heading() @ FRONT_APPROACH), self._front(), speed=0.12)

    def pull_tray_out(self) -> dict:
        s = self.sim
        sid = self._need_server()
        if sid is None:
            return self._result(False, error="go_to a server first")
        if s.tray_extension(sid) > s.cfg.tray.travel - 0.01:
            return self._result(True, note="tray already in service position")
        try:
            h = self._grip_tab(sid)
            s.move_tcp(h - (s.cfg.tray.travel + 0.01) * self._heading()[:, 0], self._front(), speed=0.10, settle=0.5)
            self._let_go_of_tab()
        except Stopped as e:
            return self._result(False, error=str(e) or "stopped")
        ext = s.tray_extension(sid)
        return self._result(ext > s.cfg.tray.travel - 0.02, server=sid, tray_extension_m=round(ext, 3))

    def push_tray_in(self) -> dict:
        s = self.sim
        sid = self._need_server()
        if sid is None:
            return self._result(False, error="go_to a server first")
        if s.holding():
            return self._result(False, error="put the part down first")
        if s.tray_extension(sid) < 0.01:
            return self._result(True, note="tray already in")
        try:
            h = self._grip_tab(sid)
            s.move_tcp(h + (s.tray_extension(sid) + 0.01) * self._heading()[:, 0], self._front(), speed=0.10, settle=0.5)
            self._let_go_of_tab()
            s.home()
        except Stopped as e:
            return self._result(False, error=str(e) or "stopped")
        return self._result(s.tray_extension(sid) < 0.02, server=sid, tray_extension_m=round(s.tray_extension(sid), 3))

    def view_tray(self, height: float = 0.46) -> dict:
        """Wrist camera straight above the open tray's fan wall + RAM bank."""
        s = self.sim
        sid = self._need_server()
        if sid is None:
            return self._result(False, error="go_to a server first")
        t = s.cfg.tray
        p = s.tray_point(sid, [(t.fan_x + t.dimm_x) / 2 - 0.01, 0, 0])
        try:
            ok = self._transit(p, z_rel=height)
        except Stopped as e:
            return self._result(False, error=str(e) or "stopped")
        return self._result(ok)

    # ------------------------------------------------------------ parts
    def pick(self, slot: str) -> dict:
        """Grasp + lift the part in `slot`: "A1".."A8"/"F1".."F5" (current server), "CD1".."CF2"
        (onboard spares) or "DD1".."DF3" (depot shelf). Watches the extraction force."""
        s = self.sim
        if s.holding():
            return self._result(False, error=f"already holding {s.holding()}")
        try:
            key = s.slot_key(slot)
        except ValueError as e:
            return self._result(False, error=str(e))
        if key not in s.layout.slots:
            return self._result(False, error=f"unknown slot {slot}")
        pid = s.occupant(key)
        if pid is None:
            return self._result(False, error=f"slot {slot} is empty")
        sd = s.layout.slots[key]
        if sd.where == "tray" and s.tray_extension(sd.server) < s.cfg.tray.travel - 0.02:
            return self._result(False, error="pull the tray out first")
        if sd.where == "depot" and np.linalg.norm(s.base_pose()[:2] - s.dock_pose()[:2]) > 0.05:
            return self._result(False, error="go_home to the depot first")
        g = s.slot_grasp_point(key)
        down = self._down()
        try:
            s.open_gripper()
            self._transit(g)
            s.move_tcp(g + [0, 0, 0.06], down, speed=0.15)
            s.move_tcp(g, down, speed=0.05)
            s.peak_force(reset=True)
            r = s.close_gripper()
            if r["holding"] != pid:
                s.open_gripper()
                s.move_tcp(g + [0, 0, 0.08], down, speed=0.15)
                return self._result(False, error="grasp missed", gripper=r)
            lifted = s.move_tcp(g + [0, 0, 0.12], down, speed=0.04, settle=0.1,
                                guard=lambda: "force" if s.pull_force() > ABORT_FORCE_N else None)
            peak = s.peak_force()
            if not lifted and s.guard_reason == "force":
                f = s.pull_force()
                s.move_tcp(g + [0, 0, 0.002], down, speed=0.02, settle=0.1)  # relieve tension first
                s.open_gripper()
                s.move_tcp(g + [0, 0, 0.08], down, speed=0.15)
                if sd.where == "tray":
                    s.health[sd.server].log("warning", f"{sd.name}: extraction aborted at {f:.1f} N (part stuck)", slot=sd.name)
                return self._result(False, error="part stuck: extraction aborted to protect the socket",
                                    peak_force_n=round(peak, 1), slot=slot)
        except Stopped as e:
            return self._result(False, error=str(e) or "stopped")
        return self._result(s.holding() == pid and s.parts[pid].where == "hand", part=pid, peak_force_n=round(peak, 1))

    def place(self, slot: str) -> dict:
        """Lower the held part into an empty, compatible `slot` and let go."""
        s = self.sim
        pid = s.holding()
        if pid is None:
            return self._result(False, error="not holding anything")
        try:
            key = s.slot_key(slot)
        except ValueError as e:
            return self._result(False, error=str(e))
        sd = s.layout.slots.get(key)
        if sd is None or sd.kind != s.parts[pid].kind:
            return self._result(False, error=f"{pid} does not fit slot {slot}")
        if s.occupant(key) is not None:
            return self._result(False, error=f"slot {slot} is occupied")
        if sd.where == "tray" and s.tray_extension(sd.server) < s.cfg.tray.travel - 0.02:
            return self._result(False, error="pull the tray out first")
        g = s.slot_grasp_point(key)
        down = self._down()
        try:
            self._transit(g)
            s.move_tcp(g + [0, 0, 0.05], down, speed=0.15)
            s.move_tcp(g + [0, 0, 0.001], down, speed=0.03, settle=0.3)
            s.open_gripper()
            s.move_tcp(g + [0, 0, 0.08], down, speed=0.15)
        except Stopped as e:
            return self._result(False, error=str(e) or "stopped")
        p = s.parts[pid]
        return self._result(p.where == "slot" and p.slot == key, part=pid, slot=slot)

    def discard(self) -> dict:
        """Drop the held part into the robot's returns bin."""
        s = self.sim
        pid = s.holding()
        if pid is None:
            return self._result(False, error="not holding anything")
        bx, by = s.cfg.robot.bin_center
        try:
            self._transit(s.to_world([bx, by, 0]), z_rel=0.26)
            s.open_gripper()
            for _ in range(40):
                s.step()
        except Stopped as e:
            return self._result(False, error=str(e) or "stopped")
        return self._result(s.parts[pid].where == "bin", part=pid)

    def find_spare(self, kind: str) -> str | None:
        for slot, pid in self.sim.storage("onboard").items():
            if pid is not None and self.sim.parts[pid].kind == kind:
                return slot
        return None

    def swap(self, slot: str) -> dict:
        """Hot-swap the part in tray `slot` of the current server: pull it, bin it, install an onboard spare."""
        s = self.sim
        try:
            key = s.slot_key(slot)
        except ValueError as e:
            return self._result(False, error=str(e))
        sd = s.layout.slots.get(key)
        if sd is None or sd.where != "tray":
            return self._result(False, error=f"{slot} is not a server slot")
        spare = self.find_spare(sd.kind)
        if spare is None:
            return self._result(False, error=f"no spare {sd.kind} on board: go_home and restock")
        steps = []
        for name, fn in (("pick", lambda: self.pick(slot)), ("discard", self.discard),
                         ("pick_spare", lambda: self.pick(spare)), ("place", lambda: self.place(slot))):
            r = fn()
            steps.append({"step": name, "ok": r["ok"], **{k: v for k, v in r.items() if k in ("error", "part", "peak_force_n")}})
            if not r["ok"]:
                return self._result(False, steps=steps, error=f"{name} failed: {r.get('error')}")
        s.home()
        return self._result(True, steps=steps)

    def restock(self) -> dict:
        """At the dock: fill empty onboard spare slots from the depot shelf."""
        s = self.sim
        if np.linalg.norm(s.base_pose()[:2] - s.dock_pose()[:2]) > 0.05:
            return self._result(False, error="go_home to the depot first")
        moved = []
        for slot, pid in s.storage("onboard").items():
            if pid is not None:
                continue
            kind = "dimm" if slot.startswith("CD") else "fan"
            src = next((d for d, p in s.storage("depot").items() if p and s.parts[p].kind == kind), None)
            if src is None:
                continue
            r = self.pick(src)
            if not r["ok"]:
                return self._result(False, moved=moved, error=f"pick {src} failed: {r.get('error')}")
            r = self.place(slot)
            if not r["ok"]:
                return self._result(False, moved=moved, error=f"place {slot} failed: {r.get('error')}")
            moved.append(f"{src}->{slot}")
        s.home()
        return self._result(True, moved=moved, onboard=s.storage("onboard"), depot=s.storage("depot"))
