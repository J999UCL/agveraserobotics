"""DataCenterSim: the MuJoCo environment API.

Everything is in the robot base frame (= MuJoCo world): +X toward the rack, +Y left, +Z up.

Physical model in one paragraph: the tray is a drawer on a slide joint that the gripper
drags by its handle. RAM sticks and fan modules are free bodies held in place by weld
constraints. Closing the gripper on a part's grasp point attaches it to the hand;
pulling a seated part builds force in its slot weld until it unseats (or, if jammed,
keeps climbing). Lowering a held part onto a free slot within tolerance seats it.
Opening the gripper anywhere else drops it (e.g. into the used-parts bin).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import mujoco
import mujoco.viewer
import numpy as np

from .build import Layout, build_model
from .config import SimCfg
from .health import PartHealth, ServerHealth
from .kinematics import ARM_DOF, IK, QUAT_DOWN_FINGERS_Y, Q_NOMINAL

GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
SEAT_STIFFNESS = 2000.0  # N/m: a healthy part unseats ~6 mm into the lift (12 N)
EQ_CONSTRAINT = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)


class Stopped(Exception):
    pass


@dataclass
class PartState:
    id: str
    kind: str
    where: str  # "slot" | "hand" | "loose" | "bin"
    slot: str | None
    jammed: bool = False
    damaged_socket: bool = False
    lockout_slot: str | None = None  # just pulled from here; can't re-seat until lifted clear


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _rot(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


class DataCenterSim:
    def __init__(self, cfg: SimCfg | None = None, viewer: bool = False, realtime: bool = False) -> None:
        self.cfg = cfg or SimCfg()
        self.model, self.layout, _ = build_model(self.cfg)
        self.m, self.d = self.model, mujoco.MjData(self.model)
        self.ik = IK(self.model)
        self.rng = np.random.default_rng(0)
        self.realtime = realtime
        self.step_hooks: list[Callable[[], None]] = []
        self._stop = False
        self._viewer = None
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._index()
        self.reset()
        if viewer:
            self._viewer = mujoco.viewer.launch_passive(self.m, self.d)
            self._viewer.cam.lookat[:] = [0.45, 0.0, 0.1]
            self._viewer.cam.distance, self._viewer.cam.azimuth, self._viewer.cam.elevation = 2.2, -140, -25

    # ================================================================== setup
    def _index(self) -> None:
        m = self.m
        self.tray_body = m.body("tray").id
        self.hand_body = m.body("hand").id
        self.tcp_site = m.site("tcp").id
        j = m.joint("tray_slide").id
        self.tray_qadr, self.tray_dadr = m.jnt_qposadr[j], m.jnt_dofadr[j]
        self.eq = {}
        self.part_qadr = {}
        self.grasp_site = {}
        for cid in self.layout.components:
            for kind in ("slot", "hand", "cart"):
                self.eq[(cid, kind)] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_{kind}_{cid}")
            self.part_qadr[cid] = m.jnt_qposadr[m.joint(f"{cid}_free").id]
            self.grasp_site[cid] = m.site(f"{cid}_grasp").id
        self.mat = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"led_{n}")
                    for n in ("ok", "fault", "identify", "off", "rebuild")}
        self.tray_slots = [s for s, d in self.layout.slots.items() if d.where == "tray"]
        self.cart_slots = [s for s, d in self.layout.slots.items() if d.where == "cart"]

    def reset(self, seed: int | None = None, faults: list[tuple[str, str]] | None = None) -> dict:
        """Everything back to the start. Default faults: RAM in A3 failing, fan in F2 stalled.
        faults: list of (tray slot, fault) e.g. [("A6", "dimm_correctable")]; seed randomizes them."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:ARM_DOF] = Q_NOMINAL
        d.qpos[7:9] = 0.04
        d.ctrl[:ARM_DOF] = Q_NOMINAL
        d.ctrl[7] = GRIP_OPEN
        self.q_cmd = Q_NOMINAL.copy()
        self.grip_cmd = GRIP_OPEN
        self.parts: dict[str, PartState] = {}
        self.health = ServerHealth()
        for i, (cid, c) in enumerate(self.layout.components.items()):
            where = self.layout.slots[c.slot].where
            self.parts[cid] = PartState(cid, c.kind, "slot", c.slot)
            serial = f"{'DM' if c.kind == 'dimm' else 'FN'}{1000 + i * 37:05d}"
            self.health.parts[cid] = PartHealth(cid, c.kind, serial)
            d.eq_active[self.eq[(cid, "slot")]] = where == "tray"
            d.eq_active[self.eq[(cid, "cart")]] = where == "cart"
            d.eq_active[self.eq[(cid, "hand")]] = False
        mujoco.mj_forward(m, d)
        if seed is not None:
            r = np.random.default_rng(seed)
            dimm_slot = f"A{r.integers(1, 9)}"
            fan_slot = f"F{r.integers(1, 6)}"
            faults = [(dimm_slot, str(r.choice(["dimm_uncorrectable", "dimm_correctable"]))),
                      (fan_slot, str(r.choice(["fan_failed", "fan_degraded"])))]
        faults = faults if faults is not None else [("A3", "dimm_uncorrectable"), ("F2", "fan_failed")]
        for slot, fault in faults:
            pid = self.installed()[slot]
            self.health.inject(pid, fault)
            self.health.log("critical" if "failed" in fault or "uncorrectable" in fault else "warning",
                            f"{slot}: {self.health.parts[pid].fault_note}", slot=slot, part=pid)
        self._held: str | None = None
        self._grip_z0 = 0.0
        self._last_force = 0.0
        self._peak_force = 0.0
        self.events: list[dict] = []
        self._refresh_leds()
        return {"faults": [list(f) for f in faults]}

    # ================================================================== time
    @property
    def time(self) -> float:
        return float(self.d.time)

    def step(self, n: int = 1) -> None:
        """Advance n control ticks (control_dt each)."""
        sub = max(1, int(round(self.cfg.control_dt / self.cfg.timestep)))
        for _ in range(n):
            self.d.ctrl[:ARM_DOF] = self.q_cmd
            self.d.ctrl[7] = self.grip_cmd
            t0 = time.perf_counter()
            for _ in range(sub):
                mujoco.mj_step(self.m, self.d)
            self._after_tick()
            for hook in self.step_hooks:
                hook()
            if self._viewer is not None:
                if self._viewer.is_running():
                    self._viewer.sync()
            if self.realtime:
                lag = self.cfg.control_dt - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(lag)

    def _checked_step(self) -> None:
        if self._stop:
            self._stop = False
            raise Stopped("stopped")
        self.step()

    def stop(self) -> None:
        """Abort the running blocking motion and hold the current joints."""
        self._stop = True
        self.q_cmd = self.d.qpos[:ARM_DOF].copy()

    # ================================================================== arm
    def joint_positions(self) -> np.ndarray:
        return self.d.qpos[:ARM_DOF].copy()

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.d.site_xmat[self.tcp_site])
        return self.d.site_xpos[self.tcp_site].copy(), q

    def solve_ik(self, pos, quat=QUAT_DOWN_FINGERS_Y, seed=None) -> tuple[np.ndarray, bool]:
        seeds = [self.q_cmd if seed is None else seed, Q_NOMINAL,
                 Q_NOMINAL + [0.8, 0, 0, 0, 0, 0, 0], Q_NOMINAL - [0.8, 0, 0, 0, 0, 0, 0]]
        best = None
        for s in seeds:
            q, ok, err = self.ik.solve(pos, quat, s)
            if ok:
                return q, True
            if best is None or err < best[1]:
                best = (q, err)
        return best[0], False

    def move_tcp(self, pos, quat=QUAT_DOWN_FINGERS_Y, speed: float = 0.25, settle: float = 0.3) -> bool:
        """Straight-line TCP move at `speed` m/s (blocking). Returns False if unreachable."""
        p0, q0 = self.tcp_pose()
        pos, quat = np.asarray(pos, float), np.asarray(quat, float)
        q_goal, ok = self.solve_ik(pos, quat)
        if not ok:
            return False
        n = max(1, int(np.linalg.norm(pos - p0) / (speed * self.cfg.control_dt)))
        # also limit by rotation so big wrist turns are not instant
        dq_max = np.abs(q_goal - self.q_cmd).max()
        n = max(n, int(dq_max / (1.5 * self.cfg.control_dt)))
        q_prev = self.q_cmd.copy()
        for i in range(1, n + 1):
            a = i / n
            qi = _slerp(q0, quat, a)
            qj, okj, _ = self.ik.solve(p0 + (pos - p0) * a, qi, q_prev, iters=60)
            if not okj:
                qj = q_prev + (q_goal - q_prev) / max(1, n - i + 1)
            self.q_cmd = qj
            q_prev = qj
            self._checked_step()
        self.q_cmd = q_goal
        for _ in range(int(settle / self.cfg.control_dt)):
            self._checked_step()
        return bool(np.linalg.norm(self.tcp_pose()[0] - pos) < 0.01)

    def move_joints(self, q, duration: float = 2.0) -> None:
        q0, q1 = self.q_cmd.copy(), np.asarray(q, float)[:ARM_DOF]
        n = max(1, int(duration / self.cfg.control_dt))
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)
            self.q_cmd = q0 + (q1 - q0) * a
            self._checked_step()

    def home(self) -> None:
        p, _ = self.tcp_pose()
        if p[2] < 0.25:  # lift straight up first so we never sweep through the tray
            self.move_tcp([p[0], p[1], 0.30], self.tcp_pose()[1])
        self.move_joints(Q_NOMINAL)

    # ================================================================ gripper
    def gripper_width(self) -> float:
        return float(self.d.qpos[7] + self.d.qpos[8])

    def open_gripper(self) -> dict:
        self.grip_cmd = GRIP_OPEN
        released = self._release()
        for _ in range(20):
            self.step()
        return {"width": round(self.gripper_width(), 4), "released": released}

    def close_gripper(self) -> dict:
        self.grip_cmd = GRIP_CLOSED
        for _ in range(25):
            self.step()
        grabbed = self._try_grasp()
        return {"width": round(self.gripper_width(), 4), "holding": grabbed}

    def holding(self) -> str | None:
        return self._held

    # ============================================================ tray drawer
    def tray_extension(self) -> float:
        return float(-self.d.qpos[self.tray_qadr])

    def tray_state(self) -> str:
        e = self.tray_extension()
        return "in rack" if e < 0.01 else ("service position" if e > self.cfg.tray.travel - 0.01 else f"out {e:.2f} m")

    def tray_handle_position(self) -> np.ndarray:
        t = self.cfg.tray
        return self.d.xpos[self.tray_body] + np.array([t.handle_x, 0, t.handle_z])

    # ================================================================ parts
    def installed(self) -> dict[str, str | None]:
        out = {s: None for s in self.tray_slots}
        for p in self.parts.values():
            if p.where == "slot" and p.slot in out:
                out[p.slot] = p.id
        return out

    def cart_inventory(self) -> dict[str, str | None]:
        out = {s: None for s in self.cart_slots}
        for p in self.parts.values():
            if p.where == "slot" and p.slot in out:
                out[p.slot] = p.id
        return out

    def slot_world_pose(self, slot: str) -> tuple[np.ndarray, np.ndarray]:
        sd = self.layout.slots[slot]
        if sd.where == "cart":
            return sd.pos.copy(), np.array([1.0, 0, 0, 0])
        tq = self.d.xquat[self.tray_body]
        return self.d.xpos[self.tray_body] + _rot(tq) @ sd.pos, tq.copy()

    def slot_grasp_point(self, slot: str) -> np.ndarray:
        """Where the TCP must be to grasp a part seated in `slot` (from the server's layout spec)."""
        pos, _ = self.slot_world_pose(slot)
        t = self.cfg.tray
        dz = t.dimm_size[2] / 2 - 0.009 if self.layout.slots[slot].kind == "dimm" else t.fan_size[2] / 2 + 0.012
        return pos + np.array([0, 0, dz])

    def part_position(self, pid: str) -> np.ndarray:
        return self.d.xpos[self.m.body(pid).id].copy()

    def pull_force(self) -> float:
        return self._last_force

    def peak_force(self, reset: bool = False) -> float:
        """Highest extraction force since the last reset (the instantaneous value drops to 0 once a part unseats)."""
        v = self._peak_force
        if reset:
            self._peak_force = 0.0
        return v

    def set_jam(self, slot: str, on: bool = True) -> None:
        pid = self.installed().get(slot) or self.cart_inventory().get(slot)
        if pid is None:
            raise ValueError(f"no part in slot {slot}")
        self.parts[pid].jammed = on

    # --- welds -------------------------------------------------------------
    def _set_weld(self, eq_id: int, p1, q1, p2, q2, anchor_in_2=None) -> None:
        """Weld body1 (pose p1,q1) to body2 (p2,q2) at this relative pose.
        MuJoCo convention (verified empirically): relpose = body2 expressed in body1's frame,
        anchor left at zero."""
        R1 = _rot(q1)
        data = self.m.eq_data[eq_id]
        data[0:3] = 0.0
        data[3:6] = R1.T @ (np.asarray(p2) - np.asarray(p1))
        data[6:10] = _quat_mul(_quat_conj(q1), q2)
        data[10] = 1.0

    def _body_pose(self, bid: int) -> tuple[np.ndarray, np.ndarray]:
        return self.d.xpos[bid].copy(), self.d.xquat[bid].copy()

    def _try_grasp(self) -> str | None:
        if self._held:
            return self._held
        w = self.gripper_width()
        if w < 0.003:  # closed on nothing
            return None
        tcp = self.d.site_xpos[self.tcp_site]
        best, best_d = None, self.cfg.grasp_tol
        for pid, p in self.parts.items():
            if p.where == "bin":
                continue
            dist = np.linalg.norm(self.d.site_xpos[self.grasp_site[pid]] - tcp)
            if dist < best_d:
                best, best_d = pid, dist
        if best is None:
            return None
        self._held = best
        part = self.parts[best]
        if part.where == "slot" and self.layout.slots[part.slot].where == "tray":
            # Still seated: stays welded to its slot only. Pulling stretches a virtual spring
            # (see _after_tick); the hand weld engages once the part unseats.
            self._grip_z0 = float(self.d.site_xpos[self.tcp_site][2])
        else:
            if part.where == "slot":  # spares lift straight off the cart shelf
                self.d.eq_active[self.eq[(best, "cart")]] = False
                self._log_event("picked_from_cart", part=best, slot=part.slot)
                part.slot = None
            self._attach_to_hand(best)
            part.where = "hand"
        self._log_event("grasped", part=best)
        return best

    def _attach_to_hand(self, pid: str) -> None:
        p1, q1 = self._body_pose(self.m.body(pid).id)
        p2, q2 = self._body_pose(self.hand_body)
        self._set_weld(self.eq[(pid, "hand")], p1, q1, p2, q2, _rot(q2).T @ (p1 - p2))
        self.d.eq_active[self.eq[(pid, "hand")]] = True

    def _release(self) -> str | None:
        pid = self._held
        if pid is None:
            return None
        self.d.eq_active[self.eq[(pid, "hand")]] = False
        self._held = None
        part = self.parts[pid]
        if part.where != "slot":
            slot = self._capture_slot(pid, cart_only=True)
            if slot:
                self._seat(pid, slot)
            else:
                part.where = "loose"
        self._log_event("released", part=pid, where=part.where, slot=part.slot)
        return pid

    def _capture_slot(self, pid: str, cart_only: bool = False) -> str | None:
        kind = self.parts[pid].kind
        pos = self.part_position(pid)
        occupied = {p.slot for p in self.parts.values() if p.where == "slot"}
        tol = self.cfg.capture_pos_tol
        for slot, sd in self.layout.slots.items():
            if sd.kind != kind or slot in occupied or (cart_only and sd.where != "cart"):
                continue
            sp, sq = self.slot_world_pose(slot)
            dxy = np.linalg.norm((pos - sp)[:2])
            dz = pos[2] - sp[2]
            if dxy < tol and -0.004 < dz < 0.015 and self._yaw_err(pid, sq) < np.radians(self.cfg.capture_yaw_tol_deg):
                return slot
        return None

    def _yaw_err(self, pid: str, slot_quat) -> float:
        q = self.d.xquat[self.m.body(pid).id]
        rel = _quat_mul(_quat_conj(slot_quat), q)
        yaw = abs(np.arctan2(2 * (rel[0] * rel[3] + rel[1] * rel[2]), 1 - 2 * (rel[2] ** 2 + rel[3] ** 2)))
        return min(yaw, abs(np.pi - yaw))  # parts are symmetric front-to-back

    def _seat(self, pid: str, slot: str) -> None:
        sd = self.layout.slots[slot]
        sp, sq = self.slot_world_pose(slot)
        bid = self.m.body(pid).id
        if sd.where == "tray":
            p2, q2 = self._body_pose(self.tray_body)
            self._set_weld(self.eq[(pid, "slot")], sp, sq, p2, q2, sd.pos)
            self.d.eq_active[self.eq[(pid, "slot")]] = True
        else:
            self._set_weld(self.eq[(pid, "cart")], sp, sq, np.zeros(3), np.array([1.0, 0, 0, 0]), sp)
            self.d.eq_active[self.eq[(pid, "cart")]] = True
        part = self.parts[pid]
        part.where, part.slot = "slot", slot
        if self._held == pid:
            self.d.eq_active[self.eq[(pid, "hand")]] = False
            self._grip_z0 = float(self.d.site_xpos[self.tcp_site][2])
        if sd.where == "tray":
            h = self.health.parts[pid]
            h.pending_post = 3.0
            self.health.log("info", f"{slot}: part inserted ({h.serial}), verifying", slot=slot, part=pid)
        self._log_event("seated", part=pid, slot=slot)
        del bid

    def _weld_force(self, eq_id: int) -> float:
        d = self.d
        rows = np.nonzero((d.efc_type[: d.nefc] == EQ_CONSTRAINT) & (d.efc_id[: d.nefc] == eq_id))[0]
        if rows.size < 3:
            return 0.0
        return float(np.linalg.norm(d.efc_force[rows[:3]]))

    def _after_tick(self) -> None:
        dt = self.cfg.control_dt
        force = 0.0
        pid = self._held
        if pid is not None:
            part = self.parts[pid]
            if part.where == "slot" and self.layout.slots[part.slot].where == "tray":
                # virtual spring between the fingers and the seated part: force = k * lift
                lift = float(self.d.site_xpos[self.tcp_site][2]) - self._grip_z0
                force = max(0.0, SEAT_STIFFNESS * lift)
                if part.jammed:
                    if force > self.cfg.jam_break_n and not part.damaged_socket:
                        part.damaged_socket = True
                        self.health.log("critical", f"{part.slot}: socket damaged by excessive extraction force",
                                        slot=part.slot, force_n=round(force, 1))
                        self._log_event("socket_damaged", slot=part.slot, force_n=round(force, 1))
                elif force > self.cfg.seat_release_n:
                    slot = part.slot
                    self.d.eq_active[self.eq[(pid, "slot")]] = False
                    self._attach_to_hand(pid)
                    part.where, part.slot, part.lockout_slot = "hand", None, slot
                    self.health.log("info", f"{slot}: part removed ({self.health.parts[pid].serial})", slot=slot, part=pid)
                    self._log_event("unseated", part=pid, slot=slot, force_n=round(force, 1))
            elif part.where == "hand":
                if part.lockout_slot is not None:
                    sp, _ = self.slot_world_pose(part.lockout_slot)
                    if np.linalg.norm(self.part_position(pid) - sp) > 0.04:
                        part.lockout_slot = None
                slot = None if part.lockout_slot else self._capture_slot(pid)
                if slot and self.layout.slots[slot].where == "tray":
                    self._seat(pid, slot)
        self._last_force = force
        self._peak_force = max(self._peak_force, force)
        for p in self.parts.values():  # loose parts that land in the bin
            if p.where == "loose":
                pos = self.part_position(p.id)
                bx, by = self.cfg.cart.bin_center
                sx, sy, sz = self.cfg.cart.bin_size
                if abs(pos[0] - bx) < sx / 2 and abs(pos[1] - by) < sy / 2 and pos[2] < sz:
                    p.where = "bin"
                    self._log_event("in_bin", part=p.id)
        for ev in self.health.update(dt, self.installed(), self.rng):
            self._log_event(**ev)
        self._tick = getattr(self, "_tick", 0) + 1
        if self._tick % 10 == 0:
            self._refresh_leds()

    def _log_event(self, event: str, **kw) -> None:
        self.events.append({"t": round(self.time, 2), "event": event, **kw})

    # ================================================================ LEDs
    def _set_led(self, geom: str, state: str) -> None:
        self.m.geom_matid[self.m.geom(geom).id] = self.mat[state]

    def _refresh_leds(self) -> None:
        any_fault = False
        for slot, pid in self.installed().items():
            if pid is None:
                if slot.startswith("A"):
                    self._set_led(f"tray_led_{slot}", "off")
                continue
            h = self.health.parts[pid]
            state = "rebuild" if h.pending_post > 0 else ("fault" if h.status != "ok" else ("off" if slot.startswith("A") else "ok"))
            any_fault |= h.status != "ok"
            if slot.startswith("A"):
                self._set_led(f"tray_led_{slot}", state)
        for pid, p in self.parts.items():
            if p.kind == "fan":
                h = self.health.parts[pid]
                self._set_led(f"{pid}_led", "fault" if h.status != "ok" else "ok")
        self._set_led("tray_status_led", "fault" if any_fault else "ok")

    # ================================================================ sensing
    def render(self, camera: str = "wrist", width: int | None = None, height: int | None = None,
               depth: bool = False) -> np.ndarray:
        w = width or self.cfg.camera.width
        h = height or self.cfg.camera.height
        key = (w, h)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.m, h, w)
        r = self._renderers[key]
        if depth:
            r.enable_depth_rendering()
        r.update_scene(self.d, camera)
        img = r.render().copy()
        if depth:
            r.disable_depth_rendering()
        return img

    def camera_intrinsics(self, camera: str = "wrist", width: int | None = None, height: int | None = None) -> np.ndarray:
        w = width or self.cfg.camera.width
        h = height or self.cfg.camera.height
        fovy = np.radians(self.m.cam_fovy[self.m.camera(camera).id])
        f = 0.5 * h / np.tan(fovy / 2)
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

    def camera_pose(self, camera: str = "wrist") -> tuple[np.ndarray, np.ndarray]:
        cid = self.m.camera(camera).id
        return self.d.cam_xpos[cid].copy(), self.d.cam_xmat[cid].reshape(3, 3).copy()

    def pixel_to_world(self, u: float, v: float, depth: float, camera: str = "wrist") -> np.ndarray:
        """Depth (meters along the optical axis) at pixel (u, v) -> world xyz."""
        K = self.camera_intrinsics(camera)
        x = (u - K[0, 2]) / K[0, 0] * depth
        y = (v - K[1, 2]) / K[1, 1] * depth
        pos, R = self.camera_pose(camera)
        return R @ np.array([x, -y, -depth]) + pos  # MuJoCo cameras: x right, y up, look down -z

    # ================================================================ reports
    def server_health(self) -> dict:
        return self.health.report(self.installed(), self.tray_state())

    def event_log(self) -> list[dict]:
        return list(self.health.sel)

    def status(self) -> dict:
        p, q = self.tcp_pose()
        return {"time": round(self.time, 2), "tcp_position": p.round(4).tolist(), "tcp_quat": q.round(4).tolist(),
                "joints": self.joint_positions().round(4).tolist(), "gripper_width": round(self.gripper_width(), 4),
                "holding": self._held, "pull_force_n": round(self._last_force, 2),
                "tray_extension_m": round(self.tray_extension(), 4), "tray": self.tray_state()}

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()


def _slerp(q0, q1, a):
    out = np.zeros(4)
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, _quat_conj(q0), q1)
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, dq, 1.0)
    inc = np.zeros(4)
    mujoco.mju_axisAngle2Quat(inc, v / (np.linalg.norm(v) + 1e-12), np.linalg.norm(v) * a)
    mujoco.mju_mulQuat(out, q0, inc)
    return out
