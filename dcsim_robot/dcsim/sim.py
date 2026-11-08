"""DataCenterSim: the MuJoCo environment.

World frame: floor at z=0, cold aisle along +X (see config.py). The robot is a mobile
manipulator: driven base joints (x, y, yaw) + a lift carry the Franka; a wheeled chassis
(kinematic body) follows underneath.

Physical model in one paragraph: each server tray is a drawer on a slide joint that the
gripper drags by its pull tab. RAM sticks and fan modules are free bodies held by weld
constraints. Closing the gripper on a part's grasp point grips it; pulling a seated part
stretches a virtual spring until it unseats (jammed parts keep building force). Unseated
parts attach to the hand; lowering one onto a free compatible slot seats it; opening the
gripper over the onboard bin drops it in (it then rides along with the robot).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import mujoco
import mujoco.viewer
import numpy as np

from .build import DEPOT_SHELF_Z, LIDAR_RANGE, LIDAR_RAYS, Layout, build_model, rot_z
from .config import SERVICE_STANDOFF, TRAY_ABOVE_BASE, SimCfg
from .health import PartHealth, ServerHealth
from .kinematics import ARM_DOF, IK, Q_NOMINAL, Q_STOW, QUAT_DOWN_FINGERS_Y, yaw_rotate

GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
MAX_JOINT_SPEED = 1.2  # rad/s cap on commanded arm joint motion
SEAT_STIFFNESS = 2000.0  # N/m: a healthy part unseats ~6 mm into the lift (12 N)
TRAVEL_LIFT = 0.55  # lift height while driving
STOP_DIST = 0.60  # lidar: stop if anything is this close in the travel direction
BUMPER_DIST = 0.12  # virtual bumper: emergency-stop event below this
MAX_WAIT = 20.0  # give up on a blocked aisle after this long


COLLISION_N = 15.0  # contact force with racks/depot/walls that counts as a collision (a normal shift has none)


class Stopped(Exception):
    pass


class Collision(Stopped):
    """The robot hit the scenery hard enough to stop."""


@dataclass
class PartState:
    id: str
    kind: str
    where: str  # "slot" | "hand" | "loose" | "bin"
    slot: str | None  # slot key, e.g. "R1-05-U20:A3", "onboard:CD1", "depot:DD2"
    jammed: bool = False
    damaged_socket: bool = False
    lockout_slot: str | None = None


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


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class DataCenterSim:
    def __init__(self, cfg: SimCfg | None = None, viewer: bool = False, realtime: bool = False) -> None:
        self.cfg = cfg or SimCfg()
        self.model, self.layout, _ = build_model(self.cfg)
        self.m, self.d = self.model, mujoco.MjData(self.model)
        self.ik = IK(self.model)
        self._sensors(False)  # lidar rays are expensive; they only run while driving (or on demand)
        self.rng = np.random.default_rng(0)
        self.realtime = realtime
        self.step_hooks: list[Callable[[], None]] = []
        self.guard_reason: str | None = None
        self._stop = False
        self.collision_watch = True
        self.last_collision: dict | None = None
        self._viewer = None
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._index()
        self.reset()
        if viewer:
            self._viewer = mujoco.viewer.launch_passive(self.m, self.d)
            self._viewer.cam.lookat[:] = [1.2, 0.0, 0.6]
            self._viewer.cam.distance, self._viewer.cam.azimuth, self._viewer.cam.elevation = 5.0, -130, -28

    # ================================================================== setup
    def _index(self) -> None:
        m = self.m
        jid = lambda n: m.joint(n).id  # noqa: E731
        self.arm_qadr = np.array([m.jnt_qposadr[jid(f"joint{i}")] for i in range(1, 8)])
        self.arm_dadr = np.array([m.jnt_dofadr[jid(f"joint{i}")] for i in range(1, 8)])
        self.finger_qadr = np.array([m.jnt_qposadr[jid(n)] for n in ("finger_joint1", "finger_joint2")])
        self.base_qadr = np.array([m.jnt_qposadr[jid(n)] for n in ("base_x", "base_y", "base_yaw", "lift")])
        self.base_dadr = np.array([m.jnt_dofadr[jid(n)] for n in ("base_x", "base_y", "base_yaw", "lift")])
        self.act_arm = np.array([m.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.act_grip = m.actuator("actuator8").id
        self.act_base = np.array([m.actuator(f"act_{n}").id for n in ("base_x", "base_y", "base_yaw", "lift")])
        self.hand_body = m.body("hand").id
        self.link0_body = m.body("link0").id
        self.depot_body = m.body("depot").id
        self.chassis_mocap = m.body_mocapid[m.body("chassis").id]
        self.tech_mocap = m.body_mocapid[m.body("technician").id]
        self.cable_mocap = m.body_mocapid[m.body("loose_cable").id]
        self.lidar_adr = m.sensor_adr[m.sensor("lidar_00").id]
        self.tcp_site = m.site("tcp").id
        self.srv = {}
        for sid, sv in self.layout.servers.items():
            j = m.joint(sv.slide_joint).id
            self.srv[sid] = {"tray": m.body(sv.tray_body).id, "qadr": m.jnt_qposadr[j],
                             "handle_eq": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, sv.handle_eq),
                             "lock_eq": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"s{sv.index}_rail_lock")}
        self.eq, self.part_qadr, self.grasp_site = {}, {}, {}
        for cid in self.layout.components:
            for kind in ("slot", "hand", "store"):
                self.eq[(cid, kind)] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_{kind}_{cid}")
            self.part_qadr[cid] = m.jnt_qposadr[m.joint(f"{cid}_free").id]
            self.grasp_site[cid] = m.site(f"{cid}_grasp").id
        bid_of = {cid: m.body(cid).id for cid in self.layout.components}
        self.part_geoms = {cid: [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g] == 2]
                           for cid, b in bid_of.items()}
        # collision watchdog: robot geoms vs fixed scenery (racks, depot, walls, room). Trays, parts and
        # the floor are excluded: touching those is part of the job.
        def root(b):
            while m.body_parentid[b] != 0:
                b = m.body_parentid[b]
            return b
        def under(b, top):
            while b != 0:
                if b == top:
                    return True
                b = m.body_parentid[b]
            return False
        robot_roots = {self.link0_body, m.body("chassis").id}
        trays = [v["tray"] for v in self.srv.values()]
        part_bodies = set(bid_of.values())
        floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.geom_robot = np.zeros(m.ngeom, bool)
        self.geom_scenery = np.zeros(m.ngeom, bool)
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if root(b) in robot_roots:
                self.geom_robot[g] = True
            elif (g != floor and b not in part_bodies and b not in (m.body("technician").id, m.body("loose_cable").id)
                  and not any(under(b, t) for t in trays)):
                self.geom_scenery[g] = True
        self.mat = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"led_{n}")
                    for n in ("ok", "fault", "identify", "off", "rebuild", "red")}

    def reset(self, seed: int | None = None, faults: list[tuple[str, str, str]] | None = None) -> dict:
        """Robot docked at the depot, all trays in, default faults (or a random set with `seed`).
        faults: list of (server_id, slot, fault), e.g. [("R1-05-U20", "A3", "dimm_uncorrectable")]."""
        m, d = self.m, self.d
        self.last_collision = None
        self.led_override: dict[tuple[str, str], str] = {}
        mujoco.mj_resetData(m, d)
        self.q_cmd = Q_NOMINAL.copy()
        self.grip_cmd = GRIP_OPEN
        dock = self.dock_pose()
        self.base_cmd = np.array([dock[0], dock[1], dock[2], DEPOT_SHELF_Z])
        d.qpos[self.base_qadr] = self.base_cmd
        d.qpos[self.arm_qadr] = Q_NOMINAL
        d.qpos[self.finger_qadr] = 0.04
        self._apply_ctrl()
        self._sync_chassis()
        mujoco.mj_kinematics(m, d)
        self.parts: dict[str, PartState] = {}
        self.health = {sid: ServerHealth(name=sid) for sid in self.layout.servers}
        for i, (cid, c) in enumerate(self.layout.components.items()):
            self.parts[cid] = PartState(cid, c.kind, "slot", c.slot_key)
            serial = f"{'DM' if c.kind == 'dimm' else 'FN'}{1000 + i * 37:05d}"
            ph = PartHealth(cid, c.kind, serial)
            for h in self.health.values():
                h.parts[cid] = ph  # parts can move between servers; every server knows every part
            d.eq_active[self.eq[(cid, "hand")]] = False
            self._place_in_slot(cid, c.slot_key)
        for sid in self.srv:
            self._latch_tray(sid, True)
        mujoco.mj_forward(m, d)
        self.active_server: str | None = None
        self._held: str | None = None
        self._on_handle: str | None = None
        self._grip_z0 = 0.0
        self._last_force = self._peak_force = 0.0
        self._tick = 0
        self.events: list[dict] = []
        self.work_orders: list[dict] = []
        self.nav_log: list[dict] = []
        self._tech_keys = None
        self.clear_cable()
        if seed is not None:
            r = np.random.default_rng(seed)
            sids = list(self.layout.servers)
            faults = [(str(r.choice(sids)), f"A{r.integers(1, 9)}", str(r.choice(["dimm_uncorrectable", "dimm_correctable"]))),
                      (str(r.choice(sids)), f"F{r.integers(1, 6)}", str(r.choice(["fan_failed", "fan_degraded"])))]
        if faults is None:
            faults = [("R1-05-U20", "A3", "dimm_uncorrectable"), ("R2-03-U14", "F2", "fan_failed")]
        for sid, slot, fault in faults:
            self.inject_fault(sid, slot, fault)
        self._refresh_leds()
        return {"faults": [list(f) for f in faults], "robot": "docked at depot"}

    def inject_fault(self, server: str, slot: str, fault: str) -> None:
        pid = self.installed(server)[slot]
        if pid is None:
            raise ValueError(f"{server}:{slot} is empty")
        h = self.health[server]
        h.inject(pid, fault)
        severity = "critical" if ("failed" in fault or "uncorrectable" in fault) else "warning"
        h.log(severity, f"{slot}: {h.parts[pid].fault_note}", slot=slot, part=pid)
        sv = self.layout.servers[server]
        self.work_orders.append({
            "id": f"WO-{len(self.work_orders) + 1:03d}", "server": server, "rack": f"R{sv.row}-{sv.rack:02d}", "u": sv.u,
            "slot": slot, "component": "RAM (DIMM)" if slot.startswith("A") else "fan module", "fault": fault,
            "severity": severity, "detail": h.parts[pid].fault_note, "status": "open",
            "opened_t": round(self.time, 1), "resolved_t": None})

    # ================================================================== time
    @property
    def time(self) -> float:
        return float(self.d.time)

    def _apply_ctrl(self) -> None:
        self.d.ctrl[self.act_arm] = self.q_cmd
        self.d.ctrl[self.act_grip] = self.grip_cmd
        self.d.ctrl[self.act_base] = self.base_cmd

    def _sync_chassis(self) -> None:
        x, y, yaw = self.d.qpos[self.base_qadr[:3]]
        off = self.cfg.robot.chassis_offset_x
        self.d.mocap_pos[self.chassis_mocap] = [x + off * np.cos(yaw), y + off * np.sin(yaw), 0.0]
        self.d.mocap_quat[self.chassis_mocap] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def step(self, n: int = 1) -> None:
        """Advance n control ticks (control_dt each)."""
        sub = max(1, int(round(self.cfg.control_dt / self.cfg.timestep)))
        for _ in range(n):
            self._apply_ctrl()
            t0 = time.perf_counter()
            self._update_technician()
            for _ in range(sub):
                mujoco.mj_step(self.m, self.d)
                self._sync_chassis()
            self._after_tick()
            for hook in self.step_hooks:
                hook()
            if self._viewer is not None and self._viewer.is_running():
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
        if self.collision_watch:
            hit = self._scenery_contact()
            if hit is not None:
                self.last_collision = hit
                self._log_event("collision", **hit)
                self.nav_log.append({"event": "collision", **hit})
                self.stop()
                self._stop = False
                raise Collision(f"collision: {hit['robot_part']} hit {hit['object']} ({hit['force_n']} N); "
                                "motion stopped")

    def _scenery_contact(self) -> dict | None:
        """Strongest contact between the robot and fixed scenery above COLLISION_N, else None."""
        n = self.d.ncon
        if n == 0:
            return None
        g = self.d.contact.geom[:n]
        rob, scen = self.geom_robot, self.geom_scenery
        idx = np.nonzero((rob[g[:, 0]] & scen[g[:, 1]]) | (scen[g[:, 0]] & rob[g[:, 1]]))[0]
        best, f6 = None, np.zeros(6)
        for i in idx:
            mujoco.mj_contactForce(self.m, self.d, int(i), f6)
            if abs(f6[0]) > COLLISION_N and (best is None or abs(f6[0]) > best[0]):
                best = (abs(f6[0]), int(i))
        if best is None:
            return None
        g1, g2 = (int(x) for x in g[best[1]])
        r, o = (g1, g2) if rob[g1] else (g2, g1)
        name = lambda gid: (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, gid)  # noqa: E731
                            or mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[gid]))
        return {"t": round(self.time, 2), "robot_part": name(r), "object": name(o), "force_n": round(best[0], 1)}

    def stop(self) -> None:
        """Abort the running blocking motion and hold position."""
        self._stop = True
        self.q_cmd = self.d.qpos[self.arm_qadr].copy()
        self.base_cmd = self.d.qpos[self.base_qadr].copy()

    # ================================================================== base
    def base_pose(self) -> np.ndarray:
        """[x, y, yaw, lift]."""
        return self.d.qpos[self.base_qadr].copy()

    def robot_quat(self, quat_robot) -> np.ndarray:
        """Robot-frame orientation (robot facing +X) -> world, using the current heading."""
        return yaw_rotate(quat_robot, float(self.d.qpos[self.base_qadr[2]]))

    def to_world(self, p_robot) -> np.ndarray:
        """Point in the Franka-base frame -> world."""
        return self.d.xpos[self.link0_body] + _rot(self.d.xquat[self.link0_body]) @ np.asarray(p_robot, float)

    def to_robot(self, p_world) -> np.ndarray:
        return _rot(self.d.xquat[self.link0_body]).T @ (np.asarray(p_world, float) - self.d.xpos[self.link0_body])

    def service_pose(self, server: str) -> np.ndarray:
        sv = self.layout.servers[server]
        xy = sv.rack_pos[:2] - SERVICE_STANDOFF * np.array([np.cos(sv.rack_yaw), np.sin(sv.rack_yaw)])
        return np.array([xy[0], xy[1], sv.rack_yaw, sv.tray_bottom - TRAY_ABOVE_BASE])

    def dock_pose(self) -> np.ndarray:
        pos, yaw = self.layout.depot
        xy = pos[:2] - SERVICE_STANDOFF * np.array([np.cos(yaw), np.sin(yaw)])
        return np.array([xy[0], xy[1], yaw, DEPOT_SHELF_Z])

    # ---------------------------------------------------------------- sensing while driving
    def _sensors(self, on: bool) -> None:
        bit = int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        if on:
            self.m.opt.disableflags &= ~bit
        else:
            self.m.opt.disableflags |= bit

    def lidar_scan(self) -> dict:
        """2D lidar on the front of the base: 37 rays from -90 to +90 deg (robot frame, 0 = ahead).
        Ranges in meters measured from the front bumper; None = nothing within 4 m."""
        self._sensors(True)
        mujoco.mj_forward(self.m, self.d)
        r = self.d.sensordata[self.lidar_adr:self.lidar_adr + LIDAR_RAYS].copy()
        self._sensors(False)
        angles = np.linspace(-90, 90, LIDAR_RAYS)
        return {"angles_deg": angles.round(1).tolist(),
                "ranges_m": [None if (x < 0 or x >= LIDAR_RANGE - 1e-6) else round(float(x), 3) for x in r]}

    def _path_clearance(self, travel_dir_robot: float) -> float:
        """Smallest lidar range within +/-35 deg of the travel direction (robot frame, rad)."""
        r = np.array(self.d.sensordata[self.lidar_adr:self.lidar_adr + LIDAR_RAYS])
        ang = np.linspace(-np.pi / 2, np.pi / 2, LIDAR_RAYS)
        cone = (np.abs(ang - travel_dir_robot) < np.radians(35)) & (r >= 0) & (r < LIDAR_RANGE - 1e-6)
        return float(r[cone].min()) if cone.any() else np.inf

    def _drive_segment(self, target: np.ndarray, watch: bool = False) -> bool:
        """Drive the base targets [x, y, yaw, lift] to target with a velocity profile.

        With watch=True the lidar is checked every tick: anything closer than STOP_DIST in the
        travel direction brakes the robot to a stop; it waits and resumes once the path has been
        clear for a moment. Returns False if blocked longer than MAX_WAIT.
        """
        rc, dt = self.cfg.robot, self.cfg.control_dt
        start = self.base_cmd.copy()
        delta = np.asarray(target, float) - start
        delta[2] = _wrap(delta[2])
        T = max(np.linalg.norm(delta[:2]) / rc.drive_speed, abs(delta[2]) / rc.turn_speed,
                abs(delta[3]) / rc.lift_speed, dt)
        sdot_max = 1.0 / T
        acc = sdot_max / 0.7  # ~0.7 s to reach full speed
        yaw = float(start[2])
        rel = np.array([np.cos(-yaw) * delta[0] - np.sin(-yaw) * delta[1],
                        np.sin(-yaw) * delta[0] + np.cos(-yaw) * delta[1]])
        travel_dir = float(np.arctan2(rel[1], rel[0])) if np.linalg.norm(rel) > 0.05 else None
        watch = watch and travel_dir is not None and abs(travel_dir) < np.radians(60)
        self._sensors(watch)
        if watch:
            mujoco.mj_forward(self.m, self.d)  # fresh lidar readings before the first check
        try:
            return self._drive_profile(start, delta, sdot_max, acc, travel_dir, watch)
        finally:
            self._sensors(False)

    def _drive_profile(self, start, delta, sdot_max, acc, travel_dir, watch) -> bool:
        dt = self.cfg.control_dt
        s, sdot, waited, clear_for, stopped = 0.0, 0.0, 0.0, 0.0, False
        while s < 1.0:
            blocked = False
            if watch and (1.0 - s) * np.linalg.norm(delta[:2]) > 0.8:
                c = self._path_clearance(travel_dir)
                if c < BUMPER_DIST:
                    self._log_event("bumper_estop", clearance_m=round(c, 3))
                blocked = c < STOP_DIST
            if blocked:
                sdot = max(0.0, sdot - 4 * acc * dt)  # brake hard
                clear_for = 0.0
                if not stopped:
                    stopped = True
                    c = round(self._path_clearance(travel_dir), 2)
                    self.nav_log.append({"event": "obstacle_stop", "t": round(self.time, 2), "clearance_m": c})
                    self._log_event("obstacle_stop", clearance_m=c)
                waited += dt
                if waited > MAX_WAIT:
                    self.nav_log.append({"event": "gave_up", "t": round(self.time, 2), "waited_s": round(waited, 1)})
                    self.base_cmd = self.d.qpos[self.base_qadr].copy()
                    return False
            else:
                if stopped:
                    clear_for += dt
                    if clear_for < 1.0:  # stay put until the path has been clear for a moment
                        self.base_cmd = start + delta * s
                        self._checked_step()
                        continue
                    stopped = False
                    self.nav_log.append({"event": "resumed", "t": round(self.time, 2), "waited_s": round(waited, 1)})
                    self._log_event("resumed", waited_s=round(waited, 1))
                brake = np.sqrt(2 * acc * max(0.0, 1.0 - s))
                sdot = min(sdot_max, sdot + acc * dt, max(brake, 0.02 * sdot_max))
            s = min(1.0, s + sdot * dt)
            self.base_cmd = start + delta * s
            self._checked_step()
        for _ in range(20):
            self._checked_step()
        return True

    def drive_route(self, waypoints: list[np.ndarray]) -> bool:
        """Follow waypoints [x, y, yaw, lift(, watch)]. The drive down the aisle is lidar-guarded;
        turning on the spot and the final approach to a rack/dock are not (the rack itself is close)."""
        self.nav_log = []
        for wp in waypoints:
            wp = np.asarray(wp, float)
            if not self._drive_segment(wp[:4], watch=len(wp) > 4 and wp[4] > 0):
                return False
        return True

    # ---------------------------------------------------------------- moving obstacle (demo)
    def spawn_technician(self, mode: str = "cross", x: float | None = None, delay: float = 0.0,
                         duration: float = 4.0) -> dict:
        """Demo: a technician walks into the aisle. mode "cross": walks across it once.
        "block": steps into the middle of the aisle, stands for duration seconds, then leaves.
        x defaults to 1.6 m ahead of the robot along the aisle."""
        if x is None:
            b = self.base_pose()
            heading = float(np.sign(np.cos(b[2]))) or 1.0
            x = float(b[0] + 1.6 * heading)
        y0, y1 = 0.75, -0.75
        t0 = self.time + delay
        if mode == "cross":
            keys = [(t0, [x, y0]), (t0 + 2.0, [x, y1])]
        elif mode == "block":
            keys = [(t0, [x, y0]), (t0 + 1.0, [x, 0.0]), (t0 + 1.0 + duration, [x, 0.0]), (t0 + 2.0 + duration, [x, y1])]
        else:
            raise ValueError("mode must be 'cross' or 'block'")
        self._tech_keys = keys
        return {"mode": mode, "x": round(x, 2), "starts_in_s": delay}

    def _update_technician(self) -> None:
        keys = self._tech_keys
        mid = self.tech_mocap
        if not keys or self.time < keys[0][0] or self.time > keys[-1][0] + 0.5:
            self.d.mocap_pos[mid] = [0, 0, -10]
            if keys and self.time > keys[-1][0] + 0.5:
                self._tech_keys = None
            return
        for (ta, pa), (tb, pb) in zip(keys, keys[1:]):
            if ta <= self.time <= tb:
                a = (self.time - ta) / max(tb - ta, 1e-6)
                pa_, pb_ = np.array(pa), np.array(pb)
                pp = pa_ + (pb_ - pa_) * a
                self.d.mocap_pos[mid] = [pp[0], pp[1], 0.0]
                dd = pb_ - pa_
                if np.linalg.norm(dd) > 1e-6:
                    yw = np.arctan2(dd[1], dd[0])
                    self.d.mocap_quat[mid] = [np.cos(yw / 2), 0, 0, np.sin(yw / 2)]
                return
        self.d.mocap_pos[mid] = [keys[-1][1][0], keys[-1][1][1], 0.0]

    def route_to(self, goal: np.ndarray) -> list[np.ndarray]:
        """Aisle route: back out to the aisle centreline, turn, drive along it, turn, pull in."""
        cur = self.base_cmd.copy()
        travel_lift = max(TRAVEL_LIFT, self.cfg.robot.lift_range[0])
        wps = []
        if abs(cur[1]) > 0.02 or abs(cur[3] - travel_lift) > 0.01:
            wps.append(np.array([cur[0], 0.0, cur[2], travel_lift]))
        dx = goal[0] - cur[0]
        if abs(dx) > 0.02:
            heading = 0.0 if dx > 0 else np.pi
            heading = cur[2] + _wrap(heading - cur[2])
            wps.append(np.array([cur[0], 0.0, heading, travel_lift]))
            wps.append(np.array([goal[0], 0.0, heading, travel_lift, 1.0]))  # 5th value 1 = lidar-watched
        last = wps[-1] if wps else cur
        face = last[2] + _wrap(goal[2] - last[2])
        wps.append(np.array([goal[0], 0.0, face, travel_lift]))
        wps.append(np.array([goal[0], goal[1], face, goal[3]]))
        return wps

    # ================================================================== arm
    def joint_positions(self) -> np.ndarray:
        return self.d.qpos[self.arm_qadr].copy()

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.d.site_xmat[self.tcp_site])
        return self.d.site_xpos[self.tcp_site].copy(), q

    def solve_ik(self, pos, quat, seed=None) -> tuple[np.ndarray, bool]:
        self.ik.sync(self.d.qpos)
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

    def move_tcp(self, pos, quat, speed: float = 0.25, settle: float = 0.3,
                 guard: Callable[[], str | None] | None = None) -> bool:
        """Smooth straight-line TCP move in WORLD coordinates (blocking).

        One continuous IK path seeded step to step (no posture hopping), joint speed capped.
        `guard` runs every tick; if it returns a reason the arm holds and this returns False.
        """
        self.guard_reason = None
        dt = self.cfg.control_dt
        self.ik.sync(self.d.qpos)
        p0, q0 = self.tcp_pose()
        pos, quat = np.asarray(pos, float), np.asarray(quat, float)
        q_goal, ok, _ = self.ik.solve(pos, quat, self.q_cmd, iters=400)
        if not ok:
            q_goal, ok = self.solve_ik(pos, quat)
            if not ok:
                return False
        n = max(1, int(np.linalg.norm(pos - p0) / (speed * dt)))
        n = max(n, int(np.abs(q_goal - self.q_cmd).max() / (MAX_JOINT_SPEED * 0.8 * dt)))
        q_prev = self.q_cmd.copy()
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)
            self.ik.sync(self.d.qpos)
            qj, okj, _ = self.ik.solve(p0 + (pos - p0) * a, _slerp(q0, quat, a), q_prev, iters=100)
            if not okj:
                qj = q_prev + (q_goal - q_prev) / max(1, n - i + 1)
            if self._command_arm(qj, guard):
                return False
            q_prev = self.q_cmd.copy()
        self.ik.sync(self.d.qpos)
        q_final, okf, _ = self.ik.solve(pos, quat, self.q_cmd, iters=400)
        q_final = q_final if okf else q_goal
        for _ in range(int(settle / dt) + 50):
            if self._command_arm(q_final, guard):
                return False
            if np.abs(self.q_cmd - q_final).max() < 1e-4 and np.abs(self.d.qvel[self.arm_dadr]).max() < 0.02:
                break
        return bool(np.linalg.norm(self.tcp_pose()[0] - pos) < 0.01)

    def _command_arm(self, q_target, guard=None) -> bool:
        lim = MAX_JOINT_SPEED * self.cfg.control_dt
        self.q_cmd = self.q_cmd + np.clip(np.asarray(q_target) - self.q_cmd, -lim, lim)
        self._checked_step()
        if guard is not None:
            reason = guard()
            if reason:
                self.guard_reason = reason
                self.q_cmd = self.d.qpos[self.arm_qadr].copy()
                return True
        return False

    def move_joints(self, q, duration: float = 2.0) -> None:
        q0, q1 = self.q_cmd.copy(), np.asarray(q, float)[:ARM_DOF]
        n = max(1, int(duration / self.cfg.control_dt))
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)
            self.q_cmd = q0 + (q1 - q0) * a
            self._checked_step()

    def home(self) -> None:
        """Arm to its ready pose (lifting straight up first if it is low)."""
        p, q = self.tcp_pose()
        if self.to_robot(p)[2] < 0.25:
            self.move_tcp(self.to_world(self.to_robot(p) * [1, 1, 0] + [0, 0, 0.30]), q)
        self.move_joints(Q_NOMINAL, duration=1.5)

    def stow(self) -> None:
        """Fold the arm over the base for driving (via the ready pose)."""
        self.home()
        self.move_joints(Q_STOW, duration=2.0)

    def unstow(self) -> None:
        self.move_joints(Q_NOMINAL, duration=2.0)

    # ================================================================ gripper
    def gripper_width(self) -> float:
        return float(self.d.qpos[self.finger_qadr].sum())

    def open_gripper(self, width: float | None = None) -> dict:
        """Open fully, or only to `width` metres between the fingers (for tight spots)."""
        self.grip_cmd = GRIP_OPEN if width is None else float(np.clip(width / 0.08, 0, 1) * GRIP_OPEN)
        if self._on_handle:
            self.d.eq_active[self.srv[self._on_handle]["handle_eq"]] = False
            self._latch_tray(self._on_handle, True)
            self._on_handle = None
        released = self._release()
        for _ in range(20):
            self.step()
        return {"width": round(self.gripper_width(), 4), "released": released}

    def close_gripper(self) -> dict:
        self.grip_cmd = GRIP_CLOSED
        for _ in range(25):
            self.step()
        if self._try_handle():
            return {"width": round(self.gripper_width(), 4), "holding": "tray_handle"}
        return {"width": round(self.gripper_width(), 4), "holding": self._try_grasp()}

    def holding(self) -> str | None:
        return self._held

    # ============================================================ trays
    def tray_extension(self, server: str) -> float:
        return float(-self.d.qpos[self.srv[server]["qadr"]])

    def tray_state(self, server: str) -> str:
        e = self.tray_extension(server)
        return "in rack" if e < 0.01 else ("service position" if e > self.cfg.tray.travel - 0.01 else f"out {e:.2f} m")

    def tray_point(self, server: str, p_local) -> np.ndarray:
        bid = self.srv[server]["tray"]
        return self.d.xpos[bid] + _rot(self.d.xquat[bid]) @ np.asarray(p_local, float)

    def tray_handle_position(self, server: str) -> np.ndarray:
        t = self.cfg.tray
        return self.tray_point(server, [t.handle_x - 0.006, 0, t.handle_z])

    def _latch_tray(self, sid: str, on: bool) -> None:
        """Rail latch: a joint equality pinning the tray's slide at its current position."""
        s = self.srv[sid]
        self.m.eq_data[s["lock_eq"]][0] = self.d.qpos[s["qadr"]]  # polycoef y0
        self.m.eq_data[s["lock_eq"]][1:5] = 0.0
        self.d.eq_active[s["lock_eq"]] = on

    def _try_handle(self) -> bool:
        tcp = self.d.site_xpos[self.tcp_site].copy()
        if self.gripper_width() < 0.003:
            return False
        for sid, s in self.srv.items():
            if np.linalg.norm(tcp - self.tray_handle_position(sid)) < self.cfg.grasp_tol:
                ph, qh = self._body_pose(self.hand_body)
                pt, qt = self._body_pose(s["tray"])
                data = self.m.eq_data[s["handle_eq"]]
                data[0:3] = _rot(qh).T @ (tcp - ph)
                data[3:6] = _rot(qt).T @ (tcp - pt)
                self.d.eq_active[s["handle_eq"]] = True
                self._latch_tray(sid, False)
                self._on_handle = sid
                self._log_event("gripped_tray_handle", server=sid)
                return True
        return False

    # ================================================================ parts
    def installed(self, server: str) -> dict[str, str | None]:
        out = {k.split(":")[1]: None for k, s in self.layout.slots.items() if s.server == server}
        for p in self.parts.values():
            if p.where == "slot" and p.slot and p.slot.startswith(server + ":"):
                out[p.slot.split(":")[1]] = p.id
        return out

    def storage(self, where: str) -> dict[str, str | None]:
        """where: "onboard" (robot) or "depot"."""
        out = {s.name: None for s in self.layout.slots.values() if s.where == where}
        for p in self.parts.values():
            if p.where == "slot" and p.slot and p.slot.startswith(where + ":"):
                out[p.slot.split(":")[1]] = p.id
        return out

    def slot_key(self, slot: str) -> str:
        """Resolve a short slot name: "A3"/"F2" -> active server, "CD1"/"CF2" -> onboard, "DD1"/"DF1" -> depot."""
        if ":" in slot:
            return slot
        if slot.startswith("C"):
            return f"onboard:{slot}"
        if slot.startswith("D"):
            return f"depot:{slot}"
        if self.active_server is None:
            raise ValueError("no active server: go_to a server first")
        return f"{self.active_server}:{slot}"

    def occupant(self, key: str) -> str | None:
        for p in self.parts.values():
            if p.where == "slot" and p.slot == key:
                return p.id
        return None

    def slot_world_pose(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        sd = self.layout.slots[key]
        bid = {"tray": None, "onboard": self.link0_body, "depot": self.depot_body}[sd.where]
        if bid is None:
            bid = self.srv[sd.server]["tray"]
        q = self.d.xquat[bid].copy()
        return self.d.xpos[bid] + _rot(q) @ sd.pos, q

    def slot_grasp_point(self, key: str) -> np.ndarray:
        pos, _ = self.slot_world_pose(key)
        t = self.cfg.tray
        dz = t.dimm_size[2] / 2 - 0.009 if self.layout.slots[key].kind == "dimm" else t.fan_size[2] / 2 + 0.012
        return pos + np.array([0, 0, dz])

    def part_position(self, pid: str) -> np.ndarray:
        return self.d.xpos[self.m.body(pid).id].copy()

    def pull_force(self) -> float:
        return self._last_force

    def peak_force(self, reset: bool = False) -> float:
        v = self._peak_force
        if reset:
            self._peak_force = 0.0
        return v

    def set_jam(self, server: str, slot: str, on: bool = True) -> None:
        pid = self.installed(server)[slot]
        if pid is None:
            raise ValueError(f"{server}:{slot} is empty")
        self.parts[pid].jammed = on

    # --- welds -------------------------------------------------------------
    def _set_weld(self, eq_id: int, p1, q1, body2: int) -> None:
        """Weld body1 (desired pose p1,q1) to body2 at its current pose. The weld pins body1's
        origin (data[3:6] = 0) to the matching point on body2 (anchor), plus relative rotation."""
        p2, q2 = self._body_pose(body2)
        self.m.eq_obj2id[eq_id] = body2
        data = self.m.eq_data[eq_id]
        data[0:3] = _rot(q2).T @ (np.asarray(p1) - p2)
        data[3:6] = 0.0
        data[6:10] = _quat_mul(_quat_conj(q1), q2)
        data[10] = 1.0

    def _body_pose(self, bid: int) -> tuple[np.ndarray, np.ndarray]:
        return self.d.xpos[bid].copy(), self.d.xquat[bid].copy()

    def _place_in_slot(self, pid: str, key: str) -> None:
        """Teleport a part into a slot and weld it there (used by reset)."""
        sd = self.layout.slots[key]
        mujoco.mj_kinematics(self.m, self.d)
        pos, quat = self.slot_world_pose(key)
        qa = self.part_qadr[pid]
        self.d.qpos[qa:qa + 3] = pos
        self.d.qpos[qa + 3:qa + 7] = quat
        bid = self.m.body(pid).id
        eq_slot, eq_store = self.eq[(pid, "slot")], self.eq[(pid, "store")]
        if sd.where == "tray":
            self._set_weld(eq_slot, pos, quat, self.srv[sd.server]["tray"])
        else:
            self._set_weld(eq_store, pos, quat, self.link0_body if sd.where == "onboard" else 0)
        self.d.eq_active[eq_slot] = sd.where == "tray"
        self.d.eq_active[eq_store] = sd.where != "tray"
        self.parts[pid].where, self.parts[pid].slot = "slot", key
        del bid

    def _attach_to_hand(self, pid: str) -> None:
        p1, q1 = self._body_pose(self.m.body(pid).id)
        self._set_weld(self.eq[(pid, "hand")], p1, q1, self.hand_body)
        self.d.eq_active[self.eq[(pid, "hand")]] = True
        self._part_collisions(pid, False)  # the weld carries it; finger contacts would only fight it

    def _part_collisions(self, pid: str, on: bool) -> None:
        for g in self.part_geoms[pid]:
            self.m.geom_contype[g] = 2 if on else 0
            self.m.geom_conaffinity[g] = 1 if on else 0

    def _try_grasp(self) -> str | None:
        if self._held:
            return self._held
        if self.gripper_width() < 0.003:
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
            self._grip_z0 = float(self.d.site_xpos[self.tcp_site][2])  # still seated: spring model
        else:
            if part.where == "slot":  # spares lift straight off their shelf
                self.d.eq_active[self.eq[(best, "store")]] = False
                self._log_event("picked_from_storage", part=best, slot=part.slot)
                part.slot = None
            self._attach_to_hand(best)
            part.where = "hand"
        self._log_event("grasped", part=best)
        return best

    def _release(self) -> str | None:
        pid = self._held
        if pid is None:
            return None
        self.d.eq_active[self.eq[(pid, "hand")]] = False
        self._held = None
        self._part_collisions(pid, True)
        part = self.parts[pid]
        if part.where != "slot":
            key = self._capture_slot(pid, storage_only=True)
            if key:
                self._seat(pid, key)
            else:
                part.where = "loose"
        self._log_event("released", part=pid, where=part.where, slot=part.slot)
        return pid

    def _capture_slot(self, pid: str, storage_only: bool = False) -> str | None:
        kind = self.parts[pid].kind
        pos = self.part_position(pid)
        occupied = {p.slot for p in self.parts.values() if p.where == "slot"}
        for key, sd in self.layout.slots.items():
            if sd.kind != kind or key in occupied or (storage_only and sd.where == "tray"):
                continue
            sp, sq = self.slot_world_pose(key)
            if np.linalg.norm((pos - sp)[:2]) < self.cfg.capture_pos_tol and -0.004 < pos[2] - sp[2] < 0.015 \
                    and self._yaw_err(pid, sq) < np.radians(self.cfg.capture_yaw_tol_deg):
                return key
        return None

    def _yaw_err(self, pid: str, slot_quat) -> float:
        rel = _quat_mul(_quat_conj(slot_quat), self.d.xquat[self.m.body(pid).id])
        yaw = abs(np.arctan2(2 * (rel[0] * rel[3] + rel[1] * rel[2]), 1 - 2 * (rel[2] ** 2 + rel[3] ** 2)))
        return min(yaw, abs(np.pi - yaw))  # parts are symmetric front-to-back

    def _seat(self, pid: str, key: str) -> None:
        sd = self.layout.slots[key]
        sp, sq = self.slot_world_pose(key)
        if sd.where == "tray":
            self._set_weld(self.eq[(pid, "slot")], sp, sq, self.srv[sd.server]["tray"])
            self.d.eq_active[self.eq[(pid, "slot")]] = True
        else:
            self._set_weld(self.eq[(pid, "store")], sp, sq, self.link0_body if sd.where == "onboard" else 0)
            self.d.eq_active[self.eq[(pid, "store")]] = True
        part = self.parts[pid]
        part.where, part.slot = "slot", key
        if self._held == pid:
            self.d.eq_active[self.eq[(pid, "hand")]] = False
            self._part_collisions(pid, True)
            self._grip_z0 = float(self.d.site_xpos[self.tcp_site][2])
        if sd.where == "tray":
            h = self.health[sd.server]
            h.parts[pid].pending_post = 3.0
            h.log("info", f"{sd.name}: part inserted ({h.parts[pid].serial}), verifying", slot=sd.name, part=pid)
        self._log_event("seated", part=pid, slot=key)

    def _after_tick(self) -> None:
        self._tick += 1
        force = 0.0
        pid = self._held
        if pid is not None:
            part = self.parts[pid]
            if part.where == "slot" and self.layout.slots[part.slot].where == "tray":
                force = max(0.0, SEAT_STIFFNESS * (float(self.d.site_xpos[self.tcp_site][2]) - self._grip_z0))
                sd = self.layout.slots[part.slot]
                h = self.health[sd.server]
                if part.jammed:
                    if force > self.cfg.jam_break_n and not part.damaged_socket:
                        part.damaged_socket = True
                        h.log("critical", f"{sd.name}: socket damaged by excessive extraction force",
                              slot=sd.name, force_n=round(force, 1))
                        self._log_event("socket_damaged", server=sd.server, slot=sd.name, force_n=round(force, 1))
                elif force > self.cfg.seat_release_n:
                    self.d.eq_active[self.eq[(pid, "slot")]] = False
                    self._attach_to_hand(pid)
                    part.where, part.slot, part.lockout_slot = "hand", None, f"{sd.server}:{sd.name}"
                    h.log("info", f"{sd.name}: part removed ({h.parts[pid].serial})", slot=sd.name, part=pid)
                    self._log_event("unseated", part=pid, server=sd.server, slot=sd.name, force_n=round(force, 1))
            elif part.where == "hand":
                if part.lockout_slot is not None:
                    sp, _ = self.slot_world_pose(part.lockout_slot)
                    if np.linalg.norm(self.part_position(pid) - sp) > 0.04:
                        part.lockout_slot = None
                key = None if part.lockout_slot else self._capture_slot(pid)
                if key and self.layout.slots[key].where == "tray":
                    self._seat(pid, key)
        self._last_force = force
        self._peak_force = max(self._peak_force, force)
        for p in self.parts.values():  # loose parts landing in the onboard bin ride along
            if p.where == "loose":
                rel = self.to_robot(self.part_position(p.id))
                bx, by = self.cfg.robot.bin_center
                sx, sy, sz = self.cfg.robot.bin_size
                vel = np.linalg.norm(self.d.qvel[self.m.jnt_dofadr[self.m.joint(f"{p.id}_free").id]:][:3])
                if abs(rel[0] - bx) < sx / 2 and abs(rel[1] - by) < sy / 2 and rel[2] < sz and vel < 0.05:
                    p1, q1 = self._body_pose(self.m.body(p.id).id)
                    self._set_weld(self.eq[(p.id, "store")], p1, q1, self.link0_body)
                    self.d.eq_active[self.eq[(p.id, "store")]] = True
                    p.where = "bin"
                    self._log_event("in_bin", part=p.id)
        for sid, h in self.health.items():
            for ev in h.update(self.cfg.control_dt, self.installed(sid), self.rng):
                self._log_event(server=sid, **ev)
        if self._tick % 10 == 0:
            self._refresh_leds()
        if self._tick % 25 == 0:
            self._update_work_orders()

    def _update_work_orders(self) -> None:
        """Tickets go open -> in_progress (robot at that server) -> resolved (new part verified OK)."""
        for wo in self.work_orders:
            if wo["status"] == "resolved":
                continue
            pid = self.installed(wo["server"])[wo["slot"]]
            ph = self.health[wo["server"]].parts.get(pid) if pid else None
            if ph is not None and ph.status == "ok" and ph.pending_post <= 0:
                wo["status"], wo["resolved_t"] = "resolved", round(self.time, 1)
                self._log_event("work_order_resolved", id=wo["id"], server=wo["server"], slot=wo["slot"])
            elif self.active_server == wo["server"] and wo["status"] == "open":
                wo["status"] = "in_progress"

    def _log_event(self, event: str, **kw) -> None:
        self.events.append({"t": round(self.time, 2), "event": event, **kw})

    # ================================================================ LEDs
    def _set_led(self, geom: str, state: str) -> None:
        self.m.geom_matid[self.m.geom(geom).id] = self.mat[state]

    def _refresh_leds(self) -> None:
        for sid, sv in self.layout.servers.items():
            any_fault = False
            pfx = f"s{sv.index}_"
            for slot, pid in self.installed(sid).items():
                if pid is None:
                    if slot.startswith("A"):
                        self._set_led(f"{pfx}led_{slot}", "off")
                    continue
                ph = self.health[sid].parts[pid]
                any_fault |= ph.status != "ok"
                if slot.startswith("A"):
                    st = "rebuild" if ph.pending_post > 0 else ("fault" if ph.status != "ok" else "off")
                    self._set_led(f"{pfx}led_{slot}", st)
            self._set_led(f"{pfx}status_led", "fault" if any_fault else "ok")
        for pid, p in self.parts.items():
            if p.kind == "fan":
                ph = next(iter(self.health.values())).parts[pid]
                self._set_led(f"{pid}_led", "fault" if ph.status != "ok" else "ok")
        for (sid, slot), state in self.led_override.items():  # demo: an LED that shows the wrong thing
            geom = self._led_geom(sid, slot)
            if geom:
                self._set_led(geom, state)

    LED_NAMES = {"green": "ok", "ok": "ok", "amber": "fault", "fault": "fault", "red": "red", "off": "off",
                 "blue": "identify", "identify": "identify"}

    def _led_geom(self, server: str, slot: str) -> str | None:
        if slot.startswith("A"):
            return f"s{self.layout.servers[server].index}_led_{slot}"
        pid = self.occupant(f"{server}:{slot}")
        return f"{pid}_led" if pid else None

    def set_led(self, server: str, slot: str, state: str | None) -> dict:
        """Demo: force a slot's LED to show `state` (green/amber/red/off/blue) whatever the part's
        health; None goes back to automatic. Fan LEDs follow the fan module that is in the slot."""
        if server not in self.layout.servers:
            raise ValueError(f"unknown server {server}")
        key = (server, slot)
        if state in (None, "auto"):
            self.led_override.pop(key, None)
        else:
            if state not in self.LED_NAMES:
                raise ValueError(f"state must be one of {sorted(self.LED_NAMES)} or None")
            self.led_override[key] = self.LED_NAMES[state]
        self._refresh_leds()
        mujoco.mj_forward(self.m, self.d)
        return {"server": server, "slot": slot, "led": state or "auto"}

    def slot_leds(self, server: str) -> dict:
        """World position of each slot's status LED (map knowledge, like a rack diagram)."""
        out = {}
        slots = [f"A{i}" for i in range(1, 9)] + [f"F{i}" for i in range(1, 6)]
        for slot in slots:
            g = self._led_geom(server, slot)
            if g:
                out[slot] = self.d.geom_xpos[self.m.geom(g).id].round(4).tolist()
        return out

    def slot_grasp_points(self, server: str) -> dict:
        """World position the gripper closes on for each slot (A1-A8, F1-F5): map knowledge, like
        slot_leds, so perception can check the path to it."""
        slots = [f"A{i}" for i in range(1, 9)] + [f"F{i}" for i in range(1, 6)]
        return {s: self.slot_grasp_point(f"{server}:{s}").round(4).tolist()
                for s in slots if f"{server}:{s}" in self.layout.slots}

    def drop_cable(self, server: str, slot: str, height: float = 0.07) -> dict:
        """Demo: a loose patch cable lands across the open tray, lying along `slot` `height` m above the
        part, right in the gripper's way. Nothing physical stops the arm: only the camera can catch it."""
        g = self.slot_grasp_point(f"{server}:{slot}")
        a1, a8 = (self.slot_grasp_point(f"{server}:A{i}") for i in (1, 8))
        row = (a8 - a1) / max(np.linalg.norm(a8 - a1), 1e-9)       # the DIMM row direction
        axis = np.cross(row, [0, 0, 1.0])                         # lie along the slot, across the row
        yaw = float(np.arctan2(axis[1], axis[0]))
        self.d.mocap_pos[self.cable_mocap] = g + [0, 0, height]
        self.d.mocap_quat[self.cable_mocap] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.m, self.d)
        return {"cable": [server, slot], "height_m": height}

    def clear_cable(self) -> dict:
        """Demo: the loose cable is picked up and out of the way."""
        self.d.mocap_pos[self.cable_mocap] = [0, 0, -10]
        mujoco.mj_forward(self.m, self.d)
        return {"cleared": True}

    def clear_technician(self) -> dict:
        """Demo: the technician walks out of the aisle now (after spawn_technician)."""
        if self._tech_keys:
            p = self.d.mocap_pos[self.tech_mocap][:2].copy()
            self._tech_keys = [(self.time, p.tolist()), (self.time + 1.0, [float(p[0]), -0.75])]
        return {"cleared": True}

    # ================================================================ sensing
    def render(self, camera: str = "wrist", width: int | None = None, height: int | None = None,
               depth: bool = False) -> np.ndarray:
        w, h = width or self.cfg.camera.width, height or self.cfg.camera.height
        if (w, h) not in self._renderers:
            self._renderers[(w, h)] = mujoco.Renderer(self.m, h, w)
        r = self._renderers[(w, h)]
        if depth:
            r.enable_depth_rendering()
        opt = mujoco.MjvOption()
        lidar_on = not (self.m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_SENSOR))
        # lidar rays only in room/follow views and only while the lidar is running (never in the wrist camera)
        opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = lidar_on and camera != "wrist"
        r.update_scene(self.d, camera, scene_option=opt)
        img = r.render().copy()
        if depth:
            r.disable_depth_rendering()
        return img

    def camera_intrinsics(self, camera: str = "wrist", width: int | None = None, height: int | None = None) -> np.ndarray:
        w, h = width or self.cfg.camera.width, height or self.cfg.camera.height
        f = 0.5 * h / np.tan(np.radians(self.m.cam_fovy[self.m.camera(camera).id]) / 2)
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

    def camera_pose(self, camera: str = "wrist") -> tuple[np.ndarray, np.ndarray]:
        cid = self.m.camera(camera).id
        return self.d.cam_xpos[cid].copy(), self.d.cam_xmat[cid].reshape(3, 3).copy()

    def pixel_to_world(self, u: float, v: float, depth: float, camera: str = "wrist") -> np.ndarray:
        K = self.camera_intrinsics(camera)
        x = (u - K[0, 2]) / K[0, 0] * depth
        y = (v - K[1, 2]) / K[1, 1] * depth
        pos, R = self.camera_pose(camera)
        return R @ np.array([x, -y, -depth]) + pos

    # ================================================================ reports
    def server_health(self, server: str) -> dict:
        return self.health[server].report(self.installed(server), self.tray_state(server))

    def fleet_health(self) -> list[dict]:
        out = []
        for sid, sv in self.layout.servers.items():
            r = self.server_health(sid)
            bad = [d["slot"] for d in r["dimms"] + r["fans"] if d["status"] not in ("ok", "verifying", "spinning up")]
            out.append({"server": sid, "row": f"R{sv.row}", "rack": f"R{sv.row}-{sv.rack:02d}", "u": sv.u,
                        "overall": r["overall"], "problem_slots": bad, "cpu_temps_c": r["cpu_temps_c"]})
        return out

    def status(self) -> dict:
        p, q = self.tcp_pose()
        b = self.base_pose()
        return {"time": round(self.time, 2), "base": {"x": round(b[0], 3), "y": round(b[1], 3),
                "heading_deg": round(float(np.degrees(_wrap(b[2]))), 1), "lift_m": round(b[3], 3)},
                "at_server": self.active_server, "tcp_position": p.round(4).tolist(),
                "joints": self.joint_positions().round(4).tolist(), "gripper_width": round(self.gripper_width(), 4),
                "holding": self._held, "pull_force_n": round(self._last_force, 2), "last_collision": self.last_collision,
                "onboard": self.storage("onboard"),
                "trays": {sid: round(self.tray_extension(sid), 3) for sid in self.layout.servers}}

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()


def _slerp(q0, q1, a):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, _quat_conj(q0), q1)
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, dq, 1.0)
    inc = np.zeros(4)
    mujoco.mju_axisAngle2Quat(inc, v / (np.linalg.norm(v) + 1e-12), np.linalg.norm(v) * a)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q0, inc)
    return out


__all__ = ["DataCenterSim", "Stopped", "PartState", "QUAT_DOWN_FINGERS_Y", "rot_z"]
