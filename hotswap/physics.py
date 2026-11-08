"""Caddy slide physics: rigid bodies on prismatic joints + a resistance model.

Each caddy's Xform sits at identity when seated, so its world X position IS the slide
displacement (negative = pulled toward the robot). A linear drive on the joint provides
all resistance, which means the pull force can be computed exactly from the same
numbers PhysX uses:

    force = stiffness * (-x) + damping * (-v)      (x, v along the slide)

    healthy : detent spring for the first 10 mm (~15 N to unseat), then just damping
    jammed  : very stiff spring for the whole travel (~20 N per mm)
    released: once pulled out 15 cm the joint is disabled and the caddy is carried freely
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdPhysics, UsdShade

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import RigidPrim

from .config import CaddyPhysicsCfg
from .rack import BayInfo


@dataclass
class CaddyState:
    jammed: bool = False
    released: bool = False
    broken: bool = False
    stiffness: float = -1.0  # last value written to USD
    force: float = 0.0
    x: float = 0.0


@dataclass
class CaddySystem:
    bays: list[BayInfo]
    cfg: CaddyPhysicsCfg
    states: list[CaddyState] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stage = stage_utils.get_current_stage(backend="usd")
        self.index = {(b.cabinet, b.bay): i for i, b in enumerate(self.bays)}
        self.states = [CaddyState() for _ in self.bays]
        self._joints = [f"{b.caddy_path}/slide" for b in self.bays]
        self._rigid: RigidPrim | None = None

    # ------------------------------------------------------------------ setup
    def setup(self, shell_paths: list[str]) -> None:
        """Call after the caddy geometry exists and BEFORE play()."""
        st = self.stage
        friction = self._tab_material()
        for b, jpath in zip(self.bays, self._joints):
            prim = st.GetPrimAtPath(b.caddy_path)
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(self.cfg.mass)
            joint = UsdPhysics.PrismaticJoint.Define(st, jpath)
            joint.CreateBody1Rel().SetTargets([Sdf.Path(b.caddy_path)])  # body0 empty = world
            joint.CreateAxisAttr("X")
            joint.CreateLowerLimitAttr(-0.30)
            joint.CreateUpperLimitAttr(0.0)
            joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateDampingAttr(self.cfg.damping)
            drive.CreateStiffnessAttr(self.cfg.detent_stiffness)
            UsdShade.MaterialBindingAPI.Apply(st.GetPrimAtPath(b.tab_path)).Bind(
                friction, UsdShade.Tokens.strongerThanDescendants, "physics"
            )
        self._collision_groups(shell_paths)

    def _tab_material(self) -> UsdShade.Material:
        mat = UsdShade.Material.Define(self.stage, "/World/PhysicsMaterials/tab_grip")
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(1.5)
        api.CreateDynamicFrictionAttr(1.2)
        PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr("max")
        return mat

    def _collision_groups(self, shell_paths: list[str]) -> None:
        """Caddies don't collide with the server shell or each other (the joint guides them)."""
        caddies = UsdPhysics.CollisionGroup.Define(self.stage, "/World/CollisionGroups/caddies")
        shell = UsdPhysics.CollisionGroup.Define(self.stage, "/World/CollisionGroups/server_shells")
        inc = caddies.GetCollidersCollectionAPI().CreateIncludesRel()
        for b in self.bays:
            inc.AddTarget(Sdf.Path(b.caddy_path))
        inc_s = shell.GetCollidersCollectionAPI().CreateIncludesRel()
        for p in shell_paths:
            inc_s.AddTarget(Sdf.Path(p))
        rel = caddies.CreateFilteredGroupsRel()
        rel.AddTarget(caddies.GetPath())
        rel.AddTarget(shell.GetPath())

    # ---------------------------------------------------------------- runtime
    @property
    def rigid(self) -> RigidPrim:
        if self._rigid is None:
            self._rigid = RigidPrim([b.caddy_path for b in self.bays])
        return self._rigid

    def update(self) -> None:
        """Call once per physics step."""
        pos, _ = self.rigid.get_world_poses()
        vel, _ = self.rigid.get_velocities()
        xs, vs = pos.numpy()[:, 0], vel.numpy()[:, 0]
        c = self.cfg
        for i, (s, x, v) in enumerate(zip(self.states, xs, vs)):
            s.x = float(x)
            if s.released:
                s.force = 0.0
                continue
            if s.jammed:
                k = c.jam_stiffness
            elif -x < c.detent_len:
                k = c.detent_stiffness
            else:
                k = 0.0
            if k != s.stiffness:
                self._set_attr(self._joints[i], "drive:linear:physics:stiffness", k)
                s.stiffness = k
            s.force = max(0.0, k * (-x) - c.damping * v)
            if s.jammed and not s.broken and s.force > c.breakaway_force:
                s.broken = True
                self.events.append({"event": "connector_broken", **self._addr(i), "force_n": round(s.force, 1)})
            if not s.jammed and -x > c.release_at:
                s.released = True
                self._set_attr(self._joints[i], "physics:jointEnabled", False)
                self.events.append({"event": "caddy_released", **self._addr(i)})

    def _set_attr(self, path: str, name: str, value) -> None:
        self.stage.GetPrimAtPath(path).GetAttribute(name).Set(value)

    def _addr(self, i: int) -> dict:
        return {"cabinet": self.bays[i].cabinet, "bay": self.bays[i].bay}

    # -------------------------------------------------------------- controls
    def state(self, cabinet: int, bay: int) -> CaddyState:
        return self.states[self.index[(cabinet, bay)]]

    def set_jam(self, cabinet: int, bay: int, on: bool) -> None:
        self.state(cabinet, bay).jammed = on

    def max_force(self) -> float:
        return max((s.force for s in self.states), default=0.0)

    def reset(self) -> None:
        n = len(self.bays)
        self.rigid.set_world_poses(positions=np.zeros((n, 3)), orientations=np.tile([1.0, 0, 0, 0], (n, 1)))
        self.rigid.set_velocities(np.zeros((n, 3)), np.zeros((n, 3)))
        for i, s in enumerate(self.states):
            if s.released:
                self._set_attr(self._joints[i], "physics:jointEnabled", True)
            self.states[i] = CaddyState()
        self.events.clear()


def shell_paths_for(cabinets: int, root_fn) -> list[str]:
    return [f"{root_fn(c)}/StorageServer/Shell" for c in range(cabinets)]

