"""A loose cable that drops down in front of the rack and hangs there until cleared.

It's a kinematic rigid body (it collides with the arm but nothing pushes it), animated
by us. That makes the disturbance identical every take, which is what you want when
recording a single-shot demo video.
"""

from __future__ import annotations

import math

import numpy as np
from pxr import UsdPhysics

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Capsule
from isaacsim.core.experimental.prims import GeomPrim, XformPrim

PATH = "/World/Obstacle/cable"
PARKED = np.array([0.0, 0.0, 20.0])  # far above the scene when not in use
LENGTH = 0.55
RADIUS = 0.012
DROP_TIME = 0.5  # s from appearing to hanging in place


class CableObstacle:
    def __init__(self) -> None:
        Capsule(PATH, radii=RADIUS, heights=LENGTH, axes="Z", colors=[(0.02, 0.02, 0.02)], positions=PARKED)
        GeomPrim(PATH, apply_collision_apis=True)
        prim = stage_utils.get_current_stage(backend="usd").GetPrimAtPath(PATH)
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
        self._xform = XformPrim(PATH)
        self.active = False
        self._t = 0.0
        self._delay = 0.0
        self._target = PARKED.copy()

    def drop(self, world_xyz, delay: float = 0.0) -> None:
        """Drop so the cable's middle ends up at `world_xyz`."""
        self._target = np.asarray(world_xyz, dtype=float)
        self._t = 0.0
        self._delay = delay
        self.active = True

    def clear(self) -> None:
        self.active = False
        self._set(PARKED)

    def update(self, dt: float) -> None:
        if not self.active:
            return
        self._t += dt
        t = self._t - self._delay
        if t < 0:
            return
        if t < DROP_TIME:  # fall from 0.6 m above, easing in
            a = t / DROP_TIME
            pos = self._target + np.array([0, 0, 0.6 * (1 - a * a)])
        else:  # gentle damped sway once it's hanging
            s = t - DROP_TIME
            sway = 0.02 * math.exp(-0.6 * s) * math.sin(4.0 * s)
            pos = self._target + np.array([0, sway, 0])
        self._set(pos)

    def world_position(self) -> np.ndarray | None:
        if not self.active or self._t < self._delay:
            return None
        return self._xform.get_world_poses()[0].numpy()[0]

    def _set(self, p) -> None:
        self._xform.set_world_poses(positions=np.asarray(p, dtype=np.float32).reshape(1, 3))
