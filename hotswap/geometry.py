"""Batch box builder + USD materials.

Everything in the rack is an axis-aligned box. We collect boxes, then create them in
one call so building ~200 prims stays fast. Import only after SimulationApp starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim

Color = tuple[float, float, float]


@dataclass
class Box:
    path: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    color: Color
    collide: bool = True
    material: str | None = None  # key into the material library, overrides color


@dataclass
class BoxBatch:
    boxes: list[Box] = field(default_factory=list)

    def add(self, path, center, size, color, collide=True, material=None) -> str:
        self.boxes.append(Box(path, tuple(center), tuple(size), tuple(color), collide, material))
        return path

    def build(self, materials: "MaterialLibrary | None" = None) -> None:
        if not self.boxes:
            return
        Cube(
            paths=[b.path for b in self.boxes],
            sizes=1.0,
            positions=np.array([b.center for b in self.boxes], dtype=np.float32),
            scales=np.array([b.size for b in self.boxes], dtype=np.float32),
            colors=np.array([b.color for b in self.boxes], dtype=np.float32),
        )
        colliders = [b.path for b in self.boxes if b.collide]
        if colliders:
            GeomPrim(colliders, apply_collision_apis=True)
        if materials is not None:
            for b in self.boxes:
                if b.material:
                    materials.bind(b.path, b.material)


def ensure_xform(path: str) -> None:
    UsdGeom.Xform.Define(stage_utils.get_current_stage(backend="usd"), path)


class MaterialLibrary:
    """UsdPreviewSurface materials under /World/Looks. Emissive ones render as glowing LEDs."""

    def __init__(self, root: str = "/World/Looks") -> None:
        self.stage = stage_utils.get_current_stage(backend="usd")
        self.root = root
        ensure_xform(root)
        self._mats: dict[str, UsdShade.Material] = {}

    def add(
        self,
        name: str,
        diffuse: Color,
        emissive: Color = (0.0, 0.0, 0.0),
        roughness: float = 0.5,
        metallic: float = 0.0,
    ) -> None:
        path = f"{self.root}/{name}"
        mat = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self._mats[name] = mat

    def bind(self, prim_path: str, name: str) -> None:
        prim = self.stage.GetPrimAtPath(prim_path)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            self._mats[name], UsdShade.Tokens.strongerThanDescendants
        )


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """wxyz orientation for a USD camera at `eye` looking at `target` (camera looks down -Z)."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return mat_to_quat(np.column_stack([r, u, -f]))


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> wxyz quaternion."""
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q = [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
