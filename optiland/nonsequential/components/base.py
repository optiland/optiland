"""Base component for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.nonsequential._utils import as_param

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.bsdf.base import BaseBSDF
    from optiland.nonsequential.components.geometry.base import AABB, ComponentGeometry
    from optiland.nonsequential.ir.bsdf_ir import BsdfIR
    from optiland.nonsequential.ir.scene_ir import SamplingPolicy
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.rng import NSQRng


class BaseComponent(ABC):
    """Abstract base class for all non-sequential optical components.

    Components define the geometry and optical interaction (reflection,
    refraction, absorption) for a surface in the NSQ scene.

    Attributes:
        cs: Coordinate system defining position and orientation in global frame.
        geometry: Shape of the component surface.
        material_front: Medium on the front side (normal-facing side).
        material_back: Medium on the back side.
        bsdf: Optional scatter model. None means specular-only.
        name: Optional human-readable label.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        material_front: NSQMaterial,
        material_back: NSQMaterial,
        bsdf: BaseBSDF | None = None,
        name: str = "",
        scatter_fraction: float = 1.0,
    ) -> None:
        """Initialize BaseComponent.

        Args:
            cs: Coordinate system for this component.
            geometry: Surface geometry.
            material_front: Medium on the front (normal-facing) side.
            material_back: Medium on the back side.
            bsdf: Optional BSDF scatter model.
            name: Optional label for this component.
            scatter_fraction: Probability that a ray striking this surface is
                routed through ``bsdf`` instead of the specular path.
                Differentiable: a ``torch.Tensor`` with
                ``requires_grad=True`` stays attached to the autograd graph
                -- see ``RefractiveComponent.interact``/
                ``ReflectiveComponent.interact`` for the detached-sample /
                attached-weight estimator that makes
                ``d(flux)/d(scatter_fraction)`` correct rather than zero.
        """
        self.cs = cs
        self.geometry = geometry
        self.material_front = material_front
        self.material_back = material_back
        self.bsdf = bsdf
        self.name = name
        self.scatter_fraction = as_param(scatter_fraction)

    def intersect(
        self, rays: NSQRayBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find the nearest intersection of alive rays with this component.

        Transforms rays to local frame, delegates to geometry, then
        transforms normals back to global frame.

        Args:
            rays: The ray bundle in global coordinates.

        Returns:
            Tuple (t, normals, hit_mask, n_geom) in global frame:
                - t: Per-ray distances [mm], shape (N,). inf if no hit.
                - normals: Surface normals in global frame, shape (N, 3).
                - hit_mask: Boolean hit mask, shape (N,).
                - n_geom: Geometric (unflipped, direction-independent)
                    surface normal in global frame, shape (N, 3). See
                    :meth:`ComponentGeometry.ray_intersect`.
        """
        translation, rot = _get_transform(self.cs)

        # Global ray data as (N, 3) arrays
        positions_g = be.stack([rays.x, rays.y, rays.z], axis=1)
        directions_g = be.stack([rays.L, rays.M, rays.N], axis=1)

        t_be = be.array(translation)
        R_be = be.array(rot)

        # Transform to local frame
        positions_l = (positions_g - t_be) @ R_be
        directions_l = directions_g @ R_be

        t_hit, normals_l, hit_mask, n_geom_l = self.geometry.ray_intersect(
            positions_l, directions_l
        )

        # T_EPSILON guard: prevent self-intersection after surface crossing
        T_EPSILON = 1e-9
        inf_like = be.ones_like(t_hit) * be.inf
        t_hit = be.where(t_hit > T_EPSILON, t_hit, inf_like)
        hit_mask = hit_mask & (t_hit > T_EPSILON)

        # Dead rays can't hit
        t_hit = be.where(rays.alive, t_hit, inf_like)
        hit_mask = hit_mask & rays.alive

        # Transform normals back to global: n_global_row = n_local_row @ R^T
        normals_g = normals_l @ R_be.T
        n_geom_g = n_geom_l @ R_be.T

        return t_hit, normals_g, hit_mask, n_geom_g

    @abstractmethod
    def interact(
        self,
        rays: NSQRayBundle,
        t: np.ndarray,
        normals: np.ndarray,
        hit_mask: np.ndarray,
        rng: NSQRng,
        bsdf_ir: BsdfIR,
        n_geom: np.ndarray,
        sampling: SamplingPolicy | None = None,
        forced_branch: str | None = None,
    ) -> None:
        """Apply optical interaction at hit points (in-place).

        Updates ray positions, directions, flux, n_current, bounce, and
        alive status for rays that hit this component.

        This is a private implementation detail of the reference NumPy/Torch
        interpreters (``optiland.nonsequential.ir.interpreter
        .apply_primitive_interactions``), not the engine's public dispatch
        contract -- a non-Python backend never calls it. ``bsdf_ir`` is what
        makes the *dispatch* IR-driven: whether to route a hit ray through
        ``self.bsdf`` is decided from ``bsdf_ir.kind`` (verified to match
        ``self.bsdf``'s actual type by the caller), not from a bare
        ``self.bsdf is not None`` check.

        Args:
            rays: Ray bundle to update in-place.
            t: Hit distances [mm], shape (N,).
            normals: Surface normals in global frame, shape (N, 3).
            hit_mask: True for rays that hit this component, shape (N,).
            rng: Keyed PCG32 RNG for stochastic interactions.
            bsdf_ir: This surface's lowered BSDF descriptor (``BsdfIR(kind=
                "none")`` when no scatter model is attached), matching
                ``self.bsdf``.
            n_geom: Geometric (unflipped) surface normal in global frame,
                shape (N, 3): points from ``material_front`` toward
                ``material_back``. ``RefractiveComponent`` uses this,
                not index proximity, to determine which material a ray is
                entering.
            sampling: The scene's rare-path sampling policy.
                Only ``RefractiveComponent`` consults it, to resolve the
                Fresnel reflect/transmit branch probability; ``None`` is
                treated as the default (unbiased, ``reflect_prob="fresnel"``)
                policy.
            forced_branch: ``"reflect"`` or ``"transmit"`` to deterministically
                force the branch instead of drawing it, or ``None`` for the
                normal stochastic draw. Used only by the NumPy forward
                engine's bounded-splitting orchestration (PR11;
                :mod:`optiland.nonsequential.ir.interpreter`) to build both
                children of a split ray; ignored by every component except
                ``RefractiveComponent``.
        """

    @property
    def bounding_box(self) -> AABB:
        """Axis-aligned bounding box in global coordinates.

        Returns:
            AABB for this component.
        """
        transform = _get_transform(self.cs)
        return self.geometry.bounding_box(transform)


def _get_transform(cs: CoordinateSystem) -> tuple[np.ndarray, np.ndarray]:
    """Extract (translation, rotation_matrix) from a CoordinateSystem.

    Returns plain numpy float64 arrays regardless of the current
    optiland backend.

    Args:
        cs: The coordinate system.

    Returns:
        Tuple (translation [mm], rotation_matrix) as numpy float64 arrays.
        rotation_matrix is (3, 3) and transforms column vectors local->global.
    """
    from optiland.backend.utils import to_numpy  # noqa: PLC0415

    t_be, R_be = cs.get_effective_transform()
    translation = to_numpy(t_be).astype(np.float64)
    rotation = to_numpy(R_be).astype(np.float64)
    return translation, rotation
