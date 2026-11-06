"""Build display polylines without neutral reference-plane backtracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.geometries import Plane
from optiland.interactions.refractive_reflective_model import RefractiveReflectiveModel
from optiland.materials import IdealMaterial
from optiland.propagation.homogeneous import HomogeneousPropagation
from optiland.rays.real_rays import RealRays
from optiland.surfaces.standard_surface import Surface

if TYPE_CHECKING:
    from collections.abc import Sequence


def neutral_reference_mask(surfaces: Sequence[Surface]) -> np.ndarray:
    """Identify removable reference vertices conservatively.

    Only ordinary planes with identity interaction in matching homogeneous
    ideal media qualify. Stops, apertures, coatings, phase/scattering events,
    endpoints and custom implementations retain their recorded vertices.
    """
    mask = np.zeros(len(surfaces), dtype=bool)
    for index in range(1, len(surfaces) - 1):
        surface = surfaces[index]
        model = surface.interaction_model
        if (
            type(surface) is not Surface
            or type(surface.geometry) is not Plane
            or type(model) is not RefractiveReflectiveModel
            or model.is_reflective
            or model.coating is not None
            or model.bsdf is not None
            or surface.aperture is not None
            or surface.is_stop
        ):
            continue
        before, after = surface.material_pre, surface.material_post
        if type(before) is not IdealMaterial or type(after) is not IdealMaterial:
            continue
        if (
            type(before.propagation_model) is not HomogeneousPropagation
            or type(after.propagation_model) is not HomogeneousPropagation
        ):
            continue
        n_before, n_after = be.to_numpy(before.index), be.to_numpy(after.index)
        k_before, k_after = be.to_numpy(before.absorp), be.to_numpy(after.absorp)
        mask[index] = (
            np.isfinite(n_before).all()
            and (n_before > 0).all()
            and np.isfinite(k_before).all()
            and (k_before >= 0).all()
            and np.array_equal(n_before, n_after)
            and np.array_equal(k_before, k_after)
        )
    return mask


def _collinear(first: np.ndarray, middle: np.ndarray, last: np.ndarray) -> bool:
    """Check 3D collinearity, including reversed and zero-length steps."""
    if not np.isfinite([first, middle, last]).all():
        return False
    incoming, outgoing = middle - first, last - middle
    lengths = np.hypot.reduce(incoming), np.hypot.reduce(outgoing)
    if lengths[0] == 0 or lengths[1] == 0:
        return True
    # Unit directions avoid overflowing the cross product of long segments.
    sine = np.linalg.norm(np.cross(incoming / lengths[0], outgoing / lengths[1]))
    return bool(sine <= 1e-10)


def physical_ray_path(
    points: np.ndarray,
    intensity: np.ndarray,
    neutral: np.ndarray,
    *,
    hide_vignetted: bool = False,
) -> np.ndarray:
    """Return a copied physical display path, preserving real interaction hits.

    The exact first zero-intensity hit remains the endpoint. Invalid intensity
    terminates the path before that sample. Nonfinite coordinates remain gaps;
    they are never bridged by reference-point removal. Prescription and recorded
    trace data, including optical-path accumulation, are not changed.
    """
    points = np.array(points, dtype=float, copy=True)
    if points.shape != (len(intensity), 3) or len(neutral) != len(intensity):
        raise ValueError(
            "Ray points, intensity and surface mask must have equal length."
        )

    blocked = np.flatnonzero((intensity <= 0) | ~np.isfinite(intensity))
    if blocked.size:
        if hide_vignetted:
            return np.empty((0, 3))
        first = blocked[0]
        end = first + 1 if intensity[first] == 0 else first
        points = points[:end]
    points[~np.isfinite(points).all(axis=1)] = np.nan

    keep: list[int] = []
    for index in range(len(points)):
        keep.append(index)
        while len(keep) >= 3 and neutral[keep[-2]]:
            if not _collinear(points[keep[-3]], points[keep[-2]], points[keep[-1]]):
                break
            keep.pop(-2)
    return points[keep]


def curved_segments(
    surfaces: Sequence[Surface], wavelength: float | None
) -> dict[int, np.ndarray]:
    """Sample ray paths through media whose propagation model curves them.

    Uses the ray state recorded on each surface by the latest trace.

    Args:
        surfaces: The traced surfaces, in order.
        wavelength: The traced wavelength in µm. ``None`` (unknown) keeps all
            segments straight.

    Returns:
        Map from the index of the surface ending a curved medium to global
        points of shape ``(num_samples, num_rays, 3)``.
    """
    segments = {}
    if wavelength is None:
        return segments
    for index in range(1, len(surfaces)):
        start, surface = surfaces[index - 1], surfaces[index]
        if be.size(start.x) == 0:
            continue
        rays = RealRays(
            start.x, start.y, start.z, start.L, start.M, start.N,
            start.intensity, be.full_like(start.x, wavelength),
        )  # fmt: skip
        path = surface.material_pre.propagation_model.display_path(rays, surface)
        if path is not None:
            segments[index] = np.asarray(be.to_numpy(path), dtype=float)
    return segments


def splice_curved_segments(
    points: np.ndarray,
    intensity: np.ndarray,
    neutral: np.ndarray,
    segments: dict[int, np.ndarray],
    ray: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert one ray's interior curved-path samples between its surface hits.

    Inserted points carry the intensity at the start of their medium and are
    never neutral reference vertices.

    Args:
        points: Surface hits of the ray, shape ``(num_surfaces, 3)``.
        intensity: Intensity at each hit.
        neutral: Removable reference-vertex mask, one entry per hit.
        segments: Output of :func:`curved_segments`.
        ray: Index of the ray in the traced bundle.

    Returns:
        The expanded ``(points, intensity, neutral)``.
    """
    rows, values, flags = [], [], []
    for index in range(len(points)):
        if index in segments:
            interior = segments[index][1:-1, ray]
            interior = interior[np.isfinite(interior).all(axis=1)]
            rows.append(interior)
            values.append(np.full(len(interior), intensity[index - 1]))
            flags.append(np.zeros(len(interior), dtype=bool))
        rows.append(points[index : index + 1])
        values.append(intensity[index : index + 1])
        flags.append(neutral[index : index + 1])
    return np.concatenate(rows), np.concatenate(values), np.concatenate(flags)
