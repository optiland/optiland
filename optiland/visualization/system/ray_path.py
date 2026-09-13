"""Build display polylines without neutral reference-plane backtracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.geometries import Plane
from optiland.interactions.refractive_reflective_model import RefractiveReflectiveModel
from optiland.materials import IdealMaterial
from optiland.propagation.homogeneous import HomogeneousPropagation
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
