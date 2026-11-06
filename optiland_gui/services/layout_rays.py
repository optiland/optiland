"""GUI layout sampling with an explicit chief identity and physical paths."""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.distribution import create_distribution
from optiland.utils import resolve_fields
from optiland.visualization.system.ray_path import (
    neutral_reference_mask,
    physical_ray_path,
)
from optiland_gui.services.job_records import check_cancelled


def ray_counts(distribution):
    """Supported total rays per field, including the chief."""
    if distribution in ("line_x", "line_y"):
        return tuple(range(1, 102, 2))
    if distribution == "hexapolar":
        return tuple(1 + 3 * n * (n + 1) for n in range(6))
    if distribution in ("random", "ring"):
        return tuple(range(1, 102))
    raise ValueError(f"Unsupported layout distribution: {distribution}")


def normalize_ray_count(count, distribution):
    """Round up to a supported bundle, clamped at the GUI's sampling limit."""
    counts = ray_counts(distribution)
    return next((value for value in counts if value >= count), counts[-1])


def layout_pupil(count, distribution):
    """Return pupil coordinates and the single chief index for this layout."""
    count = normalize_ray_count(count, distribution)
    if count == 1:
        return be.zeros(1), be.zeros(1), 0
    sample = create_distribution(distribution)
    if distribution == "hexapolar":
        sample.generate_points(ray_counts(distribution).index(count))
        return sample.x, sample.y, 0
    if distribution in ("random", "ring"):
        sample.generate_points(count - 1)
        return (
            be.concatenate((be.zeros(1), sample.x)),
            be.concatenate((be.zeros(1), sample.y)),
            0,
        )
    sample.generate_points(count)
    chief = count // 2
    # Some backend linspace implementations leave a tiny residual at zero.
    center = be.arange(count) == chief
    return be.where(center, 0.0, sample.x), be.where(center, 0.0, sample.y), chief


def trace_layout_rays(rays, count, distribution, progress, cancelled):
    """Trace an owned model; retain chief identity through gaps and blocking."""
    optic = rays.optic
    neutral = neutral_reference_mask(list(optic.surfaces))
    paths = []
    for color_index, field in enumerate(resolve_fields(optic, "all")):
        check_cancelled(cancelled)
        progress("Tracing layout rays", color_index, len(optic.fields))
        px, py, chief = layout_pupil(count, distribution)
        if len(px) == 1:
            # Auto-sized glass still needs an outline when only its central ray
            # is displayed. These four rim probes size outlines only.
            optic.trace_generic(
                *field.coord,
                Px=be.array([1.0, -1.0, 0.0, 0.0]),
                Py=be.array([0.0, 0.0, 1.0, -1.0]),
                wavelength=optic.primary_wavelength,
            )
            rays._process_traced_rays()
            check_cancelled(cancelled)
        optic.trace_generic(
            *field.coord, Px=px, Py=py, wavelength=optic.primary_wavelength
        )
        rays._process_traced_rays()
        x, y, z, intensity = (
            be.to_numpy(values) for values in (rays.x, rays.y, rays.z, rays.i)
        )
        for index in range(len(px)):
            path = physical_ray_path(
                np.column_stack((x[:, index], y[:, index], z[:, index])),
                intensity[:, index],
                neutral,
            )
            paths.append(
                {
                    "xyz": path,
                    "chief": index == chief,
                    "field": field.coord,
                    "color_index": color_index,
                }
            )
    check_cancelled(cancelled)
    return paths
