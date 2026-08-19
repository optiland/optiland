"""Shared nearest-detector dispatch for Non-Sequential Raytracing.

Both reference backends need to find, for every ray, the nearest detector it
hits (if any) among ``scene.detectors``. Before PR10 this routine was
duplicated almost verbatim in ``ArrayBackend._intersect_detectors`` and
``TorchBackend._intersect_detectors``, with subtly different grad-attachment
semantics: the NumPy version cast ``t``/normals to float64 NumPy (harmless
there, since NumPy has no autograd), while the Torch version kept ``t``
attached to the graph because the splatted landing position is
``origin + t * direction`` and detaching ``t`` silently drops the
``direction * dt/dtheta`` term from every spatial loss.

This module keeps exactly one implementation, using the Torch-safe (grad
-preserving) semantics unconditionally -- harmless under the NumPy backend,
required under Torch.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy

if TYPE_CHECKING:
    from optiland.nonsequential.detectors.base import BaseDetector
    from optiland.nonsequential.ray_bundle import NSQRayBundle


def intersect_detectors(
    rays: NSQRayBundle,
    detectors: list[BaseDetector],
) -> tuple[object, object, np.ndarray]:
    """Find the nearest detector intersection for every ray.

    Only the *dispatch* (which detector, and whether it beats the nearest
    component) is decided in NumPy -- that choice is a discrete visibility
    event with no gradient anyway. The returned ``t_min``/``hit_normals``
    stay attached to the active backend's autograd graph (a no-op detail
    under NumPy).

    Args:
        rays: Current ray bundle.
        detectors: ``scene.detectors``.

    Returns:
        ``(t_min, hit_normals, detector_indices)`` where ``t_min`` and
        ``hit_normals`` are backend arrays and ``detector_indices`` is a
        NumPy int32 array (``-1`` where no detector was hit).
    """
    N = rays.num_rays
    t_min = be.ones(N) * be.inf
    t_min_np = np.full(N, np.inf, dtype=np.float64)
    hit_normals = be.zeros((N, 3))
    det_indices = np.full(N, -1, dtype=np.int32)

    for i, det in enumerate(detectors):
        t_d, normals_d, hit_d = det.intersect(rays)
        t_d_np = to_numpy(t_d).astype(np.float64)
        hit_d_np = to_numpy(hit_d).astype(bool)
        better_np = hit_d_np & (t_d_np < t_min_np)
        better = be.array(better_np)

        t_min = be.where(better, t_d, t_min)
        hit_normals = be.where(better[:, None], normals_d, hit_normals)
        t_min_np = np.where(better_np, t_d_np, t_min_np)
        det_indices = np.where(better_np, i, det_indices)

    return t_min, hit_normals, det_indices


def detector_absorb_mask(
    det_idx: np.ndarray, detectors: list[BaseDetector]
) -> np.ndarray:
    """Per-ray absorb flag of the detector each ray hit (D-10 ``absorb``).

    Args:
        det_idx: Per-ray index into ``detectors`` of the nearest-hit
            detector, or ``-1`` where no detector was hit. NumPy int array.
        detectors: ``scene.detectors``.

    Returns:
        Boolean NumPy array, shape matching ``det_idx``: True where the hit
        detector (if any) absorbs the ray. Rays with no detector hit are
        reported as ``True`` (irrelevant -- the caller only consults this
        where a detector was actually hit).
    """
    if len(detectors) == 0:
        return np.ones_like(det_idx, dtype=bool)
    absorb_per_detector = np.array([bool(d.absorb) for d in detectors], dtype=bool)
    safe_idx = np.clip(det_idx, 0, len(detectors) - 1)
    return absorb_per_detector[safe_idx]
