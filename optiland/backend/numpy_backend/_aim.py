"""NumPy-only fast path of the paraxial aim directions. The NumPy code of
``ParaxialRayAimer._aim_rays_impl`` (its helper ``_unit_directions`` in
``optiland/rays/ray_aiming/paraxial.py``) is the reference, and this module
must give its results bit for bit.

``ParaxialRayAimer`` sends each ray from its origin toward its aim point on
the entrance pupil. The NumPy code makes one pass over all rays, with a new
array, for each step: the difference vector, its length, the unit vector,
the test for a zero length and the reversal of a ray whose aim point is
behind its origin. The kernel ``_directions`` does all steps for one ray at
a time. Each ray gets the same expressions in the same order, without
fastmath, so each result has the same bits. A NaN is at the same position,
but compiled code can give it another sign bit.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
from numba import njit

from optiland.backend.numpy_backend._compiled_math import _sqrt

if TYPE_CHECKING:
    from collections.abc import Sequence

# The reversal test (argument flip_kind of _directions). It finds a ray whose
# aim point is behind its origin.
_NO_FLIP = 0  # no test
_FLIP_Z = 1  # the z component of (aim point - origin) is negative
_FLIP_AXIS = 2  # the component of (aim point - origin) along the axis is negative


@njit(error_model="numpy", nogil=True, cache=True)
def _directions(
    origins: tuple[np.ndarray, np.ndarray, np.ndarray],
    aim_points: tuple[np.ndarray, np.ndarray, np.ndarray],
    min_distance: float,
    flip_kind: int,
    axis: tuple[float, float, float],
    directions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Write the unit vector from each ray origin to its aim point.

    A ray whose aim point is less than ``min_distance`` from its origin gets
    the direction (0, 0, 1) before the reversal test. The reversal test can
    then reverse it, as the NumPy code does. The kernel releases the GIL, so
    several Python threads can aim rays at the same time.

    Args:
        origins: x0, y0 and z0 of the rays.
        aim_points: x1, y1 and z1 of the aim points.
        min_distance: The distance below which a ray gets (0, 0, 1).
        flip_kind: The reversal test: ``_NO_FLIP``, ``_FLIP_Z`` or
            ``_FLIP_AXIS``. A ray that the test finds gets the opposite
            direction.
        axis: The axis of ``_FLIP_AXIS``, as three floats.
        directions: L, M and N. They receive the direction cosines.
    """
    x0, y0, z0 = origins
    x1, y1, z1 = aim_points
    ax, ay, az = axis
    L, M, N = directions
    for j in range(x0.shape[0]):
        dx = x1[j] - x0[j]
        dy = y1[j] - y0[j]
        dz = z1[j] - z0[j]
        distance = _sqrt(dx * dx + dy * dy + dz * dz)
        zero = distance < min_distance
        lj = 0.0 if zero else dx / distance
        mj = 0.0 if zero else dy / distance
        nj = 1.0 if zero else dz / distance
        if flip_kind != _NO_FLIP:
            along = dz if flip_kind == _FLIP_Z else dx * ax + dy * ay + dz * az
            if along < 0:
                lj, mj, nj = -lj, -mj, -nj
        L[j] = lj
        M[j] = mj
        N[j] = nj


def _is_float64_vector(value: object) -> bool:
    """Return True for a C-contiguous 1-D float64 ``np.ndarray`` (exact type)."""
    return (
        type(value) is np.ndarray
        and value.dtype == np.float64
        and value.ndim == 1
        and value.flags.c_contiguous
    )


def _axis_values(axis: Sequence[object]) -> tuple[float, float, float] | None:
    """Return the first three components of ``axis`` as floats, or None.

    The NumPy code multiplies each component with an array of one value per
    ray. A number or a 0-d or 1-D array of one value keeps the shape of that
    array, so it is in the domain. Other values give None.
    """
    values = []
    for k in range(3):
        component = np.asarray(axis[k], dtype=np.float64)
        if component.ndim > 1 or component.size != 1:
            return None
        values.append(float(component.reshape(-1)[0]))
    return values[0], values[1], values[2]


def compiled_directions(
    x0: object,
    y0: object,
    z0: object,
    x1: object,
    y1: object,
    z1: object,
    flip: bool,
    axis: Sequence[object] | None,
    min_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the directions of ``_directions``, or None outside the domain.

    Args:
        x0: The x coordinates of the ray origins.
        y0: The y coordinates of the ray origins.
        z0: The z coordinates of the ray origins.
        x1: The x coordinates of the aim points.
        y1: The y coordinates of the aim points.
        z1: The z coordinates of the aim points.
        flip: Whether a ray whose aim point is behind its origin gets the
            opposite direction.
        axis: The axis of that test, as three numbers or arrays of one
            number, or None for the z axis.
        min_distance: The distance below which a ray gets (0, 0, 1). The
            aimer passes its ``_MIN_AIM_DISTANCE``, so that both paths read
            one constant.

    Returns:
        L, M and N as new arrays, or None if Numba runs without its JIT
        compiler, if the six coordinate arrays are not C-contiguous 1-D
        float64 NumPy arrays of one length, or if a component of ``axis`` is
        not one number. The caller then runs the NumPy code.
    """
    if numba.config.DISABLE_JIT:
        return None
    arrays = (x0, y0, z0, x1, y1, z1)
    if not all(_is_float64_vector(array) for array in arrays):
        return None
    n = x0.shape[0]
    if any(array.shape[0] != n for array in arrays):
        return None
    if not flip:
        flip_kind, values = _NO_FLIP, (0.0, 0.0, 0.0)
    elif axis is None:
        flip_kind, values = _FLIP_Z, (0.0, 0.0, 0.0)
    else:
        flip_kind, values = _FLIP_AXIS, _axis_values(axis)
        if values is None:
            return None
    directions = (np.empty(n), np.empty(n), np.empty(n))
    _directions((x0, y0, z0), (x1, y1, z1), min_distance, flip_kind, values, directions)
    return directions
