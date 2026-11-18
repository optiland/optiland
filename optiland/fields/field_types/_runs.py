"""Runs of rays with the same field point.

``RealRayTracer.trace`` repeats each field point for each pupil point, so a
large bundle has few different field points, in long runs. Arithmetic that
depends only on the field point (the validity checks of the field angles,
their tangents, the launch distances) can then run once for each run, and
``FieldRuns.expand`` repeats the results to one value per ray.

NumPy computes each element of a contiguous float64 array with the same
loop, whatever its length. The values per run and per ray are new
contiguous arrays, so each ray gets the same bits. The per-ray expressions
in ``AngleField.get_ray_origins`` are the reference. The runs compare the
bits of the field coordinates, not their values: ``-0.0 == 0.0``, but the
tangent of ``-0.0`` is ``-0.0``.

Only the NumPy backend uses runs. With Torch, a gradient through a repeated
value would add its parts in another order.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland._types import BEArray, ScalarOrArray

# The fewest rays for which runs are found, and the fewest rays in a run on
# average. Below these, the arithmetic on one value per ray costs less than
# finding the runs.
_MIN_RAYS = 4096
_MIN_RUN = 64


class FieldRuns:
    """The field coordinates of a bundle as runs of equal field points.

    Attributes:
        hx: The normalized x field coordinate of each run.
        hy: The normalized y field coordinate of each run.
    """

    __slots__ = ("hx", "hy", "_lengths", "_ray_count")

    def __init__(
        self,
        hx: np.ndarray,
        hy: np.ndarray,
        lengths: np.ndarray | None,
        ray_count: int,
    ) -> None:
        """Store the runs.

        Args:
            hx: The normalized x field coordinate of each run.
            hy: The normalized y field coordinate of each run.
            lengths: The number of rays in each run, or None if all rays
                are in one run.
            ray_count: The number of rays in the bundle.
        """
        self.hx = hx
        self.hy = hy
        self._lengths = lengths
        self._ray_count = ray_count

    def expand(self, values: np.ndarray) -> np.ndarray:
        """Repeat one value for each run to one value for each ray.

        Args:
            values: One value for each run.

        Returns:
            One value for each ray. With one run, ``values`` is returned
            unchanged; it broadcasts against arrays with one value per ray.
        """
        if self._lengths is None:
            return values
        return np.repeat(values, self._lengths)

    def per_ray(self, values: np.ndarray) -> np.ndarray:
        """Repeat one value for each run to one value for each ray, always.

        Args:
            values: One value for each run.

        Returns:
            A new array with one value for each ray, also with one run.
        """
        if self._lengths is None:
            return np.repeat(values, self._ray_count)
        return np.repeat(values, self._lengths)


def _identity(values: BEArray) -> BEArray:
    """Return ``values``: the expand function of a bundle without runs."""
    return values


def field_runs(
    Hx: ScalarOrArray, Hy: ScalarOrArray, Px: ScalarOrArray, Py: ScalarOrArray
) -> FieldRuns | None:
    """Return the runs of the field coordinates, or None.

    Args:
        Hx: The normalized x field coordinate of each ray.
        Hy: The normalized y field coordinate of each ray.
        Px: The normalized x pupil coordinate of each ray.
        Py: The normalized y pupil coordinate of each ray.

    Returns:
        FieldRuns, or None if the backend is not NumPy with float64, if the
        four arrays are not 1-D float64 NumPy arrays of one length, if the
        bundle has fewer than ``_MIN_RAYS`` rays, or if the runs have fewer
        than ``_MIN_RUN`` rays on average.
    """
    if be.get_backend() != "numpy" or be.get_precision() != 64:
        return None
    arrays = (Hx, Hy, Px, Py)
    if not all(type(a) is np.ndarray and a.ndim == 1 for a in arrays):
        return None
    ray_count = Hx.shape[0]
    if ray_count < _MIN_RAYS or any(a.shape[0] != ray_count for a in arrays):
        return None
    if any(a.dtype != np.float64 for a in arrays):
        return None
    # Compare bits, not values (see the module docstring).
    bits_x = Hx.view(np.int64)
    bits_y = Hy.view(np.int64)
    if (bits_x == bits_x[0]).all() and (bits_y == bits_y[0]).all():
        return FieldRuns(Hx[:1], Hy[:1], None, ray_count)
    changed = (bits_x[1:] != bits_x[:-1]) | (bits_y[1:] != bits_y[:-1])
    starts = np.concatenate(([0], np.flatnonzero(changed) + 1))
    if starts.size * _MIN_RUN > ray_count:
        return None
    lengths = np.diff(np.append(starts, ray_count))
    return FieldRuns(Hx[starts], Hy[starts], lengths, ray_count)


def expander(runs: FieldRuns | None) -> Callable[[BEArray], BEArray]:
    """Return ``runs.expand``, or a function that returns its argument.

    Args:
        runs: The runs of the bundle, or None.

    Returns:
        The function that repeats one value for each run to one value for
        each ray.
    """
    return _identity if runs is None else runs.expand
