"""NumPy conic intersections, including shared compiled float64 loops."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit
from numba.extending import register_jitable

from optiland.backend._conic import _Candidates, _conic_candidates, _select_distance

if TYPE_CHECKING:
    from collections.abc import Callable

# Register the same pure arithmetic for inlining; no second mathematical solver.
register_jitable(inline="always")(_conic_candidates)
_FLOAT64_EPS = np.finfo(np.float64).eps


@register_jitable(inline="always")
def _scalar_where(condition: Any, left: Any, right: Any) -> Any:
    """Scalar selection used by the shared arithmetic in the compiled loop."""
    return left if condition else right


@register_jitable(inline="always")
def _float64_eps(value: Any) -> float:
    """Precision of the explicitly restricted compiled execution path."""
    return _FLOAT64_EPS


@njit(cache=True, error_model="numpy")
def _numpy_conic_distance(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    L: np.ndarray,
    M: np.ndarray,
    N: np.ndarray,
    radius: float,
    conic: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse the no-aperture float64 calculation without fastmath."""
    distance = np.empty_like(x)
    regular = np.empty(x.shape, dtype=np.bool_)
    for i in range(x.size):
        roots = _conic_candidates(
            x[i],
            y[i],
            z[i],
            L[i],
            M[i],
            N[i],
            radius,
            conic,
            _scalar_where,
            math.sqrt,
            math.copysign,
            _float64_eps,
        )
        distance[i] = (
            (roots.second if roots.pick_second else roots.first)
            if roots.solvable
            else math.nan
        )
        regular[i] = roots.regular
    return distance, regular


@njit(cache=True, error_model="numpy")
def _numpy_conic_candidates(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    L: np.ndarray,
    M: np.ndarray,
    N: np.ndarray,
    radius: float,
    conic: float,
) -> _Candidates:
    """Expose both roots when an arbitrary aperture must choose between them.

    The no-aperture loop only allocates distance and regularity arrays; this
    variant retains the additional candidate arrays needed by aperture code.
    Both loops use the same scalar arithmetic and selection policy.
    """
    first, second = np.empty_like(x), np.empty_like(x)
    valid1 = np.empty(x.shape, dtype=np.bool_)
    valid2 = np.empty(x.shape, dtype=np.bool_)
    pick2 = np.empty(x.shape, dtype=np.bool_)
    solvable = np.empty(x.shape, dtype=np.bool_)
    regular = np.empty(x.shape, dtype=np.bool_)
    for i in range(x.size):
        roots = _conic_candidates(
            x[i],
            y[i],
            z[i],
            L[i],
            M[i],
            N[i],
            radius,
            conic,
            _scalar_where,
            math.sqrt,
            math.copysign,
            _float64_eps,
        )
        first[i], second[i] = roots.first, roots.second
        valid1[i], valid2[i] = roots.first_valid, roots.second_valid
        pick2[i], solvable[i], regular[i] = (
            roots.pick_second,
            roots.solvable,
            roots.regular,
        )
    return _Candidates(first, second, valid1, valid2, pick2, solvable, regular)


def _can_fuse_numpy(values: tuple, radius: Any, conic: Any) -> bool:
    """Restrict compiled execution to matching float64 arrays and scalars."""
    first = values[0]
    return (
        type(first) is np.ndarray
        and first.ndim == 1
        and all(
            type(value) is np.ndarray
            and value.dtype == np.float64
            and value.shape == first.shape
            for value in values
        )
        and np.ndim(radius) == 0
        and np.ndim(conic) == 0
        and np.asarray(radius).dtype == np.float64
        and np.asarray(conic).dtype == np.float64
    )


def _epsilon(value: Any) -> float:
    """Use the arithmetic result's dtype, including mixed-precision inputs."""
    return float(np.finfo(getattr(value, "dtype", float)).eps)


class ConicMixin:
    """Finite-conic execution for the NumPy backend."""

    def conic_intersection(
        self,
        x: Any,
        y: Any,
        z: Any,
        L: Any,
        M: Any,
        N: Any,
        radius: Any,
        conic: Any,
        contains: Callable | None = None,
    ) -> Any:
        """Select a finite-conic intersection using compiled or native arrays.

        See ``AbstractBackend.conic_intersection`` for the numerical contract.
        """
        values = (x, y, z, L, M, N)
        if _can_fuse_numpy(values, radius, conic):
            if contains is None:
                return _numpy_conic_distance(*values, float(radius), float(conic))[0]
            roots = _numpy_conic_candidates(*values, float(radius), float(conic))
        else:
            roots = _conic_candidates(
                *values, radius, conic, np.where, np.sqrt, np.copysign, _epsilon
            )
        return _select_distance(roots, values, contains, np.where)
