"""Shared finite-conic arithmetic, independent of array libraries and execution.

Backend implementations supply elementary operations and dtype epsilon. Both
native arrays and compiled scalar loops use this root and selection policy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable


class _Candidates(NamedTuple):
    """Root values and masks shared by scalar and array execution."""

    first: Any
    second: Any
    first_valid: Any
    second_valid: Any
    pick_second: Any
    solvable: Any
    regular: Any


def _conic_candidates(
    x: Any,
    y: Any,
    z: Any,
    L: Any,
    M: Any,
    N: Any,
    radius: Any,
    conic: Any,
    where: Callable,
    sqrt: Callable,
    copysign: Callable,
    epsilon: Callable,
) -> _Candidates:
    """Compute root values, admissibility, and selection masks.

    ``where``, ``sqrt``, ``copysign``, and ``epsilon`` operate on backend arrays,
    or scalars in the compiled loop. Factoring the coefficients
    avoids cancellation of the quadratic term at and near a parabola. The
    constant term is the implicit surface residual at the ray origin.

    A nonzero coefficient must not be discarded solely because it is below
    machine epsilon: ``a``, ``b``, ``c``, and the discriminant have different
    units and scale differently under a geometric rescaling. Zero roots are
    self-crossings. The smaller root is also excluded when both its residual
    and its displacement are within roundoff at the origin.
    """
    k1 = 1 + conic
    kz = k1 * z
    transverse = x * x + y * y
    transverse_direction = L * L + M * M
    N2 = N * N
    a = transverse_direction + k1 * N2
    b = 2 * (L * x + M * y + N * (kz - radius))
    c = transverse + z * (kz - 2 * radius)
    # Sag evaluation and propagation can leave a rounded point just off the
    # surface. Bound that residual in squared-length units, using the terms
    # before cancellation. Unlike a coordinate-based distance floor, this
    # scales with the equation and retains resolvable nearby intersections.
    residual_scale = transverse + abs(z) * (abs(kz) + 2 * abs(radius))
    roundoff = 4 * epsilon(c)
    resolved_c = abs(c) > roundoff * residual_scale
    d = b * b - 4 * a * c
    d_ok = d >= 0

    # Preserve every positive radicand. Replacing *inactive* inputs before
    # sqrt keeps its backward pass finite for misses and exact tangencies.
    positive_d = d > 0
    sqrt_d = where(positive_d, sqrt(where(positive_d, d, 1.0)), 0.0)
    q = -0.5 * (b + copysign(sqrt_d, b))
    # A magnitude comparison excludes infinities and NaNs in one predicate.
    # This also avoids Torch isfinite's separate NaN and infinity masks.
    a_ok = (a != 0) & (abs(a) < math.inf)
    q_ok = (q != 0) & (abs(q) < math.inf)
    t1 = q / where(a_ok, a, 1.0)
    t2 = c / where(q_ok, q, 1.0)
    # A small implicit residual alone does not imply a self-hit: a nearly
    # tangent ray can travel a resolved distance from such an origin. Require
    # the smaller root's displacement to be below position roundoff as well.
    resolved_step = t2 * t2 * (transverse_direction + N2) > (
        roundoff * roundoff * (transverse + z * z)
    )
    solvable1 = d_ok & a_ok & (abs(t1) < math.inf)
    solvable2 = d_ok & q_ok & (abs(t2) < math.inf)
    z1 = z + t1 * N
    z2 = z + t2 * N
    valid1 = solvable1 & (t1 > 0) & (1 - k1 * z1 / radius >= 0)
    # The stable q formula makes c/q the root with smaller magnitude. Only
    # that candidate is a possible rounded self-hit; retain the other root
    # and its full derivatives through the original, unmodified coefficients.
    valid2 = (
        solvable2
        & (resolved_c | resolved_step)
        & (t2 > 0)
        & (1 - k1 * z2 / radius >= 0)
    )

    # c/q is the smaller-magnitude root, so whenever both roots are forward,
    # it is the nearer one. Select its index directly, with vertex fallback.
    vertex2 = where(solvable2, abs(z2), math.inf) < where(solvable1, abs(z1), math.inf)
    pick2 = valid2 | where(valid1, False, vertex2)
    solvable = solvable1 | solvable2
    regular = solvable & (positive_d | (a == 0))
    return _Candidates(t1, t2, valid1, valid2, pick2, solvable, regular)


def _select_distance(
    roots: _Candidates, values: tuple, contains: Callable | None, where: Callable
) -> Any:
    """Apply aperture preference without changing geometric admissibility."""
    pick2 = roots.pick_second
    if contains is not None:
        x, y, _, L, M, _ = values
        pref1 = roots.first_valid & contains(x + roots.first * L, y + roots.first * M)
        pref2 = roots.second_valid & contains(
            x + roots.second * L, y + roots.second * M
        )
        pick2 = pref2 | where(pref1, False, pick2)
    return where(roots.solvable, where(pick2, roots.second, roots.first), math.nan)
