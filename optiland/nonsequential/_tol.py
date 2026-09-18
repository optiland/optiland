"""Dtype-aware tolerances for the non-sequential ray tracer.

Every numerical tolerance in ``optiland.nonsequential`` used to be a bare
literal (``1e-9``, ``1e-12``, ``1e-30``, ...) that did not consult the active
dtype or the coordinate magnitude it was being compared against. At float32,
several of those literals sit below the resolvable step of the numbers they
guard -- a self-intersection epsilon of ``1e-9`` mm is four orders of
magnitude below the float32 step at 50 mm -- so a ray immediately re-accepts
the surface it just left. At float64 the same absolute epsilon fails once the
scene is large enough (roughly 5e4 mm) for the same reason. See
the module's own function docstrings for the rule each one applies.

The rule this module implements, in one sentence: **no tolerance is an
absolute length, angle, or denominator constant; every one is a small
multiple of the working dtype's own resolution at the magnitude in play.**

Three primitives:

- :func:`ulp` -- the spacing between adjacent representable numbers at a
  given magnitude, in the input's own dtype and backend (NumPy or Torch).
  This is the exact IEEE-754 "unit in the last place", via ``np.spacing`` or
  ``torch.nextafter``; it is *not* the cruder "dtype epsilon times the next
  power of two" shortcut that a pure-Python reference script uses when it
  cannot ask NumPy or Torch directly. The two agree to within a factor of 2
  (rounding to the exact binade vs. the next one up) -- either is a safe
  self-intersection guard at the ``k~25`` scale this module defaults to,
  because both sit four-plus orders of magnitude above the self-intersection
  artifacts they exist to reject (about 1e-6 mm, measured on this engine's
  own conic and frustum arithmetic).

- :func:`accept_t_min` -- the minimum accepted ray parameter ``t``, expressed
  as ``k`` ulps of a representative coordinate magnitude (with a 1.0 mm floor
  so a ray at the local origin still gets a non-zero threshold). Replaces
  every bare ``t > 1e-9`` / ``eps = 1e-9`` self-intersection guard.

- :func:`tiny_for` -- a division-guard epsilon that is safe for the backward
  pass: the square root of the dtype's smallest normal number, so squaring it
  (which reverse-mode autodiff does for a bare reciprocal's local derivative)
  does not underflow to zero and turn a ``where``'s discarded zero-cotangent
  branch into ``0 * inf = NaN``. Replaces every bare ``+ 1e-30`` additive
  division guard, and every bare ``1e-30``/``1e-14`` comparison threshold
  used to decide whether a denominator is degenerate.

Two exceptions, both intentional and both flagged where used: a physical
fraction (a Russian-roulette flux floor, a scatter-branch probability clamp)
is dtype-independent by definition and stays an absolute constant; and a
few construction-time geometry checks (e.g. a frustum's axial height) are
evaluated once, off the hot ray loop, in whatever precision their inputs
happen to carry, so they use :func:`ulp` directly with a Python-float floor
rather than backend dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from optiland.nonsequential._utils import is_tensor

if TYPE_CHECKING:
    from optiland._types import ScalarOrArrayT

# Default k for accept_t_min. Measured: the largest self-intersection
# residual after a real refraction on the quickstart singlet is about 0.7 ulp
# of the ray's coordinate, so 25 clears it with a wide margin while staying
# far below any genuine second hit.
DEFAULT_ACCEPT_K = 25

# Coordinate-magnitude floor [mm] for accept_t_min: a ray whose local origin
# is at (or very near) the coordinate origin still needs a non-zero minimum
# accepted t, so the ulp is never taken at magnitude 0.
_MAGNITUDE_FLOOR = 1.0

# ulp(1) at float64 -- the reference radicand_floor() scales against.
_ULP1_F64 = 2.220446049250313e-16


def ulp(x: ScalarOrArrayT) -> ScalarOrArrayT:
    """Spacing between adjacent representable numbers at magnitude ``|x|``.

    Dispatches on backend so the result is in ``x``'s own dtype: a NumPy
    array or scalar uses :func:`numpy.spacing`; a Torch tensor uses
    :func:`torch.nextafter` toward positive infinity, which is the exact
    successor distance in that tensor's dtype (and its device, and detached
    from any autograd graph on ``x`` -- this is a constant, not a quantity to
    differentiate through).

    Args:
        x: Coordinate magnitude(s). A plain Python float is treated as a
            NumPy float64 scalar (Python has no other float width).

    Returns:
        The ulp at ``|x|``, same backend and dtype as ``x`` (float64 for a
        plain Python float or int).
    """
    if is_tensor(x):
        import torch  # noqa: PLC0415

        ax = torch.abs(x).detach()
        return torch.nextafter(ax, torch.full_like(ax, float("inf"))) - ax
    return np.spacing(np.abs(np.asarray(x)))


def accept_t_min(
    origin_magnitude: ScalarOrArrayT, k: int = DEFAULT_ACCEPT_K
) -> ScalarOrArrayT:
    """Minimum accepted ray parameter ``t``, ``k`` ulps of a coordinate scale.

    A ray is accepted as hitting a
    surface only when its solved parameter exceeds this value; below it, the
    hit cannot be distinguished, in the working dtype, from the surface the
    ray has just left.

    Args:
        origin_magnitude: A representative coordinate magnitude (e.g. the
            infinity norm of the local-frame ray origins the intersection is
            about to be solved for) in the working backend/dtype. A plain
            Python float is accepted too (treated as float64). Floored at
            1.0 (in whatever units ``origin_magnitude`` carries -- mm
            throughout this package) so a ray already at the local origin
            still gets a non-zero threshold.
        k: Multiple of the local ulp. Default 25.

    Returns:
        The threshold, same backend/dtype as ``origin_magnitude``.
    """
    if is_tensor(origin_magnitude):
        import torch  # noqa: PLC0415

        mag = torch.clamp(torch.abs(origin_magnitude).detach(), min=_MAGNITUDE_FLOOR)
    else:
        mag = np.maximum(np.abs(np.asarray(origin_magnitude)), _MAGNITUDE_FLOOR)
    return k * ulp(mag)


def tiny_for(dtype_or_array: Any) -> float:
    """Division-guard epsilon safe for the backward pass, for ``dtype``.

    Returns ``sqrt(smallest_normal)`` of the given dtype. See
    A guarded reciprocal
    ``1 / (2*a + eps)`` forms ``eps**2`` in its backward pass (the local
    derivative of ``1/x`` is ``-1/x**2``); if ``eps`` is much smaller than
    this bound, ``eps**2`` underflows to exactly zero, the reciprocal's local
    derivative becomes signed infinity, and a ``where`` multiplying that by a
    discarded zero cotangent produces ``0 * inf = NaN`` that then propagates
    through the whole reverse sweep -- even though the forward pass was
    masked correctly. ``eps >= sqrt(x_min_normal)`` keeps ``eps**2`` at or
    above the smallest normal, so the local derivative stays finite and the
    zero cotangent actually zeroes it out.

    Args:
        dtype_or_array: A NumPy array/dtype, a Torch tensor/dtype, or a
            string/``numpy.dtype`` NumPy accepts (``"float32"``, ...). An
            array or tensor has its own ``.dtype`` used automatically.

    Returns:
        A plain Python float: ``1.0842e-19`` for float32, ``1.4917e-154``
        for float64.
    """
    if is_tensor(dtype_or_array):
        import torch  # noqa: PLC0415

        return float(torch.finfo(dtype_or_array.dtype).tiny) ** 0.5
    try:
        import torch  # noqa: PLC0415

        if isinstance(dtype_or_array, torch.dtype):
            return float(torch.finfo(dtype_or_array).tiny) ** 0.5
    except ImportError:
        pass
    # A NumPy array uses its own dtype; anything else (a numpy dtype object,
    # a scalar type like np.float32, or a string like "float32") is handed
    # to np.dtype() directly -- np.float32.dtype (the class, not an instance)
    # is an unbound attribute descriptor, not a dtype, so arrays must be
    # special-cased rather than blindly read through getattr(..., "dtype").
    dt = (
        dtype_or_array.dtype
        if isinstance(dtype_or_array, np.ndarray)
        else dtype_or_array
    )
    return float(np.finfo(np.dtype(dt)).tiny) ** 0.5


def radicand_floor(
    ones_like_x: ScalarOrArrayT, floor_at_f64: float = 1e-12
) -> ScalarOrArrayT:
    """Floor for a value about to be square-rooted, calibrated at float64.

    A secondary helper alongside the three primitives above, for the one
    tolerance shape that is not simply "k ulps of a coordinate": a handful of
    radicand clamps (``sqrt(max(w, floor))``) whose floor was originally a
    bare ``1e-12`` chosen so a specific forward quantity (a lens-edge sag)
    moves by no more than ``sqrt(1e-12) = 1`` micron. That budget is a design
    choice, not a resolution limit, so it should not simply shrink to the
    working dtype's own ulp(1) (replacing it with ``k * ulp(1)`` for a small
    k changes the regularization strength and, measured here, the traced
    result). Instead this scales the
    *same* calibrated budget by the working dtype's own coarseness relative
    to float64, so it is unchanged at float64 (the ratio is exactly 1.0) and
    grows at lower precision, where the un-scaled 1e-12 floor is already
    below the dtype's own resolution at magnitude 1 (float32 epsilon is
    1.19e-7) and so acts as if it were zero -- the sqrt(0) singularity it
    exists to avoid still reaches the backward pass.

    Args:
        ones_like_x: An array of ones in the working backend/dtype (e.g.
            ``be.ones_like(w)``) -- only its dtype matters.
        floor_at_f64: The calibrated float64 floor this reduces to.

    Returns:
        ``floor_at_f64`` at float64; scaled up at lower precision.
    """
    return floor_at_f64 * (ulp(ones_like_x) / _ULP1_F64)
