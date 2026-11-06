"""Dtype-aware self-intersection threshold for the non-sequential ray tracer.

A ray leaving a surface is only accepted as hitting a surface again when the
solved ray parameter ``t`` exceeds a minimum. That minimum used to be the
absolute length ``1e-9`` mm. At float32 this is far below the resolvable step
of the coordinates it guards (the float32 spacing at 50 mm is 3.8e-6 mm), so a
ray re-accepts the surface it has just left, bounces against it until the depth
cap kills it, and its flux never reaches a detector. At float64 the same
absolute value fails once coordinates reach roughly 5e4 mm, for the same reason.

Two primitives:

- :func:`ulp` -- the exact IEEE-754 spacing between adjacent representable
  numbers at a given magnitude, in the input's own dtype and backend
  (``numpy.spacing`` or ``torch.nextafter``).
- :func:`accept_t_min` -- the minimum accepted ray parameter, ``k`` ulps of a
  coordinate magnitude, with a 1 mm floor. Elementwise, so a per-ray magnitude
  gives a per-ray threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential._utils import is_tensor

if TYPE_CHECKING:
    from optiland._types import ScalarOrArrayT

# Default k for accept_t_min. At float32 (torch CPU, 200k rays) the detected
# flux of the quick-start singlet stops changing from k = 4 on, and that of a
# lens-and-mirror scene from k = 8 on; 25 leaves a margin above both while
# staying far below any genuine second hit (25 ulps at 50 mm is 1e-4 mm in
# float32).
DEFAULT_ACCEPT_K = 25

# Coordinate-magnitude floor [mm] for accept_t_min: a ray at (or very near) the
# coordinate origin still gets a non-zero minimum accepted t.
_MAGNITUDE_FLOOR = 1.0


def ulp(x: ScalarOrArrayT) -> ScalarOrArrayT:
    """Spacing between adjacent representable numbers at magnitude ``|x|``.

    A NumPy array or scalar uses :func:`numpy.spacing`; a Torch tensor uses
    :func:`torch.nextafter` toward positive infinity, on the tensor's own
    dtype and device, detached from any autograd graph (this is a constant,
    not a quantity to differentiate through).

    Args:
        x: Coordinate magnitude(s). A plain Python float is treated as a NumPy
            float64 scalar.

    Returns:
        The ulp at ``|x|``, same backend, dtype and shape as ``x``.
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

    A ray is accepted as hitting a surface only when its solved parameter
    exceeds this value; below it, the hit cannot be distinguished, in the
    working dtype, from the surface the ray has just left.

    Args:
        origin_magnitude: Coordinate magnitude(s) in the working backend and
            dtype, for example each ray's largest absolute global coordinate,
            shape (N,), which gives one threshold per ray. A plain Python float
            is accepted (treated as float64). Floored at 1.0 mm so a ray at
            the origin still gets a non-zero threshold.
        k: Multiple of the ulp. Default 25.

    Returns:
        The threshold, same backend, dtype and shape as ``origin_magnitude``.
    """
    if is_tensor(origin_magnitude):
        import torch  # noqa: PLC0415

        mag = torch.clamp(torch.abs(origin_magnitude).detach(), min=_MAGNITUDE_FLOOR)
    else:
        mag = np.maximum(np.abs(np.asarray(origin_magnitude)), _MAGNITUDE_FLOOR)
    return k * ulp(mag)
