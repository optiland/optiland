"""Non-mutating diagnostics for sampled scalar optical fields."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.physical_optics.field import ScalarField

if TYPE_CHECKING:
    from optiland._types import ScalarOrArray


@dataclass(frozen=True, slots=True)
class BoundaryDiagnostic:
    """Intensity fraction in the boundary band of a rectangular field.

    Attributes:
        boundary_fraction: Dimensionless boundary intensity divided by total
            intensity. Zero for a zero field. A NumPy scalar or a differentiable
            Torch scalar with the field's real dtype and device.
        edge_width: Width of the band at each edge, in samples.
        threshold: Finite positive reporting threshold.
        exceeds_threshold: Whether ``boundary_fraction > threshold``.

    The record is frozen; its Torch scalar remains a tensor, not a deeply
    immutable value. A low fraction does not establish alias-free propagation.
    """

    boundary_fraction: ScalarOrArray
    edge_width: int
    threshold: float
    exceeds_threshold: bool


def boundary_diagnostic(
    field: ScalarField,
    *,
    edge_width: int,
    threshold: float,
    warn: bool = False,
) -> BoundaryDiagnostic:
    """Report intensity reaching the periodic FFT grid boundary.

    Call this on a propagated field to check its current boundary occupancy.
    The band is the union of the outer ``edge_width`` rows and columns; corners
    are counted once. Uniform pixel area cancels from the power ratio, even
    when ``dx != dy``. Amplitudes are scaled before squaring to avoid intensity
    overflow or underflow. Neither the field nor its sampling is modified.

    This is a wraparound-risk indicator, not a proof of alias-free propagation:
    it cannot detect content that has already wrapped back into the interior
    or assess spatial-frequency sampling. It never pads or filters the field.

    The metric retains Torch gradients for nonzero fields. Validation and the
    Python threshold flag synchronize Torch device execution, even when
    ``warn=False``; warning formatting also reads the scalar on the host.
    The zero-field convention has a zero gradient, not a directional limit.

    Args:
        field: Scalar field on the active backend, with finite amplitudes.
        edge_width: Positive integer band width in samples, no greater than
            half the smaller grid dimension (rounded down).
        threshold: Finite positive dimensionless fraction threshold. Equality
            does not trigger the flag or warning.
        warn: Emit a ``RuntimeWarning`` when the threshold is exceeded.

    Returns:
        BoundaryDiagnostic: Frozen record containing the backend metric and
            the threshold decision.

    Raises:
        TypeError: If the field or reporting parameters have invalid types.
        ValueError: If the width, threshold, or field amplitudes are invalid.
        RuntimeError: If the field's backend is no longer active.
    """
    if not isinstance(field, ScalarField):
        raise TypeError("field must be a ScalarField.")
    field._ensure_active_backend()
    if isinstance(edge_width, bool) or not isinstance(edge_width, Integral):
        raise TypeError("edge_width must be an integer number of samples.")
    edge_width = int(edge_width)
    if not 1 <= edge_width <= min(field.shape) // 2:
        raise ValueError("edge_width must be between 1 and half the smaller dimension.")
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real scalar.")
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and greater than zero.")
    if not isinstance(warn, bool):
        raise TypeError("warn must be a bool.")
    if not bool(be.all(be.isfinite(field.data))):
        raise ValueError("field amplitudes must be finite.")

    amplitude = be.abs(field.data)
    scale = be.max(amplitude)
    if not bool(be.isfinite(scale)):
        raise ValueError("field amplitude magnitudes must be finite.")
    if bool(scale == 0):
        fraction = be.sum(amplitude**2)
    else:
        intensity = (amplitude / scale) ** 2
        width = edge_width
        total_intensity = be.sum(intensity)
        if 2 * width == min(field.shape):
            # The band covers the grid; avoid reduction-order roundoff at 1.
            boundary_intensity = total_intensity
        else:
            boundary_intensity = (
                be.sum(intensity[:width, :])
                + be.sum(intensity[-width:, :])
                + be.sum(intensity[width:-width, :width])
                + be.sum(intensity[width:-width, -width:])
            )
        # Keep reduction-order roundoff within the physical fraction bounds.
        fraction = be.clip(boundary_intensity / total_intensity, 0.0, 1.0)

    exceeds_threshold = bool(fraction > threshold)
    if warn and exceeds_threshold:
        warnings.warn(
            f"Boundary intensity fraction {float(be.to_numpy(fraction)):.6g} exceeds "
            f"threshold {threshold:.6g} in the outer {edge_width} samples; "
            "periodic FFT propagation may suffer wraparound.",
            RuntimeWarning,
            stacklevel=2,
        )
    return BoundaryDiagnostic(fraction, edge_width, threshold, exceeds_threshold)
