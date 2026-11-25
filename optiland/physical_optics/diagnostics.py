"""Non-mutating diagnostics for sampled scalar optical fields."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.physical_optics.field import ScalarField
from optiland.physical_optics.propagation import (
    _direction_cosines,
    _phase_precision,
    _validate_distance,
)

if TYPE_CHECKING:
    from optiland._types import ScalarOrArray


def _validate_reporting(threshold: float, warn: bool) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real scalar.")
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and greater than zero.")
    if not isinstance(warn, bool):
        raise TypeError("warn must be a bool.")
    return threshold


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
    threshold = _validate_reporting(threshold, warn)
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


@dataclass(frozen=True, slots=True)
class WraparoundDiagnostic:
    """Bounds on the power that periodic propagation wraps around the grid.

    Attributes:
        lower_fraction: Fraction of spectral power whose lateral shift is at
            least the grid width along x or y. It leaves the grid from every
            start position, so in the ray model at least this fraction wraps.
        upper_fraction: Fraction of spectral power whose lateral shift can
            carry it across a grid edge from somewhere in the occupied support
            box. In the ray model, at most this fraction wraps.
        support_threshold: Relative intensity that defines the support box.
        threshold: Finite positive reporting threshold.
        exceeds_threshold: Whether ``upper_fraction > threshold``.

    Both fractions are NumPy scalars or differentiable Torch scalars with the
    field's real dtype and device. The record is frozen; its Torch scalars
    remain tensors, not deeply immutable values.
    """

    lower_fraction: ScalarOrArray
    upper_fraction: ScalarOrArray
    support_threshold: float
    threshold: float
    exceeds_threshold: bool


def wraparound_diagnostic(
    field: ScalarField,
    distance: float | ScalarOrArray,
    *,
    support_threshold: float,
    threshold: float,
    warn: bool = False,
) -> WraparoundDiagnostic:
    """Bound the power that angular-spectrum propagation wraps around the grid.

    Call this on the field before propagation, with the propagation distance.
    A plane-wave component with direction cosines ``(ux, uy, uz)`` shifts
    laterally by ``distance * ux / uz`` along x, and likewise along y. On the
    periodic FFT grid, content that crosses an edge re-enters on the opposite
    side. The angular spectrum fixes the shift of each component but not where
    its content starts, so the wrapped power is bounded from both sides:

    - ``lower_fraction`` counts power that shifts by at least the grid width,
      which leaves the grid from any start position. It is the power outside
      the non-aliasing band of Matsushima and Shimobaba (2009), Eq. (17),
      for this grid zero padded to twice its size.
    - ``upper_fraction`` counts power that shifts beyond the empty margin
      between the support box and the grid edge. The support box spans the
      samples whose intensity is at least ``support_threshold`` times the peak.

    Unlike the boundary occupancy of the propagated field, the lower bound
    also detects content that has already wrapped back into the interior. The
    upper bound detects content that starts near an edge and wraps after a
    small shift, which no spectral window alone can detect.

    Both bounds assume a ray model: each spatial-frequency component carries
    its power along a single lateral shift, from anywhere in the support box.
    This misses diffraction, the sinc-like tails of the sampled propagation
    kernel, and content below ``support_threshold``, so the wrapped power can
    fall slightly outside the bounds. Evanescent components do not shift.
    Neither the field nor its sampling is modified.

    Fractions retain Torch gradients with respect to the field, so a design
    that generates it can be optimized against wraparound. Penalize
    ``upper_fraction``; ``lower_fraction`` can reach zero while power near an
    edge still wraps. The bounds are piecewise constant in ``distance``, so a
    distance tensor receives no gradient.
    The support box, validation and the Python threshold flag synchronize
    Torch device execution, even when ``warn=False``.

    Args:
        field: Scalar field before propagation, on the active backend, with
            finite amplitudes.
        distance: Signed propagation distance, in the field's spatial unit.
        support_threshold: Relative intensity in ``(0, 1]`` above which a
            sample belongs to the support box.
        threshold: Finite positive dimensionless fraction threshold for the
            upper bound. Equality does not trigger the flag or warning.
        warn: Emit a ``RuntimeWarning`` when the threshold is exceeded.

    Returns:
        WraparoundDiagnostic: Frozen record containing both backend bounds
            and the threshold decision.

    Raises:
        TypeError: If the field, distance or reporting parameters have invalid
            types.
        ValueError: If the distance, thresholds or field amplitudes are
            invalid.
        RuntimeError: If the field's backend is no longer active.

    References:
        1. K. Matsushima and T. Shimobaba, "Band-limited angular spectrum
           method for numerical simulation of free-space propagation in far
           and near fields," Opt. Express 17, 19662-19673 (2009).
    """
    if not isinstance(field, ScalarField):
        raise TypeError("field must be a ScalarField.")
    field._ensure_active_backend()
    _validate_distance(distance)
    if isinstance(support_threshold, bool) or not isinstance(support_threshold, Real):
        raise TypeError("support_threshold must be a real scalar.")
    support_threshold = float(support_threshold)
    if not 0 < support_threshold <= 1:
        raise ValueError("support_threshold must be in the interval (0, 1].")
    threshold = _validate_reporting(threshold, warn)
    if not bool(be.all(be.isfinite(field.data))):
        raise ValueError("field amplitudes must be finite.")

    amplitude = be.abs(field.data)
    scale = be.max(amplitude)
    if not bool(be.isfinite(scale)):
        raise ValueError("field amplitude magnitudes must be finite.")
    if bool(scale == 0):
        lower_fraction = upper_fraction = be.sum(amplitude**2)
    else:
        # Scaling before the FFT avoids spectral power overflow or underflow.
        spectrum = be.fft.fft2(field.data / scale)
        power = be.real(spectrum * be.conj(spectrum))
        total_power = be.sum(power)

        occupied = be.to_numpy((amplitude / scale) ** 2 >= support_threshold)
        (rows,) = occupied.any(axis=1).nonzero()
        (columns,) = occupied.any(axis=0).nonzero()
        ny, nx = field.shape
        # Empty margins between the support box and each grid edge.
        low_x = int(columns[0]) * field.dx
        high_x = (nx - 1 - int(columns[-1])) * field.dx
        low_y = int(rows[0]) * field.dy
        high_y = (ny - 1 - int(rows[-1])) * field.dy

        if not isinstance(distance, Real):
            distance = _phase_precision(distance, field.data)
        ux, uy, transverse_squared = _direction_cosines(field)
        longitudinal_squared = 1.0 - transverse_squared
        propagating = longitudinal_squared >= 0
        longitudinal = be.sqrt(be.clip(longitudinal_squared, 0.0, be.inf))
        # The lateral shift is shift_x / longitudinal. Comparing products avoids
        # dividing by zero at the propagating cutoff, where the shift diverges.
        shift_x = distance * ux
        shift_y = distance * uy
        leaves_everywhere = propagating & (
            ((be.abs(shift_x) >= nx * field.dx * longitudinal) & (shift_x != 0))
            | ((be.abs(shift_y) >= ny * field.dy * longitudinal) & (shift_y != 0))
        )
        may_leave = propagating & (
            (shift_x > high_x * longitudinal)
            | (shift_x < -low_x * longitudinal)
            | (shift_y > high_y * longitudinal)
            | (shift_y < -low_y * longitudinal)
        )
        # Keep reduction-order roundoff within the physical fraction bounds.
        lower_fraction = be.clip(
            be.sum(be.where(leaves_everywhere, power, 0.0)) / total_power, 0.0, 1.0
        )
        upper_fraction = be.clip(
            be.sum(be.where(may_leave, power, 0.0)) / total_power, 0.0, 1.0
        )

    exceeds_threshold = bool(upper_fraction > threshold)
    if warn and exceeds_threshold:
        warnings.warn(
            f"Between {float(be.to_numpy(lower_fraction)):.6g} and "
            f"{float(be.to_numpy(upper_fraction)):.6g} of the spectral power may "
            "wrap around the periodic FFT grid during propagation; the upper "
            f"bound exceeds threshold {threshold:.6g}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return WraparoundDiagnostic(
        lower_fraction,
        upper_fraction,
        support_threshold,
        threshold,
        exceeds_threshold,
    )
