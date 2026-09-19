"""Shared forward kernels for weighted wavefront evaluation.

Scalar reductions consume nonempty one-dimensional arrays. Aligned-term addition
accepts scalars and same-shaped arrays, with at least one array; broadcasting is
limited to those scalar terms. Calculations retain the input floating precision.
Overflow recovery does not promise correctly rounded arbitrary expressions.

This module has no dependency on the public evaluator or custom autograd wrappers.
The eager Torch wrappers own the analytical derivatives of scalar reductions;
the aligned arithmetic also works with ordinary backend differentiation.
"""

# ruff: noqa: I002

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

import numpy as np

import optiland.backend as be
from optiland._types import BEArrayT, ScalarOrArrayT


def _stop_gradient(value: ScalarOrArrayT) -> ScalarOrArrayT:
    """Detach a numerical scale or diagnostic probe, leaving NumPy unchanged."""
    return value.detach() if be.is_torch_tensor(value) else value


def _normalized_sqrt_weights(weights: BEArrayT) -> BEArrayT:
    """Normalize positive square-root weights without squaring their range."""
    roots = be.sqrt(weights)
    scaled = roots / be.max(roots)
    return scaled / _stable_norm(scaled)


def _stable_norm(values: BEArrayT) -> BEArrayT:
    """Scale before squaring, with a constant scale for differentiation.

    The squared terms are nonnegative and bounded by one. They require no
    cancellation recovery. Detaching scale avoids overflowing, cancelling
    derivative terms when this kernel is used in a differentiable expression.
    """
    scale = _stop_gradient(be.max(be.abs(values)))
    if scale == 0.0:
        return be.sum(values * 0.0)
    normalized = values / scale
    return scale * be.sqrt(be.sum(normalized * normalized))


def _weighted_rms_components(
    values: BEArrayT, sqrt_alpha: BEArrayT, scale: BEArrayT | float
) -> tuple[BEArrayT, BEArrayT, BEArrayT]:
    """Return scaled weighted values and norms for a strictly positive scale."""
    weighted = sqrt_alpha * (values / scale)
    return weighted, _stable_norm(weighted), _stable_norm(sqrt_alpha)


def _weighted_rms_forward(values: BEArrayT, sqrt_alpha: BEArrayT) -> BEArrayT:
    """Compute the forward population RMS without preparing derivative storage."""
    scale = be.max(be.abs(values))
    if scale == 0.0:
        return be.sum(values * 0.0)
    _, weighted_norm, weight_norm = _weighted_rms_components(values, sqrt_alpha, scale)
    return scale * (weighted_norm / weight_norm)


def _weighted_scale(weighted_centered: BEArrayT, coordinate_name: str) -> BEArrayT:
    """Return a finite nonzero spread or reject a degenerate coordinate axis."""
    if be.max(be.abs(weighted_centered)) == 0.0:
        raise ValueError(
            f"The {coordinate_name} coordinate has zero positive-weight spread."
        )
    scale = _stable_norm(weighted_centered)
    if not be.all(be.isfinite(scale)) or scale == 0.0:
        raise ValueError(
            f"The {coordinate_name} coordinate has no finite positive-weight spread."
        )
    return scale


def _two_diff(values: BEArrayT, anchor: BEArrayT | float) -> tuple[BEArrayT, BEArrayT]:
    """Return rounded elementwise differences and their subtraction errors."""
    with be.errstate(over="ignore", invalid="ignore"):
        difference = values - anchor
        virtual_anchor = values - difference
        virtual_values = difference + virtual_anchor
        anchor_error = virtual_anchor - anchor
        value_error = values - virtual_values
        error = value_error + anchor_error
    return difference, error


def _sum_samples_forward(values: BEArrayT) -> BEArrayT:
    """Reduce a sample vector, retaining low bits of cancelling finite operands.

    Same-sign finite sums use the backend reduction. Mixed signs use an expansion
    even if every operand is large. Nonfinite operands retain the ordinary result;
    callers decide whether it is an error or a reason to rescale the objective.
    This is a forward kernel, not a custom differentiation policy.
    """
    with be.errstate(over="ignore", invalid="ignore"):
        direct = be.sum(values)
    if not be.all(be.isfinite(values)):
        return direct
    mixed_sign = be.any(values < 0.0) and be.any(values > 0.0)
    if be.all(be.isfinite(direct)) and not mixed_sign:
        return direct
    result = _reduce_expansion(values)
    if not be.all(be.isfinite(result)):
        scale = be.max(be.abs(values))
        with be.errstate(over="ignore", invalid="ignore"):
            result = scale * _reduce_expansion(values / scale)
    return result


def _reduce_expansion(values: BEArrayT) -> BEArrayT:
    """Reduce a finite vector while retaining every nonzero addition error.

    A tree pass leaves a rounded total plus error terms. Repeat on this smaller
    expansion, not on ordinary sums of cancelling blocks. Sorting opposite
    extremes is necessary only when intermediate overflow is possible. A scalar
    partial expansion terminates a pass that cannot reduce the term count.
    """
    partials = values
    if be.is_torch_tensor(values):
        import torch

        maximum = torch.finfo(values.dtype).max
    else:
        maximum = float(np.finfo(values.dtype).max)
    needs_ordering = be.max(be.abs(values)) > maximum / (2 * len(values))
    while len(partials) > 2:
        previous_count = len(partials)
        reduced = partials
        errors = []
        while len(reduced) > 1:
            ordered = be.sort(reduced) if needs_ordering else reduced
            pair_count = len(ordered) // 2
            high, low = _two_diff(ordered[:pair_count], -be.flip(ordered[-pair_count:]))
            if not be.all(be.isfinite(high)):
                return be.sum(high)
            errors.append(low[low != 0.0])
            if len(ordered) % 2:
                high = be.concatenate([high, ordered[pair_count : pair_count + 1]])
            reduced = high
        partials = be.concatenate([reduced, *errors])
        if len(partials) >= previous_count:
            return _reduce_partials(partials)
    return _reduce_partials(partials)


def _reduce_partials(values: BEArrayT) -> BEArrayT:
    """Reduce a vector using nonoverlapping scalar partials in the input dtype."""
    partials = []
    with be.errstate(over="ignore", invalid="ignore"):
        for value in values:
            partial_count = 0
            for partial in partials:
                if be.abs(value) < be.abs(partial):
                    value, partial = partial, value
                high = value + partial
                low = partial - (high - value)
                if low != 0.0:
                    partials[partial_count] = low
                    partial_count += 1
                value = high
            partials[partial_count:] = [value]
        total = values[0] * 0.0
        for partial in partials:
            total = total + partial
    return total


def _add_terms(values: Sequence[BEArrayT | float]) -> BEArrayT:
    """Add aligned terms with compensation; rescale only finite overflow cases.

    This is elementwise addition, not a sample reduction. Nonfinite inputs remain
    nonfinite. Direct compensation retains small corrections at large offsets.
    """
    total = values[0]
    all_finite = be.isfinite(total)
    with be.errstate(over="ignore", invalid="ignore"):
        compensation = total * 0.0
        for value in values[1:]:
            all_finite = be.logical_and(all_finite, be.isfinite(value))
            updated = total + value
            correction = be.where(
                be.abs(total) >= be.abs(value),
                (total - updated) + value,
                (value - updated) + total,
            )
            compensation = compensation + correction
            total = updated
        result = total + compensation
    needs_fallback = be.logical_and(all_finite, be.logical_not(be.isfinite(result)))
    if be.any(needs_fallback):
        result = be.where(needs_fallback, _add_scaled_terms(values), result)
    return result


def _add_scaled_terms(values: Sequence[BEArrayT | float]) -> BEArrayT:
    """Rescale aligned terms after direct addition overflowed."""
    scale = be.abs(values[0])
    for value in values[1:]:
        scale = be.maximum(scale, be.abs(value))
    safe_scale = be.where(scale == 0.0, scale * 0.0 + 1.0, scale)
    total = values[0] / safe_scale
    for value in values[1:]:
        total = total + value / safe_scale
    with be.errstate(over="ignore", invalid="ignore"):
        return safe_scale * total


def _centered_product(
    coefficient: BEArrayT, coordinates: BEArrayT, reference: BEArrayT
) -> BEArrayT:
    """Evaluate c*(x-reference), retaining dependence on c even when c is zero.

    Finite coordinates whose subtraction overflows are differenced at half scale
    before multiplication. Invalid coordinates remain visibly nonfinite.
    """
    finite = be.isfinite(coordinates)
    product = be.copy(coordinates)
    with be.errstate(over="ignore", invalid="ignore"):
        difference = coordinates - reference
    direct = be.logical_and(finite, be.isfinite(difference))
    direct_coefficient = coefficient[direct] if coefficient.ndim else coefficient
    with be.errstate(over="ignore", invalid="ignore"):
        product[direct] = direct_coefficient * difference[direct]
    fallback = be.logical_and(finite, be.logical_not(be.isfinite(difference)))
    if be.any(fallback):
        half_difference = coordinates[fallback] * 0.5 - reference * 0.5
        fallback_coefficient = (
            coefficient[fallback] if coefficient.ndim else coefficient
        )
        with be.errstate(over="ignore", invalid="ignore"):
            product[fallback] = (fallback_coefficient * half_difference) * 2.0
    return product


def _combined_centered_product(
    coefficient_x: BEArrayT,
    x: BEArrayT,
    x_ref: BEArrayT,
    coefficient_y: BEArrayT,
    y: BEArrayT,
    y_ref: BEArrayT,
) -> BEArrayT:
    """Combine two aligned centered products before restoring common scales."""
    finite = be.logical_and(be.isfinite(x), be.isfinite(y))
    result = be.copy(x)
    invalid = be.logical_not(finite)
    if be.any(invalid):
        with be.errstate(over="ignore", invalid="ignore"):
            result[invalid] = x[invalid] + y[invalid]
    if not be.any(finite):
        return result
    x_finite, y_finite = x[finite], y[finite]
    coordinate_scale = be.maximum(be.abs(x_finite), be.abs(y_finite))
    coordinate_scale = be.maximum(coordinate_scale, be.abs(x_ref))
    coordinate_scale = be.maximum(coordinate_scale, be.abs(y_ref))
    safe_coordinate_scale = be.where(
        coordinate_scale == 0.0, coordinate_scale * 0.0 + 1.0, coordinate_scale
    )
    x_normalized = x_finite / safe_coordinate_scale - x_ref / safe_coordinate_scale
    y_normalized = y_finite / safe_coordinate_scale - y_ref / safe_coordinate_scale
    coefficient_scale = be.maximum(be.abs(coefficient_x), be.abs(coefficient_y))
    safe_coefficient_scale = be.where(
        coefficient_scale == 0.0, coefficient_scale * 0.0 + 1.0, coefficient_scale
    )
    combined = _add_terms(
        [
            (coefficient_x / safe_coefficient_scale) * x_normalized,
            (coefficient_y / safe_coefficient_scale) * y_normalized,
        ]
    )
    smaller_scale = be.minimum(coordinate_scale, coefficient_scale)
    larger_scale = be.maximum(coordinate_scale, coefficient_scale)
    with be.errstate(over="ignore", invalid="ignore"):
        result[finite] = (combined * smaller_scale) * larger_scale
    return result


@dataclass(frozen=True)
class _AffineFit(Generic[BEArrayT]):
    """Centered fit state; keep the OPD anchor separate from its correction."""

    opd_anchor: BEArrayT
    intercept_correction: BEArrayT
    slope_x: BEArrayT
    slope_y: BEArrayT
    x_ref: BEArrayT
    y_ref: BEArrayT

    def detached(self) -> "_AffineFit[BEArrayT]":
        """Return graph-independent fit state for diagnostic representability probes."""
        return _AffineFit(
            _stop_gradient(self.opd_anchor),
            _stop_gradient(self.intercept_correction),
            _stop_gradient(self.slope_x),
            _stop_gradient(self.slope_y),
            _stop_gradient(self.x_ref),
            _stop_gradient(self.y_ref),
        )

    def residual(self, opd: BEArrayT, x: BEArrayT, y: BEArrayT) -> BEArrayT:
        """Evaluate aligned residuals without rounding away the centered intercept."""
        x_term = _centered_product(self.slope_x, x, self.x_ref)
        y_term = _centered_product(self.slope_y, y, self.y_ref)
        affine_term = _add_terms([x_term, y_term])
        needs_joint = be.logical_not(be.isfinite(affine_term))
        if be.any(needs_joint):
            joint_term = _combined_centered_product(
                self.slope_x,
                x[needs_joint],
                self.x_ref,
                self.slope_y,
                y[needs_joint],
                self.y_ref,
            )
            replacements = be.where(
                be.isfinite(joint_term), joint_term, affine_term[needs_joint]
            )
            affine_term = be.copy(affine_term)
            affine_term[needs_joint] = replacements
        centered_opd = _add_terms([opd, -self.opd_anchor])
        residual = _add_terms([centered_opd, -self.intercept_correction, -affine_term])
        needs_raw = be.logical_not(be.isfinite(residual))
        if not be.any(needs_raw):
            return residual
        intercept = _add_terms([self.opd_anchor, self.intercept_correction])
        with be.errstate(over="ignore", invalid="ignore"):
            raw_residual = _add_terms(
                [
                    opd[needs_raw],
                    -intercept,
                    -(self.slope_x * x[needs_raw]),
                    self.slope_x * self.x_ref,
                    -(self.slope_y * y[needs_raw]),
                    self.slope_y * self.y_ref,
                ]
            )
        replacements = be.where(
            be.isfinite(raw_residual), raw_residual, residual[needs_raw]
        )
        residual = be.copy(residual)
        residual[needs_raw] = replacements
        return residual
