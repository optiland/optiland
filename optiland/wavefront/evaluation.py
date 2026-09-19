"""Weighted evaluation and detrending of supplied OPD samples in waves.

Validation establishes support and mode requirements once. Explicit raw, piston,
and affine paths then construct the result. Shared forward kernels live in
``_evaluation_math``; optional eager Torch derivatives live in ``_torch_evaluation``.
Finite excluded diagnostics remain differentiable when representable. Invalid or
overflowing diagnostics are isolated from the selected-sample graph.
"""

# ruff: noqa: I002

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Complex, Real
from typing import Generic, Literal

import numpy as np

import optiland.backend as be
from optiland._types import BEArrayT

from ._evaluation_math import (
    _add_terms,
    _AffineFit,
    _centered_product,
    _normalized_sqrt_weights,
    _stop_gradient,
    _sum_samples_forward,
    _two_diff,
    _weighted_rms_forward,
    _weighted_scale,
)

WavefrontRemoval = Literal["none", "piston", "piston_tilt"]

__all__ = [
    "WavefrontEvaluationNumericalError",
    "WavefrontEvaluationResult",
    "WavefrontRemoval",
    "evaluate_wavefront",
]


class WavefrontEvaluationNumericalError(RuntimeError):
    """Raised when a requested finite numerical result cannot be produced."""


@dataclass(frozen=True)
class WavefrontEvaluationResult(Generic[BEArrayT]):
    """Weighted OPD statistics and fit diagnostics.

    Attributes:
        residual_opd: Residual OPD in waves, aligned with the input samples.
        rms: Weighted population RMS on positive-weight support, in waves.
        coefficients: Centered affine coefficients ``(a, b, c)``. Piston removal
            returns ``(mean, 0, 0)``; no removal returns zeros. An absolute mean
            can round even when separately centered residuals remain accurate.
        coordinate_reference: Weighted coordinate centroid ``(x_ref, y_ref)``
            for an affine fit, otherwise ``None``.
        used_mask: Boolean mask identifying strictly positive-weight samples.
        n_used: Number of samples in the statistical support.
        fit_rank: Zero for no fit, one for piston, three for an affine fit.
        condition_number: Condition number of the weighted, centered, scaled
            affine design, not caller-unit coefficients; otherwise ``None``.
        rcond_used: Relative affine singular-value cutoff, otherwise ``None``.
    """

    residual_opd: BEArrayT
    rms: BEArrayT
    coefficients: BEArrayT
    coordinate_reference: tuple[BEArrayT, BEArrayT] | None
    used_mask: BEArrayT
    n_used: int
    fit_rank: int
    condition_number: BEArrayT | None
    rcond_used: BEArrayT | None


@dataclass(frozen=True)
class _PreparedWavefrontInputs(Generic[BEArrayT]):
    """Validated mode, aligned arrays, and selected statistical support.

    Affine mode guarantees that full and selected x/y arrays are present and
    that the selected coordinates are finite. Other modes do not use coordinates.
    """

    remove: WavefrontRemoval
    opd: BEArrayT
    x: BEArrayT | None
    y: BEArrayT | None
    used_mask: BEArrayT
    opd_used: BEArrayT
    weights_used: BEArrayT
    x_used: BEArrayT | None
    y_used: BEArrayT | None
    n_used: int


def evaluate_wavefront(
    opd_waves: BEArrayT | Sequence[float],
    *,
    remove: WavefrontRemoval,
    weights: BEArrayT | Sequence[float] | None = None,
    x: BEArrayT | Sequence[float] | None = None,
    y: BEArrayT | Sequence[float] | None = None,
    rcond: float | None = None,
) -> WavefrontEvaluationResult[BEArrayT]:
    """Evaluate explicitly weighted OPD samples without tracing.

    Weights are final effective weights, never multiplied by intensity,
    apodization, or another inferred factor. Only positive weights participate;
    fitting and RMS use the same support and population denominator.

    Active-backend arrays retain dtype, device, and OPD autograd attachment.
    Python sequences are converted to the OPD dtype on the configured backend
    device. Existing arrays must be real float32/float64 with matching dtype and
    device. Inputs must be one-dimensional, without flattening or broadcasting.
    Masked arrays and sequence conversions that erase nonzero weights are rejected.

    Args:
        opd_waves: Already referenced OPD samples in waves.
        remove: Explicitly remove ``"none"``, ``"piston"``, or ``"piston_tilt"``.
        weights: Final finite nonnegative weights; ``None`` means equal samples.
        x: Coordinates for the x slope, required for affine removal.
        y: Coordinates for the y slope, required for affine removal.
        rcond: Real scalar relative rank cutoff, finite and in [0, 1). Python
            and NumPy real scalars are accepted; booleans, strings, arrays, and
            tensors are rejected. ``None`` uses ``eps(dtype)*max(n_used, 3)``.

    Returns:
        Residual statistics and fit diagnostics. Affine coefficients represent
        ``a + b*(x-x_ref) + c*(y-y_ref)`` in the supplied coordinate chart.

    Raises:
        TypeError: For invalid representations, dtypes, devices, or cutoff types.
        ValueError: For invalid values, shapes, support, mode, cutoff, or fit rank.
        WavefrontEvaluationNumericalError: For SVD convergence failure or
            unrepresentable finite coefficients or positive-support residuals.

    Notes:
        The tested matrix is NumPy CPU and Torch CPU float32/float64. Other Torch
        devices depend on the backend's SVD support; arrays are never moved.

        OPD gradients are supported through eager ``backward()`` and
        ``torch.autograd.grad`` with coordinates, weights, and support held fixed,
        full affine rank, normal-range nonzero RMS and derivatives, and normal-range
        nonzero components ``sqrt_alpha*(residual/max(abs(residual)))`` and their
        norm. Exact zero components are allowed. Subnormal forward values remain
        supported. Higher derivatives, forward mode, ``torch.func`` transforms,
        and metadata derivatives are not supported. Detach metadata explicitly
        when constructing a fixed-metadata objective.

        Two-part centering preserves resolvable residuals even when the reported
        absolute intercept rounds. Both passes use every positive weight. The
        default rank cutoff depends on sample count; use explicit ``rcond`` when
        comparing rank decisions across different sample counts.

        Selected residuals retain the OPD graph. Rebuilding RMS preserves its
        supported derivative only if the downstream implementation is itself
        stable over the same range, for example another call with ``remove="none"``.
        Finite excluded diagnostics retain their derivatives when representable;
        invalid or overflowing diagnostics remain nonfinite and graph-independent.
    """
    prepared = _prepare_wavefront_inputs(
        opd_waves, remove=remove, weights=weights, x=x, y=y
    )
    return _evaluate_prepared(prepared, rcond=rcond)


def _evaluate_prepared(
    prepared: _PreparedWavefrontInputs[BEArrayT], *, rcond: float | None
) -> WavefrontEvaluationResult[BEArrayT]:
    """Select the validated mode after any native intensity safety check."""
    rcond_value = _resolve_rcond(rcond, prepared.opd, prepared.n_used)
    sqrt_alpha = _normalized_sqrt_weights(prepared.weights_used)
    if prepared.remove == "none":
        return _evaluate_raw(prepared, sqrt_alpha)
    if prepared.remove == "piston":
        return _evaluate_piston(prepared, sqrt_alpha)
    return _evaluate_affine(prepared, sqrt_alpha, rcond_value)


def _evaluate_raw(
    prepared: _PreparedWavefrontInputs[BEArrayT], sqrt_alpha: BEArrayT
) -> WavefrontEvaluationResult[BEArrayT]:
    """Evaluate raw RMS and preserve aligned diagnostics without fitting."""
    residual = be.copy(prepared.opd)
    invalid_excluded = be.logical_and(
        be.logical_not(prepared.used_mask), be.logical_not(be.isfinite(prepared.opd))
    )
    if be.any(invalid_excluded):
        residual[invalid_excluded] = _stop_gradient(prepared.opd[invalid_excluded])
    zero = be.sum(prepared.opd_used * 0.0)
    return WavefrontEvaluationResult(
        residual_opd=residual,
        rms=_weighted_rms(prepared.opd_used, sqrt_alpha),
        coefficients=be.concatenate([zero[None], zero[None], zero[None]]),
        coordinate_reference=None,
        used_mask=prepared.used_mask,
        n_used=prepared.n_used,
        fit_rank=0,
        condition_number=None,
        rcond_used=None,
    )


def _evaluate_piston(
    prepared: _PreparedWavefrontInputs[BEArrayT], sqrt_alpha: BEArrayT
) -> WavefrontEvaluationResult[BEArrayT]:
    """Center with a two-part weighted mean and evaluate selected residuals."""
    anchor, correction = _weighted_mean_parts(
        prepared.opd_used, prepared.weights_used, sqrt_alpha
    )
    mean = _add_terms([anchor, correction])
    centered = _add_terms([prepared.opd_used, -anchor])
    residual_used = _add_terms([centered, -correction])
    if not be.all(be.isfinite(residual_used)):
        raise WavefrontEvaluationNumericalError(
            "The piston residual cannot be represented in the input dtype."
        )
    residual = _assemble_piston_diagnostics(prepared, residual_used, anchor, correction)
    zero = mean * 0.0
    return WavefrontEvaluationResult(
        residual_opd=residual,
        rms=_weighted_rms(residual_used, sqrt_alpha),
        coefficients=be.concatenate([mean[None], zero[None], zero[None]]),
        coordinate_reference=None,
        used_mask=prepared.used_mask,
        n_used=prepared.n_used,
        fit_rank=1,
        condition_number=None,
        rcond_used=None,
    )


def _evaluate_affine(
    prepared: _PreparedWavefrontInputs[BEArrayT],
    sqrt_alpha: BEArrayT,
    rcond_value: float,
) -> WavefrontEvaluationResult[BEArrayT]:
    """Solve a centered/scaled weighted affine fit using rank-revealing SVD."""
    x_used, y_used = prepared.x_used, prepared.y_used
    assert x_used is not None and y_used is not None
    weights = prepared.weights_used
    anchor, mean_correction = _weighted_mean_parts(
        prepared.opd_used, weights, sqrt_alpha
    )
    x_ref = _weighted_mean(x_used, weights, sqrt_alpha)
    y_ref = _weighted_mean(y_used, weights, sqrt_alpha)
    weighted_x = _centered_product(sqrt_alpha, x_used, x_ref)
    weighted_y = _centered_product(sqrt_alpha, y_used, y_ref)
    x_scale = _weighted_scale(weighted_x, "x")
    y_scale = _weighted_scale(weighted_y, "y")
    design = be.concatenate(
        [
            sqrt_alpha[:, None],
            (weighted_x / x_scale)[:, None],
            (weighted_y / y_scale)[:, None],
        ],
        axis=1,
    )
    u_matrix, singular_values, vh_matrix = _affine_svd(design)
    if not be.all(be.isfinite(singular_values)):
        raise WavefrontEvaluationNumericalError(
            "The weighted affine decomposition produced nonfinite singular values."
        )
    fit_rank = len(singular_values[singular_values > singular_values[0] * rcond_value])
    if fit_rank < 3:
        raise ValueError(
            "The weighted affine design is numerically rank deficient "
            f"(rank {fit_rank}; expected 3)."
        )
    centered_opd = _add_terms([prepared.opd_used, -anchor])
    target = _centered_product(sqrt_alpha, centered_opd, mean_correction)
    projection = u_matrix.T @ target
    scaled_coefficients = vh_matrix.T @ (projection / singular_values)
    if not be.all(be.isfinite(scaled_coefficients)):
        raise WavefrontEvaluationNumericalError(
            "The weighted affine solve produced nonfinite coefficients."
        )
    intercept_correction = _add_terms([mean_correction, scaled_coefficients[0]])
    intercept = _add_terms([anchor, intercept_correction])
    with be.errstate(over="ignore", divide="ignore", invalid="ignore"):
        slope_x = scaled_coefficients[1] / x_scale
        slope_y = scaled_coefficients[2] / y_scale
    coefficients = be.concatenate([intercept[None], slope_x[None], slope_y[None]])
    lost_slope = (scaled_coefficients[1] != 0.0 and slope_x == 0.0) or (
        scaled_coefficients[2] != 0.0 and slope_y == 0.0
    )
    if not be.all(be.isfinite(coefficients)) or lost_slope:
        raise WavefrontEvaluationNumericalError(
            "The affine coefficients cannot be represented in caller coordinates."
        )
    weighted_residual = _add_terms(
        [
            target,
            -sqrt_alpha * scaled_coefficients[0],
            -design[:, 1] * scaled_coefficients[1],
            -design[:, 2] * scaled_coefficients[2],
        ]
    )
    residual_used = weighted_residual / sqrt_alpha
    if not be.all(be.isfinite(residual_used)):
        raise WavefrontEvaluationNumericalError(
            "The affine residual cannot be represented in the input dtype."
        )
    fit = _AffineFit(anchor, intercept_correction, slope_x, slope_y, x_ref, y_ref)
    residual = _assemble_affine_diagnostics(prepared, residual_used, fit)
    return WavefrontEvaluationResult(
        residual_opd=residual,
        rms=_weighted_rms(residual_used, sqrt_alpha),
        coefficients=coefficients,
        coordinate_reference=(x_ref, y_ref),
        used_mask=prepared.used_mask,
        n_used=prepared.n_used,
        fit_rank=fit_rank,
        condition_number=singular_values[0] / singular_values[-1],
        rcond_used=_local_full_like(singular_values[0], rcond_value),
    )


def _affine_svd(design: BEArrayT) -> tuple[BEArrayT, BEArrayT, BEArrayT]:
    """Translate only the active backend's decomposition failure exception."""
    decomposition_error = np.linalg.LinAlgError
    if be.is_torch_tensor(design):
        import torch

        decomposition_error = torch.linalg.LinAlgError
    try:
        return be.linalg.svd(design, full_matrices=False)
    except decomposition_error as exc:
        raise WavefrontEvaluationNumericalError(
            "The weighted affine singular-value decomposition failed."
        ) from exc


def _assemble_piston_diagnostics(
    prepared: _PreparedWavefrontInputs[BEArrayT],
    residual_used: BEArrayT,
    anchor: BEArrayT,
    correction: BEArrayT,
) -> BEArrayT:
    """Attach finite representable diagnostics; isolate invalid/overflowing ones."""
    opd, used = prepared.opd, prepared.used_mask
    residual = be.copy(opd)
    residual[used] = residual_used
    excluded = be.logical_not(used)
    finite = be.logical_and(excluded, be.isfinite(opd))
    if be.any(finite):
        values = opd[finite]
        centered = _add_terms([_stop_gradient(values), -_stop_gradient(anchor)])
        diagnostic = _add_terms([centered, -_stop_gradient(correction)])
        representable = be.isfinite(diagnostic)
        if be.any(representable):
            centered = _add_terms([values[representable], -anchor])
            diagnostic[representable] = _add_terms([centered, -correction])
        residual[finite] = diagnostic
    invalid = be.logical_and(excluded, be.logical_not(be.isfinite(opd)))
    if be.any(invalid):
        residual[invalid] = _stop_gradient(opd[invalid])
    return residual


def _assemble_affine_diagnostics(
    prepared: _PreparedWavefrontInputs[BEArrayT],
    residual_used: BEArrayT,
    fit: _AffineFit[BEArrayT],
) -> BEArrayT:
    """Probe detached extrapolations, then attach only representable derivatives.

    Finite diagnostic derivatives remain valid at zero fitted slopes. Invalid or
    overflowing probes never enter the coefficient-dependent autograd graph.
    """
    residual = be.copy(prepared.opd)
    residual[prepared.used_mask] = residual_used
    excluded = be.logical_not(prepared.used_mask)
    if not be.any(excluded):
        return residual
    assert prepared.x is not None and prepared.y is not None
    opd, x, y = prepared.opd[excluded], prepared.x[excluded], prepared.y[excluded]
    finite = be.logical_and(
        be.isfinite(opd), be.logical_and(be.isfinite(x), be.isfinite(y))
    )
    diagnostic = _local_full_like(opd, be.nan)
    if be.any(finite):
        finite_opd, finite_x, finite_y = opd[finite], x[finite], y[finite]
        probe = fit.detached().residual(
            _stop_gradient(finite_opd),
            _stop_gradient(finite_x),
            _stop_gradient(finite_y),
        )
        representable = be.isfinite(probe)
        if be.any(representable):
            probe[representable] = fit.residual(
                finite_opd[representable],
                finite_x[representable],
                finite_y[representable],
            )
        diagnostic[finite] = probe
    residual[excluded] = diagnostic
    return residual


def _prepare_wavefront_inputs(
    opd_waves: BEArrayT | Sequence[float],
    *,
    remove: WavefrontRemoval,
    weights: BEArrayT | Sequence[float] | None = None,
    x: BEArrayT | Sequence[float] | None = None,
    y: BEArrayT | Sequence[float] | None = None,
) -> _PreparedWavefrontInputs[BEArrayT]:
    """Validate representation, support, and mode requirements before arithmetic."""
    if remove not in ("none", "piston", "piston_tilt"):
        raise ValueError("remove must be one of 'none', 'piston', or 'piston_tilt'.")
    opd = _convert_input(opd_waves, "opd_waves")
    if opd.ndim != 1:
        raise ValueError("opd_waves must be one-dimensional.")
    _validate_dtype(opd, "opd_waves")
    x_array = _convert_aligned(x, "x", opd)
    y_array = _convert_aligned(y, "y", opd)
    weight_array = _convert_aligned(weights, "weights", opd)
    _validate_sequence_weight_conversion(weights, weight_array)
    if weight_array is None:
        if not be.all(be.isfinite(opd)):
            raise ValueError("opd_waves must be finite on positive-weight support.")
        used_mask = be.isfinite(opd)
        opd_used = opd[used_mask]
        weights_used = _local_full_like(opd_used, 1.0)
    else:
        if not be.all(be.isfinite(weight_array)):
            raise ValueError("weights must be finite.")
        if be.any(weight_array < 0.0):
            raise ValueError("weights must be nonnegative.")
        used_mask = weight_array > 0.0
        if not be.any(used_mask):
            raise ValueError("weights must contain at least one positive value.")
        opd_used, weights_used = opd[used_mask], weight_array[used_mask]
        if not be.all(be.isfinite(opd_used)):
            raise ValueError("opd_waves must be finite on positive-weight support.")
    n_used = len(opd_used)
    if n_used == 0:
        raise ValueError("opd_waves must contain at least one sample.")
    x_used = y_used = None
    if remove == "piston_tilt":
        if x_array is None or y_array is None:
            raise ValueError("x and y are required for remove='piston_tilt'.")
        x_used, y_used = x_array[used_mask], y_array[used_mask]
        if not be.all(be.isfinite(x_used)) or not be.all(be.isfinite(y_used)):
            raise ValueError(
                "x and y must be finite on positive-weight support for an affine fit."
            )
    return _PreparedWavefrontInputs(
        remove=remove,
        opd=opd,
        x=x_array,
        y=y_array,
        used_mask=used_mask,
        opd_used=opd_used,
        weights_used=weights_used,
        x_used=x_used,
        y_used=y_used,
        n_used=n_used,
    )


def _convert_aligned(
    value: BEArrayT | Sequence[float] | None, name: str, opd: BEArrayT
) -> BEArrayT | None:
    """Convert an optional aligned array without changing existing array precision."""
    if value is None:
        return None
    array = _convert_input(value, name, like=opd)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.shape != opd.shape:
        raise ValueError(f"{name} must have the same shape as opd_waves.")
    _validate_dtype(array, name)
    if array.dtype != opd.dtype:
        raise TypeError(f"{name} must have the same dtype as opd_waves.")
    if be.is_torch_tensor(opd) and array.device != opd.device:
        raise TypeError(f"{name} must be on the same device as opd_waves.")
    return array


def _convert_input(
    value: BEArrayT | Sequence[float], name: str, like: BEArrayT | None = None
) -> BEArrayT:
    """Preserve active arrays and convert ordinary array-like input at the boundary."""
    if np.ma.isMaskedArray(value):
        raise TypeError(f"{name} must not be a NumPy MaskedArray.")
    is_torch = be.is_torch_tensor(value)
    is_backend_array = isinstance(value, be.ndarray)
    active_array = (
        is_torch if be.get_backend() == "torch" else is_backend_array and not is_torch
    )
    if is_backend_array and not active_array:
        raise TypeError(f"{name} must use the active {be.get_backend()} backend.")
    if active_array:
        return value
    if isinstance(value, Sequence) and any(
        isinstance(item, Complex) and not isinstance(item, Real) for item in value
    ):
        raise TypeError(f"{name} must contain real values, not complex scalars.")
    try:
        return (
            be.asarray(value) if like is None else be.asarray(value, dtype=like.dtype)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real floating array-like value.") from exc


def _validate_sequence_weight_conversion(
    source: BEArrayT | Sequence[float] | None, converted: BEArrayT | None
) -> None:
    """Reject nonzero sequence weights erased by dtype conversion."""
    if (
        source is None
        or converted is None
        or isinstance(source, be.ndarray)
        or not isinstance(source, Sequence)
    ):
        return
    for source_weight, converted_weight in zip(source, converted, strict=True):
        if bool(source_weight != 0.0) and bool(converted_weight == 0.0):
            raise ValueError(
                "weights contain a nonzero sequence value that became zero "
                "in the opd_waves dtype."
            )


def _validate_dtype(array: BEArrayT, name: str) -> None:
    """Require float32 or float64 for OPD, coordinates, and evaluation weights."""
    if be.is_torch_tensor(array):
        import torch

        supported = array.dtype in (torch.float32, torch.float64)
    else:
        supported = array.dtype in (np.dtype(np.float32), np.dtype(np.float64))
    if not supported:
        raise TypeError(f"{name} must have dtype float32 or float64.")


def _local_full_like(array: BEArrayT, value: float) -> BEArrayT:
    """Create a constant using the input's dtype/device, not backend defaults."""
    if be.is_torch_tensor(array):
        return array.new_full(array.shape, value, requires_grad=False)
    return np.full_like(array, value)


def _resolve_rcond(rcond: float | None, opd: BEArrayT, n_used: int) -> float:
    """Normalize a real scalar cutoff, distinguishing wrong type from bad value."""
    if rcond is None:
        if be.is_torch_tensor(opd):
            import torch

            epsilon = torch.finfo(opd.dtype).eps
        else:
            epsilon = float(np.finfo(opd.dtype).eps)
        return epsilon * max(n_used, 3)
    if isinstance(rcond, (bool, np.bool_)) or not isinstance(rcond, Real):
        raise TypeError("rcond must be a real scalar, not a boolean, array, or tensor.")
    if rcond < 0 or rcond >= 1:
        raise ValueError("rcond must be finite, nonnegative, and less than one.")
    rcond_value = float(rcond)
    if not math.isfinite(rcond_value) or not 0.0 <= rcond_value < 1.0:
        raise ValueError("rcond must be finite, nonnegative, and less than one.")
    return rcond_value


def _weighted_mean_parts(
    values: BEArrayT, weights: BEArrayT, sqrt_alpha: BEArrayT
) -> tuple[BEArrayT, BEArrayT]:
    """Retain a mean anchor and correction, using all weights in both passes."""
    minimum, maximum = be.min(values), be.max(values)
    if minimum > 0.0:
        anchor = minimum
    elif maximum < 0.0:
        anchor = maximum
    else:
        anchor = values[0] * 0.0
    centered = _add_terms([values, -anchor])
    correction = _weighted_average(centered, weights, sqrt_alpha)
    mean = _add_terms([anchor, correction])
    if not be.all(be.isfinite(mean)):
        return anchor, correction
    difference, error = _two_diff(values, mean)
    if not be.all(be.isfinite(difference)):
        return anchor, correction
    return mean, _weighted_average(difference, weights, sqrt_alpha, error=error)


def _weighted_average(
    values: BEArrayT,
    weights: BEArrayT,
    sqrt_alpha: BEArrayT,
    *,
    error: BEArrayT | None = None,
) -> BEArrayT:
    """Average finite values with range-safe weights and compensated reductions.

    If max scaling loses a positive weight, split multiplication into square-root
    factors. Divide terms before summation, then scale values, only when the
    original numerator overflows. Products/divisions still round in the input dtype.
    """
    scaled_weights = weights / be.max(weights)
    lost_weight = be.any(be.logical_and(weights > 0.0, scaled_weights == 0.0))
    if lost_weight:
        denominator = _sum_samples(sqrt_alpha * sqrt_alpha)
        numerator_terms = (sqrt_alpha * values) * sqrt_alpha
        error_terms = None if error is None else (sqrt_alpha * error) * sqrt_alpha
    else:
        denominator = _sum_samples(scaled_weights)
        numerator_terms = scaled_weights * values
        error_terms = None if error is None else scaled_weights * error
    if error_terms is not None:
        numerator_terms = be.concatenate([numerator_terms, error_terms])
    correction = _sum_samples(numerator_terms) / denominator
    if be.all(be.isfinite(correction)):
        return correction
    correction = _sum_samples(numerator_terms / denominator)
    if be.all(be.isfinite(correction)):
        return correction
    value_scale = be.max(be.abs(values))
    if error is not None:
        value_scale = be.maximum(value_scale, be.max(be.abs(error)))
    normalized = values / value_scale
    normalized_terms = (
        (sqrt_alpha * normalized) * sqrt_alpha
        if lost_weight
        else scaled_weights * normalized
    )
    if error is not None:
        normalized_error = error / value_scale
        normalized_error_terms = (
            (sqrt_alpha * normalized_error) * sqrt_alpha
            if lost_weight
            else scaled_weights * normalized_error
        )
        normalized_terms = be.concatenate([normalized_terms, normalized_error_terms])
    correction = _sum_samples(normalized_terms) / denominator
    with be.errstate(over="ignore", invalid="ignore"):
        return value_scale * correction


def _weighted_mean(
    values: BEArrayT, weights: BEArrayT, sqrt_alpha: BEArrayT
) -> BEArrayT:
    """Round the two-part mean only when an absolute coordinate reference is needed."""
    return _add_terms(_weighted_mean_parts(values, weights, sqrt_alpha))


def _sum_samples(values: BEArrayT) -> BEArrayT:
    """Reduce samples with the analytical sum derivative in eager Torch mode."""
    if be.is_torch_tensor(values) and values.requires_grad:
        import torch

        if torch.is_grad_enabled():
            from ._torch_evaluation import _CompensatedSum

            return _CompensatedSum.apply(values)
    return _sum_samples_forward(values)


def _weighted_rms(values: BEArrayT, sqrt_alpha: BEArrayT) -> BEArrayT:
    """Use the custom first-order rule only when OPD differentiation is active."""
    if be.is_torch_tensor(values) and values.requires_grad:
        import torch

        if torch.is_grad_enabled():
            from ._torch_evaluation import _WeightedRMS

            return _WeightedRMS.apply(values, sqrt_alpha)
    return _weighted_rms_forward(values, sqrt_alpha)
