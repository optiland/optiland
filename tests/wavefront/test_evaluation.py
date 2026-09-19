# ruff: noqa: I002

import math
import warnings
from decimal import Decimal, localcontext
from itertools import permutations

import numpy as np
import pytest

import optiland.backend as be
from optiland.wavefront import (
    WavefrontEvaluationNumericalError,
    WavefrontEvaluationResult,
    evaluate_wavefront,
)
from optiland.wavefront.evaluation import _sum_samples
from tests.utils import assert_allclose, assert_array_equal


def _array(values):
    return be.array(values)


def _typed_array(values, dtype_name):
    if be.get_backend() == "numpy":
        return np.array(values, dtype=getattr(np, dtype_name))
    torch = pytest.importorskip("torch")
    return torch.tensor(values, dtype=getattr(torch, dtype_name))


def _dim_outlier_reference(offset, weak_weight, sign, dtype):
    """Return high-precision residual and RMS references for dim outliers."""
    residual, rms, _, _ = _piston_reference(
        [0.0, sign * offset, sign * (offset + 2)],
        [weak_weight, 1.0, 1.0],
        dtype,
    )
    return residual, rms


def _piston_reference(values, weights, dtype):
    """Return high-precision weighted-piston values in the requested dtype."""
    typed_values = np.asarray(values, dtype=dtype)
    typed_weights = np.asarray(weights, dtype=dtype)
    with localcontext() as context:
        context.prec = 800
        decimal_values = [Decimal.from_float(float(value)) for value in typed_values]
        decimal_weights = [
            Decimal.from_float(float(weight)) for weight in typed_weights
        ]
        weight_sum = sum(decimal_weights)
        mean = (
            sum(
                weight * value
                for weight, value in zip(decimal_weights, decimal_values, strict=True)
            )
            / weight_sum
        )
        decimal_residual = [value - mean for value in decimal_values]
        variance = (
            sum(
                weight * residual**2
                for weight, residual in zip(
                    decimal_weights, decimal_residual, strict=True
                )
            )
            / weight_sum
        )
        rms = variance.sqrt()
        gradient = [
            weight * residual / (weight_sum * rms)
            for weight, residual in zip(decimal_weights, decimal_residual, strict=True)
        ]
    return (
        np.asarray([float(value) for value in decimal_residual], dtype=dtype),
        dtype(float(rms)),
        np.asarray([float(value) for value in gradient], dtype=dtype),
        dtype(float(mean)),
    )


def _assert_selected_rms_gradient_matches(
    result, opd, weights, tolerance, *, scaled_reconstruction=False
):
    """Compare stored RMS with the same loss rebuilt from selected residuals."""
    torch = pytest.importorskip("torch")
    used_weights = weights[result.used_mask]
    used_residual = result.residual_opd[result.used_mask]
    if scaled_reconstruction:
        scale = used_residual.detach().abs().max()
        reconstructed = scale * torch.sqrt(
            torch.sum(used_weights * (used_residual / scale).square())
            / used_weights.sum()
        )
    else:
        reconstructed = torch.sqrt(
            torch.sum(used_weights * used_residual.square()) / used_weights.sum()
        )
    stored_gradient = torch.autograd.grad(result.rms, opd, retain_graph=True)[0]
    reconstructed_gradient = torch.autograd.grad(reconstructed, opd)[0]

    assert_allclose(reconstructed, result.rms, rtol=tolerance, atol=tolerance)
    assert be.all(be.isfinite(stored_gradient))
    assert be.all(be.isfinite(reconstructed_gradient))
    assert_allclose(
        reconstructed_gradient,
        stored_gradient,
        rtol=tolerance,
        atol=tolerance,
    )
    return stored_gradient


def _assert_full_piston_case(values, weights, dtype_name, order):
    """Check one weighted-piston case against a high-precision objective."""
    dtype = getattr(np, dtype_name)
    ordered_values = np.asarray(values, dtype=dtype)[list(order)]
    ordered_weights = np.asarray(weights, dtype=dtype)[list(order)]
    expected_residual, expected_rms, expected_gradient, expected_mean = (
        _piston_reference(ordered_values, ordered_weights, dtype)
    )
    opd = _typed_array(ordered_values, dtype_name)
    if be.get_backend() == "torch":
        opd.requires_grad_()
    backend_weights = _typed_array(ordered_weights, dtype_name)

    result = evaluate_wavefront(opd, weights=backend_weights, remove="piston")

    tolerance = 8e-6 if dtype_name == "float32" else 8e-14
    assert_allclose(result.coefficients[0], expected_mean, rtol=tolerance)
    assert_allclose(
        result.residual_opd,
        expected_residual,
        rtol=tolerance,
        atol=tolerance,
    )
    assert_allclose(result.rms, expected_rms, rtol=tolerance)
    residual_scale = float(
        np.sum(np.abs(ordered_weights * expected_residual), dtype=dtype)
    )
    orthogonality_tolerance = max(
        np.finfo(dtype).smallest_normal,
        residual_scale * np.finfo(dtype).eps * 32.0,
    )
    assert_allclose(
        be.sum(backend_weights * result.residual_opd),
        0.0,
        rtol=0.0,
        atol=orthogonality_tolerance,
    )
    if be.get_backend() == "torch":
        result.rms.backward()
        gradient_tolerance = 3e-6 if dtype_name == "float32" else 3e-13
        assert_allclose(
            opd.grad,
            expected_gradient,
            rtol=gradient_tolerance,
            atol=gradient_tolerance,
        )
    return result


@pytest.mark.parametrize("input_name", ["opd_waves", "weights", "x", "y"])
def test_numpy_masked_arrays_are_rejected(set_test_backend, input_name):
    masked = np.ma.array([0.0, np.nan], mask=[False, True])
    kwargs = {
        "opd_waves": _array([0.0, 1.0]),
        "weights": None,
        "x": None,
        "y": None,
    }
    kwargs[input_name] = masked

    with pytest.raises(TypeError, match=rf"{input_name} must not be.*MaskedArray"):
        evaluate_wavefront(remove="none", **kwargs)


@pytest.mark.parametrize("input_name", ["opd_waves", "weights", "x", "y"])
@pytest.mark.parametrize("scalar_type", [np.complex64, np.complex128])
@pytest.mark.parametrize("imaginary", [0.0, 2.0])
def test_complex_scalar_sequences_are_rejected_before_conversion(
    set_test_backend, input_name, scalar_type, imaginary
):
    kwargs = {"opd_waves": _array([0.0, 1.0])}
    kwargs[input_name] = [scalar_type(9.0 + imaginary * 1j), scalar_type(1.0)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match=input_name):
            evaluate_wavefront(remove="piston", **kwargs)
    assert not caught


@pytest.mark.parametrize("input_name", ["opd_waves", "weights", "x", "y"])
@pytest.mark.parametrize("representation", ["python_complex", "backend_array"])
def test_complex_input_controls_remain_rejected(
    set_test_backend, input_name, representation
):
    kwargs = {"opd_waves": _array([0.0, 1.0])}
    values = [9.0 + 2.0j, 1.0 - 2.0j]
    kwargs[input_name] = (
        values
        if representation == "python_complex"
        else _typed_array(values, "complex128")
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match=input_name):
            evaluate_wavefront(remove="piston", **kwargs)
    assert not caught


@pytest.mark.parametrize("input_name", ["opd_waves", "weights", "x", "y"])
@pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
def test_real_scalar_sequences_remain_accepted(
    set_test_backend, input_name, scalar_type
):
    kwargs = {"opd_waves": _array([0.0, 1.0])}
    kwargs[input_name] = (scalar_type(9.0), scalar_type(1.0))

    result = evaluate_wavefront(remove="piston", **kwargs)

    expected = {"opd_waves": 4.0, "weights": 0.3, "x": 0.5, "y": 0.5}
    assert_allclose(result.rms, expected[input_name], rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("tiny_weight", [1e-50, -1e-50])
def test_float32_sequence_weight_conversion_cannot_change_support(
    set_test_backend, tiny_weight
):
    opd = _typed_array([0.0, 1.0], "float32")

    with pytest.raises(ValueError, match="nonzero sequence value.*became zero"):
        evaluate_wavefront(opd, weights=[1.0, tiny_weight], remove="piston")


def test_backend_array_underflow_cannot_be_recovered(set_test_backend):
    opd = _typed_array([0.0, 1.0], "float32")
    weights = _typed_array([1.0, 1e-50], "float32")

    result = evaluate_wavefront(opd, weights=weights, remove="piston")

    assert result.n_used == 1
    assert_array_equal(result.used_mask, _typed_array([1.0, 0.0], "float32") > 0)


@pytest.mark.parametrize("remove", ["none", "piston"])
def test_constant_opd(set_test_backend, remove):
    opd = _array([2.5, 2.5, 2.5])

    result = evaluate_wavefront(opd, remove=remove)

    expected_rms = 2.5 if remove == "none" else 0.0
    assert_allclose(result.rms, expected_rms, atol=1e-14)
    assert result.fit_rank == (0 if remove == "none" else 1)
    assert result.n_used == 3


def test_two_sample_weighted_piston(set_test_backend):
    result = evaluate_wavefront(
        _array([0.0, 1.0]), weights=_array([9.0, 1.0]), remove="piston"
    )

    assert_allclose(result.coefficients, [0.1, 0.0, 0.0])
    assert_allclose(result.residual_opd, [-0.1, 0.9])
    assert_allclose(result.rms, 0.3)


@pytest.mark.parametrize(
    ("dtype_name", "offset", "weak_weight"),
    [("float32", 2**24, 1e-20), ("float64", 2**53, 1e-40)],
)
def test_dim_outlier_preserves_weighted_piston_centering(
    set_test_backend, dtype_name, offset, weak_weight
):
    dtype = getattr(np, dtype_name)
    orders = ([0, 1, 2], [2, 0, 1], [1, 2, 0])
    tolerance = 3e-6 if dtype_name == "float32" else 3e-15

    for remote_weight in (weak_weight, weak_weight * 1e-4, weak_weight * 1e-8):
        for sign in (-1, 1):
            expected_residual, expected_rms = _dim_outlier_reference(
                offset, remote_weight, sign, dtype
            )
            values = np.array([0.0, sign * offset, sign * (offset + 2)], dtype=dtype)
            weights = np.array([remote_weight, 1.0, 1.0], dtype=dtype)
            for shift in (0.0, -sign * offset):
                shifted = values + dtype(shift)
                for order in orders:
                    result = evaluate_wavefront(
                        _typed_array(shifted[list(order)], dtype_name),
                        weights=_typed_array(weights[list(order)], dtype_name),
                        remove="piston",
                    )

                    assert_allclose(
                        result.residual_opd,
                        expected_residual[list(order)],
                        rtol=tolerance,
                        atol=tolerance,
                    )
                    assert_allclose(result.rms, expected_rms, rtol=tolerance)


@pytest.mark.parametrize(
    ("dtype_name", "offset"),
    [("float32", 2**24), ("float64", 2**53)],
)
def test_full_weighted_refinement_is_continuous_across_weight_epsilon(
    set_test_backend, dtype_name, offset
):
    dtype = getattr(np, dtype_name)
    epsilon = dtype(np.finfo(dtype).eps)
    weak_weights = (
        dtype(0.5 * epsilon),
        np.nextafter(epsilon, dtype(0.0), dtype=dtype),
        epsilon,
        np.nextafter(epsilon, dtype(np.inf), dtype=dtype),
    )
    orders = tuple(permutations(range(3)))

    for weak_weight in weak_weights:
        for sign in (-1, 1):
            values = np.asarray([0.0, sign * offset, sign * (offset + 2)], dtype=dtype)
            weights = np.asarray([weak_weight, 1.0, 1.0], dtype=dtype)
            for shift in (0.0, -sign * offset):
                shifted = values + dtype(shift)
                for order in orders:
                    result = _assert_full_piston_case(
                        shifted, weights, dtype_name, order
                    )
                    if (
                        weak_weight == dtype(0.5 * epsilon)
                        and sign == 1
                        and shift == 0.0
                        and order == (0, 1, 2)
                    ):
                        assert_allclose(
                            result.residual_opd[1:],
                            [-0.5, 1.5],
                            rtol=0.0,
                            atol=0.0,
                        )


@pytest.mark.parametrize(
    ("dtype_name", "offset"),
    [("float32", 2**24), ("float64", 2**53)],
)
def test_full_weighted_refinement_handles_unequal_dominant_weights(
    set_test_backend, dtype_name, offset
):
    dtype = getattr(np, dtype_name)
    weak_weight = dtype(0.5 * np.finfo(dtype).eps)
    weights = np.asarray([weak_weight, 1.0, 3.0], dtype=dtype)

    for sign in (-1, 1):
        values = np.asarray([0.0, sign * offset, sign * (offset + 2)], dtype=dtype)
        for shift in (0.0, -sign * offset):
            shifted = values + dtype(shift)
            for order in permutations(range(3)):
                _assert_full_piston_case(shifted, weights, dtype_name, order)


@pytest.mark.parametrize(
    ("dtype_name", "offset"),
    [("float32", 2**24), ("float64", 2**53)],
)
def test_full_weighted_refinement_accumulates_multiple_weak_samples(
    set_test_backend, dtype_name, offset
):
    dtype = getattr(np, dtype_name)
    weak_weight = dtype(0.75 * np.finfo(dtype).eps)
    weights = np.asarray([weak_weight, weak_weight, 1.0, 1.0], dtype=dtype)

    for sign in (-1, 1):
        values = np.asarray([0.0, 0.0, sign * offset, sign * (offset + 2)], dtype=dtype)
        for order in permutations(range(4)):
            _assert_full_piston_case(values, weights, dtype_name, order)


@pytest.mark.parametrize(
    ("dtype_name", "offset"),
    [("float32", 2**24), ("float64", 2**53)],
)
def test_full_weighted_refinement_converges_to_zero_weight_limit(
    set_test_backend, dtype_name, offset
):
    dtype = getattr(np, dtype_name)
    epsilon = np.finfo(dtype).eps
    values = np.asarray([0.0, offset, offset + 2], dtype=dtype)
    distances = []

    for weak_weight in (2.0 * epsilon, epsilon, 0.5 * epsilon, 0.25 * epsilon, 0.0):
        weights = np.asarray([weak_weight, 1.0, 1.0], dtype=dtype)
        result = _assert_full_piston_case(values, weights, dtype_name, (0, 1, 2))
        bright_residual = be.to_numpy(result.residual_opd[1:])
        distances.append(float(np.max(np.abs(bright_residual - [-1.0, 1.0]))))

    assert all(
        current >= following
        for current, following in zip(distances, distances[1:], strict=False)
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_nonuniform_raw_rms(set_test_backend, dtype_name):
    result = evaluate_wavefront(
        _typed_array([3.0, 4.0], dtype_name),
        weights=_typed_array([1.0, 3.0], dtype_name),
        remove="none",
    )

    tolerance = 2e-6 if dtype_name == "float32" else 2e-15
    assert_allclose(result.rms, math.sqrt(14.25), rtol=tolerance)


@pytest.mark.parametrize(
    ("dtype_name", "weight_values"),
    [("float32", [3.0, 4.0, 5.0]), ("float64", [5.0, 9.0])],
)
def test_raw_rms_constant_gradient_is_weighted(
    set_test_backend, dtype_name, weight_values
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    opd = _typed_array([1.0] * len(weight_values), dtype_name).requires_grad_()
    weights = _typed_array(weight_values, dtype_name)

    result = evaluate_wavefront(opd, weights=weights, remove="none")
    result.rms.backward()

    tolerance = 2e-6 if dtype_name == "float32" else 2e-13
    assert_allclose(result.rms, 1.0, rtol=0.0, atol=0.0)
    assert_allclose(
        opd.grad, weights / torch.sum(weights), rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize(
    "opd_values",
    [[1.0, -1.0, 1.0, -1.0], [1.0, -1.0001, 0.9999, -1.0002]],
)
def test_raw_rms_equal_magnitude_and_nearby_gradients(
    set_test_backend, dtype_name, opd_values
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    opd = _typed_array(opd_values, dtype_name).requires_grad_()
    weights = _typed_array([1.0, 2.0, 3.0, 4.0], dtype_name)

    result = evaluate_wavefront(opd, weights=weights, remove="none")
    result.rms.backward()

    expected_rms = math.sqrt(
        sum(w * value**2 for w, value in zip([1, 2, 3, 4], opd_values, strict=True))
        / 10.0
    )
    expected_gradient = weights * opd.detach() / (10.0 * expected_rms)
    tolerance = 3e-6 if dtype_name == "float32" else 3e-13
    assert_allclose(result.rms, expected_rms, rtol=tolerance)
    assert_allclose(opd.grad, expected_gradient, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_raw_rms_implicit_equal_weight_gradient(set_test_backend, dtype_name):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    opd = _typed_array([1.0, -1.0, 1.0, -1.0], dtype_name).requires_grad_()

    result = evaluate_wavefront(opd, remove="none")
    result.rms.backward()

    tolerance = 2e-6 if dtype_name == "float32" else 2e-13
    assert_allclose(opd.grad, [0.25, -0.25, 0.25, -0.25], rtol=tolerance)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_subnormal_raw_rms_forward_value_is_preserved(set_test_backend, dtype_name):
    dtype = getattr(np, dtype_name)
    smallest = np.nextafter(dtype(0.0), dtype(1.0), dtype=dtype)

    result = evaluate_wavefront(
        _typed_array([smallest, smallest], dtype_name), remove="none"
    )

    assert_allclose(result.rms, smallest, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_normal_minimum_raw_rms_gradient_is_supported(set_test_backend, dtype_name):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    smallest_normal = np.finfo(getattr(np, dtype_name)).smallest_normal
    opd = _typed_array([smallest_normal, smallest_normal], dtype_name).requires_grad_()

    result = evaluate_wavefront(opd, remove="none")
    result.rms.backward()

    tolerance = 2e-6 if dtype_name == "float32" else 2e-13
    assert_allclose(result.rms, smallest_normal, rtol=0.0, atol=0.0)
    assert_allclose(opd.grad, [0.5, 0.5], rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_exact_affine_opd(set_test_backend, dtype_name):
    x = _typed_array([0.0, 1.0, 0.0, 2.0, -1.0], dtype_name)
    y = _typed_array([0.0, 0.0, 1.0, -1.0, 2.0], dtype_name)
    opd = 1.5 + 2.0 * x - 3.0 * y

    result = evaluate_wavefront(opd, x=x, y=y, remove="piston_tilt")

    tolerance = 3e-6 if dtype_name == "float32" else 2e-14
    assert_allclose(result.residual_opd, be.zeros_like(opd), atol=tolerance)
    assert_allclose(result.rms, 0.0, atol=tolerance)
    assert result.fit_rank == 3


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_asymmetric_fit_and_coefficient_origin(set_test_backend, dtype_name):
    x = _typed_array([0.0, 1.0, 0.0, 1.0], dtype_name)
    y = _typed_array([0.0, 0.0, 1.0, 1.0], dtype_name)
    opd = _typed_array([0.0, 0.0, 0.0, 1.0], dtype_name)
    weights = _typed_array([1.0, 2.0, 3.0, 4.0], dtype_name)

    result = evaluate_wavefront(opd, x=x, y=y, weights=weights, remove="piston_tilt")

    tolerance = 3e-6 if dtype_name == "float32" else 2e-14
    assert isinstance(result, WavefrontEvaluationResult)
    assert_allclose(result.coordinate_reference[0], 3.0 / 5.0, rtol=tolerance)
    assert_allclose(result.coordinate_reference[1], 7.0 / 10.0, rtol=tolerance)
    assert_allclose(
        result.coefficients,
        [2.0 / 5.0, 18.0 / 25.0, 16.0 / 25.0],
        rtol=tolerance,
        atol=tolerance,
    )
    fitted = (
        result.coefficients[0]
        + result.coefficients[1] * (x - result.coordinate_reference[0])
        + result.coefficients[2] * (y - result.coordinate_reference[1])
    )
    assert_allclose(opd - result.residual_opd, fitted, rtol=tolerance, atol=tolerance)
    raw_intercept = (
        result.coefficients[0]
        - result.coefficients[1] * result.coordinate_reference[0]
        - result.coefficients[2] * result.coordinate_reference[1]
    )
    assert_allclose(raw_intercept, -12.0 / 25.0, rtol=tolerance, atol=tolerance)
    assert_allclose(result.rms, math.sqrt(0.048), rtol=tolerance)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_close_piston_subtraction_and_offset_affine_fit(set_test_backend, dtype_name):
    offset = 1e5 if dtype_name == "float32" else 1e14
    piston = evaluate_wavefront(
        _typed_array([offset, offset + 1.0], dtype_name), remove="piston"
    )

    assert_allclose(piston.residual_opd, [-0.5, 0.5], rtol=0.0, atol=0.0)
    rms_tolerance = 1e-7 if dtype_name == "float32" else 0.0
    assert_allclose(piston.rms, 0.5, rtol=rms_tolerance, atol=0.0)

    x = _typed_array([0.0, 1.0, 0.0, 1.0], dtype_name)
    y = _typed_array([0.0, 0.0, 1.0, 1.0], dtype_name)
    opd = offset + 2.0 * x - 3.0 * y
    affine = evaluate_wavefront(opd, x=x, y=y, remove="piston_tilt")
    tolerance = 2e-5 if dtype_name == "float32" else 2e-14

    assert_allclose(affine.residual_opd, be.zeros_like(opd), atol=tolerance)
    assert_allclose(affine.rms, 0.0, atol=tolerance)


@pytest.mark.parametrize(
    ("dtype_name", "offset"),
    [("float32", float(2**24)), ("float64", float(2**53))],
)
def test_centered_residual_survives_rounded_absolute_mean(
    set_test_backend, dtype_name, offset
):
    baseline = evaluate_wavefront(_typed_array([0.0, 2.0], dtype_name), remove="piston")
    shifted = evaluate_wavefront(
        _typed_array([offset, offset + 2.0], dtype_name), remove="piston"
    )

    tolerance = 2e-6 if dtype_name == "float32" else 2e-13
    assert_allclose(shifted.residual_opd, [-1.0, 1.0], rtol=0.0, atol=0.0)
    assert_allclose(shifted.rms, 1.0, rtol=0.0, atol=0.0)
    assert_allclose(shifted.rms, baseline.rms, rtol=tolerance)

    weights = _typed_array([1.0, 3.0], dtype_name)
    weighted = evaluate_wavefront(
        _typed_array([offset, offset + 2.0], dtype_name),
        weights=weights,
        remove="piston",
    )
    assert_allclose(weighted.residual_opd, [-1.5, 0.5], rtol=0.0, atol=0.0)
    assert_allclose(be.sum(weights * weighted.residual_opd), 0.0, atol=0.0)


@pytest.mark.parametrize(
    ("dtype_name", "offset"),
    [("float32", float(2**24)), ("float64", float(2**53))],
)
def test_affine_residual_survives_rounded_absolute_intercept(
    set_test_backend, dtype_name, offset
):
    x = _typed_array([0.0, 1.0, 0.0, 1.0], dtype_name)
    y = _typed_array([0.0, 0.0, 1.0, 1.0], dtype_name)
    result = evaluate_wavefront(
        _typed_array([offset, offset, offset, offset + 2.0], dtype_name),
        x=x,
        y=y,
        remove="piston_tilt",
    )

    tolerance = 3e-6 if dtype_name == "float32" else 3e-13
    assert_allclose(result.residual_opd, [0.5, -0.5, -0.5, 0.5], rtol=tolerance)
    assert_allclose(result.rms, 0.5, rtol=tolerance)


def test_compensated_weighted_mean_retains_small_contributions(set_test_backend):
    cancellation = evaluate_wavefront(
        _typed_array([1e16, 1.0, -1e16], "float64"), remove="piston"
    )
    assert_allclose(cancellation.coefficients[0], 1.0 / 3.0, rtol=2e-15)

    if be.get_backend() != "torch":
        return
    torch = pytest.importorskip("torch")
    opd = torch.tensor([0.0, 1.0], dtype=torch.float32, requires_grad=True)
    weights = torch.tensor([1e8, 1.0], dtype=torch.float32)
    result = evaluate_wavefront(opd, weights=weights, remove="piston")
    expected_gradient = (
        (weights / weights.sum()) * result.residual_opd.detach() / result.rms.detach()
    )
    result.rms.backward()

    assert_allclose(result.coefficients[0], 1e-8, rtol=2e-6)
    assert_allclose(opd.grad, expected_gradient, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize(
    ("dtype_name", "large", "tiny"),
    [("float32", 1e38, 1e-30), ("float64", 1e308, 1e-300)],
)
def test_compensated_mean_preserves_terms_below_normalized_range(
    set_test_backend, dtype_name, large, tiny
):
    result = evaluate_wavefront(
        _typed_array([large, -large, tiny], dtype_name), remove="piston"
    )

    tolerance = 2e-6 if dtype_name == "float32" else 2e-15
    assert_allclose(result.coefficients[0], tiny / 3.0, rtol=tolerance)
    assert_allclose(result.residual_opd[-1], 2.0 * tiny / 3.0, rtol=tolerance)


@pytest.mark.parametrize(
    ("dtype_name", "tiny"),
    [("float32", 1e-30), ("float64", 1e-300)],
)
def test_rounded_maximum_full_weighted_mean_remains_representable(
    set_test_backend, dtype_name, tiny
):
    dtype = getattr(np, dtype_name)
    maximum = np.finfo(dtype).max
    values = [maximum, maximum, maximum, -1.0]
    weights = [1.0, 1.0, 1.0, tiny]
    orders = ([0, 1, 2, 3], [3, 2, 0, 1], [1, 3, 0, 2])
    expected_residual, expected_rms, _, expected_mean = _piston_reference(
        values, weights, dtype
    )

    for order in orders:
        result = evaluate_wavefront(
            _typed_array([values[index] for index in order], dtype_name),
            weights=_typed_array([weights[index] for index in order], dtype_name),
            remove="piston",
        )
        ordered_residual = expected_residual[list(order)]

        assert_allclose(result.coefficients[0], expected_mean, rtol=0.0, atol=0.0)
        assert be.all(be.isfinite(result.residual_opd))
        assert_array_equal(result.residual_opd, ordered_residual)
        assert be.all(be.isfinite(result.rms))
        assert_allclose(result.rms, expected_rms, rtol=2e-6)


@pytest.mark.parametrize(
    ("dtype_name", "tiny"),
    [("float32", 1e-30), ("float64", 1e-300)],
)
def test_opposite_extreme_pairing_preserves_tiny_sum_and_gradient(
    set_test_backend, dtype_name, tiny
):
    maximum = np.finfo(getattr(np, dtype_name)).max
    values = [maximum, maximum, -maximum, -maximum, tiny]
    orders = ([0, 1, 2, 3, 4], [4, 3, 1, 2, 0], [2, 0, 4, 1, 3])

    for order in orders:
        ordered = _typed_array([values[index] for index in order], dtype_name)
        if be.get_backend() == "torch":
            ordered.requires_grad_()

        result = _sum_samples(ordered)

        expected = _typed_array(tiny, dtype_name)
        assert_allclose(result, expected, rtol=0.0, atol=0.0)
        if be.get_backend() == "torch":
            result.backward()
            assert_array_equal(ordered.grad, _typed_array([1.0] * 5, dtype_name))


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_common_weight_scale_invariance(set_test_backend, scale):
    x = _array([-1.0, 0.0, 1.0, -0.5, 0.75])
    y = _array([0.0, 1.0, 0.25, -1.0, 2.0])
    opd = _array([0.4, -0.2, 1.1, 0.8, -0.7])
    base_weights = _array([1.0, 3.0, 2.0, 5.0, 4.0])
    baseline = evaluate_wavefront(
        opd, x=x, y=y, weights=base_weights, remove="piston_tilt"
    )

    scaled = evaluate_wavefront(
        opd, x=x, y=y, weights=base_weights * scale, remove="piston_tilt"
    )

    assert_allclose(scaled.coefficients, baseline.coefficients, rtol=2e-6, atol=2e-7)
    assert_allclose(scaled.residual_opd, baseline.residual_opd, rtol=2e-6, atol=2e-7)
    assert_allclose(scaled.rms, baseline.rms, rtol=2e-6, atol=2e-7)


def test_float32_common_weight_scale_invariance(set_test_backend):
    if be.get_backend() == "numpy":

        def array(values):
            return np.array(values, dtype=np.float32)
    else:
        torch = pytest.importorskip("torch")

        def array(values):
            return torch.tensor(values, dtype=torch.float32)

    x = array([-1.0, 0.0, 1.0, -0.5, 0.75])
    y = array([0.0, 1.0, 0.25, -1.0, 2.0])
    opd = array([0.4, -0.2, 1.1, 0.8, -0.7])
    weights = array([1.0, 3.0, 2.0, 5.0, 4.0])
    baseline = evaluate_wavefront(opd, x=x, y=y, weights=weights, remove="piston_tilt")

    for scale in (1e-20, 1e20):
        scaled = evaluate_wavefront(
            opd, x=x, y=y, weights=weights * scale, remove="piston_tilt"
        )
        assert_allclose(scaled.coefficients, baseline.coefficients, rtol=2e-5)
        assert_allclose(scaled.rms, baseline.rms, rtol=2e-5)


def test_large_opd_rms_avoids_squaring_overflow(set_test_backend):
    result = evaluate_wavefront(_array([1e200, -1e200]), remove="none")

    assert_allclose(result.rms, 1e200, rtol=1e-14)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_extreme_nonuniform_raw_rms_retains_tiny_weight(set_test_backend, dtype_name):
    dtype = getattr(np, dtype_name)
    finfo = np.finfo(dtype)
    smallest = np.nextafter(dtype(0.0), dtype(1.0), dtype=dtype)
    result = evaluate_wavefront(
        _typed_array([0.0, finfo.max], dtype_name),
        weights=_typed_array([finfo.max, smallest], dtype_name),
        remove="none",
    )
    expected = np.sqrt(dtype(finfo.max * smallest), dtype=dtype)

    tolerance = 2e-4 if dtype_name == "float32" else 2e-15
    assert result.n_used == 2
    assert_allclose(result.rms, expected, rtol=tolerance)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_largest_finite_constant_and_cancellation_are_stable(
    set_test_backend, dtype_name
):
    dtype = getattr(np, dtype_name)
    finfo = np.finfo(dtype)
    constant = _typed_array([finfo.max, finfo.max, finfo.max], dtype_name)
    result = evaluate_wavefront(
        constant,
        weights=_typed_array([0.94280368, 0.66565742, 0.13339576], dtype_name),
        remove="piston",
    )

    assert be.all(be.isfinite(result.coefficients))
    assert_allclose(result.coefficients[0], finfo.max, rtol=0.0)
    assert_allclose(result.residual_opd, _typed_array([0.0, 0.0, 0.0], dtype_name))

    large = dtype(finfo.max * 0.25)
    small = dtype(finfo.max * finfo.eps * 4.0)
    values = np.array([large, -large, large, -large, small], dtype=dtype)
    orders = ([0, 1, 2, 3, 4], [0, 2, 1, 3, 4], [4, 0, 1, 2, 3])
    means = [
        evaluate_wavefront(
            _typed_array(values[list(order)], dtype_name), remove="piston"
        ).coefficients[0]
        for order in orders
    ]
    tolerance = 2e-6 if dtype_name == "float32" else 2e-15
    expected_mean = dtype(small / 5.0)
    for mean in means:
        assert_allclose(mean, expected_mean, rtol=tolerance)
    assert_allclose(means[1], means[0], rtol=0.0)
    assert_allclose(means[2], means[0], rtol=0.0)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("weight_values", [None, [3.0, 5.0]])
@pytest.mark.parametrize("signs", [[1.0, 1.0], [1.0, -1.0]])
@pytest.mark.parametrize("magnitude_factor", [1.0, 0.9])
def test_largest_finite_raw_rms_has_finite_weighted_gradient(
    set_test_backend, dtype_name, weight_values, signs, magnitude_factor
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    maximum = torch.finfo(dtype).max
    magnitude = torch.tensor(maximum * magnitude_factor, dtype=dtype)
    opd = (torch.tensor(signs, dtype=dtype) * magnitude).requires_grad_()
    weights = (
        None if weight_values is None else torch.tensor(weight_values, dtype=dtype)
    )

    result = evaluate_wavefront(opd, weights=weights, remove="none")
    result.rms.backward()

    expected = (
        torch.full((2,), 0.5, dtype=dtype)
        if weights is None
        else weights / torch.sum(weights)
    )
    expected = expected * torch.tensor(signs, dtype=dtype)
    tolerance = 2e-6 if dtype_name == "float32" else 2e-13
    assert result.rms.requires_grad
    assert_allclose(result.rms, magnitude, rtol=0.0, atol=0.0)
    assert be.all(be.isfinite(opd.grad))
    assert_allclose(opd.grad, expected, rtol=tolerance, atol=tolerance)


def test_piston_and_affine_shift_invariance(set_test_backend):
    x = _array([-1.0, 0.0, 1.0, -0.5, 0.5])
    y = _array([0.0, 1.0, 0.0, -1.0, 2.0])
    opd = _array([0.1, 0.7, -0.3, 1.2, -0.8])
    weights = _array([1.0, 4.0, 2.0, 3.0, 5.0])
    piston = evaluate_wavefront(opd, weights=weights, remove="piston")
    affine = evaluate_wavefront(opd, x=x, y=y, weights=weights, remove="piston_tilt")

    shifted_piston = evaluate_wavefront(opd + 8.0, weights=weights, remove="piston")
    shifted_affine = evaluate_wavefront(
        opd + 8.0 - 2.0 * x + 3.0 * y,
        x=x,
        y=y,
        weights=weights,
        remove="piston_tilt",
    )

    assert_allclose(shifted_piston.residual_opd, piston.residual_opd)
    assert_allclose(shifted_piston.rms, piston.rms)
    assert_allclose(shifted_affine.residual_opd, affine.residual_opd, atol=2e-14)
    assert_allclose(shifted_affine.rms, affine.rms, atol=2e-14)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("sample_count", [32, 33, 64, 65])
def test_cancelling_reductions_at_size_boundaries(
    set_test_backend, dtype_name, sample_count
):
    """Compare permutations across fallback boundaries to a Decimal reference."""
    dtype = getattr(np, dtype_name)
    large = 2.0 ** (24 if dtype_name == "float32" else 53)
    values = np.ones(sample_count, dtype=dtype)
    values[:2] = [large, -large]
    with localcontext() as context:
        context.prec = 100
        expected = float(sum(Decimal.from_float(float(v)) for v in values))
    rng = np.random.default_rng(1729)
    orders = [np.arange(sample_count)]
    orders.extend(rng.permutation(sample_count) for _ in range(3))
    for order in orders:
        actual = _sum_samples(_typed_array(values[order], dtype_name))
        assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_large_raw_rms_gradient_with_differentiable_backend_max(
    set_test_backend, dtype_name, monkeypatch
):
    """Norm stabilization must not depend on a detached backend max reduction."""
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    # Patch the module namespace so cleanup removes the override and restores
    # dynamic backend dispatch, rather than pinning a resolved bound method.
    monkeypatch.setitem(be.__dict__, "max", torch.max)
    opd = torch.full(
        (2,), 0.9 * torch.finfo(dtype).max, dtype=dtype, requires_grad=True
    )
    result = evaluate_wavefront(
        opd, weights=torch.tensor([3.0, 5.0], dtype=dtype), remove="none"
    )
    result.rms.backward()

    assert torch.isfinite(result.rms)
    assert torch.all(torch.isfinite(opd.grad))
    tolerance = 2e-6 if dtype_name == "float32" else 2e-13
    assert_allclose(opd.grad, [0.375, 0.625], rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    ("dtype_name", "small_scale", "small_weight"),
    [("float32", 1e-30, 1e-20), ("float64", 1e-250, 1e-100)],
)
@pytest.mark.parametrize("base_values", [[1.0, 1.0], [0.5, -1.0]])
def test_raw_rms_small_gradient_is_invariant_under_opd_scaling(
    set_test_backend, dtype_name, small_scale, small_weight, base_values
):
    """Normal-range derivatives must survive tiny forward OPD scales."""
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    base = torch.tensor(base_values, dtype=dtype)
    weights = torch.tensor([small_weight, 1.0], dtype=dtype)
    alpha = weights / weights.sum()
    reference_rms = torch.sqrt(torch.sum(alpha * base.square()))
    expected = alpha * base / reference_rms
    tolerance = 3e-6 if dtype_name == "float32" else 3e-14
    gradients = []

    for scale in [1.0, small_scale]:
        opd = (base * scale).requires_grad_()
        result = evaluate_wavefront(opd, weights=weights, remove="none")
        result.rms.backward()
        gradients.append(opd.grad)

        assert_allclose(result.rms, reference_rms * scale, rtol=tolerance, atol=0.0)
        assert torch.all(torch.abs(expected) >= torch.finfo(dtype).tiny)
        assert_allclose(opd.grad, expected, rtol=tolerance, atol=0.0)

    assert_allclose(gradients[0], gradients[1], rtol=tolerance, atol=0.0)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("sample_count", [4, 29, 32, 33, 61, 64, 65, 129])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_piston_multiple_large_cancellations_preserve_mean_and_small_residuals(
    set_test_backend, dtype_name, sample_count, nonuniform
):
    """Exercise both centering passes against Decimal in deterministic orders."""
    dtype = getattr(np, dtype_name)
    large = 2.0 ** (24 if dtype_name == "float32" else 53)
    block_count = sample_count // 4
    values = np.asarray(
        [large, large + 2.0, -large, -large] * block_count + [1.0] * (sample_count % 4),
        dtype=dtype,
    )
    weights = np.ones(sample_count, dtype=dtype)
    if nonuniform:
        for block in range(block_count):
            weights[4 * block : 4 * block + 4] = 2.0 ** (block % 4 - 3)
        weights[4 * block_count :] = 0.25
    expected_residual, _, _, expected_mean = _piston_reference(values, weights, dtype)
    orders = [np.arange(sample_count), np.arange(sample_count)[::-1]]
    orders.append(np.argsort(values))
    orders.append(np.random.default_rng(29).permutation(sample_count))
    tolerance = 3e-6 if dtype_name == "float32" else 3e-14

    for order in orders:
        opd = _typed_array(values[order], dtype_name)
        if be.get_backend() == "torch":
            opd.requires_grad_()
        result = evaluate_wavefront(
            opd, weights=_typed_array(weights[order], dtype_name), remove="piston"
        )

        assert_allclose(result.coefficients[0], expected_mean, rtol=tolerance, atol=0.0)
        small = np.abs(values[order]) < 2.0
        if np.any(small):
            indices = np.flatnonzero(small).tolist()
            assert_allclose(
                result.residual_opd[indices],
                expected_residual[order][small],
                rtol=tolerance,
                atol=0.0,
            )
        if be.get_backend() == "torch":
            result.coefficients[0].backward()
            assert_allclose(
                opd.grad,
                weights[order] / weights.sum(),
                rtol=tolerance,
                atol=0.0,
            )


def test_coordinate_translation_and_rescaling(set_test_backend):
    x = _array([-1.0, 0.0, 2.0, -0.5, 1.5])
    y = _array([0.0, 1.0, -0.5, 2.0, 1.25])
    opd = _array([0.2, 0.9, -0.4, 1.3, -0.6])
    weights = _array([1.0, 2.0, 4.0, 3.0, 5.0])
    baseline = evaluate_wavefront(opd, x=x, y=y, weights=weights, remove="piston_tilt")

    transformed = evaluate_wavefront(
        opd,
        x=3.0 * x + 7.0,
        y=-2.0 * y - 4.0,
        weights=weights,
        remove="piston_tilt",
    )

    assert_allclose(transformed.residual_opd, baseline.residual_opd, atol=2e-14)
    assert_allclose(transformed.coefficients[0], baseline.coefficients[0])
    assert_allclose(transformed.coefficients[1], baseline.coefficients[1] / 3.0)
    assert_allclose(transformed.coefficients[2], baseline.coefficients[2] / -2.0)
    assert_allclose(
        transformed.coordinate_reference[0],
        3.0 * baseline.coordinate_reference[0] + 7.0,
    )
    assert_allclose(
        transformed.coordinate_reference[1],
        -2.0 * baseline.coordinate_reference[1] - 4.0,
    )


def test_permutation_and_sample_splitting_invariance(set_test_backend):
    x = _array([-1.0, 0.0, 1.0, -0.5, 0.75])
    y = _array([0.0, 1.0, 0.25, -1.0, 2.0])
    opd = _array([0.4, -0.2, 1.1, 0.8, -0.7])
    weights = _array([1.0, 3.0, 2.0, 5.0, 4.0])
    baseline = evaluate_wavefront(opd, x=x, y=y, weights=weights, remove="piston_tilt")
    order = [3, 0, 4, 1, 2]
    permuted = evaluate_wavefront(
        opd[order],
        x=x[order],
        y=y[order],
        weights=weights[order],
        remove="piston_tilt",
    )
    split = evaluate_wavefront(
        _array([0.4, -0.2, 1.1, 0.8, -0.7, 0.8]),
        x=_array([-1.0, 0.0, 1.0, -0.5, 0.75, -0.5]),
        y=_array([0.0, 1.0, 0.25, -1.0, 2.0, -1.0]),
        weights=_array([1.0, 3.0, 2.0, 2.5, 4.0, 2.5]),
        remove="piston_tilt",
    )

    assert_allclose(permuted.rms, baseline.rms)
    assert_allclose(permuted.coefficients, baseline.coefficients)
    assert_allclose(split.rms, baseline.rms)
    assert_allclose(split.coefficients, baseline.coefficients)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_weighted_affine_residual_orthogonality(set_test_backend, dtype_name):
    x = _typed_array([-1.0, 0.0, 1.0, -0.5, 0.75], dtype_name)
    y = _typed_array([0.0, 1.0, 0.25, -1.0, 2.0], dtype_name)
    opd = _typed_array([0.4, -0.2, 1.1, 0.8, -0.7], dtype_name)
    weights = _typed_array([1.0, 3.0, 2.0, 5.0, 4.0], dtype_name)

    result = evaluate_wavefront(opd, x=x, y=y, weights=weights, remove="piston_tilt")

    residual = result.residual_opd
    x_centered = x - result.coordinate_reference[0]
    y_centered = y - result.coordinate_reference[1]
    tolerance = 2e-5 if dtype_name == "float32" else 2e-14
    assert_allclose(be.sum(weights * residual), 0.0, atol=tolerance)
    assert_allclose(be.sum(weights * residual * x_centered), 0.0, atol=tolerance)
    assert_allclose(be.sum(weights * residual * y_centered), 0.0, atol=tolerance)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_weighted_piston_residual_orthogonality(set_test_backend, dtype_name):
    opd = _typed_array([0.4, -0.2, 1.1, 0.8, -0.7], dtype_name)
    weights = _typed_array([1.0, 3.0, 2.0, 5.0, 4.0], dtype_name)

    result = evaluate_wavefront(opd, weights=weights, remove="piston")

    tolerance = 2e-6 if dtype_name == "float32" else 2e-15
    assert_allclose(be.sum(weights * result.residual_opd), 0.0, atol=tolerance)


def test_zero_weight_invalid_samples_are_excluded_before_arithmetic(set_test_backend):
    opd = _array([0.2, 0.9, -0.4, 1.3, float("nan")])
    x = _array([0.0, 1.0, 0.0, 1.0, float("nan")])
    y = _array([0.0, 0.0, 1.0, 1.0, float("nan")])
    weights = _array([1.0, 2.0, 3.0, 4.0, 0.0])

    result = evaluate_wavefront(opd, x=x, y=y, weights=weights, remove="piston_tilt")
    baseline = evaluate_wavefront(
        opd[:4], x=x[:4], y=y[:4], weights=weights[:4], remove="piston_tilt"
    )

    assert_allclose(result.rms, baseline.rms)
    assert_allclose(result.residual_opd[:4], baseline.residual_opd)
    assert be.all(be.isnan(result.residual_opd[4]))
    assert_array_equal(result.used_mask, [True, True, True, True, False])
    assert result.n_used == 4


@pytest.mark.parametrize("remove", ["none", "piston", "piston_tilt"])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_invalid_excluded_diagnostics_do_not_contaminate_residual_gradients(
    set_test_backend, remove, dtype_name, invalid_value
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    opd = _typed_array([0.0, 0.0, 0.0, 1.0, invalid_value], dtype_name)
    opd.requires_grad_()
    x = _typed_array([0.0, 1.0, 0.0, 1.0, invalid_value], dtype_name)
    y = _typed_array([0.0, 0.0, 1.0, 1.0, invalid_value], dtype_name)
    weights = _typed_array([1.0, 2.0, 3.0, 4.0, 0.0], dtype_name)

    result = evaluate_wavefront(opd, x=x, y=y, weights=weights, remove=remove)

    tolerance = 3e-5 if dtype_name == "float32" else 3e-12
    gradient = _assert_selected_rms_gradient_matches(result, opd, weights, tolerance)
    assert not be.all(be.isfinite(result.residual_opd[4:]))
    assert_allclose(gradient[4], 0.0, atol=0.0)
    if remove == "piston_tilt":
        expected = [
            0.21908902300206645,
            -0.21908902300206645,
            -0.21908902300206645,
            0.21908902300206645,
            0.0,
        ]
        assert_allclose(gradient, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("coordinate_name", ["x", "y"])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_nonfinite_excluded_coordinate_does_not_contaminate_residual_gradients(
    set_test_backend, coordinate_name, dtype_name, invalid_value
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    opd = _typed_array([0.0, 0.0, 0.0, 1.0, 5.0], dtype_name)
    opd.requires_grad_()
    x_values = [0.0, 1.0, 0.0, 1.0, 0.5]
    y_values = [0.0, 0.0, 1.0, 1.0, 0.5]
    if coordinate_name == "x":
        x_values[-1] = invalid_value
    else:
        y_values[-1] = invalid_value
    weights = _typed_array([1.0, 2.0, 3.0, 4.0, 0.0], dtype_name)

    result = evaluate_wavefront(
        opd,
        x=_typed_array(x_values, dtype_name),
        y=_typed_array(y_values, dtype_name),
        weights=weights,
        remove="piston_tilt",
    )

    tolerance = 3e-5 if dtype_name == "float32" else 3e-12
    _assert_selected_rms_gradient_matches(result, opd, weights, tolerance)
    assert not be.all(be.isfinite(result.residual_opd[4:]))


@pytest.mark.parametrize("remove", ["piston", "piston_tilt"])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_finite_excluded_overflow_does_not_contaminate_residual_gradients(
    set_test_backend, remove, dtype_name
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    maximum = np.finfo(getattr(np, dtype_name)).max
    if remove == "piston":
        opd_values = [
            -0.75 * maximum,
            -0.5 * maximum,
            -0.75 * maximum,
            -0.5 * maximum,
            maximum,
        ]
        x_values = [0.0, 1.0, 0.0, 1.0, 0.5]
    else:
        opd_values = [0.0, 0.0, 0.0, 1.0, -maximum]
        x_values = [0.0, 1.0, 0.0, 1.0, maximum]
    opd = _typed_array(opd_values, dtype_name)
    opd.requires_grad_()
    weights = _typed_array([1.0, 2.0, 3.0, 4.0, 0.0], dtype_name)

    result = evaluate_wavefront(
        opd,
        x=_typed_array(x_values, dtype_name),
        y=_typed_array([0.0, 0.0, 1.0, 1.0, 0.5], dtype_name),
        weights=weights,
        remove=remove,
    )

    tolerance = 3e-5 if dtype_name == "float32" else 3e-12
    _assert_selected_rms_gradient_matches(
        result, opd, weights, tolerance, scaled_reconstruction=True
    )
    assert not be.all(be.isfinite(result.residual_opd[4:]))


def test_excluded_nan_coordinates_do_not_poison_torch_opd_gradient(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    opd = torch.tensor(
        [0.2, 0.9, -0.4, 1.3, 5.0], dtype=torch.float64, requires_grad=True
    )
    x = torch.tensor([0.0, 1.0, 0.0, 1.0, float("nan")], dtype=torch.float64)
    y = torch.tensor([0.0, 0.0, 1.0, 1.0, float("nan")], dtype=torch.float64)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0], dtype=torch.float64)

    result = evaluate_wavefront(opd, x=x, y=y, weights=weights, remove="piston_tilt")
    expected = torch.zeros_like(opd)
    expected[:4] = (
        (weights[:4] / weights[:4].sum())
        * result.residual_opd[:4].detach()
        / result.rms.detach()
    )
    result.rms.backward()

    assert be.all(be.isfinite(opd.grad))
    assert_allclose(opd.grad, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_finite_excluded_coordinate_extrapolates_with_zero_slopes(
    set_test_backend, dtype_name
):
    dtype = getattr(np, dtype_name)
    maximum = np.finfo(dtype).max
    base = dtype(-maximum * 0.5)
    delta = dtype(maximum * 0.125)
    result = evaluate_wavefront(
        _typed_array([3.0, 3.0, 3.0, 3.0, 3.0], dtype_name),
        x=_typed_array(
            [base - delta, base + delta, base - delta, base + delta, maximum],
            dtype_name,
        ),
        y=_typed_array([0.0, 0.0, 1.0, 1.0, 0.5], dtype_name),
        weights=_typed_array([1.0, 1.0, 1.0, 1.0, 0.0], dtype_name),
        remove="piston_tilt",
    )

    assert be.all(be.isfinite(result.residual_opd))
    assert_allclose(result.residual_opd[-1], 0.0, atol=0.0)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_excluded_affine_terms_cancel_before_rescaling(set_test_backend, dtype_name):
    dtype = getattr(np, dtype_name)
    maximum = np.finfo(dtype).max
    slope = dtype(maximum * 0.75)
    result = evaluate_wavefront(
        _typed_array([0.0, slope, -slope, 0.0, 0.0], dtype_name),
        x=_typed_array([0.0, 1.0, 0.0, 1.0, 2.0], dtype_name),
        y=_typed_array([0.0, 0.0, 1.0, 1.0, 2.0], dtype_name),
        weights=_typed_array([1.0, 1.0, 1.0, 1.0, 0.0], dtype_name),
        remove="piston_tilt",
    )

    tolerance = maximum * np.finfo(dtype).eps * 2.0
    assert_allclose(result.coordinate_reference, [0.5, 0.5], rtol=0.0, atol=0.0)
    assert_allclose(result.coefficients[1:], [slope, -slope], rtol=2e-6)
    assert be.all(be.isfinite(result.residual_opd))
    assert_allclose(result.residual_opd[-1], 0.0, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_unrepresentable_excluded_piston_residual_is_nonfinite(
    set_test_backend, dtype_name
):
    maximum = np.finfo(getattr(np, dtype_name)).max
    result = evaluate_wavefront(
        _typed_array([-0.75 * maximum, -0.75 * maximum, maximum], dtype_name),
        weights=_typed_array([1.0, 1.0, 0.0], dtype_name),
        remove="piston",
    )

    assert_allclose(result.rms, 0.0, atol=0.0)
    assert be.all(be.isfinite(result.residual_opd[:2]))
    assert not be.all(be.isfinite(result.residual_opd[2:]))


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_unrepresentable_excluded_affine_residual_is_nonfinite(
    set_test_backend, dtype_name
):
    maximum = np.finfo(getattr(np, dtype_name)).max
    result = evaluate_wavefront(
        _typed_array([0.0, 1.0, 0.0, 1.0, -maximum], dtype_name),
        x=_typed_array([0.0, 1.0, 0.0, 1.0, maximum], dtype_name),
        y=_typed_array([0.0, 0.0, 1.0, 1.0, 0.0], dtype_name),
        weights=_typed_array([1.0, 1.0, 1.0, 1.0, 0.0], dtype_name),
        remove="piston_tilt",
    )

    assert be.all(be.isfinite(result.residual_opd[:4]))
    assert not be.all(be.isfinite(result.residual_opd[4:]))
    assert be.all(be.isfinite(result.rms))


def test_unused_coordinates_require_shape_but_not_finiteness(set_test_backend):
    result = evaluate_wavefront(
        _array([0.0, 1.0]),
        x=_array([float("nan"), float("inf")]),
        y=_array([float("nan"), float("inf")]),
        remove="piston",
    )

    assert_allclose(result.rms, 0.5)
    with pytest.raises(ValueError, match="same shape"):
        evaluate_wavefront(_array([0.0, 1.0]), x=_array([0.0]), remove="none")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"remove": "defocus"}, "remove must"),
        ({"remove": "none", "weights": [1.0, -1.0]}, "nonnegative"),
        ({"remove": "none", "weights": [1.0, float("nan")]}, "finite"),
        ({"remove": "none", "weights": [0.0, 0.0]}, "positive"),
        ({"remove": "none", "weights": [1.0]}, "same shape"),
        ({"remove": "none", "rcond": -1.0}, "rcond"),
    ],
)
def test_invalid_arguments_fail_explicitly(set_test_backend, kwargs, message):
    with pytest.raises(ValueError, match=message):
        evaluate_wavefront(_array([0.0, 1.0]), **kwargs)


def test_empty_and_multidimensional_inputs_fail(set_test_backend):
    with pytest.raises(ValueError, match="at least one sample"):
        evaluate_wavefront(_array([]), remove="none")
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate_wavefront(_array([[0.0, 1.0]]), remove="none")
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate_wavefront(
            _array([0.0, 1.0]), weights=_array([[1.0, 1.0]]), remove="none"
        )


def test_positive_weight_invalid_data_fail(set_test_backend):
    with pytest.raises(ValueError, match="opd_waves"):
        evaluate_wavefront(
            _array([0.0, float("nan")]),
            weights=_array([1.0, 1.0]),
            remove="piston",
        )
    with pytest.raises(ValueError, match="x and y"):
        evaluate_wavefront(
            _array([0.0, 1.0, 2.0]),
            x=_array([0.0, 1.0, float("nan")]),
            y=_array([0.0, 0.0, 1.0]),
            remove="piston_tilt",
        )
    with pytest.raises(ValueError, match="required"):
        evaluate_wavefront(_array([0.0, 1.0, 2.0]), remove="piston_tilt")


@pytest.mark.parametrize("rcond", [float("nan"), float("inf"), 1.0, 10**1000])
def test_invalid_rcond_fails(set_test_backend, rcond):
    with pytest.raises(ValueError, match="rcond"):
        evaluate_wavefront(
            _array([0.0, 1.0, 2.0]),
            x=_array([0.0, 1.0, 0.0]),
            y=_array([0.0, 0.0, 1.0]),
            remove="piston_tilt",
            rcond=rcond,
        )


@pytest.mark.parametrize(
    "rcond",
    [True, False, np.bool_(False), "0.01", [0.1], np.array(0.1), np.array([0.1]), 0.1j],
)
def test_rcond_rejects_non_real_scalar_types(set_test_backend, rcond):
    with pytest.raises(TypeError, match="rcond must be a real scalar"):
        evaluate_wavefront(_array([0.0, 1.0]), remove="none", rcond=rcond)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_rcond_rejects_tensors_without_scalar_conversion(
    set_test_backend, requires_grad
):
    torch = pytest.importorskip("torch")
    rcond = torch.tensor(0.1, requires_grad=requires_grad)
    with pytest.raises(TypeError, match="rcond must be a real scalar"):
        evaluate_wavefront(_array([0.0, 1.0]), remove="none", rcond=rcond)


@pytest.mark.parametrize(
    "rcond", [0, 0.01, np.int64(0), np.float32(0.01), np.float64(0.01)]
)
def test_rcond_accepts_python_and_numpy_real_scalars(set_test_backend, rcond):
    result = evaluate_wavefront(
        _array([0.0, 1.0, 2.0, 4.0]),
        x=_array([0.0, 1.0, 0.0, 1.0]),
        y=_array([0.0, 0.0, 1.0, 1.0]),
        remove="piston_tilt",
        rcond=rcond,
    )
    assert_allclose(result.rcond_used, float(rcond), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("missing", [False, True])
def test_affine_requirements_precede_numerical_evaluation(
    set_test_backend, monkeypatch, missing
):
    from optiland.wavefront import evaluation

    def unexpected_normalization(*args, **kwargs):
        raise AssertionError("Numerical evaluation ran before validation.")

    monkeypatch.setattr(
        evaluation, "_normalized_sqrt_weights", unexpected_normalization
    )
    coordinates = {}
    if not missing:
        coordinates = {
            "x": _array([0.0, float("nan"), 0.0]),
            "y": _array([0.0, 0.0, 1.0]),
        }
    with pytest.raises(ValueError, match="x and y"):
        evaluate_wavefront(_array([0.0, 1.0, 2.0]), remove="piston_tilt", **coordinates)


def test_affine_rank_deficiency_fails(set_test_backend):
    with pytest.raises(ValueError, match="zero positive-weight spread"):
        evaluate_wavefront(
            _array([0.0, 1.0, 2.0]),
            x=_array([0.0, 1.0, 2.0]),
            y=_array([1.0, 1.0, 1.0]),
            remove="piston_tilt",
        )
    with pytest.raises(ValueError, match="rank deficient"):
        evaluate_wavefront(
            _array([0.0, 1.0, 2.0]),
            x=_array([0.0, 1.0, 2.0]),
            y=_array([0.0, 2.0, 4.0]),
            remove="piston_tilt",
        )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_extreme_weight_range_preserves_tiny_positive_rank(
    set_test_backend, dtype_name
):
    dtype = getattr(np, dtype_name)
    finfo = np.finfo(dtype)
    smallest = np.nextafter(dtype(0.0), dtype(1.0), dtype=dtype)
    result = evaluate_wavefront(
        _typed_array([0.0, 1.0, 2.0], dtype_name),
        x=_typed_array([0.0, 1.0, 0.0], dtype_name),
        y=_typed_array([0.0, 0.0, 1.0], dtype_name),
        weights=_typed_array([finfo.max, finfo.max, smallest], dtype_name),
        remove="piston_tilt",
        rcond=0.0,
    )

    assert result.n_used == 3
    assert result.fit_rank == 3
    assert be.all(be.isfinite(result.coefficients))


def test_rcond_controls_rank_and_returns_diagnostics(set_test_backend):
    x = _array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = x + _array([0.0, 1e-4, -1e-4, 1e-4, -1e-4, 0.0])
    opd = _array([0.2, -0.1, 0.7, 1.2, -0.4, 0.9])

    result = evaluate_wavefront(opd, x=x, y=y, remove="piston_tilt", rcond=1e-8)

    assert result.fit_rank == 3
    assert_allclose(result.rcond_used, 1e-8)
    assert float(be.to_numpy(result.condition_number)) > 1e3
    with pytest.raises(ValueError, match="rank deficient"):
        evaluate_wavefront(opd, x=x, y=y, remove="piston_tilt", rcond=1e-2)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_default_rcond_uses_dtype_epsilon_and_sample_count(
    set_test_backend, dtype_name
):
    opd = _typed_array([0.2, -0.1, 0.7, 1.2], dtype_name)
    result = evaluate_wavefront(
        opd,
        x=_typed_array([0.0, 1.0, 0.0, 1.0], dtype_name),
        y=_typed_array([0.0, 0.0, 1.0, 1.0], dtype_name),
        remove="piston_tilt",
    )

    expected = np.finfo(getattr(np, dtype_name)).eps * 4
    assert_allclose(result.rcond_used, expected, rtol=0.0)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_overflow_safe_affine_centering(set_test_backend, dtype_name):
    dtype = getattr(np, dtype_name)
    maximum = np.finfo(dtype).max
    x = _typed_array([-maximum, maximum, -maximum, maximum], dtype_name)
    y = _typed_array([-1.0, -1.0, 1.0, 1.0], dtype_name)
    weights = _typed_array([1.0, 3.0, 1.0, 3.0], dtype_name)
    opd = x * 0.75

    result = evaluate_wavefront(opd, x=x, y=y, weights=weights, remove="piston_tilt")

    tolerance = 3e-6 if dtype_name == "float32" else 3e-15
    assert result.fit_rank == 3
    assert be.all(be.isfinite(result.coefficients))
    assert be.all(be.isfinite(result.residual_opd))
    assert_allclose(result.coefficients[1], 0.75, rtol=tolerance)
    assert_allclose(result.rms / maximum, 0.0, atol=tolerance)


def test_condition_number_uses_scaled_weighted_design(set_test_backend):
    x_values = np.array([-2.0, -0.5, 0.25, 1.0, 3.0])
    y_values = np.array([1.0, -1.0, 2.0, 0.5, -0.25])
    weight_values = np.array([1.0, 4.0, 2.0, 3.0, 5.0])
    alpha = weight_values / weight_values.sum()
    x_centered = x_values - np.sum(alpha * x_values)
    y_centered = y_values - np.sum(alpha * y_values)
    design = np.column_stack(
        [
            np.ones_like(x_values),
            x_centered / np.sqrt(np.sum(alpha * x_centered**2)),
            y_centered / np.sqrt(np.sum(alpha * y_centered**2)),
        ]
    )
    singular_values = np.linalg.svd(design * np.sqrt(alpha)[:, None], compute_uv=False)
    expected_condition = singular_values[0] / singular_values[-1]

    result = evaluate_wavefront(
        _array([0.2, -0.1, 0.7, 1.2, -0.4]),
        x=_array(x_values),
        y=_array(y_values),
        weights=_array(weight_values),
        remove="piston_tilt",
    )

    assert_allclose(result.condition_number, expected_condition, rtol=2e-14)


def test_svd_failure_has_distinct_numerical_error(set_test_backend, monkeypatch):
    if be.get_backend() == "torch":
        torch = pytest.importorskip("torch")
        error_type = torch.linalg.LinAlgError
    else:
        error_type = np.linalg.LinAlgError

    def fail_svd(*args, **kwargs):
        raise error_type("decomposition failed")

    monkeypatch.setattr(be.linalg, "svd", fail_svd)

    with pytest.raises(WavefrontEvaluationNumericalError, match="decomposition"):
        evaluate_wavefront(
            _array([0.2, -0.1, 0.7, 1.2]),
            x=_array([0.0, 1.0, 0.0, 1.0]),
            y=_array([0.0, 0.0, 1.0, 1.0]),
            remove="piston_tilt",
        )


def test_unrelated_svd_runtime_error_propagates(set_test_backend, monkeypatch):
    failure = RuntimeError("unrelated backend runtime failure")

    def fail_svd(*args, **kwargs):
        raise failure

    monkeypatch.setattr(be.linalg, "svd", fail_svd)
    with pytest.raises(RuntimeError) as caught:
        evaluate_wavefront(
            _array([0.0, 1.0, 2.0, 4.0]),
            x=_array([0.0, 1.0, 0.0, 1.0]),
            y=_array([0.0, 0.0, 1.0, 1.0]),
            remove="piston_tilt",
        )
    assert caught.value is failure


def test_affine_projection_shape_error_is_not_reclassified(
    set_test_backend, monkeypatch
):
    original_svd = be.linalg.svd

    def malformed_svd(*args, **kwargs):
        u, s, vh = original_svd(*args, **kwargs)
        return u[:-1], s, vh

    monkeypatch.setattr(be.linalg, "svd", malformed_svd)
    with pytest.raises((ValueError, RuntimeError)) as caught:
        evaluate_wavefront(
            _array([0.0, 1.0, 2.0, 4.0]),
            x=_array([0.0, 1.0, 0.0, 1.0]),
            y=_array([0.0, 0.0, 1.0, 1.0]),
            remove="piston_tilt",
        )
    assert not isinstance(caught.value, WavefrontEvaluationNumericalError)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_unrepresentable_caller_unit_slope_is_numerical_failure(
    set_test_backend, dtype_name
):
    dtype = getattr(np, dtype_name)
    coordinate_step = dtype(np.finfo(dtype).smallest_normal / 8.0)

    with pytest.raises(WavefrontEvaluationNumericalError, match="caller coordinates"):
        evaluate_wavefront(
            _typed_array([0.0, 1.0, 0.0, 1.0], dtype_name),
            x=_typed_array([0.0, coordinate_step, 0.0, coordinate_step], dtype_name),
            y=_typed_array([0.0, 0.0, 1.0, 1.0], dtype_name),
            remove="piston_tilt",
            rcond=0.0,
        )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_unrepresentable_piston_residual_is_numerical_failure(
    set_test_backend, dtype_name
):
    maximum = np.finfo(getattr(np, dtype_name)).max

    with pytest.raises(WavefrontEvaluationNumericalError, match="piston residual"):
        evaluate_wavefront(
            _typed_array([maximum, -maximum], dtype_name),
            weights=_typed_array([3.0, 1.0], dtype_name),
            remove="piston",
        )


def test_backend_mismatch_fails_explicitly(set_test_backend):
    torch = pytest.importorskip("torch")
    if be.get_backend() == "numpy":
        foreign_opd = torch.tensor([0.0, 1.0], dtype=torch.float64)
        active_opd = np.array([0.0, 1.0], dtype=np.float64)
    else:
        foreign_opd = np.array([0.0, 1.0], dtype=np.float64)
        active_opd = torch.tensor([0.0, 1.0], dtype=torch.float64)

    with pytest.raises(TypeError, match="active .* backend"):
        evaluate_wavefront(foreign_opd, remove="none")
    with pytest.raises(TypeError, match="active .* backend"):
        evaluate_wavefront(active_opd, weights=foreign_opd, remove="none")


@pytest.mark.parametrize("dtype_name", ["float16", "int64", "bool", "complex128"])
@pytest.mark.parametrize("input_name", ["opd_waves", "weights"])
def test_unsupported_numeric_dtype_is_rejected(
    set_test_backend, dtype_name, input_name
):
    kwargs = {"opd_waves": _typed_array([0.0, 1.0], "float64")}
    kwargs[input_name] = _typed_array([1.0, 1.0], dtype_name)

    with pytest.raises(
        TypeError, match=rf"{input_name} must have dtype float32 or float64"
    ):
        evaluate_wavefront(remove="none", **kwargs)


@pytest.mark.parametrize("input_name", ["weights", "x", "y"])
def test_mixed_input_precision_is_rejected(set_test_backend, input_name):
    kwargs = {input_name: _typed_array([1.0, 1.0], "float32")}

    with pytest.raises(TypeError, match=rf"{input_name} must have the same dtype"):
        evaluate_wavefront(_typed_array([0.0, 1.0], "float64"), remove="none", **kwargs)


def test_implicit_weights_do_not_use_global_backend_array_factory(
    set_test_backend, monkeypatch
):
    opd = _array([0.2, -0.4, 0.8, 1.1])
    x = _array([0.0, 1.0, 0.0, 1.0])
    y = _array([0.0, 0.0, 1.0, 1.0])

    def unexpected_asarray(*args, **kwargs):
        raise AssertionError("Implicit weights used the global backend factory.")

    monkeypatch.setitem(be.__dict__, "asarray", unexpected_asarray)
    result = evaluate_wavefront(opd, x=x, y=y, remove="piston_tilt")

    assert be.all(be.isfinite(result.rms))


@pytest.mark.parametrize("remove", ["none", "piston", "piston_tilt"])
def test_inputs_are_not_mutated_and_residual_does_not_alias(set_test_backend, remove):
    opd = _array([0.2, -0.4, 0.8, 1.1])
    weights = _array([1.0, 3.0, 2.0, 4.0])
    x = _array([0.0, 1.0, 0.0, 1.0])
    y = _array([0.0, 0.0, 1.0, 1.0])
    originals = [be.copy(value) for value in (opd, weights, x, y)]

    result = evaluate_wavefront(opd, weights=weights, x=x, y=y, remove=remove)
    result.residual_opd[0] = 99.0

    for value, original in zip((opd, weights, x, y), originals, strict=True):
        assert_array_equal(value, original)


def test_backend_dtype_and_device_are_preserved(set_test_backend):
    if be.get_backend() == "numpy":
        module = np
        dtypes = [np.float32, np.float64]
    else:
        module = pytest.importorskip("torch")
        dtypes = [module.float32, module.float64]

    for dtype in dtypes:
        if be.get_backend() == "torch":
            opd = module.tensor([0.2, -0.4, 0.8, 1.1], dtype=dtype)
            x = module.tensor([0.0, 1.0, 0.0, 1.0], dtype=dtype)
            y = module.tensor([0.0, 0.0, 1.0, 1.0], dtype=dtype)
        else:
            opd = np.array([0.2, -0.4, 0.8, 1.1], dtype=dtype)
            x = np.array([0.0, 1.0, 0.0, 1.0], dtype=dtype)
            y = np.array([0.0, 0.0, 1.0, 1.0], dtype=dtype)
        result = evaluate_wavefront(opd, x=x, y=y, remove="piston_tilt")

        assert result.residual_opd.dtype == opd.dtype
        assert result.rms.dtype == opd.dtype
        assert result.coefficients.dtype == opd.dtype
        assert result.condition_number.dtype == opd.dtype
        if be.get_backend() == "torch":
            assert result.residual_opd.device == opd.device
            assert result.rms.device == opd.device


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("remove", ["none", "piston", "piston_tilt"])
def test_torch_opd_gradient_matches_analytic_expression(
    set_test_backend, remove, dtype_name
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    opd = torch.tensor([0.2, -0.4, 0.8, 1.1, -0.7], dtype=dtype, requires_grad=True)
    weights = torch.tensor([1.0, 3.0, 2.0, 4.0, 5.0], dtype=dtype)
    x = torch.tensor([-1.0, 0.0, 1.0, -0.5, 0.75], dtype=dtype)
    y = torch.tensor([0.0, 1.0, 0.25, -1.0, 2.0], dtype=dtype)
    kwargs = {"x": x, "y": y} if remove == "piston_tilt" else {}

    result = evaluate_wavefront(opd, weights=weights, remove=remove, **kwargs)
    expected = (
        (weights / weights.sum()) * result.residual_opd.detach() / result.rms.detach()
    )
    result.rms.backward()

    tolerance = 3e-5 if dtype_name == "float32" else 2e-12
    assert_allclose(opd.grad, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_implicit_equal_weights_preserve_symmetric_affine_gradient(
    set_test_backend, dtype_name
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    opd = _typed_array([0.0, 0.0, 0.0, 1.0], dtype_name).requires_grad_()
    x = _typed_array([0.0, 1.0, 0.0, 1.0], dtype_name)
    y = _typed_array([0.0, 0.0, 1.0, 1.0], dtype_name)

    result = evaluate_wavefront(opd, x=x, y=y, remove="piston_tilt")
    result.rms.backward()

    tolerance = 2e-6 if dtype_name == "float32" else 2e-13
    assert_allclose(result.rms, 0.25, rtol=tolerance)
    assert_allclose(
        opd.grad,
        [0.25, -0.25, -0.25, 0.25],
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_implicit_and_explicit_equal_weights_match_asymmetric_affine_gradient(
    set_test_backend, dtype_name
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    implicit_opd = _typed_array(
        [0.2, -0.4, 0.8, 1.1, -0.7], dtype_name
    ).requires_grad_()
    explicit_opd = implicit_opd.detach().clone().requires_grad_()
    x = _typed_array([-1.0, 0.0, 1.0, -0.5, 0.75], dtype_name)
    y = _typed_array([0.0, 1.0, 0.25, -1.0, 2.0], dtype_name)
    weights = _typed_array([1.0] * 5, dtype_name)

    implicit = evaluate_wavefront(implicit_opd, x=x, y=y, remove="piston_tilt")
    explicit = evaluate_wavefront(
        explicit_opd, weights=weights, x=x, y=y, remove="piston_tilt"
    )
    implicit.rms.backward()
    explicit.rms.backward()

    tolerance = 3e-5 if dtype_name == "float32" else 2e-12
    assert_allclose(implicit.rms, explicit.rms, rtol=tolerance, atol=tolerance)
    assert_allclose(
        implicit.residual_opd,
        explicit.residual_opd,
        rtol=tolerance,
        atol=tolerance,
    )
    assert_allclose(
        implicit_opd.grad, explicit_opd.grad, rtol=tolerance, atol=tolerance
    )


def test_torch_affine_rms_gradient_matches_independent_finite_difference(
    set_test_backend,
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    opd = torch.tensor(
        [0.2, -0.4, 0.8, 1.1, -0.7], dtype=torch.float64, requires_grad=True
    )
    weights = torch.tensor([1.0, 3.0, 2.0, 4.0, 5.0], dtype=torch.float64)
    x = torch.tensor([-1.0, 0.0, 1.0, -0.5, 0.75], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0, 0.25, -1.0, 2.0], dtype=torch.float64)

    result = evaluate_wavefront(opd, weights=weights, x=x, y=y, remove="piston_tilt")
    result.rms.backward()
    analytic = opd.grad.detach().clone()
    step = 1e-6
    finite_difference = torch.empty_like(opd)
    for index in range(len(opd)):
        plus = opd.detach().clone()
        minus = opd.detach().clone()
        plus[index] += step
        minus[index] -= step
        plus_rms = evaluate_wavefront(
            plus, weights=weights, x=x, y=y, remove="piston_tilt"
        ).rms
        minus_rms = evaluate_wavefront(
            minus, weights=weights, x=x, y=y, remove="piston_tilt"
        ).rms
        finite_difference[index] = (plus_rms - minus_rms) / (2.0 * step)

    assert_allclose(analytic, finite_difference, rtol=2e-7, atol=2e-9)


def test_torch_affine_rms_gradcheck_with_implicit_weights(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    opd = torch.tensor(
        [0.2, -0.4, 0.8, 1.1, -0.7], dtype=torch.float64, requires_grad=True
    )
    x = torch.tensor([-1.0, 0.0, 1.0, -0.5, 0.75], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0, 0.25, -1.0, 2.0], dtype=torch.float64)

    def rms(values):
        return evaluate_wavefront(values, x=x, y=y, remove="piston_tilt").rms

    assert torch.autograd.gradcheck(rms, (opd,), eps=1e-6, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("y_slope", [0.0, 0.25])
def test_finite_excluded_diagnostic_gradient_at_zero_fitted_slope(
    set_test_backend, dtype_name, y_slope
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    x = _typed_array([-1.0, 1.0, -1.0, 1.0, 2.0], dtype_name)
    y = _typed_array([-1.0, -1.0, 1.0, 1.0, 3.0], dtype_name)
    opd = _typed_array([1.0, -1.0, -1.0, 1.0, 0.0], dtype_name) + y_slope * y
    opd.requires_grad_()
    result = evaluate_wavefront(
        opd,
        x=x,
        y=y,
        weights=_typed_array([1, 1, 1, 1, 0], dtype_name),
        remove="piston_tilt",
    )

    diagnostic_gradient = torch.autograd.grad(
        result.residual_opd[-1], opd, retain_graph=True
    )[0]
    rms_gradient = torch.autograd.grad(result.rms, opd)[0]
    tolerance = 2e-6 if dtype_name == "float32" else 2e-13
    assert_allclose(
        diagnostic_gradient,
        [1.0, 0.0, -0.5, -1.5, 1.0],
        rtol=tolerance,
        atol=tolerance,
    )
    assert_allclose(
        rms_gradient,
        [0.25, -0.25, -0.25, 0.25, 0.0],
        rtol=tolerance,
        atol=tolerance,
    )


def test_finite_excluded_diagnostic_gradcheck_at_zero_slope(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    opd = _typed_array([1.0, -1.0, -1.0, 1.0, 0.0], "float64").requires_grad_()
    x = _typed_array([-1.0, 1.0, -1.0, 1.0, 2.0], "float64")
    y = _typed_array([-1.0, -1.0, 1.0, 1.0, 3.0], "float64")
    weights = _typed_array([1, 1, 1, 1, 0], "float64")

    def diagnostic(values):
        return evaluate_wavefront(
            values, x=x, y=y, weights=weights, remove="piston_tilt"
        ).residual_opd[-1]

    assert torch.autograd.gradcheck(diagnostic, (opd,))


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("upstream", [3.0, 1e-5])
def test_small_rms_gradient_composes_with_scalar_loss(
    set_test_backend, dtype_name, upstream
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    magnitude, weak_weight = (
        (1e-30, 1e-20) if dtype_name == "float32" else (1e-250, 1e-100)
    )
    opd = _typed_array([magnitude, magnitude], dtype_name).requires_grad_()
    weights = _typed_array([weak_weight, 1.0], dtype_name)
    result = evaluate_wavefront(opd, weights=weights, remove="none")
    (upstream * result.rms).backward()

    tolerance = 3e-6 if dtype_name == "float32" else 3e-14
    assert_allclose(
        opd.grad, upstream * weights / weights.sum(), rtol=tolerance, atol=0.0
    )


def test_rms_second_derivative_is_unsupported(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    opd = _typed_array([1.0, 2.0], "float64").requires_grad_()
    result = evaluate_wavefront(opd, remove="none")
    gradient = torch.autograd.grad(result.rms, opd, create_graph=True)[0]

    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), opd)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_stable_selected_residual_reconstruction_retains_small_gradient(
    set_test_backend, dtype_name
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    magnitude, weak_weight = (
        (1e-30, 1e-20) if dtype_name == "float32" else (1e-250, 1e-100)
    )
    opd = _typed_array([magnitude, magnitude], dtype_name).requires_grad_()
    weights = _typed_array([weak_weight, 1.0], dtype_name)
    result = evaluate_wavefront(opd, weights=weights, remove="none")
    rebuilt = evaluate_wavefront(
        result.residual_opd[result.used_mask],
        weights=weights[result.used_mask],
        remove="none",
    )
    stored_gradient = torch.autograd.grad(result.rms, opd, retain_graph=True)[0]
    rebuilt_gradient = torch.autograd.grad(rebuilt.rms, opd)[0]
    tolerance = 3e-6 if dtype_name == "float32" else 3e-14
    assert_allclose(rebuilt_gradient, weights / weights.sum(), rtol=tolerance, atol=0.0)
    assert_allclose(rebuilt_gradient, stored_gradient, rtol=tolerance, atol=0.0)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_forward_only_evaluation_bypasses_autograd_wrappers(
    set_test_backend, monkeypatch, requires_grad
):
    if be.get_backend() != "torch":
        pytest.skip("Autograd dispatch test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    from optiland.wavefront._torch_evaluation import _CompensatedSum, _WeightedRMS

    def unexpected_apply(*args, **kwargs):
        raise AssertionError("Forward-only evaluation prepared a custom backward.")

    monkeypatch.setattr(_CompensatedSum, "apply", unexpected_apply)
    monkeypatch.setattr(_WeightedRMS, "apply", unexpected_apply)
    opd = _typed_array([1.0, -1.0, 2.0], "float64")
    opd.requires_grad_(requires_grad)
    if requires_grad:
        with torch.no_grad():
            result = evaluate_wavefront(opd, remove="piston")
    else:
        result = evaluate_wavefront(opd, remove="piston")
    assert torch.isfinite(result.rms)
    assert not result.rms.requires_grad
