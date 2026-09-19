# ruff: noqa: I002

from types import SimpleNamespace
from typing import get_origin, get_type_hints
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import optiland.backend as be
from optiland.analysis.rms_vs_field import RmsWavefrontErrorVsField
from optiland.distribution import GaussianQuadrature
from optiland.samples.objectives import CookeTriplet
from optiland.wavefront import (
    OPD,
    Wavefront,
    WavefrontData,
    WavefrontEvaluationResult,
    evaluate_wavefront,
)
from optiland.wavefront.evaluation import _evaluate_prepared
from tests.utils import assert_allclose, assert_array_equal


def _typed_array(values, dtype_name="float64"):
    if be.get_backend() == "numpy":
        return np.array(values, dtype=getattr(np, dtype_name))
    torch = pytest.importorskip("torch")
    return torch.tensor(values, dtype=getattr(torch, dtype_name))


def _wavefront_data(
    *,
    opd=None,
    intensity=None,
    quadrature_weights=None,
    dtype_name="float64",
):
    x = _typed_array([0.0, 1.0, 0.0, 1.0, 0.3], dtype_name)
    y = _typed_array([0.0, 0.0, 1.0, 1.0, 0.7], dtype_name)
    if opd is None:
        opd = [0.2, -0.4, 0.8, 1.1, -0.7]
    if intensity is None:
        intensity = [1.0, 0.8, 0.6, 0.4, 0.2]
    return WavefrontData(
        pupil_x=x,
        pupil_y=y,
        pupil_z=_typed_array([0.0] * 5, dtype_name),
        opd=_typed_array(opd, dtype_name),
        intensity=_typed_array(intensity, dtype_name),
        radius=1.0,
        quadrature_weights=(
            None
            if quadrature_weights is None
            else _typed_array(quadrature_weights, dtype_name)
        ),
    )


def _assert_same_result(actual, expected):
    assert_array_equal(actual.residual_opd, expected.residual_opd)
    assert_array_equal(actual.rms, expected.rms)
    assert_array_equal(actual.coefficients, expected.coefficients)
    assert_array_equal(actual.used_mask, expected.used_mask)
    assert actual.n_used == expected.n_used
    assert actual.fit_rank == expected.fit_rank
    if expected.coordinate_reference is None:
        assert actual.coordinate_reference is None
    else:
        assert_array_equal(
            actual.coordinate_reference[0], expected.coordinate_reference[0]
        )
        assert_array_equal(
            actual.coordinate_reference[1], expected.coordinate_reference[1]
        )
    if expected.condition_number is None:
        assert actual.condition_number is None
        assert actual.rcond_used is None
    else:
        assert_array_equal(actual.condition_number, expected.condition_number)
        assert_array_equal(actual.rcond_used, expected.rcond_used)


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
    assert_allclose(stored_gradient[-1], 0.0, atol=0.0)


@pytest.mark.parametrize("remove", ["none", "piston", "piston_tilt"])
@pytest.mark.parametrize("selection", ["equal", "quadrature", "explicit"])
def test_native_evaluation_matches_direct_for_each_weight_mode(
    set_test_backend, remove, selection
):
    quadrature = [0.05, 0.1, 0.2, 0.25, 0.4]
    explicit = _typed_array([0.7, 0.2, 0.9, 0.3, 0.5])
    data = _wavefront_data(quadrature_weights=quadrature)
    if selection == "equal":
        native = data.evaluate(remove=remove)
        effective_weights = None
    elif selection == "quadrature":
        native = data.evaluate(remove=remove, use_quadrature=True)
        effective_weights = data.quadrature_weights
    else:
        native = data.evaluate(remove=remove, weights=explicit)
        effective_weights = explicit

    direct = evaluate_wavefront(
        data.opd,
        x=data.pupil_x,
        y=data.pupil_y,
        weights=effective_weights,
        remove=remove,
    )

    _assert_same_result(native, direct)


def test_native_weight_selection_errors_are_explicit(set_test_backend):
    data = _wavefront_data()

    with pytest.raises(ValueError, match="mutually exclusive"):
        data.evaluate(
            remove="piston",
            weights=_typed_array([1.0] * 5),
            use_quadrature=True,
        )
    with pytest.raises(ValueError, match="No quadrature weights"):
        data.evaluate(remove="piston", use_quadrature=True)


def test_native_evaluate_annotations_resolve_at_runtime():
    hints = get_type_hints(WavefrontData.evaluate)

    assert get_origin(hints["return"]) is WavefrontEvaluationResult
    assert "remove" in hints
    assert "weights" in hints


@pytest.mark.parametrize("source", ["intensity", "explicit", "quadrature"])
def test_native_evaluation_rejects_numpy_masked_arrays(set_test_backend, source):
    data = _wavefront_data(quadrature_weights=[0.05, 0.1, 0.2, 0.25, 0.4])
    masked = np.ma.array(
        [1.0, np.nan, 1.0, 1.0, 1.0],
        mask=[False, True, False, False, False],
    )
    kwargs = {}
    if source == "intensity":
        data.intensity = masked
    elif source == "explicit":
        kwargs["weights"] = masked
    else:
        data.quadrature_weights = masked
        kwargs["use_quadrature"] = True

    name = source if source == "intensity" else "weights"
    with pytest.raises(TypeError, match=rf"{name} must not be.*MaskedArray"):
        data.evaluate(remove="none", **kwargs)


def test_native_intensity_rejects_complex_dtype(set_test_backend):
    data = _wavefront_data()
    if be.get_backend() == "numpy":
        data.intensity = np.array([1.0] * 5, dtype=np.complex128)
    else:
        torch = pytest.importorskip("torch")
        data.intensity = torch.tensor([1.0] * 5, dtype=torch.complex128)

    with pytest.raises(TypeError, match="real numeric, non-complex dtype"):
        data.evaluate(remove="none")


@pytest.mark.parametrize("intensity_dtype", ["float32", "int64"])
def test_native_intensity_accepts_mixed_precision_and_positive_integers(
    set_test_backend, intensity_dtype
):
    data = _wavefront_data(dtype_name="float64")
    data.intensity = _typed_array([1, 2, 3, 4, 5], intensity_dtype)

    native = data.evaluate(remove="piston")
    direct = evaluate_wavefront(data.opd, remove="piston")

    _assert_same_result(native, direct)


def test_native_intensity_rejects_boolean_dtype(set_test_backend):
    data = _wavefront_data()
    data.intensity = _typed_array([True] * 5, "bool")

    with pytest.raises(TypeError, match="real numeric, non-complex dtype"):
        data.evaluate(remove="none")


@pytest.mark.parametrize("dtype", [object, np.str_])
def test_native_intensity_rejects_nonnumeric_numpy_dtype(set_test_backend, dtype):
    if be.get_backend() != "numpy":
        pytest.skip("NumPy object and string arrays have no Torch equivalent.")
    data = _wavefront_data()
    data.intensity = np.array([1, 1, 1, 1, 1], dtype=dtype)

    with pytest.raises(TypeError, match="real numeric, non-complex dtype"):
        data.evaluate(remove="none")


def test_native_intensity_validation_precedes_evaluation(set_test_backend):
    data = _wavefront_data()
    data.intensity = _typed_array([1.0] * 4)

    with (
        patch(
            "optiland.wavefront.wavefront_data._evaluate_prepared",
            side_effect=AssertionError("evaluation must not run"),
        ),
        pytest.raises(ValueError, match="one-dimensional and aligned"),
    ):
        data.evaluate(remove="none")


@pytest.mark.parametrize("tiny_weight", [1e-50, -1e-50])
def test_native_float32_sequence_weights_cannot_change_support(
    set_test_backend, tiny_weight
):
    data = _wavefront_data(dtype_name="float32")

    with pytest.raises(ValueError, match="nonzero sequence value.*became zero"):
        data.evaluate(
            remove="piston",
            weights=[1.0, tiny_weight, 1.0, 1.0, 1.0],
        )


def test_explicit_integrated_power_is_not_weighted_again(set_test_backend):
    q = _typed_array([0.05, 0.1, 0.2, 0.25, 0.4])
    data = _wavefront_data(
        intensity=[0.9, 0.7, 0.5, 0.3, 0.1],
        quadrature_weights=[0.05, 0.1, 0.2, 0.25, 0.4],
    )
    power = q * data.intensity

    composed = data.evaluate(remove="piston", weights=q * data.intensity)
    integrated = data.evaluate(remove="piston", weights=power)
    direct = evaluate_wavefront(data.opd, weights=power, remove="piston")
    double_weighted = evaluate_wavefront(
        data.opd,
        weights=power * data.intensity,
        remove="piston",
    )

    _assert_same_result(composed, integrated)
    _assert_same_result(integrated, direct)
    assert not be.allclose(integrated.rms, double_weighted.rms)


@pytest.mark.parametrize("selection", ["equal", "quadrature", "explicit"])
@pytest.mark.parametrize("bad_intensity", [0.0, -1.0, float("nan"), float("inf")])
def test_native_evaluation_rejects_unsafe_support_before_solve(
    set_test_backend, selection, bad_intensity
):
    data = _wavefront_data(
        intensity=[1.0, bad_intensity, 1.0, 1.0, 1.0],
        quadrature_weights=[0.05, 0.1, 0.2, 0.25, 0.4],
    )
    kwargs = {}
    if selection == "quadrature":
        kwargs["use_quadrature"] = True
    elif selection == "explicit":
        kwargs["weights"] = _typed_array([0.5, 0.5, 0.5, 0.5, 0.5])

    with (
        patch(
            "optiland.wavefront.wavefront_data._evaluate_prepared",
            side_effect=AssertionError("evaluation must not run"),
        ),
        pytest.raises(ValueError, match="strictly positive"),
    ):
        data.evaluate(remove="piston", **kwargs)


def test_native_intensity_value_check_precedes_rank_validation(set_test_backend):
    data = WavefrontData(
        pupil_x=_typed_array([0.0, 0.0, 0.0, 0.0]),
        pupil_y=_typed_array([0.0, 1.0, 0.0, 1.0]),
        pupil_z=_typed_array([0.0, 0.0, 0.0, 0.0]),
        opd=_typed_array([0.0, 1.0, 2.0, 3.0]),
        intensity=_typed_array([0.0, 1.0, 1.0, 1.0]),
        radius=1.0,
    )

    with (
        patch(
            "optiland.wavefront.wavefront_data._evaluate_prepared",
            side_effect=AssertionError("evaluation must not run"),
        ),
        pytest.raises(ValueError, match="intensity"),
    ):
        data.evaluate(remove="piston_tilt")


@pytest.mark.parametrize("excluded_intensity", [0.0, -1.0, float("nan"), float("inf")])
def test_explicit_zero_weights_exclude_the_same_unsafe_support(
    set_test_backend, excluded_intensity
):
    data = _wavefront_data(intensity=[1.0, excluded_intensity, 1.0, 1.0, 1.0])
    weights = _typed_array([2.0, 0.0, 3.0, 4.0, 5.0])

    with patch(
        "optiland.wavefront.wavefront_data._evaluate_prepared",
        wraps=_evaluate_prepared,
    ) as evaluator:
        native = data.evaluate(remove="piston", weights=weights)
    direct = evaluate_wavefront(data.opd, weights=weights, remove="piston")

    evaluator.assert_called_once()
    _assert_same_result(native, direct)
    assert native.n_used == 4
    assert_array_equal(native.used_mask, weights > 0)


@pytest.mark.parametrize(
    "intensity",
    [
        [1.0, 1.0, 1.0, 1.0],
        [[1.0, 1.0, 1.0, 1.0, 1.0]],
    ],
)
def test_native_evaluation_validates_intensity_alignment(set_test_backend, intensity):
    data = _wavefront_data()
    data.intensity = _typed_array(intensity)

    with pytest.raises(ValueError, match="one-dimensional and aligned"):
        data.evaluate(remove="none")


def test_repeated_cached_evaluation_is_immutable_and_does_not_retrace(
    set_test_backend,
):
    optic = CookeTriplet()
    distribution = GaussianQuadrature()
    distribution.generate_points(num_rings=3)
    field = (0.0, 1.0)
    analysis = OPD(
        optic, field, 0.55, distribution=distribution, assume_sample_order=True
    )
    data = analysis.get_data(field, 0.55)
    originals = {
        name: be.copy(getattr(data, name))
        for name in (
            "pupil_x",
            "pupil_y",
            "pupil_z",
            "opd",
            "intensity",
            "quadrature_weights",
        )
    }
    explicit = be.copy(data.quadrature_weights)
    evaluations = [
        {"remove": "none"},
        {"remove": "piston", "use_quadrature": True},
        {"remove": "piston_tilt", "weights": explicit},
    ]

    with (
        patch.object(optic, "trace", side_effect=AssertionError("unexpected trace")),
        patch.object(
            optic,
            "trace_generic",
            side_effect=AssertionError("unexpected reference trace"),
        ),
    ):
        for _ in range(2):
            for kwargs in evaluations:
                data.evaluate(**kwargs)

    for name, original in originals.items():
        assert_array_equal(getattr(data, name), original)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_native_evaluation_preserves_backend_dtype_and_device(
    set_test_backend, dtype_name
):
    data = _wavefront_data(
        quadrature_weights=[0.05, 0.1, 0.2, 0.25, 0.4],
        dtype_name=dtype_name,
    )

    result = data.evaluate(remove="piston_tilt", use_quadrature=True)

    assert result.residual_opd.dtype == data.opd.dtype
    assert result.rms.dtype == data.opd.dtype
    assert result.coefficients.dtype == data.opd.dtype
    assert result.coordinate_reference[0].dtype == data.opd.dtype
    assert result.coordinate_reference[1].dtype == data.opd.dtype
    assert result.condition_number.dtype == data.opd.dtype
    assert result.rcond_used.dtype == data.opd.dtype
    continuous = [
        result.residual_opd,
        result.rms,
        result.coefficients,
        *result.coordinate_reference,
        result.condition_number,
        result.rcond_used,
    ]
    if be.get_backend() == "torch":
        torch = pytest.importorskip("torch")
        for value in [*continuous, result.used_mask]:
            assert value.device == data.opd.device
        assert result.used_mask.dtype == torch.bool
    else:
        assert result.used_mask.dtype == np.dtype(np.bool_)


def test_native_torch_opd_gradient_matches_direct_contract(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    torch = pytest.importorskip("torch")
    data = _wavefront_data()
    data.opd.requires_grad_()
    weights = torch.tensor([1.0, 3.0, 2.0, 4.0, 5.0], dtype=data.opd.dtype)

    result = data.evaluate(remove="piston_tilt", weights=weights)
    expected = (
        (weights / weights.sum()) * result.residual_opd.detach() / result.rms.detach()
    )
    result.rms.backward()

    assert_allclose(data.opd.grad, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("remove", ["none", "piston", "piston_tilt"])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_native_invalid_excluded_diagnostics_preserve_residual_gradients(
    set_test_backend, remove, dtype_name, invalid_value
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    data = WavefrontData(
        pupil_x=_typed_array([0.0, 1.0, 0.0, 1.0, invalid_value], dtype_name),
        pupil_y=_typed_array([0.0, 0.0, 1.0, 1.0, invalid_value], dtype_name),
        pupil_z=_typed_array([0.0] * 5, dtype_name),
        opd=_typed_array([0.0, 0.0, 0.0, 1.0, invalid_value], dtype_name),
        intensity=_typed_array([1.0] * 5, dtype_name),
        radius=1.0,
    )
    data.opd.requires_grad_()
    weights = _typed_array([1.0, 2.0, 3.0, 4.0, 0.0], dtype_name)

    result = data.evaluate(remove=remove, weights=weights)

    tolerance = 3e-5 if dtype_name == "float32" else 3e-12
    _assert_selected_rms_gradient_matches(result, data.opd, weights, tolerance)
    assert not be.all(be.isfinite(result.residual_opd[4:]))


@pytest.mark.parametrize("coordinate_name", ["x", "y"])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_native_nonfinite_excluded_coordinate_preserves_residual_gradients(
    set_test_backend, coordinate_name, dtype_name, invalid_value
):
    if be.get_backend() != "torch":
        pytest.skip("Gradient test requires the Torch backend.")
    x_values = [0.0, 1.0, 0.0, 1.0, 0.5]
    y_values = [0.0, 0.0, 1.0, 1.0, 0.5]
    if coordinate_name == "x":
        x_values[-1] = invalid_value
    else:
        y_values[-1] = invalid_value
    data = WavefrontData(
        pupil_x=_typed_array(x_values, dtype_name),
        pupil_y=_typed_array(y_values, dtype_name),
        pupil_z=_typed_array([0.0] * 5, dtype_name),
        opd=_typed_array([0.0, 0.0, 0.0, 1.0, 5.0], dtype_name),
        intensity=_typed_array([1.0] * 5, dtype_name),
        radius=1.0,
    )
    data.opd.requires_grad_()
    weights = _typed_array([1.0, 2.0, 3.0, 4.0, 0.0], dtype_name)

    result = data.evaluate(remove="piston_tilt", weights=weights)

    tolerance = 3e-5 if dtype_name == "float32" else 3e-12
    _assert_selected_rms_gradient_matches(result, data.opd, weights, tolerance)
    assert not be.all(be.isfinite(result.residual_opd[4:]))


@pytest.mark.parametrize("remove", ["piston", "piston_tilt"])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_native_finite_excluded_overflow_preserves_residual_gradients(
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
    data = WavefrontData(
        pupil_x=_typed_array(x_values, dtype_name),
        pupil_y=_typed_array([0.0, 0.0, 1.0, 1.0, 0.5], dtype_name),
        pupil_z=_typed_array([0.0] * 5, dtype_name),
        opd=_typed_array(opd_values, dtype_name),
        intensity=_typed_array([1.0] * 5, dtype_name),
        radius=1.0,
    )
    data.opd.requires_grad_()
    weights = _typed_array([1.0, 2.0, 3.0, 4.0, 0.0], dtype_name)

    result = data.evaluate(remove=remove, weights=weights)

    tolerance = 3e-5 if dtype_name == "float32" else 3e-12
    _assert_selected_rms_gradient_matches(
        result, data.opd, weights, tolerance, scaled_reconstruction=True
    )
    assert not be.all(be.isfinite(result.residual_opd[4:]))


def test_real_native_quadrature_dataset_evaluates_cached_opd(set_test_backend):
    optic = CookeTriplet()
    distribution = GaussianQuadrature()
    distribution.generate_points(num_rings=3)
    field = (0.0, 1.0)
    data = OPD(
        optic,
        field,
        0.55,
        distribution=distribution,
        assume_sample_order=True,
    ).get_data(field, 0.55)

    native = data.evaluate(remove="piston", use_quadrature=True)
    direct = evaluate_wavefront(
        data.opd,
        x=data.pupil_x,
        y=data.pupil_y,
        weights=data.quadrature_weights,
        remove="piston",
    )

    _assert_same_result(native, direct)


def test_legacy_opd_rms_keeps_positive_intensity_equal_sample_support(
    set_test_backend,
):
    data = _wavefront_data(
        opd=[1.0, 100.0, 3.0, 100.0, 100.0],
        intensity=[1.0, 0.0, 9.0, 0.0, 0.0],
    )
    analysis = object.__new__(OPD)
    analysis.fields = [(0.0, 0.0)]
    analysis.wavelengths = [0.55]
    analysis.get_data = MagicMock(return_value=data)

    assert_allclose(analysis.rms(), be.sqrt(_typed_array(5.0)))


def test_legacy_rms_vs_field_keeps_all_sample_support(set_test_backend):
    data = _wavefront_data(
        opd=[3.0, 4.0, 0.0, 0.0, 0.0],
        intensity=[1.0, 0.0, 0.0, 0.0, 0.0],
    )
    analysis = object.__new__(RmsWavefrontErrorVsField)
    analysis.fields = [SimpleNamespace(coord=(0.0, 0.0))]
    analysis.wavelengths = [SimpleNamespace(value=0.55)]
    analysis.get_data = MagicMock(return_value=data)

    result = analysis._rms_wavefront_error()

    assert_allclose(result, be.sqrt(_typed_array([[5.0]])))


def test_legacy_tilt_helper_keeps_asymmetric_origin_and_ridge_behavior(
    set_test_backend,
):
    data = WavefrontData(
        pupil_x=_typed_array([0.0, 1.0, 0.0, 1.0]),
        pupil_y=_typed_array([0.0, 0.0, 1.0, 1.0]),
        pupil_z=_typed_array([0.0, 0.0, 0.0, 0.0]),
        opd=_typed_array([0.0, 0.0, 0.0, 1.0]),
        intensity=_typed_array([1.0, 2.0, 3.0, 4.0]),
        radius=1.0,
    )

    tilt_only = Wavefront.fit_and_remove_tilt(data, remove_piston=False, ridge=0.0)
    piston_tilt = Wavefront.fit_and_remove_tilt(data, remove_piston=True, ridge=0.0)
    ridged_tilt_only = Wavefront.fit_and_remove_tilt(
        data, remove_piston=False, ridge=1.0
    )
    ridged_piston_tilt = Wavefront.fit_and_remove_tilt(
        data, remove_piston=True, ridge=1.0
    )
    default = Wavefront.fit_and_remove_tilt(data)
    explicit_default = Wavefront.fit_and_remove_tilt(
        data, remove_piston=False, ridge=1e-12
    )

    assert_allclose(tilt_only, _typed_array([0.0, -0.72, -0.64, -0.36]))
    assert_allclose(piston_tilt, _typed_array([0.48, -0.24, -0.16, 0.12]))
    assert_allclose(
        ridged_tilt_only,
        _typed_array([0.0, -0.468965517241, -0.386206896552, 0.144827586207]),
    )
    assert_allclose(
        ridged_piston_tilt,
        _typed_array([0.137931034483, -0.331034482759, -0.248275862069, 0.28275862069]),
    )
    assert_array_equal(default, explicit_default)
    assert not be.array_equal(default, tilt_only)
