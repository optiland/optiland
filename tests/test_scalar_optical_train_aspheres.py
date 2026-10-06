"""Native even-asphere phase screens: independent references in millimeters."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.geometries.even_asphere import EvenAsphere
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.physical_optics.field import ScalarField
from optiland.physical_optics.train import ScalarOpticalTrain

from .utils import assert_allclose, assert_array_equal

COEFFICIENTS = (0.0004, -0.002, 0.0008)


def _array(values):
    if be.get_backend() == "torch":
        import torch

        return torch.as_tensor(values)
    return np.asarray(values)


def _field(precision="complex128", scale=1.0):
    ny, nx = 31, 47
    dx, dy, center = 0.009, 0.013, (0.023, -0.016)
    x = (np.arange(nx) - (nx - 1) / 2) * dx + center[0]
    y = (np.arange(ny) - (ny - 1) / 2) * dy + center[1]
    xx, yy = np.meshgrid(x, y)
    data = (
        (1 + 0.1 * xx)
        * np.exp(-(xx**2 + yy**2) / 0.12**2)
        * np.exp(1j * (8 * xx - 5 * yy))
    ).astype(precision)
    field = ScalarField(
        _array(data),
        dx * scale,
        0.00055 * scale,
        dy=dy * scale,
        center=(center[0] * scale, center[1] * scale),
    )
    return field, xx * scale, yy * scale, data


def _optic(
    radius=28.0,
    conic=-0.7,
    coefficients=COEFFICIENTS,
    aperture=None,
    surface_type="even_asphere",
    leading_plane=False,
):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=np.inf)
    index = 1
    if leading_plane:
        optic.surfaces.add(index=index, z=0, surface_type="plane")
        index += 1
    parameters = dict(
        index=index,
        z=1 if leading_plane else 0,
        radius=radius,
        conic=conic,
        surface_type=surface_type,
        material=IdealMaterial(1.5),
        aperture=aperture,
    )
    if surface_type == "even_asphere":
        parameters["coefficients"] = list(coefficients)
    optic.surfaces.add(**parameters)
    return optic


def _reference_sag(x, y, radius, conic, coefficients):
    """Independent curvature-form Cartesian reference, not native sag calls."""
    curvature = 0.0 if np.isinf(radius) else 1.0 / radius
    r2 = x * x + y * y
    sag = curvature * r2 / (1 + np.sqrt(1 - (1 + conic) * curvature**2 * r2))
    for power, coefficient in enumerate(coefficients, start=1):
        sag = sag + coefficient * r2**power
    return sag


@pytest.mark.parametrize(
    "radius,conic", [(28, -0.7), (-33, 0.35), (np.inf, -0.8), (-np.inf, 0.7)]
)
def test_asphere_independent_cartesian_phase_shifted_rectangular_grid(
    set_test_backend, radius, conic
):
    field, x, y, data = _field()
    optic = _optic(radius, conic)
    sag = _reference_sag(x, y, radius, conic, COEFFICIENTS)
    phase = 2 * np.pi / field.wavelength * (1 - 1.5) * sag
    expected = data * np.exp(1j * phase)
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert_allclose(output.data, expected, rtol=0, atol=3e-13)
    assert_allclose(output.power, field.power, rtol=3e-14, atol=0)
    assert output.center == field.center
    assert output.data.dtype == field.data.dtype
    assert output.refractive_index == 1.5


@pytest.mark.parametrize("radius", [-33.0, np.inf])
def test_asphere_coefficient_units_physical_scaling(set_test_backend, radius):
    scale = 2.5
    field, _, _, _ = _field()
    scaled_field, _, _, _ = _field(scale=scale)
    scaled_coefficients = [
        coefficient * scale ** (1 - 2 * power)
        for power, coefficient in enumerate(COEFFICIENTS, start=1)
    ]
    original = ScalarOpticalTrain.from_optic(_optic(radius)).propagate(field)
    scaled = ScalarOpticalTrain.from_optic(
        _optic(radius * scale, coefficients=scaled_coefficients)
    ).propagate(scaled_field)
    assert_allclose(scaled.data, original.data, rtol=0, atol=3e-13)
    assert_allclose(scaled.power, scale**2 * original.power, rtol=3e-14, atol=0)


@pytest.mark.parametrize(
    "radius,conic", [(28, -1), (-33, 0.35), (np.inf, -0.8), (-np.inf, 0.7)]
)
def test_zero_asphere_coefficients_equal_standalone_native_conic(
    set_test_backend, radius, conic
):
    field, _, _, _ = _field()
    asphere = _optic(radius, conic, coefficients=[0, 0, 0])
    conic_optic = _optic(radius, conic, surface_type="standard")
    asphere_output = ScalarOpticalTrain.from_optic(asphere).propagate(field)
    conic_output = ScalarOpticalTrain.from_optic(conic_optic).propagate(field)
    assert_array_equal(asphere_output.data, conic_output.data)


def test_asphere_native_sag_called_on_read_only_parameter_view(
    set_test_backend, monkeypatch
):
    optic = _optic()
    geometry = optic.surfaces[1].geometry
    geometry_state = dict(vars(geometry))
    coefficient_items = tuple(geometry.coefficients)
    field, x, y, data = _field()
    field_before = be.copy(field.data)
    calls = []
    native_sag = EvenAsphere.sag

    def record(self, x, y):
        calls.append(self)
        return native_sag(self, x, y)

    monkeypatch.setattr(EvenAsphere, "sag", record)
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert len(calls) == 1 and calls[0] is not geometry
    assert calls[0].coefficients is not geometry.coefficients
    assert vars(geometry).keys() == geometry_state.keys()
    for name, value in geometry_state.items():
        assert getattr(geometry, name) is value
    for before, after in zip(coefficient_items, geometry.coefficients, strict=True):
        assert before is after
    assert_array_equal(field.data, field_before)
    expected = data * np.exp(
        -1j * np.pi / field.wavelength * _reference_sag(x, y, 28, -0.7, COEFFICIENTS)
    )
    assert_allclose(output.data, expected, rtol=0, atol=3e-13)


@pytest.mark.parametrize("radius", [0.1, -0.1])
def test_asphere_aperture_protects_outside_conic_domain(set_test_backend, radius):
    field, x, y, data = _field()
    mask = x**2 + y**2 <= 0.04**2
    optic = _optic(radius, conic=0, aperture=RadialAperture(0.04))
    with np.errstate(invalid="raise"):
        output = ScalarOpticalTrain.from_optic(optic).propagate(field)
        sag = _reference_sag(
            np.where(mask, x, 0), np.where(mask, y, 0), radius, 0, COEFFICIENTS
        )
    expected = np.where(mask, data * np.exp(-1j * np.pi / field.wavelength * sag), 0)
    assert_allclose(output.data, expected, rtol=0, atol=3e-13)
    assert_allclose(
        output.power,
        np.sum(np.abs(data[mask]) ** 2) * field.dx * field.dy,
        rtol=3e-14,
        atol=0,
    )
    assert bool(be.all(be.isfinite(output.data)))


@pytest.mark.parametrize(
    "kind",
    [
        "nan_coefficient",
        "inf_coefficient",
        "complex_coefficient",
        "vector_coefficient",
        "zero_radius",
        "nan_radius",
        "inf_conic",
        "domain",
    ],
)
def test_later_asphere_invalidity_rejected_before_any_asm(
    set_test_backend, monkeypatch, kind
):
    optic = _optic(leading_plane=True)
    train = ScalarOpticalTrain.from_optic(optic)
    geometry = optic.surfaces[2].geometry
    if kind == "nan_coefficient":
        geometry.coefficients[0] = np.nan
    elif kind == "inf_coefficient":
        geometry.coefficients[0] = np.inf
    elif kind == "complex_coefficient":
        geometry.coefficients[0] = 0.0004 + 0.1j
    elif kind == "vector_coefficient":
        geometry.coefficients[0] = be.array([0.0004, 0.0008])
    elif kind == "zero_radius":
        geometry.radius = be.array(0)
    elif kind == "nan_radius":
        geometry.radius = be.array(np.nan)
    elif kind == "inf_conic":
        geometry.k = be.array(np.inf)
    else:
        geometry.radius = be.array(0.01)
        geometry.k = be.array(0)

    def forbidden(*args, **kwargs):
        pytest.fail("ASM ran before the later asphere was preflighted")

    monkeypatch.setattr(ScalarField, "propagate", forbidden)
    with np.errstate(invalid="ignore"), pytest.raises(ValueError):
        train.propagate(_field()[0])


@pytest.mark.parametrize("kind", ["none", "nested_list", "matrix"])
def test_asphere_invalid_coefficient_container_rejected(set_test_backend, kind):
    optic = _optic()
    geometry = optic.surfaces[1].geometry
    if kind == "none":
        geometry.coefficients = None
    elif kind == "nested_list":
        geometry.coefficients = [[0.0004, 0.0008]]
    else:
        geometry.coefficients = be.array([[0.0004, 0.0008]])
    with pytest.raises(ValueError, match="scalar"):
        ScalarOpticalTrain.from_optic(optic)


@pytest.mark.parametrize("parameter", ["radius", "conic", "coefficient"])
def test_asphere_parameter_overflow_in_field_precision_rejected(
    set_test_backend, parameter
):
    optic = _optic()
    geometry = optic.surfaces[1].geometry
    if parameter == "radius":
        geometry.radius = be.array(1e200)
    elif parameter == "conic":
        geometry.k = be.array(1e200)
    else:
        geometry.coefficients[0] = be.array(1e200)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="field-precision"):
        ScalarOpticalTrain.from_optic(optic).propagate(_field(precision="complex64")[0])


class _CustomTrainAsphere(EvenAsphere):
    pass


def test_custom_asphere_subclass_remains_unsupported(set_test_backend):
    optic = _optic()
    optic.surfaces[1].geometry = _CustomTrainAsphere(
        optic.surfaces[1].geometry.cs, 28, -0.7, coefficients=COEFFICIENTS
    )
    with pytest.raises(ValueError, match="unsupported geometry"):
        ScalarOpticalTrain.from_optic(optic)


@pytest.mark.parametrize("radius", [28.0, np.inf])
@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_torch_asphere_nonleaf_coefficient_gradients_and_mixed_precision(
    set_test_backend, radius, precision
):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific gradient test")
    import torch

    # The fixture configures float64; source scalar parameters stay float64
    # even when the sampled field is complex64.
    assert be.get_precision() == 64
    field, _, _, _ = _field(precision=precision)
    field.data.requires_grad_()
    primitive = torch.tensor(
        [0.0002, 0.004, 0.0004], dtype=torch.float64, requires_grad=True
    )
    scales = torch.tensor([2.0, -0.5, 1.0], dtype=torch.float64)

    def coefficients():
        return [primitive[0] * 2, primitive[1] * -0.5, primitive[2] + 0.0004]

    optic = _optic(radius)
    geometry = optic.surfaces[1].geometry
    geometry.coefficients = coefficients()
    source_coefficients = geometry.coefficients
    source_radius, source_conic = geometry.radius, geometry.k
    assert all(not coefficient.is_leaf for coefficient in source_coefficients)
    train = ScalarOpticalTrain.from_optic(optic)
    output = train.propagate(field)
    output.data.real.sum().backward()
    assert output.data.dtype == field.data.dtype
    assert output.data.device == field.data.device
    assert output.center == field.center
    assert geometry.coefficients is source_coefficients
    assert geometry.radius is source_radius and geometry.k is source_conic
    assert primitive.grad is not None and torch.isfinite(primitive.grad).all()
    assert field.data.grad is not None and torch.isfinite(field.data.grad).all()
    x, y = field.coordinates()
    xx, yy = be.meshgrid(x, y)
    r2 = xx**2 + yy**2
    alpha = 2 * np.pi / field.wavelength * (1 - 1.5)
    expected = (
        torch.stack(
            [
                -alpha * torch.sum(output.data.detach().imag * r2**power)
                for power in (1, 2, 3)
            ]
        ).double()
        * scales
    )
    tolerance = 4e-6 if precision == "complex64" else 3e-12
    assert_allclose(
        primitive.grad,
        expected,
        rtol=tolerance,
        atol=2e-5 if precision == "complex64" else 1e-9,
    )
    if np.isinf(radius):
        # Infinite radius/conic describe a constant flat base, not an infinity
        # arithmetic graph. Coefficient gradients must not contaminate them.
        assert source_radius.grad is None and source_conic.grad is None
    else:
        assert source_radius.grad is not None and torch.isfinite(source_radius.grad)
        assert source_conic.grad is not None and torch.isfinite(source_conic.grad)

    if precision == "complex128":
        original = primitive.detach().clone()
        direction = torch.tensor([1.0, 0.5, -0.25], dtype=torch.float64)
        derivative = torch.sum(primitive.grad * direction)
        for step in (1e-6, 1e-7, 1e-8):
            with torch.no_grad():
                primitive.copy_(original + step * direction)
                geometry.coefficients = coefficients()
                plus = train.propagate(field).data.real.sum().item()
                primitive.copy_(original - step * direction)
                geometry.coefficients = coefficients()
                minus = train.propagate(field).data.real.sum().item()
                primitive.copy_(original)
                geometry.coefficients = source_coefficients
            assert_allclose(
                derivative, (plus - minus) / (2 * step), rtol=5e-7, atol=1e-6
            )
