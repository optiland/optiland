"""Independent axial Beer--Lambert references; lengths and wavelengths in mm."""

from __future__ import annotations

from copy import copy, deepcopy

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import AbbeMaterial, AbbeMaterialE, BaseMaterial, IdealMaterial
from optiland.optic import Optic
from optiland.physical_optics.field import ScalarField
from optiland.physical_optics.train import ScalarOpticalTrain
from optiland.samples.objectives import CookeTriplet
from optiland.surfaces.image_surface import ImageSurface

from .utils import assert_allclose, assert_array_equal


def _array(values):
    if be.get_backend() == "torch":
        import torch

        return torch.as_tensor(values)
    return np.asarray(values)


def _plane(z=0, material="air", **kwargs):
    return dict(z=z, surface_type="plane", material=material, **kwargs)


def _optic(*surfaces, incident=None):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=np.inf, material=incident or "air")
    for index, parameters in enumerate(surfaces, start=1):
        optic.surfaces.add(index=index, **parameters)
    return optic


def _field(*, wavelength=0.00055, n=1, precision="complex128", power=2.3):
    # Explicit SI power calibration: data is sqrt(W/mm^2), not V/m.
    shape, dx, dy = (24, 30), 0.01, 0.012
    amplitude = np.sqrt(power / (np.prod(shape) * dx * dy))
    return ScalarField(
        _array(np.full(shape, amplitude + 0j, dtype=precision)),
        dx,
        wavelength,
        dy=dy,
        refractive_index=n,
        center=(0.025, -0.01),
    )


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize(
    "layers",
    [
        ((1.6, 0.017, 0.00317),),
        ((1.6, 0.017, 0.00317), (1.2, 0.003, 0.00129), (1.8, 0.023, 0.00211)),
    ],
)
def test_planar_slab_complex_amplitude_phase_and_calibrated_power(
    set_test_backend, precision, layers
):
    z = 0.0
    surfaces = []
    for n, kappa, distance in layers:
        surfaces.append(_plane(z=z, material=IdealMaterial(n, kappa)))
        z += distance
    surfaces.append(_plane(z=z))
    field = _field(precision=precision)
    original = be.copy(field.data)
    train = ScalarOpticalTrain.from_optic(_optic(*surfaces), absorption="axial")
    output = train.propagate(field)
    optical_length = sum(n * distance for n, _, distance in layers)
    extinction_length = sum(kappa * distance for _, kappa, distance in layers)
    amplitude = np.exp(-2 * np.pi * extinction_length / field.wavelength)
    expected = (
        np.asarray(be.to_numpy(field.data))
        * amplitude
        * np.exp(2j * np.pi * optical_length / field.wavelength)
    )
    tolerance = 2e-6 if precision == "complex64" else 5e-13
    assert_allclose(output.data, expected, rtol=tolerance, atol=tolerance)
    assert_allclose(field.power, 2.3, rtol=tolerance, atol=0)
    assert_allclose(output.power, 2.3 * amplitude**2, rtol=tolerance, atol=0)
    assert output.refractive_index == 1  # No complex medium metadata.
    assert output.data.dtype == field.data.dtype
    assert output.center == field.center
    assert (output.dx, output.dy, output.shape) == (field.dx, field.dy, field.shape)
    assert_array_equal(field.data, original)


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_shifted_rectangular_fourier_mode_uses_real_index_and_axial_loss(
    set_test_backend, precision
):
    ny, nx, mx, my = 24, 30, 4, -3
    dx, dy, wavelength, n, kappa, distance = 0.01, 0.012, 0.00055, 1.6, 0.017, 0.00317
    mode = np.exp(
        2j
        * np.pi
        * (mx * np.arange(nx)[None, :] / nx + my * np.arange(ny)[:, None] / ny)
    )
    field = ScalarField(
        _array(mode.astype(precision)), dx, wavelength, dy=dy, center=(0.035, -0.021)
    )
    optic = _optic(_plane(material=IdealMaterial(n, kappa)), _plane(z=distance))
    kz = np.sqrt(
        (2 * np.pi * n / wavelength) ** 2
        - (2 * np.pi * mx / (nx * dx)) ** 2
        - (2 * np.pi * my / (ny * dy)) ** 2
    )
    expected = mode * np.exp(
        1j * kz * distance - 2 * np.pi * kappa * distance / wavelength
    )
    output = ScalarOpticalTrain.from_optic(optic, absorption="axial").propagate(field)
    tolerance = 2e-6 if precision == "complex64" else 5e-13
    assert_allclose(output.data, expected, rtol=tolerance, atol=tolerance)
    for actual, original in zip(output.coordinates(), field.coordinates(), strict=True):
        assert_array_equal(actual, original)


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize("gap_kind", ["python", "backend_scalar"])
@pytest.mark.parametrize("default_precision", ["float64", "float32"])
def test_attenuation_preserves_field_dtype_with_python_or_backend_scalar_gap(
    set_test_backend, precision, gap_kind, default_precision
):
    original_precision = be.get_precision()
    try:
        be.set_precision(default_precision)
        material = IdealMaterial(1.6, 0.017)
        # Keep independent n/k metadata precise; only vertex creation uses the
        # selected default precision. The field may have a different precision.
        material.index = _array(np.array([1.6], dtype=np.float64))
        material.absorp = _array(np.array([0.017], dtype=np.float64))
        optic = _optic(_plane(material=material), _plane(z=0.00317))
        optic.surfaces[1].geometry.cs.z = 0.0
        gap = 0.00317 if gap_kind == "python" else be.array(0.00317)
        optic.surfaces[2].geometry.cs.z = gap
        # CoordinateSystem setters convert Python values to backend precision.
        # Compare the actual stored distance, not an unrounded authoring value.
        source_distance = be.to_numpy(
            optic.surfaces[2].geometry.cs.z - optic.surfaces[1].geometry.cs.z
        ).item()
        field = _field(precision=precision)
        output = ScalarOpticalTrain.from_optic(optic, absorption="axial").propagate(
            field
        )
        expected = field.data * np.exp(
            2 * np.pi * (-0.017 + 1.6j) * source_distance / field.wavelength
        )
        tolerance = 2e-6 if precision == "complex64" else 5e-13
        assert output.data.dtype == field.data.dtype
        if be.get_backend() == "torch":
            assert output.data.device == field.data.device
        assert_allclose(output.data, expected, rtol=tolerance, atol=tolerance)
        assert_allclose(
            output.power / field.power,
            np.exp(-4 * np.pi * 0.017 * source_distance / field.wavelength),
            rtol=tolerance,
            atol=0,
        )
    finally:
        be.set_precision("float64" if original_precision == 64 else "float32")


@pytest.mark.parametrize("changes_extinction", [False, True])
def test_axial_image_marker_cannot_change_extinction(
    set_test_backend, changes_extinction
):
    material = IdealMaterial(1.6, 0.017)
    optic = _optic(_plane(material=material), _plane(z=0.00317, material=material))
    image = ImageSurface(
        previous_surface=optic.surfaces[1],
        geometry=optic.surfaces[2].geometry,
        material_post=IdealMaterial(1.6, 0.018) if changes_extinction else material,
    )
    train = ScalarOpticalTrain((image,), start_surface=2, absorption="axial")
    field = _field(n=1.6)
    if changes_extinction:
        with pytest.raises(ValueError, match="ImageSurface cannot change"):
            train.propagate(field)
    else:
        assert_array_equal(train.propagate(field).data, field.data)


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_zero_extinction_is_bitwise_equal_to_strict_default(
    set_test_backend, precision
):
    optic = _optic(
        _plane(material=IdealMaterial(1.6)), _plane(z=0.00317), _plane(z=0.00423)
    )
    rng = np.random.default_rng(2037)
    field = ScalarField(
        _array(
            (rng.normal(size=(24, 30)) + 1j * rng.normal(size=(24, 30))).astype(
                precision
            )
        ),
        0.01,
        0.00055,
        dy=0.012,
        center=(0.025, -0.01),
    )
    default = ScalarOpticalTrain.from_optic(optic).propagate(field)
    explicit = ScalarOpticalTrain.from_optic(optic, absorption="reject").propagate(
        field
    )
    axial = ScalarOpticalTrain.from_optic(optic, absorption="axial").propagate(field)
    assert_array_equal(axial.data, default.data)
    assert_array_equal(explicit.data, default.data)
    assert_array_equal(axial.power, default.power)


@pytest.mark.parametrize("kappa", [np.nextafter(0.0, 1.0), 4.379e-9, 0.017])
@pytest.mark.parametrize("explicit", [False, True])
def test_default_rejects_every_nonzero_extinction(set_test_backend, kappa, explicit):
    optic = _optic(_plane(material=IdealMaterial(1.6, kappa)))
    kwargs = {"absorption": "reject"} if explicit else {}
    with pytest.raises(ValueError, match="lossless"):
        ScalarOpticalTrain.from_optic(optic, **kwargs).propagate(_field())


@pytest.mark.parametrize("mode", ["ignore", "AXIAL", "", None, True, []])
@pytest.mark.parametrize("factory", ["from_optic", "direct"])
def test_invalid_mode_rejected(set_test_backend, mode, factory):
    optic = _optic(_plane())
    with pytest.raises(ValueError, match="absorption"):
        if factory == "from_optic":
            ScalarOpticalTrain.from_optic(optic, absorption=mode)
        else:
            ScalarOpticalTrain(tuple(optic.surfaces)[1:], absorption=mode)


@pytest.mark.parametrize(
    "n,kappa",
    [
        (1.6, -0.001),
        (1.6, -np.nextafter(0.0, 1.0)),
        (1.6, np.nan),
        (1.6, np.inf),
        (1.6, 0.01 + 0.02j),
        (0, 0.01),
        (-1, 0.01),
        (np.nan, 0.01),
        (np.inf, 0.01),
        (1.5 + 0.02j, 0.01),
    ],
)
def test_axial_requires_finite_positive_n_and_passive_real_kappa(
    set_test_backend, n, kappa
):
    material = IdealMaterial(1.6)
    material.index = _array(np.atleast_1d(n))
    material.absorp = _array(np.atleast_1d(kappa))
    optic = _optic(_plane(material=material))
    with pytest.raises(ValueError, match="finite|positive|passive|real scalar"):
        ScalarOpticalTrain.from_optic(optic, absorption="axial").propagate(_field())


def test_zero_gaps_input_plane_and_trailing_medium_are_not_absorbing_paths(
    set_test_backend,
):
    optic = _optic(
        _plane(z=10, material=IdealMaterial(1.4, 0.21)),
        _plane(z=10, material=IdealMaterial(1.6, 0.017)),
        _plane(z=10.00317, material=IdealMaterial(1.2, 0.9)),
        incident=IdealMaterial(1.1, 0.31),
    )
    optic.surfaces[3].thickness = 10000
    field = _field(n=1.1)
    train = ScalarOpticalTrain(tuple(optic.surfaces)[1:], absorption="axial")
    output = train.propagate(field)
    gap = 10.00317 - 10  # The stored vertices, not authoring thickness.
    expected = field.data * np.exp(
        2j * np.pi * 1.6 * gap / field.wavelength
        - 2 * np.pi * 0.017 * gap / field.wavelength
    )
    assert_allclose(output.data, expected, rtol=5e-13, atol=5e-13)
    assert output.refractive_index == 1.2
    single = ScalarOpticalTrain.from_optic(optic, end_surface=1, absorption="axial")
    assert_array_equal(single.propagate(field).data, field.data)


class _DispersiveAbsorber(BaseMaterial):
    def _calculate_n(self, wavelength, **kwargs):
        return 1 + 0.2 * wavelength

    def _calculate_k(self, wavelength, **kwargs):
        return 0.02 * wavelength

    def spectral_range(self, property_name="n"):
        return (0.4, 0.7)


@pytest.mark.parametrize("wavelength_um", [0.45, 0.55, 0.65])
def test_kappa_lookup_uses_microns_but_attenuation_uses_vacuum_mm(
    set_test_backend, wavelength_um
):
    wavelength = wavelength_um / 1000
    distance, n, kappa = 0.00317, 1 + 0.2 * wavelength_um, 0.02 * wavelength_um
    optic = _optic(_plane(material=_DispersiveAbsorber()), _plane(z=distance))
    field = _field(wavelength=wavelength)
    output = ScalarOpticalTrain.from_optic(optic, absorption="axial").propagate(field)
    expected = field.data * np.exp(
        2j * np.pi * n * distance / wavelength
        - 2 * np.pi * kappa * distance / wavelength
    )
    assert_allclose(output.data, expected, rtol=5e-13, atol=5e-13)
    assert_allclose(
        output.power / field.power,
        np.exp(-4 * np.pi * kappa * distance / wavelength),
        rtol=5e-13,
        atol=0,
    )
    with pytest.raises(ValueError, match="wavelength"):
        ScalarOpticalTrain.from_optic(optic, absorption="axial").propagate(
            _field(wavelength=wavelength_um)
        )


def test_axial_does_not_relax_incident_medium_or_negative_gap_checks(set_test_backend):
    optic = _optic(_plane(material=IdealMaterial(1.6, 0.017)), _plane(z=0.00317))
    with pytest.raises(ValueError, match="incident material"):
        ScalarOpticalTrain.from_optic(optic, absorption="axial").propagate(
            _field(n=1.2)
        )
    optic.surfaces[2].geometry.cs.z = -0.00317
    with pytest.raises(ValueError, match="negative vertex"):
        ScalarOpticalTrain.from_optic(optic, absorption="axial")


@pytest.mark.parametrize("bad_kappa", [-0.01, np.nan, np.inf])
def test_later_invalid_extinction_is_atomic_after_live_edits(
    set_test_backend, monkeypatch, bad_kappa
):
    material = IdealMaterial(1.6, 0.017)
    optic = _optic(
        _plane(material=material),
        _plane(z=0.00317),
        _plane(z=0.005, material=IdealMaterial(1.4)),
    )
    train = ScalarOpticalTrain.from_optic(optic, absorption="axial")
    field = _field()
    original = be.copy(field.data)
    material_before = (material._n_cache.copy(), material._k_cache.copy())
    optic.surfaces[3].material_post.absorp = be.array([bad_kappa])

    def forbidden(*args, **kwargs):
        pytest.fail("ASM ran before all extinction coefficients were validated")

    monkeypatch.setattr(ScalarField, "propagate", forbidden)
    with pytest.raises(ValueError, match="finite|passive"):
        train.propagate(field)
    assert_array_equal(field.data, original)
    assert material._n_cache == material_before[0]
    assert material._k_cache == material_before[1]


@pytest.mark.parametrize("value", [complex(np.nan, 0), complex(0, np.inf)])
def test_nonfinite_field_rejected_before_axial_propagation(
    set_test_backend, monkeypatch, value
):
    optic = _optic(_plane(material=IdealMaterial(1.6, 0.017)), _plane(z=0.00317))
    field = _field()
    field.data[2, 3] = value

    def forbidden(*args, **kwargs):
        pytest.fail("ASM ran before finite input validation")

    monkeypatch.setattr(ScalarField, "propagate", forbidden)
    with pytest.raises(ValueError, match="field data must be finite"):
        ScalarOpticalTrain.from_optic(optic, absorption="axial").propagate(field)


@pytest.mark.parametrize("kind", ["polynomial", "buchdahl", "e_line"])
def test_axial_leaves_abbe_models_and_source_caches_unchanged(set_test_backend, kind):
    material = (
        AbbeMaterialE(1.5, 60)
        if kind == "e_line"
        else AbbeMaterial(1.5, 60, model=kind)
    )
    material.n(0.6)
    material.k(0.6)
    model = material.model
    coefficients = {
        name: getattr(model, name)
        for name in ("_p", "v1", "v2", "v3")
        if hasattr(model, name)
    }
    caches = (
        material._n_cache.copy(),
        material._k_cache.copy(),
        material._cache_context,
    )
    optic = _optic(_plane(material=material), _plane(z=0.00317))
    ScalarOpticalTrain.from_optic(optic, absorption="axial").propagate(_field())
    assert material.model is model
    for name, original in coefficients.items():
        assert getattr(model, name) is original
    assert material._cache_context == caches[2]
    for cache, original in zip(
        (material._n_cache, material._k_cache), caches[:2], strict=True
    ):
        assert cache.keys() == original.keys()
        for key in original:
            assert cache[key] is original[key]


@pytest.mark.parametrize("distance", [0.0, 0.00317])
@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_torch_gap_gradient_includes_loss_and_forward_derivative_at_zero(
    set_test_backend, distance, precision
):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific gradient test")
    import torch

    n, kappa = 1.6, 0.017
    optic = _optic(_plane(material=IdealMaterial(n, kappa)), _plane(z=distance))
    vertex = torch.tensor(distance, dtype=torch.float64, requires_grad=True)
    optic.surfaces[2].geometry.cs.z = vertex
    field = _field(precision=precision)
    field.data.requires_grad_(True)
    train = ScalarOpticalTrain.from_optic(optic, absorption="axial")
    output = train.propagate(field)
    output.data.real.mean().backward(retain_graph=True)
    q = 2 * np.pi * (-kappa + 1j * n) / field.wavelength
    input_amplitude = float(field.data.real[0, 0].detach())
    expected = input_amplitude * (q * np.exp(q * distance)).real
    tolerance = 3e-6 if precision == "complex64" else 2e-12
    assert_allclose(vertex.grad, expected, rtol=tolerance, atol=1e-8)
    derivative = vertex.grad.detach().clone()
    assert torch.isfinite(field.data.grad).all()
    assert output.data.dtype == field.data.dtype
    assert output.data.device == field.data.device
    vertex.grad = None
    output.power.backward()
    expected_power_derivative = (
        -4 * np.pi * kappa / field.wavelength * float(output.power.detach())
    )
    assert_allclose(vertex.grad, expected_power_derivative, rtol=tolerance, atol=1e-8)
    if precision == "complex128":
        for step in (1e-7, 1e-8, 1e-9):
            with torch.no_grad():
                vertex.fill_(distance + step)
                plus = train.propagate(field).data.real.mean().item()
                vertex.fill_(distance)
                center = train.propagate(field).data.real.mean().item()
                if distance > 0:
                    vertex.fill_(distance - step)
                    minus = train.propagate(field).data.real.mean().item()
                    finite_difference = (plus - minus) / (2 * step)
                    rtol = 3e-6
                else:
                    # A forward three-point difference honors the nonnegative
                    # gap domain.
                    vertex.fill_(2 * step)
                    twice = train.propagate(field).data.real.mean().item()
                    finite_difference = (-3 * center + 4 * plus - twice) / (2 * step)
                    rtol = 2e-5
                vertex.fill_(distance)
            assert_allclose(derivative, finite_difference, rtol=rtol, atol=1e-6)


def test_material_n_and_kappa_remain_scalar_metadata_without_gradients(
    set_test_backend,
):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific gradient test")
    import torch

    material = IdealMaterial(1.6, 0.017)
    n = torch.tensor([1.6], dtype=torch.float64, requires_grad=True)
    kappa = torch.tensor([0.017], dtype=torch.float64, requires_grad=True)
    material.index, material.absorp = n, kappa
    optic = _optic(_plane(material=material), _plane(z=0.00317))
    field = _field()
    field.data.requires_grad_(True)
    output = ScalarOpticalTrain.from_optic(optic, absorption="axial").propagate(field)
    output.power.backward()
    assert n.grad is None and kappa.grad is None
    assert field.data.grad is not None and torch.isfinite(field.data.grad).all()
    assert material.index is n and material.absorp is kappa


def _material_properties(material, wavelength_um):
    # Independent read-only catalog evaluation; do not use the train's helper.
    view = copy(material)
    view._n_cache, view._k_cache, view._cache_context = {}, {}, None
    return be.to_numpy(view.n(wavelength_um)).item(), be.to_numpy(
        view.k(wavelength_um)
    ).item()


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_catalog_cooke_triplet_axial_power_and_lossless_prescription_reference(
    set_test_backend, precision
):
    optic = CookeTriplet()
    wavelength = 0.00055
    assert all(surface.aperture is None for surface in optic.surfaces)
    materials = [surface.material_post for surface in optic.surfaces]
    for material in materials:
        material.n(0.6)
        material.k(0.6)
    source = deepcopy(optic.to_dict())
    caches = [
        (material._n_cache.copy(), material._k_cache.copy(), material._cache_context)
        for material in materials
    ]
    properties = [_material_properties(material, 0.55) for material in materials]
    assert properties[1][1] > 0 and properties[3][1] > 0  # Actual SK16 and F2.
    with pytest.raises(ValueError, match="lossless"):
        ScalarOpticalTrain.from_optic(optic).propagate(_field())
    ny, nx, dx, dy = 64, 80, 0.008, 0.009
    xx, yy = np.meshgrid(
        (np.arange(nx) - (nx - 1) / 2) * dx + 0.015,
        (np.arange(ny) - (ny - 1) / 2) * dy - 0.01,
    )
    data = np.exp(-(xx**2 + yy**2) / 0.07**2 + 13j * yy)
    data *= np.sqrt(1.7 / (np.sum(np.abs(data) ** 2) * dx * dy))
    field = ScalarField(
        _array(data.astype(precision)),
        dx,
        wavelength,
        dy=dy,
        center=(0.015, -0.01),
        refractive_index=1,
    )
    original_data = be.copy(field.data)
    output = ScalarOpticalTrain.from_optic(optic, absorption="axial").propagate(field)
    # Clone only the actual prescription, replacing glass by fixed real n at
    # this wavelength. Geometry, gaps, screens and field remain identical.
    parameters = []
    vertices = [float(surface.geometry.cs.z) for surface in optic.surfaces[1:]]
    for surface, (n, _) in zip(optic.surfaces[1:], properties[1:], strict=True):
        geometry = surface.geometry
        entry = dict(z=float(geometry.cs.z), material=IdealMaterial(n))
        if hasattr(geometry, "k"):
            entry.update(radius=float(geometry.radius), conic=float(geometry.k))
        else:
            entry.update(surface_type="plane")
        parameters.append(entry)
    lossless_optic = _optic(*parameters, incident=IdealMaterial(properties[0][0]))
    lossless = ScalarOpticalTrain.from_optic(lossless_optic).propagate(field)
    extinction_length = sum(
        properties[index + 1][1] * (z2 - z1)
        for index, (z1, z2) in enumerate(zip(vertices[:-1], vertices[1:], strict=True))
    )
    expected_amplitude = np.exp(-2 * np.pi * extinction_length / wavelength)
    tolerance = 4e-6 if precision == "complex64" else 3e-12
    assert bool(be.all(be.isfinite(output.data)))
    assert_allclose(field.power, 1.7, rtol=tolerance, atol=0)
    assert_allclose(lossless.power, field.power, rtol=tolerance, atol=0)
    assert_allclose(
        output.power / field.power, expected_amplitude**2, rtol=tolerance, atol=0
    )
    assert_allclose(
        output.data,
        lossless.data * expected_amplitude,
        rtol=tolerance,
        atol=tolerance * np.max(np.abs(be.to_numpy(lossless.data))),
    )
    assert_array_equal(field.data, original_data)
    assert optic.to_dict() == source
    for material, (n_cache, k_cache, context) in zip(materials, caches, strict=True):
        assert material._cache_context == context
        for cache, original in (
            (material._n_cache, n_cache),
            (material._k_cache, k_cache),
        ):
            assert cache.keys() == original.keys()
            for key in original:
                assert cache[key] is original[key]
