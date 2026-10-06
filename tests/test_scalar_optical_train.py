"""Independent scalar phase-screen and propagation references, in millimeters."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.odd_asphere import OddAsphere
from optiland.materials import AbbeMaterial, AbbeMaterialE, BaseMaterial, IdealMaterial
from optiland.optic import Optic
from optiland.phase.constant import ConstantPhaseProfile
from optiland.phase.grid import GridPhaseProfile
from optiland.phase.height_profile import HeightProfile
from optiland.phase.radial import RadialPhaseProfile
from optiland.physical_apertures import (
    EllipticalAperture,
    RadialAperture,
    RectangularAperture,
)
from optiland.physical_optics.field import ScalarField, gaussian_field
from optiland.physical_optics.train import ScalarOpticalTrain
from optiland.surfaces.image_surface import ImageSurface

from .utils import assert_allclose, assert_array_equal


def _array(values):
    if be.get_backend() == "torch":
        import torch

        return torch.as_tensor(values)
    return np.asarray(values)


def _optic(*surfaces, incident=None):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=np.inf, material=incident or "air")
    for index, parameters in enumerate(surfaces, start=1):
        optic.surfaces.add(index=index, **parameters)
    return optic


def _plane(z=0, material="air", **kwargs):
    return dict(z=z, surface_type="plane", material=material, **kwargs)


def _uniform(shape=(32, 40), dx=0.01, dy=0.012, wavelength=0.0005, n=1):
    return ScalarField(
        _array(np.ones(shape, dtype=np.complex128)),
        dx=dx,
        dy=dy,
        wavelength=wavelength,
        refractive_index=n,
    )


def test_planar_media_phase_and_power(set_test_backend):
    """Each positive gap uses its outgoing medium, not vacuum or input n."""
    optic = _optic(
        _plane(z=0, material=IdealMaterial(1.5)),
        _plane(z=0.00313, material=IdealMaterial(1.2)),
        _plane(z=0.0054, material="air"),
    )
    field = _uniform()
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    # Nonintegral optical cycles ensure a missing/wrong medium is detectable.
    expected = np.exp(
        1j * (2 * np.pi / field.wavelength) * (1.5 * 0.00313 + 1.2 * 0.00227)
    )
    assert_allclose(output.data, expected, rtol=0, atol=3e-13)
    assert_allclose(output.power, field.power, rtol=2e-14, atol=0)
    assert output.refractive_index == 1
    assert (output.dx, output.dy, output.shape) == (field.dx, field.dy, field.shape)


def test_nonconstant_fourier_mode_exact_medium_dispersion(set_test_backend):
    """An exact DFT mode independently tests kz rather than piston alone."""
    ny, nx, mx, my = 32, 40, 5, -3
    dx, dy, wavelength = 0.01, 0.012, 0.005
    grid_phase = (
        2
        * np.pi
        * (mx * np.arange(nx)[None, :] / nx + my * np.arange(ny)[:, None] / ny)
    )
    data = np.exp(1j * grid_phase)
    field = ScalarField(_array(data), dx, wavelength, dy=dy)
    d1, d2, n1, n2 = 0.1321, 0.0913, 1.5, 1.2
    optic = _optic(
        _plane(material=IdealMaterial(n1)),
        _plane(z=d1, material=IdealMaterial(n2)),
        _plane(z=d1 + d2),
    )
    transverse_k2 = (2 * np.pi * mx / (nx * dx)) ** 2 + (
        2 * np.pi * my / (ny * dy)
    ) ** 2
    kz1 = np.sqrt((2 * np.pi * n1 / wavelength) ** 2 - transverse_k2)
    kz2 = np.sqrt((2 * np.pi * n2 / wavelength) ** 2 - transverse_k2)
    expected = data * np.exp(1j * (d1 * kz1 + d2 * kz2))
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert_allclose(output.data, expected, rtol=0, atol=3e-13)
    assert_allclose(output.power, field.power, rtol=3e-14, atol=0)


def test_thick_glass_quadratic_interfaces_independent_reduced_abcd(set_test_backend):
    """Finite glass gap is d/n in the reduced (height, n*angle) ABCD frame."""
    wavelength, waist, glass, thickness = 0.0005, 0.2, 1.5, 4.0
    r1, r2 = 50.0, -45.0
    entry = np.array([[1, 0], [-(glass - 1) / r1, 1]])
    propagation = np.array([[1, thickness / glass], [0, 1]])
    exit_surface = np.array([[1, 0], [-(1 - glass) / r2, 1]])
    a, b, c, d = (exit_surface @ propagation @ entry).ravel()
    reduced_q = 1j * np.pi * waist**2 / wavelength
    reduced_q = (a * reduced_q + b) / (c * reduced_q + d)
    focus = -reduced_q.real
    reference_waist = np.sqrt(wavelength * reduced_q.imag / np.pi)
    dx, size = 0.004, 512
    assert focus > 0 and reference_waist / dx > 8
    assert size * dx / 2 > 5 * waist
    optic = _optic(
        dict(z=0, radius=r1, conic=-1, material=IdealMaterial(glass)),
        dict(z=thickness, radius=r2, conic=-1, material="air"),
        _plane(z=thickness + focus),
    )
    field = gaussian_field((size, size), dx, wavelength, waist)
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    x, y = output.coordinates()
    xx, yy = be.meshgrid(x, y)
    measured = be.sqrt(
        2 * be.sum((xx**2 + yy**2) * output.intensity) / be.sum(output.intensity)
    )
    expected_intensity = (waist / reference_waist) ** 2 * be.exp(
        -2 * (xx**2 + yy**2) / reference_waist**2
    )
    relative_l2 = be.sqrt(
        be.sum((output.intensity - expected_intensity) ** 2)
        / be.sum(expected_intensity**2)
    )
    assert_allclose(measured, reference_waist, rtol=2e-4, atol=0)
    assert float(be.to_numpy(relative_l2)) < 2e-4
    assert_allclose(output.power, field.power, rtol=3e-13, atol=0)


@pytest.mark.parametrize("kind", ["polynomial", "buchdahl", "e_line"])
def test_native_abbe_model_state_is_not_replaced(set_test_backend, kind):
    material = (
        AbbeMaterialE(1.5, 60)
        if kind == "e_line"
        else AbbeMaterial(1.5, 60, model=kind)
    )
    model = material.model
    stored = {
        name: getattr(model, name)
        for name in ("_p", "v1", "v2", "v3")
        if hasattr(model, name)
    }
    source_caches = (material._n_cache.copy(), material._k_cache.copy())
    optic = _optic(_plane(material=material))
    ScalarOpticalTrain.from_optic(optic).propagate(_uniform())
    assert material.model is model
    for name, value in stored.items():
        assert getattr(model, name) is value
    assert material._n_cache == source_caches[0]
    assert material._k_cache == source_caches[1]


def test_inclusive_bounds_object_and_no_trailing_thickness(set_test_backend):
    optic = _optic(
        _plane(z=2, material=IdealMaterial(1.5)),
        _plane(z=5, material=IdealMaterial(1.2)),
        _plane(z=9),
    )
    # Vertex coordinates, not these deliberately inconsistent authoring values,
    # define distances. The infinite object distance is never propagated.
    optic.surfaces[1].thickness = 500
    optic.surfaces[2].thickness = 1000
    train = ScalarOpticalTrain.from_optic(optic, start_surface=2, end_surface=2)
    field = _uniform(n=1.5)
    output = train.propagate(field)
    assert_array_equal(output.data, field.data)
    assert output.refractive_index == 1.2
    assert (train.start_surface, train.end_surface) == (2, 2)


@pytest.mark.parametrize("changes_medium", [False, True])
def test_native_image_surface_is_only_a_planar_medium_preserving_marker(
    set_test_backend, changes_medium
):
    optic = _optic(_plane(material=IdealMaterial(1.5)), _plane(z=1))
    material = IdealMaterial(1.2) if changes_medium else optic.surfaces[1].material_post
    image = ImageSurface(
        previous_surface=optic.surfaces[1],
        geometry=optic.surfaces[2].geometry,
        material_post=material,
    )
    train = ScalarOpticalTrain((image,), start_surface=2)
    field = _uniform(n=1.5)
    if changes_medium:
        with pytest.raises(ValueError, match="ImageSurface cannot change"):
            train.propagate(field)
    else:
        output = train.propagate(field)
        assert_array_equal(output.data, field.data)
        assert output.refractive_index == 1.5


@pytest.mark.parametrize(
    "start,end", [(0, None), (-1, 2), (2, 1), (1, 4), (1.2, 2), (True, 2)]
)
def test_invalid_bounds(set_test_backend, start, end):
    optic = _optic(_plane(), _plane(z=1))
    with pytest.raises(ValueError, match="start_surface"):
        ScalarOpticalTrain.from_optic(optic, start, end)


def test_rectangular_aperture_independent_sampled_power(set_test_backend):
    aperture = RectangularAperture(-0.065, 0.045, -0.042, 0.018)
    optic = _optic(_plane(aperture=aperture))
    field = _uniform(shape=(20, 24))
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    x = (np.arange(24) - 11.5) * field.dx
    y = (np.arange(20) - 9.5) * field.dy
    expected = (
        (x[None, :] >= -0.065)
        & (x[None, :] <= 0.045)
        & (y[:, None] >= -0.042)
        & (y[:, None] <= 0.018)
    )
    assert_array_equal(output.data, expected.astype(complex))
    assert_allclose(
        output.power, np.sum(expected) * field.dx * field.dy, rtol=1e-14, atol=0
    )
    assert float(be.to_numpy(output.power)) < float(be.to_numpy(field.power))


@pytest.mark.parametrize("physical_stop", [False, True])
def test_global_ray_aperture_does_not_clip_or_change_supplied_field(
    set_test_backend, monkeypatch, physical_stop
):
    aperture = RectangularAperture(-0.07, 0.06, -0.09, 0.07) if physical_stop else None
    optic = _optic(_plane(is_stop=True, aperture=aperture), _plane(z=0.00313))

    class ForbiddenParaxial:
        def __getattr__(self, name):
            pytest.fail(f"train attempted paraxial stop inference via {name}")

    paraxial = ForbiddenParaxial()
    monkeypatch.setattr(optic, "paraxial", paraxial)
    field = gaussian_field((32, 40), 0.01, 0.0005, 0.06, dy=0.012)
    field_before = be.copy(field.data)
    train = ScalarOpticalTrain.from_optic(optic)
    reference = train.propagate(field)
    for aperture_type, value in (
        ("EPD", 0.02),
        ("EPD", 1000),
        ("imageFNO", 4),
        ("objectNA", 0.3),
    ):
        optic.set_aperture(aperture_type, value)
        source_before = deepcopy(optic.to_dict())
        output = train.propagate(field)
        assert_array_equal(output.data, reference.data)
        assert_array_equal(field.data, field_before)
        assert optic.to_dict() == source_before
        assert optic.paraxial is paraxial
        assert optic.surfaces[1].aperture is aperture
    if not physical_stop:
        assert_allclose(reference.power, field.power, rtol=2e-14, atol=0)


@pytest.mark.parametrize("aperture_type", ["radial", "ellipse"])
def test_other_supported_apertures(set_test_backend, aperture_type):
    field = _uniform()
    x, y = field.coordinates()
    xx, yy = be.meshgrid(x, y)
    if aperture_type == "radial":
        aperture = RadialAperture(0.1, 0.03)
        expected = (xx**2 + yy**2 <= 0.1**2) & (xx**2 + yy**2 >= 0.03**2)
    else:
        aperture = EllipticalAperture(0.1, 0.07, 0.02, -0.01)
        expected = ((xx - 0.02) / 0.1) ** 2 + ((yy + 0.01) / 0.07) ** 2 <= 1
    output = ScalarOpticalTrain.from_optic(_optic(_plane(aperture=aperture))).propagate(
        field
    )
    assert_array_equal(output.data, expected + 0j)


@pytest.mark.parametrize("radius", [25.0, -25.0])
def test_quadratic_conic_phase_sign(set_test_backend, radius):
    optic = _optic(dict(z=0, radius=radius, conic=-1, material=IdealMaterial(1.5)))
    field = _uniform()
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    x = (np.arange(field.shape[1]) - (field.shape[1] - 1) / 2) * field.dx
    y = (np.arange(field.shape[0]) - (field.shape[0] - 1) / 2) * field.dy
    sag = (x[None, :] ** 2 + y[:, None] ** 2) / (2 * radius)
    expected = np.exp(1j * 2 * np.pi / field.wavelength * (1 - 1.5) * sag)
    assert_allclose(output.data, expected, rtol=0, atol=2e-14)
    assert_allclose(output.power, field.power, rtol=2e-14, atol=0)


def test_spherical_sag_and_explicit_screen_asm_equivalence(set_test_backend):
    radius, gap = 12.0, 2.0
    optic = _optic(
        dict(z=0, radius=radius, material=IdealMaterial(1.5)),
        _plane(z=gap),
    )
    field = gaussian_field((64, 80), 0.008, 0.0005, 0.08, dy=0.009)
    # Add arbitrary phase and amplitude structure; no ray-derived field is used.
    x, y = field.coordinates()
    xx, yy = be.meshgrid(x, y)
    field = ScalarField(
        field.data * (1 + 0.1 * xx) * be.exp(15j * yy),
        dx=field.dx,
        dy=field.dy,
        wavelength=field.wavelength,
    )
    r2 = xx**2 + yy**2
    # Independent rationalized sphere expression, rather than geometry.sag.
    sag = r2 / (radius + be.sqrt(radius**2 - r2))
    screen_data = field.data * be.exp(-1j * np.pi / field.wavelength * sag)
    explicit = ScalarField(
        screen_data, field.dx, field.wavelength, dy=field.dy, refractive_index=1.5
    ).propagate(gap)
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert_allclose(output.data, explicit.data, rtol=2e-13, atol=2e-13)
    assert output.refractive_index == 1


@pytest.mark.parametrize("dx,size", [(0.008, 256), (0.004, 512)])
def test_gaussian_focus_independent_abcd_reference(set_test_backend, dx, size):
    """Low NA, >5 input waists padding, and >=5 focused-waist samples."""
    wavelength, waist, focal_length = 0.0005, 0.2, 50.0
    rayleigh = np.pi * waist**2 / wavelength
    q_after = 1 / (1 / (1j * rayleigh) - 1 / focal_length)
    focus_distance = -q_after.real
    focused_waist = np.sqrt(wavelength * q_after.imag / np.pi)
    assert size * dx / 2 >= 5 * waist
    assert focused_waist / dx >= 4.8
    assert 2 * np.pi / wavelength * (3 * waist) / focal_length * dx < np.pi
    optic = _optic(
        dict(z=0, radius=0.5 * focal_length, conic=-1, material=IdealMaterial(1.5)),
        _plane(z=0),
        _plane(z=focus_distance),
    )
    field = gaussian_field((size, size), dx, wavelength, waist)
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    x, y = output.coordinates()
    xx, yy = be.meshgrid(x, y)
    measured_width = be.sqrt(
        2 * be.sum((xx**2 + yy**2) * output.intensity) / be.sum(output.intensity)
    )
    # 2e-4 bounds the ASM-vs-paraxial correction at NA~0.004 and sampling.
    assert_allclose(measured_width, focused_waist, rtol=2e-4, atol=0)
    expected_intensity = (waist / focused_waist) ** 2 * be.exp(
        -2 * (xx**2 + yy**2) / focused_waist**2
    )
    relative_l2 = be.sqrt(
        be.sum((output.intensity - expected_intensity) ** 2)
        / be.sum(expected_intensity**2)
    )
    assert float(be.to_numpy(relative_l2)) < 2e-4
    assert_allclose(output.power, field.power, rtol=3e-13, atol=0)


class _DispersiveMaterial(BaseMaterial):
    def _calculate_n(self, wavelength, **kwargs):
        return 1 + 0.2 * wavelength

    def _calculate_k(self, wavelength, **kwargs):
        return 0.0

    def spectral_range(self, property_name="n"):
        return (0.4, 0.7)


def test_wavelength_mm_to_material_microns_and_incident_medium(set_test_backend):
    optic = _optic(_plane(z=0), incident=_DispersiveMaterial())
    train = ScalarOpticalTrain.from_optic(optic)
    field = _uniform(wavelength=0.00055, n=1.11)
    assert_allclose(train.propagate(field).data, field.data, rtol=0, atol=0)
    with pytest.raises(ValueError, match="incident material"):
        train.propagate(_uniform(wavelength=0.00055, n=1))
    with pytest.raises(ValueError, match="wavelength"):
        train.propagate(_uniform(wavelength=0.55, n=1.11))


@pytest.mark.parametrize("bad_index,bad_k", [(0, 0), (-1, 0), (np.nan, 0), (1.5, 0.01)])
def test_unsupported_material_values(set_test_backend, bad_index, bad_k):
    optic = _optic(_plane(material=IdealMaterial(bad_index, bad_k)))
    with pytest.raises(ValueError, match="finite|lossless"):
        ScalarOpticalTrain.from_optic(optic).propagate(_uniform())


class _ComplexMaterial(_DispersiveMaterial):
    def _calculate_n(self, wavelength, **kwargs):
        return 1.5 + 0.01j


def test_complex_index_rejected(set_test_backend):
    optic = _optic(_plane(material=_ComplexMaterial()))
    with pytest.raises(ValueError, match="real scalar"):
        ScalarOpticalTrain.from_optic(optic).propagate(_uniform())


class _CustomConstantPhase(ConstantPhaseProfile):
    pass


@pytest.mark.parametrize(
    "kind",
    [
        "custom_phase",
        "grid_phase",
        "height_phase",
        "curved_phase",
        "curved_lens",
        "cylindrical_lens",
    ],
)
def test_unaudited_phase_and_thin_lens_interactions_rejected(set_test_backend, kind):
    if kind == "custom_phase":
        optic = _optic(_plane(phase_profile=_CustomConstantPhase(0.3)))
    elif kind in ("grid_phase", "height_phase"):
        coordinates = be.array([-0.2, -0.1, 0.1, 0.2])
        profile_type = GridPhaseProfile if kind == "grid_phase" else HeightProfile
        optic = _optic(
            _plane(
                phase_profile=profile_type(coordinates, coordinates, be.zeros((4, 4)))
            )
        )
    elif kind == "curved_phase":
        optic = _optic(dict(z=0, radius=25, phase_profile=ConstantPhaseProfile(0.3)))
    elif kind == "curved_lens":
        optic = _optic(dict(z=0, radius=25, interaction_type="thin_lens", f=50))
    else:
        optic = _optic(dict(z=0, surface_type="paraxial", f=50))
        optic.surfaces[1].interaction_model.f = be.array([50, 25])
    with pytest.raises(
        ValueError, match="unsupported phase|require a native planar|real scalar"
    ):
        ScalarOpticalTrain.from_optic(optic)


@pytest.mark.parametrize("focal_length", [50.0, -50.0, np.inf, -np.inf])
def test_native_thin_lens_quadratic_phase_reduced_power(set_test_backend, focal_length):
    field = _uniform()
    optic = _optic(
        dict(z=0, surface_type="paraxial", f=focal_length, material=IdealMaterial(1.6))
    )
    x, y = field.coordinates()
    xx, yy = be.meshgrid(x, y)
    expected = be.exp(-1j * np.pi / field.wavelength * (xx**2 + yy**2) / focal_length)
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert_allclose(output.data, expected, rtol=0, atol=3e-14)
    assert_allclose(output.power, field.power, rtol=2e-14, atol=0)
    assert output.refractive_index == 1.6


@pytest.mark.parametrize(
    "focal_length,n_out", [(50, 1), (50, 1.6), (-50, 1), (-50, 1.6)]
)
def test_native_thin_lens_gaussian_reduced_q_reference(
    set_test_backend, focal_length, n_out
):
    wavelength, waist, dx, size = 0.0005, 0.2, 0.005, 512
    # Reduced Q=q/n: a lens subtracts 1/f, independently of output n.
    q_after = 1 / (1 / (1j * np.pi * waist**2 / wavelength) - 1 / focal_length)
    distance = -n_out * q_after.real if focal_length > 0 else n_out * 35
    q_output = q_after + distance / n_out
    reference_width = np.sqrt(-wavelength / (np.pi * (1 / q_output).imag))
    assert size * dx / 2 > 3.5 * max(waist, reference_width)
    assert reference_width / dx > 7
    material = IdealMaterial(n_out)
    optic = _optic(
        dict(z=0, surface_type="paraxial", f=focal_length, material=material),
        _plane(z=distance, material=material),
    )
    field = gaussian_field((size, size), dx, wavelength, waist)
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    x, y = output.coordinates()
    xx, yy = be.meshgrid(x, y)
    measured = be.sqrt(
        2 * be.sum((xx**2 + yy**2) * output.intensity) / be.sum(output.intensity)
    )
    expected_intensity = (waist / reference_width) ** 2 * be.exp(
        -2 * (xx**2 + yy**2) / reference_width**2
    )
    relative_l2 = be.sqrt(
        be.sum((output.intensity - expected_intensity) ** 2)
        / be.sum(expected_intensity**2)
    )
    assert_allclose(measured, reference_width, rtol=2e-4, atol=0)
    assert float(be.to_numpy(relative_l2)) < 2e-4
    assert_allclose(output.power, field.power, rtol=3e-13, atol=0)
    assert output.refractive_index == n_out


@pytest.mark.parametrize("focal_length", [0, np.nan])
def test_native_thin_lens_invalid_f_rejected(set_test_backend, focal_length):
    optic = _optic(dict(z=0, surface_type="paraxial", f=focal_length))
    with pytest.raises(ValueError, match="lens f"):
        ScalarOpticalTrain.from_optic(optic)


@pytest.mark.parametrize("efficiency", [0.0, 0.36, 1.0])
def test_native_constant_phase_efficiency_exact_power_loss(
    set_test_backend, monkeypatch, efficiency
):
    # Native profiles currently inherit unit efficiency. Exercise the existing
    # public property contract without modifying their owned implementation.
    monkeypatch.setattr(
        ConstantPhaseProfile, "efficiency", property(lambda self: efficiency)
    )
    profile = ConstantPhaseProfile(0.37)
    optic = _optic(_plane(phase_profile=profile))
    field = _uniform()
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert_allclose(
        output.data, np.sqrt(efficiency) * np.exp(0.37j), rtol=0, atol=2e-15
    )
    assert_allclose(output.power, efficiency * field.power, rtol=2e-14, atol=1e-15)
    assert profile.phase == 0.37
    assert optic.surfaces[1].interaction_model.phase_profile is profile
    assert profile.parent_surface is optic.surfaces[1]


@pytest.mark.parametrize("coefficients", [[], [20, -130, 300]])
def test_native_radial_phase_radians_positive_sign_and_center(
    set_test_backend, coefficients
):
    center = (0.025, -0.01)
    field = ScalarField(
        _array(np.ones((24, 30), dtype=np.complex128)),
        0.01,
        0.0005,
        dy=0.012,
        center=center,
    )
    profile = RadialPhaseProfile(coefficients)
    optic = _optic(_plane(phase_profile=profile, material=IdealMaterial(1.6)))
    x = (np.arange(30) - 14.5) * field.dx + center[0]
    y = (np.arange(24) - 11.5) * field.dy + center[1]
    xx, yy = np.meshgrid(x, y)
    r2 = xx**2 + yy**2
    phase = sum(
        coefficient * r2**power
        for power, coefficient in enumerate(coefficients, start=1)
    )
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert_allclose(output.data, np.exp(1j * phase), rtol=0, atol=3e-15)
    assert output.center == center
    assert output.refractive_index == 1.6
    assert profile.coefficients is coefficients
    assert_array_equal(field.data, np.ones(field.shape))


def test_native_get_phase_receives_micron_wavelength_on_read_only_view(
    set_test_backend, monkeypatch
):
    calls = []
    original = ConstantPhaseProfile.get_phase

    def record(self, x, y, wavelength):
        calls.append((self, wavelength))
        return original(self, x, y, wavelength)

    monkeypatch.setattr(ConstantPhaseProfile, "get_phase", record)
    profile = ConstantPhaseProfile(0.2)
    optic = _optic(_plane(phase_profile=profile))
    ScalarOpticalTrain.from_optic(optic).propagate(_uniform(wavelength=0.00055))
    assert len(calls) == 1
    assert calls[0][0] is not profile
    assert calls[0][1] == pytest.approx(0.55)


@pytest.mark.parametrize("efficiency", [-0.01, 1.01, np.nan, np.inf, 0.5 + 0.1j])
def test_native_phase_invalid_efficiency_rejected(
    set_test_backend, monkeypatch, efficiency
):
    monkeypatch.setattr(
        ConstantPhaseProfile, "efficiency", property(lambda self: efficiency)
    )
    optic = _optic(_plane(phase_profile=ConstantPhaseProfile(0.2)))
    with pytest.raises(ValueError, match="efficiency"):
        ScalarOpticalTrain.from_optic(optic)


@pytest.mark.parametrize("kind", ["constant", "radial"])
def test_native_phase_nonfinite_parameters_rejected(set_test_backend, kind):
    profile = (
        ConstantPhaseProfile(np.nan)
        if kind == "constant"
        else RadialPhaseProfile([1, np.inf])
    )
    with pytest.raises(ValueError, match="finite"):
        ScalarOpticalTrain.from_optic(_optic(_plane(phase_profile=profile)))


@pytest.mark.parametrize(
    "value",
    [
        complex(np.nan, 0),
        complex(np.inf, 0),
        complex(-np.inf, 0),
        complex(0, np.nan),
        complex(0, np.inf),
    ],
)
def test_nonfinite_input_field_preflight_before_propagation(
    set_test_backend, monkeypatch, value
):
    optic = _optic(_plane(), _plane(z=1))
    train = ScalarOpticalTrain.from_optic(optic)
    data = np.ones((16, 20), dtype=np.complex128)
    data[3, 7] = value
    field = ScalarField(_array(data), 0.01, 0.0005)

    def forbidden(*args, **kwargs):
        pytest.fail("ASM ran before finite input validation")

    monkeypatch.setattr(ScalarField, "propagate", forbidden)
    with pytest.raises(ValueError, match="field data must be finite"):
        train.propagate(field)


def test_finite_zero_field_is_valid(set_test_backend):
    field = ScalarField(_array(np.zeros((16, 20), dtype=np.complex128)), 0.01, 0.0005)
    output = ScalarOpticalTrain.from_optic(_optic(_plane(), _plane(z=1))).propagate(
        field
    )
    assert_array_equal(output.data, field.data)
    assert float(be.to_numpy(output.power)) == 0


@pytest.mark.parametrize("kind", ["lens_f", "phase_coefficient", "efficiency"])
def test_native_screen_preflight_atomic_after_edits(
    set_test_backend, monkeypatch, kind
):
    profile = RadialPhaseProfile([20])
    if kind == "lens_f":
        optic = _optic(_plane(), dict(z=1, surface_type="paraxial", f=50))
    else:
        optic = _optic(_plane(), _plane(z=1, phase_profile=profile))
    train = ScalarOpticalTrain.from_optic(optic)
    if kind == "lens_f":
        optic.surfaces[2].interaction_model.f = be.array(0)
    elif kind == "phase_coefficient":
        profile.coefficients[0] = np.inf
    else:
        monkeypatch.setattr(
            RadialPhaseProfile, "efficiency", property(lambda self: np.nan)
        )

    def forbidden(*args, **kwargs):
        pytest.fail("ASM ran before native screen preflight completed")

    monkeypatch.setattr(ScalarField, "propagate", forbidden)
    with pytest.raises(ValueError):
        train.propagate(_uniform())


@pytest.mark.parametrize(
    "kind",
    [
        "x",
        "y",
        "rx",
        "ry",
        "rz",
        "reference",
        "mirror",
        "coating",
        "scatter",
        "interaction",
        "geometry",
        "grin",
        "negative_gap",
        "multi_configuration",
        "aperture",
    ],
)
def test_unsupported_physics_rejected_at_construction(set_test_backend, kind):
    optic = _optic(_plane(), _plane(z=1))
    surface = optic.surfaces[2]
    model = surface.interaction_model
    if kind in ("x", "y", "rx", "ry", "rz"):
        setattr(surface.geometry.cs, kind, 0.01)
    elif kind == "reference":
        surface.geometry.cs.reference_cs = CoordinateSystem()
    elif kind == "mirror":
        model.is_reflective = True
    elif kind == "coating":
        surface.set_fresnel_coating()
    elif kind == "scatter":
        model.bsdf = object()
    elif kind == "interaction":
        surface.interaction_model = object()
    elif kind == "geometry":
        surface.geometry = OddAsphere(surface.geometry.cs, 25, coefficients=[0.1])
    elif kind == "grin":
        surface.material_post.propagation_model = object()
    elif kind == "negative_gap":
        surface.geometry.cs.z = -1
    elif kind == "multi_configuration":
        surface.geometry.cs.z = be.array([1, 2])
    else:
        surface.aperture = object()
    with pytest.raises(ValueError, match="unsupported|homogeneous|real scalar"):
        ScalarOpticalTrain.from_optic(optic)


@pytest.mark.parametrize("kind", ["mirror", "nonfinite_sag", "extinction"])
def test_preflight_atomic_after_source_edits(set_test_backend, monkeypatch, kind):
    optic = _optic(_plane(), dict(z=1, radius=10, material=IdealMaterial(1.5)))
    train = ScalarOpticalTrain.from_optic(optic)
    if kind == "mirror":
        optic.surfaces[2].interaction_model.is_reflective = True
    elif kind == "nonfinite_sag":
        optic.surfaces[2].geometry.radius = be.array(0.01)
    else:
        optic.surfaces[2].material_post.absorp = be.array([0.01])

    def forbidden(*args, **kwargs):
        pytest.fail("propagation ran before all surfaces were validated")

    monkeypatch.setattr(ScalarField, "propagate", forbidden)
    with np.errstate(invalid="ignore"), pytest.raises(ValueError):
        train.propagate(_uniform())


def test_invalid_sag_outside_aperture_is_not_evaluated(set_test_backend):
    optic = _optic(
        dict(
            z=0, radius=0.1, material=IdealMaterial(1.5), aperture=RadialAperture(0.05)
        )
    )
    field = _uniform()
    with np.errstate(invalid="raise"):
        output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert bool(be.all(be.isfinite(output.data)))


def test_source_optic_field_and_material_caches_unchanged(set_test_backend):
    optic = _optic(dict(z=0, radius=10, material=IdealMaterial(1.5)), _plane(z=2))
    field = gaussian_field((32, 32), 0.01, 0.0005, 0.06)
    field_before = be.copy(field.data)
    optic_before = deepcopy(optic.to_dict())
    materials = [surface.material_post for surface in optic.surfaces]
    for material in materials:
        material.n(0.6)
    cache_before = [dict(material._n_cache) for material in materials]
    contexts_before = [material._cache_context for material in materials]
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert output is not field
    assert_array_equal(field.data, field_before)
    assert optic.to_dict() == optic_before
    assert [material._cache_context for material in materials] == contexts_before
    for material, previous in zip(materials, cache_before, strict=True):
        assert material._n_cache.keys() == previous.keys()
        for key in previous:
            assert material._n_cache[key] is previous[key]


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_torch_geometry_field_dtype_device_and_gradients(set_test_backend, precision):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific gradient test")
    import torch

    optic = _optic(dict(z=0, radius=25, conic=-1, material=IdealMaterial(1.5)))
    radius = torch.tensor(25.0, dtype=torch.float64, requires_grad=True)
    optic.surfaces[1].geometry.radius = radius
    data = torch.ones((20, 24), dtype=getattr(torch, precision), requires_grad=True)
    field = ScalarField(data, 0.01, 0.0005)
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert output.data.dtype == data.dtype
    assert output.data.device == data.device
    loss = output.data.real.sum()
    loss.backward()
    assert data.grad is not None and torch.isfinite(data.grad).all()
    x, y = field.coordinates()
    xx, yy = be.meshgrid(x, y)
    phase = -np.pi / field.wavelength * (xx**2 + yy**2) / (2 * radius.detach())
    expected_derivative = torch.sum(torch.sin(phase) * phase / radius.detach())
    assert_allclose(radius.grad, expected_derivative, rtol=2e-6, atol=1e-7)
    if precision == "complex128":
        # Central differences over multiple steps independently check the
        # geometry-to-screen autodiff path without a second implementation.
        train = ScalarOpticalTrain.from_optic(optic)
        for step in (1e-2, 1e-3, 1e-4):
            with torch.no_grad():
                radius.add_(step)
                plus = train.propagate(field).data.real.sum().item()
                radius.sub_(2 * step)
                minus = train.propagate(field).data.real.sum().item()
                radius.add_(step)
            assert_allclose(
                radius.grad, (plus - minus) / (2 * step), rtol=3e-7, atol=1e-8
            )


@pytest.mark.parametrize("distance", [0.0, 0.003])
def test_torch_vertex_gap_gradients_including_zero(set_test_backend, distance):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific gradient test")
    import torch

    optic = _optic(_plane(material=IdealMaterial(1.5)), _plane(z=distance))
    vertex = torch.tensor(distance, dtype=torch.float64, requires_grad=True)
    optic.surfaces[2].geometry.cs.z = vertex
    field = _uniform()
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    output.data.imag.sum().backward()
    k = 2 * np.pi * 1.5 / field.wavelength
    expected = np.prod(field.shape) * k * np.cos(k * distance)
    assert_allclose(vertex.grad, expected, rtol=1e-12, atol=1e-8)


def test_large_vertex_origin_does_not_erase_float32_gap(set_test_backend):
    # The binary-exact gap avoids representational error in the source vertices;
    # a nonintegral optical cycle makes a lost (zero) gap readily detectable.
    translated = _optic(_plane(z=1e8), _plane(z=1e8 + 1.125))
    centered = _optic(_plane(z=0), _plane(z=1.125))
    field = ScalarField(_array(np.ones((16, 16), dtype=np.complex64)), 0.01, 0.00053)
    actual = ScalarOpticalTrain.from_optic(translated).propagate(field)
    expected = ScalarOpticalTrain.from_optic(centered).propagate(field)
    assert actual.data.dtype == field.data.dtype
    assert_array_equal(actual.data, expected.data)


def test_single_element_vertex_coordinates_are_scalar_metadata(set_test_backend):
    optic = _optic(_plane(), _plane(z=0.00313))
    optic.surfaces[1].geometry.cs.z = be.array([0.0])
    optic.surfaces[2].geometry.cs.z = be.array([0.00313])
    field = _uniform(wavelength=0.00053)
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    expected = np.exp(2j * np.pi / field.wavelength * 0.00313)
    assert_allclose(output.data, expected, rtol=0, atol=3e-13)
    assert optic.surfaces[2].geometry.cs.z.shape == (1,)


def test_shifted_grid_preserved_through_screen_and_gap(set_test_backend):
    center = (0.025, -0.01)
    field = ScalarField(
        _array(np.ones((24, 30), dtype=np.complex128)),
        0.01,
        0.0005,
        dy=0.012,
        center=center,
    )
    aperture = RectangularAperture(-0.08, 0.06, -0.05, 0.08)
    optic = _optic(
        dict(z=0, radius=25, conic=-1, material=IdealMaterial(1.5), aperture=aperture),
        _plane(z=0.003),
    )
    x = (np.arange(30) - 14.5) * field.dx + center[0]
    y = (np.arange(24) - 11.5) * field.dy + center[1]
    xx, yy = np.meshgrid(x, y)
    mask = (xx >= -0.08) & (xx <= 0.06) & (yy >= -0.05) & (yy <= 0.08)
    sag = (xx**2 + yy**2) / 50
    data = _array(mask * np.exp(-1j * np.pi / field.wavelength * sag))
    explicit = ScalarField(
        data,
        field.dx,
        field.wavelength,
        dy=field.dy,
        refractive_index=1.5,
        center=center,
    ).propagate(0.003)
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert output.center == center
    assert_allclose(output.data, explicit.data, rtol=2e-13, atol=2e-13)
    for actual, expected in zip(output.coordinates(), field.coordinates(), strict=True):
        assert_array_equal(actual, expected)


def test_native_radial_phase_spatial_gradient_spectral_moment(set_test_backend):
    """A positive quadratic phase gives positive mean kx on a shifted beam."""
    size, dx, waist, beam_x, coefficient = 256, 0.004, 0.06, 0.08, 50.0
    axis = (np.arange(size) - (size - 1) / 2) * dx
    xx, yy = np.meshgrid(axis, axis)
    amplitude = np.exp(-((xx - beam_x) ** 2 + yy**2) / waist**2)
    field = ScalarField(_array(amplitude.astype(complex)), dx, 0.0005)
    profile = RadialPhaseProfile([coefficient])
    output = ScalarOpticalTrain.from_optic(
        _optic(_plane(phase_profile=profile))
    ).propagate(field)
    spectrum_power = np.abs(np.fft.fft2(be.to_numpy(output.data))) ** 2
    k_axis = 2 * np.pi * np.fft.fftfreq(size, dx)
    kx_mean = np.sum(spectrum_power * k_axis[None, :]) / np.sum(spectrum_power)
    ky_mean = np.sum(spectrum_power * k_axis[:, None]) / np.sum(spectrum_power)
    assert_allclose(kx_mean, 2 * coefficient * beam_x, rtol=2e-13, atol=1e-12)
    assert_allclose(ky_mean, 0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["thin_lens", "constant", "radial"])
@pytest.mark.parametrize(
    "field_precision,default_precision",
    [("complex64", "float64"), ("complex128", "float32")],
)
def test_native_screens_preserve_nondefault_field_precision(
    set_test_backend, kind, field_precision, default_precision
):
    original_precision = be.get_precision()
    try:
        be.set_precision(default_precision)
        field = ScalarField(
            _array(np.ones((20, 24), dtype=field_precision)),
            0.01,
            0.00053,
            center=(0.025, -0.01),
        )
        x, y = field.coordinates()
        xx, yy = np.meshgrid(be.to_numpy(x), be.to_numpy(y))
        if kind == "thin_lens":
            optic = _optic(dict(z=0, surface_type="paraxial", f=50))
            focal_length = 50.123456789
            optic.surfaces[1].interaction_model.f = _array(
                np.array(focal_length, dtype=np.float64)
            )
            phase = -np.pi / field.wavelength * (xx**2 + yy**2) / focal_length
        elif kind == "constant":
            phase = 0.1234567890123
            optic = _optic(_plane(phase_profile=ConstantPhaseProfile(phase)))
        else:
            coefficients = [20.123456789, -130.234567891]
            optic = _optic(_plane(phase_profile=RadialPhaseProfile(coefficients)))
            phase = (
                coefficients[0] * (xx**2 + yy**2)
                + coefficients[1] * (xx**2 + yy**2) ** 2
            )
        output = ScalarOpticalTrain.from_optic(optic).propagate(field)
        assert output.data.dtype == field.data.dtype
        assert output.center == field.center
        tolerance = 2e-6 if field_precision == "complex64" else 3e-13
        assert_allclose(output.data, np.exp(1j * phase), rtol=0, atol=tolerance)
    finally:
        be.set_precision("float64" if original_precision == 64 else "float32")


@pytest.mark.parametrize("kind", ["thin_lens", "constant", "radial"])
@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_torch_native_screen_parameter_and_field_gradients(
    set_test_backend, monkeypatch, kind, precision
):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific gradient test")
    import torch

    data = torch.ones((20, 24), dtype=getattr(torch, precision), requires_grad=True)
    field = ScalarField(data, 0.01, 0.0005, center=(0.025, -0.01))
    efficiency = torch.tensor(0.36, dtype=torch.float64, requires_grad=True)
    if kind == "thin_lens":
        parameter = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
        optic = _optic(dict(z=0, surface_type="paraxial", f=50))
        optic.surfaces[1].interaction_model.f = parameter
    elif kind == "constant":
        parameter = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
        profile = ConstantPhaseProfile(parameter)
        monkeypatch.setattr(
            ConstantPhaseProfile, "efficiency", property(lambda self: efficiency)
        )
        optic = _optic(_plane(phase_profile=profile))
    else:
        parameter = torch.tensor(
            [20.0, -130.0], dtype=torch.float64, requires_grad=True
        )
        profile = RadialPhaseProfile(parameter)
        optic = _optic(_plane(phase_profile=profile))
    train = ScalarOpticalTrain.from_optic(optic)
    output = train.propagate(field)
    output.data.real.sum().backward()
    assert output.data.dtype == data.dtype and output.data.device == data.device
    assert output.center == field.center
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert data.grad is not None and torch.isfinite(data.grad).all()
    assert_array_equal(field.data, torch.ones_like(field.data))
    x, y = field.coordinates()
    xx, yy = be.meshgrid(x, y)
    r2 = xx**2 + yy**2
    values = parameter.detach().to(dtype=field.data.real.dtype)
    if kind == "thin_lens":
        phase = -np.pi / field.wavelength * r2 / values
        expected = torch.sum(torch.sin(phase) * phase / values)
        assert optic.surfaces[1].interaction_model.f is parameter
    elif kind == "constant":
        amplitude = torch.sqrt(efficiency.detach().to(dtype=field.data.real.dtype))
        expected = -data.numel() * amplitude * torch.sin(values)
        expected_efficiency = data.numel() * torch.cos(values) / (2 * amplitude)
        assert_allclose(efficiency.grad, expected_efficiency, rtol=3e-6, atol=1e-7)
        assert profile.phase is parameter
    else:
        phase = values[0] * r2 + values[1] * r2**2
        expected = torch.stack(
            [-torch.sum(torch.sin(phase) * r2), -torch.sum(torch.sin(phase) * r2**2)]
        )
        assert profile.coefficients is parameter
    assert_allclose(parameter.grad, expected, rtol=3e-6, atol=1e-7)
    if precision == "complex128":
        original = parameter.detach().clone()
        direction = torch.ones_like(parameter)
        if kind == "radial":
            direction[1] = 0
        directional_derivative = torch.sum(parameter.grad * direction)
        for step in (1e-3, 1e-4, 1e-5):
            with torch.no_grad():
                parameter.copy_(original + step * direction)
                plus = train.propagate(field).data.real.sum().item()
                parameter.copy_(original - step * direction)
                minus = train.propagate(field).data.real.sum().item()
                parameter.copy_(original)
            assert_allclose(
                directional_derivative,
                (plus - minus) / (2 * step),
                rtol=3e-7,
                atol=1e-8,
            )


@pytest.mark.parametrize("kind", ["thin_lens", "constant", "radial"])
@pytest.mark.parametrize("unsupported", ["mirror", "coating", "scatter"])
def test_native_screen_interaction_safety_gates(set_test_backend, kind, unsupported):
    if kind == "thin_lens":
        optic = _optic(dict(z=0, surface_type="paraxial", f=50))
    else:
        profile = (
            ConstantPhaseProfile(0.2)
            if kind == "constant"
            else RadialPhaseProfile([20])
        )
        optic = _optic(_plane(phase_profile=profile))
    model = optic.surfaces[1].interaction_model
    if unsupported == "mirror":
        model.is_reflective = True
    elif unsupported == "coating":
        optic.surfaces[1].set_fresnel_coating()
    else:
        model.bsdf = object()
    with pytest.raises(ValueError, match="unsupported"):
        ScalarOpticalTrain.from_optic(optic)


def test_type_validation(set_test_backend):
    with pytest.raises(TypeError, match="Optic"):
        ScalarOpticalTrain.from_optic(object())
    train = ScalarOpticalTrain.from_optic(_optic(_plane()))
    with pytest.raises(TypeError, match="ScalarField"):
        train.propagate(be.ones((4, 4)))
