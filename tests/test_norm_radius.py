from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.optic import Optic


@pytest.fixture
def zernike_optic():
    optic = Optic()
    optic.surfaces.add(surface_type="standard", thickness=be.inf, index=0)  # object
    optic.surfaces.add(
        surface_type="standard", is_stop=True, radius=10, thickness=20, index=1
    )
    # Add a Zernike surface which has a norm_radius
    optic.surfaces.add(surface_type="zernike", radius=-10, thickness=50, index=2)
    optic.surfaces.add(surface_type="standard", index=3)  # image
    optic.set_aperture("EPD", 5.0)
    optic.fields.set_type("angle")
    optic.fields.add(0.0)
    optic.wavelengths.add(0.55)
    return optic


def test_init_kwargs_behavior(set_test_backend):
    # Tests that adding a surface with `norm_radius` as a kwarg locks it
    optic = Optic()
    optic.surfaces.add(surface_type="standard", thickness=10, index=0)
    optic.surfaces.add(
        surface_type="standard", is_stop=True, radius=10, thickness=20, index=1
    )
    optic.surfaces.add(
        surface_type="zernike", radius=-10, thickness=50, index=2, norm_radius=111.0
    )
    optic.set_aperture("EPD", 5.0)
    optic.fields.set_type("angle")
    optic.fields.add(0.0)
    optic.wavelengths.add(0.55)

    zernike_surface = optic.surfaces[2]
    assert getattr(zernike_surface.geometry, "normalization_mode", "auto") == "manual"
    assert zernike_surface.geometry.norm_radius == 111.0

    optic.updater.update_paraxial()
    assert getattr(zernike_surface.geometry, "normalization_mode", "auto") == "manual"
    assert zernike_surface.geometry.norm_radius == 111.0


def test_default_behavior(zernike_optic, set_test_backend):
    # Without calling set_norm_radius, the norm_radius should be auto-updated
    zernike_optic.updater.update_paraxial()
    zernike_surface = zernike_optic.surfaces[2]

    # Check if norm_radius has been set to 1.25 * semi_aperture
    semi_aperture = zernike_surface.semi_aperture

    semi_aperture_val = (
        float(semi_aperture) if hasattr(semi_aperture, "item") else semi_aperture
    )
    norm_radius_val = (
        float(zernike_surface.geometry.norm_radius)
        if hasattr(zernike_surface.geometry.norm_radius, "item")
        else zernike_surface.geometry.norm_radius
    )

    assert norm_radius_val == pytest.approx(semi_aperture_val * 1.25)
    assert getattr(zernike_surface.geometry, "normalization_mode", "auto") == "auto"


def test_fixed_behavior(zernike_optic, set_test_backend):
    # Set norm_radius explicitly
    custom_norm_radius = 42.0
    zernike_optic.updater.set_norm_radius(custom_norm_radius, 2)  # surface index 2

    zernike_surface = zernike_optic.surfaces[2]
    assert getattr(zernike_surface.geometry, "normalization_mode", "auto") == "manual"
    assert zernike_surface.geometry.norm_radius == custom_norm_radius

    # update paraxial should not change it
    zernike_optic.updater.update_paraxial()
    assert zernike_surface.geometry.norm_radius == custom_norm_radius


def test_reversibility(zernike_optic, set_test_backend):
    # Fix it
    custom_norm_radius = 42.0
    zernike_optic.updater.set_norm_radius(custom_norm_radius, 2)

    # Unfix it
    zernike_optic.updater.set_norm_radius(custom_norm_radius, 2, is_fixed=False)

    zernike_surface = zernike_optic.surfaces[2]
    assert getattr(zernike_surface.geometry, "normalization_mode", "manual") == "auto"

    # Now it should auto-scale
    zernike_optic.updater.update_paraxial()
    semi_aperture = zernike_surface.semi_aperture
    # Auto-scaling kicks in overriding custom
    semi_aperture_val = (
        float(semi_aperture) if hasattr(semi_aperture, "item") else semi_aperture
    )
    norm_radius_val = (
        float(zernike_surface.geometry.norm_radius)
        if hasattr(zernike_surface.geometry.norm_radius, "item")
        else zernike_surface.geometry.norm_radius
    )

    assert norm_radius_val == pytest.approx(semi_aperture_val * 1.25)
    assert norm_radius_val != custom_norm_radius


def _build_tma_578(dy_last):
    """Three-mirror Zernike TMA from optiland/optiland#578.

    Reproduces the reporter's example: three off-axis (tilted + decentered)
    Zernike mirrors with no coefficients (i.e. spherical/conic in shape).
    With `dy_last=-75` the system traces cleanly; decentering the last
    mirror by only 5 mm (`dy_last=-70`) previously raised
    ``ValueError: Zernike coordinates must be normalized to [-1, 1]`` because
    the paraxial marginal/chief ray heights used to size the normalization
    radius are an axisymmetric estimate that under-represents the true ray
    footprint once surfaces are decentered/tilted.
    """
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, thickness=120, is_stop=True)
    optic.surfaces.add(
        index=2,
        radius=-400,
        thickness=-125,
        conic=0,
        material="mirror",
        rx=be.radians(-10.0),
        surface_type="zernike",
        zernike_type="fringe",
        coefficients=[],
    )
    optic.surfaces.add(
        index=3,
        radius=300,
        thickness=125,
        conic=0,
        material="mirror",
        rx=be.radians(-3.0),
        dy=-45,
        surface_type="zernike",
        zernike_type="fringe",
        coefficients=[],
    )
    optic.surfaces.add(
        index=4,
        radius=-100,
        thickness=-135,
        conic=0,
        material="mirror",
        rx=be.radians(2.0),
        dy=dy_last,
        surface_type="zernike",
        zernike_type="fringe",
        coefficients=[],
    )
    optic.surfaces.add(index=5, dy=-100)
    optic.set_aperture(aperture_type="EPD", value=30)
    optic.fields.set_type("angle")
    optic.fields.add(x=0, y=-5)
    optic.fields.add(x=0, y=0)
    optic.fields.add(x=0, y=5)
    optic.wavelengths.add(value=0.7, is_primary=True)
    return optic


def test_off_axis_zernike_normalization_regression_578(set_test_backend):
    """Decentering an off-axis Zernike TMA by 5 mm must not raise (#578)."""
    optic = _build_tma_578(dy_last=-70)
    optic.updater.update_paraxial()

    # This is the exact field/pupil combination that previously overflowed
    # the auto-sized normalization radius of the last (most off-axis)
    # mirror.
    optic.trace_generic(Hx=0, Hy=-1, Px=0, Py=1, wavelength=0.7)


def test_off_axis_zernike_normalization_working_baseline_578(set_test_backend):
    """The reporter's original (working) decenter must keep tracing cleanly."""
    optic = _build_tma_578(dy_last=-75)
    optic.updater.update_paraxial()

    optic.trace_generic(Hx=0, Hy=-1, Px=0, Py=1, wavelength=0.7)
    optic.trace_generic(Hx=0, Hy=1, Px=0, Py=-1, wavelength=0.7)


def test_optimizer_precedence(zernike_optic, set_test_backend):
    # Fix it
    custom_norm_radius = 42.0
    zernike_optic.updater.set_norm_radius(custom_norm_radius, 2)

    zernike_surface = zernike_optic.surfaces[2]

    # Mock optimizer setting it as variable
    zernike_surface.is_norm_radius_variable = True

    # Simulate optimizer varying the radius
    optimizer_driven_radius = 55.0
    zernike_surface.geometry.norm_radius = optimizer_driven_radius

    # update shouldn't override the optimizer's new value, despite is_fixed=True
    zernike_optic.updater.update_paraxial()

    assert getattr(zernike_surface.geometry, "normalization_mode", "auto") == "manual"
    assert zernike_surface.geometry.norm_radius == optimizer_driven_radius
