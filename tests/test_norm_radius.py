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
