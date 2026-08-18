"""Coating and mirror-reflectance tests for Non-Sequential Raytracing.

Covers D-2 (RefractiveComponent honoring an attached optiland.coatings
coating instead of the bare Fresnel split) and D-3 (ReflectiveComponent
requiring an explicit reflectance instead of an implicit perfect mirror).

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coatings import FresnelCoating, SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    MirrorConfig,
    NSQMaterial,
    NSQScene,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry
from optiland.nonsequential.components.reflective import ReflectiveComponent

GREEN = 0.55


def _glass():
    return NSQMaterial.from_glass("N-BK7")


def _coated_interface_scene(coating, *, num_rays=60_000):
    """Collimated beam at normal incidence on a single coated interface.

    Returns (reflected_total_flux, transmitted_total_flux, input_flux).
    """
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-5.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(GREEN),
            total_flux=1.0,
            aperture_radius=1.0,
        ),
    )
    scene.add_component(
        "I",
        RefractiveComponent(
            cs=CoordinateSystem(z=0.0),
            geometry=PlaneGeometry(),
            material_front=VACUUM,
            material_back=_glass(),
            coating=coating,
            name="I",
        ),
    )
    scene.add_detector(
        "R",
        CoordinateSystem(z=-15.0),
        IrradianceDetectorConfig(width=10, height=10, num_pixels_x=8, num_pixels_y=8),
    )
    scene.add_detector(
        "T",
        CoordinateSystem(z=15.0),
        IrradianceDetectorConfig(width=10, height=10, num_pixels_x=8, num_pixels_y=8),
    )
    result = scene.trace(num_rays=num_rays, seed=7)
    return (
        result.detectors["R"].total_flux,
        result.detectors["T"].total_flux,
        1.0,
    )


class TestCoatingOverridesFresnel:
    def test_coating_reflectance_and_transmittance_match_flux_split(self):
        """NSQ's coating-driven R/T must agree with the SimpleCoating values,
        not the bare (uncoated) Fresnel reflectance at this interface."""
        coating = SimpleCoating(transmittance=0.85, reflectance=0.10)
        r_flux, t_flux, in_flux = _coated_interface_scene(coating)

        assert r_flux == pytest.approx(coating.reflectance * in_flux, abs=0.01)
        assert t_flux == pytest.approx(coating.transmittance * in_flux, abs=0.01)
        # Absorptance (0.05) is unaccounted-for flux, not a third detector hit.
        assert r_flux + t_flux == pytest.approx(
            (coating.reflectance + coating.transmittance) * in_flux, abs=0.01
        )

    def test_uncoated_interface_still_uses_bare_fresnel(self):
        """No coating attached -> unchanged pre-PR7 behaviour: R+T == 1."""
        r_flux, t_flux, in_flux = _coated_interface_scene(coating=None)
        assert r_flux + t_flux == pytest.approx(in_flux, abs=0.01)
        # Normal incidence, VACUUM -> N-BK7: small but nonzero Fresnel R.
        assert 0.0 < r_flux < 0.1

    def test_polarized_coating_on_refractive_surface_raises(self):
        with pytest.raises(NotImplementedError, match="polarized"):
            RefractiveComponent(
                cs=CoordinateSystem(z=0.0),
                geometry=PlaneGeometry(),
                material_front=VACUUM,
                material_back=_glass(),
                coating=FresnelCoating(None, None),
                name="I",
            )


class TestMirrorReflectanceIsRequired:
    def test_reflective_component_requires_reflectance(self):
        with pytest.raises(TypeError):
            ReflectiveComponent(cs=CoordinateSystem(z=0.0), geometry=PlaneGeometry())

    def test_reflective_component_rejects_polarized_coating(self):
        coating = FresnelCoating(None, None)
        with pytest.raises(NotImplementedError, match="polarized"):
            ReflectiveComponent(
                cs=CoordinateSystem(z=0.0),
                geometry=PlaneGeometry(),
                reflectance=coating,
            )

    def test_constant_reflectance_scales_flux(self):
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=-10.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(GREEN),
                total_flux=1.0,
                aperture_radius=1.0,
            ),
        )
        scene.add_mirror(
            "M",
            CoordinateSystem(z=0.0),
            MirrorConfig(radius=np.inf, reflectance=0.5, aperture_radius=10.0),
        )
        scene.add_detector(
            "R",
            CoordinateSystem(z=-20.0),
            IrradianceDetectorConfig(
                width=10, height=10, num_pixels_x=8, num_pixels_y=8
            ),
        )
        result = scene.trace(num_rays=20_000, seed=3)
        assert result.detectors["R"].total_flux == pytest.approx(0.5, abs=0.02)

    def test_coating_reflectance_on_mirror_scales_flux(self):
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=-10.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(GREEN),
                total_flux=1.0,
                aperture_radius=1.0,
            ),
        )
        scene.add_mirror(
            "M",
            CoordinateSystem(z=0.0),
            MirrorConfig(
                radius=np.inf,
                reflectance=SimpleCoating(transmittance=0.0, reflectance=0.8),
                aperture_radius=10.0,
            ),
        )
        scene.add_detector(
            "R",
            CoordinateSystem(z=-20.0),
            IrradianceDetectorConfig(
                width=10, height=10, num_pixels_x=8, num_pixels_y=8
            ),
        )
        result = scene.trace(num_rays=20_000, seed=3)
        assert result.detectors["R"].total_flux == pytest.approx(0.8, abs=0.02)

    def test_make_surface_reflective_override_without_reflectance_raises(self):
        from optiland.nonsequential.components.configs import (
            InteractionType,
            SurfaceConfig,
        )
        from optiland.nonsequential.components.lens import _make_surface

        with pytest.raises(ValueError, match="reflectance"):
            _make_surface(
                CoordinateSystem(z=0.0),
                PlaneGeometry(),
                VACUUM,
                VACUUM,
                SurfaceConfig(interaction=InteractionType.REFLECTIVE),
                InteractionType.ABSORBING,
                name="edge",
            )
