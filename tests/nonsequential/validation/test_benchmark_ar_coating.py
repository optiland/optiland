"""AR-coated surface: NSQ reflectance vs optiland.coatings directly.

An ``optiland.coatings.SimpleCoating`` attached to a refractive interface
replaces the bare Fresnel reflectance with the coating's own ``.reflectance``
(D-2) -- the same number ``optiland.coatings`` reports, so the
sequential and non-sequential engines agree. This traces the reflected and
transmitted flux off an AR-coated interface and checks both against the
coating object's own values directly (not a re-derived Fresnel formula --
the whole point is that NSQ defers to the coating, not to its own physics).

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

from optiland.coatings import SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQMaterial,
    NSQScene,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry


@pytest.mark.parametrize("r_coating,t_coating", [(0.005, 0.995), (0.02, 0.98)])
def test_coated_interface_flux_split_matches_coating_object(r_coating, t_coating):
    coating = SimpleCoating(reflectance=r_coating, transmittance=t_coating)
    glass = NSQMaterial.from_glass("N-BK7")

    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-5.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=1.0
        ),
    )
    scene.add_component(
        "I",
        RefractiveComponent(
            cs=CoordinateSystem(z=0.0),
            geometry=PlaneGeometry(),
            material_front=VACUUM,
            material_back=glass,
            coating=coating,
        ),
    )
    scene.add_detector(
        "R",
        CoordinateSystem(z=-15.0),
        IrradianceDetectorConfig(width=10, height=10, num_pixels_x=4, num_pixels_y=4),
    )
    scene.add_detector(
        "T",
        CoordinateSystem(z=15.0),
        IrradianceDetectorConfig(width=10, height=10, num_pixels_x=4, num_pixels_y=4),
    )
    result = scene.trace(num_rays=100_000, seed=7)

    r_sim = result.detectors["R"].total_flux_float
    t_sim = result.detectors["T"].total_flux_float

    # The same numbers optiland.coatings.SimpleCoating itself reports --
    # not a Fresnel formula re-derived independently.
    assert r_sim == pytest.approx(coating.reflectance, abs=0.01)
    assert t_sim == pytest.approx(coating.transmittance, abs=0.01)
