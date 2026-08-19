"""Uncoated plane-parallel window: closed-form total transmittance.

For a plane-parallel plate of index n with two uncoated (bare Fresnel)
surfaces at normal incidence, the total transmittance including *every*
internal reflection (the ray bounces back and forth, losing R at each
surface, and Monte Carlo must sum the resulting geometric series correctly
in expectation) is the classic closed form::

    T = (1 - R)^2 / (1 - R^2) = 2n / (n^2 + 1),  R = ((n-1)/(n+1))^2

This is a strong end-to-end check on the Fresnel branch estimator: it only
comes out right if the reflect/transmit split at *each* surface is unbiased,
not just the first one.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.materials import Material
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend


@pytest.mark.parametrize("glass", ["N-BK7", "SF11"])
def test_window_transmittance_matches_closed_form(glass):
    n = float(np.asarray(Material(glass).n(0.55)).ravel()[0])
    t_theory = 2.0 * n / (n**2 + 1.0)

    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=3.0),
    )
    # Large radii (~flat plate); the front/back aperture is well inside the
    # collimated beam is not, so most flux clears the barrel edge.
    scene.add_lens(
        "W",
        CoordinateSystem(z=10.0),
        LensConfig(
            r1=1.0e9,
            r2=1.0e9,
            thickness=2.0,
            material=glass,
            front_aperture_radius=10.0,
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=50.0),
        IrradianceDetectorConfig(
            width=20, height=20, num_pixels_x=8, num_pixels_y=8, splat="hard"
        ),
    )
    result = scene.trace(
        num_rays=200_000, seed=1, max_depth=40, backend=NumpyBackend(seed=1)
    )
    t_sim = result.total_flux_detected / result.total_flux_in

    assert t_sim == pytest.approx(t_theory, rel=0.01)
    assert result.flux_conservation_error < 1e-6
