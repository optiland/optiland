"""Point source -> small flat patch: exact inverse-square law.

An isotropic point source (``half_angle_deg=180``) emits uniformly over the
full 4*pi steradian sphere by construction (uniform sampling on the sphere,
see ``PointSource.generate``), so the irradiance on a small flat patch a
distance ``d`` away, facing the source, is exactly ``E = F / (4*pi*d**2)`` --
no lens, no material, nothing else in the scene to get subtly wrong.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    IrradianceDetectorConfig,
    NSQScene,
    PointSourceConfig,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend


@pytest.mark.parametrize("distance", [20.0, 50.0, 100.0])
def test_irradiance_matches_inverse_square_law(distance):
    total_flux = 2.0
    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        PointSourceConfig(spectrum=spec, total_flux=total_flux, half_angle_deg=180.0),
    )
    # Patch small relative to distance, so it samples a near-uniform patch of
    # the sphere (a finite-size correction would otherwise bias E low).
    patch = 0.02 * distance
    scene.add_detector(
        "D",
        CoordinateSystem(z=distance),
        IrradianceDetectorConfig(
            width=patch, height=patch, num_pixels_x=1, num_pixels_y=1, splat="hard"
        ),
    )
    result = scene.trace(num_rays=2_000_000, seed=1, backend=NumpyBackend(seed=1))
    e_sim = float(result.detectors["D"].irradiance[0, 0])
    e_theory = total_flux / (4.0 * np.pi * distance**2)

    # Dominated by shot noise: a patch this small only catches a tiny
    # fraction of the 4*pi sphere's rays (~patch_area / (4*pi*d^2) of them),
    # so relative error scales as 1/sqrt(hits), not 1/sqrt(num_rays) --
    # 15% is the honest bound for this ray count, not a loose fudge factor.
    assert e_sim == pytest.approx(e_theory, rel=0.15)
