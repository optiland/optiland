"""Small Lambertian disc -> parallel plane: cos^4 falloff.

For a small Lambertian-emitting disc (radiant intensity I(theta) = I0 *
cos(theta) about the disc normal) illuminating a plane parallel to the disc,
the irradiance at an off-axis point falls off as ``E(theta) = E(0) *
cos^4(theta)``, where theta is the angle subtended from the disc's axis
(the classic photometric "cos-fourth law": one cosine from the source's
Lambertian intensity falloff, one from the detector's foreshortened
projected area, and 1/r^2 with r = d/cos(theta) contributing cos^2(theta)).

``ExtendedSourceConfig(half_angle_deg=90)`` is exactly this: a finite-area
source with cosine-weighted (Lambertian) emission over the hemisphere.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    ExtendedSourceConfig,
    IrradianceDetectorConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend


def test_cos4_falloff_off_axis():
    distance = 30.0
    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        ExtendedSourceConfig(
            spectrum=spec, total_flux=1.0, width=0.2, height=0.2, half_angle_deg=90.0
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=distance),
        IrradianceDetectorConfig(
            width=40, height=2, num_pixels_x=80, num_pixels_y=1, splat="hard"
        ),
    )
    result = scene.trace(num_rays=4_000_000, seed=1, backend=NumpyBackend(seed=1))
    irr = result.detectors["D"]
    x = irr.x_coords
    e = irr.irradiance[0, :]

    # On-axis irradiance (theta ~= 0) as the E(0) reference.
    e0 = e[np.argmin(np.abs(x))]
    theta = np.arctan(x / distance)
    predicted = e0 * np.cos(theta) ** 4

    # Restrict to |x| < 15 mm (theta up to ~27 deg): near the edge of the
    # detector, the bin-averaged angle differs enough from the bin-centre
    # angle used in `predicted` that pixel-width effects start to matter
    # more than the physics being tested.
    mask = np.abs(x) < 15.0
    rel_err = np.abs(e[mask] - predicted[mask]) / predicted[mask]
    assert np.mean(rel_err) < 0.08
    assert np.max(rel_err) < 0.20
