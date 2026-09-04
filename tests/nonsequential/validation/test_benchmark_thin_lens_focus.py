"""Thin lens focus: focal spot position vs the paraxial prediction.

A thin symmetric biconvex lens' focal distance (measured from its centre)
is, to first order, the thin-lens form of the lensmaker's equation::

    f = 1 / ((n - 1) * (1/R1 - 1/R2))

Rather than searching for the exact focal plane (expensive: many detector
traces), this asserts the weaker but still discriminating statement the
paraxial prediction actually makes: the spot is smaller at the predicted
focus than at a plane defocused by several mm on either side. A wrong focal
-length calculation (e.g. a sign error, or an index mix-up) would put the
true minimum well outside this window and fail the comparison; a several
-mm defocus is well beyond this system's depth of focus.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

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


def _rms_spot_radius(scene, num_rays=100_000, seed=1) -> float:
    result = scene.trace(num_rays=num_rays, seed=seed, backend=NumpyBackend(seed=seed))
    irr = result.detectors["D"]
    xx, yy = np.meshgrid(irr.x_coords, irr.y_coords)
    w = irr.irradiance
    total = w.sum()
    cx = (xx * w).sum() / total
    cy = (yy * w).sum() / total
    return float(np.sqrt((((xx - cx) ** 2 + (yy - cy) ** 2) * w).sum() / total))


def _scene_with_detector_at(r1, r2, thickness, material, z_det):
    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=2.0),
    )
    scene.add_lens(
        "L",
        CoordinateSystem(z=0.0),
        LensConfig(
            r1=r1,
            r2=r2,
            thickness=thickness,
            material=material,
            front_aperture_radius=10.0,
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=z_det),
        IrradianceDetectorConfig(
            width=4, height=4, num_pixels_x=64, num_pixels_y=64, splat="hard"
        ),
    )
    return scene


def test_focal_spot_is_sharpest_near_paraxial_prediction():
    r1, r2, thickness, material = 100.0, -100.0, 1.0, "N-BK7"
    n = float(np.asarray(Material(material).n(0.55)).ravel()[0])
    f_thin = 1.0 / ((n - 1.0) * (1.0 / r1 - 1.0 / r2))
    z_focus = thickness / 2.0 + f_thin

    rms_at_focus = _rms_spot_radius(
        _scene_with_detector_at(r1, r2, thickness, material, z_focus)
    )
    rms_defocused_near = _rms_spot_radius(
        _scene_with_detector_at(r1, r2, thickness, material, z_focus - 5.0)
    )
    rms_defocused_far = _rms_spot_radius(
        _scene_with_detector_at(r1, r2, thickness, material, z_focus + 5.0)
    )

    assert rms_at_focus < 0.75 * rms_defocused_near
    assert rms_at_focus < 0.75 * rms_defocused_far
