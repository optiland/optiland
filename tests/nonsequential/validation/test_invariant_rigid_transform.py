"""Rigid invariance: a global rotation+translation of the whole
scene leaves detector results unchanged, in the detector's own frame.

Every component's ``CoordinateSystem`` is built relative to a shared
``reference_cs`` (a single top-level "world" transform); the physics must
not care that the world frame itself is rotated and translated -- only
relative geometry matters. Detector maps are compared in the detector's own
local (x, y) coordinates, which is what "in the detector's own frame" means:
the map itself, not its position in some external frame, must be identical.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend


def _build_scene(world_cs: CoordinateSystem | None) -> NSQScene:
    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()

    def cs(z: float, **kwargs) -> CoordinateSystem:
        return CoordinateSystem(z=z, reference_cs=world_cs, **kwargs)

    scene.add_source(
        "S",
        cs(0.0),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=8.0),
    )
    scene.add_lens(
        "L1",
        cs(50.0),
        LensConfig(
            r1=60.0,
            r2=-60.0,
            thickness=6.0,
            material="N-BK7",
            front_aperture_radius=12.0,
        ),
    )
    scene.add_detector(
        "D1",
        cs(200.0),
        IrradianceDetectorConfig(
            width=40, height=40, num_pixels_x=32, num_pixels_y=32, splat="hard"
        ),
    )
    return scene


def test_rigid_transform_leaves_detector_map_unchanged():
    baseline = _build_scene(None)
    r0 = baseline.trace(num_rays=20_000, seed=1, backend=NumpyBackend(seed=1))

    world = CoordinateSystem(x=37.0, y=-12.0, z=500.0, rx=0.3, ry=-0.2, rz=0.7)
    transformed = _build_scene(world)
    r1 = transformed.trace(num_rays=20_000, seed=1, backend=NumpyBackend(seed=1))

    d0 = r0.detectors["D1"].irradiance
    d1 = r1.detectors["D1"].irradiance
    np.testing.assert_allclose(d0, d1, rtol=1e-9, atol=1e-15)
    assert r0.total_flux_detected == r1.total_flux_detected
