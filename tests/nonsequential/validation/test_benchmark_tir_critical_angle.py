"""Total internal reflection: sharp transition at the critical angle.

Same isolated single-interface rig as ``test_benchmark_fresnel_sweep.py``,
but glass -> vacuum, sweeping through the critical angle
``theta_c = asin(n2/n1)``. Below theta_c, transmittance follows the ordinary
Fresnel formula; at and above theta_c it must be *exactly* zero -- TIR is a
hard cutoff, not a soft one, so this checks the transition is sharp rather
than smeared out by (for example) an angle-averaging or interpolation bug.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

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
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry


def _transmittance_at(theta_deg: float, glass) -> float:
    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=3.0),
    )
    rx = np.radians(theta_deg)
    comp = RefractiveComponent(
        CoordinateSystem(z=10.0, rx=rx),
        PlaneGeometry(),
        material_front=glass,
        material_back=VACUUM,
    )
    scene.add_component("IF", comp)
    scene.add_detector(
        "D",
        CoordinateSystem(z=10.5, rx=rx),
        IrradianceDetectorConfig(
            width=200, height=200, num_pixels_x=4, num_pixels_y=4, splat="hard"
        ),
    )
    result = scene.trace(
        num_rays=100_000, seed=1, max_depth=2, backend=NumpyBackend(seed=1)
    )
    return result.total_flux_detected / result.total_flux_in


def test_transmittance_vanishes_above_critical_angle():
    glass = NSQMaterial.from_glass("N-BK7")
    n = float(np.asarray(glass.n(0.55)).ravel()[0])
    theta_c = np.degrees(np.arcsin(1.0 / n))

    t_below = _transmittance_at(theta_c - 5.0, glass)
    t_at_edge = _transmittance_at(theta_c - 0.5, glass)
    t_above = _transmittance_at(theta_c + 0.5, glass)
    t_well_above = _transmittance_at(theta_c + 10.0, glass)

    assert t_below > 0.3  # well below theta_c: mostly transmits
    assert 0.0 < t_at_edge < t_below  # approaching theta_c: T shrinking
    assert t_above == 0.0  # at/above theta_c: TIR, exactly
    assert t_well_above == 0.0
