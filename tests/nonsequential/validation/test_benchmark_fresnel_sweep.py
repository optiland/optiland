"""Single interface, swept incidence angle: unpolarized Fresnel R.

A bare vacuum/glass ``RefractiveComponent`` built directly on an infinite
``PlaneGeometry`` (no lens barrel, no second surface) isolates exactly one
Fresnel interface. Tilting the interface's coordinate system sweeps the
angle of incidence of an on-axis collimated beam. The detector sits at a
small standoff *parallel to the tilted interface* -- close enough that the
transmitted ray's path length in glass before detection is negligible (a
detector far away would pick up real, non-negligible Beer-Lambert bulk
absorption over that path, which is not what this benchmark is measuring).

NSQ tracks unpolarized reflectance only (no polarization state -- see
``optiland.nonsequential.components.refractive``), so this checks
``R_unpol(theta) = 0.5 * (R_s(theta) + R_p(theta))`` against the standard
Fresnel equations; R_s and R_p are not independently observable from the
engine and so are not separately validated.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

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


def _fresnel_r_unpol(theta_deg: float, n1: float, n2: float) -> float:
    theta = np.radians(theta_deg)
    sin_t2 = (n1 / n2 * np.sin(theta)) ** 2
    if sin_t2 >= 1.0:
        return 1.0
    cos_t = np.sqrt(1.0 - sin_t2)
    cos_i = np.cos(theta)
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return 0.5 * (rs**2 + rp**2)


@pytest.mark.parametrize("theta_deg", [0.0, 15.0, 30.0, 45.0, 60.0])
def test_transmittance_matches_fresnel_unpolarized(theta_deg):
    glass = NSQMaterial.from_glass("N-BK7")
    n = float(np.asarray(glass.n(0.55)).ravel()[0])
    r_theory = _fresnel_r_unpol(theta_deg, 1.0, n)
    t_theory = 1.0 - r_theory

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
        material_front=VACUUM,
        material_back=glass,
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
        num_rays=200_000, seed=1, max_depth=2, backend=NumpyBackend(seed=1)
    )
    t_sim = result.total_flux_detected / result.total_flux_in

    assert t_sim == pytest.approx(t_theory, abs=0.01)
