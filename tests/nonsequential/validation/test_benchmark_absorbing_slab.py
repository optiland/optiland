"""Absorbing slab: Beer-Lambert transmittance, sweeping k and L.

``exp(-4*pi*k*L/lambda)`` for the flux surviving a path length L through a
medium with extinction coefficient k, at wavelength lambda -- swept over
several (k, L) pairs, independent of ``test_nsq_absorption.py``'s single
-case check.

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


class _ConstantMaterial:
    """Bare stand-in exposing only .n()/.k() for a fixed (n, k) pair."""

    def __init__(self, n: float, k: float) -> None:
        self._n = n
        self._k = k
        self.name = f"const_n{n}_k{k}"

    def n(self, wavelength_um):
        return self._n

    def k(self, wavelength_um):
        return self._k


@pytest.mark.parametrize("k", [1e-4, 5e-4, 2e-3])
@pytest.mark.parametrize("thickness", [1.0, 5.0])
def test_beer_lambert_matches_closed_form(k, thickness):
    wavelength_um = 0.55
    # Index barely above 1: Fresnel reflectance is ~6e-8 (negligible at any
    # ray count this test uses), so transmittance isolates the Beer-Lambert
    # factor alone rather than needing the full multi-bounce Fresnel-window
    # formula on top of it.
    n_index = 1.001
    material = NSQMaterial(optiland_material=_ConstantMaterial(n_index, k))

    spec = Spectrum.monochromatic(wavelength_um)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=3.0),
    )
    # Two parallel interfaces bounding a slab of exactly `thickness` mm of
    # the absorbing medium; index-matched-ish choice keeps most flux
    # transmitting so the Beer-Lambert signal isn't swamped by Fresnel loss.
    entry = RefractiveComponent(
        CoordinateSystem(z=10.0),
        PlaneGeometry(),
        material_front=VACUUM,
        material_back=material,
    )
    exit_ = RefractiveComponent(
        CoordinateSystem(z=10.0 + thickness),
        PlaneGeometry(),
        material_front=material,
        material_back=VACUUM,
    )
    scene.add_component("entry", entry)
    scene.add_component("exit", exit_)
    scene.add_detector(
        "D",
        CoordinateSystem(z=10.0 + thickness + 0.5),
        IrradianceDetectorConfig(
            width=200, height=200, num_pixels_x=4, num_pixels_y=4, splat="hard"
        ),
    )
    result = scene.trace(
        num_rays=200_000, seed=1, max_depth=4, backend=NumpyBackend(seed=1)
    )

    t_theory = np.exp(-4.0 * np.pi * k * thickness * 1e3 / wavelength_um)

    t_sim = result.total_flux_detected / result.total_flux_in
    assert t_sim == pytest.approx(t_theory, rel=0.03)
