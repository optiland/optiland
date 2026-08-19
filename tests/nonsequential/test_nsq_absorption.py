"""Beer-Lambert bulk absorption tests for Non-Sequential Raytracing (D-13).

Covers NSQMaterial.k(), rays.k_current tracking across a refractive
interface, and the trace-loop attenuation itself -- checked against the
same alpha = 4*pi*k/wavelength_um formula used by the sequential engine's
optiland.propagation.homogeneous.HomogeneousPropagation, so NSQ and the
sequential tracer attenuate a glass path by the same amount.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optiland.coatings import SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.materials.ideal import IdealMaterial
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

GREEN = 0.55


def _tinted(n: float = 1.5, k: float = 1e-6) -> NSQMaterial:
    return NSQMaterial(optiland_material=IdealMaterial(n=n, k=k))


class TestNSQMaterialK:
    def test_vacuum_k_is_zero(self):
        wl = np.array([0.5, 0.55, 0.6])
        k = np.asarray(VACUUM.k(wl))
        np.testing.assert_allclose(k, 0.0)

    def test_scalar_wavelength_vacuum_k(self):
        assert float(np.asarray(VACUUM.k(0.55))) == 0.0

    def test_catalog_material_delegates_to_underlying_k(self):
        mat = _tinted(k=0.002)
        k = float(np.asarray(mat.k(0.55)))
        assert k == pytest.approx(0.002)


class TestKCurrentTracking:
    def test_k_current_updates_on_transmission(self):
        """A ray refracting into a tinted material picks up its k."""
        from optiland.nonsequential.ir.bsdf_ir import BsdfIR
        from optiland.nonsequential.ray_bundle import NSQRayBundle

        tinted = _tinted(k=0.003)
        comp = RefractiveComponent(
            cs=CoordinateSystem(z=0.0),
            geometry=PlaneGeometry(),
            material_front=VACUUM,
            material_back=tinted,
            coating=SimpleCoating(transmittance=1.0, reflectance=0.0),
            name="I",
        )
        n = 100
        rays = NSQRayBundle(
            x=np.zeros(n),
            y=np.zeros(n),
            z=np.full(n, -1.0),
            L=np.zeros(n),
            M=np.zeros(n),
            N=np.ones(n),
            flux=np.ones(n),
            wavelength=np.full(n, GREEN),
            n_current=np.ones(n),
            bounce=np.zeros(n, dtype=np.int32),
            alive=np.ones(n, dtype=bool),
            ray_id=np.arange(n, dtype=np.int64),
        )
        assert np.all(rays.k_current == 0.0)
        t = np.ones(n)
        normals = np.tile([0.0, 0.0, -1.0], (n, 1))
        n_geom = np.tile([0.0, 0.0, 1.0], (n, 1))
        hit_mask = np.ones(n, dtype=bool)
        from optiland.nonsequential.rng import NSQRng

        comp.interact(
            rays, t, normals, hit_mask, NSQRng(0), BsdfIR(kind="none"), n_geom
        )
        np.testing.assert_allclose(rays.k_current, 0.003)


def _slab_transmitted_flux(k: float, thickness_mm: float, *, num_rays=20_000) -> float:
    """Collimated beam through a lossless-Fresnel (coating R=0,T=1) tinted
    slab of the given thickness; only Beer-Lambert should attenuate it."""
    tinted = _tinted(k=k)
    lossless = SimpleCoating(transmittance=1.0, reflectance=0.0)

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
    scene.add_component(
        "entry",
        RefractiveComponent(
            cs=CoordinateSystem(z=0.0),
            geometry=PlaneGeometry(),
            material_front=VACUUM,
            material_back=tinted,
            coating=lossless,
            name="entry",
        ),
    )
    scene.add_component(
        "exit",
        RefractiveComponent(
            cs=CoordinateSystem(z=thickness_mm),
            geometry=PlaneGeometry(),
            material_front=tinted,
            material_back=VACUUM,
            coating=lossless,
            name="exit",
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=thickness_mm + 20.0),
        IrradianceDetectorConfig(width=10, height=10, num_pixels_x=8, num_pixels_y=8),
    )
    result = scene.trace(num_rays=num_rays, seed=11)
    return result.detectors["D"].total_flux, result


class TestBeerLambertAttenuation:
    def test_transmitted_flux_matches_beer_lambert(self):
        k = 1e-6
        thickness_mm = 10.0
        flux, _ = _slab_transmitted_flux(k, thickness_mm)

        alpha = 4.0 * math.pi * k / GREEN  # [1/um]
        expected_T = math.exp(-alpha * thickness_mm * 1e3)  # mm -> um
        assert 0.0 < expected_T < 1.0
        assert flux == pytest.approx(expected_T, abs=0.01)

    def test_thicker_slab_transmits_less(self):
        k = 5e-6
        thin, _ = _slab_transmitted_flux(k, 5.0)
        thick, _ = _slab_transmitted_flux(k, 20.0)
        assert thick < thin

    def test_zero_k_is_unattenuated(self):
        flux, _ = _slab_transmitted_flux(0.0, 10.0)
        assert flux == pytest.approx(1.0, abs=0.01)

    def test_flux_ledger_accounts_for_bulk_absorption(self):
        flux, result = _slab_transmitted_flux(1e-6, 10.0)
        assert result.total_flux_bulk_absorbed > 0.0
        balance = (
            result.total_flux_in
            - result.total_flux_detected
            - result.total_flux_absorbed
            - result.total_flux_bulk_absorbed
            - result.total_flux_escaped
            - result.total_flux_lost
        )
        assert abs(balance) / result.total_flux_in < 1e-9

    def test_vacuum_material_has_no_bulk_absorption(self):
        """Sanity check: an all-vacuum scene reports zero bulk absorption."""
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
        scene.add_detector(
            "D",
            CoordinateSystem(z=10.0),
            IrradianceDetectorConfig(
                width=10, height=10, num_pixels_x=8, num_pixels_y=8
            ),
        )
        result = scene.trace(num_rays=1_000, seed=1)
        assert result.total_flux_bulk_absorbed == 0.0
