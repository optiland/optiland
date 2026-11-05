"""TabulatedBSDF: angle of incidence and flux weights.

The scattered flux of a tabulated BSDF must equal the table's total
integrated scatter, TIS(theta_i) = 2 pi Int b(theta_i, theta_s) cos sin
dtheta_s, for reflective and transmissive lobes alike, also for narrow lobes
(where pi * b >> 1), and theta_i must be the actual angle of incidence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    FarFieldDetectorConfig,
    FinitePlaneGeometry,
    NSQRng,
    NSQScene,
    RefractiveComponent,
    Spectrum,
    TabulatedBSDF,
)

THETA_S = np.arange(0.0, 90.0 + 1e-9, 0.25)


def _gaussian_rows(fwhm_deg: float, tis_per_row: dict[float, float]) -> np.ndarray:
    """Rows (theta_i, theta_s, b) of a Gaussian lobe scaled to a TIS per row."""
    sig = math.radians(fwhm_deg) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    th = np.radians(THETA_S)
    shape = np.exp(-(th**2) / (2.0 * sig**2))
    unit = 2.0 * math.pi * np.trapezoid(shape * np.cos(th) * np.sin(th), th)
    rows = [
        (ti, ts, v)
        for ti, tis in tis_per_row.items()
        for ts, v in zip(THETA_S, tis / unit * shape, strict=True)
    ]
    return np.array(rows)


def _write(tmp_path, rows: np.ndarray, name: str = "bsdf.csv"):
    path = tmp_path / name
    np.savetxt(path, rows, delimiter=",")
    return path


def _sample(bsdf, theta_i_deg: float, n_rays: int = 200_000, seed: int = 0):
    """Sample with the incoming ray at theta_i; normals face the incoming ray."""
    ti = math.radians(theta_i_deg)
    incident = np.tile([math.sin(ti), 0.0, -math.cos(ti)], (n_rays, 1))
    normal = np.tile([0.0, 0.0, 1.0], (n_rays, 1))
    dirs, weights, transmitted = bsdf.sample(
        n_rays,
        incident,
        normal,
        np.full(n_rays, 0.55),
        NSQRng(seed),
        np.arange(n_rays),
        np.zeros(n_rays, dtype=np.int32),
    )
    return np.asarray(dirs), np.asarray(weights), np.asarray(transmitted)


@pytest.mark.parametrize("fwhm", [10.0, 20.0, 60.0])
@pytest.mark.parametrize("transmissive_fraction", [0.0, 1.0])
def test_scattered_flux_equals_table_tis(tmp_path, fwhm, transmissive_fraction):
    """Mean weight = TIS for narrow and wide lobes, in both hemispheres.

    Before the fix: weights were clip(pi * b, 0, 1) on cosine-sampled
    directions (about 0.06 W instead of 0.9 for FWHM 10 deg), and every
    transmitted draw was evaluated at theta_i = 90 deg.
    """
    tis = 0.9
    rows = _gaussian_rows(fwhm, {ti: tis for ti in range(0, 81, 10)})
    bsdf = TabulatedBSDF(_write(tmp_path, rows), transmissive_fraction)
    dirs, weights, transmitted = _sample(bsdf, 0.0)

    assert np.mean(weights) == pytest.approx(tis, rel=2e-3)
    assert np.all(transmitted == bool(transmissive_fraction))
    # the incident ray travels along -z: transmitted rays continue along -z,
    # reflected rays go back along +z
    expected_side = -1.0 if transmissive_fraction else 1.0
    assert np.all(np.sign(dirs[:, 2]) == expected_side)


@pytest.mark.parametrize("fwhm", [10.0, 60.0])
def test_angular_distribution_follows_the_table(tmp_path, fwhm):
    """Weighted <cos theta_s> matches Int b cos^2 sin / Int b cos sin."""
    rows = _gaussian_rows(fwhm, {0.0: 0.9, 80.0: 0.9})
    bsdf = TabulatedBSDF(_write(tmp_path, rows), transmissive_fraction=1.0)
    dirs, weights, _ = _sample(bsdf, 0.0)
    measured = np.sum(weights * np.abs(dirs[:, 2])) / np.sum(weights)

    th = np.radians(THETA_S)
    b = rows[rows[:, 0] == 0.0, 2]
    expected = np.trapezoid(b * np.cos(th) ** 2 * np.sin(th), th) / np.trapezoid(
        b * np.cos(th) * np.sin(th), th
    )
    assert measured == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize("transmissive_fraction", [0.0, 1.0])
def test_theta_i_is_the_angle_of_incidence(tmp_path, transmissive_fraction):
    """Rows depend on theta_i: TIS 0.9 at 0 deg, 0.5 at 60 deg (linear between)."""
    rows = _gaussian_rows(30.0, {0.0: 0.9, 60.0: 0.5})
    bsdf = TabulatedBSDF(_write(tmp_path, rows), transmissive_fraction)

    for theta_i, expected in ((0.0, 0.9), (30.0, 0.7), (60.0, 0.5)):
        _, weights, _ = _sample(bsdf, theta_i, n_rays=100_000)
        assert np.mean(weights) == pytest.approx(expected, rel=5e-3), theta_i


def test_incidence_outside_the_table_scatters_nothing(tmp_path):
    """Outside the tabulated theta_i range the weight is 0 (interpolator fill)."""
    rows = _gaussian_rows(30.0, {0.0: 0.9, 40.0: 0.9})
    bsdf = TabulatedBSDF(_write(tmp_path, rows), transmissive_fraction=1.0)
    _, weights, _ = _sample(bsdf, 60.0, n_rays=1_000)
    assert np.all(weights == 0.0)


def test_transmissive_diffuser_scene_flux(tmp_path, set_test_backend):
    """Collimated beam on a transmissive diffuser: far-field flux = TIS."""
    tis = 0.9
    rows = _gaussian_rows(20.0, {ti: tis for ti in range(0, 81, 10)})
    bsdf = TabulatedBSDF(_write(tmp_path, rows), transmissive_fraction=1.0)

    scene = NSQScene()
    scene.add_source(
        "Beam",
        CoordinateSystem(z=-5.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=2.0
        ),
    )
    scene.add_component(
        "Diffuser",
        RefractiveComponent(
            CoordinateSystem(),
            FinitePlaneGeometry(aperture_radius=10.0),
            VACUUM,
            VACUUM,
            bsdf=bsdf,
            name="Diffuser",
            scatter_fraction=1.0,
        ),
    )
    scene.add_detector(
        "FarField",
        CoordinateSystem(z=0.5),
        FarFieldDetectorConfig(num_theta=90, num_phi=36),
    )
    result = scene.trace(num_rays=50_000, seed=1)
    flux = float(np.asarray(result.detectors["FarField"].total_flux))
    assert flux == pytest.approx(tis, rel=5e-3)
