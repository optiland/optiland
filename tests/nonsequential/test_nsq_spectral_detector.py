"""Tests for the NSQ SpectralDetector.

Covers the wavelength binning contract (bin edges are in µm, matching
``Spectrum`` and ``rays.wavelength``), flux conservation across bins, and the
nanometre guard.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    NSQScene,
    SpectralDetectorConfig,
    Spectrum,
)
from optiland.nonsequential.detectors.spectral import SpectralDetector

RGB_WAVELENGTHS = np.array([0.45, 0.55, 0.65])
RGB_WEIGHTS = np.array([1.0, 1.5, 1.0])


def _rgb_scene(wl_min: float, wl_max: float, num_bins: int) -> NSQScene:
    """Collimated RGB beam straight onto a spectral detector, no optics."""
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-10.0),
        CollimatedSourceConfig(
            spectrum=Spectrum(wavelengths=RGB_WAVELENGTHS, weights=RGB_WEIGHTS),
            total_flux=1.0,
            aperture_radius=2.0,
        ),
    )
    scene.add_detector(
        "SD",
        CoordinateSystem(z=10.0),
        SpectralDetectorConfig(
            width=10.0,
            height=10.0,
            num_pixels_x=8,
            num_pixels_y=8,
            wl_min=wl_min,
            wl_max=wl_max,
            num_bins=num_bins,
        ),
    )
    return scene


def test_flux_lands_in_the_bin_matching_its_wavelength():
    """Each source line deposits its flux in the bin that spans it."""
    result = _rgb_scene(0.4, 0.7, 30).trace(num_rays=6_000, seed=42)
    spectral = result.detectors["SD"]

    per_bin = spectral.irradiance.sum(axis=(0, 1))
    occupied = np.flatnonzero(per_bin)

    # 0.4-0.7 µm in 30 bins is 0.01 µm per bin: 0.45 -> bin 5, 0.55 -> 15,
    # 0.65 -> 25.
    assert occupied.tolist() == [5, 15, 25]
    # Each source line sits on a bin edge, so its bin centre lands half a bin
    # width (0.005 µm) above it.
    bin_width = 0.3 / 30
    for idx, wl in zip(occupied, RGB_WAVELENGTHS, strict=True):
        assert spectral.wavelengths[idx] == pytest.approx(wl + bin_width / 2)


def test_relative_bin_flux_follows_spectrum_weights():
    """Bin flux is proportional to the spectrum weights it was drawn from."""
    result = _rgb_scene(0.4, 0.7, 30).trace(num_rays=40_000, seed=7)
    per_bin = result.detectors["SD"].irradiance.sum(axis=(0, 1))
    occupied = per_bin[np.flatnonzero(per_bin)]

    expected = RGB_WEIGHTS / RGB_WEIGHTS.sum()
    np.testing.assert_allclose(occupied / occupied.sum(), expected, atol=0.01)


def test_binned_flux_sums_to_total_flux():
    """Summing over the wavelength axis recovers the broadband total."""
    result = _rgb_scene(0.4, 0.7, 30).trace(num_rays=6_000, seed=42)
    spectral = result.detectors["SD"]

    pixel_area = (10.0 / 8) * (10.0 / 8)
    assert spectral.irradiance.sum() * pixel_area == pytest.approx(
        spectral.total_flux, rel=1e-9
    )
    # Nothing blocks the beam, so every launched watt is detected.
    assert spectral.total_flux == pytest.approx(1.0, rel=1e-6)


def test_out_of_range_wavelengths_clip_into_the_end_bins():
    """A range narrower than the spectrum clips rather than dropping flux."""
    result = _rgb_scene(0.5, 0.6, 10).trace(num_rays=6_000, seed=42)
    spectral = result.detectors["SD"]

    per_bin = spectral.irradiance.sum(axis=(0, 1))
    assert per_bin[0] > 0.0  # 0.45 µm clipped up into the first bin
    assert per_bin[-1] > 0.0  # 0.65 µm clipped down into the last bin
    assert spectral.total_flux == pytest.approx(1.0, rel=1e-6)


def test_nanometre_bin_edges_are_rejected():
    """Bin edges in nm would silently collapse into bin 0, so they raise."""
    with pytest.raises(ValueError, match="must be in µm"):
        SpectralDetector(
            cs=CoordinateSystem(z=0.0),
            width=10.0,
            height=10.0,
            num_pixels_x=4,
            num_pixels_y=4,
            wavelength_bins=np.linspace(400.0, 700.0, 11),
        )


def test_spectral_detector_config_defaults_span_the_visible_in_um():
    config = SpectralDetectorConfig(width=1.0, height=1.0)
    assert config.wl_min == pytest.approx(0.4)
    assert config.wl_max == pytest.approx(0.7)
