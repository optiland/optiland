"""Config dataclasses for NSQ detectors.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IrradianceDetectorConfig:
    """Configuration for an IrradianceDetector.

    Attributes:
        width: Detector width [mm].
        height: Detector height [mm].
        n_pixels_x: Number of pixels along x.
        n_pixels_y: Number of pixels along y.
    """

    width: float
    height: float
    n_pixels_x: int = 256
    n_pixels_y: int = 256


@dataclass
class SpectralDetectorConfig:
    """Configuration for a SpectralDetector.

    Attributes:
        width: Detector width [mm].
        height: Detector height [mm].
        wl_min: Minimum wavelength for spectral binning [nm].
        wl_max: Maximum wavelength for spectral binning [nm].
        n_bins: Number of wavelength bins.
    """

    width: float
    height: float
    wl_min: float = 400.0
    wl_max: float = 700.0
    n_bins: int = 100


@dataclass
class FarFieldDetectorConfig:
    """Configuration for a FarFieldDetector.

    Attributes:
        n_theta: Number of polar angle bins.
        n_phi: Number of azimuthal angle bins.
    """

    n_theta: int = 90
    n_phi: int = 360


@dataclass
class RayDatabaseConfig:
    """Configuration for a RayDatabaseDetector.

    Attributes:
        width: Detector width [mm].
        height: Detector height [mm].
        max_rays: Maximum number of rays to store (0 = unlimited).
    """

    width: float
    height: float
    max_rays: int = 0
