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
        num_pixels_x: Number of pixels along x.
        num_pixels_y: Number of pixels along y.
    """

    width: float
    height: float
    num_pixels_x: int = 256
    num_pixels_y: int = 256


@dataclass
class SpectralDetectorConfig:
    """Configuration for a SpectralDetector.

    Attributes:
        width: Detector width [mm].
        height: Detector height [mm].
        num_pixels_x: Number of pixels along x.
        num_pixels_y: Number of pixels along y.
        wl_min: Minimum wavelength for spectral binning [nm].
        wl_max: Maximum wavelength for spectral binning [nm].
        num_bins: Number of wavelength bins.
    """

    width: float
    height: float
    num_pixels_x: int = 256
    num_pixels_y: int = 256
    wl_min: float = 400.0
    wl_max: float = 700.0
    num_bins: int = 100


@dataclass
class FarFieldDetectorConfig:
    """Configuration for a FarFieldDetector.

    Attributes:
        num_theta: Number of polar angle bins.
        num_phi: Number of azimuthal angle bins.
    """

    num_theta: int = 90
    num_phi: int = 360


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
