"""Config dataclasses for NSQ detectors.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class IrradianceDetectorConfig:
    """Configuration for an IrradianceDetector.

    Attributes:
        width: Detector width [mm].
        height: Detector height [mm].
        num_pixels_x: Number of pixels along x.
        num_pixels_y: Number of pixels along y.
        splat: Splatting mode — 'bilinear', 'gaussian', or 'hard'.
        splat_sigma: Gaussian splat sigma in pixels (used when splat='gaussian').
        absorb: Whether a hit terminates the ray. False makes the detector
            transmissive: the hit is recorded and the ray continues on its
            unchanged direction, enabling mid-system beam sampling.
    """

    width: float
    height: float
    num_pixels_x: int = 256
    num_pixels_y: int = 256
    splat: Literal["bilinear", "gaussian", "hard"] = "bilinear"
    splat_sigma: float = 0.5
    absorb: bool = True


@dataclass
class SpectralDetectorConfig:
    """Configuration for a SpectralDetector.

    Attributes:
        width: Detector width [mm].
        height: Detector height [mm].
        num_pixels_x: Number of pixels along x.
        num_pixels_y: Number of pixels along y.
        wl_min: Minimum wavelength for spectral binning [µm].
        wl_max: Maximum wavelength for spectral binning [µm].
        num_bins: Number of wavelength bins.
        splat: Spatial (x, y) splatting mode — 'bilinear', 'gaussian', or
            'hard'. The wavelength bin is always hard-assigned.
        splat_sigma: Gaussian splat sigma in pixels (used when
            ``splat='gaussian'``).
        absorb: Whether a hit terminates the ray.

    Note:
        Wavelengths are in **micrometres**, matching ``Spectrum`` and every
        other wavelength in Optiland. Visible light spans 0.4-0.7 µm, so a
        detector spanning the visible is ``wl_min=0.4, wl_max=0.7``.
    """

    width: float
    height: float
    num_pixels_x: int = 256
    num_pixels_y: int = 256
    wl_min: float = 0.4
    wl_max: float = 0.7
    num_bins: int = 100
    splat: Literal["bilinear", "gaussian", "hard"] = "bilinear"
    splat_sigma: float = 0.5
    absorb: bool = True


@dataclass
class FarFieldDetectorConfig:
    """Configuration for a FarFieldDetector.

    Attributes:
        num_theta: Number of polar angle bins.
        num_phi: Number of azimuthal angle bins.
        absorb: Whether a hit terminates the ray.
    """

    num_theta: int = 90
    num_phi: int = 360
    absorb: bool = True


@dataclass
class RayDatabaseConfig:
    """Configuration for a RayDatabaseDetector.

    Attributes:
        width: Detector width [mm].
        height: Detector height [mm].
        max_rays: Maximum number of rays to store (0 = unlimited).
        absorb: Whether a hit terminates the ray.
    """

    width: float
    height: float
    max_rays: int = 0
    absorb: bool = True
