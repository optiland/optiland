"""Config dataclasses for NSQ sources.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.sources.base import Spectrum


@dataclass
class PointSourceConfig:
    """Configuration for a PointSource.

    Attributes:
        spectrum: Wavelength distribution.
        total_flux: Total emitted flux [W].
        half_angle_deg: Half-angle of the emission cone [deg].
            90 = hemisphere, 180 = full sphere (isotropic).
    """

    spectrum: Spectrum
    total_flux: float = 1.0
    half_angle_deg: float = 90.0


@dataclass
class CollimatedSourceConfig:
    """Configuration for a CollimatedSource.

    Attributes:
        spectrum: Wavelength distribution.
        total_flux: Total emitted flux [W].
        aperture_radius: Beam semi-diameter [mm].
        profile: Spatial profile -- ``'tophat'`` or ``'gaussian'``.
    """

    spectrum: Spectrum
    total_flux: float = 1.0
    aperture_radius: float = 1.0
    profile: str = "tophat"


@dataclass
class ExtendedSourceConfig:
    """Configuration for an ExtendedSource.

    Attributes:
        spectrum: Wavelength distribution.
        total_flux: Total emitted flux [W].
        width: Source width [mm].
        height: Source height [mm].
        half_angle_deg: Half-angle of the emission cone [deg].
    """

    spectrum: Spectrum
    total_flux: float = 1.0
    width: float = 1.0
    height: float = 1.0
    half_angle_deg: float = 90.0
