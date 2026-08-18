"""Config dataclasses for NSQ sources.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.sources.base import Spectrum


@dataclass
class PointSourceConfig:
    """Configuration for a PointSource.

    Attributes:
        spectrum: Wavelength distribution.
        total_flux: Total emitted flux [W].
        half_angle_deg: Half-angle of the emission cone [deg].
            90 = hemisphere, 180 = full sphere (isotropic).
        medium: Medium the source is embedded in (default: vacuum).
    """

    spectrum: Spectrum
    total_flux: float = 1.0
    half_angle_deg: float = 90.0
    medium: NSQMaterial | None = field(default=None)


@dataclass
class CollimatedSourceConfig:
    """Configuration for a CollimatedSource.

    Attributes:
        spectrum: Wavelength distribution.
        total_flux: Total emitted flux [W].
        aperture_radius: Beam semi-diameter [mm].
        profile: Spatial profile -- ``'tophat'`` or ``'gaussian'``.
        gaussian_sigma: Gaussian sigma [mm]. Defaults to aperture_radius / 2.
        medium: Medium the source is embedded in (default: vacuum).
    """

    spectrum: Spectrum
    total_flux: float = 1.0
    aperture_radius: float = 1.0
    profile: str = "tophat"
    gaussian_sigma: float | None = None
    medium: NSQMaterial | None = field(default=None)


@dataclass
class ExtendedSourceConfig:
    """Configuration for an ExtendedSource.

    Attributes:
        spectrum: Wavelength distribution.
        total_flux: Total emitted flux [W].
        width: Source width [mm].
        height: Source height [mm].
        aperture_radius: Circular aperture radius [mm]. If set, overrides
            width/height for a circular source.
        half_angle_deg: Half-angle of the emission cone [deg]. Below 90 the
            rays are distributed uniformly within the cone; at 90 or above
            they are cosine-weighted over the full hemisphere (Lambertian),
            and values above 90 behave the same as 90.
        medium: Medium the source is embedded in (default: vacuum).
    """

    spectrum: Spectrum
    total_flux: float = 1.0
    width: float = 1.0
    height: float = 1.0
    aperture_radius: float | None = None
    half_angle_deg: float = 90.0
    medium: NSQMaterial | None = field(default=None)
