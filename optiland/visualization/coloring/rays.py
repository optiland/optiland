"""Ray Coloring Module

This module contains the strategies that choose the color of each drawn ray, and
the data handed to them.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import to_rgba

if TYPE_CHECKING:
    from matplotlib.typing import ColorType

    from optiland.optic import Optic
    from optiland.visualization.coloring.scale import ColorScale


@dataclass(frozen=True)
class RayBundleRef:
    """Identifies one traced ray bundle.

    Attributes:
        field_index (int): Index among the drawn fields.
        field (tuple[float, float]): Field coordinates.
        wavelength_index (int): Index among the drawn wavelengths.
        wavelength (float): Wavelength in micrometers.
        num_fields (int): Number of drawn fields.
        num_wavelengths (int): Number of drawn wavelengths.
        is_reference (bool): True for chief/marginal reference rays.
    """

    field_index: int
    field: tuple[float, float]
    wavelength_index: int
    wavelength: float
    num_fields: int
    num_wavelengths: int
    is_reference: bool


@dataclass(frozen=True)
class TracedRays:
    """NumPy copies of the bundle just traced.

    Attributes:
        x, y, z (np.ndarray): Global coordinates, shape ``(num_surfaces, num_rays)``.
        intensity (np.ndarray): Intensity at each surface, same shape.
        opd (np.ndarray | None): Optical path difference recorded at each
            surface, same shape, or None if the trace did not record it.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    intensity: np.ndarray
    opd: np.ndarray | None = None

    @property
    def num_rays(self) -> int:
        return self.z.shape[1]


class RayColorizer(ABC):
    """Strategy that picks the color of each ray in a bundle."""

    def prepare(self, optic: Optic) -> None:  # noqa: B027
        """Hook called once per draw, before any :meth:`ray_colors` call.

        Args:
            optic (Optic): The system being drawn.
        """

    @abstractmethod
    def ray_colors(self, bundle: RayBundleRef, rays: TracedRays) -> np.ndarray | None:
        """Return per-ray colors for a bundle.

        Args:
            bundle (RayBundleRef): The bundle that was traced.
            rays (TracedRays): Its traced rays.

        Returns:
            np.ndarray | None: RGBA array of shape ``(num_rays, 4)``, or None to
            keep the default bundle color. A row of NaN keeps the default
            color for that ray only.
        """

    @property
    def scale(self) -> ColorScale | None:
        """ColorScale | None: Resolved scale to show as a colorbar, if any."""
        return None


class UniformRayColorizer(RayColorizer):
    """Gives every ray the same color.

    Args:
        color (ColorType): Anything ``matplotlib.colors.to_rgba`` accepts.
    """

    def __init__(self, color: ColorType) -> None:
        self._rgba = to_rgba(color)

    def ray_colors(self, bundle: RayBundleRef, rays: TracedRays) -> np.ndarray | None:
        return np.tile(self._rgba, (rays.num_rays, 1))
