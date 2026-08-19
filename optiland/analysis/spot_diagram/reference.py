"""Spot Diagram Reference Strategies

This module provides configurable reference point strategies for centering
spot diagram data. The center of calculation can be either the chief ray
intersection or the geometric centroid of the traced rays.

Kramer Harrison, 2025
"""

from __future__ import annotations

import abc
from enum import StrEnum
from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from optiland._types import BEArray
    from optiland.optic import Optic

    from .core import SpotData


class SpotReferenceType(StrEnum):
    """Defines the available reference point types for spot centering."""

    CHIEF_RAY = "chief_ray"
    CENTROID = "centroid"


class SpotReferenceStrategy(abc.ABC):
    """Abstract base class for spot diagram reference strategies."""

    @abc.abstractmethod
    def get_centers(
        self,
        data: list[list[SpotData]],
        ref_wavelength_index: int,
        optic: Optic,
        fields: list[tuple[float, float]],
        wavelength: float,
        coordinates: str,
    ) -> list[tuple[BEArray, BEArray]]:
        """Computes the (x, y) center for each field.

        Args:
            data: Nested list of spot data, ordered [field][wavelength].
            ref_wavelength_index: Index of the reference wavelength.
            optic: The optical system being analyzed.
            fields: List of (Hx, Hy) field coordinates.
            wavelength: Reference wavelength value in micrometers.
            coordinates: Coordinate system ('global' or 'local').

        Returns:
            A list of (x, y) center tuples, one per field.
        """


class CentroidReference(SpotReferenceStrategy):
    """Centers spots using the geometric centroid of rays at the reference
    wavelength."""

    def get_centers(
        self,
        data: list[list[SpotData]],
        ref_wavelength_index: int,
        optic: Optic,
        fields: list[tuple[float, float]],
        wavelength: float,
        coordinates: str,
    ) -> list[tuple[BEArray, BEArray]]:
        """Computes centroids from the mean of ray intersections."""
        return [
            (
                be.mean(field_data[ref_wavelength_index].x),
                be.mean(field_data[ref_wavelength_index].y),
            )
            for field_data in data
        ]


class ChiefRayReference(SpotReferenceStrategy):
    """Centers spots using the chief ray intersection at the reference
    wavelength."""

    def get_centers(
        self,
        data: list[list[SpotData]],
        ref_wavelength_index: int,
        optic: Optic,
        fields: list[tuple[float, float]],
        wavelength: float,
        coordinates: str,
    ) -> list[tuple[BEArray, BEArray]]:
        """Computes centers from chief ray (Px=0, Py=0) intersections."""
        from optiland.visualization.system.utils import transform

        centers = []
        for H_x, H_y in fields:
            ray = optic.trace_generic(Hx=H_x, Hy=H_y, Px=0, Py=0, wavelength=wavelength)
            if coordinates == "local":
                x, y, _ = transform(
                    ray.x, ray.y, ray.z, optic.image_surface, is_global=True
                )
                centers.append((x.ravel()[0], y.ravel()[0]))
            else:
                centers.append((ray.x.ravel()[0], ray.y.ravel()[0]))
        return centers


def create_reference_strategy(
    reference: str | SpotReferenceType,
) -> SpotReferenceStrategy:
    """Factory function to create a reference strategy from a type string.

    Args:
        reference: The reference type, either a SpotReferenceType enum value
            or a string ('chief_ray' or 'centroid').

    Returns:
        The corresponding SpotReferenceStrategy instance.

    Raises:
        ValueError: If the reference type is not recognized.
    """
    ref = SpotReferenceType(reference)
    strategies = {
        SpotReferenceType.CHIEF_RAY: ChiefRayReference,
        SpotReferenceType.CENTROID: CentroidReference,
    }
    return strategies[ref]()
