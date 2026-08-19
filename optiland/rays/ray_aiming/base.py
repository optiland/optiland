"""
Base Ray Aimer Module

This module defines the abstract base class for ray aiming algorithms.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland._types import ScalarOrArrayT
    from optiland.optic import Optic


class BaseRayAimer(ABC):
    """
    Abstract base class for ray aiming algorithms.

    This class defines the interface for calculating the initial ray coordinates
    and direction cosines required to aim rays at a specific pupil coordinate
    on the stop surface.

    Attributes:
        optic (Optic): The optical system to trace.
        kwargs (dict): Additional parameters for the aiming algorithm.
    """

    def __init__(self, optic: Optic, **kwargs: Any) -> None:
        """
        Initialize the BaseRayAimer.

        Args:
            optic (Optic): The optical system instance.
            **kwargs: Specific parameters for the aiming algorithm.
        """
        self.optic = optic
        self.kwargs = kwargs

    @abstractmethod
    def aim_rays(
        self,
        fields: tuple[ScalarOrArrayT, ScalarOrArrayT],
        wavelengths: ScalarOrArrayT,
        pupil_coords: tuple[ScalarOrArrayT, ScalarOrArrayT],
    ) -> tuple[
        ScalarOrArrayT,
        ScalarOrArrayT,
        ScalarOrArrayT,
        ScalarOrArrayT,
        ScalarOrArrayT,
        ScalarOrArrayT,
    ]:
        """
        Calculate ray starting coordinates and direction cosines.

        Args:
            fields: Normalized field coordinates (Hx, Hy).
            wavelengths: Wavelengths for the rays.
            pupil_coords: Normalized pupil coordinates (Px, Py).

        Returns:
            Tuple containing:
                - x: Starting x-coordinate on the object/entrance surface.
                - y: Starting y-coordinate on the object/entrance surface.
                - z: Starting z-coordinate on the object/entrance surface.
                - L: Direction cosine L.
                - M: Direction cosine M.
                - N: Direction cosine N.
        """
