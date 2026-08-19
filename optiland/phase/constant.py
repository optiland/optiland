"""
Provides a constant phase profile.
"""

from __future__ import annotations

from optiland import backend as be
from optiland.phase.base import BasePhaseProfile


class ConstantPhaseProfile(BasePhaseProfile):
    """A phase profile that is constant everywhere.

    This class represents a phase profile that has a constant value and zero
    gradient at all points. It is useful for representing surfaces that have no
    phase contribution.

    Args:
        phase (float, optional): The constant phase value. Defaults to 0.0.
    """

    phase_type = "constant"

    def __init__(self, phase: float = 0.0):
        super().__init__()
        self.phase = phase

    def get_phase(
        self, x: be.Array, y: be.Array, wavelength: be.Array = None
    ) -> be.Array:
        """Calculates the phase added by the profile at coordinates (x, y).

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.

        Returns:
            The phase at each (x, y) coordinate.
        """
        return be.full_like(x, self.phase)

    def get_gradient(
        self, x: be.Array, y: be.Array, wavelength: be.Array = None
    ) -> tuple[be.Array, be.Array, be.Array]:
        """Calculates the gradient of the phase at coordinates (x, y).

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.

        Returns:
            A tuple containing the x, y, and z components of the phase
            gradient, which are always zero for this profile.
        """
        return be.zeros_like(x), be.zeros_like(y), be.zeros_like(x)

    def get_paraxial_gradient(
        self, y: be.Array, wavelength: be.Array = None
    ) -> be.Array:
        """Calculates the paraxial phase gradient at y-coordinate.

        This is the gradient d_phi/dy evaluated at x=0.

        Args:
            y: The y-coordinates of the points of interest.

        Returns:
            The paraxial phase gradient at each y-coordinate, which is always
            zero for this profile.
        """
        return be.zeros_like(y)

    def to_dict(self) -> dict:
        """Serializes the phase profile to a dictionary.

        Returns:
            A dictionary representation of the phase profile.
        """
        data = super().to_dict()
        data["phase"] = self.phase
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ConstantPhaseProfile:
        """Deserializes a phase profile from a dictionary.

        Args:
            data: A dictionary representation of a phase profile.

        Returns:
            An instance of a `ConstantPhaseProfile`.
        """
        return cls(phase=data.get("phase", 0.0))
