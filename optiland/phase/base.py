"""
Provides the abstract base class for all phase profile strategies.
"""

from __future__ import annotations

import abc
import typing

if typing.TYPE_CHECKING:
    from optiland import backend as be
    from optiland.surfaces.standard_surface import Surface


class BasePhaseProfile(abc.ABC):
    """Abstract base class for defining phase profiles on a surface.

    This class defines the interface that all phase profile strategies must
    implement. It uses a registry pattern to handle serialization and
    deserialization, allowing for easy extension with custom phase profiles.
    """

    _registry = {}

    def __init__(self):
        self.parent_surface: Surface | None = None

    def __init_subclass__(cls, **kwargs):
        """Registers subclasses for deserialization."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "phase_type"):
            cls._registry[cls.phase_type] = cls

    @property
    def efficiency(self) -> float:
        """The diffraction efficiency of the phase profile.

        Returns:
            The efficiency, a value between 0 and 1.
        """
        return 1.0

    @abc.abstractmethod
    def get_phase(self, x: be.Array, y: be.Array, wavelength: be.Array) -> be.Array:
        """Calculates the phase added by the profile at coordinates (x, y).

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.

        Returns:
            The phase at each (x, y) coordinate.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_gradient(
        self, x: be.Array, y: be.Array, wavelength: be.Array
    ) -> tuple[be.Array, be.Array, be.Array]:
        """Calculates the gradient of the phase at coordinates (x, y).

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.

        Returns:
            A tuple containing the x, y, and z components of the phase gradient
            (d_phi/dx, d_phi/dy, d_phi/dz).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_paraxial_gradient(self, y: be.Array, wavelength: be.Array) -> be.Array:
        """Calculates the paraxial phase gradient at y-coordinate.

        This is the gradient d_phi/dy evaluated at x=0.

        Args:
            y: The y-coordinates of the points of interest.

        Returns:
            The paraxial phase gradient at each y-coordinate.
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Serializes the phase profile to a dictionary.

        Returns:
            A dictionary representation of the phase profile.
        """
        return {"phase_type": self.phase_type}

    @classmethod
    def from_dict(cls, data: dict) -> BasePhaseProfile:
        """Deserializes a phase profile from a dictionary.

        Args:
            data: A dictionary representation of a phase profile.

        Returns:
            An instance of a `BasePhaseProfile` subclass.

        Raises:
            ValueError: If the `phase_type` is unknown.
        """
        phase_type = data.get("phase_type")
        if phase_type not in cls._registry:
            raise ValueError(f"Unknown phase profile type: {phase_type}")

        # Delegate to the correct subclass's from_dict
        return cls._registry[phase_type].from_dict(data)
