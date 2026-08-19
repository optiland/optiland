"""Thickness Solve Module

This module defines `ThicknessSolve`, an abstract base class for solves
that adjust surface positions to satisfy a condition.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from optiland.solves.base import BaseSolve


class ThicknessSolve(BaseSolve, ABC):
    """Abstract base class for thickness solves.

    This class provides the common structure for solves that aim to achieve a
    specific ray height at a given surface by adjusting the z-position of that
    surface and subsequent surfaces.

    Attributes:
        optic (Optic): The optic object.
        surface_idx (int): The index of the surface where the ray height
            is to be controlled.
        height (float): The target height of the ray on the specified surface.
    """

    def __init__(self, optic, surface_idx: int, height: float):
        """Initializes a ThicknessSolve object.

        Args:
            optic (Optic): The optic object.
            surface_idx (int): The index of the surface.
            height (float): The target height of the ray.
        """
        if surface_idx is None:
            raise ValueError("'surface_idx' argument must be provided.")
        super().__init__()
        self.optic = optic
        self.surface_idx = surface_idx
        self.height = height

    @abstractmethod
    def _get_ray_y_u(self):
        """Gets the ray height (y) and slope (u) arrays for the relevant ray.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - ya: ray heights at each surface.
                - ua: ray slopes at each surface.
        """
        pass  # pragma: no cover

    def apply(self):
        """Applies the thickness solve to the optic.

        This method calculates the necessary shift in z-position for the
        target surface and all subsequent surfaces to achieve the desired
        ray height.
        """
        y, u = self._get_ray_y_u()

        # Ensure surface_idx is within bounds for y and u
        if not (0 <= self.surface_idx < len(y) and 0 <= self.surface_idx < len(u)):
            raise IndexError(
                f"surface_idx {self.surface_idx} is out of bounds for ray data arrays "
                f"of length {len(y)}."
            )

        if y[self.surface_idx] is None:
            raise ValueError(f"Ray height at surface {self.surface_idx} is None. ")

        u_incident = u[0] if self.surface_idx == 0 else u[self.surface_idx - 1]

        if u_incident == 0:
            # Cannot apply thickness solve if ray is parallel to axis
            return

        offset = (self.height - y[self.surface_idx]) / u_incident

        # Shift current surface and all subsequent surfaces
        num_surfaces_in_group = len(self.optic.surfaces)
        if not (0 <= self.surface_idx < num_surfaces_in_group):
            raise IndexError(
                f"surface_idx {self.surface_idx} is out of bounds for surface group "
                f"of length {num_surfaces_in_group}."
            )

        for i in range(self.surface_idx, num_surfaces_in_group):
            surface = self.optic.surfaces[i]
            current_z = surface.geometry.cs.z
            new_z = current_z + offset
            surface.geometry.cs.z = new_z

    def to_dict(self):
        """Returns a dictionary representation of the solve."""
        solve_dict = super().to_dict()
        solve_dict.update(
            {
                "surface_idx": self.surface_idx,
                "height": self.height,
            }
        )
        return solve_dict

    @classmethod
    def from_dict(cls, optic, data):
        """Creates a solve instance from a dictionary representation."""
        if cls is ThicknessSolve:
            raise TypeError(
                "ThicknessSolve is an abstract class and cannot be "
                "instantiated directly."
            )
        return cls(optic, data["surface_idx"], data["height"])


class MarginalRayHeightThicknessSolve(ThicknessSolve):
    """Solves for a target marginal ray height on a specific surface."""

    def _get_ray_y_u(self):
        """Gets the marginal ray height (y) and slope (u) arrays."""
        return self.optic.paraxial.marginal_ray()


class ChiefRayHeightThicknessSolve(ThicknessSolve):
    """Solves for a target chief ray height on a specific surface."""

    def _get_ray_y_u(self):
        """Gets the chief ray height (y) and slope (u) arrays."""
        return self.optic.paraxial.chief_ray()
