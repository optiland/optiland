"""Field Module

This module defines the Field class, which represents a field in an optical system.

Kramer Harrison, 2025
"""

from __future__ import annotations


class Field:
    """Represents a field with specific properties.

    Attributes:
        x (float): The x-coordinate of the field.
        y (float): The y-coordinate of the field.
        vx (float): The vignette factor in the x-direction.
        vy (float): The vignette factor in the y-direction.
        weight (float): The relative importance scalar for this field.
            Non-negative; 0.0 means excluded from weighted contexts.

    """

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        vignette_factor_x: float = 0.0,
        vignette_factor_y: float = 0.0,
        weight: float = 1.0,
    ) -> None:
        self.x = x
        self.y = y
        self.vx = vignette_factor_x
        self.vy = vignette_factor_y
        self.weight = weight  # uses the validated setter

    @property
    def weight(self) -> float:
        """float: Non-negative relative importance scalar for this field."""
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"Field weight must be non-negative, got {value}.")
        self._weight = float(value)

    def to_dict(self) -> dict:
        """Convert the field to a dictionary.

        Returns:
            A dictionary representation of the field.

        """
        return {
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, field_dict: dict) -> Field:
        """Create a field from a dictionary.

        Args:
            field_dict: A dictionary representation of the field.

        Returns:
            A field object created from the dictionary.

        """
        return cls(
            field_dict.get("x", 0),
            field_dict.get("y", 0),
            field_dict.get("vx", 0.0),
            field_dict.get("vy", 0.0),
            field_dict.get("weight", 1.0),
        )
