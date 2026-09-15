"""Material Formatter Module

This module provides a flexible mechanism for formatting material information
for display. It uses a registry pattern to allow different material types to
be formatted in specific ways without modifying the core visualization code.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland import materials

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.surfaces import Surface


class MaterialFormatter:
    """A registry-based formatter for material information."""

    _formatters: dict[type, Callable[[Surface], str]] = {}
    _default_formatter: Callable[[Surface], str] | None = None

    @classmethod
    def register(cls, material_type: type, formatter: Callable[[Surface], str]):
        """Registers a formatter function for a specific material type.

        Args:
            material_type: The class of the material to format.
            formatter: A function that takes a Surface object and returns a
                formatted string description of its material.
        """
        cls._formatters[material_type] = formatter

    @classmethod
    def set_default(cls, formatter: Callable[[Surface], str]):
        """Sets the default formatter for unknown material types."""
        cls._default_formatter = formatter

    @classmethod
    def format(cls, surface: Surface) -> str:
        """Formats the material information for a given surface.

        Args:
            surface: The surface whose material information is to be formatted.

        Returns:
            str: A formatted string description of the material.
        """
        # specialized check for mirror
        if surface.interaction_model.is_reflective:
            return "Mirror"

        if hasattr(surface, "material_post"):
            material = surface.material_post
            # check for exact match first
            formatter = cls._formatters.get(type(material))

            # check for subclass match if no exact match
            if not formatter:
                for mat_type, fmt in cls._formatters.items():
                    if isinstance(material, mat_type):
                        formatter = fmt
                        break

            if formatter:
                return formatter(surface)

        if cls._default_formatter:
            return cls._default_formatter(surface)

        if isinstance(getattr(surface, "material_post", None), materials.BaseMaterial):
            return _format_builtin(surface)

        raise ValueError("Unknown material type")


def _format_builtin(surface: Surface) -> str:
    return surface.material_post.display_name


for _material_type in (
    materials.Material,
    materials.MaterialFile,
    materials.IdealMaterial,
    materials.AbbeMaterial,
    materials.AbbeMaterialE,
    materials.DataMaterial,
):
    MaterialFormatter.register(_material_type, _format_builtin)
