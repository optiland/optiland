"""Coating Factory Module

This module contains the CoatingFactory class, which is responsible for generating
an appropriate coating instance given an input configuration. The class ensures
that coatings are properly assigned based on material properties and optical
requirements.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.coatings import BaseCoating, FresnelCoating

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial


class CoatingFactory:
    """Factory class for creating coating instances.

    This class provides a method for generating coatings based on user-defined
    parameters. It supports both predefined coating instances and dynamically
    generated coatings based on material interactions.

    Methods:
        See :meth:`create` for creating a coating instance based on the
        given input.

    """

    @staticmethod
    def create(
        coating: str | BaseCoating,
        material_pre: BaseMaterial,
        material_post: BaseMaterial,
    ) -> BaseCoating | None:
        """Creates a coating instance based on the given input.

        Args:
            coating (str | BaseCoating): The coating specification. It can be either
                a string indicating a predefined coating type or an instance of
                BaseCoating.
            material_pre (BaseMaterial): The material before the surface.
            material_post (BaseMaterial): The material after the surface.

        Returns:
            BaseCoating | None: A coating instance if applicable, otherwise None.
        """
        if isinstance(coating, BaseCoating):
            return coating

        if isinstance(coating, str) and coating.lower() == "fresnel":
            return FresnelCoating(material_pre, material_post)

        return None
