"""Base Field Definition Module

This module defines the abstract base class for field types in optical systems.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from optiland._suggest import options_hint

if TYPE_CHECKING:
    from optiland import Optic
    from optiland._types import BEArray, ScalarOrArray


class BaseFieldDefinition(ABC):
    """Abstract base class for defining how fields map to ray properties."""

    _registry: ClassVar[dict[str, type[BaseFieldDefinition]]] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator to register a field type by name.

        Args:
            name: The string key used to look up this field type.

        Returns:
            A decorator that registers the subclass and returns it unchanged.

        """

        def decorator(subclass: type[BaseFieldDefinition]) -> type[BaseFieldDefinition]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, field_type: str) -> BaseFieldDefinition:
        """Instantiate a field definition by its registered name.

        Args:
            field_type: The registered name of the field type.

        Returns:
            A new instance of the corresponding field definition.

        Raises:
            ValueError: If ``field_type`` is not in the registry.

        """
        if field_type not in cls._registry:
            raise ValueError(
                f"Unknown field type, got {field_type!r}."
                f"{options_hint(str(field_type), cls._registry)} "
                "Set it with lens.fields.set_type('angle')."
            )
        return cls._registry[field_type]()

    @abstractmethod
    def get_ray_origins(
        self,
        optic: Optic,
        Hx: ScalarOrArray,
        Hy: ScalarOrArray,
        Px: ScalarOrArray,
        Py: ScalarOrArray,
        vx: ScalarOrArray,
        vy: ScalarOrArray,
    ) -> tuple[ScalarOrArray, ScalarOrArray, ScalarOrArray]:
        """Calculate the initial positions for rays originating at the object.

        Args:
            Hx: Normalized x field coordinate.
            Hy: Normalized y field coordinate.
            Px: x-coordinate of the pupil point.
            Py: y-coordinate of the pupil point.
            vx: Vignetting factor in the x-direction.
            vy: Vignetting factor in the y-direction.

        Returns:
            A tuple containing the x, y, and z coordinates of the
                object position.

        """
        pass  # pragma: no cover

    @abstractmethod
    def get_paraxial_object_position(
        self, optic: Optic, Hy: ScalarOrArray, y1: ScalarOrArray, EPL: ScalarOrArray
    ) -> tuple[BEArray, BEArray]:
        """Calculate the position of the object in the paraxial optical system.

        Args:
            Hy: The normalized field height.
            y1: The initial y-coordinate of the ray.
            EPL: The entrance pupil location.

        Returns:
            A tuple containing the y and z coordinates of the object
                position.

        """
        pass  # pragma: no cover

    @abstractmethod
    def scale_chief_ray_for_field(
        self,
        optic: Optic,
        y_obj_unit: ScalarOrArray,
        u_obj_unit: ScalarOrArray,
        y_img_unit: ScalarOrArray,
    ) -> ScalarOrArray:
        """Calculates the scaling factor for a unit chief ray based on the field
        definition.

        This is used in the paraxial chief_ray calculation. It uses the results
        of a forward and backward "unit" trace from the stop to determine the
        final scaling factor.

        Args:
            optic: The optical system.
            y_obj_unit: The object-space height of the unit ray.
            u_obj_unit: The object-space angle of the unit ray.
            y_img_unit: The image-space height of the unit ray.

        Returns:
            The scaling factor.

        """
        pass  # pragma: no cover

    def to_dict(self) -> dict:
        """Convert the field definition to a dictionary.

        Returns:
            dict: A dictionary representation of the field definition.

        """
        return {"field_type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, field_def_dict: dict) -> BaseFieldDefinition:
        """Create a field definition from a dictionary.

        Args:
            field_def_dict (dict): A dictionary representation of the field
                definition.

        Returns:
            BaseFieldDefinition: A field definition object created from the
                dictionary.

        Raises:
            ValueError: If ``field_type`` is missing or not in the registry.

        """
        if "field_type" not in field_def_dict:
            raise ValueError("Missing required keys: field_type")

        # Ensure subclasses are imported so their @register decorators run.
        from optiland.fields.field_types import (  # noqa: F401
            AngleField,
            ObjectHeightField,
            ParaxialImageHeightField,
            RealImageHeightField,
        )

        class_name = field_def_dict["field_type"]
        # Registry keys are class names (e.g. "AngleField"); look up by name.
        for _key, klass in cls._registry.items():
            if klass.__name__ == class_name:
                return klass()
        known = [klass.__name__ for klass in cls._registry.values()]
        raise ValueError(
            f"Unknown field definition {class_name!r} in the field data."
            f"{options_hint(str(class_name), known)}"
        )
