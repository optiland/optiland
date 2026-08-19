"""Field Group Module

This module defines the FieldGroup class, which manages a collection of fields
in an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.fields.field import Field
from optiland.fields.field_types import BaseFieldDefinition

if TYPE_CHECKING:
    from optiland._types import ScalarOrArray


class FieldGroup:
    """A class representing a group of fields.

    Attributes:
        fields (list): A list of fields in the group.
        telecentric (bool): Whether the system is telecentric in object space.

    Methods:
        get_vig_factor(Hx, Hy): Returns the vignetting factors for given Hx
            and Hy values.
        get_field_coords: Returns the normalized coordinates of the fields.
        add_field(field): Adds a field to the group.
        get_field(field_number): Returns the field at the specified index.

    """

    def __init__(self):
        self.fields = []
        self.field_definition: BaseFieldDefinition | None = None
        self.telecentric = False

    def require_definition(self) -> BaseFieldDefinition:
        """Return the field definition, or raise if none has been set.

        Field coordinates cannot be interpreted without a field type, so
        ray tracing fails with an opaque ``AttributeError`` if this is
        skipped. This turns that into an actionable message.

        Returns:
            The active field definition.

        Raises:
            ValueError: If no field type has been set on the system.
        """
        if self.field_definition is None:
            raise ValueError(
                "No field type is defined on the optical system, so field "
                "coordinates cannot be interpreted. Set one with "
                "lens.fields.set_type('angle') for an object at infinity, or "
                "lens.fields.set_type('object_height') for a finite object."
            )
        return self.field_definition

    @property
    def x_fields(self):
        """be.ndarray: x field values."""
        return be.array([field.x for field in self.fields])

    @property
    def y_fields(self):
        """be.ndarray: y field values."""
        return be.array([field.y for field in self.fields])

    @property
    def max_x_field(self):
        """float: Maximum field value in the x-direction."""
        return be.max(self.x_fields)

    @property
    def max_y_field(self):
        """float: Maximum field value in the y-direction."""
        return be.max(self.y_fields)

    @property
    def max_field(self):
        """float: Maximum radial field value."""
        if not self.fields:
            return 0.0
        return be.max(be.sqrt(self.x_fields**2 + self.y_fields**2))

    @property
    def num_fields(self):
        """int: number of fields in field group"""
        return len(self.fields)

    def __getitem__(self, index):
        return self.fields[index]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    @property
    def vx(self):
        """be.ndarray: Vignetting factors in x for each field."""
        return be.array([field.vx for field in self.fields])

    @property
    def vy(self):
        """be.ndarray: Vignetting factors in y for each field."""
        return be.array([field.vy for field in self.fields])

    def get_vig_factor(self, Hx, Hy):
        """Calculates the vignetting factors for a given field position.

        Note that the vignetting factors are interpolated using the nearest
        neighbor method.

        Args
            Hx (float): The normalized x component of the field.
            Hy (float): The normalized y component of the field.

        Returns:
            vx_new (float): The interpolated x-component of the
                vignetting factor.
            vy_new (float): The interpolated y-component of the
                vignetting factor.

        """
        max_field = self.max_field
        if max_field == 0:
            x_fields = self.x_fields
            y_fields = self.y_fields
        else:
            x_fields = self.x_fields / max_field
            y_fields = self.y_fields / max_field
        fields = be.stack((x_fields, y_fields), axis=-1)
        v_data = be.stack((self.vx, self.vy), axis=-1)
        result = be.nearest_nd_interpolator(fields, v_data, Hx, Hy)
        vx_new = result[..., 0]
        vy_new = result[..., 1]
        return vx_new, vy_new

    def get_field_coords(self) -> list[tuple[ScalarOrArray, ScalarOrArray]]:
        """Returns the coordinates of the fields.

        If the maximum field size is 0, it returns a single coordinate (0, 0).
        Otherwise, it calculates the normalized coordinates for each field
        based on the maximum field size.

        Returns:
            A list of tuples, where each tuple
            contains the (normalized_x, normalized_y) coordinates of a field.

        """
        max_field = self.max_field
        if max_field == 0:
            return [(0, 0)]
        return [
            (float(x / max_field), float(y / max_field))
            for x, y in zip(self.x_fields, self.y_fields, strict=False)
        ]

    @property
    def weights(self) -> tuple[float, ...]:
        """tuple[float, ...]: Weights for all fields as a tuple."""
        return tuple(field.weight for field in self.fields)

    def add(
        self,
        y: float,
        x: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        weight: float = 1.0,
    ):
        """Add a field to the list of fields.

        Args:
            y: The y-coordinate of the field.
            x: The x-coordinate of the field.
                Defaults to 0.0.
            vx: The x-component of the field's vignetting
                factor. Defaults to 0.0.
            vy: The y-component of the field's vignetting
                factor. Defaults to 0.0.
            weight: Non-negative relative importance scalar (default 1.0). A
                weight of 0.0 means this field is excluded from optimization
                and weighted analysis but is still present in standalone
                analysis outputs. Negative values raise ValueError.

        """
        new_field = Field(x, y, vx, vy, weight)
        self.fields.append(new_field)

    def set_type(self, field_type: str) -> None:
        """Set the type of field used in the optical system.

        Args:
            field_type: The type of field, e.g., 'angle',
                'object_height', or 'paraxial_image_height'.
        """
        self.field_definition = BaseFieldDefinition.create(field_type)

    def get_field(self, field_number: int) -> Field:
        """Retrieve the field at the specified field_number.

        Args:
            field_number (int): The index of the field to retrieve.

        Returns:
            Field: The field at the specified index.

        Raises:
            IndexError: If the field_number is out of range.

        """
        return self.fields[field_number]

    def remove(self, field_number: int) -> None:
        """Remove the field at the specified field_number.

        Args:
            field_number (int): The index of the field to remove.

        Raises:
            IndexError: If the field_number is out of range.

        """
        self.fields.pop(field_number)

    def set_telecentric(self, is_telecentric):
        """Specify whether the system is telecentric in object space.

        Args:
            is_telecentric (bool): Whether the system is telecentric in object
                space.

        """
        self.telecentric = is_telecentric

    def to_dict(self):
        """Convert the field group to a dictionary.

        Returns:
            dict: A dictionary representation of the field group.

        """
        data = {
            "fields": [field.to_dict() for field in self.fields],
            "telecentric": self.telecentric,
            "field_definition": (
                self.field_definition.to_dict()
                if self.field_definition is not None
                else None
            ),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Create a field group from a dictionary.

        Args:
            data (dict): A dictionary representation of the field group.

        Returns:
            FieldGroup: A field group object created from the dictionary.

        """
        field_group = cls()
        for field_dict in data["fields"]:
            field_group.add(
                y=field_dict.get("y", 0.0),
                x=field_dict.get("x", 0.0),
                vx=field_dict.get("vx", 0.0),
                vy=field_dict.get("vy", 0.0),
                weight=field_dict.get("weight", 1.0),
            )
        field_group.set_telecentric(data["telecentric"])

        if data.get("field_definition"):
            field_group.field_definition = BaseFieldDefinition.from_dict(
                data["field_definition"]
            )
        return field_group
