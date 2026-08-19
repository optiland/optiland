"""Pickup Module

The pickup module contains classes for managing and performing pickup
operations on an optic surface. A pickup operation involves copying an
attribute value from one surface to another surface, optionally scaling and
offsetting the value.

It also supports generic pickups between arbitrary attributes of any
Optic instance.

Kramer Harrison, 2024
"""

from __future__ import annotations

from optiland.utils import get_attr_by_path, set_attr_by_path


class PickupManager:
    """A class for managing multiple pickup operations on an optic surface

    Args:
        optic (Optic): The optic object on which the pickup operations are
            performed.

    Attributes:
        pickups (list): A list of Pickup objects representing the pickup
            operations to be performed.

    Methods:
        add(): Adds a new pickup operation to the manager.
        apply(): Applies all pickup operations in the manager.
        clear(): Clears all pickup operations in the manager.

    """

    def __init__(self, optic):
        self.optic = optic
        self.pickups = []

    def __len__(self):
        return len(self.pickups)

    def add(
        self,
        source_surface_idx,
        attr_type,
        target_surface_idx,
        scale=1,
        offset=0,
        source_optic=None,
    ):
        """Adds a new pickup operation to the manager.

        Args:
            source_surface_idx (int): The index of the source surface in the
                optic's surface group.
            attr_type (str): The type of attribute to be picked up ('radius',
                'conic', or 'thickness'). Can also be a generic path, where `[i]`
                should be used to indicate the surface index (e.g.
                `surfaces.surfaces[i].geometry.coefficients`).
            target_surface_idx (int): The index of the target surface in the
                optic's surface group.
            scale (float, optional): The scaling factor applied to the picked
                up value. Defaults to 1.
            offset (float, optional): The offset added to the picked up value.
                Defaults to 0.
            source_optic (Optic, optional): The generic source optic. Defaults
                to None (same optic).

        """
        pickup = Pickup(
            self.optic,
            source_surface_idx,
            attr_type,
            target_surface_idx,
            scale,
            offset,
            source_optic,
        )
        pickup.apply()
        self.pickups.append(pickup)

    def apply(self):
        """Applies all pickup operations in the manager."""
        for pickup in self.pickups:
            pickup.apply()

    def clear(self):
        """Clears all pickup operations in the manager."""
        self.pickups.clear()

    def remap_surface_indices(self, remap_func):
        """Remaps the source and target surface indices of all pickups.

        Args:
            remap_func (callable): A function that takes an old surface index
                                   and returns a new surface index.
        """
        for pickup in self.pickups:
            pickup.source_surface_idx = remap_func(pickup.source_surface_idx)
            pickup.target_surface_idx = remap_func(pickup.target_surface_idx)

    def to_dict(self):
        """Returns a dictionary representation of the pickup manager.

        Returns:
            dict: A dictionary representation of the pickup manager.

        """
        return [pickup.to_dict() for pickup in self.pickups]

    @classmethod
    def from_dict(cls, optic, data):
        """Creates a PickupManager object from a dictionary representation.

        Args:
            optic (Optic): The optic object on which the pickup operations are
                performed.
            data (dict): A dictionary representation of the pickup manager.

        Returns:
            PickupManager: A PickupManager object created from the dictionary
                representation.

        """
        manager = cls(optic)
        for pickup_data in data:
            manager.add(**pickup_data)
        return manager


class Pickup:
    """A class representing a pickup on an optic surface

    Args:
        optic (Optic): The optic object on which the pickup operation is
            performed.
        source_surface_idx (int): The index of the source surface in the
            optic's surface group.
        attr_type (str): The type of attribute to be picked up ('radius',
            'conic', or 'thickness'). Can also be a generic path, where `[i]`
            should be used to indicate the surface index (e.g.
            `surfaces.surfaces[i].geometry.coefficients`).
        target_surface_idx (int): The index of the target surface in the
            optic's surface group.
        scale (float, optional): The scaling factor applied to the picked up
            value. Defaults to 1.
        offset (float, optional): The offset added to the picked up value.
            Defaults to 0.
        source_optic (Optic, optional): The source optic to pick up from.
            Defaults to None (uses self.optic).

    Methods:
        apply(): Applies the pickup operation by scaling and offsetting the
            picked up value and setting it on the target surface.

    Raises:
        ValueError: If an invalid source attribute is specified.

    """

    def __init__(
        self,
        optic,
        source_surface_idx,
        attr_type,
        target_surface_idx,
        scale=1,
        offset=0,
        source_optic=None,
    ):
        self.optic = optic
        self.source_surface_idx = source_surface_idx
        self.attr_type = attr_type
        self.target_surface_idx = target_surface_idx
        self.scale = scale
        self.offset = offset
        self.source_optic = source_optic if source_optic else optic

    def apply(self):
        """Updates the target surface based on the source surface attribute.

        This method calculates the new value by multiplying the current value
        by the scale factor and adding the offset. The new value is then set
        on the target surface.
        """
        old_value = self._get_value()

        # Optimization: if scale is 1 and offset is 0, just copy the value.
        # This allows pickups to work with non-numeric types (e.g., strings).
        if self.scale == 1 and self.offset == 0:
            self._set_value(old_value)
            return

        new_value = self.scale * old_value + self.offset
        self._set_value(new_value)

    def _get_value(self):
        """Returns the value of the source surface attribute.

        Returns:
            The value of the attribute.

        """
        # Legacy support
        surface = self.source_optic.surfaces[self.source_surface_idx]
        if self.attr_type == "radius":
            return surface.geometry.radius
        if self.attr_type == "conic":
            return surface.geometry.k
        if self.attr_type == "thickness":
            return self.source_optic.surfaces.get_thickness(self.source_surface_idx)

        # Generic path support
        try:
            path = self.attr_type.replace("[i]", f"[{self.source_surface_idx}]")
            val = get_attr_by_path(self.source_optic, path)
            return val
        except AttributeError:
            raise ValueError("Invalid source attribute") from None

    def _set_value(self, value):
        """Sets the value of the target surface attribute.

        Args:
            value (float): The value to set for the attribute.

        """
        # Legacy support
        if self.attr_type == "radius":
            self.optic.updater.set_radius(value, self.target_surface_idx)
            return
        elif self.attr_type == "conic":
            self.optic.updater.set_conic(value, self.target_surface_idx)
            return
        elif self.attr_type == "thickness":
            self.optic.updater.set_thickness(value, self.target_surface_idx)
            return

        # Generic path support
        try:
            path = self.attr_type.replace("[i]", f"[{self.target_surface_idx}]")
            # Check existence first to ensure no arbitrary attribute creation
            get_attr_by_path(self.optic, path)
            set_attr_by_path(self.optic, path, value)
        except Exception:
            raise ValueError("Invalid target attribute") from None

    def to_dict(self):
        """Returns a dictionary representation of the pickup operation.

        Returns:
            dict: A dictionary representation of the pickup operation.

        """
        return {
            "source_surface_idx": self.source_surface_idx,
            "attr_type": self.attr_type,
            "target_surface_idx": self.target_surface_idx,
            "scale": self.scale,
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, optic, data):
        """Creates a Pickup object from a dictionary representation.

        Args:
            optic (Optic): The optic object on which the pickup operation is
                performed.
            data (dict): A dictionary representation of the pickup operation.

        Returns:
            Pickup: A Pickup object created from the dictionary representation.

        """
        return cls(
            optic,
            data.get("source_surface_idx"),
            data.get("attr_type"),
            data.get("target_surface_idx"),
            data.get("scale", 1),
            data.get("offset", 0),
        )
