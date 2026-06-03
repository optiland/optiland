"""Coordinate System Factory Module

This module contains the CoordinateSystemFactory class, which is responsible
for generating coordinate system instances based on input parameters. The class
interfaces with the surface factory to ensure that each surface is assigned the
appropriate coordinate system, including positional and rotational attributes.

Kramer Harrison, 2025
"""

from __future__ import annotations

from optiland.coordinate_system import CoordinateSystem


class CoordinateSystemFactory:
    """Factory class for creating CoordinateSystem instances.

    This class abstracts the creation of coordinate systems, ensuring that the
    correct transformations are applied to each surface within an optical system.
    """

    def __init__(self, surface_factory):
        self.surface_factory = surface_factory

    def create(self, index, surface_group, **kwargs):
        """Creates and returns a CoordinateSystem instance.

        Args:
            index (int): The index of the surface within the optical system.
            surface_group (SurfaceGroup): The group containing all surfaces.
            **kwargs: Additional keyword arguments specifying position and rotation.
                - x (float): X-coordinate (if absolute positioning is used).
                - y (float): Y-coordinate (if absolute positioning is used).
                - z (float): Z-coordinate (if absolute positioning is used).
                - dx (float): X displacement relative to the previous surface.
                - dy (float): Y displacement relative to the previous surface.
                - thickness (float): Thickness of the surface (if relative positioning
                    is used).
                - rx (float): Rotation about the X-axis.
                - ry (float): Rotation about the Y-axis.
                - rz (float): Rotation about the Z-axis.

        Returns:
            CoordinateSystem: The configured coordinate system instance.

        Raises:
            ValueError: If conflicting positional parameters are provided.
        """
        if "z" in kwargs:
            if "thickness" in kwargs:
                raise ValueError('Cannot define both "thickness" and "z".')
            if "dx" in kwargs or "dy" in kwargs:
                raise ValueError(
                    'Cannot define "dx" or "dy" when using absolute "x", "y", "z".',
                )

            x = kwargs.get("x", 0)
            y = kwargs.get("y", 0)
            z = kwargs["z"]
            self.surface_factory.use_absolute_cs = True
        else:
            if self.surface_factory.use_absolute_cs:
                raise ValueError(
                    'Cannot pass "thickness" after defining '
                    '"x", "y", "z" position for a previous '
                    "surface.",
                )

            thickness = kwargs.get("thickness", 0)
            x = kwargs.get("dx", 0)
            y = kwargs.get("dy", 0)

            if index == 0:  # object surface
                z = -thickness
            elif index == 1:
                z = 0  # first surface, always at zero
            else:
                prev_surface = surface_group.surfaces[index - 1]
                z_prev = prev_surface.geometry.cs.z
                t_prev = prev_surface.thickness
                # Extract plain floats so the new coordinate is a leaf scalar,
                # not a non-leaf tensor created by arithmetic on traced values.
                if hasattr(z_prev, "item"):
                    z_prev = z_prev.item()
                if hasattr(t_prev, "item"):
                    t_prev = t_prev.item()
                z = float(z_prev) + float(t_prev)

        rx = kwargs.get("rx", 0)
        ry = kwargs.get("ry", 0)
        rz = kwargs.get("rz", 0)

        return CoordinateSystem(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)
