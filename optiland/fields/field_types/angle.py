"""Angle Field Module

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.utils import globalize_coordinates

from .base import BaseFieldDefinition


@BaseFieldDefinition.register("angle")
class AngleField(BaseFieldDefinition):
    """Defines fields by angle (in degrees) relative to the optical axis."""

    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """Calculate the initial positions for rays originating at the object.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or be.ndarray): x-coordinate of the pupil point.
            Py (float or be.ndarray): y-coordinate of the pupil point.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.

        """
        obj = optic.object_surface
        EPL = optic.paraxial.EPL()
        max_field = be.array(optic.fields.max_field)
        field_x = max_field * be.array(Hx)
        field_y = max_field * be.array(Hy)

        if obj.is_infinite:
            EPD = optic.paraxial.EPD()
            offset = self._get_starting_z_offset(optic)
            x = -be.tan(be.radians(field_x)) * (offset + EPL)
            y = -be.tan(be.radians(field_y)) * (offset + EPL)
            z = optic.surfaces.positions[1] - offset
            x0 = be.array(Px) * EPD / 2 * be.array(vx) + x
            y0 = be.array(Py) * EPD / 2 * be.array(vy) + y
            z0 = be.full_like(Px, z)
        else:
            dist_to_ep = optic.paraxial.entrance_pupil_z() - optic.surfaces.positions[0]
            x_local = be.atleast_1d(be.array(-be.tan(be.radians(field_x)) * dist_to_ep))
            y_local = be.atleast_1d(be.array(-be.tan(be.radians(field_y)) * dist_to_ep))
            z_local = obj.geometry.sag(x_local, y_local)

            # Globalize the local coordinates
            x0, y0, z0 = globalize_coordinates(obj, x_local, y_local, z_local)

            if be.size(x0) == 1:
                x0 = be.full_like(be.atleast_1d(Px), x0)
            if be.size(y0) == 1:
                y0 = be.full_like(be.atleast_1d(Px), y0)
            if be.size(z0) == 1:
                z0 = be.full_like(be.atleast_1d(Px), z0)
        return x0, y0, z0

    def get_paraxial_object_position(self, optic, Hy, y1, EPL):
        """Calculate the position of the object in the paraxial optical system.

        Args:
            Hy (float): The normalized field height.
            y1 (ndarray): The initial y-coordinate of the ray.
            EPL (float): The entrance pupil location.

        Returns:
            tuple: A tuple containing the y and z coordinates of the object
                position.

        """
        max_field = be.array(optic.fields.max_field)
        field_y = max_field * be.array(Hy)
        y = -be.tan(be.radians(field_y)) * EPL
        z = optic.surfaces.positions[1]
        y0 = y1 + y
        z0 = be.ones_like(y1) * z
        return y0, z0

    def scale_chief_ray_for_field(self, optic, y_obj_unit, u_obj_unit, y_img_unit):
        """Calculates the scaling factor for a unit chief ray based on the field
        definition.

        This is used in the paraxial chief_ray calculation. It uses the results
        of a forward and backward "unit" trace from the stop to determine the
        final scaling factor.

        Args:
            optic (Optic): The optical system.
            y_obj_unit (float): The object-space height of the unit ray.
            u_obj_unit (float): The object-space angle of the unit ray.
            y_img_unit (float): The image-space height of the unit ray.

        Returns:
            float: The scaling factor.
        """
        max_field_angle = optic.fields.max_y_field
        target_slope = be.tan(be.deg2rad(max_field_angle))
        return target_slope / u_obj_unit

    def _get_starting_z_offset(self, optic):
        """Calculate the starting ray z-coordinate offset for systems with an
        object at infinity. This is relative to the first surface of the optic.

        This method chooses a starting point that is equivalent to the entrance
        pupil diameter of the optic.

        Args:
            optic (Optic): The optical system being traced.

        Returns:
            float: The z-coordinate offset relative to the first surface.

        """
        z = optic.surfaces.positions[1:-1] - optic.surfaces.positions[1]
        offset = optic.paraxial.EPD()
        return offset - be.min(z)
