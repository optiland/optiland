"""Paraxial Image Height Field Module

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.utils import globalize_coordinates

from .base import BaseFieldDefinition


@BaseFieldDefinition.register("paraxial_image_height")
class ParaxialImageHeightField(BaseFieldDefinition):
    """Defines fields by the chief ray's paraxial height at the image plane."""

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

        Raises:
            ValueError: If the field type is "object_height" for an object at
                infinity.
        """
        y_img_target = optic.fields.max_field * Hy
        x_img_target = optic.fields.max_field * Hx

        y_img_unit, _ = self._trace_unit_chief_ray(optic, plane="image")
        y_obj_unit, u_obj_unit = self._trace_unit_chief_ray(optic, plane="object")

        if optic.object_surface.is_infinite:
            u_obj_y = u_obj_unit * (y_img_target / y_img_unit)
            u_obj_x = u_obj_unit * (x_img_target / y_img_unit)

            EPL = optic.paraxial.EPL()
            EPD = optic.paraxial.EPD()
            offset = self._get_starting_z_offset(optic)

            x = -u_obj_x * (offset + EPL)
            y = -u_obj_y * (offset + EPL)
            z = optic.surfaces.positions[1] - offset
            x0 = Px * EPD / 2 * vx + x
            y0 = Py * EPD / 2 * vy + y
            z0 = be.full_like(Px, z)
        else:
            y_obj = y_obj_unit * (y_img_target / y_img_unit)
            x_obj = y_obj_unit * (x_img_target / y_img_unit)
            x_local = be.atleast_1d(be.array(x_obj))
            y_local = be.atleast_1d(be.array(y_obj))
            z_local = optic.object_surface.geometry.sag(x_local, y_local)

            # Globalize the local coordinates
            x0, y0, z0 = globalize_coordinates(
                optic.object_surface, x_local, y_local, z_local
            )

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
        y_img_target = optic.fields.max_field * Hy
        y_img_unit, _ = self._trace_unit_chief_ray(optic, plane="image")
        y_obj_unit, u_obj_unit = self._trace_unit_chief_ray(optic, plane="object")

        if optic.object_surface.is_infinite:
            u_obj = u_obj_unit * (y_img_target / y_img_unit)
            y = u_obj * -EPL
            z = optic.surfaces.positions[1]
            y0 = y1 + y
            z0 = be.ones_like(y1) * z
        else:
            y_obj = y_obj_unit * (y_img_target / y_img_unit)
            y = y_obj
            z = optic.object_surface.geometry.cs.z
            y0 = be.ones_like(y1) * y
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
        max_image_height = optic.fields.max_y_field
        return max_image_height / y_img_unit

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
        z = optic.surfaces.positions[1:-1]
        offset = optic.paraxial.EPD()
        return offset - be.min(z)

    def _trace_unit_chief_ray(self, optic, plane="image"):
        stop_idx = optic.surfaces.stop_index
        num_surf = optic.surfaces.num_surfaces
        pos = optic.surfaces.positions
        wavelength = optic.primary_wavelength

        if plane == "image":
            z_start = pos[stop_idx]
            y, u = optic.paraxial.trace_generic(
                y=0, u=1, z=z_start, wavelength=wavelength, skip=stop_idx
            )
            return y[-1], u[-1]
        elif plane == "object":
            z_start = pos[-1] - pos[stop_idx]
            skip = num_surf - stop_idx
            y, u = optic.paraxial.trace_generic(
                y=0, u=1, z=z_start, wavelength=wavelength, reverse=True, skip=skip
            )
            return y[-1], u[-1]
