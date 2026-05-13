"""Object Height Field Module

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be

from .base import BaseFieldDefinition


@BaseFieldDefinition.register("object_height")
class ObjectHeightField(BaseFieldDefinition):
    """Defines fields by height on the object surface."""

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
        self._validate_object_infinite(optic)
        obj = optic.object_surface
        max_field = optic.fields.max_field
        x_local = be.atleast_1d(be.array(max_field * Hx))
        y_local = be.atleast_1d(be.array(max_field * Hy))
        z_local = obj.geometry.sag(x_local, y_local)

        # Globalize the local coordinates
        eff_translation, eff_rot_mat = obj.geometry.cs.get_effective_transform()

        # Handle both scalar and array inputs for field coordinates
        # and ensure the points are correctly transformed to the global system.
        points_local = be.stack([x_local, y_local, z_local], axis=0)
        if len(be.shape(points_local)) == 1:
            points_local = be.unsqueeze_last(points_local)

        points_global = be.matmul(eff_rot_mat, points_local) + be.reshape(
            eff_translation, (3, 1)
        )

        x0 = be.ravel(points_global[0, :])
        y0 = be.ravel(points_global[1, :])
        z0 = be.ravel(points_global[2, :])

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

        Raises:
            ValueError: If the field type is "object_height" and the object is
                at infinity.

        """
        self._validate_object_infinite(optic)
        obj = optic.object_surface
        field_y = optic.fields.max_field * Hy
        y = -field_y
        z = obj.geometry.cs.z
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
        max_field_height = optic.fields.max_y_field
        return max_field_height / y_obj_unit

    def _validate_object_infinite(self, optic):
        """Check if the object surface is at infinity.

        Args:
            optic (Optic): The optical system being traced.

        Raises:
            ValueError: If the object surface is at infinity.
        """
        if optic.object_surface.is_infinite:
            raise ValueError("Object surface is at infinity.")
