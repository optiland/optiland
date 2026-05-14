"""Real Image Height Field Module

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be

from .base import BaseFieldDefinition
from .paraxial_image_height import ParaxialImageHeightField


@BaseFieldDefinition.register("real_image_height")
class RealImageHeightField(BaseFieldDefinition):
    """Defines fields by the chief ray's real height at the image plane."""

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
        # Initial guess using ParaxialImageHeightField logic
        paraxial_field = ParaxialImageHeightField()
        paraxial_field.scale_chief_ray_for_field(
            optic,
            *paraxial_field._trace_unit_chief_ray(optic, plane="object"),
            paraxial_field._trace_unit_chief_ray(optic, plane="image")[0],
        )

        max_field = optic.fields.max_field
        target_x = max_field * Hx
        target_y = max_field * Hy

        # Initial guess for the field parameters (angle or height)
        # We solve for the field parameters that result in the target image height
        if optic.object_surface.is_infinite:
            y_img_unit, _ = paraxial_field._trace_unit_chief_ray(optic, plane="image")
            _, u_obj_unit = paraxial_field._trace_unit_chief_ray(optic, plane="object")

            val_x = u_obj_unit * (target_x / y_img_unit)
            val_y = u_obj_unit * (target_y / y_img_unit)
        else:
            # For finite object, variables are object heights
            y_img_unit, _ = paraxial_field._trace_unit_chief_ray(optic, plane="image")
            y_obj_unit, _ = paraxial_field._trace_unit_chief_ray(optic, plane="object")

            val_x = y_obj_unit * (target_x / y_img_unit)
            val_y = y_obj_unit * (target_y / y_img_unit)

        # Iterative solver (Newton's method)
        num_iterations = 10
        tol = 1e-12

        # We need the paraxial magnification/scaling for the Jacobian
        # d(ImageHeight)/d(FieldParam)
        if optic.object_surface.is_infinite:
            jacobian = y_img_unit / u_obj_unit
        else:
            jacobian = y_img_unit / y_obj_unit

        prev_val_x = None
        prev_curr_x = None
        prev_val_y = None
        prev_curr_y = None

        for _ in range(num_iterations):
            # 1. Generate chief rays for current object guess
            rays = self._generate_chief_rays(optic, val_x, val_y)

            # 2. Trace rays
            optic.surfaces.trace(rays)

            # Propagate to image surface
            last_surface = optic.surfaces[-1]
            last_surface.material_post.propagation_model.propagate(
                rays, last_surface.thickness
            )

            # 3. Get intersection at image plane
            curr_x = rays.x
            curr_y = rays.y

            err_x = curr_x - target_x
            err_y = curr_y - target_y

            # Check convergence (using max error across all fields)
            if be.max(be.abs(err_x)) < tol and be.max(be.abs(err_y)) < tol:
                break

            # 4. Update guess
            # Use Secant method if possible, else Paraxial Jacobian
            if prev_val_x is not None and be.any(be.abs(val_x - prev_val_x) > 1e-12):
                d_val_x = val_x - prev_val_x
                d_curr_x = curr_x - prev_curr_x

                # Avoid division by zero
                mask_x = be.abs(d_val_x) > 1e-12
                jac_update_x = be.ones_like(val_x) * jacobian  # Fallback
                jac_update_x[mask_x] = d_curr_x[mask_x] / d_val_x[mask_x]

                # Store current values before updating val_x
                prev_val_x = be.copy(val_x)
                prev_curr_x = be.copy(curr_x)

                # Apply update X
                val_x -= err_x / jac_update_x

                d_val_y = val_y - prev_val_y
                d_curr_y = curr_y - prev_curr_y

                # Avoid division by zero
                mask_y = be.abs(d_val_y) > 1e-12
                jac_update_y = be.ones_like(val_y) * jacobian  # Fallback
                jac_update_y[mask_y] = d_curr_y[mask_y] / d_val_y[mask_y]

                prev_val_y = be.copy(val_y)  # Save OLD val
                prev_curr_y = be.copy(curr_y)

                val_y -= err_y / jac_update_y
            else:
                prev_val_x = be.copy(val_x)
                prev_curr_x = be.copy(curr_x)
                val_x -= err_x / jacobian

                prev_val_y = be.copy(val_y)
                prev_curr_y = be.copy(curr_y)
                val_y -= err_y / jacobian

        # Generate final ray origins using the converged field parameters
        return self._compute_ray_origins_from_params(
            optic, val_x, val_y, Px, Py, vx, vy
        )

    def _generate_chief_rays(self, optic, val_x, val_y):
        """Generate chief rays (Px=0, Py=0) for the given field parameters."""
        from optiland.rays import RealRays

        zeros = be.zeros_like(val_x)

        # We use _compute_ray_origins_from_params with Px=0, Py=0, vx=0, vy=0
        x0, y0, z0 = self._compute_ray_origins_from_params(
            optic, val_x, val_y, zeros, zeros, 0, 0
        )

        EPL = optic.paraxial.EPL()
        # EPL is relative to the first surface (index 1)
        z_pupil = optic.surfaces.positions[1] + EPL

        x1 = be.zeros_like(x0)
        y1 = be.zeros_like(y0)
        z1 = be.full_like(x0, z_pupil)

        mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        intensity = be.ones_like(x0)
        wavelength = be.full_like(x0, optic.primary_wavelength)

        return RealRays(x0, y0, z0, L, M, N, intensity, wavelength)

    def _compute_ray_origins_from_params(self, optic, val_x, val_y, Px, Py, vx, vy):
        """Compute ray origins given field parameters and pupil coords."""
        if optic.object_surface.is_infinite:
            EPL = optic.paraxial.EPL()
            EPD = optic.paraxial.EPD()
            offset = self._get_starting_z_offset(optic)

            x = -val_x * (offset + EPL)
            y = -val_y * (offset + EPL)
            z = optic.surfaces.positions[1] - offset

            x0 = Px * EPD / 2 * vx + x
            y0 = Py * EPD / 2 * vy + y
            z0 = be.full_like(Px, z)
        else:
            # val_x, val_y are object heights
            x0 = val_x
            y0 = val_y

            # Ensure correct shape
            if be.size(x0) == 1:
                x0 = be.full_like(Px, x0)
            if be.size(y0) == 1:
                y0 = be.full_like(Px, y0)

            z0 = (
                optic.object_surface.geometry.sag(x0, y0)
                + optic.object_surface.geometry.cs.z
            )
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
        return ParaxialImageHeightField().get_paraxial_object_position(
            optic, Hy, y1, EPL
        )

    def scale_chief_ray_for_field(self, optic, y_obj_unit, u_obj_unit, y_img_unit):
        """Calculates the scaling factor for a unit chief ray based on the field
        definition.

        Args:
            optic (Optic): The optical system.
            y_obj_unit (float): The object-space height of the unit ray.
            u_obj_unit (float): The object-space angle of the unit ray.
            y_img_unit (float): The image-space height of the unit ray.

        Returns:
            float: The scaling factor.
        """
        return ParaxialImageHeightField().scale_chief_ray_for_field(
            optic, y_obj_unit, u_obj_unit, y_img_unit
        )

    def _get_starting_z_offset(self, optic):
        z = optic.surfaces.positions[1:-1]
        offset = optic.paraxial.EPD()
        return offset - be.min(z)
