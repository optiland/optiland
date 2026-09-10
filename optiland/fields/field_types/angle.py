"""Angle Field Module

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.paraxial_path import (
    AMBIGUOUS_WIDE_ANGLE_FIELD,
    UnsupportedParaxialGeometryError,
    require_nonsingular_tangent_angles,
)
from optiland.utils import globalize_coordinates

from .base import BaseFieldDefinition

# Below this angle (degrees) a field component is treated as zero for the
# wide-angle ambiguity check.
_MATERIAL_ANGLE_DEG = 1e-9


def _validate_unambiguous_field(field_x, field_y) -> None:
    """Reject two-dimensional angle fields whose direction is ambiguous.

    The current component-angle representation composes ``tan(fx)`` and
    ``tan(fy)`` with a hemisphere rule based on the total angle. That
    uniquely defines a direction for one-dimensional fields at any angle
    and for two-dimensional fields inside 90 degrees; a two-dimensional
    field at or beyond 90 degrees has no unique three-dimensional direction
    in this representation, so it is rejected rather than silently mapped.

    Raises:
        UnsupportedParaxialGeometryError: With code
            ``AMBIGUOUS_WIDE_ANGLE_FIELD`` when both components are
            materially nonzero and either component or the total field
            angle crosses 90 degrees.
    """
    fx = be.to_numpy(be.atleast_1d(be.array(field_x))).reshape(-1)
    fy = be.to_numpy(be.atleast_1d(be.array(field_y))).reshape(-1)
    if fx.size == 1 and fy.size > 1:
        fx = fx.repeat(fy.size)
    if fy.size == 1 and fx.size > 1:
        fy = fy.repeat(fx.size)
    both = (abs(fx) > _MATERIAL_ANGLE_DEG) & (abs(fy) > _MATERIAL_ANGLE_DEG)
    total = (fx**2 + fy**2) ** 0.5
    wide = (abs(fx) >= 90.0) | (abs(fy) >= 90.0) | (total >= 90.0)
    bad = both & wide
    if bad.any():
        i = int(bad.nonzero()[0][0])
        raise UnsupportedParaxialGeometryError(
            f"[{AMBIGUOUS_WIDE_ANGLE_FIELD}] the two-dimensional angle "
            f"field (fx={fx[i]:.6g} deg, fy={fy[i]:.6g} deg, total "
            f"{total[i]:.6g} deg) reaches 90 degrees. The current "
            "component-angle representation does not uniquely define a "
            "three-dimensional direction in that regime; use a "
            "one-dimensional field, or wait for an explicit wide-angle "
            "field convention (polar angle + azimuth or direction "
            "cosines)."
        )


@BaseFieldDefinition.register("angle")
class AngleField(BaseFieldDefinition):
    """Defines fields by angle (in degrees) relative to the optical axis.

    For an object at infinity the field direction is composed from the two
    component angles against the entry axis; ``(u, v)`` pupil/field offsets
    use the entry frame's transverse basis for folded or off-axis-entered
    systems. Two-dimensional fields at or beyond 90 degrees total are
    ambiguous under this representation and are rejected (see
    :func:`_validate_unambiguous_field`).
    """

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
            _validate_unambiguous_field(field_x, field_y)
            require_nonsingular_tangent_angles(
                field_x, field_y, operation="infinite-conjugate ray origins"
            )
            EPD = optic.paraxial.EPD()
            offset = self._get_starting_z_offset(optic)
            d = offset + EPL

            # Past 90 deg, N = cos(theta) is negative, so the launch point
            # must sit downstream of the pupil (ray runs backward into it)
            # rather than upstream -- otherwise (L, M, N), which for
            # infinite conjugates is never refined past this seed, folds
            # back into (-90, 90) instead of reaching the true angle.
            theta_total = be.sqrt(field_x**2 + field_y**2)
            s = be.where(be.cos(be.radians(theta_total)) < 0, -1.0, 1.0)

            frame = optic.surfaces._entry_frame()
            if frame is None:
                z_pupil = optic.paraxial.entrance_pupil_axial_position()
                x = -s * be.tan(be.radians(field_x)) * d
                y = -s * be.tan(be.radians(field_y)) * d
                z = z_pupil - s * d
                x0 = be.array(Px) * EPD / 2 * be.array(vx) + x
                y0 = be.array(Py) * EPD / 2 * be.array(vy) + y
                z0 = be.zeros_like(Px) + z
            else:
                # The beam path is folded off the z axis: launch on the
                # entry line, a distance d behind the entrance pupil's
                # apparent (unfolded) position, with the pupil and field
                # offsets laid out on the entry frame's transverse basis.
                # Field angles measure against the entry axis, which is the
                # same thing the z-based branch means wherever both are
                # defined. See SurfaceGroup._entry_frame.
                anchor, axial, d0, u0, v0 = frame
                ep = optic.paraxial.entrance_pupil_axial_position()
                back = ep - axial - s * d
                tu = (
                    be.array(Px) * EPD / 2 * be.array(vx)
                    - s * be.tan(be.radians(field_x)) * d
                )
                tv = (
                    be.array(Py) * EPD / 2 * be.array(vy)
                    - s * be.tan(be.radians(field_y)) * d
                )
                x0 = anchor[0] + back * d0[0] + tu * u0[0] + tv * v0[0]
                y0 = anchor[1] + back * d0[1] + tu * u0[1] + tv * v0[1]
                z0 = anchor[2] + back * d0[2] + tu * u0[2] + tv * v0[2]
        else:
            require_nonsingular_tangent_angles(
                field_x, field_y, operation="finite-conjugate ray origins"
            )
            dist_to_ep = (
                optic.paraxial.entrance_pupil_axial_position()
                - optic.surfaces.positions[0]
            )
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
        require_nonsingular_tangent_angles(
            field_y, operation="paraxial object-position construction"
        )
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
        require_nonsingular_tangent_angles(
            max_field_angle, operation="chief-ray field scaling"
        )
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
