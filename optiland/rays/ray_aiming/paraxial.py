"""
Paraxial Ray Aimer Module

This module implements the paraxial ray aiming algorithm, which aims rays
at the paraxial entrance pupil.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.aperture import ObjectNAAperture
from optiland.fields.field_types import AngleField
from optiland.paraxial_path import (
    UnsupportedParaxialGeometryError,
    paraxial_seed_scope,
)
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.registry import register_aimer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optiland._types import BEArray, ScalarOrArray, ScalarOrArrayT

# A ray whose aim point is closer to its origin than this distance gets the
# direction (0, 0, 1). The compiled directions take the same value.
_MIN_AIM_DISTANCE = 1e-9


def _unit_directions(
    x0: BEArray,
    y0: BEArray,
    z0: BEArray,
    x1: BEArray,
    y1: BEArray,
    z1: BEArray,
    flip: bool,
    axis: Sequence[ScalarOrArray] | None,
) -> tuple[BEArray, BEArray, BEArray]:
    """Return the unit vectors from the ray origins to their aim points.

    These NumPy expressions are the reference of the compiled directions
    (``optiland.backend.numpy_backend._aim``), which must give their bits.

    Args:
        x0: The x coordinates of the ray origins.
        y0: The y coordinates of the ray origins.
        z0: The z coordinates of the ray origins.
        x1: The x coordinates of the aim points.
        y1: The y coordinates of the aim points.
        z1: The z coordinates of the aim points.
        flip: Whether a ray whose aim point is behind its origin gets the
            opposite direction.
        axis: The axis of that test (three components), or None for the
            z axis.

    Returns:
        The direction cosines L, M and N. A ray whose aim point is less than
        ``_MIN_AIM_DISTANCE`` from its origin gets (0, 0, 1) before the
        reversal test.
    """
    mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)

    # Handle case where ray origin and pupil point are the same
    is_zero = mag < _MIN_AIM_DISTANCE
    mag = be.where(is_zero, 1.0, mag)

    L = be.where(is_zero, 0.0, (x1 - x0) / mag)
    M = be.where(is_zero, 0.0, (y1 - y0) / mag)
    N = be.where(is_zero, 1.0, (z1 - z0) / mag)

    if flip:
        if axis is None:
            along = z1 - z0
        else:
            along = (x1 - x0) * axis[0] + (y1 - y0) * axis[1] + (z1 - z0) * axis[2]
        behind = along < 0
        L = be.where(behind, -L, L)
        M = be.where(behind, -M, M)
        N = be.where(behind, -N, N)
    return L, M, N


@register_aimer("paraxial")
class ParaxialRayAimer(BaseRayAimer):
    """
    Paraxial ray aiming algorithm.

    This aimer targets the paraxial entrance pupil of the optical system.
    It handles both finite and infinite object distances, as well as
    telecentric object spaces.
    """

    def aim_rays(
        self,
        fields: tuple[ScalarOrArrayT, ScalarOrArrayT],
        wavelengths: ScalarOrArrayT,  # noqa: ARG002
        pupil_coords: tuple[ScalarOrArrayT, ScalarOrArrayT],
    ) -> tuple[
        ScalarOrArrayT,
        ScalarOrArrayT,
        ScalarOrArrayT,
        ScalarOrArrayT,
        ScalarOrArrayT,
        ScalarOrArrayT,
    ]:
        """
        Calculate ray starting coordinates and direction cosines targeting the
        paraxial entrance pupil.

        Args:
            fields: Normalized field coordinates (Hx, Hy).
            wavelengths: Wavelengths for the rays (unused in paraxial aimer).
            pupil_coords: Normalized pupil coordinates (Px, Py).

        Returns:
            Tuple containing:
                - x: Starting x-coordinate.
                - y: Starting y-coordinate.
                - z: Starting z-coordinate.
                - L: Direction cosine L.
                - M: Direction cosine M.
                - N: Direction cosine N.
        """
        Hx, Hy = fields
        Px, Py = pupil_coords

        # Ensure backend arrays
        Hx = be.as_array_1d(Hx)
        Hy = be.as_array_1d(Hy)
        Px = be.as_array_1d(Px)
        Py = be.as_array_1d(Py)

        # Ray aiming uses scalar paraxial values as launch seeds for real
        # rays; out-of-domain geometries (e.g. a slightly tilted stop
        # mirror) warn instead of raising here. Direct first-order analysis
        # stays strict.
        with paraxial_seed_scope():
            return self._aim_rays_impl(Hx, Hy, Px, Py)

    def _aim_rays_impl(self, Hx, Hy, Px, Py):
        """Body of :meth:`aim_rays`, run inside the paraxial seed scope."""

        # (Px, Py) arrive scaled by (1 - vx, 1 - vy) in
        # RayGenerator.generate_rays. So the pupil scale factors of
        # get_ray_origins are 1.0, and the formulas below do not scale Px and
        # Py again.
        x0, y0, z0 = self.optic.fields.require_definition().get_ray_origins(
            self.optic, Hx, Hy, Px, Py, 1.0, 1.0
        )

        # The entry frame of a folded system (see SurfaceGroup._entry_frame).
        # It stays None for a telecentric object space.
        frame = None
        if self.optic.obj_space_telecentric:
            self._check_telecentric_compatibility()
            # The telecentric launches below use global +z as their axis
            # and global x/y for pupil offsets; this construction is only
            # valid while the beam enters along +z. Reject other entries
            # rather than silently constructing invalid targets.
            path = self.optic.surfaces.build_paraxial_path()
            if not path.entry_is_positive_z:
                raise UnsupportedParaxialGeometryError(
                    "Object-space telecentric ray aiming is only supported "
                    "for systems entered along global +z; this system's "
                    "entry direction is off that axis. A local-coordinate "
                    "telecentric construction is not yet implemented. Real "
                    "ray tracing remains available."
                )
            if isinstance(self.optic.aperture, ObjectNAAperture):
                # Share the aperture's NA = n*sin(theta) conversion, reference
                # wavelength and validation with its EPD calculation.
                sine = self.optic.aperture.object_space_sine(self.optic)
                # Work directly with directions to avoid a target at 1/sine.
                # Normalization retains the existing slope-based pupil sampling.
                L = Px * sine
                M = Py * sine
                N = be.ones_like(Px) * be.sqrt(1 - sine**2)
                norm = be.sqrt(L**2 + M**2 + N**2)
                return x0, y0, z0, L / norm, M / norm, N / norm

            sin = self.optic.aperture.value
            z = be.sqrt(1 - sin**2) / sin + z0
            z1 = be.full_like(Px, z)
            x1 = Px + x0
            y1 = Py + y0
        else:
            EPD = self.optic.paraxial.EPD()
            frame = self.optic.surfaces._entry_frame()

            if frame is None:
                epl_global = self.optic.paraxial.entrance_pupil_axial_position()

                x1 = Px * EPD / 2
                y1 = Py * EPD / 2
                z1 = be.full_like(Px, epl_global)
            else:
                # The beam path is folded off the z axis, so the pupil disc
                # cannot live in a z-perpendicular plane. Rays aim at the
                # entrance pupil's apparent position in object space: on the
                # entry line at the pupil's axial coordinate, with the pupil
                # offsets laid out on the entry frame's transverse basis.
                # See SurfaceGroup._entry_frame for why the position is
                # never refolded onto a downstream leg.
                anchor, axial, d0, u0, v0 = frame
                ep = self.optic.paraxial.entrance_pupil_axial_position()
                a = Px * EPD / 2
                b = Py * EPD / 2
                x1 = anchor[0] + (ep - axial) * d0[0] + a * u0[0] + b * v0[0]
                y1 = anchor[1] + (ep - axial) * d0[1] + a * u0[1] + b * v0[1]
                z1 = anchor[2] + (ep - axial) * d0[2] + a * u0[2] + b * v0[2]

        # A finite object closer to the lens than its virtual entrance pupil has the
        # pupil behind it: the aim point is then behind the launch point, and a ray
        # sent toward it would leave the lens, its optical path counted backwards.
        # Send it toward the lens along the same line through the pupil point.
        finite = not self.optic.object_surface.is_infinite
        flip = finite and not self.optic.obj_space_telecentric
        # The entry axis d0 of a folded system, or None for the z axis.
        axis = None if frame is None else d0

        directions = None
        if be.get_backend() == "numpy":
            # Imported here, not at the top of the module: this looks up the
            # module attribute at each call, so the tests can replace it.
            from optiland.backend.numpy_backend._aim import compiled_directions

            directions = compiled_directions(
                x0, y0, z0, x1, y1, z1, flip, axis, _MIN_AIM_DISTANCE
            )
        if directions is None:
            directions = _unit_directions(x0, y0, z0, x1, y1, z1, flip, axis)
        return x0, y0, z0, *directions

    def _check_telecentric_compatibility(self) -> None:
        """Video compatibility checks for telecentric object space."""
        if isinstance(self.optic.fields.field_definition, AngleField):
            raise ValueError(
                'Field type cannot be "angle" for telecentric object space.'
            )
        if not self.optic.aperture.supports_telecentric:
            raise ValueError(
                f'Aperture type "{self.optic.aperture.ap_type}" is not compatible '
                f"with telecentric object space."
            )
