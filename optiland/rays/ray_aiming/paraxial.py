"""
Paraxial Ray Aimer Module

This module implements the paraxial ray aiming algorithm, which aims rays
at the paraxial entrance pupil.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.fields.field_types import AngleField
from optiland.paraxial_path import (
    UnsupportedParaxialGeometryError,
    paraxial_seed_scope,
)
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.registry import register_aimer

if TYPE_CHECKING:
    from optiland._types import ScalarOrArrayT


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

        vxf, vyf = self.optic.fields.get_vig_factor(Hx, Hy)
        vx = 1 - be.array(vxf)
        vy = 1 - be.array(vyf)

        x0, y0, z0 = self.optic.fields.require_definition().get_ray_origins(
            self.optic, Hx, Hy, Px, Py, vx, vy
        )

        if self.optic.obj_space_telecentric:
            self._check_telecentric_compatibility()
            # The telecentric launch construction below displaces along
            # global z and lays pupil offsets in global x/y; it is only
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
            sin = self.optic.aperture.value
            z = be.sqrt(1 - sin**2) / sin + z0
            z1 = be.full_like(Px, z)
            x1 = Px * vx + x0
            y1 = Py * vy + y0
        else:
            EPD = self.optic.paraxial.EPD()
            frame = self.optic.surfaces._entry_frame()

            if frame is None:
                epl_global = self.optic.paraxial.entrance_pupil_axial_position()

                x1 = Px * EPD * vx / 2
                y1 = Py * EPD * vy / 2
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
                a = Px * EPD * vx / 2
                b = Py * EPD * vy / 2
                x1 = anchor[0] + (ep - axial) * d0[0] + a * u0[0] + b * v0[0]
                y1 = anchor[1] + (ep - axial) * d0[1] + a * u0[1] + b * v0[1]
                z1 = anchor[2] + (ep - axial) * d0[2] + a * u0[2] + b * v0[2]

        mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)

        # Handle case where ray origin and pupil point are the same
        is_zero = mag < 1e-9
        mag = be.where(is_zero, 1.0, mag)

        L = be.where(is_zero, 0.0, (x1 - x0) / mag)
        M = be.where(is_zero, 0.0, (y1 - y0) / mag)
        N = be.where(is_zero, 1.0, (z1 - z0) / mag)

        return x0, y0, z0, L, M, N

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
