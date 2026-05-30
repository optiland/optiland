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

        vxf, vyf = self.optic.fields.get_vig_factor(Hx, Hy)
        vx = 1 - be.array(vxf)
        vy = 1 - be.array(vyf)

        x0, y0, z0 = self.optic.fields.field_definition.get_ray_origins(
            self.optic, Hx, Hy, Px, Py, vx, vy
        )

        if self.optic.obj_space_telecentric:
            self._check_telecentric_compatibility()
            sin = self.optic.aperture.value
            z = be.sqrt(1 - sin**2) / sin + z0
            z1 = be.full_like(Px, z)
            x1 = Px * vx + x0
            y1 = Py * vy + y0
        else:
            EPD = self.optic.paraxial.EPD()
            epl_global = self.optic.paraxial.entrance_pupil_z()

            x1 = Px * EPD * vx / 2
            y1 = Py * EPD * vy / 2
            z1 = be.full_like(Px, epl_global)

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
