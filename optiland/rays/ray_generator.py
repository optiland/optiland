"""Ray Generator

This module contains the RayGenerator class, which is used to generate rays
for tracing through an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.rays.polarized_rays import PolarizedRays
from optiland.rays.ray_aiming.registry import create_ray_aimer
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from optiland._types import ScalarOrArray


class RayGenerator:
    """Generator class for creating rays."""

    def __init__(self, optic):
        self.optic = optic
        self.aimer = create_ray_aimer("paraxial", optic)

    def set_ray_aiming(
        self, mode: str, max_iter: int = 10, tol: float = 1e-6, **kwargs
    ):
        """Set the ray aiming strategy.

        Args:
            mode (str): The name of the ray aiming strategy. Options include
                "paraxial", "iterative", and "robust".
            max_iter (int, optional): Maximum iterations for iterative/robust
                methods. Defaults to 10.
            tol (float, optional): Convergence tolerance for iterative/robust
                methods. Defaults to 1e-6.
            **kwargs: Additional parameters passed to the aimer constructor.
        """
        self.aimer = create_ray_aimer(
            mode, self.optic, max_iter=max_iter, tol=tol, **kwargs
        )

    def generate_rays(
        self,
        Hx: float,
        Hy: float,
        Px: ScalarOrArray,
        Py: ScalarOrArray,
        wavelength: ScalarOrArray,
    ) -> RealRays:
        """Generates rays for tracing based on the given parameters.

        Args:
            Hx: Normalized x field coordinate.
            Hy: Normalized y field coordinate.
            Px: x-coordinate of the pupil point.
            Py: y-coordinate of the pupil point.
            wavelength: Wavelength of the rays.

        Returns:
            RealRays object containing the generated rays.

        """
        if hasattr(self.optic.ray_tracer, "ray_aiming_config"):
            config = self.optic.ray_tracer.ray_aiming_config
            if not hasattr(self, "_current_config") or self._current_config != config:
                self.set_ray_aiming(**config)
                self._current_config = config.copy()

        # Aim rays using the configured strategy
        x0, y0, z0, L, M, N = self.aimer.aim_rays(
            (Hx, Hy),
            wavelength,
            (Px, Py),
        )

        apodization = self.optic.apodization
        if apodization:
            intensity = apodization.get_intensity(Px, Py)
        else:
            intensity = be.ones_like(Px)

        wavelength = be.ones_like(x0) * wavelength

        if self.optic.polarization == "ignore":
            if self.optic.surfaces.uses_polarization:
                raise ValueError(
                    "Polarization must be set when surfaces have "
                    "polarization-dependent coatings.",
                )
            rays = RealRays(
                x0, y0, z0, L, M, N, intensity=intensity, wavelength=wavelength
            )
            return rays
        return PolarizedRays(x0, y0, z0, L, M, N, intensity, wavelength)
