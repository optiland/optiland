"""Real Ray Tracer Module

This module contains the RealRayTracer class, which is responsible for tracing
real rays through an optical system. It uses the RayGenerator class to create
rays and the surface group of the optical system to trace them.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.distribution import create_distribution
from optiland.rays import PolarizedRays, RayGenerator
from optiland.raytrace.base import BaseRayTracer

if TYPE_CHECKING:
    from optiland._types import DistributionType
    from optiland.distribution import BaseDistribution


class RealRayTracer(BaseRayTracer):
    """Class to trace real rays through an optical system

    This class is responsible for building rays (via a ray generator) and tracing these
    through an optical system (via the surface group).

    Args:
        optic (Optic): The optical system to be traced.
    """

    def __init__(self, optic):
        super().__init__(optic)
        self.ray_generator = RayGenerator(optic)
        self.ray_aiming_config = {
            "mode": "paraxial",
            "max_iter": 10,
            "tol": 1e-6,
        }

    def set_aiming(self, mode: str, max_iter: int = 10, tol: float = 1e-6, **kwargs):
        """Configure the ray aiming strategy.

        Args:
            mode: The aiming mode ("paraxial", "iterative", "robust").
            max_iter: Maximum iterations for iterative solvers.
            tol: Convergence tolerance for iterative solvers.
            **kwargs: Additional configuration parameters.
        """
        self.ray_aiming_config = {
            "mode": mode,
            "max_iter": max_iter,
            "tol": tol,
            **kwargs,
        }

    def trace(
        self,
        Hx,
        Hy,
        wavelength,
        num_rays: int | None = 100,
        distribution: DistributionType | BaseDistribution | None = "hexapolar",
    ):
        """Trace a distribution of rays through the optical system.

        Args:
            Hx (float or numpy.ndarray): The normalized x field coordinate.
            Hy (float or numpy.ndarray): The normalized y field coordinate.
            wavelength (float): The wavelength of the rays.
            num_rays (int, optional): The number of rays to be traced. Defaults
                to 100.
            distribution (str or Distribution, optional): The distribution of
                the rays. Defaults to 'hexapolar'.

        Returns:
            RealRays: The RealRays object containing the traced rays."
        """
        self._validate_normalized_coordinates(Hx, Hy, "field")

        if isinstance(distribution, str):
            distribution = create_distribution(distribution)
            distribution.generate_points(num_rays)
        Px = distribution.x
        Py = distribution.y

        Hx = be.atleast_1d(Hx)
        Hy = be.atleast_1d(Hy)

        # expand coordinates to create a ray for each field point at each pupil point
        num_fields = len(Hx)
        num_pupil_points = len(Px)

        Hx_full = be.repeat(Hx, num_pupil_points)
        Hy_full = be.repeat(Hy, num_pupil_points)
        Px_full = be.tile(Px, num_fields)
        Py_full = be.tile(Py, num_fields)

        rays = self.ray_generator.generate_rays(
            Hx_full, Hy_full, Px_full, Py_full, wavelength
        )
        self.optic.surfaces.trace(rays)

        # Propagate to the image surface
        if self.optic.image_surface:
            last_surface = self.optic.surfaces[-1]
            last_surface.material_post.propagation_model.propagate(
                rays, last_surface.thickness
            )

        if isinstance(rays, PolarizedRays):
            rays.update_intensity(self.optic.polarization_state)

        # update ray intensity
        self.optic.surfaces.intensity[-1, :] = rays.i

        return rays

    def trace_generic(self, Hx, Hy, Px, Py, wavelength):
        """Trace generic rays through the optical system.

        Args:
            Hx (float or numpy.ndarray): The normalized x field coordinate.
            Hy (float or numpy.ndarray): The normalized y field coordinate.
            Px (float or numpy.ndarray): The normalized x pupil coordinate.
            Py (float or numpy.ndarray): The normalized y pupil coordinate
            wavelength (float): The wavelength of the rays.

        """
        self._validate_normalized_coordinates(Hx, Hy, "field")
        self._validate_normalized_coordinates(Px, Py, "pupil")

        vx, vy = self.optic.fields.get_vig_factor(Hx, Hy)

        Px = Px * (1 - vx)
        Py = Py * (1 - vy)

        # assure all variables are arrays of the same size
        Hx, Hy, Px, Py = self._validate_array_size(Hx, Hy, Px, Py)

        rays = self.ray_generator.generate_rays(Hx, Hy, Px, Py, wavelength)
        self.optic.surfaces.trace(rays)

        # Propagate to the image surface
        last_surface = self.optic.surfaces[-1]
        last_surface.material_post.propagation_model.propagate(
            rays, last_surface.thickness
        )

        # update intensity
        self.optic.surfaces.intensity[-1, :] = rays.i

        return rays

    def _validate_normalized_coordinates(self, x, y, coord_type="field"):
        """Validate that normalized coordinates are within the range (-1, 1).

        Args:
            x (float or numpy.ndarray): The normalized x coordinate.
            y (float or numpy.ndarray): The normalized y coordinate.
            coord_type (str): The type of coordinates being validated
                              ('field' or 'pupil').

        Raises:
            ValueError: If the coordinates are not within the range (-1, 1).
        """
        valid_x = be.all((x >= -1) & (x <= 1))
        valid_y = be.all((y >= -1) & (y <= 1))
        if not (valid_x and valid_y):
            raise ValueError(
                f"Normalized {coord_type} coordinates must be within (-1, 1)"
            )

    def _validate_array_size(self, *arrays):
        """Ensure all input variables are arrays of the same size.

        Args:
            *arrays: A variable number of inputs (float, int, or numpy.ndarray).

        Returns:
            list: A list of numpy arrays, all of the same size.
        """
        max_size = max([be.size(be.array(arr)) for arr in arrays])
        return [
            (
                be.full((max_size,), value)
                if isinstance(value, float | int)
                else value
                if be.is_array_like(value)
                else None
            )
            for value in arrays
        ]
