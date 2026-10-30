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
from optiland.raytrace import _resume
from optiland.raytrace.base import BaseRayTracer
from optiland.raytrace.partial_trace import partial_trace_start
from optiland.raytrace.snapshot import TraceSnapshot, capture_rays

if TYPE_CHECKING:
    from optiland._types import DistributionType
    from optiland.distribution import BaseDistribution
    from optiland.rays import RealRays


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
        record: bool = True,
        *,
        retain_launch: bool = False,
    ):
        """Trace a distribution of rays through the optical system.

        Args:
            Hx (float or numpy.ndarray): The normalized x field coordinate.
            Hy (float or numpy.ndarray): The normalized y field coordinate.
            wavelength (float): The wavelength of the rays.
            num_rays (int, optional): The sampling parameter that determines the
                number of rays in the pupil. Its meaning depends on the value of
                `distribution`.
                Defaults to 100.
            distribution (str or Distribution, optional): The distribution of
                the rays. Defaults to 'hexapolar'.
            record (bool, optional): Whether to store per-surface snapshots of
                the traced rays for later analysis. Pass False to cut peak
                memory roughly in half when only the returned rays are needed
                (e.g. large GPU traces). Defaults to True.
            retain_launch: Whether to retain the generated ray state for an
                analysis. Defaults to False.

        Returns:
            RealRays: The RealRays object containing the traced rays."

        Notes:
            The interpretation of the `num_rays` argument depends on the value of
            `distribution`:

            - 'hexapolar': the number of rings N; the pupil holds
              ``1 + 3 * N * (N + 1)`` points (e.g. 8 -> 217, 16 -> 817,
              32 -> 3169).
            - 'uniform': the number of grid points per axis n; about
              ``pi / 4 * n**2`` points remain after masking to the unit disk.
            - 'cross': the number of points along each axis.
            - 'random', 'sobol', 'ring', and the 'line' distributions: the
              total number of points.

            If a `BaseDistribution` object is passed, `num_rays` is ignored and
            its existing points are used. The returned rays hold one ray per
            pupil point for every field point; ``len(rays.x)`` gives the count.
        """
        self._validate_normalized_coordinates(Hx, Hy, "field")

        if isinstance(distribution, str):
            distribution = create_distribution(distribution)
            distribution.generate_points(num_rays)
        Px = distribution.x
        Py = distribution.y

        # With gradients globally disabled, the whole trace runs without
        # autograd bookkeeping; with gradients enabled this context is a
        # no-op and differentiable tracing works as before.
        with be.no_grad_unless_enabled():
            Hx = be.atleast_1d(Hx)
            Hy = be.atleast_1d(Hy)

            # expand coordinates: one ray per field point at each pupil point
            num_fields = len(Hx)
            num_pupil_points = len(Px)

            Hx_full = be.repeat(Hx, num_pupil_points)
            Hy_full = be.repeat(Hy, num_pupil_points)
            Px_full = be.tile(Px, num_fields)
            Py_full = be.tile(Py, num_fields)

            rays = self.ray_generator.generate_rays(
                Hx_full,
                Hy_full,
                Px_full,
                Py_full,
                wavelength,
                retain_launch=retain_launch,
            )
            self.optic.surfaces.trace(rays, record=record)

            # Propagate to the image surface
            if self.optic.image_surface:
                last_surface = self.optic.surfaces[-1]
                last_surface.material_post.propagation_model.propagate(
                    rays, last_surface.thickness
                )

            if isinstance(rays, PolarizedRays):
                rays.update_intensity(self.optic.polarization_state)

        return rays

    def trace_generic(
        self,
        Hx,
        Hy,
        Px,
        Py,
        wavelength,
        record: bool = True,
        *,
        retain_launch: bool = False,
    ):
        """Trace generic rays through the optical system.

        Args:
            Hx (float or numpy.ndarray): The normalized x field coordinate.
            Hy (float or numpy.ndarray): The normalized y field coordinate.
            Px (float or numpy.ndarray): The normalized x pupil coordinate.
            Py (float or numpy.ndarray): The normalized y pupil coordinate
            wavelength (float): The wavelength of the rays.
            record (bool, optional): Whether to store per-surface snapshots of
                the traced rays. Defaults to True.
            retain_launch: Whether to retain the generated ray state for an
                analysis. Defaults to False.

        """
        self._validate_normalized_coordinates(Hx, Hy, "field")
        self._validate_normalized_coordinates(Px, Py, "pupil")

        # Partial-trace finite differences record or resume this trace.
        hook = _resume.active()
        if hook is not None and not retain_launch:
            rays = hook.trace_generic(self, Hx, Hy, Px, Py, wavelength, record)
            if rays is not None:
                return rays

        with be.no_grad_unless_enabled():
            # Vignetting is applied once, in RayGenerator.generate_rays.
            # assure all variables are arrays of the same size
            Hx, Hy, Px, Py = self._validate_array_size(Hx, Hy, Px, Py)

            rays = self.ray_generator.generate_rays(
                Hx, Hy, Px, Py, wavelength, retain_launch=retain_launch
            )
            self.optic.surfaces.trace(rays, record=record)

            # Propagate to the image surface
            last_surface = self.optic.surfaces[-1]
            last_surface.material_post.propagation_model.propagate(
                rays, last_surface.thickness
            )

        return rays

    def partial_trace_start(self, variable) -> int | None:
        """Surface index where a trace may resume after ``variable`` changes.

        Conservative: an index is returned only when every surface before it,
        the launched rays and the ray aiming are provably unaffected by the
        variable; otherwise None, meaning a full trace. See
        :func:`optiland.raytrace.partial_trace.partial_trace_start`.

        Args:
            variable: An optimization ``Variable`` (or its behavior object).

        Returns:
            The surface index to resume from with :meth:`_trace_from`, or None.
        """
        return partial_trace_start(self.optic, variable)

    def _trace(
        self,
        Hx,
        Hy,
        Px,
        Py,
        wavelength,
        record: bool = True,
    ) -> tuple[RealRays, TraceSnapshot]:
        """Trace generic rays and snapshot the ray state before each surface.

        Private (opt v2 PR3.1). Traces exactly as :meth:`trace_generic` does,
        with the same returned rays and recorded surface data, and also
        returns a :class:`TraceSnapshot` that :meth:`_trace_from` resumes.

        Args:
            Hx (float or numpy.ndarray): The normalized x field coordinate.
            Hy (float or numpy.ndarray): The normalized y field coordinate.
            Px (float or numpy.ndarray): The normalized x pupil coordinate.
            Py (float or numpy.ndarray): The normalized y pupil coordinate.
            wavelength (float): The wavelength of the rays.
            record (bool, optional): Whether to store per-surface snapshots of
                the traced rays on the surfaces. Defaults to True.

        Returns:
            The traced rays and the snapshot of the ray state before each
            surface.
        """
        self._validate_normalized_coordinates(Hx, Hy, "field")
        self._validate_normalized_coordinates(Px, Py, "pupil")

        with be.no_grad_unless_enabled():
            Hx, Hy, Px, Py = self._validate_array_size(Hx, Hy, Px, Py)

            rays = self.ray_generator.generate_rays(Hx, Hy, Px, Py, wavelength)
            surfaces = self.optic.surfaces.surfaces
            self.optic.surfaces.reset()
            states = []
            for surface in surfaces:
                states.append(capture_rays(rays))
                surface.trace(rays, record=record)

            self._propagate_to_image(rays)

        return rays, TraceSnapshot(tuple(states), tuple(surfaces))

    def _trace_from(
        self, k: int, snapshot: TraceSnapshot, record: bool = True
    ) -> RealRays:
        """Resume a snapshot trace at surface ``k``.

        Private (opt v2 PR3.1). Starts from the ray state stored before
        surface ``k`` and traces surfaces ``k`` to the image, writing their
        surface data exactly as a full trace would. Surfaces before ``k`` keep
        their recorded data. The result equals a full trace only when nothing
        before surface ``k`` changed since the snapshot was taken, which
        :meth:`partial_trace_start` decides. The snapshot is not modified, so
        it can be resumed any number of times.

        Args:
            k: The surface index to resume at.
            snapshot: A snapshot from :meth:`_trace` on this optic.
            record (bool, optional): Whether to store per-surface snapshots of
                the traced rays on the surfaces. Defaults to True.

        Returns:
            The traced rays.

        Raises:
            ValueError: If the optic's surfaces are not the ones recorded.
            IndexError: If ``k`` is not a recorded surface index.
        """
        surfaces = self.optic.surfaces.surfaces
        if not snapshot.matches(surfaces):
            raise ValueError(
                "The snapshot was recorded on a different surface list; take a "
                "new snapshot after adding, removing or replacing surfaces."
            )
        if not 0 <= k < len(snapshot):
            raise IndexError(
                f"Cannot resume at surface {k}: the snapshot holds surfaces "
                f"0 to {len(snapshot) - 1}."
            )

        with be.no_grad_unless_enabled():
            rays = snapshot.state_before(k)
            for surface in surfaces[k:]:
                surface.trace(rays, record=record)

            self._propagate_to_image(rays)

        return rays

    def _propagate_to_image(self, rays: RealRays) -> None:
        """Propagate rays by the thickness after the last surface.

        Args:
            rays: The rays traced through every surface.
        """
        last_surface = self.optic.surfaces[-1]
        last_surface.material_post.propagation_model.propagate(
            rays, last_surface.thickness
        )

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
                f"Normalized {coord_type} coordinates must lie within "
                f"[-1, 1], got x={x!r}, y={y!r}. These are normalized "
                f"coordinates: {coord_type} extremes are +/-1, not physical "
                "units."
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
