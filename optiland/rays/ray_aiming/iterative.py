"""Iterative Ray Aiming Module

This module implements the iterative ray aiming algorithm with robust
derivative calculation for wide-angle systems.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.rays import RealRays
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.paraxial import ParaxialRayAimer
from optiland.rays.ray_aiming.registry import register_aimer

if TYPE_CHECKING:
    from optiland.optic import Optic


@register_aimer("iterative")
class IterativeRayAimer(BaseRayAimer):
    """Iterative ray aiming strategy using Modified Newton-Raphson.

    This class implements an iterative ray aiming algorithm that solves for the
    initial ray coordinates (x, y) or directions (L, M) required to hit a specific
    target on the stop surface. It uses a Modified Newton-Raphson method with
    a paraxial Jacobian estimate and Broyden rank-1 updates to achieve fast
    super-linear convergence without expensive finite-difference recalculations.

    Attributes:
        optic (Optic): The optical system being traced.
        max_iter (int): Maximum number of iterations allowed.
        tol (float): Convergence tolerance for ray aiming error.
        _paraxial_aimer (ParaxialRayAimer): Helper to generate initial guesses.
    """

    def __init__(
        self,
        optic: Optic,
        max_iter: int = 20,
        tol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        """Initialize the IterativeRayAimer.

        Args:
            optic (Optic): The optical system to aim rays for.
            max_iter (int, optional): Maximum number of iterations. Defaults to 20.
            tol (float, optional): Error tolerance for convergence. Defaults to 1e-8.
            **kwargs: Additional keyword arguments passed to BaseRayAimer.
        """
        super().__init__(optic, **kwargs)
        self.max_iter = max_iter
        self.tol = tol
        self._paraxial_aimer = ParaxialRayAimer(optic)
        self.last_iterations = 0

    def aim_rays(
        self,
        fields: tuple,
        wavelengths: Any,
        pupil_coords: tuple,
        initial_guess: tuple | None = None,
    ) -> tuple:
        """Calculate ray starting coordinates using iterative aiming.

        This method solves the inverse ray tracing problem to find the starting
        coordinates (on the object surface) or directions (for finite objects)
        such that the ray passes through the specified pupil coordinates on the
        stop surface.

        Args:
            fields (tuple): Field coordinates (Hy, Hx) or (angle_x, angle_y).
            wavelengths (Any): Wavelengths of the rays in microns.
            pupil_coords (tuple): Normalized pupil coordinates (Px, Py).
            initial_guess (tuple | None, optional): Optional starting guess
                (x, y, z, L, M, N). If None, a paraxial guess is used.

        Returns:
            tuple: A tuple containing the solved ray parameters (x, y, z, L, M, N).

        Raises:
            ValueError: If initial guess produces NaNs or if the solver fails
                to converge within max_iter.
        """
        if initial_guess:
            x, y, z, L, M, N = initial_guess
        else:
            # Helper to ensure fields and pupil coords are backend arrays
            Hx, Hy = fields
            Hx = be.as_array_1d(Hx)
            Hy = be.as_array_1d(Hy)
            fields = (Hx, Hy)

            Px, Py = pupil_coords
            Px = be.as_array_1d(Px)
            Py = be.as_array_1d(Py)
            pupil_coords = (Px, Py)

            x, y, z, L, M, N = self._paraxial_aimer.aim_rays(
                fields, wavelengths, pupil_coords
            )

        # Ensure arrays
        x = be.as_array_1d(x)
        y = be.as_array_1d(y)
        z = be.as_array_1d(z)
        L = be.as_array_1d(L)
        M = be.as_array_1d(M)
        N = be.as_array_1d(N)

        Px, Py = pupil_coords
        Px = be.as_array_1d(Px)
        Py = be.as_array_1d(Py)
        stop_idx = self.optic.surfaces.stop_index
        is_inf = getattr(self.optic.object_surface, "is_infinite", False)

        # Determine target coordinates
        # Use initialization strategy to find the effective stop radius.
        from optiland.rays.ray_aiming.initialization import get_stop_radius_strategy

        strategy = get_stop_radius_strategy(self.optic, "iterative")
        r_stop = strategy.calculate_stop_radius()
        rx = ry = r_stop

        tx, ty = Px * rx, Py * ry
        # Ensure proper broadcasting for indexing later
        tx = tx * be.ones_like(x)
        ty = ty * be.ones_like(y)

        x, y, z, L, M, N, converged, had_initial_nan = self._solve_core(
            x, y, z, L, M, N, wavelengths, stop_idx, is_inf, tx, ty
        )

        if had_initial_nan:
            raise ValueError(
                "Initial ray aiming guess produced NaNs. "
                "Consider using the 'robust' method instead."
            )

        if not be.all(converged):
            raise ValueError("Iterative aimer failed to converge.")

        return x, y, z, L, M, N

    def _solve_core(
        self,
        x: Any,
        y: Any,
        z: Any,
        L: Any,
        M: Any,
        N: Any,
        wavelengths: Any,
        stop_idx: int,
        is_inf: bool,
        tx: Any,
        ty: Any,
    ) -> tuple:
        """Core 2-DOF Newton/Broyden solve against an arbitrary local-stop
        target, without raising.

        This is the reusable solver core: it drives ``(x, y)`` (infinite
        conjugates) or ``(L, M)`` (finite conjugates) so that the ray lands
        at local-stop coordinates ``(tx, ty)``. Unlike the public
        :meth:`aim_rays`, this never raises -- NaNs and non-convergence are
        reported per-ray via the returned mask, so callers such as
        ``RobustRayAimer`` can treat individual ray failures gracefully
        instead of aborting the whole batch.

        Args:
            x, y, z, L, M, N: Initial ray launch guess.
            wavelengths: Wavelengths of the rays.
            stop_idx: Index of the stop surface.
            is_inf: Whether the object is at infinity.
            tx, ty: Target local-stop coordinates for each ray.

        Returns:
            tuple: ``(x, y, z, L, M, N, converged, had_initial_nan)`` where
            ``converged`` is a per-ray boolean mask and ``had_initial_nan``
            indicates whether the initial guess produced NaN errors for any
            ray.
        """
        tol_sq = self.tol**2

        # Initial trace (all rays)
        rays = self._trace_subset(x, y, z, L, M, N, wavelengths, stop_idx, is_inf)
        lx, ly = self._get_local_stop_coords(rays, stop_idx)
        ex, ey = lx - tx, ly - ty

        had_initial_nan = bool(be.any(be.isnan(ex)) or be.any(be.isnan(ey)))

        num_rays = len(x)
        full_indices = be.arange_indices(num_rays)

        # Precompute Paraxial Jacobian Factor
        wl_mean = (
            be.mean(wavelengths) if hasattr(wavelengths, "__len__") else wavelengths
        )
        J_factor = self._get_paraxial_jacobian(float(wl_mean), stop_idx, is_inf)
        if abs(J_factor) < 1e-12:
            J_factor = 1e-12

        # Initialize Jacobian Estimate (J)
        # J = [[j11, j12], [j21, j22]]
        J11 = be.full(num_rays, J_factor)
        J12 = be.zeros(num_rays)
        J21 = be.zeros(num_rays)
        J22 = be.full(num_rays, J_factor)

        # Ensure we are not modifying leaf variables in-place
        x = be.copy(x)
        y = be.copy(y)
        z = be.copy(z)
        L = be.copy(L)
        M = be.copy(M)
        N = be.copy(N)

        converged = ex**2 + ey**2 < tol_sq
        self.last_iterations = 0

        for _iter_idx in range(self.max_iter):
            # Check convergence
            error_sq = ex**2 + ey**2
            converged = error_sq < tol_sq

            if be.all(converged):
                break

            stuck = be.logical_or(converged, be.isnan(error_sq))
            if be.all(stuck):
                break

            self.last_iterations = _iter_idx + 1

            # Active Set Strategy: only process non-converged rays
            active_mask = ~converged
            # Ensure indices are integers
            idx = full_indices[active_mask]

            # Extract active data
            ex_curr = ex[idx]
            ey_curr = ey[idx]

            # Update solution (Newton Step)
            # [dx] = - [J]^-1 * [error]
            # Determinant for active rays
            det = J11[idx] * J22[idx] - J12[idx] * J21[idx]

            # Prevent division by zero
            det = be.where(be.abs(det) < 1e-12, 1e-12, det)

            # Invert 2x2 matrix analytically
            J11_inv = J22[idx]
            J12_inv = -J12[idx]
            J21_inv = -J21[idx]
            J22_inv = J11[idx]

            dp1 = -(J11_inv * ex_curr + J12_inv * ey_curr) / det
            dp2 = -(J21_inv * ex_curr + J22_inv * ey_curr) / det

            # Update parameters
            if is_inf:
                x[idx] += dp1
                y[idx] += dp2
            else:
                L[idx] += dp1
                M[idx] += dp2

            # Recalculate errors with new parameters
            rays = self._trace_subset(x, y, z, L, M, N, wavelengths, stop_idx, is_inf)
            lx, ly = self._get_local_stop_coords(rays, stop_idx)
            ex_new = lx - tx
            ey_new = ly - ty

            # Extract new errors for active set
            ex_next = ex_new[idx]
            ey_next = ey_new[idx]

            # --- Broyden Update ---
            # y_k = ex_next - ex_curr
            # s_k = dp1, dp2
            dEx = ex_next - ex_curr
            dEy = ey_next - ey_curr

            dx = dp1
            dy = dp2

            # Update J
            # J += (y - J*s) * s^T / (s^T * s)

            # Calculate J*s (using OLD J)
            Js_x = J11[idx] * dx + J12[idx] * dy
            Js_y = J21[idx] * dx + J22[idx] * dy

            Rx = dEx - Js_x
            Ry = dEy - Js_y

            # Norm sq of step s
            norm_sq = dx**2 + dy**2
            norm_sq = be.maximum(norm_sq, 1e-20)

            # Update J (Avoid in-place leaf errors by copying first)
            J11 = be.copy(J11)
            J12 = be.copy(J12)
            J21 = be.copy(J21)
            J22 = be.copy(J22)

            J11[idx] += Rx * dx / norm_sq
            J12[idx] += Rx * dy / norm_sq
            J21[idx] += Ry * dx / norm_sq
            J22[idx] += Ry * dy / norm_sq

            # Update 'ex' and 'ey' arrays for next iter
            ex = ex_new
            ey = ey_new

        converged = ex**2 + ey**2 < tol_sq
        return x, y, z, L, M, N, converged, had_initial_nan

    def _get_paraxial_jacobian(
        self, wavelength: float, stop_idx: int, is_inf: bool
    ) -> float:
        """Estimate the Jacobian (magnification) using paraxial trace.

        This method performs a paraxial ray trace to estimate the sensitivity of
        the stop height to changes in the initial ray parameter.

        Args:
            wavelength (float): The wavelength for the trace.
            stop_idx (int): The index of the stop surface.
            is_inf (bool): Whether the object is at infinity.

        Returns:
            float: The estimated Jacobian factor (dy_stop / d_param).
        """
        para = self.optic.paraxial
        if is_inf:
            z_start = para.surfaces.positions[1]
            y, _ = para.trace_generic(1.0, 0.0, z_start, wavelength, skip=1)
            return y[stop_idx]
        else:
            obj_z = self.optic.object_surface.geometry.cs.z
            y, _ = para.trace_generic(0.0, 1.0, obj_z, wavelength)
            return y[stop_idx]

    def _get_local_stop_coords(self, rays: RealRays, stop_idx: int) -> tuple:
        """Get ray intersection coordinates in the stop surface's local frame.

        After tracing, ray coordinates are in the global frame. This method
        transforms them back to the stop surface's local coordinate system
        so they can be compared against local-frame targets (Px*r, Py*r).

        Args:
            rays (RealRays): Traced rays (in global coordinates).
            stop_idx (int): The index of the stop surface.

        Returns:
            tuple: Local (x, y) coordinates on the stop surface.
        """
        stop_cs = self.optic.surfaces[stop_idx].geometry.cs

        # Create a temporary copy of rays to avoid mutating the originals
        temp = RealRays(
            be.copy(rays.x),
            be.copy(rays.y),
            be.copy(rays.z),
            be.copy(rays.L),
            be.copy(rays.M),
            be.copy(rays.N),
            intensity=be.copy(rays.i),
            wavelength=rays.w,
        )
        stop_cs.localize(temp)
        return temp.x, temp.y

    def _trace_subset(
        self,
        x: Any,
        y: Any,
        z: Any,
        L: Any,
        M: Any,
        N: Any,
        wl: Any,
        stop: int,
        is_inf: bool,
    ) -> RealRays:
        """Trace a subset of rays through the system up to the stop surface.

        Args:
            x, y, z: Ray positions.
            L, M, N: Ray direction cosines.
            wl: Wavelengths.
            stop (int): Index of the stop surface.
            is_inf (bool): Whether the object is at infinity (determines start surface).

        Returns:
            RealRays: The traced rays at the stop surface.
        """
        rays = RealRays(x, y, z, L, M, N, intensity=be.ones_like(x), wavelength=wl)
        start = 1 if is_inf else 0
        for i in range(start, stop + 1):
            self.optic.surfaces[i].trace(rays)
        return rays
