"""Iterative Ray Aiming Module

This module implements the iterative ray aiming algorithm with robust
derivative calculation for wide-angle systems.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland.paraxial_path import paraxial_seed_scope
from optiland.rays import RealRays
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.parameterization import (
    BoundLaunch,
    LaunchParameterization,
    SolveReport,
)
from optiland.rays.ray_aiming.paraxial import ParaxialRayAimer
from optiland.rays.ray_aiming.registry import register_aimer
from optiland.utils import machine_eps, solve_2x2

if TYPE_CHECKING:
    from optiland.optic import Optic


def _max_abs_residual(ex: Any, ey: Any) -> float:
    """Max-abs stop residual over rays, ignoring NaN (inf if all NaN)."""
    values = np.concatenate(
        [np.abs(be.to_numpy(ex)).reshape(-1), np.abs(be.to_numpy(ey)).reshape(-1)]
    )
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else float("inf")


# Maximum number of step halvings in the per-ray backtracking line search
# used by the Newton/Broyden core (see ``_solve_core``).
_MAX_BACKTRACK = 8


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
        self.last_report: SolveReport | None = None
        # Debug/report state written by _finite_difference_jacobian.
        self._last_fd_steps: tuple | None = None
        self._last_jacobian_fallback: Any = None

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
        # Scalar paraxial values are only seeds here -- the Newton polish
        # against real traces provides the exactness -- so out-of-domain
        # geometries warn instead of raising inside this scope.
        with paraxial_seed_scope():
            return self._aim_rays_scoped(
                fields, wavelengths, pupil_coords, initial_guess
            )

    def _aim_rays_scoped(
        self,
        fields: tuple,
        wavelengths: Any,
        pupil_coords: tuple,
        initial_guess: tuple | None = None,
    ) -> tuple:
        """Body of :meth:`aim_rays`, run inside the paraxial seed scope."""
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

        x, y, z, L, M, N, converged, had_initial_nan, report = self._solve_core(
            x, y, z, L, M, N, wavelengths, stop_idx, is_inf, tx, ty
        )
        self.last_report = report

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
        param: LaunchParameterization | None = None,
    ) -> tuple:
        """Core 2-DOF Newton/Broyden solve against an arbitrary local-stop
        target, without raising.

        This is the reusable solver core: it drives two true transverse
        parameters ``(xi, eta)`` (see
        :class:`~optiland.rays.ray_aiming.parameterization.LaunchParameterization`)
        so that the ray lands at local-stop coordinates ``(tx, ty)``. For
        infinite conjugates the launch point moves in the entry-frame
        transverse plane at fixed direction; for finite conjugates the
        object point is fixed and the direction rotates in a per-ray
        tangent basis, staying unit-norm through every trial. A
        displacement along the beam direction is never a solver degree of
        freedom. Unlike the public :meth:`aim_rays`, this never raises --
        NaNs and non-convergence are reported per-ray via the returned
        mask, so callers such as ``RobustRayAimer`` can treat individual
        ray failures gracefully instead of aborting the whole batch.

        Args:
            x, y, z, L, M, N: Initial ray launch guess (the seed state).
            wavelengths: Wavelengths of the rays.
            stop_idx: Index of the stop surface.
            is_inf: Whether the object is at infinity.
            tx, ty: Target local-stop coordinates for each ray.
            param: Optional prebuilt launch parameterization; built from
                the optic's entry frame when omitted.

        Returns:
            tuple: ``(x, y, z, L, M, N, converged, had_initial_nan,
            report)`` where ``converged`` is a per-ray boolean mask,
            ``had_initial_nan`` indicates whether the seed produced NaN
            errors for any ray, and ``report`` is a
            :class:`~optiland.rays.ray_aiming.parameterization.SolveReport`
            with seed/final residuals and iteration counts.
        """
        tol_sq = self.tol**2

        if param is None:
            param = LaunchParameterization.for_optic(self.optic, bool(is_inf))
        bound = param.bind(x, y, z, L, M, N)

        num_rays = len(bound.x0)
        xi = be.zeros(num_rays)
        eta = be.zeros(num_rays)

        # Initial trace (all rays); launch(0, 0) reproduces the seed state.
        x, y, z, L, M, N = bound.launch(xi, eta)
        rays = self._trace_subset(x, y, z, L, M, N, wavelengths, stop_idx, is_inf)
        lx, ly = self._get_local_stop_coords(rays, stop_idx)
        ex, ey = lx - tx, ly - ty

        had_initial_nan = bool(be.any(be.isnan(ex)) or be.any(be.isnan(ey)))
        seed_residual = _max_abs_residual(ex, ey)

        full_indices = be.arange_indices(num_rays)

        # Initialize the per-ray 2x2 Jacobian by central finite differences
        # on (xi, eta). A paraxial estimate is only a scalar magnitude (equal
        # on both axes, off-diagonal zero) and cannot represent the sign flip
        # or cross-coupling a tilted/decentered stop induces -- e.g. a 90 deg
        # fold makes d(ey)/d(eta) negative, so an assumed-positive diagonal
        # Jacobian steps the wrong way and Broyden then diverges to NaN
        # (issue #654). Four extra traces capture the true local response.
        step_scale = self._fd_step_scale(is_inf, tx, ty)
        J11, J12, J21, J22 = self._finite_difference_jacobian(
            bound, xi, eta, wavelengths, stop_idx, is_inf, lx, ly, step_scale
        )
        fallback_used = bool(be.any(self._last_jacobian_fallback))
        jacobian_refreshes = 0
        # Rays whose Jacobian stayed ill-conditioned even after a refresh
        # and the paraxial substitution: hold them (zero step) without
        # re-triggering the refresh ladder every iteration.
        hopeless = be.zeros(num_rays) > 0.0

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

            # Newton step dp = -J^-1 e via the shared, scale-invariant,
            # conditioning-aware 2x2 solve: the determinant's sign is
            # preserved for every valid ray (never clamped to an arbitrary
            # positive value), so the step direction cannot be silently
            # reversed.
            solve = solve_2x2(J11[idx], J12[idx], J21[idx], J22[idx], ex_curr, ey_curr)
            needs_ladder = be.logical_and(
                be.logical_not(solve.valid),
                be.logical_not(hopeless[idx]),
            )
            if bool(be.any(needs_ladder)):
                # Iteration-time conditioning ladder: (1) refresh the
                # ill-conditioned active rays with a fresh central-difference
                # Jacobian at the current state and re-evaluate; (2) if still
                # ill-conditioned, substitute the sign-preserving paraxial
                # diagonal and report the fallback; (3) if even that is
                # singular, hold the ray (zero step) and let it surface as
                # non-converged -- a step is never fabricated.
                jacobian_refreshes += 1
                F11, F12, F21, F22 = self._finite_difference_jacobian(
                    bound,
                    xi,
                    eta,
                    wavelengths,
                    stop_idx,
                    is_inf,
                    None,
                    None,
                    step_scale,
                )
                fallback_used = fallback_used or bool(
                    be.any(self._last_jacobian_fallback[idx][needs_ladder])
                )
                bad_idx = idx[needs_ladder]
                J11 = be.copy(J11)
                J12 = be.copy(J12)
                J21 = be.copy(J21)
                J22 = be.copy(J22)
                J11[bad_idx] = F11[bad_idx]
                J12[bad_idx] = F12[bad_idx]
                J21[bad_idx] = F21[bad_idx]
                J22[bad_idx] = F22[bad_idx]
                solve = solve_2x2(
                    J11[idx], J12[idx], J21[idx], J22[idx], ex_curr, ey_curr
                )

                still_bad = be.logical_and(be.logical_not(solve.valid), needs_ladder)
                if bool(be.any(still_bad)):
                    fallback_used = True
                    wl_mean = (
                        be.mean(wavelengths)
                        if hasattr(wavelengths, "__len__")
                        else wavelengths
                    )
                    j_par = float(
                        be.to_numpy(
                            self._get_paraxial_jacobian(
                                float(wl_mean), stop_idx, is_inf
                            )
                        ).ravel()[0]
                    )
                    sub_idx = idx[still_bad]
                    J11[sub_idx] = j_par
                    J22[sub_idx] = j_par
                    J12[sub_idx] = 0.0
                    J21[sub_idx] = 0.0
                    solve = solve_2x2(
                        J11[idx], J12[idx], J21[idx], J22[idx], ex_curr, ey_curr
                    )
                    # Anything still invalid is genuinely degenerate: mark it
                    # so later iterations skip the ladder for it.
                    hopeless = be.copy(hopeless)
                    hopeless[idx] = be.logical_or(
                        hopeless[idx],
                        be.logical_and(be.logical_not(solve.valid), still_bad),
                    )

            dp1 = -solve.x1
            dp2 = -solve.x2

            # --- Damped update with per-ray backtracking line search ---
            # A full Newton/Broyden step can overshoot into a region where a
            # ray misses a surface (NaN error) or the error simply grows.
            # Halving the step per ray until the error strictly decreases keeps
            # a single bad step from poisoning a ray into permanent NaN -- the
            # divergence/NaN failure mode in issue #654. A ray that cannot
            # improve holds its last finite state (accepted step 0) and is
            # reported as non-converged rather than as NaN.
            p1_base = be.copy(xi[idx])
            p2_base = be.copy(eta[idx])

            old_err_sq = ex_curr**2 + ey_curr**2
            alpha = be.ones_like(ex_curr)
            acc_dp1 = be.zeros_like(ex_curr)
            acc_dp2 = be.zeros_like(ey_curr)
            acc_ex = be.copy(ex_curr)
            acc_ey = be.copy(ey_curr)
            searching = be.ones_like(ex_curr) > 0.0

            for _bt in range(_MAX_BACKTRACK):
                xi = be.copy(xi)
                eta = be.copy(eta)
                xi[idx] = p1_base + alpha * dp1
                eta[idx] = p2_base + alpha * dp2

                x, y, z, L, M, N = bound.launch(xi, eta)
                rays = self._trace_subset(
                    x, y, z, L, M, N, wavelengths, stop_idx, is_inf
                )
                lx, ly = self._get_local_stop_coords(rays, stop_idx)
                ex_try = lx[idx] - tx[idx]
                ey_try = ly[idx] - ty[idx]
                new_err_sq = ex_try**2 + ey_try**2

                improved = be.logical_and(
                    searching,
                    be.logical_and(
                        be.logical_not(be.isnan(new_err_sq)),
                        new_err_sq < old_err_sq,
                    ),
                )
                acc_dp1 = be.where(improved, alpha * dp1, acc_dp1)
                acc_dp2 = be.where(improved, alpha * dp2, acc_dp2)
                acc_ex = be.where(improved, ex_try, acc_ex)
                acc_ey = be.where(improved, ey_try, acc_ey)
                searching = be.logical_and(searching, be.logical_not(improved))
                if not be.any(searching):
                    break
                alpha = alpha * 0.5

            # Commit the accepted (possibly zero) step for each active ray.
            xi = be.copy(xi)
            eta = be.copy(eta)
            xi[idx] = p1_base + acc_dp1
            eta[idx] = p2_base + acc_dp2

            # --- Broyden Update (using the accepted step) ---
            # J += (y - J*s) * s^T / (s^T * s)
            dEx = acc_ex - ex_curr
            dEy = acc_ey - ey_curr

            dx = acc_dp1
            dy = acc_dp2

            # Calculate J*s (using OLD J)
            Js_x = J11[idx] * dx + J12[idx] * dy
            Js_y = J21[idx] * dx + J22[idx] * dy

            Rx = dEx - Js_x
            Ry = dEy - Js_y

            # Norm sq of step s. The update is skipped entirely for a
            # zero/unaccepted step (the line search committed nothing) or a
            # step at round-off level relative to the local parameter scale
            # -- dividing such a step's residual by its vanishing norm would
            # amplify pure noise into J. The floor is dtype- and
            # scale-aware: |s| <= eps_mach * max(S, |xi|, |eta|) is
            # indistinguishable from rounding of the parameters themselves.
            norm_sq = dx**2 + dy**2
            step_ref = be.maximum(
                be.maximum(be.abs(p1_base), be.abs(p2_base)),
                step_scale,
            )
            floor_sq = (machine_eps(dx) * step_ref) ** 2
            do_update = norm_sq > floor_sq
            safe_norm_sq = be.where(do_update, norm_sq, be.ones_like(norm_sq))
            zero_upd = be.zeros_like(norm_sq)

            # Update J (Avoid in-place leaf errors by copying first)
            J11 = be.copy(J11)
            J12 = be.copy(J12)
            J21 = be.copy(J21)
            J22 = be.copy(J22)

            J11[idx] += be.where(do_update, Rx * dx / safe_norm_sq, zero_upd)
            J12[idx] += be.where(do_update, Rx * dy / safe_norm_sq, zero_upd)
            J21[idx] += be.where(do_update, Ry * dx / safe_norm_sq, zero_upd)
            J22[idx] += be.where(do_update, Ry * dy / safe_norm_sq, zero_upd)

            # Write the accepted errors back for the next iteration.
            ex = be.copy(ex)
            ey = be.copy(ey)
            ex[idx] = acc_ex
            ey[idx] = acc_ey

        converged = ex**2 + ey**2 < tol_sq
        x, y, z, L, M, N = bound.launch(xi, eta)
        num_converged = int(be.to_numpy(converged).reshape(-1).sum())
        report = SolveReport(
            seed_residual=seed_residual,
            final_residual=_max_abs_residual(ex, ey),
            converged=bool(be.all(converged)),
            iterations=self.last_iterations,
            num_rays=num_rays,
            num_converged=num_converged,
            fallback_used=fallback_used,
            jacobian_refreshes=jacobian_refreshes,
        )
        return x, y, z, L, M, N, converged, had_initial_nan, report

    def _fd_step_scale(self, is_inf: bool, tx: Any, ty: Any) -> float:
        """Characteristic parameter scale ``S`` for finite-difference steps.

        For infinite conjugates the solver parameters ``(xi, eta)`` are
        lengths, so ``S`` is a physical stop scale: the largest target
        magnitude of the batch (edge probes and pupil batches carry the
        stop radius), floored at 1 mm for pure chief batches whose targets
        are all zero. For finite conjugates the parameters are
        dimensionless tangent perturbations of a unit direction, so the
        local scale is 1.
        """
        if not is_inf:
            return 1.0
        values = np.concatenate(
            [
                np.abs(be.to_numpy(tx)).reshape(-1),
                np.abs(be.to_numpy(ty)).reshape(-1),
            ]
        )
        finite = values[np.isfinite(values)]
        scale = float(finite.max()) if finite.size else 0.0
        return max(scale, 1.0)

    def _finite_difference_jacobian(
        self,
        bound: BoundLaunch,
        xi: Any,
        eta: Any,
        wavelengths: Any,
        stop_idx: int,
        is_inf: bool,
        lx: Any,
        ly: Any,
        step_scale: float = 1.0,
    ) -> tuple:
        """Per-ray 2x2 Jacobian ``d(local stop x, y)/d(xi, eta)`` by central
        finite differences.

        The free degrees of freedom are the two transverse launch parameters
        of the bound :class:`LaunchParameterization` -- both are independent
        transverse coordinates for any entry direction, so neither Jacobian
        column can vanish just because the system is entered off the z axis.
        Unlike the paraxial magnitude estimate, this captures the sign and
        cross-coupling of tilted or decentered stops, which is required for
        the Newton step to be a descent direction on such systems (issue
        #654).

        The differences are central with dtype- and scale-aware per-ray
        steps ``h = eps_mach^(1/3) * max(S, |parameter|)`` -- the cube root
        is the standard optimum balancing a central difference's truncation
        error against round-off, and ``S`` comes from
        :meth:`_fd_step_scale`. The selected steps are recorded on
        ``self._last_fd_steps`` for debugging.

        Rays whose perturbed traces are unusable (a perturbed ray missed a
        surface -> NaN entries) fall back to the sign-preserving paraxial
        diagonal -- the raw paraxial response, never clamped to an
        arbitrary magnitude -- and are recorded on
        ``self._last_jacobian_fallback`` so the solve core can report the
        substitution. Ill-conditioning is *not* judged here: the Newton
        core evaluates a scale-invariant reciprocal condition at solve time
        and refreshes/falls back explicitly.

        Args:
            bound: The bound launch parameterization for this solve.
            xi, eta: Current solver parameters.
            wavelengths: Ray wavelengths.
            stop_idx: Index of the stop surface.
            is_inf: Whether the object is at infinity.
            lx, ly: Unperturbed local-stop coordinates of the current state
                (kept for signature stability; the central difference does
                not evaluate the center point).
            step_scale: Characteristic parameter scale ``S``.

        Returns:
            tuple: ``(J11, J12, J21, J22)`` per-ray Jacobian entries, with
            ``J = [[d lx/d xi, d lx/d eta], [d ly/d xi, d ly/d eta]]``.
        """
        del lx, ly  # central differences do not use the center point
        eps_mach = machine_eps(xi)
        h0 = eps_mach ** (1.0 / 3.0)
        h_xi = h0 * be.maximum(be.abs(xi), step_scale)
        h_eta = h0 * be.maximum(be.abs(eta), step_scale)
        self._last_fd_steps = (h_xi, h_eta)

        # The four probe bundles are concatenated into one trace: every step
        # of the pipeline (launch, surface trace, local stop coordinates) is
        # elementwise per ray, so the batched results are identical to four
        # separate traces while paying the per-surface Python overhead once
        # instead of four times.
        probes = (
            bound.launch(xi + h_xi, eta),
            bound.launch(xi - h_xi, eta),
            bound.launch(xi, eta + h_eta),
            bound.launch(xi, eta - h_eta),
        )
        batched = tuple(
            be.concatenate([launch[i] for launch in probes]) for i in range(6)
        )
        wl = wavelengths
        n = batched[0].shape[0] // 4
        if getattr(wl, "shape", None) and wl.shape and wl.shape[0] == n and n > 1:
            wl = be.concatenate([wl, wl, wl, wl])
        rays = self._trace_subset(*batched, wl, stop_idx, is_inf)
        lx_all, ly_all = self._get_local_stop_coords(rays, stop_idx)

        lx_xp, lx_xm, lx_ep, lx_em = (
            lx_all[0:n],
            lx_all[n : 2 * n],
            lx_all[2 * n : 3 * n],
            lx_all[3 * n :],
        )
        ly_xp, ly_xm, ly_ep, ly_em = (
            ly_all[0:n],
            ly_all[n : 2 * n],
            ly_all[2 * n : 3 * n],
            ly_all[3 * n :],
        )

        J11 = (lx_xp - lx_xm) / (2.0 * h_xi)
        J21 = (ly_xp - ly_xm) / (2.0 * h_xi)
        J12 = (lx_ep - lx_em) / (2.0 * h_eta)
        J22 = (ly_ep - ly_em) / (2.0 * h_eta)

        # Sign-preserving paraxial diagonal for rays whose finite difference
        # is unusable (NaN from a missed surface). The raw paraxial response
        # keeps its sign and magnitude; if it is itself degenerate the
        # conditioning-aware solve will hold the ray and report
        # non-convergence rather than fabricate a step.
        finite = be.logical_and(
            be.logical_and(be.isfinite(J11), be.isfinite(J12)),
            be.logical_and(be.isfinite(J21), be.isfinite(J22)),
        )
        fallback = be.logical_not(finite)
        if bool(be.any(fallback)):
            num_rays = len(bound.x0)
            wl_mean = (
                be.mean(wavelengths) if hasattr(wavelengths, "__len__") else wavelengths
            )
            j_par = float(
                be.to_numpy(
                    self._get_paraxial_jacobian(float(wl_mean), stop_idx, is_inf)
                ).ravel()[0]
            )
            j_par_arr = be.full(num_rays, j_par)
            zeros = be.zeros(num_rays)
            J11 = be.where(fallback, j_par_arr, J11)
            J22 = be.where(fallback, j_par_arr, J22)
            J12 = be.where(fallback, zeros, J12)
            J21 = be.where(fallback, zeros, J21)
        self._last_jacobian_fallback = fallback
        return J11, J12, J21, J22

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
            # skip=1 drops the object surface, so the returned array is indexed
            # by (surface_index - skip): surface k lives at heights[k - 1].
            # Indexing with the bare stop_idx reads the surface *after* the
            # stop, which collapses to ~0 whenever that surface sits near a
            # focus (stop on/near the last surface) -- yielding a near-zero
            # Jacobian and a Newton step that overshoots to NaN (issue #654).
            skip = 1
            z_start = para.surfaces.positions[1]
            y, _ = para.trace_generic(1.0, 0.0, z_start, wavelength, skip=skip)
            return y[stop_idx - skip]
        else:
            # Unfolded axial position of the object -- identical to its
            # global cs.z on the canonical path, and frame-correct for
            # folded/off-axis entries.
            obj_z = para.surfaces.positions[0, 0]
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
