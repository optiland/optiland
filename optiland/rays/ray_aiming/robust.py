"""Robust Ray Aiming Module

This module implements a chief-ray calibrated robust ray aiming algorithm.

For each field, a cheap chief-ray solve plus four cardinal edge probes
(``(+-1, 0)``, ``(0, +-1)`` on the stop) are fit to a 2x2 affine launch
model (see ``pupil_map.py``). That model seeds every requested pupil point,
which is then driven to its exact target by the same Newton/Broyden polish
used by ``IterativeRayAimer``. Calibration is warm-started from the
previous fit (or the nearest already-solved field) rather than a paraxial
guess extrapolated across the whole field range, which is what allows this
to converge cold at extreme field angles without the recursive homotopy
subdivision the previous implementation relied on.

See ``optiland/jupyter/SPEC_ray_aiming_20260703.md`` for the full design.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.initialization import get_stop_radius_strategy
from optiland.rays.ray_aiming.iterative import IterativeRayAimer
from optiland.rays.ray_aiming.pupil_map import PupilMap, PupilMapCache, to_float
from optiland.rays.ray_aiming.registry import register_aimer

if TYPE_CHECKING:
    from optiland.optic import Optic

# Cardinal edge probes on the stop, in (Px, Py) order: east, west, north, south.
_EDGE_PROBES = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))


@contextlib.contextmanager
def _cached_paraxial_constants(optic: Optic):
    """Temporarily memoize ``Paraxial.EPD``/``EPL`` on this optic.

    Both are system-wide constants (independent of field/pupil), but each
    call re-traces the whole system. The chief-ray field-marching fallback
    (:meth:`RobustRayAimer._march_chief`) calls ``ParaxialRayAimer.aim_rays``
    -- which calls these -- once per marching attempt, and a cold extreme
    field can need dozens of attempts; without caching, that cost dominates
    total aiming time. Scoped and reversible: the original bound methods are
    restored on exit, so this never leaks stale values past one calibration.
    """
    para = optic.paraxial
    orig_epd = para.EPD
    orig_epl = para.EPL
    cache: dict[str, Any] = {}

    def cached_epd():
        if "epd" not in cache:
            cache["epd"] = orig_epd()
        return cache["epd"]

    def cached_epl():
        if "epl" not in cache:
            cache["epl"] = orig_epl()
        return cache["epl"]

    para.EPD = cached_epd
    para.EPL = cached_epl
    try:
        yield
    finally:
        para.EPD = orig_epd
        para.EPL = orig_epl


@contextlib.contextmanager
def _relaxed_tolerance(iterative, tol: float):
    """Temporarily loosen ``iterative.tol`` for cheap intermediate solves.

    Used only by the chief-ray marching fallback: an intermediate marching
    step just needs to be "good enough" to warm-start the next one, since
    the final per-ray polish (elsewhere, always at full tolerance) is what
    actually guarantees exactness (D1). Tighter-than-needed intermediate
    tolerance costs several extra Newton iterations per step for no
    accuracy benefit that survives to the final result.
    """
    orig_tol = iterative.tol
    iterative.tol = max(tol, orig_tol)
    try:
        yield
    finally:
        iterative.tol = orig_tol


@register_aimer("robust")
class RobustRayAimer(BaseRayAimer):
    """Chief-ray calibrated robust ray aiming algorithm.

    Designed to handle challenging optical systems (wide-angle, fisheye)
    where a cold paraxial seed for the iterative solver is too far from the
    real solution to converge directly. Per field, a chief-ray calibration
    (§4.2 of the spec) produces a cheap affine seed model; every requested
    ray is then polished to exactness (§4.3) via the reused
    ``IterativeRayAimer`` Newton/Broyden core. Individual ray failures
    (vignetting, TIR) are reported as NaN rather than aborting the batch.

    Attributes:
        optic (Optic): The optical system instance.
        max_iter (int): Maximum number of iterations for the internal solver.
        tol (float): Numerical tolerance for convergence.
        scale_fields (bool): Retained for constructor-signature stability;
            the calibration-based algorithm does not use homotopy
            field-scaling, so this is a no-op.
    """

    def __init__(
        self,
        optic: Optic,
        max_iter: int = 20,
        tol: float = 1e-8,
        scale_fields: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the RobustRayAimer.

        Args:
            optic (Optic): The optical system to aim rays for.
            max_iter (int, optional): Maximum number of iterations. Defaults to 20.
            tol (float, optional): Error tolerance for convergence. Defaults to 1e-8.
            scale_fields (bool, optional): No-op, retained for backward
                compatibility. Defaults to True.
            **kwargs: Additional keyword arguments passed to BaseRayAimer.
        """
        super().__init__(optic, **kwargs)
        self.scale_fields = scale_fields
        self.max_iter = max_iter
        self.tol = tol
        self._iterative = IterativeRayAimer(optic, max_iter=max_iter, tol=tol)
        self._paraxial = self._iterative._paraxial_aimer
        self._cache = PupilMapCache()

    def aim_rays(
        self,
        fields: tuple,
        wavelengths: Any,
        pupil_coords: tuple,
        initial_guess: tuple | None = None,
    ) -> tuple:
        """Calculate ray starting coordinates using chief-ray calibration.

        Args:
            fields (tuple): Field coordinates ``(Hx, Hy)``.
            wavelengths (Any): Wavelengths in microns.
            pupil_coords (tuple): Normalized pupil coordinates ``(Px, Py)``.
            initial_guess (tuple | None, optional): Optional starting guess.
                If provided, the method first attempts to solve directly
                using the iterative solver with this guess; only on failure
                does it fall back to the full calibrated solve below.

        Returns:
            tuple: Solved ray parameters ``(x, y, z, L, M, N)``.

        Raises:
            ValueError: If every ray for a field fails to converge (a
                misconfiguration, not ordinary partial vignetting).
        """
        if initial_guess is not None:
            try:
                return self._iterative.aim_rays(
                    fields, wavelengths, pupil_coords, initial_guess=initial_guess
                )
            except ValueError:
                # Fall through to the calibrated solve below.
                pass

        Px, Py = pupil_coords
        Px = be.as_array_1d(Px)
        Py = be.as_array_1d(Py)
        n = len(Px)

        Hx, Hy = fields
        Hx = be.as_array_1d(Hx)
        Hy = be.as_array_1d(Hy)
        if len(Hx) == 1 and n > 1:
            Hx = Hx * be.ones(n)
        if len(Hy) == 1 and n > 1:
            Hy = Hy * be.ones(n)

        if hasattr(wavelengths, "__len__"):
            wl_arr = be.as_array_1d(wavelengths)
            if len(wl_arr) == 1 and n > 1:
                wl_arr = wl_arr * be.ones(n)
        else:
            wl_arr = be.ones(n) * float(wavelengths)

        Hx_list = be.to_numpy(Hx).reshape(-1).tolist()
        Hy_list = be.to_numpy(Hy).reshape(-1).tolist()
        wl_list = be.to_numpy(wl_arr).reshape(-1).tolist()

        stop_idx = self.optic.surfaces.stop_index
        is_inf = getattr(self.optic.object_surface, "is_infinite", False)
        r_stop = get_stop_radius_strategy(self.optic, "robust").calculate_stop_radius()

        self._cache.sync(self.optic)

        # Group rays by field (D3: reuse the same pupil map across pupil
        # distributions for the same field), then process fields ordered
        # by radial magnitude so later (larger) fields can warm-start from
        # already-solved smaller ones (D8 field marching).
        groups: dict[tuple[float, float, float], list[int]] = {}
        for i in range(n):
            key = (Hx_list[i], Hy_list[i], wl_list[i])
            groups.setdefault(key, []).append(i)

        ordered_keys = sorted(groups, key=lambda k: k[0] ** 2 + k[1] ** 2)

        order_parts: list[list[int]] = []
        x_parts: list[Any] = []
        y_parts: list[Any] = []
        z_parts: list[Any] = []
        L_parts: list[Any] = []
        M_parts: list[Any] = []
        N_parts: list[Any] = []

        for key in ordered_keys:
            idx = groups[key]
            Hxk, Hyk, wlk = key

            pmap = self._cache.get_fresh(Hxk, Hyk, wlk)
            if pmap is None:
                seed_map = self._cache.get_stale(Hxk, Hyk, wlk)
                if seed_map is None:
                    seed_map = self._cache.nearest(Hxk, Hyk)
                with _cached_paraxial_constants(self.optic):
                    pmap = self._calibrate_field(
                        Hxk, Hyk, wlk, stop_idx, is_inf, r_stop, seed_map
                    )
                self._cache.put(Hxk, Hyk, wlk, pmap)

            Px_g = Px[idx]
            Py_g = Py[idx]
            wl_g = wl_arr[idx]

            x0, y0, z0, L0, M0, N0 = pmap.seed(Px_g, Py_g)
            tx = Px_g * r_stop
            ty = Py_g * r_stop

            x, y, z, L, M, N, converged, _ = self._iterative._solve_core(
                x0, y0, z0, L0, M0, N0, wl_g, stop_idx, is_inf, tx, ty
            )

            if not be.any(converged):
                raise ValueError(
                    "RobustRayAimer: every ray failed to converge for field "
                    f"(Hx={Hxk}, Hy={Hyk}); check the system configuration."
                )

            # Renormalize direction cosines (G3 invariant).
            norm = be.sqrt(L**2 + M**2 + N**2)
            L = L / norm
            M = M / norm
            N = N / norm

            # Graceful per-ray failure (D6): non-converged rays -> NaN.
            x = be.where(converged, x, be.nan)
            y = be.where(converged, y, be.nan)
            z = be.where(converged, z, be.nan)
            L = be.where(converged, L, be.nan)
            M = be.where(converged, M, be.nan)
            N = be.where(converged, N, be.nan)

            order_parts.append(idx)
            x_parts.append(x)
            y_parts.append(y)
            z_parts.append(z)
            L_parts.append(L)
            M_parts.append(M)
            N_parts.append(N)

        order = [i for part in order_parts for i in part]
        inv_perm = [0] * n
        for pos, orig in enumerate(order):
            inv_perm[orig] = pos

        x_out = be.concatenate(x_parts)[inv_perm]
        y_out = be.concatenate(y_parts)[inv_perm]
        z_out = be.concatenate(z_parts)[inv_perm]
        L_out = be.concatenate(L_parts)[inv_perm]
        M_out = be.concatenate(M_parts)[inv_perm]
        N_out = be.concatenate(N_parts)[inv_perm]

        return x_out, y_out, z_out, L_out, M_out, N_out

    def _calibrate_field(
        self,
        Hx: float,
        Hy: float,
        wl: float,
        stop_idx: int,
        is_inf: bool,
        r_stop: float,
        seed_map: PupilMap | None,
    ) -> PupilMap:
        """Chief solve + 4 edge probes -> affine :class:`PupilMap` (§4.2)."""
        chief = self._solve_chief(Hx, Hy, wl, stop_idx, is_inf, seed_map)
        probes = [
            self._solve_probe(wl, stop_idx, is_inf, px, py, r_stop, chief)
            for px, py in _EDGE_PROBES
        ]
        return self._fit_affine(chief, probes, is_inf)

    def _solve_chief(
        self,
        Hx: float,
        Hy: float,
        wl: float,
        stop_idx: int,
        is_inf: bool,
        seed_map: PupilMap | None,
    ) -> tuple[float, float, float, float, float, float]:
        """Solve the chief ray (stop target (0, 0)) for this field.

        Seed order: warm-started map for this field or the nearest
        already-solved field, then a direct paraxial guess. If both fail --
        the paraxial seed can be too far from the real solution at extreme
        field angles to converge in one Newton solve -- fall back to
        marching the chief ray outward in field angle from the axis
        (:meth:`_march_chief`), which is what makes a *cold* extreme-field
        solve (e.g. WideAngle170FOV) converge without recursive subdivision.

        The fixed launch components (direction for infinite conjugates,
        object position for finite ones) always come fresh from *this*
        field's paraxial trace, never from ``seed_map`` -- they encode the
        field angle itself, so reusing another field's fixed components
        would silently solve the wrong (e.g. on-axis) problem even though
        Newton still converges. Only the free 2-DOF is warm-started from
        the seed map's chief launch.
        """
        wl_a = be.array([wl])
        tx = be.array([0.0])
        ty = be.array([0.0])

        px0, py0, pz0, pL0, pM0, pN0 = self._paraxial.aim_rays(
            (be.array([Hx]), be.array([Hy])),
            wl_a,
            (be.array([0.0]), be.array([0.0])),
        )

        if seed_map is not None:
            sx0, sy0, _sz0, sL0, sM0, _sN0 = seed_map.seed(
                be.array([0.0]), be.array([0.0])
            )
            if is_inf:
                x0, y0 = sx0, sy0
                z0, L0, M0, N0 = pz0, pL0, pM0, pN0
            else:
                L0, M0 = sL0, sM0
                x0, y0, z0, N0 = px0, py0, pz0, pN0

            x, y, z, L, M, N, converged, _ = self._iterative._solve_core(
                x0, y0, z0, L0, M0, N0, wl_a, stop_idx, is_inf, tx, ty
            )
            if be.any(converged):
                return (
                    to_float(x),
                    to_float(y),
                    to_float(z),
                    to_float(L),
                    to_float(M),
                    to_float(N),
                )

        x, y, z, L, M, N, converged, _ = self._iterative._solve_core(
            px0, py0, pz0, pL0, pM0, pN0, wl_a, stop_idx, is_inf, tx, ty
        )
        if be.any(converged):
            return (
                to_float(x),
                to_float(y),
                to_float(z),
                to_float(L),
                to_float(M),
                to_float(N),
            )

        marched = self._march_chief(Hx, Hy, wl_a, stop_idx, is_inf, tx, ty)
        if marched is None:
            raise ValueError(
                f"RobustRayAimer: chief ray failed to converge for field "
                f"(Hx={Hx}, Hy={Hy}) after marching from the axis; check "
                f"the system configuration."
            )
        return marched

    def _march_chief(
        self,
        Hx: float,
        Hy: float,
        wl_a: Any,
        stop_idx: int,
        is_inf: bool,
        tx: Any,
        ty: Any,
        max_attempts: int = 150,
        min_dt: float = 1e-4,
    ) -> tuple[float, float, float, float, float, float] | None:
        """March the chief ray from the axis out to (Hx, Hy) in field angle.

        A step-halving walk -- each step a single-ray 2-DOF solve
        warm-started from the *last successfully converged* launch, never
        from a failed one -- replaces the old recursive homotopy as the
        cold-start robustness mechanism (D8). It is bounded (a fixed attempt
        budget, no recursion) and physically monotonic: only the free launch
        DOF carries over between steps, while the fixed DOF (z, and
        direction for infinite conjugates / object position for finite ones)
        is refreshed from the paraxial trace at each step's actual field
        angle.

        A step size is never grown back up after a success: this system's
        maximum reliable step tends to shrink (never grow) as the field
        angle increases, so re-attempting a larger step every time just
        wastes evaluations that repeatedly fail the same way.

        Returns ``None`` if the walk cannot reach ``t=1`` (the actual
        target field) within the attempt budget -- the caller must treat
        this as a hard failure, not silently accept whatever intermediate
        field angle happened to converge. Returning a wrong-but-converged
        intermediate result here is exactly the failure mode this method
        exists to prevent (see SPEC_ray_aiming_20260703.md D8): the fixed
        launch DOF encodes the field angle itself, so a caller that used a
        partial march's result as the final chief ray would be aiming at
        the wrong field entirely, not just aiming imprecisely.
        """
        t = 0.0
        # t=0 (the axis) is trivial and always converges: L=M=0, N=+-1.
        launch = self._paraxial.aim_rays(
            (be.array([0.0]), be.array([0.0])), wl_a, (be.array([0.0]), be.array([0.0]))
        )
        launch = tuple(to_float(v) for v in launch)

        dt = 1.0
        relaxed_tol = max(self._iterative.tol, 1e-4)
        for _attempt in range(max_attempts):
            if t >= 1.0:
                # Intermediate steps used a relaxed tolerance as a cheap
                # warm-start; do one final full-tolerance solve so the
                # chief anchor itself is exact, not just "close enough".
                x0 = be.array([launch[0]])
                y0 = be.array([launch[1]])
                z0 = be.array([launch[2]])
                L0 = be.array([launch[3]])
                M0 = be.array([launch[4]])
                N0 = be.array([launch[5]])
                x, y, z, L, M, N, converged, _ = self._iterative._solve_core(
                    x0, y0, z0, L0, M0, N0, wl_a, stop_idx, is_inf, tx, ty
                )
                if be.any(converged):
                    return (
                        to_float(x),
                        to_float(y),
                        to_float(z),
                        to_float(L),
                        to_float(M),
                        to_float(N),
                    )
                return launch

            t_next = min(t + dt, 1.0)
            Hxt, Hyt = Hx * t_next, Hy * t_next
            px0, py0, pz0, pL0, pM0, pN0 = self._paraxial.aim_rays(
                (be.array([Hxt]), be.array([Hyt])),
                wl_a,
                (be.array([0.0]), be.array([0.0])),
            )

            if is_inf:
                x0 = be.array([launch[0]])
                y0 = be.array([launch[1]])
                z0, L0, M0, N0 = pz0, pL0, pM0, pN0
            else:
                L0 = be.array([launch[3]])
                M0 = be.array([launch[4]])
                x0, y0, z0, N0 = px0, py0, pz0, pN0

            with _relaxed_tolerance(self._iterative, relaxed_tol):
                x, y, z, L, M, N, converged, _ = self._iterative._solve_core(
                    x0, y0, z0, L0, M0, N0, wl_a, stop_idx, is_inf, tx, ty
                )

            if be.any(converged):
                launch = (
                    to_float(x),
                    to_float(y),
                    to_float(z),
                    to_float(L),
                    to_float(M),
                    to_float(N),
                )
                t = t_next
                # Do not grow dt back up -- see docstring.
            else:
                # Retry the SAME target angle at half the step, warm-started
                # from the last known-good launch -- never advance t on a
                # failed step.
                dt /= 2.0
                if dt < min_dt:
                    return None

        return launch if t >= 1.0 else None

    def _solve_probe(
        self,
        wl: float,
        stop_idx: int,
        is_inf: bool,
        Px_e: float,
        Py_e: float,
        r_stop: float,
        chief: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float, float, float, float, float]:
        """Solve one cardinal edge probe, seeded from the chief launch."""
        x0, y0, z0, L0, M0, N0 = (be.array([v]) for v in chief)
        wl_a = be.array([wl])
        tx = be.array([Px_e * r_stop])
        ty = be.array([Py_e * r_stop])

        x, y, z, L, M, N, converged, _ = self._iterative._solve_core(
            x0, y0, z0, L0, M0, N0, wl_a, stop_idx, is_inf, tx, ty
        )

        if not be.any(converged):
            # An unreachable edge probe still yields a usable (if less
            # accurate) linear seed by falling back to the chief launch,
            # rather than propagating NaN into the affine fit.
            return chief

        return (
            to_float(x),
            to_float(y),
            to_float(z),
            to_float(L),
            to_float(M),
            to_float(N),
        )

    def _fit_affine(
        self,
        chief: tuple[float, float, float, float, float, float],
        probes: list[tuple[float, float, float, float, float, float]],
        is_inf: bool,
    ) -> PupilMap:
        """Fit the 2x2 affine launch model from the chief ray + 4 probes."""
        x_c, y_c, z_c, L_c, M_c, N_c = chief
        p_east, p_west, p_north, p_south = probes

        def free(v: tuple) -> tuple[float, float]:
            x, y, _z, L, M, _N = v
            return (x, y) if is_inf else (L, M)

        c1, c2 = free(chief)
        e1, e2 = free(p_east)
        w1, w2 = free(p_west)
        n1, n2 = free(p_north)
        s1, s2 = free(p_south)

        A = (
            ((e1 - w1) / 2.0, (n1 - s1) / 2.0),
            ((e2 - w2) / 2.0, (n2 - s2) / 2.0),
        )

        fixed = (z_c, L_c, M_c, N_c) if is_inf else (z_c, x_c, y_c, N_c)

        return PupilMap(c=(c1, c2), A=A, is_infinite=is_inf, fixed=fixed)
