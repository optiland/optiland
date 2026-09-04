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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.paraxial_path import paraxial_seed_scope
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.initialization import get_stop_radius_strategy
from optiland.rays.ray_aiming.iterative import IterativeRayAimer
from optiland.rays.ray_aiming.parameterization import (
    LaunchParameterization,
    SolveReport,
)
from optiland.rays.ray_aiming.pupil_map import PupilMap, PupilMapCache, to_float
from optiland.rays.ray_aiming.registry import register_aimer

if TYPE_CHECKING:
    from optiland.optic import Optic

# Cardinal edge probes on the stop, in (Px, Py) order: east, west, north, south.
_EDGE_PROBES = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))


@dataclass(frozen=True)
class RobustFieldReport:
    """Per-field outcome of one robust aiming call.

    Attributes:
        Hx, Hy: Normalized field coordinates of the group.
        wavelength: Wavelength of the group in micrometers.
        final_polish: :class:`SolveReport` of the group's final Newton
            polish (or of the failed chief solve when calibration itself
            failed).
        chief_seed_strategy: How the chief anchor was obtained --
            ``"initial_guess"`` (caller-supplied guess solved directly),
            ``"cached_map"`` (fresh cached map reused, no calibration),
            ``"warm_map"`` (chief solved from a stale/nearest cached map),
            ``"direct_paraxial"`` (chief solved from the fresh paraxial
            seed), ``"marching"`` (field-marching fallback), ``"scan"``
            (transverse scan fallback) or ``"failed"``.
        used_cached_map: Whether a fresh cached pupil map was reused
            without recalibration.
        edge_probe_fallbacks: Number of cardinal edge probes that failed to
            converge and fell back to the chief launch in the affine fit.
        calibration_used: Whether a fresh chief-plus-probes calibration ran
            for this field.
        fallback_used: Whether any substitute strategy was used for this
            field: a failed caller-supplied guess, chief marching or scan,
            edge-probe fallbacks, or a Jacobian conditioning fallback
            inside the final polish.
    """

    Hx: float
    Hy: float
    wavelength: float
    final_polish: SolveReport
    chief_seed_strategy: str
    used_cached_map: bool
    edge_probe_fallbacks: int
    calibration_used: bool
    fallback_used: bool


@dataclass(frozen=True)
class RobustSolveReport:
    """Aggregate outcome of one :meth:`RobustRayAimer.aim_rays` call.

    Robust aiming may return NaN for individual vignetted/unreachable
    rays; ``converged`` is therefore defined as
    ``num_converged == num_rays`` and the exact counts are retained.

    Attributes:
        field_reports: One :class:`RobustFieldReport` per processed field
            group, in processing order.
        num_rays: Total number of requested rays.
        num_converged: Number of rays that met the solver tolerance.
        converged: ``num_converged == num_rays``.
        seed_residual: Worst (max) seed residual across the field polishes.
        final_residual: Worst (max) final residual across the field
            polishes.
        final_polish_iterations: Largest Newton iteration count among the
            field polishes.
        fallback_used: Whether any field used a substitute strategy (see
            :attr:`RobustFieldReport.fallback_used`).
    """

    field_reports: tuple
    num_rays: int
    num_converged: int
    converged: bool
    seed_residual: float
    final_residual: float
    final_polish_iterations: int
    fallback_used: bool


@dataclass(frozen=True)
class _CalibrationRecord:
    """Bookkeeping of one fresh per-field calibration."""

    chief_strategy: str
    chief_report: SolveReport | None
    edge_probe_fallbacks: int


def _scan_candidate_offsets(
    g_xi: float, g_eta: float, Hx: float, Hy: float, n: int
) -> tuple:
    """Transverse ``(xi, eta)`` offsets of the chief-scan candidates.

    The offsets parameterize a line *through the fresh paraxial seed*: the
    candidate at index ``n // 2`` is exactly ``(0, 0)`` (the seed itself),
    and every other candidate displaces the seed within the entry frame's
    transverse plane only. The sweep direction comes from the local field
    coordinates ``(Hx, Hy)`` normalized -- the meridional ``eta`` axis for
    a zero/degenerate field -- and the sweep half-width scales with the
    seed's own transverse offset from the first-surface anchor
    (``g_xi, g_eta``), which is invariant under a rigid translation of the
    system.

    Args:
        g_xi, g_eta: Transverse components of (seed - first-surface vertex)
            in the entry frame.
        Hx, Hy: Normalized field coordinates of the chief solve.
        n: Number of candidates (odd keeps the exact seed in the sweep).

    Returns:
        tuple: ``(xi_offsets, eta_offsets)`` backend arrays of length ``n``.
    """
    h_norm = (Hx * Hx + Hy * Hy) ** 0.5
    if h_norm < 1e-12:
        dir_xi, dir_eta = 0.0, 1.0
    else:
        dir_xi, dir_eta = Hx / h_norm, Hy / h_norm

    seed_offset = (g_xi * g_xi + g_eta * g_eta) ** 0.5
    scale = max(50.0, 20.0 * seed_offset)

    r = be.linspace(-scale, scale, n)
    # Force the center candidate to exactly zero so it reproduces the seed
    # bit-for-bit (linspace's midpoint is only zero to round-off).
    center = be.arange_indices(n) == n // 2
    r = be.where(center, be.zeros_like(r), r)
    return dir_xi * r, dir_eta * r


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

    def cached_epl(path=None):
        # ``path`` only avoids a rebuild inside the wrapped call; the cached
        # value is the same either way (geometry is fixed within this scope).
        if "epl" not in cache:
            cache["epl"] = orig_epl(path=path)
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
        #: Aggregate report of the most recent :meth:`aim_rays` call. Set
        #: before raising on a total field failure, so the failure can be
        #: inspected.
        self.last_report: RobustSolveReport | None = None
        # Report of the last chief-solve attempt, kept for the failure path.
        self._last_chief_failure_report: SolveReport | None = None

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
        # Scalar paraxial values are only seeds here -- every returned ray
        # is Newton-polished against real traces -- so out-of-domain
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
        guess_failed = False
        if initial_guess is not None:
            try:
                result = self._iterative.aim_rays(
                    fields, wavelengths, pupil_coords, initial_guess=initial_guess
                )
            except ValueError:
                # Fall through to the calibrated solve below; using
                # calibration after a failed requested guess is a fallback
                # and is reported as such.
                guess_failed = True
            else:
                self.last_report = self._report_for_initial_guess(fields, wavelengths)
                return result

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

        # One launch parameterization per aiming call: all solves below share
        # the same entry frame, so the local transverse basis is built once
        # and passed through rather than re-derived per solve.
        param = LaunchParameterization.for_optic(self.optic, bool(is_inf))

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
        field_reports: list[RobustFieldReport] = []
        num_converged_total = 0

        def _finalize_report() -> None:
            """Assemble the aggregate report from the field reports so far."""
            polishes = [fr.final_polish for fr in field_reports]
            self.last_report = RobustSolveReport(
                field_reports=tuple(field_reports),
                num_rays=n,
                num_converged=num_converged_total,
                converged=num_converged_total == n,
                seed_residual=max(
                    (p.seed_residual for p in polishes), default=float("inf")
                ),
                final_residual=max(
                    (p.final_residual for p in polishes), default=float("inf")
                ),
                final_polish_iterations=max(
                    (p.iterations for p in polishes), default=0
                ),
                fallback_used=guess_failed
                or any(fr.fallback_used for fr in field_reports),
            )

        for key in ordered_keys:
            idx = groups[key]
            Hxk, Hyk, wlk = key

            used_cached_map = False
            calibration: _CalibrationRecord | None = None
            pmap = self._cache.get_fresh(Hxk, Hyk, wlk)
            if pmap is not None:
                used_cached_map = True
            else:
                seed_map = self._cache.get_stale(Hxk, Hyk, wlk)
                if seed_map is None:
                    seed_map = self._cache.nearest(Hxk, Hyk)
                try:
                    with _cached_paraxial_constants(self.optic):
                        pmap, calibration = self._calibrate_field(
                            Hxk,
                            Hyk,
                            wlk,
                            stop_idx,
                            is_inf,
                            r_stop,
                            seed_map,
                            param,
                        )
                except ValueError:
                    # Total chief failure for this field: record what was
                    # attempted, publish the partial report, then re-raise
                    # so the failure can be inspected via last_report.
                    failed_polish = self._last_chief_failure_report
                    if failed_polish is None:
                        failed_polish = SolveReport(
                            seed_residual=float("inf"),
                            final_residual=float("inf"),
                            converged=False,
                            iterations=0,
                            num_rays=len(idx),
                            num_converged=0,
                        )
                    field_reports.append(
                        RobustFieldReport(
                            Hx=Hxk,
                            Hy=Hyk,
                            wavelength=wlk,
                            final_polish=failed_polish,
                            chief_seed_strategy="failed",
                            used_cached_map=False,
                            edge_probe_fallbacks=0,
                            calibration_used=True,
                            fallback_used=True,
                        )
                    )
                    _finalize_report()
                    raise
                self._cache.put(Hxk, Hyk, wlk, pmap)

            Px_g = Px[idx]
            Py_g = Py[idx]
            wl_g = wl_arr[idx]

            x0, y0, z0, L0, M0, N0 = pmap.seed(Px_g, Py_g)
            tx = Px_g * r_stop
            ty = Py_g * r_stop

            x, y, z, L, M, N, converged, _, polish = self._iterative._solve_core(
                x0, y0, z0, L0, M0, N0, wl_g, stop_idx, is_inf, tx, ty, param=param
            )

            group_converged = int(be.to_numpy(converged).reshape(-1).sum())
            num_converged_total += group_converged

            if calibration is None:
                strategy = "cached_map"
                probe_fallbacks = 0
            else:
                strategy = calibration.chief_strategy
                probe_fallbacks = calibration.edge_probe_fallbacks

            field_reports.append(
                RobustFieldReport(
                    Hx=Hxk,
                    Hy=Hyk,
                    wavelength=wlk,
                    final_polish=polish,
                    chief_seed_strategy=strategy,
                    used_cached_map=used_cached_map,
                    edge_probe_fallbacks=probe_fallbacks,
                    calibration_used=calibration is not None,
                    fallback_used=guess_failed
                    or strategy in ("marching", "scan")
                    or probe_fallbacks > 0
                    or polish.fallback_used,
                )
            )

            if not be.any(converged):
                # Publish the report -- including this field's failed
                # polish -- before raising, so the failure is inspectable.
                _finalize_report()
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

        _finalize_report()
        return x_out, y_out, z_out, L_out, M_out, N_out

    def _report_for_initial_guess(
        self, fields: tuple, wavelengths: Any
    ) -> RobustSolveReport:
        """Aggregate report for a call solved directly from a caller guess."""
        polish = self._iterative.last_report
        if polish is None:  # pragma: no cover - aim_rays always sets it
            polish = SolveReport(
                seed_residual=float("nan"),
                final_residual=float("nan"),
                converged=True,
                iterations=0,
                num_rays=0,
                num_converged=0,
            )
        Hx, Hy = fields
        hx0 = float(be.to_numpy(be.as_array_1d(Hx)).reshape(-1)[0])
        hy0 = float(be.to_numpy(be.as_array_1d(Hy)).reshape(-1)[0])
        if hasattr(wavelengths, "__len__"):
            wl0 = float(be.to_numpy(be.as_array_1d(wavelengths)).reshape(-1)[0])
        else:
            wl0 = float(wavelengths)
        field_report = RobustFieldReport(
            Hx=hx0,
            Hy=hy0,
            wavelength=wl0,
            final_polish=polish,
            chief_seed_strategy="initial_guess",
            used_cached_map=False,
            edge_probe_fallbacks=0,
            calibration_used=False,
            fallback_used=polish.fallback_used,
        )
        return RobustSolveReport(
            field_reports=(field_report,),
            num_rays=polish.num_rays,
            num_converged=polish.num_converged,
            converged=polish.num_converged == polish.num_rays,
            seed_residual=polish.seed_residual,
            final_residual=polish.final_residual,
            final_polish_iterations=polish.iterations,
            fallback_used=polish.fallback_used,
        )

    def _calibrate_field(
        self,
        Hx: float,
        Hy: float,
        wl: float,
        stop_idx: int,
        is_inf: bool,
        r_stop: float,
        seed_map: PupilMap | None,
        param: LaunchParameterization | None = None,
    ) -> tuple[PupilMap, _CalibrationRecord]:
        """Chief solve + 4 edge probes -> affine :class:`PupilMap` (§4.2).

        Returns:
            tuple: The fitted map and the :class:`_CalibrationRecord`
            describing how the chief anchor was obtained and how many edge
            probes fell back to the chief launch.
        """
        if param is None:
            param = LaunchParameterization.for_optic(self.optic, bool(is_inf))
        chief, strategy, chief_report = self._solve_chief(
            Hx, Hy, wl, stop_idx, is_inf, seed_map, param
        )
        probes = []
        probe_fallbacks = 0
        for px, py in _EDGE_PROBES:
            probe, fell_back = self._solve_probe(
                wl, stop_idx, is_inf, px, py, r_stop, chief, param
            )
            probes.append(probe)
            probe_fallbacks += int(fell_back)
        record = _CalibrationRecord(
            chief_strategy=strategy,
            chief_report=chief_report,
            edge_probe_fallbacks=probe_fallbacks,
        )
        return self._fit_affine(chief, probes, param), record

    def _solve_chief(
        self,
        Hx: float,
        Hy: float,
        wl: float,
        stop_idx: int,
        is_inf: bool,
        seed_map: PupilMap | None,
        param: LaunchParameterization | None = None,
    ) -> tuple[tuple, str, SolveReport | None]:
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
        Newton still converges. Only the free transverse 2-DOF is
        warm-started: the seed map's chief launch is projected onto this
        field's fresh seed through the shared local parameterization.

        Returns:
            tuple: ``(launch, strategy, report)`` -- the solved chief
            launch state, which strategy produced it (``"warm_map"``,
            ``"direct_paraxial"``, ``"marching"`` or ``"scan"``), and the
            :class:`SolveReport` of the producing solve (``None`` when the
            producing step exposes no single report).

        Raises:
            ValueError: If every strategy fails. The report of the last
                attempted solve is stored on
                ``self._last_chief_failure_report`` before raising.
        """
        if param is None:
            param = LaunchParameterization.for_optic(self.optic, bool(is_inf))
        wl_a = be.array([wl])
        tx = be.array([0.0])
        ty = be.array([0.0])
        self._last_chief_failure_report = None

        px0, py0, pz0, pL0, pM0, pN0 = self._paraxial.aim_rays(
            (be.array([Hx]), be.array([Hy])),
            wl_a,
            (be.array([0.0]), be.array([0.0])),
        )

        if seed_map is not None:
            sx0, sy0, sz0, sL0, sM0, sN0 = seed_map.seed(
                be.array([0.0]), be.array([0.0])
            )
            # Carry only the free transverse offsets of the stored chief
            # over to this field's fresh seed.
            bound = param.bind(px0, py0, pz0, pL0, pM0, pN0)
            xi, eta = bound.project(sx0, sy0, sz0, sL0, sM0, sN0)
            x0, y0, z0, L0, M0, N0 = bound.launch(xi, eta)

            x, y, z, L, M, N, converged, _, report = self._iterative._solve_core(
                x0, y0, z0, L0, M0, N0, wl_a, stop_idx, is_inf, tx, ty, param=param
            )
            self._last_chief_failure_report = report
            if be.any(converged):
                launch = (
                    to_float(x),
                    to_float(y),
                    to_float(z),
                    to_float(L),
                    to_float(M),
                    to_float(N),
                )
                return launch, "warm_map", report

        x, y, z, L, M, N, converged, _, report = self._iterative._solve_core(
            px0, py0, pz0, pL0, pM0, pN0, wl_a, stop_idx, is_inf, tx, ty, param=param
        )
        self._last_chief_failure_report = report
        if be.any(converged):
            launch = (
                to_float(x),
                to_float(y),
                to_float(z),
                to_float(L),
                to_float(M),
                to_float(N),
            )
            return launch, "direct_paraxial", report

        marched = self._march_chief(Hx, Hy, wl_a, stop_idx, is_inf, tx, ty, param)
        if marched is not None:
            launch, report = marched
            if report is not None:
                self._last_chief_failure_report = report
            return launch, "marching", report

        if is_inf:
            scanned = self._scan_chief(
                px0, py0, pz0, pL0, pM0, pN0, wl_a, stop_idx, tx, ty, param, Hx, Hy
            )
            if scanned is not None:
                launch, report = scanned
                self._last_chief_failure_report = report
                return launch, "scan", report

        raise ValueError(
            f"RobustRayAimer: chief ray failed to converge for field "
            f"(Hx={Hx}, Hy={Hy}) after marching from the axis; check "
            f"the system configuration."
        )

    def _scan_chief(
        self,
        px0: Any,
        py0: Any,
        pz0: Any,
        pL0: Any,
        pM0: Any,
        pN0: Any,
        wl_a: Any,
        stop_idx: int,
        tx: Any,
        ty: Any,
        param: LaunchParameterization,
        Hx: float = 0.0,
        Hy: float = 0.0,
        n: int = 2001,
    ) -> tuple[tuple, SolveReport] | None:
        """Last-resort chief-ray seed search for extreme (beyond +-90 degree)
        field angles.

        Sweeps candidate launch points along the transverse line *through
        the fresh paraxial seed* ``(px0 ... pN0)`` and returns the first
        one the Newton polish converges from, for when neither the paraxial
        guess nor field marching converges. The parameterization is bound
        to repeated copies of the seed, so the ``r = 0`` candidate
        reproduces the seed exactly and every candidate displacement is
        transverse to the entry direction. The sweep direction comes from
        the field coordinates ``(Hx, Hy)`` (the meridional axis for a
        degenerate field) and the sweep width from the seed's transverse
        offset relative to the first-surface anchor -- both invariant under
        a rigid translation of the system (see
        :func:`_scan_candidate_offsets`).
        """
        # Seed offset relative to the first physical surface's vertex,
        # expressed on the entry frame's transverse basis. Anchoring at the
        # vertex (not the global origin) keeps the sweep width invariant
        # under rigid translations.
        path = self.optic.surfaces.build_paraxial_path()
        anchor = path.vertices_gcs[1]
        u, v = param.u, param.v
        g_rel = (
            to_float(px0) - to_float(anchor[0]),
            to_float(py0) - to_float(anchor[1]),
            to_float(pz0) - to_float(anchor[2]),
        )
        g_xi = g_rel[0] * u[0] + g_rel[1] * u[1] + g_rel[2] * u[2]
        g_eta = g_rel[0] * v[0] + g_rel[1] * v[1] + g_rel[2] * v[2]

        xi_off, eta_off = _scan_candidate_offsets(g_xi, g_eta, Hx, Hy, n)

        ones = be.ones(n)
        sx = ones * to_float(px0)
        sy = ones * to_float(py0)
        sz = ones * to_float(pz0)
        L0 = ones * to_float(pL0)
        M0 = ones * to_float(pM0)
        N0 = ones * to_float(pN0)
        wl_b = ones * to_float(wl_a)
        tx_b = ones * to_float(tx)
        ty_b = ones * to_float(ty)

        # Candidates sit on the transverse line through the seed: binding
        # the parameterization at repeated copies of the seed makes
        # launch(0, 0) the seed itself and keeps the xi/eta displacements
        # in the transverse plane for any entry direction.
        bound = param.bind(sx, sy, sz, L0, M0, N0)
        x0, y0, z0, L0, M0, N0 = bound.launch(xi_off, eta_off)

        x, y, z, L, M, N, converged, _, report = self._iterative._solve_core(
            x0, y0, z0, L0, M0, N0, wl_b, stop_idx, True, tx_b, ty_b, param=param
        )
        if not be.any(converged):
            return None

        # Among the converged candidates, take the one whose starting
        # offset from the fresh seed is smallest: a wide sweep can also
        # converge onto physically extreme solution branches far from the
        # seed, and an extreme "chief" would poison the probe/affine
        # calibration built on top of it. Nearest-to-seed stays on the
        # seed's own branch.
        conv_np = be.to_numpy(converged).reshape(-1)
        offsets_np = (
            be.to_numpy(xi_off).reshape(-1) ** 2 + be.to_numpy(eta_off).reshape(-1) ** 2
        )
        candidates = conv_np.nonzero()[0]
        idx = int(candidates[offsets_np[candidates].argmin()])
        launch = (
            to_float(x[idx : idx + 1]),
            to_float(y[idx : idx + 1]),
            to_float(z[idx : idx + 1]),
            to_float(L[idx : idx + 1]),
            to_float(M[idx : idx + 1]),
            to_float(N[idx : idx + 1]),
        )
        return launch, report

    def _march_chief(
        self,
        Hx: float,
        Hy: float,
        wl_a: Any,
        stop_idx: int,
        is_inf: bool,
        tx: Any,
        ty: Any,
        param: LaunchParameterization,
        max_attempts: int = 150,
        min_dt: float = 1e-4,
    ) -> tuple[tuple, SolveReport | None] | None:
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

        On success, returns ``(launch, report)`` where ``report`` is the
        :class:`SolveReport` of the final full-tolerance solve (``None``
        when the attempt budget ran out exactly at ``t = 1`` before that
        solve could run).
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
                x, y, z, L, M, N, converged, _, report = self._iterative._solve_core(
                    x0,
                    y0,
                    z0,
                    L0,
                    M0,
                    N0,
                    wl_a,
                    stop_idx,
                    is_inf,
                    tx,
                    ty,
                    param=param,
                )
                if be.any(converged):
                    return (
                        to_float(x),
                        to_float(y),
                        to_float(z),
                        to_float(L),
                        to_float(M),
                        to_float(N),
                    ), report
                return launch, report

            t_next = min(t + dt, 1.0)
            Hxt, Hyt = Hx * t_next, Hy * t_next
            px0, py0, pz0, pL0, pM0, pN0 = self._paraxial.aim_rays(
                (be.array([Hxt]), be.array([Hyt])),
                wl_a,
                (be.array([0.0]), be.array([0.0])),
            )

            # Carry only the free transverse offsets of the last converged
            # launch onto this step's fresh paraxial seed (the fixed DOF --
            # direction for infinite conjugates, object point for finite
            # ones -- encodes the field angle and must stay fresh).
            bound = param.bind(px0, py0, pz0, pL0, pM0, pN0)
            xi, eta = bound.project(
                be.array([launch[0]]),
                be.array([launch[1]]),
                be.array([launch[2]]),
                be.array([launch[3]]),
                be.array([launch[4]]),
                be.array([launch[5]]),
            )
            x0, y0, z0, L0, M0, N0 = bound.launch(xi, eta)

            with _relaxed_tolerance(self._iterative, relaxed_tol):
                x, y, z, L, M, N, converged, _, _r = self._iterative._solve_core(
                    x0,
                    y0,
                    z0,
                    L0,
                    M0,
                    N0,
                    wl_a,
                    stop_idx,
                    is_inf,
                    tx,
                    ty,
                    param=param,
                )

            if not be.any(converged) and is_inf:
                # The transverse launch warm-started from the previous step
                # can occasionally be a worse seed than a fresh paraxial
                # guess at the new angle (e.g. right where marching first
                # takes a large stride); retry once from the fresh guess
                # before giving up and shrinking the step.
                with _relaxed_tolerance(self._iterative, relaxed_tol):
                    x, y, z, L, M, N, converged, _, _r = self._iterative._solve_core(
                        px0,
                        py0,
                        pz0,
                        pL0,
                        pM0,
                        pN0,
                        wl_a,
                        stop_idx,
                        is_inf,
                        tx,
                        ty,
                        param=param,
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

        return (launch, None) if t >= 1.0 else None

    def _solve_probe(
        self,
        wl: float,
        stop_idx: int,
        is_inf: bool,
        Px_e: float,
        Py_e: float,
        r_stop: float,
        chief: tuple[float, float, float, float, float, float],
        param: LaunchParameterization,
    ) -> tuple[tuple, bool]:
        """Solve one cardinal edge probe, seeded from the chief launch.

        Returns:
            tuple: ``(launch, fell_back)`` -- the probe launch state and
            whether it fell back to the chief launch because the probe
            failed to converge.
        """
        x0, y0, z0, L0, M0, N0 = (be.array([v]) for v in chief)
        wl_a = be.array([wl])
        tx = be.array([Px_e * r_stop])
        ty = be.array([Py_e * r_stop])

        x, y, z, L, M, N, converged, _, _r = self._iterative._solve_core(
            x0, y0, z0, L0, M0, N0, wl_a, stop_idx, is_inf, tx, ty, param=param
        )

        if not be.any(converged):
            # An unreachable edge probe still yields a usable (if less
            # accurate) linear seed by falling back to the chief launch,
            # rather than propagating NaN into the affine fit.
            return chief, True

        return (
            to_float(x),
            to_float(y),
            to_float(z),
            to_float(L),
            to_float(M),
            to_float(N),
        ), False

    def _fit_affine(
        self,
        chief: tuple[float, float, float, float, float, float],
        probes: list[tuple[float, float, float, float, float, float]],
        param: LaunchParameterization,
    ) -> PupilMap:
        """Fit the 2x2 affine launch model from the chief ray + 4 probes.

        The probes' launch states are projected into the chief-bound local
        transverse parameterization, so the fitted offsets are (xi, eta)
        coordinates -- valid for any entry direction, and stored as plain
        floats (detached by design).
        """
        p_east, p_west, p_north, p_south = probes
        bound = param.bind(*(be.array([v]) for v in chief))

        def free(state: tuple) -> tuple[float, float]:
            xi, eta = bound.project(*(be.array([v]) for v in state))
            return to_float(xi), to_float(eta)

        e1, e2 = free(p_east)
        w1, w2 = free(p_west)
        n1, n2 = free(p_north)
        s1, s2 = free(p_south)

        A = (
            ((e1 - w1) / 2.0, (n1 - s1) / 2.0),
            ((e2 - w2) / 2.0, (n2 - s2) / 2.0),
        )

        return PupilMap(base=chief, A=A, param=param)
