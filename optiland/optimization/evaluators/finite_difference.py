"""Finite-Difference Evaluator

The CPU evaluator of the native optimizers (opt v2 PR4.1). Each design
evaluation writes ``x`` into the variables, updates the optics, opens one
evaluation epoch and reads ``problem.residual_vector()``, so residuals are
``effective_weight * delta`` (decision V7) and the merit is ``sum(r ** 2)``.

Jacobians use forward differences by default (``scheme="central"`` opt-in)
with the step ``h_i = rel_step * max(|x_i|, typical_i)``, floored at
``abs_step``. A step that would cross a variable bound flips toward the
interior. Three things keep a Jacobian cheap:

* **r0 reuse**: ``jacobian(x, r0)`` takes the residuals already computed at
  ``x`` instead of evaluating them again.
* **Partial trace**: on the numpy backend, a column whose variable changes only
  surface ``s`` (see :func:`~optiland.raytrace.partial_trace.partial_trace_start`)
  resumes every real-ray trace at ``s`` from the baseline evaluation's
  snapshots. Non-trace operands are recomputed normally. Columns are grouped
  by ``s``.
* **Aim policy** (decision V14): ``aim_policy="exact"`` re-aims per column;
  ``"frozen"`` reuses the baseline aim solutions (see
  :mod:`optiland.optimization.evaluators.aim`).

Failures follow :mod:`optiland.optimization.failure`: a failed evaluation
yields ``NaN`` rows, or raises :class:`OptimizationFailure` with
``on_failure="raise"``.

Kramer Harrison, 2026
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland import _epoch
from optiland.optimization.evaluators.aim import resolve_aim_policy
from optiland.optimization.failure import (
    ConfigurationError,
    OptimizationFailure,
    describe_rows,
    failing_rows,
    normalize_failure_mode,
)
from optiland.raytrace import _resume

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland.optimization.evaluators.aim import AimPolicy
    from optiland.optimization.problem import OptimizationProblem

_EPS = float(np.finfo(float).eps)

#: Default relative steps: optimal for smooth residuals in double precision.
DEFAULT_REL_STEP = {"forward": _EPS**0.5, "central": _EPS ** (1.0 / 3.0)}

# Baseline evaluations remembered for r0 reuse (the accepted point and the
# trial evaluated after it).
_BASELINE_CACHE_SIZE = 2


@dataclass
class _Baseline:
    """One remembered baseline evaluation."""

    r: np.ndarray
    epoch: _epoch.EvaluationEpoch
    store: _resume.TraceStore | None


class FiniteDiffEvaluator:
    """Evaluate residuals and finite-difference Jacobians of a problem.

    Args:
        problem: The optimization problem.
        scheme: ``"forward"`` (default, ``n`` evaluations per Jacobian) or
            ``"central"`` (``2n``).
        rel_step: Relative step. Defaults to ``sqrt(eps)`` (forward) or
            ``eps ** (1/3)`` (central).
        abs_step: Absolute floor of the step, for variables near zero with a
            small ``typical_x``. Defaults to 1e-10.
        typical_x: Typical magnitude of each (scaled) variable, a scalar or
            one value per variable. Defaults to 1.
        aim_policy: ``"exact"`` (default) or ``"frozen"``, or an
            :class:`~optiland.optimization.evaluators.aim.AimPolicy`.
        partial_trace: Resume traces of eligible columns from the baseline
            snapshots (numpy backend only). Defaults to True.
        on_failure: ``"reject"`` (default: failed rows become ``NaN``) or
            ``"raise"`` (raise :class:`OptimizationFailure`).

    Attributes:
        nfev: Residual evaluations so far, Jacobian columns included.
        njev: Jacobians computed so far.
        resumed_traces: Traces resumed from a snapshot so far.
        problem: The wrapped problem.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        *,
        scheme: str = "forward",
        rel_step: float | None = None,
        abs_step: float = 1e-10,
        typical_x: ArrayLike | float = 1.0,
        aim_policy: str | AimPolicy | None = "exact",
        partial_trace: bool = True,
        on_failure: str = "reject",
    ) -> None:
        if scheme not in DEFAULT_REL_STEP:
            raise ConfigurationError(
                f"Unknown scheme {scheme!r}; expected 'forward' or 'central'."
            )
        if any(isinstance(var.value, str) for var in problem.variables):
            raise ConfigurationError(
                "Glass/material variables are not supported by finite "
                "differences; use GlassExpert."
            )
        self.problem = problem
        self.n_vars = len(problem.variables)
        self.backend = be.get_backend()
        self.scheme = scheme
        self.rel_step = DEFAULT_REL_STEP[scheme] if rel_step is None else rel_step
        self.abs_step = float(abs_step)
        self.typical_x = np.broadcast_to(
            np.asarray(typical_x, dtype=float), (self.n_vars,)
        ).copy()
        self.aim_policy = resolve_aim_policy(aim_policy)
        self.on_failure = normalize_failure_mode(on_failure)
        self.nfev = 0
        self.njev = 0
        self.resumed_traces = 0
        self._optics = tuple({id(v.optic): v.optic for v in problem.variables}.values())
        self._m: int | None = None
        self._baselines: OrderedDict[bytes, _Baseline] = OrderedDict()
        self._partial = (
            partial_trace
            and self.backend == "numpy"
            and any(start is not None for start in self.partial_trace_starts())
        )

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    def read_x(self) -> np.ndarray:
        """Return the current scaled variable values as a float array."""
        return np.array(
            [float(be.to_numpy(var.value)) for var in self.problem.variables]
        )

    def write_x(self, x: ArrayLike) -> None:
        """Write ``x`` into the variables, then update the optics."""
        for var, value in zip(self.problem.variables, x, strict=True):
            var.update(float(value))
        self.problem.update_optics()

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds of the scaled variables (``±inf`` if unset)."""
        lower = np.full(self.n_vars, -np.inf)
        upper = np.full(self.n_vars, np.inf)
        for i, var in enumerate(self.problem.variables):
            lo, hi = var.bounds
            if lo is not None:
                lower[i] = float(be.to_numpy(lo))
            if hi is not None:
                upper[i] = float(be.to_numpy(hi))
        return lower, upper

    def partial_trace_starts(self) -> list[int | None]:
        """Resume surface of each variable (None: full trace), at the current state."""
        starts = []
        for var in self.problem.variables:
            tracer = getattr(var.optic, "ray_tracer", None)
            find = getattr(tracer, "partial_trace_start", None)
            try:
                starts.append(find(var) if find is not None else None)
            except Exception:  # unrecognized configuration: full trace
                starts.append(None)
        return starts

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def residuals(self, x: ArrayLike) -> np.ndarray:
        """Return the residuals at ``x`` as a float array.

        Failed rows are ``NaN`` (``on_failure="reject"``). The evaluation is
        remembered as a baseline for :meth:`jacobian`.
        """
        x = np.asarray(x, dtype=float)
        store = _resume.TraceStore() if self._partial else None
        epoch = _epoch.EvaluationEpoch()
        hook = _resume.recording(store) if store is not None else None
        r = self._evaluate(x, epoch, hook)
        # Snapshots stay valid even when some rows failed: they key on traces.
        self._remember(x, _Baseline(r, epoch, store))
        return r

    def value(self, x: ArrayLike) -> float:
        """Return the merit ``sum(r ** 2)`` at ``x`` (``nan`` on failure)."""
        r = self.residuals(x)
        return float(r @ r)

    def gradient(self, x: ArrayLike, r0: ArrayLike | None = None) -> np.ndarray:
        """Return the merit gradient ``2 J^T r`` at ``x``."""
        x = np.asarray(x, dtype=float)
        r = self.residuals(x) if r0 is None else np.asarray(r0, dtype=float)
        return 2.0 * self.jacobian(x, r0=r).T @ r

    def _evaluate(
        self, x: np.ndarray, epoch: _epoch.EvaluationEpoch, hook: Any
    ) -> np.ndarray:
        """One design evaluation: write, update, epoch, residual vector."""
        self.nfev += 1
        try:
            self.write_x(x)
            with (
                _epoch.evaluation_epoch(self._optics, epoch=epoch),
                hook or nullcontext(),
            ):
                r = self.problem.residual_vector()
            r = np.asarray(be.to_numpy(r), dtype=float).reshape(-1)
        except _epoch.EpochViolationError:
            raise  # debug guard: a contract breach is a bug, not a bad design
        except Exception as exc:
            if self.on_failure == "raise":
                raise OptimizationFailure(f"Design evaluation failed: {exc}") from exc
            return np.full(self._num_rows(), np.nan)
        if self._m is None:
            self._m = r.size
        if self.on_failure == "raise":
            rows = failing_rows(r)
            if rows.size:
                raise OptimizationFailure(
                    "Non-finite residuals for operand(s) "
                    f"{describe_rows(self.problem, rows)}."
                )
        return r

    def _num_rows(self) -> int:
        return self._m if self._m is not None else max(len(self.problem.operands), 1)

    def _remember(self, x: np.ndarray, baseline: _Baseline) -> None:
        key = x.tobytes()
        self._baselines.pop(key, None)
        self._baselines[key] = baseline
        while len(self._baselines) > _BASELINE_CACHE_SIZE:
            self._baselines.popitem(last=False)

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------

    def jacobian(self, x: ArrayLike, r0: ArrayLike | None = None) -> np.ndarray:
        """Finite-difference Jacobian of the residuals at ``x``.

        Args:
            x: The point.
            r0: Residuals already computed at ``x``. When given, the baseline
                is not evaluated again; it is re-evaluated only when partial
                trace or frozen aiming needs the baseline's trace snapshots or
                aim solutions and they are no longer remembered.

        Returns:
            An ``(m, n)`` array. Entries that cannot be differenced (failed
            rows on both sides of ``x``) are 0, so the optimizer sees no
            information rather than ``NaN``.

        Raises:
            OptimizationFailure: Under ``on_failure="raise"``, if any
                evaluation fails.
        """
        x = np.asarray(x, dtype=float)
        self.njev += 1
        baseline = self._baselines.get(x.tobytes())
        needs_baseline = self._partial or self.aim_policy.needs_baseline
        if baseline is None and (r0 is None or needs_baseline):
            self.residuals(x)
            baseline = self._baselines[x.tobytes()]
        r0 = baseline.r if r0 is None else np.asarray(r0, dtype=float).reshape(-1)

        self.write_x(x)  # eligibility is decided at the baseline state
        starts = self.partial_trace_starts() if self._partial else [None] * self.n_vars
        steps = self._steps(x)
        lower, upper = self.bounds()

        J = np.zeros((r0.size, self.n_vars))
        # Columns sharing a resume surface run together; full traces last.
        order = sorted(
            range(self.n_vars), key=lambda j: (starts[j] is None, starts[j] or 0, j)
        )
        for j in order:
            J[:, j] = self._column(
                x, j, steps[j], lower[j], upper[j], r0, baseline, starts[j]
            )

        self.write_x(x)  # leave the optics at x
        return J

    def _steps(self, x: np.ndarray) -> np.ndarray:
        h = self.rel_step * np.maximum(np.abs(x), self.typical_x)
        h = np.maximum(h, self.abs_step)
        return (x + h) - x  # exactly representable steps

    def _column(
        self,
        x: np.ndarray,
        j: int,
        h: float,
        lo: float,
        hi: float,
        r0: np.ndarray,
        baseline: _Baseline | None,
        start: int | None,
    ) -> np.ndarray:
        """One Jacobian column, with bound-aware steps and failure fallback."""
        fits_up = x[j] + h <= hi
        fits_down = x[j] - h >= lo
        if self.scheme == "central" and fits_up and fits_down:
            r_up = self._shifted(x, j, h, baseline, start)
            r_down = self._shifted(x, j, -h, baseline, start)
            col = (r_up - r_down) / (2.0 * h)
            bad = ~np.isfinite(col)
            if bad.any():  # fall back to whichever one-sided difference works
                fwd = (r_up - r0) / h
                bwd = (r0 - r_down) / h
                col[bad] = np.where(np.isfinite(fwd[bad]), fwd[bad], bwd[bad])
            return np.where(np.isfinite(col), col, 0.0)

        step = self._interior_step(x[j], h, lo, hi, fits_up, fits_down)
        if step == 0.0:
            return np.zeros_like(r0)  # the variable is pinned by its bounds
        col = (self._shifted(x, j, step, baseline, start) - r0) / step
        bad = ~np.isfinite(col)
        if bad.any():
            back = -step if (fits_down if step > 0 else fits_up) else 0.0
            if back != 0.0:
                other = (self._shifted(x, j, back, baseline, start) - r0) / back
                col[bad] = other[bad]
        return np.where(np.isfinite(col), col, 0.0)

    @staticmethod
    def _interior_step(
        xj: float, h: float, lo: float, hi: float, fits_up: bool, fits_down: bool
    ) -> float:
        """Forward step of size ``h``, flipped toward the interior near a bound."""
        if fits_up:
            return h
        if fits_down:
            return -h
        room_up, room_down = hi - xj, xj - lo
        return room_up if room_up >= room_down else -room_down

    def _shifted(
        self,
        x: np.ndarray,
        j: int,
        step: float,
        baseline: _Baseline | None,
        start: int | None,
    ) -> np.ndarray:
        """Residuals at ``x + step * e_j`` (one Jacobian evaluation)."""
        xp = x.copy()
        xp[j] += step
        epoch = _epoch.EvaluationEpoch()
        self.aim_policy.prepare_column(epoch, baseline.epoch if baseline else None)
        hook = None
        store = baseline.store if baseline is not None else None
        if start is not None and store is not None:
            hook = _resume.resuming(store, self.problem.variables[j].optic, start)
            before = store.resumed
        r = self._evaluate(xp, epoch, hook)
        if hook is not None:
            self.resumed_traces += store.resumed - before
        return r
