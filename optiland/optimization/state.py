"""Optimization State and Result

The per-iteration state shared by the native optimizers, stopping criteria and
observers, and the result returned by the native cores and ``minimize()``.

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scipy.optimize import OptimizeResult


@dataclass
class OptimizationState:
    """Mutable state of a run, updated in place after every accepted step.

    Observers and stopping criteria receive a reference to this object and
    must treat it as read-only. *Experimental.*

    Attributes:
        x: Current variable vector (scaled space).
        value: Current merit, ``sum(r ** 2)``.
        residuals: Current residual vector, or None.
        gradient: Current gradient (``J^T r`` for least squares), or None.
        jacobian: Current Jacobian, or None.
        iteration: Number of accepted steps so far.
        n_evals: Residual/merit evaluations so far (Jacobian columns included).
        step_norm: ``||dx||`` of the last accepted step, or None.
        grad_norm: L2 norm of ``gradient``, or None.
        stop_reason: Why the run ended; None while it is running.
    """

    x: Any
    value: float
    residuals: Any | None = None
    gradient: Any | None = None
    jacobian: Any | None = None
    iteration: int = 0
    n_evals: int = 0
    step_norm: float | None = None
    grad_norm: float | None = None
    stop_reason: str | None = None


class OptimizationResult(OptimizeResult):
    """Result of a native core or of ``minimize()``. *Experimental.*

    A ``scipy.optimize.OptimizeResult`` (attribute access on a dict), so code
    that reads ``x``, ``fun``, ``success``, ``message``, ``nfev`` or ``nit``
    from a SciPy result works unchanged. Fields added here:

    Attributes:
        stop_reason: Human-readable reason the run ended. Equals ``message``.
        n_evals: Residual/merit evaluations, Jacobian columns included.
        history: Per-step records of the attached ``HistoryObserver``, or None.
        multipliers: Native DLS only. Multiplier of each hard row, and of each
            active bound, by label; ``d(sum r^2)/dx + A^T mu = 0`` at a solution,
            with ``mu >= 0`` on inequalities.
        active_set: Native DLS only. Labels of the working rows.
        max_constraint_violation: Native DLS only. Largest violation of a hard
            row, in row units.
        constraint_report: Native DLS only. Per-row feasibility report, see
            :func:`~optiland.optimization.constraint.row_report`.
        method: The resolved method (``minimize()`` only).
    """
