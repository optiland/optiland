"""LevenbergController — adaptive damping with inner λ-search (D7/D11/D12).

Solves ``(JᵀJ + λ·diag(JᵀJ))Δ = −Jᵀr`` (Moré diagonal damping) via
``numpy.linalg.solve``.  Runs the full internal accept/reject λ-search so that
one ``transform()`` call produces one *accepted* step outcome (D11).

Non-finite trial value → reject + increase λ (D12).

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import StepInfo, StepOutcome

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState


class LevenbergController:
    """Levenberg-Marquardt step controller (Moré diagonal damping).

    Forms and solves ``(JᵀJ + λ·diag(JᵀJ))Δ = −Jᵀr`` via
    ``numpy.linalg.solve``.  Runs the internal accept/reject λ-search (D11):
    a trial that decreases cost → accept (decrease λ); a trial that increases
    cost or yields non-finite values → reject (increase λ).

    λ is stored in ``state.scratch['lambda']`` so the value persists across
    steps and is visible to power users.

    Args:
        lam_init: Initial damping factor λ₀ (default 1e-3).
        lam_factor_up: Multiplicative λ increase on rejection (default 10).
        lam_factor_down: Multiplicative λ decrease on acceptance (default 0.1).
        lam_max: Upper bound; step is considered failed above this (default 1e16).
        min_diag: Floor for ``diag(JᵀJ)`` entries to avoid zero damping
            (default 1e-8).
        max_trials: Maximum inner λ-search trials per step (default 20).
    """

    def __init__(
        self,
        lam_init: float = 1e-3,
        lam_factor_up: float = 10.0,
        lam_factor_down: float = 0.1,
        lam_max: float = 1e16,
        min_diag: float = 1e-8,
        max_trials: int = 20,
    ) -> None:
        self._lam_init = lam_init
        self._lam_up = lam_factor_up
        self._lam_down = lam_factor_down
        self._lam_max = lam_max
        self._min_diag = min_diag
        self._max_trials = max_trials

    def reset(self, state: OptimizationState) -> None:
        """Seed λ in ``state.scratch`` at the start of a run."""
        state.scratch["lambda"] = self._lam_init

    def transform(
        self, direction: Any, info: StepInfo, state: OptimizationState
    ) -> StepOutcome:
        """Run the inner λ-search and return the first accepted step.

        ``direction`` is ignored; ``LevenbergController`` computes its own
        descent direction from the normal equations.

        Args:
            direction: Unused (LM computes its own direction from J/r).
            info: Must carry ``residuals``, ``jacobian``, and ``value_fn``.
            state: Mutable; ``state.x``, ``state.value``, and
                ``state.scratch['lambda']`` are read / updated.

        Returns:
            ``StepOutcome`` with ``accepted=True`` and ``delta_x`` on success,
            or ``accepted=False`` when all trials fail (λ blow-up).
        """
        J: np.ndarray = np.asarray(info.jacobian, dtype=float)  # (m, n)
        r: np.ndarray = np.asarray(info.residuals, dtype=float)  # (m,)
        x: np.ndarray = np.asarray(state.x, dtype=float)  # (n,)

        JTJ: np.ndarray = J.T @ J  # (n, n)
        JTr: np.ndarray = J.T @ r  # (n,)
        # Moré diagonal: clamp zero entries so damping is always positive
        d: np.ndarray = np.maximum(np.abs(np.diag(JTJ)), self._min_diag)

        lam: float = state.scratch.get("lambda", self._lam_init)
        current_val: float = _to_float(state.value)
        n_trials: int = 0

        while n_trials < self._max_trials and lam <= self._lam_max:
            A = JTJ + lam * np.diag(d)
            try:
                dx = np.linalg.solve(A, -JTr)
            except np.linalg.LinAlgError:
                lam *= self._lam_up
                n_trials += 1
                continue

            x_trial = x + dx
            f_trial = info.value_fn(x_trial)
            n_trials += 1

            if not math.isfinite(f_trial) or f_trial >= current_val:
                # Reject: shrink trust region
                lam *= self._lam_up
                continue

            # Accept: expand trust region
            lam = max(lam * self._lam_down, 1e-15)
            state.scratch["lambda"] = lam
            return StepOutcome(
                delta_x=dx,
                accepted=True,
                info={"n_trials": n_trials, "trial_value": f_trial},
            )

        # All trials failed (λ blow-up or max_trials exceeded)
        state.scratch["lambda"] = lam
        return StepOutcome(delta_x=None, accepted=False, info={"n_trials": n_trials})


def _to_float(v: Any) -> float:
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)
