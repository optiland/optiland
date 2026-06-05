"""LevenbergController — adaptive damping with inner λ-search.

Solves ``(JᵀJ + λ·diag(JᵀJ))Δ = −Jᵀr`` (Moré diagonal damping) via
``numpy.linalg.solve``.  Runs the full internal accept/reject λ-search so that
one ``transform()`` call produces one *accepted* step outcome.

Damping adaption is the **Nielsen gain-ratio** scheme (D6, §4.4): with
predicted reduction ``L = ½·dxᵀ(λ·D·dx − g)`` (``g = Jᵀr``) and actual
reduction ``ΔF = F(x) − F(x+dx)`` (``F = Σrᵢ²``), the ratio ``ρ = ΔF / L``
drives both acceptance and the λ update —

* ``ρ > 0`` → accept; ``λ ← λ·max(⅓, 1−(2ρ−1)³)``, reset ``ν ← 2``;
* ``ρ ≤ 0`` (or non-finite trial) → reject; ``λ ← λ·ν``, ``ν ← 2ν``.

This replaces the previous flat ``×10 / ×0.1`` schedule.

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
    ``numpy.linalg.solve``.  Runs the internal accept/reject λ-search:
    a trial that decreases cost → accept (decrease λ); a trial that increases
    cost or yields non-finite values → reject (increase λ).

    λ is stored in ``state.scratch['lambda']`` so the value persists across
    steps and is visible to power users.

    Args:
        lam_init: Initial damping factor λ₀ (default 1e-3).
        lam_factor_up: **Legacy** flat-schedule λ increase, retained for
            backward-compatible construction; unused by the Nielsen adaption.
        lam_factor_down: **Legacy** flat-schedule λ decrease, retained for
            backward-compatible construction; unused by the Nielsen adaption.
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
        """Seed λ and the Nielsen ν factor in ``state.scratch``."""
        state.scratch["lambda"] = self._lam_init
        state.scratch["nu"] = 2.0

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
        from optiland.backend.utils import to_numpy as _to_numpy

        J: np.ndarray = np.asarray(_to_numpy(info.jacobian), dtype=float)  # (m, n)
        r: np.ndarray = np.asarray(_to_numpy(info.residuals), dtype=float)  # (m,)
        x: np.ndarray = np.asarray(state.x, dtype=float)  # (n,)

        JTJ: np.ndarray = J.T @ J  # (n, n)
        JTr: np.ndarray = J.T @ r  # (n,) == g
        # Moré diagonal: clamp zero entries so damping is always positive
        d: np.ndarray = np.maximum(np.abs(np.diag(JTJ)), self._min_diag)

        # Diagnostics (D12): gradient norm of the merit and JᵀJ conditioning.
        state.grad_norm = float(np.linalg.norm(2.0 * JTr))
        with np.errstate(all="ignore"):
            try:
                state.cond_estimate = float(np.linalg.cond(JTJ))
            except np.linalg.LinAlgError:
                state.cond_estimate = None

        lam: float = state.scratch.get("lambda", self._lam_init)
        nu: float = state.scratch.get("nu", 2.0)
        current_val: float = _to_float(state.value)
        n_trials: int = 0

        while n_trials < self._max_trials and lam <= self._lam_max:
            A = JTJ + lam * np.diag(d)
            try:
                dx = np.linalg.solve(A, -JTr)
            except np.linalg.LinAlgError:
                lam *= nu
                nu *= 2.0
                n_trials += 1
                continue

            x_trial = x + dx
            f_trial = info.value_fn(x_trial)
            n_trials += 1

            # Predicted reduction L = ½·dxᵀ(λ·D·dx − g), g = Jᵀr.
            predicted = 0.5 * float(dx @ (lam * d * dx - JTr))
            actual = current_val - f_trial  # ΔF for F = Σrᵢ²
            rho = actual / predicted if predicted > 0 else -1.0

            if (not math.isfinite(f_trial)) or rho <= 0.0:
                # Reject: increase damping geometrically (Nielsen).
                lam *= nu
                nu *= 2.0
                continue

            # Accept: gain-ratio-controlled λ decrease, reset ν.
            lam = max(lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), 1e-15)
            state.scratch["lambda"] = lam
            state.scratch["nu"] = 2.0
            return StepOutcome(
                delta_x=dx,
                accepted=True,
                info={"n_trials": n_trials, "trial_value": f_trial, "rho": rho},
            )

        # All trials failed (λ blow-up or max_trials exceeded)
        state.scratch["lambda"] = lam
        state.scratch["nu"] = nu
        return StepOutcome(delta_x=None, accepted=False, info={"n_trials": n_trials})


def _to_float(v: Any) -> float:
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)
