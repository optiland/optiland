"""LineSearchController — Armijo backtracking line search.

Shrinks the step length α by ``rho`` until the Armijo sufficient-decrease
condition is satisfied::

    f(x + α·d) ≤ f(x) + c1·α·(∇f·d)

Requires ``info.gradient`` and ``info.value_fn`` in the ``StepInfo``.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import StepInfo, StepOutcome


class LineSearchController:
    """Armijo (sufficient-decrease) backtracking line search.

    Shrinks the step length α by ``rho`` each trial until
    ``f(x + α·d) ≤ f(x) + c1·α·(∇f·d)`` holds or ``max_steps`` is
    exceeded.  If the condition is never satisfied the last-tried step is
    returned with ``accepted=False``.

    Args:
        c1: Sufficient-decrease constant (Armijo, default 1e-4).
        rho: Backtracking factor per trial — each trial multiplies α by
            ``rho`` (default 0.5).
        max_steps: Maximum backtracking trials (default 30).
        alpha_init: Starting step length (default 1.0).
    """

    def __init__(
        self,
        c1: float = 1e-4,
        rho: float = 0.5,
        max_steps: int = 30,
        alpha_init: float = 1.0,
    ) -> None:
        self._c1 = c1
        self._rho = rho
        self._max_steps = max_steps
        self._alpha_init = alpha_init

    def reset(self, state: Any) -> None:  # noqa: ARG002
        pass

    def transform(self, direction: Any, info: StepInfo, state: Any) -> StepOutcome:
        """Apply Armijo backtracking to ``direction``.

        Args:
            direction: Descent direction ``d`` (shape ``(n,)``).
            info: Must carry ``gradient`` and ``value_fn``.
            state: Provides ``state.x`` and ``state.value``.

        Returns:
            ``StepOutcome`` with ``alpha * direction`` as ``delta_x``.
            ``accepted=True`` when Armijo holds; ``accepted=False`` if the
            condition was never met (best-effort step returned).
        """
        d: np.ndarray = np.asarray(direction, dtype=float)
        g: np.ndarray = np.asarray(info.gradient, dtype=float)
        x: np.ndarray = np.asarray(state.x, dtype=float)
        f0: float = _to_float(state.value)
        slope: float = float(g @ d)

        alpha = self._alpha_init
        n_trials = 0
        for _ in range(self._max_steps):
            x_trial = x + alpha * d
            f_trial = info.value_fn(x_trial)
            n_trials += 1
            if math.isfinite(f_trial) and f_trial <= f0 + self._c1 * alpha * slope:
                return StepOutcome(
                    delta_x=alpha * d,
                    accepted=True,
                    info={
                        "n_trials": n_trials,
                        "alpha": alpha,
                        "trial_value": f_trial,
                    },
                )
            alpha *= self._rho

        # Armijo never satisfied — return last trial as best effort
        return StepOutcome(
            delta_x=alpha * d,
            accepted=False,
            info={"n_trials": n_trials, "alpha": alpha},
        )


def _to_float(v: Any) -> float:
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)
