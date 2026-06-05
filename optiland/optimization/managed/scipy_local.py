"""ScipyLocalAdapter — private core for scipy.optimize.minimize (local methods).

Calls scipy directly without going through the deprecated ``OptimizerGeneric``
class — no DeprecationWarning is triggered from the facade path.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.optimization.failure import (
    OptimizationFailure,
    normalize_failure_mode,
    penalty_merit,
)
from optiland.optimization.state import Capability

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem

_CAPS = frozenset(
    {
        Capability.OBSERVE,
        Capability.MUTATE_TERMINATION,
        Capability.SCIPY_CONSTRAINTS,
        Capability.BOUNDS,
    }
)

_LOCAL_SCIPY_METHODS = {
    "l-bfgs-b",
    "bfgs",
    "slsqp",
    "trust-constr",
    "cobyla",
    "nelder-mead",
    "cg",
    "newton-cg",
    "tnc",
    "powell",
}


class ScipyLocalAdapter:
    """Private-core adapter for SciPy local optimizers.

    Calls ``scipy.optimize.minimize`` directly — does not instantiate any
    deprecated legacy class.

    Args:
        problem: The optimization problem.
        method: SciPy ``minimize`` method string (default ``None`` = auto-select).
    """

    capabilities: frozenset[Capability] = _CAPS
    requires_backend: str | None = None
    method_name: str = "scipy_local"

    def __init__(
        self,
        problem: OptimizationProblem,
        method: str | None = None,
        on_failure: str = "penalty",
    ):
        self.problem = problem
        self.method = method
        self._on_failure = normalize_failure_mode(on_failure)
        self._ref_merit: float = 1.0

    def run(
        self,
        *,
        callback: Any = None,
        maxiter: int = 1000,
        tol: float = 1e-3,
        on_failure: str | None = None,
        **_: Any,
    ) -> Any:
        """Run scipy.optimize.minimize and write back optimal params."""
        if on_failure is not None:
            self._on_failure = normalize_failure_mode(on_failure)
        from scipy import optimize

        x0 = be.to_numpy([var.value for var in self.problem.variables])
        bounds = tuple([var.bounds for var in self.problem.variables])
        options = {"maxiter": maxiter}

        scipy_method = self.method if self.method != "Default" else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.minimize(
                self._fun,
                x0,
                method=scipy_method,
                bounds=bounds,
                options=options,
                tol=tol,
                callback=callback,
            )

        for i, var in enumerate(self.problem.variables):
            var.update(result.x[i])
        self.problem.update_optics()
        return result

    def _fun(self, x: Any) -> float:
        import math

        for i, var in enumerate(self.problem.variables):
            var.update(be.array(x[i]))
        self.problem.update_optics()
        try:
            rss = self.problem.sum_squared()
            val = float(be.to_numpy(rss))
        except Exception as exc:  # noqa: BLE001
            return self._fun_failure(exc)
        if not math.isfinite(val):
            return self._fun_failure(None)
        self._ref_merit = val
        return val

    def _fun_failure(self, exc: Exception | None) -> float:
        """Map a non-finite / failed merit to the active failure policy.

        ``scipy.optimize.minimize`` cannot consume ``NaN``, so any policy other
        than ``"raise"`` resolves to a finite, scale-proportional penalty (a
        multiple of the last finite merit), never a hardcoded ``1e10``.
        """
        if self._on_failure == "raise":
            raise OptimizationFailure(
                "Merit evaluation failed (non-finite or ray-trace error)."
            ) from exc
        return penalty_merit(self._ref_merit)
