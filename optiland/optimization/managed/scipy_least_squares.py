"""ScipyLeastSquaresAdapter — wraps LeastSquares (scipy.optimize.least_squares).

Note: ``scipy.optimize.least_squares`` has no ``callback`` argument, so
``on_step`` observers are not fired during the run.  ``on_start`` and
``on_end`` still fire normally.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optiland.optimization.state import Capability

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem

_CAPS = frozenset(
    {
        Capability.OBSERVE,
        Capability.SCIPY_CONSTRAINTS,
        Capability.BOUNDS,
    }
)


class ScipyLeastSquaresAdapter:
    """Adapter wrapping ``LeastSquares`` behind the managed adapter interface.

    Args:
        problem: The optimization problem.
        method_choice: SciPy least_squares method (``"lm"``, ``"trf"``,
            ``"dogbox"``).  Default ``"lm"``.
    """

    capabilities: frozenset[Capability] = _CAPS
    requires_backend: str | None = None
    method_name: str = "least_squares"

    def __init__(self, problem: OptimizationProblem, method_choice: str = "lm"):
        from optiland.optimization.optimizer.scipy.least_squares import LeastSquares

        self.problem = problem
        self.method_choice = method_choice
        self._inner = LeastSquares(problem)

    def run(  # noqa: ARG002
        self,
        *,
        callback: Any = None,
        maxiter: int | None = None,
        tol: float = 1e-3,
        **_: Any,
    ) -> Any:
        """Delegate to ``LeastSquares.optimize``."""
        return self._inner.optimize(
            maxiter=maxiter,
            tol=tol,
            method_choice=self.method_choice,
            disp=False,
        )
