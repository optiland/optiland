"""ConstraintStrategy protocol and CompositeStrategy.

Strategies are thin adapters over existing constraint mechanisms:
  - ``BoxBoundsStrategy`` reads ``var.bounds``.
  - ``ScipyNativeStrategy`` forwards bounds/constraints to SciPy.

Hard equality / inequality constraints are declared on the problem
(``add_constraint``) and solved by the KKT active-set controller.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState


@runtime_checkable
class ConstraintStrategy(Protocol):
    """Protocol for constraint strategies.

    ``prepare`` is called once before the run.  ``apply_to_step`` is called
    every accepted step by stepped optimizers that declare
    ``PER_STEP_CONSTRAINTS``.  ``to_scipy`` is called once to build the
    ``bounds=`` / ``constraints=`` arguments for SciPy.
    """

    def prepare(self, evaluator: Any, variables: Any) -> None: ...  # pragma: no cover

    def apply_to_step(self, x_proposed: Any, state: OptimizationState) -> Any:
        """Return ``x_proposed`` (possibly clamped / projected)."""
        ...  # pragma: no cover

    def to_scipy(self) -> list | None: ...  # pragma: no cover

    def is_feasible(self, x: Any, tol: float = 1e-8) -> bool: ...  # pragma: no cover


class CompositeStrategy:
    """Chains multiple ``ConstraintStrategy`` objects.

    ``apply_to_step`` applies each strategy in order.
    ``to_scipy`` merges all non-None results.

    Args:
        strategies: Sequence of strategies to compose.
    """

    def __init__(self, strategies: list[ConstraintStrategy]):
        self._strategies = list(strategies)

    def prepare(self, evaluator: Any, variables: Any) -> None:
        for s in self._strategies:
            s.prepare(evaluator, variables)

    def apply_to_step(self, x_proposed: Any, state: OptimizationState) -> Any:
        x = x_proposed
        for s in self._strategies:
            x = s.apply_to_step(x, state)
        return x

    def to_scipy(self) -> list | None:
        merged: list = []
        for s in self._strategies:
            r = s.to_scipy()
            if r:
                merged.extend(r)
        return merged or None

    def is_feasible(self, x: Any, tol: float = 1e-8) -> bool:
        return all(s.is_feasible(x, tol) for s in self._strategies)
