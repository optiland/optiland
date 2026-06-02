"""ScipyNativeStrategy — forwards bounds and constraints directly to SciPy.

A thin pass-through wrapper.  ``apply_to_step`` is a no-op (SciPy handles
feasibility internally).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState


class ScipyNativeStrategy:
    """Forwards pre-built SciPy bounds / constraint dicts to the ManagedDriver.

    Args:
        scipy_constraints: A list of SciPy constraint dicts, or None.
        scipy_bounds: A SciPy-compatible bounds sequence, or None.
    """

    def __init__(
        self,
        scipy_constraints: list | None = None,
        scipy_bounds: Any = None,
    ):
        self._constraints = scipy_constraints
        self._scipy_bounds = scipy_bounds

    def prepare(self, evaluator: Any, variables: Any) -> None:  # noqa: ARG002
        pass

    def apply_to_step(self, x_proposed: Any, state: OptimizationState) -> Any:  # noqa: ARG002
        return x_proposed  # SciPy handles feasibility

    def to_scipy(self) -> list | None:
        result: list = []
        if self._constraints:
            result.extend(self._constraints)
        if self._scipy_bounds is not None:
            result.append({"type": "_bounds_", "data": self._scipy_bounds})
        return result or None

    def is_feasible(self, x: Any, tol: float = 1e-8) -> bool:  # noqa: ARG002
        return True  # SciPy handles feasibility; we don't duplicate the check
