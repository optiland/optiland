"""StepController protocol and StepInfo / StepOutcome value objects.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState


@dataclass
class StepInfo:
    """Read-only context passed from the optimizer to the controller.

    Attributes:
        direction: Raw descent / Gauss-Newton direction (shape ``(n,)``).
        residuals: Current residual vector (shape ``(m,)``); may be None.
        jacobian: Current Jacobian (shape ``(m, n)``); may be None.
        gradient: Current gradient (shape ``(n,)``); may be None.
        value_fn: Callable ``(x) -> scalar``; used by line-search controllers.
    """

    direction: Any
    residuals: Any = None
    jacobian: Any = None
    gradient: Any = None
    value_fn: Any = None
    extra: dict = field(default_factory=dict)


@dataclass
class StepOutcome:
    """Result returned by a step controller.

    Attributes:
        delta_x: Accepted step vector (shape ``(n,)``).
        accepted: False means the controller rejected the step; the optimizer
            should retry (e.g. increase λ in LM).
        info: Controller-private debug/state dict.
    """

    delta_x: Any
    accepted: bool = True
    info: dict = field(default_factory=dict)


@runtime_checkable
class StepController(Protocol):
    """Protocol for step controllers.

    A step controller takes a proposed descent direction and transforms it
    into an accepted step (scaling / damping / line-search).  It may
    reject the step (``accepted=False``) to ask the optimizer to retry.
    """

    def reset(self, state: OptimizationState) -> None:
        """Reset internal state at the start of a run."""
        ...  # pragma: no cover

    def transform(  # noqa: E501
        self, direction: Any, info: StepInfo, state: OptimizationState
    ) -> StepOutcome:
        """Map a proposed direction to an accepted step."""
        ...  # pragma: no cover
