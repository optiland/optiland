"""IdentityController — pass-through step controller.

Returns the proposed direction unchanged with ``accepted=True``.  Used as
the default controller for first-order stepped optimizers (torch Adam/SGD)
where the step computation is internal to the optimizer.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import StepInfo, StepOutcome

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState


class IdentityController:
    """Pass-through: returns the direction unchanged."""

    def reset(self, state: OptimizationState) -> None:  # noqa: ARG002
        pass

    def transform(  # noqa: ARG002
        self, direction, info: StepInfo, state: OptimizationState
    ) -> StepOutcome:
        return StepOutcome(delta_x=direction, accepted=True)
