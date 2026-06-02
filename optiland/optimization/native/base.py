"""SteppedOptimizer ABC — base for optimizers the driver loop owns.

Concrete stepped optimizers (TorchOptimizer, LevenbergMarquardt) inherit
from this class and declare their ``capabilities`` and ``requires_backend``.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland.optimization.constraints.base import ConstraintStrategy
    from optiland.optimization.control.base import StepController
    from optiland.optimization.evaluators.base import Evaluator
    from optiland.optimization.state import Capability, OptimizationState


class SteppedOptimizer(ABC):
    """Abstract base for optimizers whose loop is owned by ``SteppedDriver``.

    Subclasses must declare ``capabilities`` and override the three abstract
    methods.  Each ``step()`` call represents **one accepted step** (D11);
    inner retries (e.g. LM λ-search) are hidden inside ``step()``.
    """

    capabilities: frozenset[Capability] = frozenset()
    requires_backend: str | None = None

    @abstractmethod
    def initialize(
        self,
        evaluator: Evaluator,
        x0: Any,
        *,
        controller: StepController,
        constraints: ConstraintStrategy | None,
    ) -> OptimizationState:
        """Set up internal state and return the initial ``OptimizationState``."""
        ...  # pragma: no cover

    @abstractmethod
    def step(self, state: OptimizationState) -> OptimizationState:
        """Execute one accepted step; mutate ``state`` in place and return it."""
        ...  # pragma: no cover

    @abstractmethod
    def converged(self, state: OptimizationState) -> bool:
        """Return True when the optimizer's *intrinsic* convergence criterion is met.

        External stopping criteria are checked by the driver, not here.
        """
        ...  # pragma: no cover
