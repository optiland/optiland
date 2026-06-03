"""TorchOptimizer — stepped adapter wrapping torch.optim (Adam / SGD).

Decomposes ``TorchBaseOptimizer.optimize()`` into ``initialize`` + ``step``
so that ``SteppedDriver`` owns the loop and observers / criteria integrate
cleanly.

Design (D9): ``loss.backward()`` is called inside the grad-mode context;
``.item()`` is NOT called in the hot loop — ``state.value`` remains a tensor
until ``OptimizationResult`` is built.

Param ↔ surface sync is delegated to :class:`TorchParameterSync` (WS1).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.optimization.evaluators._param_sync import TorchParameterSync
from optiland.optimization.state import Capability, OptimizationState

from .base import SteppedOptimizer

if TYPE_CHECKING:
    from optiland.optimization.constraints.base import ConstraintStrategy
    from optiland.optimization.control.base import StepController
    from optiland.optimization.evaluators.base import Evaluator

try:
    import torch
    import torch.nn as nn
    import torch.optim as _optim
    from torch.optim.lr_scheduler import ExponentialLR
except (ImportError, ModuleNotFoundError):
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _optim = None  # type: ignore[assignment]
    ExponentialLR = None  # type: ignore[assignment]


_CAPS = frozenset(
    {
        Capability.OBSERVE,
        Capability.STEP_CONTROL,
        Capability.MUTATE_TERMINATION,
        Capability.BOUNDS,
    }
)


class TorchOptimizer(SteppedOptimizer):
    """Stepped optimizer wrapping ``torch.optim.Adam`` or ``torch.optim.SGD``.

    Args:
        optimizer_name: ``"adam"`` (default) or ``"sgd"``.
        lr: Learning rate (default 0.01).
        gamma: ExponentialLR decay factor (default 0.99).
    """

    capabilities: frozenset[Capability] = _CAPS
    requires_backend: str = "torch"

    def __init__(
        self,
        optimizer_name: str = "adam",
        lr: float = 1e-2,
        gamma: float = 0.99,
    ):
        if torch is None:
            raise ImportError("torch is required for TorchOptimizer")
        self._optimizer_name = optimizer_name
        self._lr = lr
        self._gamma = gamma
        self._sync: TorchParameterSync | None = None
        self._torch_opt: Any = None
        self._scheduler: Any = None

    # ------------------------------------------------------------------
    # SteppedOptimizer interface
    # ------------------------------------------------------------------

    def initialize(
        self,
        evaluator: Evaluator,
        x0: Any,
        *,
        controller: StepController,
        constraints: ConstraintStrategy | None,
    ) -> OptimizationState:
        """Create sync, optimizer, and scheduler; return initial state."""
        problem = evaluator.problem

        self._sync = TorchParameterSync(problem)
        # Load x0 so params reflect the starting point
        self._sync.load_x(x0)

        # Build the torch optimizer over sync.params
        if self._optimizer_name == "adam":
            self._torch_opt = _optim.Adam(self._sync.params, lr=self._lr)
        elif self._optimizer_name == "sgd":
            self._torch_opt = _optim.SGD(self._sync.params, lr=self._lr)
        else:
            raise ValueError(f"Unknown torch optimizer: {self._optimizer_name!r}")

        self._scheduler = ExponentialLR(self._torch_opt, gamma=self._gamma)

        # Push initial params into surfaces (no grad needed for initial value)
        with torch.no_grad():
            self._sync.write_params()

        with be.grad_mode.temporary_enable():
            init_val = problem.sum_squared()

        return OptimizationState(
            x=self._sync.read_x(),
            value=init_val.detach(),
            iteration=0,
            n_value_evals=1,
        )

    def step(self, state: OptimizationState) -> OptimizationState:
        """Run one gradient step; mutate state in place and return it."""
        self._torch_opt.zero_grad()

        with be.grad_mode.temporary_enable():
            self._sync.write_params()
            loss = self._sync.problem.sum_squared()
            loss.backward()

        self._torch_opt.step()
        self._sync.clamp_bounds()
        self._scheduler.step()

        state.x = self._sync.read_x()
        state.value = loss.detach()
        state.iteration += 1
        state.n_value_evals += 1
        return state

    def converged(self, state: OptimizationState) -> bool:  # noqa: ARG002
        """TorchOptimizer has no intrinsic convergence; criteria own this."""
        return False
