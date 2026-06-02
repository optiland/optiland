"""TorchOptimizer — stepped adapter wrapping torch.optim (Adam / SGD).

Decomposes ``TorchBaseOptimizer.optimize()`` into ``initialize`` + ``step``
so that ``SteppedDriver`` owns the loop and observers / criteria integrate
cleanly.

Design (D9): ``loss.backward()`` is called inside the grad-mode context;
``.item()`` is NOT called in the hot loop — ``state.value`` remains a tensor
until ``OptimizationResult`` is built.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
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
        self._params: list[Any] = []
        self._torch_opt: Any = None
        self._scheduler: Any = None
        self._problem: Any = None

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
        """Create params, optimizer, and scheduler; return initial state."""
        problem = evaluator.problem
        self._problem = problem

        if not be.grad_mode.requires_grad:
            be.grad_mode.enable()

        # Create nn.Parameter leaf tensors from current variable values
        self._params = [
            nn.Parameter(be.array(float(var.value))) for var in problem.variables
        ]

        # Build the torch optimizer
        if self._optimizer_name == "adam":
            self._torch_opt = _optim.Adam(self._params, lr=self._lr)
        elif self._optimizer_name == "sgd":
            self._torch_opt = _optim.SGD(self._params, lr=self._lr)
        else:
            raise ValueError(f"Unknown torch optimizer: {self._optimizer_name!r}")

        self._scheduler = ExponentialLR(self._torch_opt, gamma=self._gamma)

        # Initial value (no grad needed yet)
        with torch.no_grad():
            for i, param in enumerate(self._params):
                problem.variables[i].update(param)
            problem.update_optics()

        with be.grad_mode.temporary_enable():
            init_val = problem.sum_squared()

        return OptimizationState(
            x=torch.stack([p.detach().clone() for p in self._params]),
            value=init_val.detach(),
            iteration=0,
            n_value_evals=1,
        )

    def step(self, state: OptimizationState) -> OptimizationState:
        """Run one gradient step; mutate state in place and return it."""
        self._torch_opt.zero_grad()

        with be.grad_mode.temporary_enable():
            # Push params into variables (maintains autograd graph)
            for i, param in enumerate(self._params):
                self._problem.variables[i].update(param)
            self._problem.update_optics()

            loss = self._problem.sum_squared()
            loss.backward()

        self._torch_opt.step()
        self._apply_bounds()
        self._scheduler.step()

        state.x = torch.stack([p.detach().clone() for p in self._params])
        state.value = loss.detach()
        state.iteration += 1
        state.n_value_evals += 1
        return state

    def converged(self, state: OptimizationState) -> bool:  # noqa: ARG002
        """TorchOptimizer has no intrinsic convergence; criteria own this."""
        return False

    # ------------------------------------------------------------------
    # Bounds enforcement
    # ------------------------------------------------------------------

    def _apply_bounds(self) -> None:
        with torch.no_grad():
            for i, param in enumerate(self._params):
                lo, hi = self._problem.variables[i].bounds
                if lo is not None and hi is not None:
                    param.data.clamp_(lo, hi)
                elif lo is not None:
                    param.data.clamp_(min=lo)
                elif hi is not None:
                    param.data.clamp_(max=hi)
