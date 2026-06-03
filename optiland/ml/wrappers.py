"""Machine Learning Wrappers

This module contains wrappers for integrating Optiland with PyTorch.
``OpticalSystemModule`` is the escape hatch for the data-scientist persona:
it wraps an optical system as a differentiable ``nn.Module`` so the optic can
be embedded in a larger network or trained with a custom loss and user-owned
loop.  For the common "just optimize this lens" case, use
``optiland.optimization.minimize()`` instead.

**Side effects:** ``OpticalSystemModule.forward()`` mutates the optic in place
on every call — surface parameters are updated to the current ``nn.Parameter``
values.  If you need a clean snapshot of the starting design, take it before
constructing this module.

Param ↔ surface sync is delegated to :class:`TorchParameterSync` (WS1), the
single shared sync core.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.optic import Optic
    from optiland.optimization.problem import OptimizationProblem


class OpticalSystemModule(nn.Module if nn is not None else object):
    """PyTorch ``nn.Module`` that wraps an Optiland ``OptimizationProblem``.

    Exposes the optical system's variables as trainable ``nn.Parameter``
    objects so the system can be composed into larger networks or trained with
    a custom loss function in a user-owned loop.

    **When to use this vs ``minimize()``:** use ``OpticalSystemModule`` when
    *you* own the training loop (custom loss, batched data, optic as a layer).
    Use ``minimize()`` when Optiland should own the loop.

    **Side effects:** ``forward()`` mutates the optic in place — on each call
    surface parameters are set to the current ``nn.Parameter`` values.
    ``result.x`` (when used with a torch optimizer) provides a copy of the
    final parameter vector.

    Args:
        optic: The optical system definition.
        problem: The optimization problem defining variables and objectives.
        objective_fn: Optional callable taking no arguments that returns a
            scalar ``torch.Tensor`` representing the loss.  If ``None``,
            ``problem.sum_squared()`` is used as the default.
    """

    def __init__(
        self,
        optic: Optic,
        problem: OptimizationProblem,
        objective_fn: Callable[[], torch.Tensor] | None = None,
    ):
        super().__init__()
        if torch is None:
            raise RuntimeError(
                "OpticalSystemModule requires the 'torch' package. "
                "Install PyTorch to use this class."
            )
        if be.get_backend() != "torch":
            raise RuntimeError("OpticalSystemModule requires the 'torch' backend.")

        if not be.grad_mode.requires_grad:
            warnings.warn("Gradient tracking is enabled for PyTorch.", stacklevel=2)
            be.grad_mode.enable()

        self.optic = optic
        self.problem = problem

        # Delegate param↔surface sync to the shared core (WS1)
        from optiland.optimization.evaluators._param_sync import TorchParameterSync

        self._sync = TorchParameterSync(problem)

        # Register params as nn.ParameterList so module.parameters() finds them
        # (required for composition into parent nn.Module — data-scientist use case).
        self.params = nn.ParameterList(self._sync.params)

        self.objective_fn = (
            objective_fn if objective_fn is not None else self._default_loss
        )

    def _default_loss(self) -> torch.Tensor:
        """Default loss: ``problem.sum_squared()``."""
        return self.problem.sum_squared()

    def apply_bounds(self) -> None:
        """Clamp all params to their variable bounds in-place (no grad)."""
        self._sync.clamp_bounds()

    def forward(self) -> torch.Tensor:
        """Forward pass: sync params → update optics → compute objective.

        **Side effect:** mutates the optic in place — surface parameters are
        updated to the current ``nn.Parameter`` values on every call.

        Users may override this method to implement custom forward behaviour.

        Returns:
            A differentiable scalar tensor (the objective value).
        """
        self._sync.write_params()
        loss = self.objective_fn()
        return loss
