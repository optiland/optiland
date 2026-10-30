"""Torch Parameter Sync

:class:`TorchParameterSync` is the single owner of the ``nn.Parameter`` <->
``problem.variables`` synchronization used by the autograd evaluator. No other
code should inline ``var.update(param)`` + ``update_optics()`` loops for it.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem

try:
    import torch
    from torch import nn
except (ImportError, ModuleNotFoundError):  # pragma: no cover - torch optional
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class TorchParameterSync:
    """Own the ``nn.Parameter`` <-> ``problem.variables`` sync.

    Responsibilities (and only these):

    * build one ``nn.Parameter`` per variable on the backend's device and
      precision (via ``be.array``);
    * :meth:`write_params`: push the parameters into the variables, then
      update the optics, keeping the autograd graph (no ``.item()``);
    * :meth:`load_x`: copy a 1-D tensor or array into the parameters
      (data only);
    * :meth:`read_x`: stack the detached parameter values.

    Args:
        problem: The optimization problem whose variables are wrapped.

    Raises:
        ImportError: If torch is not installed.
    """

    def __init__(self, problem: OptimizationProblem) -> None:
        if torch is None:
            raise ImportError("torch is required for TorchParameterSync")
        self.problem = problem
        if not be.grad_mode.requires_grad:
            be.grad_mode.enable()
        self.params: list[Any] = [
            nn.Parameter(be.array(float(be.to_numpy(var.value))))
            for var in problem.variables
        ]

    def write_params(self) -> None:
        """Push the parameter values into the variables and update the optics.

        Call inside a grad-enabled context to keep the graph live.
        """
        for var, param in zip(self.problem.variables, self.params, strict=True):
            var.update(param)
        self.problem.update_optics()

    def load_x(self, x: Any) -> None:
        """Copy the values of ``x`` into the parameters (no grad tracking).

        Args:
            x: 1-D tensor or array of length ``n_vars``.
        """
        with torch.no_grad():
            for i, param in enumerate(self.params):
                value = x[i]
                if isinstance(value, torch.Tensor):
                    param.data.copy_(value.detach().to(param.device, param.dtype))
                else:
                    param.data.fill_(float(value))

    def read_x(self) -> Any:
        """Return the parameter values stacked as a 1-D detached tensor."""
        return torch.stack([param.detach().clone() for param in self.params])
