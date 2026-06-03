"""TorchParameterSync — single owner of the nn.Parameter ↔ problem.variables sync.

All three torch consumers (AutogradEvaluator, TorchOptimizer, OpticalSystemModule)
delegate to this class.  No other file should inline var.update(param) + update_optics()
loops.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem

try:
    import torch
    import torch.nn as nn
except (ImportError, ModuleNotFoundError):
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class TorchParameterSync:
    """Single owner of the ``nn.Parameter`` ↔ ``problem.variables`` sync.

    Responsibilities (and only these):

    * build one ``nn.Parameter`` per variable, on the backend's current device;
    * :meth:`write_params`: push parameter values into variables then call
      ``problem.update_optics()`` — preserving the autograd graph (no
      ``.item()``, no ``no_grad``);
    * :meth:`load_x`: copy a 1-D tensor/array into the params (data-only);
    * :meth:`read_x`: stack detached param values to a 1-D on-device tensor;
    * :meth:`clamp_bounds`: in-place bounds clamp for all params.

    Params are constructed via ``be.array(...)`` so they inherit the
    configured device and precision — CUDA-correct by construction.

    Args:
        problem: The optimization problem whose variables are wrapped.
    """

    def __init__(self, problem: OptimizationProblem) -> None:
        if torch is None:
            raise ImportError("torch is required for TorchParameterSync")
        self.problem = problem
        if not be.grad_mode.requires_grad:
            be.grad_mode.enable()
        self.params: list[Any] = [
            nn.Parameter(be.array(float(var.value))) for var in problem.variables
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write_params(self) -> None:
        """Push current param values into variables and update optics.

        Must be called inside a grad-enabled context when the graph needs to
        stay live (e.g. during forward/backward in autograd paths).
        Does **not** call ``.item()`` — the graph is preserved.
        """
        for i, param in enumerate(self.params):
            self.problem.variables[i].update(param)
        self.problem.update_optics()

    def load_x(self, x: Any) -> None:
        """Copy values from ``x`` into params (data-only, no grad tracking).

        Args:
            x: 1-D tensor or array of length ``n_vars``.  Values are written
               via ``param.data.copy_`` (vectorised, no per-element Python loop).
        """
        import torch as _torch

        with _torch.no_grad():
            for i, param in enumerate(self.params):
                val = x[i]
                if isinstance(val, _torch.Tensor):
                    param.data.copy_(val.detach().to(param.device, param.dtype))
                else:
                    param.data.fill_(float(val))

    def read_x(self) -> Any:
        """Return current param values stacked as a 1-D detached tensor."""
        return torch.stack([p.detach().clone() for p in self.params])

    def clamp_bounds(self) -> None:
        """Clamp all params to their variable bounds in-place (no grad)."""
        with torch.no_grad():
            for i, param in enumerate(self.params):
                lo, hi = self.problem.variables[i].bounds
                if lo is not None and hi is not None:
                    param.data.clamp_(lo, hi)
                elif lo is not None:
                    param.data.clamp_(min=lo)
                elif hi is not None:
                    param.data.clamp_(max=hi)
