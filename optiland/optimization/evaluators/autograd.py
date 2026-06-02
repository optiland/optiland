"""AutogradEvaluator — torch autograd path

Computes value, gradient, and Jacobian using PyTorch's autograd engine.
Uses the stateful forward pass (D4): parameters are pushed into surfaces via
``var.update(param)`` each call, then ``sum_squared()`` / ``weighted_residuals()``
are evaluated through the existing trace pipeline.

The ``jacobian()`` row-by-row approach (D4 default) loops over the ``m``
residuals and calls ``torch.autograd.grad`` with ``retain_graph=True``.  A
faster functional path (``torch.func.jacrev``) is deferred to Phase 4.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.optimization.state import EvalCapability

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem

try:
    import torch
    import torch.nn as nn
except (ImportError, ModuleNotFoundError):
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

_PROVIDES = frozenset(
    {
        EvalCapability.VALUE,
        EvalCapability.RESIDUALS,
        EvalCapability.GRADIENT,
        EvalCapability.JACOBIAN,
    }
)


class AutogradEvaluator:
    """Evaluator for the torch backend using PyTorch autograd.

    Maintains a list of ``nn.Parameter`` objects (one per variable) that
    serve as the leaf tensors for gradient computation.  Values are propagated
    into the problem's variables via ``var.update(param)`` before each trace,
    mirroring ``TorchBaseOptimizer``'s approach.

    Args:
        problem: The optimization problem to wrap.
    """

    def __init__(self, problem: OptimizationProblem):
        if torch is None:
            raise ImportError("torch is required for AutogradEvaluator")
        assert be.get_backend() == "torch", (
            "AutogradEvaluator requires the 'torch' backend. "
            "Call be.set_backend('torch') first."
        )
        assert not any(isinstance(var.value, str) for var in problem.variables), (
            "Glass/material variables are not supported by AutogradEvaluator. "
            "Use GlassExpert directly."
        )
        self.problem = problem
        self.backend: str = "torch"
        self.n_vars: int = len(list(problem.variables))
        self.provides: frozenset[EvalCapability] = _PROVIDES

        # Leaf tensors — the canonical "x" for this evaluator
        if not be.grad_mode.requires_grad:
            be.grad_mode.enable()
        self._params: list[nn.Parameter] = [
            nn.Parameter(be.array(float(var.value))) for var in problem.variables
        ]

    # ------------------------------------------------------------------
    # read / write
    # ------------------------------------------------------------------

    def read_x(self) -> Any:
        """Return current param values stacked as a 1-D detached tensor."""
        return torch.stack([p.detach().clone() for p in self._params])

    def write_x(self, x: Any) -> None:
        """Sync params from ``x``, then push into variables and update optics.

        ``x`` may be a 1-D tensor or a numpy array; values are written as
        scalars so the params remain the leaf nodes.
        """
        with torch.no_grad():
            for i, param in enumerate(self._params):
                v = x[i]
                val = float(v.item()) if hasattr(v, "item") else float(v)
                param.data.fill_(val)
        for i, param in enumerate(self._params):
            self.problem.variables[i].update(param)
        self.problem.update_optics()

    # ------------------------------------------------------------------
    # Evaluation primitives
    # ------------------------------------------------------------------

    def value(self, x: Any) -> Any:
        """Evaluate merit at ``x``; returns a backend tensor (no ``.item()``)."""
        self.write_x(x)
        return self.problem.sum_squared()

    def residuals(self, x: Any) -> Any:
        """Return ``weighted_residuals()`` at ``x`` as a backend tensor."""
        self.write_x(x)
        return self.problem.weighted_residuals()

    # ------------------------------------------------------------------
    # Autograd gradient
    # ------------------------------------------------------------------

    def gradient(self, x: Any) -> Any:
        """Compute d(value)/d(x) via ``backward()``.

        Clears existing gradients before the forward pass and returns a
        detached tensor of shape ``(n,)``.  Does **not** call ``.item()``
        (no device sync in the hot loop — D9).
        """
        with torch.no_grad():
            for i, param in enumerate(self._params):
                v = x[i]
                val = float(v.item()) if hasattr(v, "item") else float(v)
                param.data.fill_(val)
        for param in self._params:
            if param.grad is not None:
                param.grad.zero_()

        with be.grad_mode.temporary_enable():
            for i, param in enumerate(self._params):
                self.problem.variables[i].update(param)
            self.problem.update_optics()
            loss = self.problem.sum_squared()
            loss.backward()

        return torch.stack([p.grad.detach().clone() for p in self._params])

    # ------------------------------------------------------------------
    # Autograd Jacobian — stateful row-by-row path (D4)
    # ------------------------------------------------------------------

    def jacobian(self, x: Any) -> Any:
        """Compute d(residuals)/d(x) row-by-row via ``torch.autograd.grad``.

        Returns a tensor of shape ``(m, n)``.  Each row requires one backward
        pass with ``retain_graph=True`` for all but the last row.

        This is the default *stateful* path (D4).  A functional
        ``torch.func.jacrev`` path is deferred to Phase 4.
        """
        with torch.no_grad():
            for i, param in enumerate(self._params):
                v = x[i]
                val = float(v.item()) if hasattr(v, "item") else float(v)
                param.data.fill_(val)

        with be.grad_mode.temporary_enable():
            for i, param in enumerate(self._params):
                self.problem.variables[i].update(param)
            self.problem.update_optics()
            r = self.problem.weighted_residuals()

        m = r.shape[0]
        rows = []
        for i in range(m):
            grads = torch.autograd.grad(
                r[i], self._params, retain_graph=(i < m - 1), allow_unused=True
            )
            row = torch.cat(
                [g.reshape(-1) if g is not None else torch.zeros(1) for g in grads]
            )
            rows.append(row)
        return torch.stack(rows)  # (m, n)
