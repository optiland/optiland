"""Torch Adam and SGD behind ``minimize(method="adam" | "sgd")``.

Private core: runs the existing ``TorchAdamOptimizer`` / ``TorchSGDOptimizer``
loop (fixed number of steps, autograd gradients of the merit) and drives
observers and stopping criteria from its per-step callback.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.optimization.failure import ConfigurationError
from optiland.optimization.state import OptimizationResult

if TYPE_CHECKING:
    from optiland.optimization.managed.base import RunMonitor
    from optiland.optimization.problem import OptimizationProblem

#: Torch first-order methods reachable through ``minimize()``.
TORCH_METHODS: frozenset[str] = frozenset({"adam", "sgd"})

#: Steps taken when ``maxiter`` is None.
DEFAULT_STEPS = 100


def run_torch_first_order(
    problem: OptimizationProblem,
    method: str,
    monitor: RunMonitor,
    *,
    maxiter: int | None,
    lr: float = 1e-2,
    gamma: float = 0.99,
) -> OptimizationResult:
    """Minimize the merit with a torch first-order optimizer.

    Args:
        problem: The optimization problem.
        method: ``"adam"`` or ``"sgd"``.
        monitor: Drives the observers and stopping criteria.
        maxiter: Number of steps, or None for :data:`DEFAULT_STEPS`.
        lr: Learning rate.
        gamma: Per-step learning-rate decay.

    Returns:
        The result; the problem's optics are left at the last step (or at the
        step where a stopping criterion fired).

    Raises:
        ConfigurationError: If the torch backend is not active, or a variable
            is a glass choice.
    """
    from optiland.optimization.optimizer.torch.adam import TorchAdamOptimizer
    from optiland.optimization.optimizer.torch.sgd import TorchSGDOptimizer

    if be.get_backend() != "torch":
        raise ConfigurationError(
            f"Method {method!r} requires the torch backend; call "
            "be.set_backend('torch') first."
        )
    if any(isinstance(var.value, str) for var in problem.variables):
        raise ConfigurationError(
            "Glass/material variables are not supported here; use GlassExpert."
        )
    optimizer = (TorchAdamOptimizer if method == "adam" else TorchSGDOptimizer)(problem)
    n_steps = DEFAULT_STEPS if maxiter is None else int(maxiter)
    x0 = np.array([float(p.item()) for p in optimizer.params])
    monitor.start(x0, float(be.to_numpy(problem.sum_squared())), 0)

    def callback(step: int, loss: float) -> None:
        x = np.array([float(p.item()) for p in optimizer.params])
        monitor.step(x, loss, step + 1)

    stopped = False
    try:
        raw = optimizer.optimize(
            n_steps=n_steps, lr=lr, gamma=gamma, disp=False, callback=callback
        )
        x, fun = np.array(raw.x, dtype=float), float(raw.fun)
    except StopIteration:
        stopped = True
        x = np.array([float(p.item()) for p in optimizer.params])
        for var, value in zip(problem.variables, optimizer.params, strict=True):
            var.update(value)
        problem.update_optics()
        fun = float(be.to_numpy(problem.sum_squared()))

    message = monitor.stop_reason if stopped else f"{n_steps} steps completed"
    result = OptimizationResult(
        x=x,
        fun=fun,
        success=True,
        status=-4 if stopped else 0,
        message=message,
        nfev=monitor.state.n_evals,
        nit=monitor.state.iteration,
        stop_reason=message,
        n_evals=monitor.state.n_evals,
        history=None,
        method=method,
    )
    return monitor.finish(result)
