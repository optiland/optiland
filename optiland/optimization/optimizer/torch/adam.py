"""Torch Adam Optimizer

This module contains an implementation of the Adam optimizer for PyTorch.

Kramer Harrison, 2025
"""

from __future__ import annotations

try:
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import ExponentialLR, LRScheduler
except (ImportError, ModuleNotFoundError):
    torch = None
    optim = None
    ExponentialLR = None
    LRScheduler = None

from .base import TorchBaseOptimizer


class TorchAdamOptimizer(TorchBaseOptimizer):
    """PyTorch Adam optimizer.

    .. deprecated::
        Use ``optiland.optimization.minimize(problem, method='adam')``
        instead.  Removal in v0.7.0.
    """

    def __init__(self, problem):
        import warnings

        warnings.warn(
            "TorchAdamOptimizer is deprecated; use "
            "optiland.optimization.minimize(problem, method='adam'). "
            "Removal in v0.7.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(problem)

    def _create_optimizer_and_scheduler(
        self, lr: float, gamma: float
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """
        Creates and returns the Adam optimizer and an ExponentialLR scheduler.

        Args:
            lr (float): The learning rate.
            gamma (float): The decay factor for the learning rate.

        Returns:
            tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]: The
                optimizer and learning rate scheduler.
        """
        optimizer = optim.Adam(self.params, lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        return optimizer, scheduler
