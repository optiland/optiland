"""Machine Learning Wrappers

This module contains wrappers for integrating Optiland with PyTorch. The
OpticalSystemModule class is a key component that allows users to define and optimize
optical systems within the PyTorch ecosystem. The `forward` method can be customized or
overridden to implement specific optical system behaviors.

Kramer Harrison, 2025
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
    """
    A PyTorch nn.Module that wraps an Optiland OptimizationProblem.

    This class exposes the optical system's variables as trainable nn.Parameters,
    allowing the entire system to be integrated as a differentiable layer into
    larger machine learning models.

    Args:
        optic (Optic): The optical system definition.
        problem (OptimizationProblem): The optimization problem defining variables and
            objectives.
        objective_fn (Callable[[], torch.Tensor] | None): An optional callable tha
            takes no arguments and returns a scalar PyTorch tensor representing the loss
            or metric to be optimized. If None, problem.sum_squared() is used as
            default.
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

        # Ensure gradients are enabled for PyTorch operations
        if not be.grad_mode.requires_grad:
            warnings.warn("Gradient tracking is enabled for PyTorch.", stacklevel=2)
            be.grad_mode.enable()

        self.optic = optic
        self.problem = problem

        # Initialize parameters as torch.nn.Parameter objects
        initial_params = [var.value for var in self.problem.variables]
        self.params = nn.ParameterList(
            [torch.nn.Parameter(be.array(p)) for p in initial_params]
        )

        # Store the original variable definitions
        self._original_variables = self.problem.variables

        # Store the user-provided objective function or set a default
        self.objective_fn = (
            objective_fn if objective_fn is not None else self._default_loss
        )

    def _default_loss(self) -> torch.Tensor:
        """
        The default loss function, which computes the sum of squared errors from the
        provided optimization problem.

        Returns:
            torch.Tensor: The computed loss value.
        """
        return self.problem.sum_squared()

    def _sync_params_to_problem(self):
        """
        Pushes the current tensor values from the nn.Parameters into the problem
        variables. This operation is part of the computation graph.
        """
        # Drop the graph of the last forward pass from shared arrays first,
        # so that this pass's backward does not run into it.
        for var in self._original_variables:
            var.release_graph()
        for i, param in enumerate(self.params):
            var = self._original_variables[i]
            var.update(param)

    def apply_bounds(self):
        """
        Applies the defined bounds to the parameters in-place.
        This should be called after each optimizer step to enforce constraints.
        """
        with torch.no_grad():  # Operations here shouldn't be part of the gradient graph
            for i, param in enumerate(self.params):
                var = self._original_variables[i]
                # Params and var.bounds are both in scaled space already
                min_val, max_val = var.bounds

                # Clamp the parameter data to the defined bounds
                if min_val is not None and max_val is not None:
                    param.data.clamp_(min_val, max_val)
                elif min_val is not None:
                    param.data.clamp_(min=min_val)
                elif max_val is not None:
                    param.data.clamp_(max=max_val)

    def forward(self) -> torch.Tensor:
        """
        Defines the forward pass of the optical system.

        This implementation synchronizes the PyTorch parameters with the Optiland
        problem variables, updates the optics, and then computes the loss using
        either the user-provided objective function or the default sum of squared
        errors. The output is a differentiable scalar tensor.

        Users are encouraged to customize or override this method to suit their
        specific optimization objectives.
        """
        # 1. Synchronize the nn.Parameter values with the Optiland problem variables.
        self._sync_params_to_problem()

        # 2. Update dependent properties within the optical system
        self.problem.update_optics()

        # 3. Compute the objective using the stored objective_fn
        loss = self.objective_fn()
        return loss
