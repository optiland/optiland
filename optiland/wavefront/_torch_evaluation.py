"""Optional Torch analytical derivatives for weighted RMS and accurate sums."""

# ruff: noqa: I002

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx, once_differentiable

from ._evaluation_math import _sum_samples_forward, _weighted_rms_components


class _CompensatedSum(torch.autograd.Function):
    """Attach the exact sum derivative to an error-compensated forward result."""

    @staticmethod
    def forward(ctx: FunctionCtx, values: Tensor) -> Tensor:
        """Keep all input derivatives, including operands whose value is zero."""
        ctx.input_shape = values.shape
        return _sum_samples_forward(values)

    @staticmethod
    def backward(ctx: FunctionCtx, gradient: Tensor) -> tuple[Tensor]:
        """Broadcast the upstream derivative independently of rounding recovery."""
        return (gradient.expand(ctx.input_shape),)


class _WeightedRMS(torch.autograd.Function):
    """Separate forward range scaling from the analytical OPD derivative.

    For root weights q, scaled values z, and h = ||q*z||, the derivative is
    (q/||q||) * (q*z/h). Both factors are bounded by one. In particular, the
    backward never multiplies by the OPD scale only to divide by it later.
    Weight derivatives and higher-order derivatives are deliberately unsupported.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, values: Tensor, sqrt_alpha: Tensor) -> Tensor:
        """Compute the stable forward value and retain its analytical direction."""
        scale = values.detach().abs().max()
        if scale == 0.0:
            derivative = torch.zeros_like(values)
            rms = values.sum() * 0.0
        else:
            weighted, weighted_norm, weight_norm = _weighted_rms_components(
                values, sqrt_alpha, scale
            )
            derivative = (sqrt_alpha / weight_norm) * (weighted / weighted_norm)
            rms = scale * (weighted_norm / weight_norm)
        ctx.save_for_backward(derivative)
        return rms

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, gradient: Tensor) -> tuple[Tensor, None]:
        """Apply the first-order OPD derivative with fixed weights and support."""
        (derivative,) = ctx.saved_tensors
        return gradient * derivative, None
