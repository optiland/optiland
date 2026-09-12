"""Torch conic execution, CPU acceleration, and implicit derivatives.

Part of the optional Torch backend. CPU float64 tensors expose NumPy views of
their existing storage; CUDA tensors retain native Torch operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from optiland.backend._conic import _Candidates, _conic_candidates, _select_distance
from optiland.backend.numpy_backend.conic import (
    _numpy_conic_candidates,
    _numpy_conic_distance,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def can_fuse_cpu(values: tuple, radius: Any, conic: Any) -> bool:
    """Check the storage and shape contract required by the compiled loop."""
    first = values[0]
    parameters = (radius, conic)
    return (
        isinstance(first, Tensor)
        and first.ndim == 1
        and all(
            type(value) in (Tensor, torch.nn.Parameter)
            and value.device.type == "cpu"
            and value.dtype == torch.float64
            and value.layout == torch.strided
            and not value.is_conj()
            and not value.is_neg()
            for value in (*values, *parameters)
        )
        and all(value.shape == first.shape for value in values)
        and all(value.ndim <= 1 and value.numel() == 1 for value in parameters)
    )


def _partials(inputs: tuple[Tensor, ...], distance: Tensor, regular: Tensor) -> tuple:
    """Differentiate F(p + t*v, R, k) = 0 on the selected regular branch.

    With hit (u, v, w), half the directional derivative is
    D = u*L + v*M + ((1+k)*w-R)*N. Each partial is -F_parameter/(2*D).
    Keeping the saved output distance in this graph supports double backward.
    Inactive inputs are replaced before arithmetic to avoid 0*NaN derivatives.
    """
    x, y, z, L, M, N, radius, conic = (
        torch.where(regular, value, 0.0) for value in inputs
    )
    t = torch.where(regular, distance, 0.0)
    u, v, w = x + t * L, y + t * M, z + t * N
    g = (1 + conic) * w - radius
    denominator = u * L + v * M + g * N
    safe = regular & (denominator != 0) & torch.isfinite(denominator)
    inverse = torch.where(safe, 1.0, 0.0) / torch.where(safe, denominator, 1.0)
    dx, dy, dz = -u * inverse, -v * inverse, -g * inverse
    return dx, dy, dz, dx * t, dy * t, dz * t, w * inverse, -0.5 * w * w * inverse


class _ConicCPU(torch.autograd.Function):
    """Use compiled forward values and the conic's exact implicit derivative."""

    @staticmethod
    def forward(contains: Callable | None, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        values = tuple(value.detach().numpy() for value in inputs[:6])
        if contains is not None:
            roots = _Candidates(
                *(
                    torch.from_numpy(value)
                    for value in _numpy_conic_candidates(
                        *values, inputs[6].item(), inputs[7].item()
                    )
                )
            )
            return _select_distance(
                roots, inputs[:6], contains, torch.where
            ), roots.regular
        distance, regular = _numpy_conic_distance(
            *values, inputs[6].item(), inputs[7].item()
        )
        return torch.from_numpy(distance), torch.from_numpy(regular)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
        inputs = inputs[1:]
        distance, regular = output
        ctx.mark_non_differentiable(regular)
        ctx.save_for_backward(*inputs, distance, regular)
        ctx.save_for_forward(*inputs, distance, regular)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor, unused: Tensor) -> tuple:
        *inputs, distance, regular = ctx.saved_tensors
        return (
            None,
            *(
                (gradient * partial).sum_to_size(value.shape)
                for value, partial in zip(
                    inputs, _partials(inputs, distance, regular), strict=True
                )
            ),
        )

    @staticmethod
    def jvp(
        ctx: Any, contains_tangent: None, *tangents: Tensor | None
    ) -> tuple[Tensor, None]:
        *inputs, distance, regular = ctx.saved_tensors
        result = torch.zeros_like(distance)
        for partial, tangent in zip(
            _partials(inputs, distance, regular), tangents, strict=True
        ):
            if tangent is not None:
                result = result + partial * tangent
        return result, None

    @staticmethod
    def vmap(
        info: Any, in_dims: tuple, contains: Callable | None, *inputs: Tensor
    ) -> tuple:
        # Explicit batching preserves NumPy's storage access contract even
        # under torch.func transforms. Ordinary ray bundles use one fused call.
        in_dims = in_dims[1:]
        if info.batch_size == 0:
            shape = list(inputs[0].shape)
            if in_dims[0] is not None:
                del shape[in_dims[0]]
            distance = inputs[0].new_empty((0, *shape))
            return (distance, distance.new_empty(distance.shape, dtype=torch.bool)), (
                0,
                0,
            )
        outputs = [
            _ConicCPU.apply(
                contains,
                *(
                    value if dim is None else value.select(dim, i)
                    for value, dim in zip(inputs, in_dims, strict=True)
                ),
            )
            for i in range(info.batch_size)
        ]
        return tuple(torch.stack(values) for values in zip(*outputs, strict=True)), (
            0,
            0,
        )


def _epsilon(value: Tensor) -> float:
    """Use the native arithmetic result's floating dtype."""
    return torch.finfo(value.dtype).eps


class ConicMixin:
    """Finite-conic execution and differentiation for the Torch backend."""

    def conic_intersection(
        self,
        x: Tensor,
        y: Tensor,
        z: Tensor,
        L: Tensor,
        M: Tensor,
        N: Tensor,
        radius: Tensor,
        conic: Tensor,
        contains: Callable | None = None,
    ) -> Tensor:
        """Select a finite-conic intersection without changing dtype or device.

        See ``AbstractBackend.conic_intersection`` for the numerical contract.
        CPU float64 tensors may use shared compiled loops with implicit
        derivatives; other inputs retain native Torch operations and autograd.
        """
        values = (x, y, z, L, M, N)
        if can_fuse_cpu(values, radius, conic):
            return _ConicCPU.apply(contains, *values, radius, conic)[0]
        # Native where pairs scalar constants with existing tensors. No host
        # scalar tensor construction or device copy is needed for each guard.
        roots = _conic_candidates(
            *values, radius, conic, torch.where, torch.sqrt, torch.copysign, _epsilon
        )
        distance = _select_distance(roots, values, contains, torch.where)
        if distance.requires_grad:
            distance = torch.where(roots.regular, distance, distance.detach())
        return distance
