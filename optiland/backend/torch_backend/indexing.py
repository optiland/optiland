"""
PyTorch backend -- array utilities, shape, and indexing operations.

Provides IndexingMixin, one of the mixins composed into
TorchBackend (see optiland/backend/torch_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


class IndexingMixin:
    """Array utilities, shape, and indexing operations."""

    # ------------------------------------------------------------------
    # Array utilities
    # ------------------------------------------------------------------

    def cast(self, x: Any) -> Tensor:
        """Cast x to the current floating-point precision.

        Args:
            x: Input data.

        Returns:
            Tensor: Cast tensor.
        """
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, device=self._device(), dtype=self._dtype())
        return x.to(device=self._device(), dtype=self._dtype())

    def is_array_like(self, x: Any) -> bool:
        """Return True if x is a tensor, ndarray, list, or tuple.

        Args:
            x: Object to check.

        Returns:
            bool: True if x is array-like.
        """
        return isinstance(x, torch.Tensor | np.ndarray | list | tuple)

    def arange_indices(self, start: Any, stop: Any = None, step: int = 1) -> Tensor:
        """Create a tensor of integer indices.

        Args:
            start: Start index (or stop if stop is None).
            stop: Stop index.
            step: Step size.

        Returns:
            Tensor: Long integer index tensor.
        """
        if stop is None:
            stop = start
            start = 0
        return torch.arange(
            start,
            stop,
            step,
            device=self._device(),
            dtype=torch.long,
            requires_grad=False,
        )

    def copy(self, x: Tensor) -> Tensor:
        """Return a copy of x.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Cloned tensor.
        """
        return x.clone()

    def size(self, x: Tensor) -> int:
        """Return the total number of elements in x.

        Args:
            x: Input tensor.

        Returns:
            int: Number of elements.
        """
        return torch.numel(x)

    def shape(self, tensor: Tensor) -> tuple[int, ...]:
        """Return the shape of a tensor.

        Args:
            tensor: Input tensor.

        Returns:
            tuple[int, ...]: Shape of the tensor.
        """
        return tensor.shape

    def isscalar(self, x: Any) -> bool:
        """Return True if x is a 0-dimensional tensor.

        Args:
            x: Input.

        Returns:
            bool: Whether x is a scalar tensor.
        """
        return torch.is_tensor(x) and x.dim() == 0

    def ravel(self, x: Tensor) -> Tensor:
        """Return a contiguous flattened tensor.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Flattened tensor.
        """
        return x.reshape(-1)

    # ------------------------------------------------------------------
    # Shape and indexing
    # ------------------------------------------------------------------

    def transpose(self, x: Tensor, axes: Sequence[int] | None = None) -> Tensor:
        """Permute the dimensions of x.

        Args:
            x: Input tensor.
            axes: The dimensions to permute. If None, reverses all dimensions.

        Returns:
            Tensor: Transposed tensor.
        """
        if axes is None:
            return x.t() if x.dim() == 2 else x.permute(*range(x.dim())[::-1])
        return x.permute(axes)

    def reshape(self, x: Tensor, shape: Sequence[int]) -> Tensor:
        """Return x with a new shape.

        Args:
            x: Input tensor.
            shape: New shape.

        Returns:
            Tensor: Reshaped tensor.
        """
        return x.view(shape)

    def atleast_1d(self, x: Any) -> Tensor:
        """Convert x to a tensor with at least one dimension.

        Args:
            x: Input data.

        Returns:
            Tensor: At least 1-D tensor.
        """
        t = torch.as_tensor(x, dtype=self._dtype(), device=self._device())
        return t.unsqueeze(0) if t.ndim == 0 else t

    def atleast_2d(self, x: Any) -> Tensor:
        """Convert x to a tensor with at least two dimensions.

        Args:
            x: Input data.

        Returns:
            Tensor: At least 2-D tensor.
        """
        t = torch.as_tensor(x, dtype=self._dtype(), device=self._device())
        if t.ndim == 0:
            return t.unsqueeze(0).unsqueeze(0)
        if t.ndim == 1:
            return t.unsqueeze(0)
        return t

    def as_array_1d(self, data: Any) -> Tensor:
        """Force conversion to a 1-D tensor.

        Args:
            data: Scalar, list, tuple, or tensor.

        Returns:
            Tensor: 1-D tensor.

        Raises:
            ValueError: If data type is not supported.
        """
        if isinstance(data, int | float):
            return self.array([data])
        if isinstance(data, list | tuple):
            return self.array(data)
        if self.is_array_like(data):
            return self.array(data).reshape(-1)
        raise ValueError("Unsupported type for as_array_1d")

    def stack(self, xs: Sequence[Any], axis: int = 0) -> Tensor:
        """Join a sequence of tensors along a new axis.

        Args:
            xs: Sequence of tensors.
            axis: Axis along which to stack.

        Returns:
            Tensor: Stacked tensor.
        """
        return torch.stack([self.cast(x) for x in xs], dim=axis)

    def concatenate(self, arrays: Sequence[Any], axis: int = 0) -> Tensor:
        """Join tensors along an existing axis.

        Args:
            arrays: Sequence of tensors to concatenate.
            axis: Axis along which to concatenate.

        Returns:
            Tensor: Concatenated tensor.
        """
        return torch.cat(arrays, dim=axis)

    def flip(self, x: Tensor) -> Tensor:
        """Reverse the order of elements along axis 0.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Flipped tensor.
        """
        return torch.flip(x, dims=(0,))

    def roll(self, x: Tensor, shift: Any, axis: Any = ()) -> Tensor:
        """Roll tensor elements along the given axis.

        Args:
            x: Input tensor.
            shift: Number of places to shift.
            axis: Axis or axes along which to roll.

        Returns:
            Tensor: Rolled tensor.
        """
        return torch.roll(x, shifts=shift, dims=axis)

    def repeat(self, x: Tensor, repeats: int) -> Tensor:
        """Repeat elements of x.

        Args:
            x: Input tensor.
            repeats: Number of repetitions.

        Returns:
            Tensor: Repeated tensor.
        """
        return torch.repeat_interleave(x, repeats)

    def broadcast_to(self, x: Tensor, shape: Sequence[int]) -> Tensor:
        """Broadcast x to the given shape.

        Args:
            x: Input tensor.
            shape: Target shape.

        Returns:
            Tensor: Broadcast tensor.
        """
        return x.expand(shape)

    def tile(self, x: Tensor, dims: Any) -> Tensor:
        """Construct a tensor by tiling x.

        Args:
            x: Input tensor.
            dims: Number of repetitions per dimension.

        Returns:
            Tensor: Tiled tensor.
        """
        return torch.tile(x, dims if isinstance(dims, list | tuple) else (dims,))

    def expand_dims(self, x: Tensor, axis: int) -> Tensor:
        """Insert a new axis into x.

        Args:
            x: Input tensor.
            axis: Position of the new axis.

        Returns:
            Tensor: Tensor with new axis.
        """
        return torch.unsqueeze(x, axis)

    def meshgrid(self, *arrays: Tensor) -> tuple[Tensor, ...]:
        """Return coordinate matrices from coordinate vectors (xy indexing).

        Args:
            *arrays: 1-D tensors representing grid coordinates.

        Returns:
            tuple[Tensor, ...]: Coordinate matrices.
        """
        return torch.meshgrid(*arrays, indexing="xy")

    def unsqueeze_last(self, x: Tensor) -> Tensor:
        """Add a trailing dimension to x.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Tensor with extra trailing dimension.
        """
        return x.unsqueeze(-1)
