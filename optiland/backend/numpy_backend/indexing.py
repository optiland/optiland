"""
NumPy backend -- array utilities, shape, and indexing operations.

Provides IndexingMixin, one of the mixins composed into
NumpyBackend (see optiland/backend/numpy_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray


class IndexingMixin:
    """Array utilities, shape, and indexing operations."""

    # ------------------------------------------------------------------
    # Array utilities
    # ------------------------------------------------------------------

    def cast(self, x: ArrayLike) -> NDArray:
        """Cast x to the current floating-point dtype.

        Args:
            x: Input data.

        Returns:
            NDArray: Array cast to current precision.
        """
        return np.array(x, dtype=self._dtype)

    def is_array_like(self, x: Any) -> bool:
        """Return True if x is a list, tuple, or ndarray.

        Args:
            x: Object to check.

        Returns:
            bool: True if x is array-like.
        """
        return isinstance(x, np.ndarray | list | tuple)

    def arange_indices(self, start: Any, stop: Any = None, step: int = 1) -> NDArray:
        """Create an integer array of indices.

        Args:
            start: Start index (or stop if stop is None).
            stop: Stop index.
            step: Step size.

        Returns:
            NDArray: Integer index array.
        """
        return np.arange(start, stop, step, dtype=np.int64)

    def ravel(self, x: ArrayLike) -> NDArray:
        """Return a contiguous flattened array cast to float.

        Args:
            x: Input array.

        Returns:
            NDArray: 1-D float array.
        """
        return np.ravel(x).astype(float)

    # ------------------------------------------------------------------
    # Shape and indexing
    # ------------------------------------------------------------------

    def transpose(self, x: ArrayLike, axes: Sequence[int] | None = None) -> NDArray:
        """Permute the dimensions of x.

        Args:
            x: Input array.
            axes: Permutation of dimensions.

        Returns:
            NDArray: Transposed array.
        """
        return np.transpose(x, axes)

    def reshape(self, x: ArrayLike, shape: Sequence[int]) -> NDArray:
        """Return x with a new shape.

        Args:
            x: Input array.
            shape: New shape.

        Returns:
            NDArray: Reshaped array.
        """
        return np.reshape(x, shape)

    def atleast_1d(self, x: ArrayLike) -> NDArray:
        """Convert x to an array with at least one dimension.

        Args:
            x: Input data.

        Returns:
            NDArray: Array with at least 1 dimension, cast to float.
        """
        return np.atleast_1d(x).astype(float)

    def atleast_2d(self, x: ArrayLike) -> NDArray:
        """Convert x to an array with at least two dimensions.

        Args:
            x: Input data.

        Returns:
            NDArray: Array with at least 2 dimensions.
        """
        return np.atleast_2d(x)

    def as_array_1d(self, data: Any) -> NDArray:
        """Force conversion to a 1-D array.

        Args:
            data: Scalar, list, tuple, or array.

        Returns:
            NDArray: 1-D array.

        Raises:
            ValueError: If data type is not supported.
        """
        if isinstance(data, int | float):
            return self.array([data])
        elif isinstance(data, list | tuple):
            return self.array(data)
        elif self.is_array_like(data):
            return data.reshape(-1)
        else:
            raise ValueError(
                "Unsupported input type: expected scalar, list, tuple, or array-like."
            )

    def stack(self, xs: Sequence[ArrayLike], axis: int = 0) -> NDArray:
        """Join a sequence of arrays along a new axis.

        Args:
            xs: Sequence of arrays.
            axis: Axis along which to stack.

        Returns:
            NDArray: Stacked array.
        """
        return np.stack(xs, axis=axis)

    def concatenate(self, arrays: Sequence[ArrayLike], axis: int = 0) -> NDArray:
        """Join arrays along an existing axis.

        Args:
            arrays: Sequence of arrays to concatenate.
            axis: Axis along which to concatenate.

        Returns:
            NDArray: Concatenated array.
        """
        return np.concatenate(arrays, axis=axis)

    def flip(self, x: ArrayLike) -> NDArray:
        """Reverse the order of elements along axis 0.

        Args:
            x: Input array.

        Returns:
            NDArray: Flipped array.
        """
        return np.flip(x, axis=0)

    def roll(self, x: ArrayLike, shift: Any, axis: Any = ()) -> NDArray:
        """Roll x elements along the given axis.

        Args:
            x: Input array.
            shift: Number of places to shift.
            axis: Axis or axes along which to roll.

        Returns:
            NDArray: Rolled array.
        """
        return np.roll(x, shift, axis=axis if axis != () else None)

    def repeat(self, x: ArrayLike, repeats: int) -> NDArray:
        """Repeat elements of x.

        Args:
            x: Input array.
            repeats: Number of repetitions.

        Returns:
            NDArray: Repeated array.
        """
        return np.repeat(x, repeats)

    def broadcast_to(self, x: ArrayLike, shape: Sequence[int]) -> NDArray:
        """Broadcast x to the given shape.

        Args:
            x: Input array.
            shape: Target shape.

        Returns:
            NDArray: Broadcast view.
        """
        return np.broadcast_to(x, shape)

    def tile(self, x: ArrayLike, dims: Any) -> NDArray:
        """Construct an array by tiling x.

        Args:
            x: Input array.
            dims: Number of repetitions per dimension.

        Returns:
            NDArray: Tiled array.
        """
        return np.tile(x, dims)

    def expand_dims(self, x: ArrayLike, axis: int) -> NDArray:
        """Insert a new axis into x.

        Args:
            x: Input array.
            axis: Position of the new axis.

        Returns:
            NDArray: Expanded array.
        """
        return np.expand_dims(x, axis)

    def meshgrid(self, *arrays: ArrayLike) -> tuple[NDArray, ...]:
        """Return coordinate matrices from coordinate vectors (xy indexing).

        Args:
            *arrays: 1-D arrays representing grid coordinates.

        Returns:
            tuple[NDArray, ...]: Coordinate matrices.
        """
        return np.meshgrid(*arrays, indexing="xy")

    def unsqueeze_last(self, x: ArrayLike) -> NDArray:
        """Add a trailing dimension to x.

        Args:
            x: Input array.

        Returns:
            NDArray: Array with an extra trailing dimension.
        """
        return x[..., np.newaxis]
