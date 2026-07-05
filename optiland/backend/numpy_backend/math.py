"""
NumPy backend -- reduction operations.

Provides MathMixin, one of the mixins composed into
NumpyBackend (see optiland/backend/numpy_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class MathMixin:
    """Reduction operations."""

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self, x: ArrayLike, axis: int | None = None) -> NDArray:
        """Sum array elements over a given axis.

        Args:
            x: Input array.
            axis: Axis along which to sum.

        Returns:
            NDArray: Sum of x.
        """
        return np.sum(x, axis=axis)

    def mean(
        self, x: ArrayLike, axis: int | None = None, keepdims: bool = False
    ) -> NDArray:
        """Compute the arithmetic mean along an axis.

        Args:
            x: Input array.
            axis: Axis along which to compute the mean.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            NDArray: Mean of x.
        """
        return np.mean(x, axis=axis, keepdims=keepdims)

    def std(self, x: ArrayLike, axis: int | None = None) -> NDArray:
        """Compute the standard deviation along an axis.

        Args:
            x: Input array.
            axis: Axis along which to compute the std.

        Returns:
            NDArray: Standard deviation.
        """
        return np.std(x, axis=axis)

    def max(self, x: ArrayLike) -> Any:
        """Return the maximum value of x.

        Args:
            x: Input array.

        Returns:
            float or NDArray: Maximum value.
        """
        return np.max(x)

    def min(self, x: ArrayLike) -> Any:
        """Return the minimum value of x.

        Args:
            x: Input array.

        Returns:
            float or NDArray: Minimum value.
        """
        return np.min(x)

    def argmin(self, x: ArrayLike, axis: int | None = None) -> NDArray:
        """Return indices of the minimum values along an axis.

        Args:
            x: Input array.
            axis: Axis along which to find the minimum.

        Returns:
            NDArray: Index array.
        """
        return np.argmin(x, axis=axis)

    def argwhere(self, x: ArrayLike) -> NDArray:
        """Return indices of non-zero elements.

        Args:
            x: Input array.

        Returns:
            NDArray: Index array of shape (N, ndim).
        """
        return np.argwhere(x)

    def clip(self, x: ArrayLike, a_min: Any, a_max: Any) -> NDArray:
        """Clip values in x to [a_min, a_max].

        Args:
            x: Input array.
            a_min: Minimum value.
            a_max: Maximum value.

        Returns:
            NDArray: Clipped array.
        """
        return np.clip(x, a_min, a_max)

    def where(self, condition: Any, x: Any, y: Any) -> NDArray:
        """Return elements from x or y depending on condition.

        Args:
            condition: Boolean array.
            x: Values where condition is True.
            y: Values where condition is False.

        Returns:
            NDArray: Output array.
        """
        return np.where(condition, x, y)

    def maximum(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        """Element-wise maximum of a and b.

        Args:
            a: First input array.
            b: Second input array.

        Returns:
            NDArray: Element-wise maximum.
        """
        return np.maximum(a, b)

    def minimum(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        """Element-wise minimum of a and b.

        Args:
            a: First input array.
            b: Second input array.

        Returns:
            NDArray: Element-wise minimum.
        """
        return np.minimum(a, b)

    def fmax(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        """Element-wise maximum, ignoring NaNs.

        Args:
            a: First input array.
            b: Second input array.

        Returns:
            NDArray: Element-wise maximum ignoring NaN.
        """
        return np.fmax(a, b)

    def power(self, x: ArrayLike, y: ArrayLike) -> NDArray:
        """Return x raised to the power y.

        Args:
            x: Base array.
            y: Exponent array.

        Returns:
            NDArray: x ** y.
        """
        return np.power(x, y)

    def diff(self, x: ArrayLike, n: int = 1, axis: int = -1, **kwargs: Any) -> NDArray:
        """Calculate the n-th discrete difference along the given axis.

        Args:
            x: Input array.
            n: Number of times to apply the difference.
            axis: Axis along which to compute differences.
            **kwargs: Additional keyword arguments forwarded to ``np.diff``
                (e.g. ``prepend``, ``append``).

        Returns:
            NDArray: Differences array.
        """
        return np.diff(x, n=n, axis=axis, **kwargs)

    def all(self, x: Any) -> bool:
        """Return True if all elements of x are True.

        Args:
            x: Input array.

        Returns:
            bool: Whether all elements are True.
        """
        return bool(np.all(x))

    def any(self, x: Any) -> bool:
        """Return True if any element of x is True.

        Args:
            x: Input array.

        Returns:
            bool: Whether any element is True.
        """
        return bool(np.any(x))

    def nanmax(
        self, x: ArrayLike, axis: int | None = None, keepdim: bool = False
    ) -> NDArray:
        """Return the maximum value, ignoring NaNs.

        Args:
            x: Input array.
            axis: Axis along which to compute the maximum.
            keepdim: Whether to keep reduced dimensions.

        Returns:
            NDArray: Maximum value ignoring NaN.
        """
        return np.nanmax(x, axis=axis, keepdims=keepdim)

    def sort(self, x: ArrayLike, axis: int = -1) -> NDArray:
        """Return a sorted copy of x.

        Args:
            x: Input array.
            axis: Axis along which to sort.

        Returns:
            NDArray: Sorted array.
        """
        return np.sort(x, axis=axis)

    def isclose(
        self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8
    ) -> NDArray:
        """Return a boolean array where elements are close.

        Args:
            a: First input.
            b: Second input.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            NDArray: Boolean array.
        """
        return np.isclose(a, b, rtol=rtol, atol=atol)

    def radians(self, x: ArrayLike) -> NDArray:
        """Convert angles from degrees to radians.

        Args:
            x: Angle in degrees.

        Returns:
            NDArray: Angle in radians.
        """
        return np.radians(x)

    def degrees(self, x: ArrayLike) -> NDArray:
        """Convert angles from radians to degrees.

        Args:
            x: Angle in radians.

        Returns:
            NDArray: Angle in degrees.
        """
        return np.degrees(x)
