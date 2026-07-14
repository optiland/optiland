"""
NumPy backend -- array creation, identity, and precision operations.

Provides CreationMixin, one of the mixins composed into
NumpyBackend (see optiland/backend/numpy_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray


class CreationMixin:
    """Array creation, identity, and precision operations."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "numpy"

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------

    @property
    def _dtype(self) -> type:
        """Return the NumPy dtype for the current precision."""
        return np.float32 if self._precision == "float32" else np.float64

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        """Set the floating-point precision.

        Args:
            precision: Either ``'float32'`` or ``'float64'``.
        """
        if precision not in ("float32", "float64"):
            raise ValueError("Precision must be 'float32' or 'float64'.")
        self._precision = precision

    def get_precision(self) -> int:
        """Return the current precision as an integer (32 or 64)."""
        return 32 if self._precision == "float32" else 64

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    def array(self, x: ArrayLike) -> NDArray:
        """Create a NumPy array cast to the current precision.

        Args:
            x: Input data.

        Returns:
            NDArray: NumPy array with dtype matching current precision.
        """
        return np.array(x, dtype=self._dtype)

    def zeros(self, shape: Sequence[int], dtype: Any = None) -> NDArray:
        """Return a zero array of given shape with current precision dtype.

        Args:
            shape: Shape of the output array.
            dtype: Optional dtype override.

        Returns:
            NDArray: Zero array.
        """
        return np.zeros(shape, dtype=dtype if dtype is not None else self._dtype)

    def ones(self, shape: Sequence[int], dtype: Any = None) -> NDArray:
        """Return an array of ones with current precision dtype.

        Args:
            shape: Shape of the output array.
            dtype: Optional dtype override.

        Returns:
            NDArray: Ones array.
        """
        return np.ones(shape, dtype=dtype if dtype is not None else self._dtype)

    def full(self, shape: Sequence[int], fill_value: Any, dtype: Any = None) -> NDArray:
        """Return a constant-filled array with current precision dtype.

        Args:
            shape: Shape of the output array.
            fill_value: Fill value.
            dtype: Optional dtype override.

        Returns:
            NDArray: Filled array.
        """
        _dtype = dtype if dtype is not None else self._dtype
        return np.full(shape, fill_value, dtype=_dtype)

    def linspace(self, start: float, stop: float, num: int = 50) -> NDArray:
        """Return evenly spaced numbers over an interval.

        Args:
            start: Start of the interval.
            stop: End of the interval.
            num: Number of samples.

        Returns:
            NDArray: Evenly spaced samples.
        """
        return np.linspace(start, stop, num, dtype=self._dtype)

    def arange(self, *args: Any, **kwargs: Any) -> NDArray:
        """Return evenly spaced values within a given interval.

        Args:
            *args: start, stop, step (same as np.arange).
            **kwargs: Additional keyword arguments passed to np.arange.

        Returns:
            NDArray: Array of evenly spaced values.
        """
        return np.arange(*args, **kwargs)

    def zeros_like(self, x: ArrayLike) -> NDArray:
        """Return a zero array with the same shape as x.

        Args:
            x: Reference array.

        Returns:
            NDArray: Zero array.
        """
        return np.zeros_like(x, dtype=self._dtype)

    def ones_like(self, x: ArrayLike) -> NDArray:
        """Return an array of ones with the same shape as x.

        Args:
            x: Reference array.

        Returns:
            NDArray: Ones array.
        """
        return np.ones_like(x, dtype=self._dtype)

    def full_like(self, x: ArrayLike, fill_value: Any) -> NDArray:
        """Return a full array with the same shape as x.

        Args:
            x: Reference array.
            fill_value: Fill value.

        Returns:
            NDArray: Filled array.
        """
        return np.full_like(x, fill_value, dtype=self._dtype)

    def empty(self, shape: Sequence[int]) -> NDArray:
        """Return an uninitialized array of the given shape.

        Args:
            shape: Shape of the output array.

        Returns:
            NDArray: Uninitialized array.
        """
        return np.empty(shape, dtype=self._dtype)

    def empty_like(self, x: ArrayLike) -> NDArray:
        """Return an uninitialized array with the same shape as x.

        Args:
            x: Reference array.

        Returns:
            NDArray: Uninitialized array.
        """
        return np.empty_like(x, dtype=self._dtype)

    def eye(self, n: int) -> NDArray:
        """Return a 2D identity matrix.

        Args:
            n: Size of the identity matrix.

        Returns:
            NDArray: Identity matrix.
        """
        return np.eye(n, dtype=self._dtype)

    def asarray(self, x: ArrayLike, **kwargs: Any) -> NDArray:
        """Convert x to a NumPy array without copying if possible.

        Args:
            x: Input data.
            **kwargs: Keyword arguments forwarded to ``np.asarray``
                (e.g. ``dtype``).

        Returns:
            NDArray: NumPy array view (or copy if necessary).
        """
        dtype = kwargs.pop("dtype", self._dtype)
        return np.asarray(x, dtype=dtype, **kwargs)
