"""
NumPy backend -- miscellaneous and NumPy-specific helper operations.

Provides MiscMixin, one of the mixins composed into
NumpyBackend (see optiland/backend/numpy_backend/__init__.py).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.path import Path
from scipy.spatial.transform import Rotation as R
from scipy.special import gamma

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from numpy.typing import ArrayLike, NDArray


class MiscMixin:
    """Miscellaneous and NumPy-specific helper operations."""

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def factorial(self, n: Any) -> NDArray:
        """Compute the factorial of n using the gamma function.

        Args:
            n: Non-negative integer or array of integers.

        Returns:
            NDArray: Factorial values.
        """
        return gamma(n + 1)

    def path_contains_points(self, vertices: NDArray, points: NDArray) -> NDArray:
        """Return a boolean mask of points inside the polygon.

        Args:
            vertices: Polygon vertices as (N, 2) array.
            points: Query points as (M, 2) array.

        Returns:
            NDArray: Boolean mask of shape (M,).
        """
        path = Path(vertices)
        mask = path.contains_points(points)
        return np.asarray(mask, dtype=bool)

    def pad(
        self,
        tensor: NDArray,
        pad_width: Any,
        mode: str = "constant",
        constant_values: float | None = 0,
    ) -> NDArray:
        """Pad an array.

        Args:
            tensor: Input array.
            pad_width: Number of values padded per axis.
            mode: Padding mode (only ``'constant'`` is supported).
            constant_values: Value used for constant padding.

        Returns:
            NDArray: Padded array.
        """
        if mode == "constant":
            return np.pad(tensor, pad_width, mode=mode, constant_values=constant_values)

        return np.pad(tensor, pad_width, mode=mode)

    def vectorize(self, pyfunc: Callable[..., Any]) -> Callable[..., Any]:
        """Vectorize a scalar Python function.

        Args:
            pyfunc: The scalar function to vectorize.

        Returns:
            Callable: Vectorized function.
        """
        return np.vectorize(pyfunc)

    @contextlib.contextmanager
    def errstate(self, **kwargs: Any) -> Generator[None, None, None]:  # type: ignore[override]
        """Context manager for NumPy floating-point error state.

        Args:
            **kwargs: Keyword arguments forwarded to ``np.errstate``.

        Yields:
            None
        """
        with np.errstate(**kwargs):
            yield

    def histogram(self, x: ArrayLike, bins: Any = 10) -> tuple[NDArray, NDArray]:
        """Compute a histogram of x.

        Args:
            x: Input data.
            bins: Number of bins or bin edges.

        Returns:
            tuple[NDArray, NDArray]: Bin counts and bin edges.
        """
        return np.histogram(x, bins=bins)

    def histogram2d(
        self,
        x: ArrayLike,
        y: ArrayLike,
        bins: Any,
        weights: NDArray | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Compute a 2-D histogram.

        Args:
            x: x-coordinates of the sample points.
            y: y-coordinates of the sample points.
            bins: Bin specification (list of two edge arrays).
            weights: Optional weights for each sample.

        Returns:
            tuple[NDArray, NDArray, NDArray]: Histogram, x edges, y edges.
        """
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)
        return hist, xedges, yedges

    def copy_to(self, source: NDArray, destination: NDArray) -> None:
        """Copy source array into destination in-place.

        Args:
            source: Source array.
            destination: Destination array (modified in place).
        """
        np.copyto(destination, source)

    # ------------------------------------------------------------------
    # Numpy-specific helpers (not in ABC — kept for backward compatibility)
    # ------------------------------------------------------------------

    def from_matrix(self, matrix: NDArray) -> R:
        """Create a SciPy Rotation from a rotation matrix.

        Args:
            matrix: Rotation matrix.

        Returns:
            Rotation: SciPy Rotation object.
        """
        return R.from_matrix(matrix)

    def from_euler(self, euler: NDArray) -> R:
        """Create a SciPy Rotation from Euler angles.

        Args:
            euler: Euler angles in the 'xyz' convention.

        Returns:
            Rotation: SciPy Rotation object.
        """
        return R.from_euler("xyz", euler)
