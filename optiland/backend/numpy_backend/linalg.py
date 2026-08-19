"""
NumPy backend -- linear algebra operations.

Provides LinalgMixin, one of the mixins composed into
NumpyBackend (see optiland/backend/numpy_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class LinalgMixin:
    """Linear algebra operations."""

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def matmul(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        """Matrix product of two arrays.

        Args:
            a: First matrix.
            b: Second matrix.

        Returns:
            NDArray: Matrix product.
        """
        return np.matmul(a, b)

    def cross(
        self,
        a: ArrayLike,
        b: ArrayLike,
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: int | None = None,
    ) -> NDArray:
        """Return the cross product of two vectors.

        Args:
            a: First vector array.
            b: Second vector array.
            axisa: Axis of a that defines the vector(s).
            axisb: Axis of b that defines the vector(s).
            axisc: Axis of c that contains the cross product vector.
            axis: If defined, the axis of a, b and c that defines the vectors.

        Returns:
            NDArray: Cross product.
        """
        return np.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

    def batched_chain_matmul3(
        self, a: ArrayLike, b: ArrayLike, c: ArrayLike
    ) -> NDArray:
        """Compute a @ b @ c with promoted dtype.

        Args:
            a: First matrix.
            b: Second matrix.
            c: Third matrix.

        Returns:
            NDArray: Result of a @ b @ c.
        """
        dtype = np.result_type(a, b, c)
        return np.matmul(np.matmul(a.astype(dtype), b.astype(dtype)), c.astype(dtype))

    def matrix_vector_multiply_and_squeeze(
        self, p: NDArray, E: NDArray, backend: Literal["numpy"] = "numpy"
    ) -> NDArray:
        """Multiply p @ E[..., newaxis] and squeeze trailing dimension.

        Args:
            p: Matrix array.
            E: Vector array.
            backend: Unused; kept for backward compatibility.

        Returns:
            NDArray: Result with trailing dimension squeezed.
        """
        return np.squeeze(np.matmul(p, E[:, :, np.newaxis]), axis=2)

    def mult_p_E(self, p: NDArray, E: NDArray) -> NDArray:
        """Complex matrix-vector multiply used for polarized fields.

        Args:
            p: Jones matrix array.
            E: Electric field array.

        Returns:
            NDArray: Result of complex matrix-vector multiplication.
        """
        return np.squeeze(np.matmul(p, E[:, :, np.newaxis]), axis=2)

    def lstsq(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        """Compute the least-squares solution to a @ x = b.

        Args:
            a: Left-hand side matrix (M, N).
            b: Right-hand side matrix (M,) or (M, K).

        Returns:
            NDArray: Least-squares solution (N,) or (N, K).
        """
        return np.linalg.lstsq(a, b, rcond=None)[0]

    def to_complex(self, x: NDArray) -> NDArray:
        """Cast x to complex128.

        Args:
            x: Input array.

        Returns:
            NDArray: Complex128 array.
        """
        return x.astype(np.complex128) if np.isrealobj(x) else x
