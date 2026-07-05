"""
PyTorch backend -- linear algebra operations.

Provides LinalgMixin, one of the mixins composed into
TorchBackend (see optiland/backend/torch_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


class LinalgMixin:
    """Linear algebra operations."""

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Matrix product of two tensors with promoted dtype.

        Args:
            a: First matrix.
            b: Second matrix.

        Returns:
            Tensor: Matrix product.
        """
        dtype = torch.promote_types(a.dtype, b.dtype)
        return torch.matmul(a.to(dtype), b.to(dtype))

    def cross(
        self,
        a: Tensor,
        b: Tensor,
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: int | None = None,
    ) -> Tensor:
        """Return the cross product of two vectors.

        Args:
            a: First vector tensor.
            b: Second vector tensor.
            axisa: Axis of a defining the vector(s).
            axisb: Axis of b defining the vector(s).
            axisc: Axis of c containing the cross product.
            axis: If set, applies to axisa, axisb, and axisc.

        Returns:
            Tensor: Cross product.
        """
        if axis is not None:
            axisa = axisb = axisc = axis
        a_moved = torch.movedim(a, axisa, -1)
        b_moved = torch.movedim(b, axisb, -1)
        c = torch.linalg.cross(a_moved, b_moved, dim=-1)
        return torch.movedim(c, -1, axisc)

    def batched_chain_matmul3(self, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        """Compute a @ b @ c with promoted dtype.

        Args:
            a: First matrix.
            b: Second matrix.
            c: Third matrix.

        Returns:
            Tensor: Result of a @ b @ c.
        """
        dtype = torch.promote_types(torch.promote_types(a.dtype, b.dtype), c.dtype)
        return torch.matmul(torch.matmul(a.to(dtype), b.to(dtype)), c.to(dtype))

    def matrix_vector_multiply_and_squeeze(self, p: Tensor, E: Tensor) -> Tensor:
        """Multiply p @ E[..., newaxis] and squeeze trailing dimension.

        Args:
            p: Matrix tensor.
            E: Vector tensor.

        Returns:
            Tensor: Result with trailing dimension squeezed.
        """
        return torch.matmul(p, E.unsqueeze(2)).squeeze(2)

    def mult_p_E(self, p: Tensor, E: Tensor) -> Tensor:
        """Complex matrix-vector multiply for polarized fields.

        Args:
            p: Jones matrix tensor.
            E: Electric field tensor.

        Returns:
            Tensor: Complex matrix-vector product.
        """
        p_c = p.to(torch.complex128)
        try:
            E_c = E.to(torch.complex128)
        except Exception:
            E_c = torch.tensor(E, device=self._device(), dtype=torch.complex128)
        return torch.squeeze(torch.matmul(p_c, E_c.unsqueeze(2)), dim=2)

    def lstsq(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute the least-squares solution to a @ x = b.

        Args:
            a: Left-hand side matrix (M, N).
            b: Right-hand side matrix (M,) or (M, K).

        Returns:
            Tensor: Least-squares solution.
        """
        return torch.linalg.lstsq(a, b).solution

    def to_complex(self, x: Tensor) -> Tensor:
        """Cast x to complex128.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Complex128 tensor.
        """
        return x.to(torch.complex128)
