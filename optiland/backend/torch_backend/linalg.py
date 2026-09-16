"""
PyTorch backend -- linear algebra operations.

Provides LinalgMixin, one of the mixins composed into
TorchBackend (see optiland/backend/torch_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
        if self._is_metal(a) or self._is_metal(b):
            # Promote the other operand exactly (``.to(float64)`` on a plain
            # mps tensor is rejected by torch); dispatch does the product.
            return torch.matmul(self.cast(a), self.cast(b))
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
        if self._is_metal(a) or self._is_metal(b) or self._is_metal(c):
            return torch.matmul(torch.matmul(self.cast(a), self.cast(b)), self.cast(c))
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
        if self._is_metal(p) or self._is_metal(E):
            # Emulated float64: complex128 does not exist on the Apple GPU, so
            # the product is formed on the CPU in complex128 (see to_complex).
            p_c = self._to_cpu_complex(p)
            E_c = self._to_cpu_complex(E)
            return torch.squeeze(torch.matmul(p_c, E_c.unsqueeze(2)), dim=2)
        cdtype = self._complex_dtype_for(p)
        p_c = p.to(cdtype)
        try:
            E_c = E.to(cdtype)
        except Exception:
            E_c = torch.tensor(E, device=self._device(), dtype=cdtype)
        return torch.squeeze(torch.matmul(p_c, E_c.unsqueeze(2)), dim=2)

    def _to_cpu_complex(self, x: Any) -> Tensor:
        """CPU complex128 copy of ``x`` (decoding a MetalFloat64 exactly)."""
        if isinstance(x, torch.Tensor):
            if x.device.type == "mps":
                x = x.to("cpu")  # MetalFloat64 -> real float64 via dispatch
            return x.to(torch.complex128)
        return torch.tensor(x, dtype=torch.complex128)

    def _complex_dtype_for(self, x: Tensor) -> torch.dtype:
        """Complex dtype for ``x``: complex128 except on MPS, which has none.

        On the Apple GPU native tensors are float32 and only complex64 exists;
        emulated float64 tensors (``MetalFloat64``) are declared complex128,
        which currently lives on the CPU (see ``to_complex``).
        """
        device = getattr(x, "device", None)
        if device is not None and device.type == "mps" and x.dtype == torch.float32:
            return torch.complex64
        return torch.complex128

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
        """Cast x to complex128 (complex64 for native float32 tensors on MPS).

        With emulated float64 on ``mps`` the result is a CPU complex128 tensor:
        the Apple GPU has no complex128, so complex data (polarization, thin
        films, FFT-based PSFs) currently lives on the CPU when emulated. The
        decode is exact; gradients do not flow back through it.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Complex tensor.
        """
        if not isinstance(x, torch.Tensor):
            x = self.array(x)
        if self._is_metal(x):
            return self._to_cpu_complex(x)
        return x.to(self._complex_dtype_for(x))
