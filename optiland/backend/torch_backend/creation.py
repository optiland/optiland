"""
PyTorch backend -- internal helpers and array creation operations.

Provides CreationMixin, one of the mixins composed into
TorchBackend (see optiland/backend/torch_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike
    from torch import Tensor


class CreationMixin:
    """Internal helpers and array creation operations."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dtype(self) -> torch.dtype:
        """Return the current torch dtype."""
        return self._config.get_precision()

    def _device(self) -> str:
        """Return the current device string."""
        return self._config.get_device()

    def _grad(self) -> bool:
        """Return whether gradients are enabled."""
        return self._config.grad_mode.requires_grad

    _NP_TO_TORCH: dict[Any, torch.dtype] = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
        np.int32: torch.int32,
        np.int64: torch.int64,
    }

    def _resolve_dtype(self, dtype: Any) -> torch.dtype:
        """Resolve a dtype argument to a torch.dtype.

        Accepts None (uses backend default), numpy dtypes, or torch dtypes.

        Args:
            dtype: Dtype to resolve.

        Returns:
            torch.dtype: Resolved torch dtype.
        """
        if dtype is None:
            return self._dtype()
        if isinstance(dtype, torch.dtype):
            return dtype
        return self._NP_TO_TORCH.get(dtype, self._dtype())

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    def array(self, x: ArrayLike) -> Tensor:
        """Create a tensor with current device, precision, and grad settings.

        Args:
            x: Input data.

        Returns:
            Tensor: Backend tensor.
        """
        if isinstance(x, torch.Tensor):
            return x

        if isinstance(x, list | tuple) and len(x) > 0:
            # Check if any element is a Tensor
            if any(isinstance(v, torch.Tensor) for v in x):
                # Ensure all are tensors and stack them to preserve gradients
                tensors = [
                    v
                    if isinstance(v, torch.Tensor)
                    else torch.tensor(v, device=self._device(), dtype=self._dtype())
                    for v in x
                ]
                # Normalize 0-d (scalar) tensors to 1-d to ensure consistent
                # shapes before stacking (e.g. mix of [] and [1] tensors)
                if len(set(t.shape for t in tensors)) > 1:
                    tensors = [t.unsqueeze(0) if t.dim() == 0 else t for t in tensors]
                try:
                    return torch.stack(tensors)
                except RuntimeError:
                    return torch.cat(tensors)
            elif isinstance(x[0], np.ndarray):
                x = np.array(x)

        return torch.tensor(
            x,
            device=self._device(),
            dtype=self._dtype(),
            requires_grad=self._grad(),
        )

    def zeros(self, shape: Sequence[int], dtype: Any = None) -> Tensor:
        """Return a zero tensor of given shape.

        Args:
            shape: Shape of the output tensor.
            dtype: Optional dtype override.

        Returns:
            Tensor: Zero tensor.
        """
        return torch.zeros(
            shape,
            device=self._device(),
            dtype=self._resolve_dtype(dtype),
            requires_grad=self._grad(),
        )

    def ones(self, shape: Sequence[int], dtype: Any = None) -> Tensor:
        """Return a ones tensor of given shape.

        Args:
            shape: Shape of the output tensor.
            dtype: Optional dtype override.

        Returns:
            Tensor: Ones tensor.
        """
        return torch.ones(
            shape,
            device=self._device(),
            dtype=self._resolve_dtype(dtype),
            requires_grad=self._grad(),
        )

    def full(self, shape: Sequence[int], fill_value: Any, dtype: Any = None) -> Tensor:
        """Return a constant-filled tensor of given shape.

        Args:
            shape: Shape of the output tensor.
            fill_value: Fill value.
            dtype: Optional dtype override.

        Returns:
            Tensor: Filled tensor.
        """
        val = fill_value.item() if isinstance(fill_value, torch.Tensor) else fill_value
        if not isinstance(shape, list | tuple):
            try:
                shape = (int(shape),)
            except Exception:
                shape = (shape,)
        return torch.full(
            shape,
            val,
            device=self._device(),
            dtype=self._resolve_dtype(dtype),
            requires_grad=self._grad(),
        )

    def linspace(self, start: float, stop: float, num: int = 50) -> Tensor:
        """Return evenly spaced numbers over an interval.

        Args:
            start: Start of the interval.
            stop: End of the interval.
            num: Number of samples.

        Returns:
            Tensor: Evenly spaced samples.
        """
        return torch.linspace(
            start,
            stop,
            num,
            device=self._device(),
            dtype=self._dtype(),
            requires_grad=self._grad(),
        )

    def arange(self, *args: Any, **kwargs: Any) -> Tensor:
        """Return evenly spaced values within a given interval.

        Args:
            *args: start, stop, step (positional).
            **kwargs: Keyword arguments passed to torch.arange.

        Returns:
            Tensor: Evenly spaced values.
        """
        if len(args) == 1:
            start, end, step = 0, args[0], 1
        elif len(args) == 2:
            start, end = args
            step = kwargs.pop("step", 1)
        elif len(args) == 3:
            start, end, step = args
        else:
            raise TypeError(
                f"arange expected 1, 2, or 3 positional arguments, got {len(args)}"
            )

        for val in (start, end, step):
            if isinstance(val, torch.Tensor):
                val = val.item()

        if isinstance(start, torch.Tensor):
            start = start.item()
        if isinstance(end, torch.Tensor):
            end = end.item()
        if isinstance(step, torch.Tensor):
            step = step.item()

        return torch.arange(
            start,
            end,
            step,
            device=self._device(),
            dtype=self._dtype(),
            requires_grad=self._grad(),
        )

    def zeros_like(self, x: Any) -> Tensor:
        """Return a zero tensor with the same shape as x.

        Args:
            x: Reference tensor.

        Returns:
            Tensor: Zero tensor.
        """
        return torch.zeros_like(
            self.array(x),
            device=self._device(),
            dtype=self._dtype(),
            requires_grad=self._grad(),
        )

    def ones_like(self, x: Any) -> Tensor:
        """Return a ones tensor with the same shape as x.

        Args:
            x: Reference tensor.

        Returns:
            Tensor: Ones tensor.
        """
        return torch.ones_like(
            self.array(x),
            device=self._device(),
            dtype=self._dtype(),
            requires_grad=self._grad(),
        )

    def full_like(self, x: Any, fill_value: Any) -> Tensor:
        """Return a full tensor with the same shape as x.

        Args:
            x: Reference tensor.
            fill_value: Fill value.

        Returns:
            Tensor: Filled tensor.
        """
        x_t = self.array(x)
        val = fill_value.item() if isinstance(fill_value, torch.Tensor) else fill_value
        return torch.full_like(
            x_t,
            val,
            device=self._device(),
            dtype=self._dtype(),
            requires_grad=self._grad(),
        )

    def empty(self, shape: Sequence[int]) -> Tensor:
        """Return an uninitialized tensor of given shape.

        Args:
            shape: Shape of the output tensor.

        Returns:
            Tensor: Uninitialized tensor.
        """
        return torch.empty(
            shape,
            device=self._device(),
            dtype=self._dtype(),
        )

    def empty_like(self, x: Any) -> Tensor:
        """Return an uninitialized tensor with the same shape as x.

        Args:
            x: Reference tensor.

        Returns:
            Tensor: Uninitialized tensor.
        """
        return torch.empty_like(
            self.array(x),
            device=self._device(),
            dtype=self._dtype(),
        )

    def eye(self, n: int) -> Tensor:
        """Return a 2D identity matrix.

        Args:
            n: Size of the matrix.

        Returns:
            Tensor: Identity matrix.
        """
        return torch.eye(n, device=self._device(), dtype=self._dtype())

    def asarray(self, x: Any, **kwargs: Any) -> Tensor:
        """Convert x to a tensor without copying if possible.

        Args:
            x: Input data.
            **kwargs: Keyword arguments forwarded to ``torch.as_tensor``
                (e.g. ``dtype``). NumPy dtypes are automatically converted
                to the equivalent torch dtype.

        Returns:
            Tensor: Backend tensor.
        """
        import numpy as _np

        _NP_TO_TORCH = {
            _np.float32: torch.float32,
            _np.float64: torch.float64,
            _np.int32: torch.int32,
            _np.int64: torch.int64,
            _np.complex64: torch.complex64,
            _np.complex128: torch.complex128,
            _np.bool_: torch.bool,
        }
        dtype = kwargs.pop("dtype", self._dtype())
        if isinstance(dtype, type) and dtype in _NP_TO_TORCH:
            dtype = _NP_TO_TORCH[dtype]
        elif hasattr(dtype, "type") and dtype.type in _NP_TO_TORCH:
            dtype = _NP_TO_TORCH[dtype.type]
        return torch.as_tensor(x, device=self._device(), dtype=dtype)

    def load(self, filename: str) -> Tensor:
        """Load a NumPy file and convert to a tensor.

        Args:
            filename: Path to a ``.npy`` file.

        Returns:
            Tensor: Loaded tensor.
        """
        data = np.load(filename)
        return self.array(data)
