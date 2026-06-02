"""
PyTorch backend — implements AbstractBackend using PyTorch.

Kramer Harrison, 2025
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from optiland.backend.base import AbstractBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from numpy.typing import ArrayLike
    from torch import Generator as TorchGenerator
    from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration helpers (private to this module)
# ---------------------------------------------------------------------------


class GradMode:
    """Control global gradient computation for the torch backend."""

    def __init__(self) -> None:
        self.requires_grad: bool = False

    def enable(self) -> None:
        """Enable gradient computation."""
        self.requires_grad = True

    def disable(self) -> None:
        """Disable gradient computation."""
        self.requires_grad = False

    @contextlib.contextmanager
    def temporary_enable(self) -> Generator[None, None, None]:
        """Context manager that temporarily enables gradient computation."""
        old = self.requires_grad
        self.requires_grad = True
        try:
            yield
        finally:
            self.requires_grad = old


class _Config:
    """Internal configuration container for TorchBackend."""

    def __init__(self) -> None:
        self.device: Literal["cpu", "cuda"] = "cpu"
        self.precision: torch.dtype = torch.float32
        self.grad_mode: GradMode = GradMode()

    def set_device(self, device: Literal["cpu", "cuda"]) -> None:
        """Set the compute device.

        Args:
            device: ``'cpu'`` or ``'cuda'``.

        Raises:
            ValueError: If device is not ``'cpu'`` or ``'cuda'``, or if CUDA
                is requested but unavailable.
        """
        if device not in ("cpu", "cuda"):
            raise ValueError("Device must be 'cpu' or 'cuda'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        self.device = device

    def get_device(self) -> Literal["cpu", "cuda"]:
        """Return the current device."""
        return self.device

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        """Set the floating-point precision.

        Args:
            precision: ``'float32'`` or ``'float64'``.

        Raises:
            ValueError: If precision is not valid.
        """
        if precision == "float32":
            self.precision = torch.float32
        elif precision == "float64":
            self.precision = torch.float64
        else:
            raise ValueError("Precision must be 'float32' or 'float64'.")

    def get_precision(self) -> torch.dtype:
        """Return the current torch dtype."""
        return self.precision


class TorchBackend(AbstractBackend):
    """Backend implementation using PyTorch.

    Attributes:
        _lib: The torch module (used by passthrough methods).
        _config: Internal configuration (device, precision, grad_mode).
    """

    _lib = torch

    def __init__(self) -> None:
        self._config = _Config()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "torch"

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_gradients(self) -> bool:
        """Return True — PyTorch supports automatic differentiation."""
        return True

    @property
    def supports_gpu(self) -> bool:
        """Return True if CUDA is available."""
        return torch.cuda.is_available()

    # ------------------------------------------------------------------
    # Capability-gated overrides (torch has real implementations)
    # ------------------------------------------------------------------

    @property
    def grad_mode(self) -> GradMode:
        """Return the GradMode controller."""
        return self._config.grad_mode

    @property
    def autograd(self) -> Any:
        """Return the torch.autograd submodule."""
        return torch.autograd

    @property
    def nn(self) -> Any:
        """Return the torch.nn submodule."""
        return torch.nn

    def set_device(self, device: str) -> None:
        """Set the compute device.

        Args:
            device: ``'cpu'`` or ``'cuda'``.
        """
        self._config.set_device(device)  # type: ignore[arg-type]

    def get_device(self) -> str:
        """Return the current compute device."""
        return self._config.get_device()

    def get_complex_precision(self) -> torch.dtype:
        """Return the complex dtype matching the current float precision.

        Returns:
            torch.dtype: ``torch.complex64`` or ``torch.complex128``.

        Raises:
            ValueError: If the current precision is unsupported.
        """
        prec = self._config.get_precision()
        if prec == torch.float32:
            return torch.complex64
        elif prec == torch.float64:
            return torch.complex128
        else:
            raise ValueError("Unsupported precision for complex dtype.")

    def tensor(self, data: Any, **kwargs: Any) -> Tensor:
        """Create a tensor from data with full kwargs support.

        Args:
            data: Input data (scalar, list, numpy array, etc.).
            **kwargs: Forwarded to ``torch.tensor`` (e.g. ``requires_grad``,
                ``dtype``, ``device``).

        Returns:
            Tensor: New tensor.
        """
        kwargs.setdefault("device", self._device())
        kwargs.setdefault("dtype", self._dtype())
        return torch.tensor(data, **kwargs)

    def copy_to(self, source: Tensor, destination: Tensor) -> None:
        """In-place copy from source to destination tensor.

        Safely handles tensors that require gradients.

        Args:
            source: Source tensor.
            destination: Destination tensor (modified in place).
        """
        if destination.requires_grad:
            destination.data.copy_(source)
        else:
            destination.copy_(source)

    def to_tensor(
        self,
        data: ArrayLike,
        device: str | torch.device | None = None,
    ) -> Tensor:
        """Convert data to a PyTorch tensor with the backend's precision.

        Args:
            data: The data to convert.
            device: Optional device override.

        Returns:
            Tensor: Converted tensor.
        """
        current_device = device or self._config.get_device()
        current_precision = self._config.get_precision()
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, device=current_device, dtype=current_precision)
        return data.to(device=current_device, dtype=current_precision)

    def get_bilinear_weights(
        self, coords: Tensor, bin_edges: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Compute differentiable bilinear interpolation weights.

        Args:
            coords: Ray coordinates tensor of shape (N, 2).
            bin_edges: Sequence of two edge tensors [x_edges, y_edges].

        Returns:
            tuple[Tensor, Tensor]: (all_indices, all_weights).
        """
        x_edges, y_edges = bin_edges
        x = coords[:, 0].contiguous()
        y = coords[:, 1].contiguous()

        valid_mask = (
            (x >= x_edges[0])
            & (x <= x_edges[-1])
            & (y >= y_edges[0])
            & (y <= y_edges[-1])
        )

        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        ix = torch.searchsorted(x_centers, x, right=True) - 1
        iy = torch.searchsorted(y_centers, y, right=True) - 1
        ix = torch.clamp(ix, 0, len(x_centers) - 2)
        iy = torch.clamp(iy, 0, len(y_centers) - 2)

        x0, x1 = x_centers[ix], x_centers[ix + 1]
        y0, y1 = y_centers[iy], y_centers[iy + 1]

        wx = (x - x0) / (x1 - x0 + 1e-9)
        wy = (y - y0) / (y1 - y0 + 1e-9)

        w00 = (1 - wx) * (1 - wy)
        w01 = (1 - wx) * wy
        w10 = wx * (1 - wy)
        w11 = wx * wy

        all_indices = torch.stack(
            [
                torch.stack([ix, iy], dim=1),
                torch.stack([ix, iy + 1], dim=1),
                torch.stack([ix + 1, iy], dim=1),
                torch.stack([ix + 1, iy + 1], dim=1),
            ],
            dim=1,
        )
        all_weights = torch.stack([w00, w01, w10, w11], dim=1)
        all_weights = all_weights * valid_mask.unsqueeze(1).to(all_weights.dtype)
        return all_indices, all_weights

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        """Set the floating-point precision.

        Args:
            precision: ``'float32'`` or ``'float64'``.
        """
        self._config.set_precision(precision)

    def get_precision(self) -> int:
        """Return the current precision as an integer (32 or 64)."""
        return 32 if self._config.get_precision() == torch.float32 else 64

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

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self, x: Any, axis: int | None = None) -> Tensor:
        """Sum tensor elements over a given axis.

        Args:
            x: Input tensor.
            axis: Dimension along which to sum.

        Returns:
            Tensor: Sum.
        """
        return torch.sum(x, dim=axis) if axis is not None else torch.sum(x)

    def mean(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Compute the arithmetic mean, ignoring NaNs.

        Args:
            x: Input tensor.
            axis: Dimension along which to compute the mean.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            Tensor: Mean.
        """
        x = self.array(x)
        mask = ~torch.isnan(x)
        cnt = mask.sum(dim=axis, keepdim=keepdims).to(x.dtype)
        x0 = torch.where(mask, x, torch.tensor(0.0, dtype=x.dtype, device=x.device))
        s = x0.sum(dim=axis, keepdim=keepdims)
        return torch.where(
            cnt > 0,
            s / cnt,
            torch.tensor(float("nan"), dtype=x.dtype, device=x.device),
        )

    def std(self, x: Any, axis: int | None = None) -> Tensor:
        """Compute the standard deviation along an axis.

        Args:
            x: Input tensor.
            axis: Dimension along which to compute std.

        Returns:
            Tensor: Standard deviation.
        """
        return torch.std(x, dim=axis) if axis is not None else torch.std(x)

    def max(self, x: Any) -> Any:
        """Return the maximum value of x.

        Args:
            x: Input tensor or array.

        Returns:
            float: Maximum value as a Python scalar.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().max().item()
        return np.max(x)

    def min(self, x: Any) -> Any:
        """Return the minimum value of x.

        Args:
            x: Input tensor or array.

        Returns:
            float: Minimum value as a Python scalar.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().min().item()
        return np.min(x)

    def argmin(self, x: Any, axis: int | None = None) -> Tensor:
        """Return indices of the minimum values along a dimension.

        Args:
            x: Input tensor.
            axis: Dimension along which to find the minimum.

        Returns:
            Tensor: Index tensor.
        """
        return torch.argmin(x, dim=axis)

    def argwhere(self, x: Any) -> Tensor:
        """Return indices of non-zero elements.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Index tensor of shape (N, ndim).
        """
        return torch.nonzero(x, as_tuple=False)

    def clip(self, x: Any, a_min: Any, a_max: Any) -> Tensor:
        """Clip values in x to [a_min, a_max].

        Args:
            x: Input tensor.
            a_min: Minimum value.
            a_max: Maximum value.

        Returns:
            Tensor: Clipped tensor.
        """
        return torch.clamp(x, a_min, a_max)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Return elements from x or y depending on condition.

        Args:
            condition: Boolean tensor or bool.
            x: Values where condition is True.
            y: Values where condition is False.

        Returns:
            Tensor: Output tensor.
        """
        if isinstance(condition, bool):
            return x if condition else y
        return torch.where(condition, x, y)

    def maximum(self, a: Any, b: Any) -> Tensor:
        """Element-wise maximum of a and b.

        Args:
            a: First input tensor.
            b: Second input tensor.

        Returns:
            Tensor: Element-wise maximum.
        """
        return torch.maximum(self.array(a), self.array(b))

    def minimum(self, a: Any, b: Any) -> Tensor:
        """Element-wise minimum of a and b.

        Args:
            a: First input tensor.
            b: Second input tensor.

        Returns:
            Tensor: Element-wise minimum.
        """
        return torch.minimum(self.array(a), self.array(b))

    def fmax(self, a: Any, b: Any) -> Tensor:
        """Element-wise maximum, ignoring NaNs.

        Args:
            a: First input tensor.
            b: Second input tensor.

        Returns:
            Tensor: Element-wise maximum ignoring NaN.
        """
        return torch.fmax(self.array(a), self.array(b))

    def power(self, x: Any, y: Any) -> Tensor:
        """Return x raised to the power y.

        Args:
            x: Base tensor.
            y: Exponent tensor.

        Returns:
            Tensor: x ** y.
        """
        return torch.pow(self.array(x), self.array(y))

    def diff(self, x: Any, n: int = 1, axis: int = -1, **kwargs: Any) -> Tensor:
        """Calculate the n-th discrete difference along a dimension.

        Args:
            x: Input tensor.
            n: Number of times to apply the difference.
            axis: Dimension along which to compute differences.
            **kwargs: Additional keyword arguments forwarded to ``torch.diff``
                (e.g. ``prepend``, ``append``).

        Returns:
            Tensor: Differences tensor.
        """
        return torch.diff(x, n=n, dim=axis, **kwargs)

    def all(self, x: Any) -> bool:
        """Return True if all elements of x are True.

        Args:
            x: Input tensor or bool.

        Returns:
            bool: Whether all elements are True.
        """
        if isinstance(x, bool):
            return x
        t = torch.as_tensor(x, dtype=self._dtype(), device=self._device())
        return bool(torch.all(t).item())

    def any(self, x: Any) -> bool:
        """Return True if any element of x is True.

        Args:
            x: Input tensor or bool.

        Returns:
            bool: Whether any element is True.
        """
        if isinstance(x, bool):
            return x
        t = torch.as_tensor(x, dtype=self._dtype(), device=self._device())
        return bool(torch.any(t).item())

    def nanmax(
        self, x: Tensor, axis: int | None = None, keepdim: bool = False
    ) -> Tensor:
        """Return the maximum value, ignoring NaNs.

        Args:
            x: Input tensor.
            axis: Dimension along which to compute the maximum.
            keepdim: Whether to keep reduced dimensions.

        Returns:
            Tensor: Maximum value ignoring NaN.
        """
        nan_mask = torch.isnan(x)
        replaced = x.clone()
        replaced[nan_mask] = float("-inf")
        if axis is not None:
            result, _ = torch.max(replaced, dim=axis, keepdim=keepdim)
        else:
            result = torch.max(replaced)
        return result

    def sort(self, x: Any, axis: int = -1) -> Tensor:
        """Return a sorted tensor along the given dimension.

        Args:
            x: Input tensor.
            axis: Dimension along which to sort.

        Returns:
            Tensor: Sorted tensor (values only, not the indices).
        """
        return torch.sort(x, dim=axis).values

    def isclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> Tensor:
        """Return a boolean tensor where elements are close.

        Args:
            a: First input.
            b: Second input.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            Tensor: Boolean tensor.
        """
        return torch.isclose(self.array(a), self.array(b), rtol=rtol, atol=atol)

    def radians(self, x: Any) -> Tensor:
        """Convert angles from degrees to radians.

        Args:
            x: Angle in degrees.

        Returns:
            Tensor: Angle in radians.
        """
        return torch.deg2rad(self.array(x))

    def degrees(self, x: Any) -> Tensor:
        """Convert angles from radians to degrees.

        Args:
            x: Angle in radians.

        Returns:
            Tensor: Angle in degrees.
        """
        return torch.rad2deg(self.array(x))

    # ------------------------------------------------------------------
    # Passthrough overrides — explicit implementations with array() cast
    # These override the passthrough methods inherited from AbstractBackend
    # to ensure Python scalars and lists are accepted (via self.array()).
    # ------------------------------------------------------------------

    def tan(self, x: Any) -> Tensor:
        """Compute the tangent of x (element-wise).

        Args:
            x: Input data.

        Returns:
            Tensor: Tangent values.
        """
        return torch.tan(self.array(x))

    def arcsin(self, x: Any) -> Tensor:
        """Compute the arcsine of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Arcsine values.
        """
        return torch.arcsin(self.array(x))

    def arccos(self, x: Any) -> Tensor:
        """Compute the arccosine of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Arccosine values.
        """
        return torch.arccos(self.array(x))

    def arctan(self, x: Any) -> Tensor:
        """Compute the arctangent of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Arctangent values.
        """
        return torch.arctan(self.array(x))

    def arctan2(self, y: Any, x: Any) -> Tensor:
        """Compute the element-wise arctan2.

        Args:
            y: y-coordinates.
            x: x-coordinates.

        Returns:
            Tensor: Arctangent values (angle in radians).
        """
        return torch.arctan2(self.array(y), self.array(x))

    def sinh(self, x: Any) -> Tensor:
        """Compute the hyperbolic sine of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Hyperbolic sine values.
        """
        return torch.sinh(self.array(x))

    def cosh(self, x: Any) -> Tensor:
        """Compute the hyperbolic cosine of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Hyperbolic cosine values.
        """
        return torch.cosh(self.array(x))

    def tanh(self, x: Any) -> Tensor:
        """Compute the hyperbolic tangent of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Hyperbolic tangent values.
        """
        return torch.tanh(self.array(x))

    def log(self, x: Any) -> Tensor:
        """Compute the natural logarithm of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Natural log values.
        """
        return torch.log(self.array(x))

    def log10(self, x: Any) -> Tensor:
        """Compute the base-10 logarithm of x.

        Args:
            x: Input data.

        Returns:
            Tensor: log10 values.
        """
        return torch.log10(self.array(x))

    def sign(self, x: Any) -> Tensor:
        """Compute the sign of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Sign values (-1, 0, or 1).
        """
        return torch.sign(self.array(x))

    def floor(self, x: Any) -> Tensor:
        """Round down to the nearest integer.

        Args:
            x: Input data.

        Returns:
            Tensor: Floor values.
        """
        return torch.floor(self.array(x))

    def ceil(self, x: Any) -> Tensor:
        """Round up to the nearest integer.

        Args:
            x: Input data.

        Returns:
            Tensor: Ceiling values.
        """
        return torch.ceil(self.array(x))

    def hypot(self, x: Any, y: Any) -> Tensor:
        """Compute the hypotenuse given legs x and y.

        Args:
            x: First leg.
            y: Second leg.

        Returns:
            Tensor: Hypotenuse values.
        """
        return torch.hypot(self.array(x), self.array(y))

    def deg2rad(self, x: Any) -> Tensor:
        """Convert angles from degrees to radians.

        Args:
            x: Angle in degrees.

        Returns:
            Tensor: Angle in radians.
        """
        return torch.deg2rad(self.array(x))

    def rad2deg(self, x: Any) -> Tensor:
        """Convert angles from radians to degrees.

        Args:
            x: Angle in radians.

        Returns:
            Tensor: Angle in degrees.
        """
        return torch.rad2deg(self.array(x))

    def conj(self, x: Any) -> Tensor:
        """Compute the complex conjugate of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Complex conjugate.
        """
        return torch.conj(self.array(x))

    def real(self, x: Any) -> Tensor:
        """Return the real part of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Real part.
        """
        return torch.real(self.array(x))

    def imag(self, x: Any) -> Tensor:
        """Return the imaginary part of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Imaginary part.
        """
        return torch.imag(self.array(x))

    def allclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Return True if all elements in a and b are close.

        Args:
            a: First input.
            b: Second input.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            bool: Whether all elements are close.
        """
        return bool(torch.allclose(self.array(a), self.array(b), rtol=rtol, atol=atol))

    def copysign(self, x: Any, y: Any) -> Tensor:
        """Return x with the sign of y (element-wise).

        Args:
            x: Magnitude array.
            y: Sign array.

        Returns:
            Tensor: x with sign from y.
        """
        return torch.copysign(self.array(x), self.array(y))

    def sin(self, x: Any) -> Tensor:
        """Compute the sine of x (element-wise).

        Args:
            x: Input data.

        Returns:
            Tensor: Sine values.
        """
        return torch.sin(self.array(x))

    def cos(self, x: Any) -> Tensor:
        """Compute the cosine of x (element-wise).

        Args:
            x: Input data.

        Returns:
            Tensor: Cosine values.
        """
        return torch.cos(self.array(x))

    def sqrt(self, x: Any) -> Tensor:
        """Compute the square root of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Square root values.
        """
        return torch.sqrt(self.array(x))

    def exp(self, x: Any) -> Tensor:
        """Compute the exponential of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Exponential values.
        """
        return torch.exp(self.array(x))

    def abs(self, x: Any) -> Tensor:
        """Compute the absolute value of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Absolute values.
        """
        return torch.abs(self.array(x))

    def log2(self, x: Any) -> Tensor:
        """Compute the base-2 logarithm of x.

        Args:
            x: Input data.

        Returns:
            Tensor: log2 values.
        """
        return torch.log2(self.array(x))

    def isinf(self, x: Any) -> Any:
        """Check if input is infinity, accepting scalars and tensors.

        Args:
            x: Input (scalar, ndarray, or Tensor).

        Returns:
            bool or Tensor: Whether x is infinite.
        """
        if isinstance(x, torch.Tensor):
            return torch.isinf(x)
        return torch.isinf(torch.tensor(x, dtype=self._dtype()))

    def isnan(self, x: Any) -> Any:
        """Check if input is NaN, accepting scalars and tensors.

        Args:
            x: Input (scalar, ndarray, or Tensor).

        Returns:
            bool or Tensor: Whether x is NaN.
        """
        if isinstance(x, torch.Tensor):
            return torch.isnan(x)
        return torch.isnan(torch.tensor(x, dtype=self._dtype()))

    def isfinite(self, x: Any) -> Any:
        """Check if input is finite, accepting scalars and tensors.

        Args:
            x: Input (scalar, ndarray, or Tensor).

        Returns:
            bool or Tensor: Whether x is finite.
        """
        if isinstance(x, torch.Tensor):
            return torch.isfinite(x)
        return torch.isfinite(torch.tensor(x, dtype=self._dtype()))

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

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interp(self, x: Any, xp: Any, fp: Any) -> Tensor:
        """1-D linear interpolation.

        Args:
            x: x-coordinates of the interpolated values.
            xp: x-coordinates of the data points.
            fp: y-coordinates of the data points.

        Returns:
            Tensor: Interpolated values.
        """
        x = torch.as_tensor(x, dtype=self._dtype(), device=self._device())
        xp = torch.as_tensor(xp, dtype=self._dtype(), device=self._device())
        fp = torch.as_tensor(fp, dtype=self._dtype(), device=self._device())
        sorted_indices = torch.argsort(xp)
        xp = xp[sorted_indices]
        fp = fp[sorted_indices]
        x_clipped = torch.clip(x, xp[0], xp[-1])
        indices = torch.searchsorted(xp, x_clipped, right=True)
        indices = torch.clamp(indices, 1, len(xp) - 1)
        x0 = xp[indices - 1]
        x1 = xp[indices]
        y0 = fp[indices - 1]
        y1 = fp[indices]
        return y0 + (y1 - y0) * (x_clipped - x0) / (x1 - x0)

    def nearest_nd_interpolator(
        self, points: Tensor, values: Tensor, Hx: Tensor, Hy: Tensor
    ) -> Tensor:
        """Nearest-neighbour interpolation on an N-D dataset.

        Args:
            points: Known sample points of shape (N, D).
            values: Values at the sample points.
            Hx: Query x coordinates.
            Hy: Query y coordinates.

        Returns:
            Tensor: Interpolated values.

        Raises:
            ValueError: If Hx or Hy is None.
        """
        if Hx is None or Hy is None:
            raise ValueError("Hx and Hy must be provided")
        Hx, Hy = self.array(Hx), self.array(Hy)
        Hx, Hy = torch.broadcast_tensors(Hx, Hy)
        q_flat = torch.stack([Hx, Hy], dim=-1).reshape(-1, 2)
        d = torch.cdist(q_flat, points.to(dtype=q_flat.dtype, device=q_flat.device))
        idx = d.argmin(dim=1)
        vals = values.view(points.shape[0], -1)
        out = vals[idx].view(*Hx.shape, -1)
        return out.squeeze(-1) if out.shape[-1] == 1 else out

    def grid_sample(
        self,
        input: Tensor,
        grid: Tensor,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ) -> Tensor:
        """Sample input using torch.nn.functional.grid_sample.

        Args:
            input: Input tensor of shape (N, C, H_in, W_in).
            grid: Grid tensor of shape (N, H_out, W_out, 2).
            mode: Interpolation mode.
            padding_mode: Padding mode.
            align_corners: Whether to align corners.

        Returns:
            Tensor: Output tensor of shape (N, C, H_out, W_out).
        """
        return F.grid_sample(
            input,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    # ------------------------------------------------------------------
    # Polynomial
    # ------------------------------------------------------------------

    def polyfit(self, x: Tensor, y: Tensor, degree: int) -> Tensor:
        """Least-squares polynomial fit.

        Args:
            x: x-coordinates of the sample points.
            y: y-coordinates of the sample points.
            degree: Degree of the polynomial.

        Returns:
            Tensor: Polynomial coefficients, highest power first.
        """
        X = torch.stack([x**i for i in range(degree, -1, -1)], dim=1)
        result = torch.linalg.lstsq(X, y.unsqueeze(1))
        coeffs = result.solution
        return coeffs[: degree + 1].squeeze()

    def polyval(self, coeffs: Any, x: Any) -> Any:
        """Evaluate a polynomial at specific values.

        Args:
            coeffs: Polynomial coefficients, highest power first.
            x: Values at which to evaluate.

        Returns:
            Tensor or float: Evaluated polynomial.
        """
        return sum(c * x**i for i, c in enumerate(reversed(coeffs)))

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def fftconvolve(
        self, in1: Tensor, in2: Tensor, mode: Literal["full", "valid", "same"] = "full"
    ) -> Tensor:
        """FFT-based convolution using PyTorch.

        Args:
            in1: First input tensor (N-D).
            in2: Second input tensor (N-D).
            mode: Convolution mode (``'full'``, ``'valid'``, ``'same'``).

        Returns:
            Tensor: Convolved tensor.

        Raises:
            ValueError: If inputs have different dimensionality or mode is
                unknown.
        """
        in1 = self.array(in1)
        in2 = self.array(in2)

        ndim = in1.ndim
        if in2.ndim != ndim:
            raise ValueError("Inputs must have the same dimensionality.")

        axes = tuple(range(ndim)) if ndim < 2 else (-2, -1)

        s1 = in1.shape
        s2 = in2.shape

        fft_shape = list(in1.shape)
        for axis in axes:
            fft_shape[axis] = s1[axis] + s2[axis] - 1

        IN1 = torch.fft.fftn(in1, s=[fft_shape[axis] for axis in axes], dim=axes)
        IN2 = torch.fft.fftn(in2, s=[fft_shape[axis] for axis in axes], dim=axes)
        ret = torch.fft.ifftn(
            IN1 * IN2, s=[fft_shape[axis] for axis in axes], dim=axes
        ).real

        if mode == "full":
            return ret

        crop_slices = [slice(None)] * ndim

        if mode == "same":
            for axis in axes:
                start = (s2[axis] - 1) // 2
                end = start + s1[axis]
                crop_slices[axis] = slice(start, end)
            return ret[tuple(crop_slices)]

        if mode == "valid":
            for axis in axes:
                start = s2[axis] - 1
                end = s1[axis]
                crop_slices[axis] = slice(start, end)
            return ret[tuple(crop_slices)]

        raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    # Random number generation
    # ------------------------------------------------------------------

    def default_rng(self, seed: int | None = None) -> TorchGenerator:
        """Return a PyTorch random number generator.

        Args:
            seed: Optional seed.

        Returns:
            Generator: PyTorch Generator.
        """
        if seed is None:
            seed = torch.initial_seed()
        return torch.Generator(device=self._device()).manual_seed(seed)

    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Any = None,
        generator: TorchGenerator | None = None,
    ) -> Tensor:
        """Uniform random samples in [low, high).

        Args:
            low: Lower boundary.
            high: Upper boundary.
            size: Output shape.
            generator: Optional torch Generator.

        Returns:
            Tensor: Uniform random samples.
        """
        size = size or 1
        gen_args = {"generator": generator} if generator else {}
        return torch.empty(size, device=self._device(), dtype=self._dtype()).uniform_(
            low, high, **gen_args
        )

    def rand(self, *size: int) -> Tensor:
        """Random values from a uniform distribution on [0, 1).

        Args:
            *size: Shape of the output tensor.

        Returns:
            Tensor: Random values.
        """
        if not size:
            size = (1,)
        return torch.rand(
            size,
            device=self._device(),
            dtype=self._dtype(),
            requires_grad=self._grad(),
        )

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Any = None,
        generator: TorchGenerator | None = None,
    ) -> Tensor:
        """Random samples from a Gaussian distribution.

        Args:
            loc: Mean of the distribution.
            scale: Standard deviation.
            size: Output shape.
            generator: Optional torch Generator.

        Returns:
            Tensor: Normal random samples.
        """
        size = size or (1,)
        gen_args = {"generator": generator} if generator else {}
        return (
            torch.randn(size, device=self._device(), dtype=self._dtype(), **gen_args)
            * scale
            + loc
        )

    def sobol_sampler(
        self,
        dim: int,
        num_samples: int,
        scramble: bool = True,
        seed: int | None = None,
    ) -> Tensor:
        """Generate quasi-random samples using Sobol sequences.

        Args:
            dim: Dimension of the samples.
            num_samples: Number of samples to generate.
            scramble: Whether to scramble the sequence.
            seed: Random seed for scrambling.

        Returns:
            Tensor: Samples of shape (num_samples_pow2, dim).
        """
        if num_samples > 0:
            num_samples_pow2 = 1 << (num_samples - 1).bit_length()
        else:
            num_samples_pow2 = num_samples
        sobol_engine = torch.quasirandom.SobolEngine(
            dimension=dim, scramble=scramble, seed=seed
        )
        samples = sobol_engine.draw(num_samples_pow2)
        return samples[:num_samples].to(device=self._device(), dtype=self._dtype())

    def erfinv(self, x: Any) -> Tensor:
        """Inverse error function.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Inverse error function of x.
        """
        return torch.erfinv(self.array(x))

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def factorial(self, n: Any) -> Tensor:
        """Compute the factorial of n using the log-gamma function.

        Args:
            n: Non-negative integer or tensor.

        Returns:
            Tensor: Factorial values.
        """
        return torch.lgamma(self.array(n + 1)).exp()

    def path_contains_points(self, vertices: Tensor, points: Tensor) -> Tensor:
        """Return a boolean mask of points inside the polygon.

        Uses a vectorized ray-crossing algorithm.

        Args:
            vertices: Polygon vertices as (N, 2) tensor (closed implicitly).
            points: Query points as (M, 2) tensor.

        Returns:
            Tensor: Boolean mask of shape (M,).
        """
        vx, vy = vertices[:, 0], vertices[:, 1]
        px = points[:, 0].unsqueeze(1)
        py = points[:, 1].unsqueeze(1)

        vx_next = torch.roll(vx, -1)
        vy_next = torch.roll(vy, -1)

        cond = (vy > py) != (vy_next > py)
        slope = (vx_next - vx) / (vy_next - vy)
        x_int = vx + slope * (py - vy)
        cross = cond & (px < x_int)

        inside = torch.sum(cross, dim=1) % 2 == 1
        return inside

    def pad(
        self,
        tensor: Tensor,
        pad_width: Any,
        mode: str = "constant",
        constant_values: float | None = 0,
    ) -> Tensor:
        """Pad a tensor.

        Args:
            tensor: Input tensor.
            pad_width: Padding per axis.
            mode: Padding mode.
            constant_values: Value used for constant padding.

        Returns:
            Tensor: Padded tensor.
        """
        if len(pad_width) == 2:
            (pt, pb), (pl, pr) = pad_width
        elif len(pad_width) == 4:
            if pad_width[0] != (0, 0) or pad_width[1] != (0, 0):
                raise NotImplementedError(
                    "Padding batch or channel dimensions is not supported"
                )
            (pt, pb), (pl, pr) = pad_width[2:]
        else:
            raise ValueError("pad_width must have length 2 or 4")

        if mode == "constant":
            return F.pad(
                tensor, (pl, pr, pt, pb), mode="constant", value=constant_values
            )

        if mode in ("reflect", "replicate", "circular"):
            return F.pad(tensor, (pl, pr, pt, pb), mode=mode)

        raise NotImplementedError(f"Unsupported padding mode: {mode}")

    def vectorize(self, pyfunc: Callable[..., Any]) -> Callable[..., Any]:
        """Vectorize a scalar Python function over tensor inputs.

        Args:
            pyfunc: The scalar function to vectorize.

        Returns:
            Callable: Vectorized function.
        """

        def wrapped(x: Tensor) -> Tensor:
            flat = x.reshape(-1)
            mapped = [pyfunc(xi) for xi in flat]
            out = torch.stack(
                [
                    (
                        m
                        if isinstance(m, torch.Tensor)
                        else torch.tensor(m, dtype=self._dtype(), device=self._device())
                    )
                    for m in mapped
                ]
            )
            return out.view(x.shape)

        return wrapped

    @contextlib.contextmanager
    def errstate(self, **kwargs: Any) -> Generator[None, None, None]:  # type: ignore[override]
        """No-op context manager (torch has no equivalent of np.errstate).

        Args:
            **kwargs: Ignored.

        Yields:
            None
        """
        yield

    def histogram(self, x: Any, bins: Any = 10) -> tuple[Tensor, Tensor]:
        """Compute a histogram of x.

        Args:
            x: Input tensor.
            bins: Number of bins or bin edge tensor.

        Returns:
            tuple[Tensor, Tensor]: Bin counts and bin edges.
        """
        if isinstance(bins, int):
            return torch.histogram(x.float(), bins=bins)
        return torch.histogram(x.float(), bins=bins.float())

    def histogram2d(
        self,
        x: Tensor,
        y: Tensor,
        bins: Any,
        weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute a 2-D histogram.

        Args:
            x: x-coordinates of the sample points.
            y: y-coordinates of the sample points.
            bins: List or tuple of two edge tensors.
            weights: Optional weights for each sample.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Histogram, x edges, y edges.

        Raises:
            ValueError: If bins is not a list/tuple of two edge tensors.
        """
        if not isinstance(bins, list | tuple) or len(bins) != 2:
            raise ValueError("`bins` must be a list or tuple of two edge tensors.")

        x_edges, y_edges = bins[0], bins[1]
        nx = x_edges.numel() - 1
        ny = y_edges.numel() - 1

        x_bin_indices = torch.searchsorted(x_edges, x, right=False) - 1
        y_bin_indices = torch.searchsorted(y_edges, y, right=False) - 1
        x_bin_indices = torch.clamp(x_bin_indices, 0, nx - 1)
        y_bin_indices = torch.clamp(y_bin_indices, 0, ny - 1)

        mask = (
            (x >= x_edges[0])
            & (x <= x_edges[-1])
            & (y >= y_edges[0])
            & (y <= y_edges[-1])
        )

        if weights is None:
            weights = torch.ones_like(x)

        valid_x = x_bin_indices[mask]
        valid_y = y_bin_indices[mask]
        valid_w = weights[mask]

        linear_indices = (valid_y * nx + valid_x).long()
        hist_flat = torch.zeros(nx * ny, device=x.device, dtype=valid_w.dtype)
        hist_flat.index_add_(0, linear_indices, valid_w)
        hist = hist_flat.reshape(ny, nx).T

        return hist, x_edges, y_edges
