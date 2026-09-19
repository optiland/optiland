"""
PyTorch backend -- identity, capability flags, overrides, and precision.

Provides CapabilitiesMixin, used by TorchBackend (see
optiland/backend/torch_backend/__init__.py).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from numpy.typing import ArrayLike
    from torch import Tensor

    from optiland.backend.torch_backend.config import GradMode


class CapabilitiesMixin:
    """Identity, capability flags, overrides, and precision."""

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
        """Return True if a GPU device (CUDA or Apple MPS) is available."""
        return torch.cuda.is_available() or torch.backends.mps.is_available()

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

    @contextlib.contextmanager
    def no_grad_unless_enabled(self) -> Generator[None, None, None]:
        """Suppress autograd bookkeeping unless gradients were requested.

        With ``grad_mode`` disabled no tensor requires grad, so no graph is
        built either way; running under ``torch.no_grad`` additionally skips
        the per-op autograd dispatch and version-counter work, which adds up
        over the hundreds of kernels a trace launches. With ``grad_mode``
        enabled this is a no-op so differentiable workflows are untouched.
        """
        if self._config.grad_mode.requires_grad:
            yield
        else:
            with torch.no_grad():
                yield

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

        This is the *declared* complex precision. With emulated float64 on
        ``mps`` (``MetalFloat64``) it is still ``torch.complex128``, but the
        Apple GPU has no complex128, so complex data currently lives on the
        CPU: ``to_complex`` / ``mult_p_E`` decode emulated tensors to CPU
        complex128 (see ``linalg.py``) and ``tensor(..., dtype=complex128)``
        allocates on the CPU.

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
        if self._emulated() and torch.device(kwargs["device"]).type == "mps":
            if kwargs["dtype"] == torch.float64:
                return self._factories().tensor(
                    data, requires_grad=bool(kwargs.get("requires_grad", False))
                )
            if kwargs["dtype"] == torch.complex128:
                # No complex128 on the Apple GPU: complex data lives on the CPU.
                kwargs["device"] = "cpu"
        return torch.tensor(data, **kwargs)

    def copy_to(self, source: Tensor, destination: Tensor) -> None:
        """In-place copy from source to destination tensor.

        Safely handles tensors that require gradients.

        Args:
            source: Source tensor.
            destination: Destination tensor (modified in place).
        """
        if self._is_metal(destination):
            # ``copy_`` dispatches to the MetalFloat64 handler, which writes the
            # (exactly promoted) source into the destination's components.
            with torch.no_grad():
                destination.copy_(source)
        elif destination.requires_grad:
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
        if (
            self._emulated()
            and torch.device(current_device).type == "mps"
            and current_precision == torch.float64
        ):
            factories = self._factories()
            if not isinstance(data, torch.Tensor):
                return factories.tensor(data)
            return factories.as_tensor(data)
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
        # ``.to(float64)`` on a plain mps bool tensor is rejected by torch; when
        # the weights are emulated float64, ``cast`` promotes the mask exactly.
        mask = valid_mask.unsqueeze(1)
        mask = (
            self.cast(mask)
            if self._is_metal(all_weights)
            else mask.to(all_weights.dtype)
        )
        all_weights = all_weights * mask
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
    # Metal (emulated float64 on the Apple GPU) passthroughs
    # ------------------------------------------------------------------

    @staticmethod
    def metal_stats() -> dict[str, int]:
        """Per-op counters of the Metal emulation (``gpu:<op>``,
        ``cpu_fallback:<op>``).

        Returns:
            dict[str, int]: A copy of the counters.
        """
        from optiland.backend.torch_backend import metal

        return metal.stats()

    @staticmethod
    def metal_reset_stats() -> None:
        """Zero the Metal emulation counters."""
        from optiland.backend.torch_backend import metal

        metal.reset_stats()

    @staticmethod
    def metal_mode() -> str:
        """Representation used by new emulated tensors (``'df64'`` or ``'sf64'``)."""
        from optiland.backend.torch_backend import metal

        return metal.get_mode()

    @staticmethod
    def set_metal_mode(mode: str) -> None:
        """Choose the representation for new emulated tensors.

        Args:
            mode: ``'df64'`` (double-single, default) or ``'sf64'`` (exact
                software binary64, slower).
        """
        from optiland.backend.torch_backend import metal

        metal.set_mode(mode)
