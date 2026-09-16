"""
PyTorch backend — configuration helpers (private to the backend package).

Provides GradMode/_Config, shared by TorchBackend's __init__ and the
capabilities mixin.

Kramer Harrison, 2025
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from collections.abc import Generator


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
        self.device: Literal["cpu", "cuda", "mps"] = "cpu"
        self.precision: torch.dtype = torch.float32
        self.grad_mode: GradMode = GradMode()

    def set_device(self, device: Literal["cpu", "cuda", "mps"]) -> None:
        """Set the compute device.

        Args:
            device: ``'cpu'``, ``'cuda'`` or ``'mps'`` (Apple GPU; float64 on
                ``mps`` is emulated, see ``optiland.backend.torch_backend.metal``).

        Raises:
            ValueError: If device is not one of the supported names, or if the
                requested accelerator is unavailable.
        """
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError("Device must be 'cpu', 'cuda' or 'mps'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS (Apple GPU) is not available.")
        self.device = device

    def get_device(self) -> Literal["cpu", "cuda", "mps"]:
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
