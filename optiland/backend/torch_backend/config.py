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
            device: ``'cpu'``, ``'cuda'``, or ``'mps'`` (Apple GPU).

        Raises:
            ValueError: If device is not one of the three above; if CUDA is
                requested but unavailable; if ``mps`` is requested but
                unavailable; or if ``mps`` is requested while the active
                precision is float64 -- Apple's Metal shader language has no
                double type (Metal Shading Language Specification sec 2.3),
                so torch's own mps backend refuses float64 tensors. That
                refusal is real and correct; this only turns it into a
                clear, immediate error instead of one raised the first time
                a tensor is allocated.
        """
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError("Device must be 'cpu', 'cuda', or 'mps'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        if device == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError(
                    "MPS (Apple GPU) is not available on this machine/torch build."
                )
            if self.precision == torch.float64:
                raise ValueError(
                    "MPS (Apple GPU) has no float64 support (Metal has no double "
                    "type). Call set_precision('float32') before set_device('mps'), "
                    "or stay on 'cpu'/'cuda' for float64 work."
                )
        self.device = device

    def get_device(self) -> Literal["cpu", "cuda", "mps"]:
        """Return the current device."""
        return self.device

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        """Set the floating-point precision.

        Args:
            precision: ``'float32'`` or ``'float64'``.

        Raises:
            ValueError: If precision is not valid, or if ``'float64'`` is
                requested while the active device is ``'mps'`` -- see
                :meth:`set_device`.
        """
        if precision == "float32":
            self.precision = torch.float32
        elif precision == "float64":
            if self.device == "mps":
                raise ValueError(
                    "MPS (Apple GPU) has no float64 support (Metal has no double "
                    "type). Call set_device('cpu') or set_device('cuda') first, or "
                    "stay on float32 for mps."
                )
            self.precision = torch.float64
        else:
            raise ValueError("Precision must be 'float32' or 'float64'.")

    def get_precision(self) -> torch.dtype:
        """Return the current torch dtype."""
        return self.precision
