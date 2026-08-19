"""Backends subpackage for Non-Sequential Raytracing."""

from __future__ import annotations

from .base import TracerBackend
from .numpy_backend import NumpyBackend

__all__ = ["NumpyBackend", "TracerBackend"]

try:
    from .torch_backend import TorchBackend  # noqa: F401

    __all__ += ["TorchBackend"]
except ImportError:
    pass
