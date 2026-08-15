"""
PyTorch backend — implements AbstractBackend using PyTorch.

The implementation is split across same-package modules by operation
category (capabilities, creation, indexing, reductions, passthrough,
linalg, interpolation, random, misc); this module composes them into
the concrete ``TorchBackend`` class. See ``capabilities.py`` etc. for
the actual method bodies, and ``config.py`` for the ``GradMode``/
``_Config`` helpers.

Kramer Harrison, 2025
"""

from __future__ import annotations

import torch

from optiland.backend.base import AbstractBackend
from optiland.backend.torch_backend.capabilities import CapabilitiesMixin
from optiland.backend.torch_backend.config import _Config
from optiland.backend.torch_backend.creation import CreationMixin
from optiland.backend.torch_backend.indexing import IndexingMixin
from optiland.backend.torch_backend.interpolation import InterpolationMixin
from optiland.backend.torch_backend.linalg import LinalgMixin
from optiland.backend.torch_backend.misc import MiscMixin
from optiland.backend.torch_backend.passthrough import PassthroughMixin
from optiland.backend.torch_backend.random import RandomMixin
from optiland.backend.torch_backend.reductions import ReductionsMixin


class TorchBackend(
    CapabilitiesMixin,
    CreationMixin,
    IndexingMixin,
    ReductionsMixin,
    PassthroughMixin,
    LinalgMixin,
    InterpolationMixin,
    RandomMixin,
    MiscMixin,
    AbstractBackend,
):
    """Backend implementation using PyTorch.

    Attributes:
        _lib: The torch module (used by passthrough methods).
        _config: Internal configuration (device, precision, grad_mode).
    """

    _lib = torch

    def __init__(self) -> None:
        self._config = _Config()
