"""
PyTorch backend — implements AbstractBackend using PyTorch.

The implementation is split across same-directory ``_torch_*`` mixin
modules by operation category (capabilities, creation, indexing,
reductions, passthrough, linalg, interpolation, random, misc); this
module composes them into the concrete ``TorchBackend`` class. See
``_torch_capabilities.py`` etc. for the actual method bodies, and
``_torch_config.py`` for the ``GradMode``/``_Config`` helpers.

Kramer Harrison, 2025
"""

from __future__ import annotations

import torch

from optiland.backend._torch_capabilities import TorchCapabilitiesMixin
from optiland.backend._torch_config import _Config
from optiland.backend._torch_creation import TorchCreationMixin
from optiland.backend._torch_indexing import TorchIndexingMixin
from optiland.backend._torch_interpolation import TorchInterpolationMixin
from optiland.backend._torch_linalg import TorchLinalgMixin
from optiland.backend._torch_misc import TorchMiscMixin
from optiland.backend._torch_passthrough import TorchPassthroughMixin
from optiland.backend._torch_random import TorchRandomMixin
from optiland.backend._torch_reductions import TorchReductionsMixin
from optiland.backend.base import AbstractBackend


class TorchBackend(
    TorchCapabilitiesMixin,
    TorchCreationMixin,
    TorchIndexingMixin,
    TorchReductionsMixin,
    TorchPassthroughMixin,
    TorchLinalgMixin,
    TorchInterpolationMixin,
    TorchRandomMixin,
    TorchMiscMixin,
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
