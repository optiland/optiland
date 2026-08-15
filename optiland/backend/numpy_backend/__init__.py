"""
NumPy backend — implements AbstractBackend using NumPy and SciPy.

The implementation is split across same-package modules by operation
category (creation, indexing, math, linalg, interpolation, random,
misc); this module composes them into the concrete ``NumpyBackend``
class. See ``creation.py`` etc. for the actual method bodies.

Kramer Harrison, 2024, 2025
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from optiland.backend.base import AbstractBackend
from optiland.backend.numpy_backend.creation import CreationMixin
from optiland.backend.numpy_backend.indexing import IndexingMixin
from optiland.backend.numpy_backend.interpolation import InterpolationMixin
from optiland.backend.numpy_backend.linalg import LinalgMixin
from optiland.backend.numpy_backend.math import MathMixin
from optiland.backend.numpy_backend.misc import MiscMixin
from optiland.backend.numpy_backend.random import RandomMixin


class NumpyBackend(
    CreationMixin,
    IndexingMixin,
    MathMixin,
    LinalgMixin,
    InterpolationMixin,
    RandomMixin,
    MiscMixin,
    AbstractBackend,
):
    """Backend implementation using NumPy and SciPy.

    Attributes:
        _lib: The NumPy module (used by passthrough methods).
        _precision: Current floating-point precision string.
    """

    _lib = np
    _precision: Literal["float32", "float64"] = "float64"
