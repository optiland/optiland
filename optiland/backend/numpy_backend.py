"""
NumPy backend — implements AbstractBackend using NumPy and SciPy.

The implementation is split across same-directory ``_numpy_*`` mixin
modules by operation category (creation, indexing, math, linalg,
interpolation, random, misc); this module composes them into the
concrete ``NumpyBackend`` class. See ``_numpy_creation.py`` etc. for the
actual method bodies.

Kramer Harrison, 2024, 2025
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from optiland.backend._numpy_creation import NumpyCreationMixin
from optiland.backend._numpy_indexing import NumpyIndexingMixin
from optiland.backend._numpy_interpolation import NumpyInterpolationMixin
from optiland.backend._numpy_linalg import NumpyLinalgMixin
from optiland.backend._numpy_math import NumpyMathMixin
from optiland.backend._numpy_misc import NumpyMiscMixin
from optiland.backend._numpy_random import NumpyRandomMixin
from optiland.backend.base import AbstractBackend


class NumpyBackend(
    NumpyCreationMixin,
    NumpyIndexingMixin,
    NumpyMathMixin,
    NumpyLinalgMixin,
    NumpyInterpolationMixin,
    NumpyRandomMixin,
    NumpyMiscMixin,
    AbstractBackend,
):
    """Backend implementation using NumPy and SciPy.

    Attributes:
        _lib: The NumPy module (used by passthrough methods).
        _precision: Current floating-point precision string.
    """

    _lib = np
    _precision: Literal["float32", "float64"] = "float64"
