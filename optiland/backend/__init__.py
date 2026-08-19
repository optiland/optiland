"""
Backend Module

Provides a unified module-level interface for numerical operations backed by
either NumPy or PyTorch. The default backend is NumPy; switch with
``set_backend('torch')``.

Usage::

    import optiland.backend as be
    be.set_backend('torch')      # switch to PyTorch
    x = be.array([1.0, 2.0])    # uses the active backend
    y = be.sin(x)

Module-level constants (backend-independent)::

    be.inf, be.nan, be.pi, be.newaxis, be.e

Note on ``to_numpy``
--------------------
``be.to_numpy`` is a **boundary utility** for converting backend arrays to
NumPy at system boundaries (tests, IO, visualization). It is not a
computation function and breaks the backend abstraction. Internal code should
import it directly from ``optiland.backend.utils``::

    from optiland.backend.utils import to_numpy

Note on thread safety
---------------------
``set_backend`` modifies global module state and is **not thread-safe**. It is
intended to be called once at program startup or at the beginning of a test.
Concurrent calls from multiple threads are not supported.

Kramer Harrison, 2025
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as _np

from optiland._suggest import options_hint
from optiland.backend.base import BackendCapabilityError  # noqa: F401
from optiland.backend.numpy_backend import NumpyBackend
from optiland.backend.utils import is_torch_tensor, to_numpy  # noqa: F401

if TYPE_CHECKING:
    from optiland._types import BEArrayT
    from optiland.backend.base import AbstractBackend

# ---------------------------------------------------------------------------
# Backend-independent constants
# ---------------------------------------------------------------------------
inf = float("inf")
nan = float("nan")
pi = math.pi
e = math.e
newaxis = None  # equivalent to np.newaxis

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# ndarray: either a NumPy ndarray or a PyTorch Tensor
try:
    import torch as _torch

    ndarray = (_np.ndarray, _torch.Tensor)
    _torch_importable = True
except (ImportError, OSError):
    ndarray = _np.ndarray  # type: ignore[assignment]
    _torch_importable = False

# NumPy dtype aliases — for compatibility with callers that use be.float32 / be.float64
float32 = _np.float32
float64 = _np.float64

# finfo — expose numpy's finfo for machine-epsilon queries
finfo = _np.finfo


def array_equal(a: BEArrayT, b: BEArrayT) -> bool:
    """Element-wise equality test for arrays/tensors in the active backend.

    Args:
        a: First array or tensor.
        b: Second array or tensor.

    Returns:
        bool: True if a and b are equal element-wise.
    """
    if _current_backend == "torch" and _torch_importable:
        return _torch.equal(a, b)
    return _np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Backend registry (singletons)
# ---------------------------------------------------------------------------
_backends: dict[str, AbstractBackend] = {
    "numpy": NumpyBackend(),
}

try:
    from optiland.backend.torch_backend import TorchBackend

    _backends["torch"] = TorchBackend()
    _torch_available = True
except (ImportError, OSError):
    _torch_available = False

_current_backend: str = "numpy"


# ---------------------------------------------------------------------------
# Backend management
# ---------------------------------------------------------------------------


def set_backend(name: str) -> None:
    """Set the current backend.

    Args:
        name: Backend name. Must be one of the registered backends (see
            ``list_available_backends()``).

    Raises:
        ValueError: If the backend name is not registered.

    Note:
        This function modifies global module state and is **not thread-safe**.
        It is intended to be called once at program startup or at the
        beginning of a test. Concurrent calls from multiple threads are not
        supported.
    """
    global _current_backend
    if name not in _backends:
        available = list_available_backends()
        raise ValueError(
            f"Unknown backend {name!r}.{options_hint(str(name), available)} "
            "Install the extra for a missing backend, e.g. "
            "pip install optiland[torch]."
        )
    _current_backend = name


def get_backend() -> str:
    """Return the name of the current active backend.

    Returns:
        str: Backend name (e.g. ``'numpy'`` or ``'torch'``).
    """
    return _current_backend


def list_available_backends() -> list[str]:
    """Return a list of all registered backend names.

    Returns:
        list[str]: Available backend names.
    """
    return list(_backends.keys())


# ---------------------------------------------------------------------------
# Module-level attribute dispatch
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> object:
    """Delegate attribute access to the active backend instance.

    This enables the ``be.sin(x)``, ``be.array(...)``, etc. call pattern at
    module level. The active backend instance is an ``AbstractBackend``
    subclass, so every attribute resolved here is guaranteed by the ABC
    contract (or raises ``AttributeError`` with a clear message).

    Args:
        name: Attribute name to resolve.

    Returns:
        object: The attribute from the active backend.

    Raises:
        AttributeError: If the active backend has no such attribute.
    """
    if name in globals():
        return globals()[name]
    instance = _backends[_current_backend]
    try:
        return getattr(instance, name)
    except AttributeError:
        raise AttributeError(
            f"Backend '{_current_backend}' has no attribute '{name}'."
        ) from None


def __dir__() -> list[str]:
    """Extend the module's directory listing to include backend attributes.

    Returns:
        list[str]: Sorted list of all available names.
    """
    instance = _backends[_current_backend]
    return sorted(set(globals().keys()) | set(dir(instance)))
