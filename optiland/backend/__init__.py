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
Concurrent calls from multiple threads are not supported. The same applies to
``set_trace_threads``, which sets the threads of the compiled NumPy trace.

Kramer Harrison, 2025
"""

from __future__ import annotations

import math
import numbers
import os
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
# Threads of the compiled trace
# ---------------------------------------------------------------------------
_trace_threads: int = 1


def set_trace_threads(n: int) -> None:
    """Set the number of threads of the compiled NumPy trace. The default is 1.

    In its domain, the NumPy backend traces real rays with a compiled kernel
    (the developer page "Performance Internals" lists the domain). With
    ``n`` above 1, the kernel traces a bundle of 8192 rays or more in parts,
    on the calling thread and ``n - 1`` threads of a private pool of Python
    threads. The kernel releases the GIL. The results are the same, bit for
    bit, as with one thread.

    All other traces use one thread: traces outside the domain of the
    compiled trace, smaller bundles, and all traces with the Torch backend.
    The setting is independent of other thread settings, in both
    directions: it does not change, and does not depend on,
    ``numba.set_num_threads``, ``NUMBA_NUM_THREADS``,
    ``torch.set_num_threads`` or BLAS variables such as
    ``OMP_NUM_THREADS``.

    Args:
        n: The number of threads, 1 or more. -1 sets the number of logical
            CPUs: ``os.process_cpu_count()`` on Python 3.13 and later, which
            counts the CPUs that the process can use, and ``os.cpu_count()``
            before, which counts all CPUs of the system and ignores the CPU
            affinity mask of the process. The number is fixed when this
            function is called. If Python cannot find it, -1 sets 1.

    Raises:
        TypeError: If ``n`` is not an integer, for example a float or a
            bool.
        ValueError: If ``n`` is 0 or below -1.

    Note:
        This function modifies global module state and is **not
        thread-safe**, like ``set_backend``. Call it before other threads
        start to trace.
    """
    global _trace_threads
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise TypeError(
            f"The number of trace threads must be an integer, not {type(n).__name__}."
        )
    if n == 0 or n < -1:
        raise ValueError(
            f"The number of trace threads must be -1 or 1 or more, not {n}."
        )
    if n == -1:
        # process_cpu_count (Python 3.13 and later) respects the CPU affinity
        # mask; cpu_count does not. Both give None if the number is unknown.
        count_cpus = getattr(os, "process_cpu_count", os.cpu_count)
        n = count_cpus() or 1
    _trace_threads = int(n)


def get_trace_threads() -> int:
    """Return the number of threads of the compiled NumPy trace.

    Returns:
        int: The number that ``set_trace_threads`` set, or 1 by default.
    """
    return _trace_threads


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
