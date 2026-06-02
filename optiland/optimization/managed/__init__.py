"""Managed optimizer adapters — wrap existing SciPy-based optimizers.

Each adapter delegates to the corresponding existing class and exposes a
uniform ``run(callback, **method_options)`` interface consumed by
``ManagedDriver``.

Kramer Harrison, 2026
"""

from __future__ import annotations

from .scipy_global import ScipyGlobalAdapter  # noqa: F401
from .scipy_least_squares import ScipyLeastSquaresAdapter  # noqa: F401
from .scipy_local import ScipyLocalAdapter  # noqa: F401
