"""Deprecation utilities for Optiland.

Provides a single decorator for marking callables as deprecated with a
consistent message format.

Kramer Harrison, 2026
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(replacement: str, removal: str = "v0.7.0") -> Callable[[F], F]:
    """Mark a callable as deprecated.

    Emits a :class:`DeprecationWarning` on every call, pointing callers to the
    replacement API.

    Args:
        replacement: The replacement expression, e.g.
            ``"optic.updater.set_radius()"``.
        removal: Version string for the planned removal.
            Defaults to ``"v0.7.0"``.

    Returns:
        A decorator that wraps the target function with the warning.

    Example::

        @deprecated("optic.updater.set_radius()")
        def set_radius(self, value: float, surface_number: int) -> None:
            self.updater.set_radius(value, surface_number)

    """

    def decorator(func: F) -> F:
        msg = (
            f"{func.__qualname__} is deprecated and will be removed in "
            f"{removal}; use {replacement} instead."
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
