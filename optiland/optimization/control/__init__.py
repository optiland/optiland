"""Step control for the native optimizers (opt v2 M4). Private.

Kramer Harrison, 2026
"""

from __future__ import annotations

from .levenberg import LevenbergController

__all__ = ["LevenbergController"]


def __getattr__(name: str):
    # kkt imports the operand stack: load it on first use.
    if name == "KKTController":
        from .kkt import KKTController

        return KKTController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
