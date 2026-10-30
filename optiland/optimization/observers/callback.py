"""CallbackObserver: SciPy-style callbacks as an observer.

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import OptimizeResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.optimization.state import OptimizationResult, OptimizationState


def callback_wants_result(callback: Callable) -> bool:
    """SciPy convention: a sole parameter named ``intermediate_result``."""
    try:
        params = list(inspect.signature(callback).parameters)
    except (TypeError, ValueError):
        return False
    return params == ["intermediate_result"]


class CallbackObserver:
    """Calls a SciPy-style callback after every accepted step. *Experimental.*

    The callback receives ``x`` or, when its only parameter is named
    ``intermediate_result``, an ``OptimizeResult`` with ``x``, ``fun``
    (residuals), ``cost`` (``0.5 * merit``) and ``nit``. Raising
    ``StopIteration`` ends the run.

    Args:
        callback: The callable to wrap.
    """

    def __init__(self, callback: Callable) -> None:
        self._callback = callback
        self._wants_result = callback_wants_result(callback)

    def on_start(self, state: OptimizationState) -> None:
        """Do nothing."""

    def on_step(self, state: OptimizationState) -> None:
        """Call the wrapped callback."""
        x = np.array(state.x, dtype=float)
        if self._wants_result:
            self._callback(
                intermediate_result=OptimizeResult(
                    x=x,
                    fun=state.residuals,
                    cost=0.5 * float(state.value),
                    nit=state.iteration,
                )
            )
        else:
            self._callback(x)

    def on_end(self, state: OptimizationState, result: OptimizationResult) -> None:
        """Do nothing."""
