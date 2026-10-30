"""Evaluators: residuals and Jacobians of an optimization problem at ``x``.

Private building blocks of the native optimizers (opt v2 M4). Optimizers
depend on the protocols in :mod:`.base`; :class:`FiniteDiffEvaluator` is the
numpy path and :class:`AutogradEvaluator` the torch autograd path.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import Any

from .aim import AimPolicy, ExactAim, FrozenAim, resolve_aim_policy
from .base import (
    Evaluator,
    SupportsJacobian,
    SupportsResiduals,
    SupportsValue,
    SupportsVariables,
)
from .finite_difference import FiniteDiffEvaluator

__all__ = [
    "AimPolicy",
    "AutogradEvaluator",
    "Evaluator",
    "ExactAim",
    "FiniteDiffEvaluator",
    "FrozenAim",
    "SupportsJacobian",
    "SupportsResiduals",
    "SupportsValue",
    "SupportsVariables",
    "resolve_aim_policy",
]


def __getattr__(name: str) -> Any:
    # torch is optional: import the autograd evaluator on first use.
    if name == "AutogradEvaluator":
        from .autograd import AutogradEvaluator

        return AutogradEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
