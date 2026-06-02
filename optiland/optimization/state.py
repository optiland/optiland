"""Optimization State & Result Module

Defines the mutable per-iteration state and the immutable end-of-run result
passed between the driver, optimizer, controllers, observers, and criteria.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Capability(Enum):
    """Capabilities an optimizer may declare."""

    OBSERVE = auto()
    STEP_CONTROL = auto()
    MUTATE_TERMINATION = auto()
    PER_STEP_CONSTRAINTS = auto()
    SCIPY_CONSTRAINTS = auto()
    BOUNDS = auto()


class EvalCapability(Enum):
    """Capabilities an evaluator may declare."""

    VALUE = auto()
    RESIDUALS = auto()
    GRADIENT = auto()
    JACOBIAN = auto()


@dataclass
class OptimizationState:
    """Mutable state carried across iterations.

    Mutated **in place** by the driver and optimizer to avoid per-iteration
    allocation.  Observers receive a reference to this object; they must treat
    it as read-only by contract.

    Attributes:
        x: Current parameter vector in scaled space.
        value: Current merit value (sum_squared), as a backend scalar.
        residuals: Last residual vector (LM path); None for gradient methods.
        gradient: Last gradient vector; None when not computed.
        jacobian: Last Jacobian matrix (m × n); None when not computed.
        iteration: Number of **accepted** steps completed.
        n_value_evals: Total merit evaluations (incl. trials).
        n_jac_evals: Total Jacobian evaluations.
        n_trial_evals: Inner λ-search trial evaluations (LM only).
        step_norm: |Δx| of the last accepted step.
        converged: Set True by a criterion / intrinsic convergence test.
        stop_reason: Human-readable reason string, set on termination.
        scratch: Optimizer-private scratch space (LM λ, etc.).
    """

    x: Any
    value: Any
    residuals: Any | None = None
    gradient: Any | None = None
    jacobian: Any | None = None
    iteration: int = 0
    n_value_evals: int = 0
    n_jac_evals: int = 0
    n_trial_evals: int = 0
    step_norm: float | None = None
    converged: bool = False
    stop_reason: str | None = None
    scratch: dict = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Immutable end-of-run result.

    Exposes a duck-compatible superset of ``scipy.optimize.OptimizeResult``
    so that existing code reading ``.x``, ``.fun``, ``.success``, or
    ``.message`` continues to work.

    Attributes:
        x: Final parameter vector (backend array, scaled space).
        success: Whether the run terminated successfully.
        status: Human-readable stop reason.
        value: Final merit value (Python float, synced once at end).
        initial_value: Merit at the start of the run.
        improvement_pct: ``(initial - final) / initial * 100``.
        iterations: Number of accepted steps.
        n_value_evals: Total value evaluations.
        n_jac_evals: Total Jacobian evaluations.
        wall_time_s: Wall-clock seconds for the full run.
        history: Per-iteration snapshots from HistoryObserver, or None.
        backend: ``"numpy"`` or ``"torch"``.
        method: Resolved method string.
    """

    x: Any
    success: bool
    status: str
    value: float
    initial_value: float
    improvement_pct: float
    iterations: int
    n_value_evals: int
    n_jac_evals: int
    wall_time_s: float
    history: list | None
    backend: str
    method: str

    @property
    def fun(self) -> float:
        """SciPy-compatible alias for ``value``."""
        return self.value

    @property
    def message(self) -> str:
        """SciPy-compatible alias for ``status``."""
        return self.status
