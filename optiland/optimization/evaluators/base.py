"""Evaluator Protocols

Structural protocols (PEP 544) that every concrete Evaluator must satisfy.
Using Protocol lets third parties implement their own evaluators without
importing Optiland base classes.

Differentiation provenance (autograd vs finite-diff) belongs to the Evaluator,
not the optimizer. Optimizers depend on this protocol only.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from optiland.optimization.state import EvalCapability  # noqa: TC001

# Type aliases kept intentionally loose — the concrete type depends on backend.
Array = Any
Scalar = Any


@runtime_checkable
class SupportsValue(Protocol):
    """Evaluator that can compute the scalar merit value."""

    def value(self, x: Array) -> Scalar:
        """Return the merit (sum_squared) at ``x`` without calling ``.item()``."""
        ...  # pragma: no cover


@runtime_checkable
class SupportsResiduals(Protocol):
    """Evaluator that can compute the residual vector."""

    def residuals(self, x: Array) -> Array:
        """Return ``weighted_residuals()`` at ``x`` (shape ``(m,)``)."""
        ...  # pragma: no cover


@runtime_checkable
class SupportsGradient(Protocol):
    """Evaluator that can compute the gradient of the merit w.r.t. ``x``."""

    def gradient(self, x: Array) -> Array:
        """Return d(value)/d(x) (shape ``(n,)``)."""
        ...  # pragma: no cover


@runtime_checkable
class SupportsJacobian(Protocol):
    """Evaluator that can compute the Jacobian of the residuals."""

    def jacobian(self, x: Array) -> Array:
        """Return d(residuals)/d(x) (shape ``(m, n)``)."""
        ...  # pragma: no cover


@runtime_checkable
class Evaluator(Protocol):
    """Full evaluator protocol consumed by optimizers and drivers.

    Every optimizer should use only the minimal sub-protocol it needs (ISP).
    The concrete implementations are ``FiniteDiffEvaluator`` (numpy) and
    ``AutogradEvaluator`` (torch).

    Attributes:
        n_vars: Number of optimization variables (``n``).
        backend: ``"numpy"`` or ``"torch"``.
        provides: Set of ``EvalCapability`` members this evaluator supplies.
    """

    n_vars: int
    backend: str
    provides: frozenset[EvalCapability]

    def value(self, x: Array) -> Scalar: ...  # pragma: no cover

    def residuals(self, x: Array) -> Array: ...  # pragma: no cover

    def gradient(self, x: Array) -> Array: ...  # pragma: no cover

    def jacobian(self, x: Array) -> Array: ...  # pragma: no cover

    def read_x(self) -> Array:
        """Gather ``var.value`` for every variable into a 1-D array."""
        ...  # pragma: no cover

    def write_x(self, x: Array) -> None:
        """Scatter ``x`` back into variables then call ``update_optics()``."""
        ...  # pragma: no cover
