"""Evaluator Protocols

Structural protocols (PEP 544) for the objects that evaluate an
:class:`~optiland.optimization.problem.OptimizationProblem` at a variable
vector ``x``. Differentiation provenance (finite differences or autograd)
belongs to the evaluator, not to the optimizer, and each optimizer depends only
on the sub-protocol it uses (ISP).

Residuals follow the problem's convention everywhere: ``r_i =
effective_weight_i * delta_i`` (``OptimizationProblem.residual_vector``), so
the merit ``value(x)`` equals ``sum(r ** 2)`` and ``sum_squared()``.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Loose aliases: the concrete array type depends on the evaluator's backend.
Array = Any
Scalar = Any


@runtime_checkable
class SupportsValue(Protocol):
    """Evaluator that computes the scalar merit."""

    def value(self, x: Array) -> Scalar:
        """Return the merit ``sum(r ** 2)`` at ``x``."""
        ...  # pragma: no cover


@runtime_checkable
class SupportsResiduals(Protocol):
    """Evaluator that computes the residual vector."""

    def residuals(self, x: Array) -> Array:
        """Return the residuals ``effective_weight * delta`` at ``x``, shape (m,)."""
        ...  # pragma: no cover


@runtime_checkable
class SupportsJacobian(Protocol):
    """Evaluator that computes the Jacobian of the residuals."""

    def jacobian(self, x: Array, r0: Array | None = None) -> Array:
        """Return ``d(residuals)/d(x)`` at ``x``, shape ``(m, n)``.

        Args:
            x: The point.
            r0: The residuals already computed at ``x``, if known. Evaluators
                that need a baseline reuse it instead of evaluating again.
        """
        ...  # pragma: no cover


@runtime_checkable
class SupportsConstraints(Protocol):
    """Evaluator that computes hard-constraint rows with the residuals.

    Rows are signed ``effective_weight * delta`` values from the same
    evaluation as the residuals at ``x``: reading them costs no extra design
    evaluation, and their Jacobian comes from the same columns as ``J``.
    """

    n_constraints: int

    def constraint_rows(self) -> tuple[Any, Any, list[str]]:
        """Return ``(equality mask, tolerances, labels)`` of the rows."""
        ...  # pragma: no cover

    def constraints(self, x: Array) -> Array:
        """Return the signed rows at ``x``, shape ``(c,)``."""
        ...  # pragma: no cover

    def constraint_jacobian(self, x: Array) -> Array:
        """Return ``d(rows)/d(x)`` at ``x``, shape ``(c, n)``."""
        ...  # pragma: no cover


@runtime_checkable
class SupportsVariables(Protocol):
    """Evaluator that reads and writes the variable vector."""

    n_vars: int
    backend: str

    def read_x(self) -> Array:
        """Gather the current (scaled) variable values into a 1-D array."""
        ...  # pragma: no cover

    def write_x(self, x: Array) -> None:
        """Write ``x`` into the variables, then update the optics."""
        ...  # pragma: no cover


@runtime_checkable
class Evaluator(
    SupportsValue, SupportsResiduals, SupportsJacobian, SupportsVariables, Protocol
):
    """Full evaluator protocol.

    Implemented by :class:`~optiland.optimization.evaluators.FiniteDiffEvaluator`
    and :class:`~optiland.optimization.evaluators.AutogradEvaluator`.

    Attributes:
        n_vars: Number of optimization variables ``n``.
        backend: ``"numpy"`` or ``"torch"``.
    """
