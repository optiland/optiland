"""ConstraintManager — ordered collection of hard constraints (§4.1).

Mirrors :class:`OperandManager`: ``add`` / ``clear`` / iteration plus
vectorized residual assembly (``c_vector`` equalities, ``g_vector``
inequalities) and finite-difference Jacobian assembly that reuses an
evaluator's ``read_x`` / ``write_x`` machinery rather than re-implementing the
parameter sync.

The manager is **solver-free**: it stores constraint data and evaluates
residuals / Jacobians on request, exactly like the problem stores merit
operands.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import Any

import numpy as np

import optiland.backend as be

from .constraint import EQUALITY, INEQUALITY, Constraint, ConstraintRow


class ConstraintManager:
    """Ordered, iterable collection of :class:`Constraint` objects."""

    def __init__(self) -> None:
        self.constraints: list[Constraint] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add(
        self,
        operand_type: str | None = None,
        target: float | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        weight: float = 1.0,
        input_data: dict | None = None,
        tol: float = 1e-8,
        scale: float | None = None,
    ) -> Constraint:
        """Append a constraint (same spec shape as ``add_operand``)."""
        c = Constraint(
            operand_type=operand_type,
            target=target,
            min_val=min_val,
            max_val=max_val,
            weight=weight,
            scale=scale,
            input_data=input_data if input_data is not None else {},
            tol=tol,
        )
        self.constraints.append(c)
        return c

    def clear(self) -> None:
        self.constraints = []

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __iter__(self):
        return iter(self.constraints)

    def __len__(self) -> int:
        return len(self.constraints)

    def __getitem__(self, index: int) -> Constraint:
        return self.constraints[index]

    @property
    def n_equality(self) -> int:
        return sum(c.n_equality for c in self.constraints)

    @property
    def n_inequality(self) -> int:
        return sum(c.n_inequality for c in self.constraints)

    @property
    def has_constraints(self) -> bool:
        return len(self.constraints) > 0

    # ------------------------------------------------------------------
    # Residual assembly (evaluated at the current optic state)
    # ------------------------------------------------------------------

    def _rows(self, kind: str) -> list[ConstraintRow]:
        rows: list[ConstraintRow] = []
        for c in self.constraints:
            rows.extend(r for r in c.rows() if r.kind == kind)
        return rows

    def equality_rows(self) -> list[ConstraintRow]:
        return self._rows(EQUALITY)

    def inequality_rows(self) -> list[ConstraintRow]:
        return self._rows(INEQUALITY)

    def c_vector(self) -> Any:
        """Equality residual vector ``c(x)`` at the current state."""
        rows = self.equality_rows()
        if not rows:
            return be.array([])
        return be.stack([be.array(r.residual) for r in rows])

    def g_vector(self) -> Any:
        """Inequality residual vector ``g(x)`` (``<= 0`` feasible)."""
        rows = self.inequality_rows()
        if not rows:
            return be.array([])
        return be.stack([be.array(r.residual) for r in rows])

    def labels(self, kind: str) -> list[str]:
        return [r.label for r in self._rows(kind)]

    # ------------------------------------------------------------------
    # Finite-difference Jacobian assembly (reuses evaluator read/write)
    # ------------------------------------------------------------------

    def _fd_jacobian(self, evaluator: Any, vector_fn, n_rows: int) -> np.ndarray:
        """Forward finite-difference Jacobian of ``vector_fn`` w.r.t. ``x``.

        Reuses ``evaluator.read_x`` / ``evaluator.write_x`` so the parameter
        sync (and any scaling) matches the merit path exactly.  Returns an
        ``(n_rows, n_vars)`` numpy array; the optic is restored to ``x`` on exit.
        """
        x = np.asarray(be.to_numpy(evaluator.read_x()), dtype=float)
        n_vars = x.shape[0]
        J = np.zeros((n_rows, n_vars))
        if n_rows == 0 or n_vars == 0:
            evaluator.write_x(x)
            return J
        evaluator.write_x(x)
        base = np.asarray(be.to_numpy(vector_fn()), dtype=float)
        for i in range(n_vars):
            h = 1e-6 * abs(x[i]) + 1e-8
            xp = x.copy()
            xp[i] += h
            evaluator.write_x(xp)
            pert = np.asarray(be.to_numpy(vector_fn()), dtype=float)
            J[:, i] = (pert - base) / h
        evaluator.write_x(x)
        return J

    def equality_jacobian(self, evaluator: Any) -> np.ndarray:
        """``A_eq`` = ∂c/∂x via finite differences, shape ``(n_eq, n_vars)``."""
        return self._fd_jacobian(evaluator, self.c_vector, self.n_equality)

    def inequality_jacobian(self, evaluator: Any) -> np.ndarray:
        """``A_ineq`` = ∂g/∂x via finite differences, shape ``(n_ineq, n_vars)``."""
        return self._fd_jacobian(evaluator, self.g_vector, self.n_inequality)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def report(self) -> list[dict]:
        """Per-row feasibility report at the current state."""
        out: list[dict] = []
        for kind in (EQUALITY, INEQUALITY):
            for r in self._rows(kind):
                res = float(be.to_numpy(be.array(r.residual)))
                violation = abs(res) if kind == EQUALITY else max(0.0, res)
                out.append(
                    {
                        "label": r.label,
                        "kind": kind,
                        "residual": res,
                        "violation": violation,
                        "feasible": violation <= r.tol,
                    }
                )
        return out

    def max_violation(self) -> float:
        """Largest feasibility violation across all rows (0.0 if none)."""
        rep = self.report()
        return max((row["violation"] for row in rep), default=0.0)
