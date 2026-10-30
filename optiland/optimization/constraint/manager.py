"""Constraint Manager

Ordered collection of the hard constraints of an optimization problem,
mirroring :class:`~optiland.optimization.operand.OperandManager`. The manager
only holds data and evaluates rows on request; solving is the job of the
native DLS (:mod:`optiland.optimization.control.kkt`).

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be

from .constraint import EQUALITY, INEQUALITY, Constraint, ConstraintRow, row_violations

if TYPE_CHECKING:
    from collections.abc import Iterator

    from optiland.optimization.operand.operand import Operand


def row_report(
    labels: list[str], equality: np.ndarray, tols: np.ndarray, values: np.ndarray
) -> list[dict]:
    """Per-row feasibility report.

    Args:
        labels: Row labels.
        equality: Boolean mask of the equality rows.
        tols: Row tolerances.
        values: Row values.

    Returns:
        One dict per row with ``label``, ``kind``, ``residual``, ``violation``
        and ``feasible``.
    """
    violations = row_violations(equality, values)
    return [
        {
            "label": labels[i],
            "kind": EQUALITY if equality[i] else INEQUALITY,
            "residual": float(values[i]),
            "violation": float(violations[i]),
            "feasible": bool(violations[i] <= tols[i]),
        }
        for i in range(len(labels))
    ]


class ConstraintManager:
    """Ordered, iterable collection of :class:`Constraint`. *Experimental.*

    Attributes:
        constraints: The constraints, in declaration order.
    """

    def __init__(self) -> None:
        self.constraints: list[Constraint] = []

    def add(
        self,
        operand_type: str | None = None,
        target: float | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        weight: float = 1.0,
        input_data: dict | None = None,
        tol: float = 1e-8,
    ) -> Constraint:
        """Append a constraint and return it (same spec as ``add_operand``)."""
        constraint = Constraint(
            operand_type=operand_type,
            target=target,
            min_val=min_val,
            max_val=max_val,
            weight=weight,
            input_data=input_data,
            tol=tol,
        )
        self.constraints.append(constraint)
        return constraint

    def clear(self) -> None:
        """Remove all constraints."""
        self.constraints = []

    def __iter__(self) -> Iterator[Constraint]:
        return iter(self.constraints)

    def __len__(self) -> int:
        return len(self.constraints)

    def __getitem__(self, index: int) -> Constraint:
        return self.constraints[index]

    @property
    def rows(self) -> list[ConstraintRow]:
        """All rows, in declaration order."""
        return [row for constraint in self.constraints for row in constraint.rows]

    @property
    def row_operands(self) -> list[Operand]:
        """The operand behind each row, for evaluators sharing the merit's bundles."""
        return [row.operand for row in self.rows]

    @property
    def signs(self) -> list[float]:
        """The sign of each row."""
        return [row.sign for row in self.rows]

    @property
    def n_rows(self) -> int:
        """Number of rows."""
        return sum(len(constraint.rows) for constraint in self.constraints)

    @property
    def n_equality(self) -> int:
        """Number of equality rows."""
        return sum(row.kind == EQUALITY for row in self.rows)

    @property
    def n_inequality(self) -> int:
        """Number of inequality rows."""
        return sum(row.kind == INEQUALITY for row in self.rows)

    @property
    def equality_mask(self) -> np.ndarray:
        """Boolean mask of the equality rows."""
        return np.array([row.kind == EQUALITY for row in self.rows], dtype=bool)

    @property
    def tols(self) -> np.ndarray:
        """Feasibility tolerance of each row."""
        return np.array([row.tol for row in self.rows], dtype=float)

    @property
    def labels(self) -> list[str]:
        """Unique row labels, ``"#<row> <operand> <relation> <limit>"``."""
        return [f"#{i} {row.label}" for i, row in enumerate(self.rows)]

    def vector(self) -> Any:
        """Signed row values at the current optic state, one call per row."""
        rows = self.rows
        if not rows:
            return be.array([])
        return be.stack([row.residual() for row in rows])

    def report(self, values: Any = None) -> list[dict]:
        """Per-row feasibility report.

        Args:
            values: Row values already computed; evaluated when None.

        Returns:
            See :func:`row_report`.
        """
        values = self.vector() if values is None else values
        values = np.asarray(be.to_numpy(values), dtype=float).reshape(-1)
        return row_report(self.labels, self.equality_mask, self.tols, values)

    def max_violation(self, values: Any = None) -> float:
        """Largest row violation (0.0 without constraints).

        Args:
            values: Row values already computed; evaluated when None.
        """
        return max((row["violation"] for row in self.report(values)), default=0.0)
