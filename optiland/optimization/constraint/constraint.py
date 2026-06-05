"""Constraint — a thin, backend-agnostic hard-constraint specification (§4.1).

A ``Constraint`` reuses the same spec shape as an :class:`Operand`
(``operand_type`` + ``input_data`` + ``target`` / ``min_val`` / ``max_val``),
but its *role* is different: it is a **hard** equality / inequality the KKT
active-set controller enforces, not a merit term.

Row conventions (each constraint expands to one or two rows):

* ``target=v``   → **equality**   ``c(x) = scale·(value(x) − v) = 0``
* ``min_val=lo`` → **inequality** ``g(x) = scale·(lo − value(x)) ≤ 0``
* ``max_val=hi`` → **inequality** ``g(x) = scale·(value(x) − hi) ≤ 0``
* both ``min_val`` and ``max_val`` → two inequality rows.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optiland.optimization.operand.operand import operand_registry

EQUALITY = "equality"
INEQUALITY = "inequality"


@dataclass
class ConstraintRow:
    """One scalar constraint row produced by a :class:`Constraint`.

    Attributes:
        kind: ``"equality"`` (``c == 0``) or ``"inequality"`` (``g <= 0``).
        residual: Current (scaled) residual value.
        tol: Feasibility tolerance for this row.
        label: Human-readable identifier (for diagnostics / active set).
    """

    kind: str
    residual: Any
    tol: float
    label: str


@dataclass
class Constraint:
    """A hard equality / inequality constraint built from an operand spec.

    Args:
        operand_type: Registered operand metric name (e.g. ``"f2"``,
            ``"edge_thickness"``).
        target: Equality target (mutually exclusive with bounds).
        min_val: Lower inequality bound (``value >= min_val``).
        max_val: Upper inequality bound (``value <= max_val``).
        weight: Reserved for parity with operands (unused by the hard KKT path).
        scale: Optional per-row scale; ``None`` means "auto-scale later" and is
            treated as ``1.0`` here.
        input_data: Operand input data dict (must include ``optic`` etc.).
        tol: Feasibility tolerance applied to every row of this constraint.
    """

    operand_type: str | None = None
    target: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    weight: float = 1.0
    scale: float | None = None
    input_data: dict | None = None
    tol: float = 1e-8

    def __post_init__(self) -> None:
        if self.input_data is None:
            self.input_data = {}
        if self.target is not None and (
            self.min_val is not None or self.max_val is not None
        ):
            raise ValueError(
                f"{self.operand_type} constraint cannot mix an equality target "
                "with min_val/max_val bounds."
            )
        if (
            self.min_val is not None
            and self.max_val is not None
            and self.min_val > self.max_val
        ):
            raise ValueError(
                f"{self.operand_type} constraint: min_val is greater than max_val."
            )
        if self.target is None and self.min_val is None and self.max_val is None:
            raise ValueError(
                f"{self.operand_type} constraint needs a target or a min/max bound."
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @property
    def value(self) -> Any:
        """Evaluate the underlying operand metric at the current optic state."""
        fn = operand_registry.get(self.operand_type)
        if fn is None:
            raise ValueError(f"Unknown operand type: {self.operand_type}")
        return fn(**self.input_data)

    @property
    def _scale(self) -> float:
        return 1.0 if self.scale is None else float(self.scale)

    @property
    def n_equality(self) -> int:
        return 1 if self.target is not None else 0

    @property
    def n_inequality(self) -> int:
        if self.target is not None:
            return 0
        return int(self.min_val is not None) + int(self.max_val is not None)

    def rows(self) -> list[ConstraintRow]:
        """Return the (scaled) constraint rows evaluated at the current state."""
        s = self._scale
        v = self.value
        rows: list[ConstraintRow] = []
        if self.target is not None:
            rows.append(
                ConstraintRow(
                    EQUALITY,
                    s * (v - self.target),
                    self.tol,
                    f"{self.operand_type}=={self.target}",
                )
            )
            return rows
        if self.min_val is not None:
            rows.append(
                ConstraintRow(
                    INEQUALITY,
                    s * (self.min_val - v),
                    self.tol,
                    f"{self.operand_type}>={self.min_val}",
                )
            )
        if self.max_val is not None:
            rows.append(
                ConstraintRow(
                    INEQUALITY,
                    s * (v - self.max_val),
                    self.tol,
                    f"{self.operand_type}<={self.max_val}",
                )
            )
        return rows
