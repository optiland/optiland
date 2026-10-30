"""Hard Constraints

A :class:`Constraint` uses the spec of an operand (``operand_type``,
``input_data``, ``target`` / ``min_val`` / ``max_val``, ``weight``), but it is
a **hard** condition enforced by the native DLS, not a merit term.

Residuals follow decision V7, ``effective_weight * delta``. Each constraint
expands to one or two signed rows, each backed by an :class:`Operand` whose
target is the limit, so a row's value comes from the same evaluation as the
merit:

* ``target=v``: equality ``ew * (value - v) = 0``.
* ``min_val=lo``: inequality ``ew * (lo - value) <= 0``.
* ``max_val=hi``: inequality ``ew * (value - hi) <= 0``.

A row whose effective weight is zero (a zero-weight field or wavelength) is
exactly 0 and constrains nothing.

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import optiland.backend as be
from optiland.optimization.failure import ConfigurationError
from optiland.optimization.operand.operand import Operand, operand_registry

EQUALITY = "equality"
INEQUALITY = "inequality"


def row_violations(equality: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Violation of each signed row: ``|c|`` or ``max(g, 0)``.

    Args:
        equality: Boolean mask of the equality rows.
        values: Row values.

    Returns:
        The non-negative violations, same shape as ``values``.
    """
    return np.where(equality, np.abs(values), np.maximum(values, 0.0))


@dataclass(frozen=True)
class ConstraintRow:
    """One signed scalar row of a :class:`Constraint`.

    Attributes:
        kind: ``"equality"`` (``c = 0``) or ``"inequality"`` (``g <= 0``).
        sign: ``-1`` for a lower bound, else ``+1``.
        operand: Operand whose target is the limit; its residual is
            ``ew * (value - limit)``.
        tol: Feasibility tolerance of the row, in row units.
        label: Human-readable description.
    """

    kind: str
    sign: float
    operand: Operand
    tol: float
    label: str

    def residual(self) -> Any:
        """Row value at the current optic state, ``sign * ew * delta``."""
        weight = self.operand.effective_weight()
        if weight == 0.0:
            return be.array(0.0)
        value = self.operand.value
        if be.size(value) == 1:
            value = be.ravel(value)[0]
        return self.sign * weight * (value - self.operand.target)


@dataclass
class Constraint:
    """A hard equality or inequality built from an operand spec. *Experimental.*

    Args:
        operand_type: Registered operand name, e.g. ``"f2"``.
        target: Equality target; excludes ``min_val`` and ``max_val``.
        min_val: Lower bound (``value >= min_val``).
        max_val: Upper bound (``value <= max_val``).
        weight: Positive row scale, the operand weight of ``ew * delta``.
        input_data: Operand input data.
        tol: Feasibility tolerance of each row, in row units.

    Attributes:
        rows: The signed rows of this constraint.

    Raises:
        ConfigurationError: If the operand is unknown, the limits are missing,
            non-finite or inconsistent, or ``weight`` or ``tol`` is not
            positive and finite.
    """

    operand_type: str | None = None
    target: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    weight: float = 1.0
    input_data: dict | None = None
    tol: float = 1e-8
    rows: list[ConstraintRow] = field(init=False)

    def __post_init__(self) -> None:
        name = self.operand_type
        if self.input_data is None:
            self.input_data = {}
        if name not in operand_registry:
            raise ConfigurationError(f"Unknown operand type {name!r}.")
        if self.target is not None and (
            self.min_val is not None or self.max_val is not None
        ):
            raise ConfigurationError(
                f"{name} constraint cannot mix a target with min_val/max_val."
            )
        limits = [
            limit
            for limit in (self.target, self.min_val, self.max_val)
            if limit is not None
        ]
        if not limits:
            raise ConfigurationError(f"{name} constraint needs a target or a bound.")
        if not all(math.isfinite(float(limit)) for limit in limits):
            raise ConfigurationError(f"{name} constraint limits must be finite.")
        if (
            self.min_val is not None
            and self.max_val is not None
            and self.min_val > self.max_val
        ):
            raise ConfigurationError(f"{name} constraint: min_val exceeds max_val.")
        if not (math.isfinite(self.weight) and self.weight > 0):
            raise ConfigurationError(f"{name} constraint weight must be positive.")
        if not (math.isfinite(self.tol) and self.tol > 0):
            raise ConfigurationError(f"{name} constraint tol must be positive.")
        self.rows = self._build_rows()

    def _build_rows(self) -> list[ConstraintRow]:
        def row(kind: str, sign: float, limit: float, relation: str) -> ConstraintRow:
            operand = Operand(
                operand_type=self.operand_type,
                target=float(limit),
                weight=self.weight,
                input_data=self.input_data,
            )
            label = f"{self.operand_type} {relation} {float(limit):g}"
            return ConstraintRow(kind, sign, operand, self.tol, label)

        if self.target is not None:
            return [row(EQUALITY, 1.0, self.target, "==")]
        rows = []
        if self.min_val is not None:
            rows.append(row(INEQUALITY, -1.0, self.min_val, ">="))
        if self.max_val is not None:
            rows.append(row(INEQUALITY, 1.0, self.max_val, "<="))
        return rows

    @property
    def value(self) -> Any:
        """Operand value at the current optic state."""
        return self.rows[0].operand.value
