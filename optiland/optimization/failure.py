"""Failure Policy Module

The shared non-finite / trace-failure policy of the optimization evaluators
(opt v2 Section 8.4):

* A ray failure or an exception inside one design evaluation yields ``NaN`` in
  the affected residual rows, never an exception out of the evaluator. The
  batched evaluator already isolates failed traces to their operands; an
  exception from the whole evaluation turns every row into ``NaN``.
* Native stepped optimizers treat a non-finite trial as a rejected trial.
* ``on_failure="raise"`` raises :class:`OptimizationFailure` instead, for
  users who want to stop at the first failure.
* A non-finite *initial* point raises :class:`ConfigurationError` naming the
  failing operands: an untraceable design cannot be optimized.

SciPy adapters cannot consume ``NaN`` and keep their scale-aware penalty
(opt v2 PR1.1).

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem

#: Valid ``on_failure`` policies of the native evaluators.
FAILURE_MODES: tuple[str, ...] = ("reject", "raise")

#: Scale of the stand-in value SciPy adapters use for a failed evaluation,
#: relative to the last finite residual norm (floored at 1).
PENALTY_FACTOR = 10.0


class ConfigurationError(ValueError):
    """The optimization cannot start from the given configuration."""


class OptimizationFailure(RuntimeError):
    """A design evaluation failed under ``on_failure="raise"``."""


def normalize_failure_mode(mode: str | None) -> str:
    """Validate an ``on_failure`` policy.

    Args:
        mode: ``"reject"`` or ``"raise"`` (case-insensitive), or None for the
            default ``"reject"``.

    Returns:
        The lowercase policy.

    Raises:
        ConfigurationError: If ``mode`` is not a known policy.
    """
    if mode is None:
        return "reject"
    norm = str(mode).lower()
    if norm not in FAILURE_MODES:
        raise ConfigurationError(
            f"Unknown on_failure policy {mode!r}; expected one of "
            f"{list(FAILURE_MODES)}."
        )
    return norm


def failing_rows(residuals: Any) -> np.ndarray:
    """Indices of the non-finite entries of a residual vector."""
    return np.flatnonzero(~np.isfinite(np.asarray(residuals, dtype=float)))


def describe_rows(problem: OptimizationProblem, rows: Any) -> str:
    """Name the operands behind residual rows, for error messages.

    Rows follow ``OptimizationProblem.residuals_and_constraints``: one row per
    operand, in order, then the constraint rows.

    Args:
        problem: The optimization problem.
        rows: Residual row indices.

    Returns:
        A comma-separated list such as ``"#3 rms_spot_size, #4 f2"``.
    """
    operands = [*problem.operands, *problem.constraints.row_operands]
    names = []
    for row in rows:
        row = int(row)
        if row < len(operands):
            names.append(f"#{row} {operands[row].operand_type}")
        else:
            names.append(f"#{row}")
    return ", ".join(names)


def check_initial_point(problem: OptimizationProblem, residuals: Any) -> None:
    """Raise when the starting design has non-finite residuals.

    Args:
        problem: The optimization problem.
        residuals: The residual vector at the starting design.

    Raises:
        ConfigurationError: If any residual is non-finite.
    """
    rows = failing_rows(residuals)
    if rows.size:
        raise ConfigurationError(
            "Cannot optimize from an untraceable design: the starting point "
            f"has non-finite residuals for operand(s) {describe_rows(problem, rows)}."
        )


def reject_hard_constraints(problem: OptimizationProblem, solver: str) -> None:
    """Raise when a solver that ignores hard constraints meets one.

    Args:
        problem: The optimization problem.
        solver: The solver's name, for the message.

    Raises:
        ConfigurationError: If the problem has hard constraints.
    """
    if len(getattr(problem, "constraints", ())):
        raise ConfigurationError(
            f"{solver} does not support hard constraints. Use minimize(problem, "
            "method='dls') or LeastSquares.optimize(method_choice='dls'), or "
            "express the limits as penalty operands with add_operand(min_val=..., "
            "max_val=...), which work with every optimizer."
        )


def penalty_residual(reference_norm: float, n: int) -> float:
    """Per-row stand-in residual for rows that could not be evaluated.

    Scaled to the last finite residual norm (the PR1.1 rule of the legacy
    ``LeastSquares``), never a hardcoded constant.

    Args:
        reference_norm: The last finite residual norm.
        n: Number of residual rows.

    Returns:
        ``PENALTY_FACTOR * max(reference_norm, 1) / sqrt(n)``.
    """
    return PENALTY_FACTOR * max(float(reference_norm), 1.0) / math.sqrt(max(n, 1))


def penalty_merit(reference_norm: float) -> float:
    """Stand-in merit for a failed evaluation: the squared penalty norm.

    Args:
        reference_norm: The last finite residual norm.

    Returns:
        ``(PENALTY_FACTOR * max(reference_norm, 1)) ** 2``, the sum of squares
        of the :func:`penalty_residual` rows.
    """
    return (PENALTY_FACTOR * max(float(reference_norm), 1.0)) ** 2
