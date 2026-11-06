"""Evaluate a frozen operand list once per row on one owned optical model."""

from __future__ import annotations

import math
import time

import optiland.backend as be
from optiland.optimization import OptimizationProblem
from optiland.optimization.operand.operand import Operand

from .job_records import CalculationCancelled, check_cancelled
from .optimization_jobs import add_operands, operand_arguments

MAX_OPERAND_ROWS = 10000


def merit_shares(optic, definitions, operand_metadata) -> list[float]:
    """Return each operand's percentage of the merit ``sum(r ** 2)``.

    The residuals come from the same problem the optimizer builds, so the
    shares use its weights, bounds and batched evaluation.
    """
    problem = OptimizationProblem()
    add_operands(problem, optic, definitions, operand_metadata)
    squared = be.to_numpy(problem.residual_vector()) ** 2
    total = float(squared.sum())
    if not math.isfinite(total) or total <= 0.0:
        return [0.0] * len(definitions)
    return [float(100.0 * value / total) for value in squared]


def prepare_operand_values(snapshot, parameters, progress, cancelled):
    """Return plain value/error rows; partial batches carry the same job token."""
    definitions = parameters["operands"]
    if len(definitions) > MAX_OPERAND_ROWS:
        raise ValueError(f"Operand preview supports at most {MAX_OPERAND_ROWS} rows.")
    check_cancelled(cancelled)
    progress("Preparing operand values", 0, len(definitions))
    optic = snapshot.restore()
    optic.updater.update()
    rows, pending = [], []
    last_report = time.monotonic()
    metadata = parameters["operand_metadata"]
    for index, definition in enumerate(definitions):
        check_cancelled(cancelled)
        try:
            # An explicit target prevents Operand.__post_init__ evaluating the
            # metric to infer a default. The sole evaluation is .value below.
            operand = Operand(
                operand_type=definition["type"],
                target=0.0,
                input_data=operand_arguments(optic, definition, metadata),
            )
            value = float(operand.value)
            if not math.isfinite(value):
                raise ValueError("The calculation returned a non-finite value.")
            row = {"index": index, "value": value, "error": ""}
        except CalculationCancelled:
            raise
        except Exception as error:
            row = {
                "index": index,
                "value": None,
                "error": f"{type(error).__name__}: {error}"[:2000],
            }
        check_cancelled(cancelled)
        rows.append(row)
        pending.append(row)
        now = time.monotonic()
        if now - last_report >= 0.2 and index + 1 < len(definitions):
            progress(
                f"Calculated {index + 1} of {len(definitions)} operand values",
                index + 1,
                len(definitions),
                details={"rows": pending},
            )
            pending = []
            last_report = now
    # The merit is defined only when every operand evaluates.
    if rows and not any(row["error"] for row in rows):
        check_cancelled(cancelled)
        for row, share in zip(
            rows, merit_shares(optic, definitions, metadata), strict=True
        ):
            row["share"] = share
        pending = rows
    progress(
        f"Calculated {len(definitions)} of {len(definitions)} operand values",
        len(definitions),
        len(definitions),
        details={"rows": pending},
    )
    return {"rows": rows}
