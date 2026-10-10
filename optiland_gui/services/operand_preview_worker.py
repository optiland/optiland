"""Evaluate a frozen operand list once per row on one owned optical model."""

from __future__ import annotations

import math
import time

from optiland.optimization.operand.operand import Operand

from .job_records import CalculationCancelled, check_cancelled
from .optimization_jobs import operand_arguments

MAX_OPERAND_ROWS = 10000


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
        if now - last_report >= 0.2 or index + 1 == len(definitions):
            progress(
                f"Calculated {index + 1} of {len(definitions)} operand values",
                index + 1,
                len(definitions),
                details={"rows": pending},
            )
            pending = []
            last_report = now
    return {"rows": rows}
