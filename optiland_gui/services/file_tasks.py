"""Parse and initialize owned optical files in the calculation worker."""

from __future__ import annotations

import json

from optiland.fileio import load_codev_file, load_zemax_file
from optiland.optic import Optic
from optiland_gui.services.job_records import check_cancelled
from optiland_gui.services.model_initialization import initialize_loaded_optic
from optiland_gui.services.prepared_optic import PreparedOptic


def load_file(snapshot, parameters, progress, cancelled):
    """Parse and validate a candidate without touching the displayed document."""
    from optiland_gui.services.file_service import json_inf_nan_hook

    parameters["backend"].apply()
    progress("Reading optical file")
    file_format = parameters["format"]
    if file_format == "optiland":
        with open(parameters["path"], encoding="utf-8") as stream:
            candidate = Optic.from_dict(
                json.load(stream, object_hook=json_inf_nan_hook)
            )
    elif file_format == "zemax":
        candidate = load_zemax_file(parameters["path"])
    elif file_format == "codev":
        candidate = load_codev_file(parameters["path"])
    elif file_format == "object":
        candidate = snapshot.restore()
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    check_cancelled(cancelled)
    progress("Validating optical document")
    initialize_loaded_optic(candidate)
    check_cancelled(cancelled)
    return PreparedOptic.capture(candidate)
