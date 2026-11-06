"""Reconstruct history candidates in the calculation process."""

from __future__ import annotations

from optiland_gui.services.job_records import check_cancelled
from optiland_gui.services.prepared_optic import PreparedOptic


def prepare_history(snapshot, parameters, progress, cancelled):
    """Return an owned model reconstructed exclusively from prescription records."""
    progress("Preparing history")
    optic = snapshot.restore()
    check_cancelled(cancelled)
    optic.updater.update()
    check_cancelled(cancelled)
    return PreparedOptic.capture(optic)
