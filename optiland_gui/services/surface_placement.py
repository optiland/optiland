"""Guarded, undoable surface-placement jobs owned by the current GUI document."""

from __future__ import annotations

import math

from PySide6.QtCore import QObject, Signal, Slot

from optiland_gui.services.placement_tasks import placement_values, validate_changes
from optiland_gui.services.prepared_optic import PreparedOptic


class SurfacePlacement(QObject):
    """Apply finite placement drafts only to the document that produced them."""

    state_changed = Signal()
    about_to_apply = Signal(object)
    target = "surface-placement"

    def __init__(self, connector):
        super().__init__(connector)
        self.connector = connector
        self.jobs = connector.calculation_jobs
        self.request = None
        self.busy = False
        self.message = ""
        self.jobs.finished.connect(self._finished)
        self.jobs.progress.connect(self._progress)

    def submit(self, surface, frame, changes, edit_token):
        if getattr(self.connector, "_calculation_shutdown_started", False):
            return False
        try:
            if edit_token != self.connector.document_state.edit_token:
                raise ValueError(
                    "The design changed. Reload placement before applying."
                )
            surfaces = tuple(self.connector.get_optic().surfaces)
            row = next((i for i, value in enumerate(surfaces) if value is surface), -1)
            if row < 0:
                raise ValueError(
                    "The surface is no longer available. Reload placement."
                )
            current = placement_values(surface, frame)
            changes = validate_changes(changes)
            if not all(math.isfinite(value) for value in (current | changes).values()):
                raise ValueError(
                    "Enter a finite position for this surface, including Z."
                )
            changes = {
                key: value for key, value in changes.items() if current[key] != value
            }
            if not changes:
                self.message = "Placement is unchanged."
                self.state_changed.emit()
                return False
            snapshot = PreparedOptic.capture(self.connector.get_optic())
            undo = self.connector._capture_optic_state()
            self.request = self.jobs.submit(
                self.target,
                "optiland_gui.services.placement_tasks:prepare_placement",
                snapshot,
                {"row": row, "frame": frame, "changes": changes},
                context=(edit_token, undo),
            )
            self.busy = True
            self.message = "Applying placement…"
            self.state_changed.emit()
            return True
        except Exception as exc:
            self.message = str(exc)
            self.state_changed.emit()
            return False

    @Slot()
    def cancel(self):
        self.jobs.cancel_target(self.target)

    @Slot(object, dict)
    def _progress(self, request, message):
        if self.request is not None and request.job_id == self.request.job_id:
            self.message = message["stage"]
            self.state_changed.emit()

    @Slot(object)
    def _finished(self, result):
        if self.request is None or result.request.job_id != self.request.job_id:
            return
        request, self.request = self.request, None
        self.busy = False
        self.message = "Placement cancelled. The design was preserved."
        if result.status == "succeeded":
            token, undo = request.context
            if (
                not result.current
                or not self.jobs.is_current(request)
                or (token != self.connector.document_state.edit_token)
            ):
                self.message = "The design changed. Reload placement before applying."
            else:
                try:
                    if not isinstance(result.data, PreparedOptic):
                        raise ValueError("Invalid placement result.")
                    candidate = result.data.restore()
                    # Only index-preserving placement candidates use this rebind.
                    if len(candidate.surfaces) != len(
                        self.connector.get_optic().surfaces
                    ):
                        raise ValueError("Placement changed the surface structure.")
                    self.about_to_apply.emit(candidate)
                    self.connector._undo_redo_manager.add_state(undo)
                    self.connector._optic = candidate
                    self.message = "Placement applied. Undo is available."
                    self.connector.set_modified(True)
                    self.connector.notify_change("replacement")
                except Exception as exc:
                    self.message = f"Placement failed: {exc}"
        elif result.status == "failed":
            lines = [line.strip() for line in result.error.splitlines() if line.strip()]
            self.message = "Placement failed: " + (
                lines[-1][:1000] if lines else "unknown error"
            )
        self.state_changed.emit()
