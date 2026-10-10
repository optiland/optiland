"""GUI ownership and explicit application of detached scientific script results."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QObject, Signal, Slot

from optiland_gui.services.calculation_jobs import CalculationJobs
from optiland_gui.services.job_records import BackendConfig, OpticSnapshot
from optiland_gui.services.scientific_tasks import SCRIPT_BYTES


class ScientificScripts(QObject):
    """Run outside Qt, then offer an owned candidate for an explicit guarded Apply."""

    status_changed = Signal(str, bool)
    output_changed = Signal(str)
    candidate_changed = Signal(bool)
    commands_ready = Signal(object)

    def __init__(self, connector):
        super().__init__(connector)
        self.connector = connector
        self._restarting = False
        self._create_jobs()
        self.request = None
        self.candidate = None
        self._edit_token = None
        self._output = ""
        connector.document_state.committed.connect(self._document_changed)
        connector.document_state.changed.connect(self._document_changed)

    def _create_jobs(self):
        self.jobs = CalculationJobs(self.connector.document_state, self, max_pending=1)
        self.connector.register_calculation_service(self.jobs)
        self.jobs.progress.connect(self._progress)
        self.jobs.finished.connect(self._finished)

    def run(self, code):
        if getattr(self.connector, "_calculation_shutdown_started", False):
            return None
        if self._restarting:
            self.status_changed.emit(
                "Wait for the scientific process to restart.", True
            )
            return None
        self.cancel()
        self.request = None
        try:
            if len(code.encode("utf-8")) > SCRIPT_BYTES:
                raise ValueError("Scientific scripts are limited to 1 MiB.")
            snapshot = OpticSnapshot.capture(self.connector.get_optic())
            self._edit_token = self.connector.document_state.edit_token
            self.request = self.jobs.submit(
                "scientific-script",
                "optiland_gui.services.scientific_tasks:execute_scientific",
                snapshot,
                {"code": code},
                cancel_on_document_change=False,
                context=self._edit_token,
            )
        except Exception as exc:
            self.status_changed.emit(f"Unable to run: {exc}", False)
            return None
        self._output = ""
        self.output_changed.emit("")
        self.status_changed.emit("Running scientific script…", True)
        return self.request

    @Slot()
    def cancel(self):
        self.candidate = None
        self.candidate_changed.emit(False)
        self.jobs.cancel_target("scientific-script")

    @Slot()
    def restart(self):
        if self._restarting:
            return
        self._restarting = True
        self.cancel()
        self.status_changed.emit("Restarting scientific process…", True)
        self.jobs.stopped.connect(self._restart_finished)
        self.jobs.shutdown()

    @Slot()
    def _restart_finished(self):
        old = self.jobs
        self.connector.calculation_services.remove(old)
        self.request = None
        self._restarting = False
        if not getattr(self.connector, "_calculation_shutdown_started", False):
            self._create_jobs()
            self.status_changed.emit("Scientific process reset. Ready to run.", False)
        old.deleteLater()

    def can_apply(self):
        return bool(
            self.candidate is not None
            and not getattr(self.connector, "_calculation_shutdown_started", False)
            and self.request is not None
            and self.jobs.is_current(self.request)
            and self.request.context == self.connector.document_state.edit_token
        )

    @Slot()
    def apply_candidate(self):
        if not self.can_apply():
            self.status_changed.emit(
                "Document changed; rerun the script before applying its result.", False
            )
            return False
        try:
            candidate = replace(
                self.candidate, backend=BackendConfig.capture()
            ).restore()
            undo = self.connector._capture_optic_state()
            if not self.can_apply():
                return False
            self.connector._undo_redo_manager.add_state(undo)
            self.connector._optic = candidate
            self.connector.set_modified(True)
            self.candidate = None
            self.connector.notify_change("replacement")
            self.candidate_changed.emit(False)
            self.status_changed.emit("Script result applied. Undo is available.", False)
            return True
        except Exception as exc:
            self.status_changed.emit(f"Unable to apply script result: {exc}", False)
            return False

    @Slot(object)
    def _document_changed(self, change):
        self.candidate_changed.emit(self.can_apply())

    @Slot(object, dict)
    def _progress(self, request, message):
        if self.request is None or request.job_id != self.request.job_id:
            return
        details = message.get("details") or {}
        if "output" in details:
            self._output = details["output"]
            self.output_changed.emit(self._output)
        self.status_changed.emit(message["stage"], True)

    @Slot(object)
    def _finished(self, result):
        if self.request is None or result.request.job_id != self.request.job_id:
            return
        if result.status != "succeeded":
            if result.status == "failed":
                self._output += "\n" + result.error[:16000]
                self.output_changed.emit(self._output)
            self.status_changed.emit(
                "Script cancelled."
                if result.status == "cancelled"
                else "Script failed. See scientific output for details.",
                False,
            )
            return
        self._output = result.data["output"] + result.data["error"]
        self.output_changed.emit(self._output)
        self.candidate = result.data["snapshot"]
        applicable = self.can_apply()
        self.candidate_changed.emit(applicable)
        self.status_changed.emit(
            "Script failed; the current document was preserved."
            if result.data["error"]
            else "Result ready. Apply Script Result to update the document."
            if applicable
            else "Result is from an earlier document revision; rerun to apply.",
            False,
        )
        if applicable:
            # The receiving GUI slot rechecks currency before each command.
            self.commands_ready.emit(result.data["commands"])
