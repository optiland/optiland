"""GUI-owned file transactions using the shared calculation lifecycle."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from optiland_gui.services.job_records import BackendConfig, OpticSnapshot


@dataclass
class FileOperation:
    """GUI-owned transaction identity, snapshot and publication state."""

    identifier: str
    kind: str
    path: str
    file_format: str
    target: str
    edit_token: object
    snapshot: OpticSnapshot | None = None
    phase: str = "preparing"
    staged_path: str | None = None
    staged_identity: tuple[int, int] | None = None
    digest: str | None = None
    load_options: dict = field(default_factory=dict)
    save_sequence: int = 0


class FileOperations(QObject):
    """Keep the current document and destination until an explicit commit point."""

    state_changed = Signal(str, bool, bool)
    candidate_conflict = Signal(str, str)
    completed = Signal()
    settled = Signal()
    close_aborted = Signal()

    def __init__(self, service, connector):
        super().__init__(connector)
        self.service, self.connector = service, connector
        self.jobs = connector.calculation_jobs
        self._operations = {}
        self._requests = {}
        self._candidate = None
        self._latest_open = None
        self._closing = False
        self._close_edit_token = None
        self._save_sequence = 0
        self._confirmed_save_sequence = {}
        self.jobs.finished.connect(self._finished)
        self.jobs.progress.connect(self._progress)
        self.jobs.transaction_progress.connect(self._transaction_progress)

    @property
    def busy(self):
        return bool(self._operations)

    @property
    def has_pending_writes(self):
        return any(operation.kind != "open" for operation in self._operations.values())

    def request_load(self, path, file_format=None, *, snapshot=None, load_options=None):
        if self._closing:
            return None
        self._candidate = None
        if file_format is None:
            file_format = "zemax" if str(path).lower().endswith(".zmx") else "optiland"
        operation = self._new_operation(
            "open", path, file_format, "file-open", snapshot
        )
        self._latest_open = operation.identifier
        operation.load_options = dict(load_options or {})
        return self._queue(operation, "load_file")

    def request_output(self, path, file_format="optiland"):
        if self._closing:
            return None
        try:
            snapshot = OpticSnapshot.capture(self.connector.get_optic())
        except Exception as exc:
            self._notify(f"Unable to capture document: {exc}", "error")
            return None
        path = os.path.abspath(path)
        target = "file-output-" + os.path.normcase(path)
        kind = "save" if file_format == "optiland" else "export"
        operation = self._new_operation(kind, path, file_format, target, snapshot)
        if kind == "save":
            operation.save_sequence = self.next_save_sequence()
        operation.staged_path = str(
            Path(path).with_name(f".optiland-{operation.identifier}.tmp")
        )
        return self._queue(operation, "prepare_output")

    def path_busy(self, path):
        path = os.path.normcase(os.path.abspath(path))
        return any(
            op.kind != "open" and os.path.normcase(op.path) == path
            for op in self._operations.values()
        )

    def _new_operation(self, kind, path, file_format, target, snapshot):
        operation = FileOperation(
            uuid.uuid4().hex,
            kind,
            os.path.abspath(path),
            file_format,
            target,
            self.connector.document_state.edit_token,
            snapshot,
        )
        self._operations[operation.identifier] = operation
        return operation

    def _queue(self, operation, handler):
        if operation.identifier not in self._operations:
            return None
        parameters = {
            "path": operation.path,
            "format": operation.file_format,
            "staged_path": operation.staged_path,
            "staged_identity": operation.staged_identity,
            "digest": operation.digest,
            "backend": BackendConfig.capture(),
            "load_options": operation.load_options,
        }
        try:
            request = self.jobs.submit(
                operation.target
                if operation.phase == "preparing"
                else f"file-{operation.phase}-{operation.identifier}",
                f"optiland_gui.services.file_tasks:{handler}",
                operation.snapshot,
                parameters,
                replace=operation.phase == "preparing",
                cancel_on_document_change=False,
                cancellable=operation.phase == "preparing",
                context=operation,
            )
        except Exception as exc:
            if operation.phase != "preparing" and "queue is full" in str(exc):
                # A terminal preparation can need a publication/cleanup slot
                # while other callers filled the finite queue. Keep ownership
                # and retry asynchronously instead of abandoning a staged file.
                QTimer.singleShot(50, lambda: self._queue(operation, handler))
                self._changed()
                return None
            self._operations.pop(operation.identifier, None)
            if operation.kind != "open" and operation.phase != "cleaning":
                self._abort_close(f"The file operation for {operation.path} failed")
            self._notify(f"File operation could not start: {exc}", "error")
            self._changed()
            return None
        self._requests[request.job_id] = operation
        self._changed()
        return request

    @Slot()
    def cancel_pending(self):
        for operation in list(self._operations.values()):
            if operation.phase == "preparing":
                self.jobs.cancel_target(operation.target)

    def cancel_load(self):
        self._candidate = None
        self._latest_open = None
        self.jobs.cancel_target("file-open")

    def begin_close(self):
        """Finish accepted writes before shutdown; cancel unrelated calculations."""
        self._closing = True
        self._close_edit_token = self.connector.document_state.edit_token
        self._candidate = None
        writes = {
            operation.target
            for operation in self._operations.values()
            if operation.kind != "open" and operation.phase == "preparing"
        }
        self.jobs.cancel_cancellable(exclude_targets=writes)
        self._changed()

    def _abort_close(self, reason):
        if self._closing:
            self._closing = False
            self._notify(
                f"Close cancelled: {reason}. The document remains open.", "error"
            )
            self.close_aborted.emit()

    def confirm_close_revision(self):
        """Recheck the document before advancing from file drain to shutdown."""
        if self.connector.document_state.edit_token != self._close_edit_token:
            self._abort_close("The document changed while waiting for file operations")
            return False
        return self._closing

    def cancel_close(self):
        """Resume normal operation after cancelling the close intent."""
        self._closing = False
        self._close_edit_token = None
        self._changed()

    @Slot(object, dict)
    def _transaction_progress(self, request, message):
        operation = self._requests.get(request.job_id)
        identity = (message.get("details") or {}).get("staged_identity")
        if operation is not None and identity is not None:
            operation.staged_identity = tuple(identity)

    @Slot(object, dict)
    def _progress(self, request, message):
        if request.job_id in self._requests:
            self._changed(message["stage"])

    @Slot(object)
    def _finished(self, result):
        operation = self._requests.pop(result.request.job_id, None)
        if operation is None:
            return
        phase = operation.phase
        if phase == "cleaning":
            if result.status != "succeeded":
                self._notify(
                    f"Temporary-file cleanup failed: {operation.staged_path}", "warning"
                )
            self._complete(operation)
            return
        if phase == "publishing":
            if result.status == "succeeded" and not result.outcome_unknown:
                self._published(operation)
            else:
                operation.phase = "reconciling"
                self._queue(operation, "reconcile_output")
            return
        if phase == "reconciling":
            if result.status == "succeeded" and result.data["matches"]:
                self._published(operation)
            else:
                self._abort_close(
                    f"The save to {operation.path} could not be confirmed"
                )
                self._notify(
                    "Save outcome could not be confirmed. "
                    f"Check {operation.path} before retrying.",
                    "error",
                )
                self._cleanup(operation)
            return
        if result.status != "succeeded":
            if operation.kind != "open":
                self._abort_close(f"The write to {operation.path} was {result.status}")
            if result.status == "failed":
                self._notify(f"File operation failed: {result.error}", "error")
            self._cleanup(operation)
            return
        if operation.kind == "open":
            if self._closing or operation.identifier != self._latest_open:
                self._complete(operation)
            elif operation.edit_token != self.connector.document_state.edit_token:
                self._candidate = (operation, result.data)
                self._complete(operation)
                self.candidate_conflict.emit(operation.identifier, operation.path)
            else:
                self._accept(operation, result.data, operation.edit_token)
                self._complete(operation)
        else:
            operation.digest = result.data["digest"]
            operation.staged_identity = tuple(result.data["staged_identity"])
            operation.phase = "publishing"
            self._queue(operation, "publish_output")

    def resolve_candidate(self, identifier, accept, expected_edit_token):
        if self._candidate is None or self._candidate[0].identifier != identifier:
            return False
        operation, snapshot = self._candidate
        self._candidate = None
        if accept:
            return self._accept(operation, snapshot, expected_edit_token)
        return False

    def _accept(self, operation, snapshot, expected_edit_token):
        if (
            self._closing
            or operation.identifier != self._latest_open
            or self.connector.document_state.edit_token != expected_edit_token
        ):
            self._notify(
                "The current document changed again; "
                "the loaded candidate was not applied.",
                "warning",
            )
            return False
        try:
            # Reconstruction binds ordinary model objects; parsing, catalog
            # resolution, normalization and solves ran in the owned worker.
            candidate = replace(snapshot, backend=BackendConfig.capture()).restore()
            if self.connector.document_state.edit_token != expected_edit_token:
                return False
            native = operation.file_format == "optiland"
            self.service._publish_candidate(
                candidate,
                operation.path if native else None,
                modified=not native,
                validated=True,
            )
            self._notify(f"Opened — {Path(operation.path).name}", "info")
            return True
        except Exception as exc:
            self._notify(f"Unable to install loaded document: {exc}", "error")
            return False

    def _published(self, operation):
        if operation.kind == "save":
            self.confirm_saved_document(
                operation.path, operation.edit_token, operation.save_sequence
            )
        verb = "Saved" if operation.kind == "save" else "Exported"
        self._notify(f"{verb} — {Path(operation.path).name}", "success")
        self._cleanup(operation)

    def next_save_sequence(self):
        """Order native save choices across asynchronous and public sync paths."""
        self._save_sequence += 1
        return self._save_sequence

    def confirm_saved_document(self, path, edit_token, sequence):
        """A late older confirmation cannot change the active file or clean state."""
        token = self.connector.document_state.edit_token
        if (
            token.document_id == edit_token.document_id
            and sequence > self._confirmed_save_sequence.get(token.document_id, -1)
        ):
            self._confirmed_save_sequence[token.document_id] = sequence
            self.service._current_filepath = path
            if token == edit_token:
                self.connector.set_modified(False)

    def _cleanup(self, operation):
        if operation.staged_path is not None:
            operation.phase = "cleaning"
            self._queue(operation, "cleanup_output")
        else:
            self._complete(operation)

    def _complete(self, operation):
        self._operations.pop(operation.identifier, None)
        self.completed.emit()
        self._changed()

    def _notify(self, text, severity):
        self.service._toast(text, severity)

    def _changed(self, stage=None):
        cancellable = any(op.phase == "preparing" for op in self._operations.values())
        if self._operations:
            operation = next(reversed(self._operations.values()))
            label = stage or (
                "Finishing save…"
                if operation.phase in ("publishing", "reconciling")
                else "Cleaning up file operation…"
                if operation.phase == "cleaning"
                else "Preparing file operation…"
            )
            label += f" {Path(operation.path).name}"
        else:
            label = "File operations finished."
        if self._closing and self.busy:
            label = "Finishing requested file operations before closing. " + label
        self.state_changed.emit(label, self.busy, cancellable)
        if self._closing and not self.busy and self.confirm_close_revision():
            self.settled.emit()
