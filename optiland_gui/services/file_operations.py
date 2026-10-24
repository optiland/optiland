"""Owned asynchronous file loading and guarded candidate installation."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from optiland_gui.services.job_records import BackendConfig
from optiland_gui.services.prepared_optic import PreparedOptic


@dataclass
class LoadOperation:
    """Identity and document revision belonging to one explicit load."""

    identifier: str
    path: str
    file_format: str
    edit_token: object


class FileOperations(QObject):
    """Prepare file candidates without replacing edits until acceptance."""

    state_changed = Signal(str, bool, bool)
    candidate_conflict = Signal(str, str)
    completed = Signal()

    def __init__(self, service, connector):
        super().__init__(connector)
        self.service, self.connector = service, connector
        self.jobs = connector.calculation_jobs
        self._operations = {}
        self._candidate = None
        self._latest_open = None
        self._closing = False
        self.jobs.finished.connect(self._finished)
        self.jobs.progress.connect(self._progress)

    @property
    def busy(self):
        return bool(self._operations)

    def request_load(self, path, file_format=None, *, snapshot=None, load_options=None):
        if self._closing:
            return None
        self.cancel_pending()
        operation = LoadOperation(
            uuid.uuid4().hex,
            os.path.abspath(path),
            file_format
            or ("zemax" if str(path).lower().endswith(".zmx") else "optiland"),
            self.connector.document_state.edit_token,
        )
        self._latest_open = operation.identifier
        self._operations[operation.identifier] = operation
        try:
            request = self.jobs.submit(
                "file-open",
                "optiland_gui.services.file_tasks:load_file",
                snapshot,
                {
                    "path": operation.path,
                    "format": operation.file_format,
                    "backend": BackendConfig.capture(),
                    "load_options": dict(load_options or {}),
                },
                cancel_on_document_change=False,
                context=operation,
            )
        except Exception as exc:
            self._operations.pop(operation.identifier, None)
            self._notify(f"File operation could not start: {exc}", "error")
            self._changed()
            return None
        self._changed()
        return request

    @Slot()
    def cancel_pending(self):
        self._candidate = None
        self._latest_open = None
        self.jobs.cancel_target("file-open")

    @Slot(object, dict)
    def _progress(self, request, message):
        if getattr(request.context, "identifier", None) in self._operations:
            self._changed(message["stage"])

    @Slot(object)
    def _finished(self, result):
        operation = result.request.context
        identifier = getattr(operation, "identifier", None)
        if identifier not in self._operations:
            return
        self._operations.pop(identifier)
        if result.status == "failed":
            self._notify(f"File operation failed: {result.error}", "error")
        elif (
            result.status == "succeeded"
            and not self._closing
            and identifier == self._latest_open
        ):
            if operation.edit_token != self.connector.document_state.edit_token:
                self._candidate = operation, result.data
                self.candidate_conflict.emit(identifier, operation.path)
            else:
                self._accept(operation, result.data, operation.edit_token)
        self.completed.emit()
        self._changed()

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
            if not isinstance(snapshot, PreparedOptic):
                raise ValueError("Invalid prepared optical document.")
            # Install the owned worker model without repeating material lookup,
            # normalization or optical solves on the GUI thread.
            candidate = snapshot.restore()
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

    def _notify(self, text, severity):
        self.service._toast(text, severity)

    def _changed(self, stage=None):
        label = stage or (
            "Preparing optical file…" if self.busy else "File loading finished."
        )
        self.state_changed.emit(label, self.busy, self.busy)
