"""GUI-owned operand-preview generations over the shared calculation service."""

from __future__ import annotations

import pickle
import uuid
from collections import defaultdict, deque

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from .job_records import BackendConfig, OpticSnapshot
from .operand_preview_worker import MAX_OPERAND_ROWS


def definition_key(definitions):
    """Compare captured definitions without retaining mutable script dictionaries."""
    return pickle.dumps(definitions, protocol=5)


class OperandPreviews(QObject):
    """Keep old values labelled while replacing obsolete preview work."""

    rowsChanged = Signal(list)
    statusChanged = Signal(str, int, int)

    def __init__(self, connector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.jobs = connector.calculation_jobs
        self.target = f"operand-preview:{uuid.uuid4().hex}"
        self.rows = []
        self._definitions = []
        self._key = definition_key([])
        self._request = None
        self._visible = False
        self._explicit = False
        self._dirty = True
        self._backend = None
        self._token = connector.document_state.token
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(75)
        self._timer.timeout.connect(self._submit)
        self.jobs.progress.connect(self._on_progress)
        self.jobs.finished.connect(self._on_finished)
        self.jobs.state_changed.connect(self._on_state)
        connector.document_state.changed.connect(self._document_changed)
        # Destruction revokes this target, without a callback into a dead widget.
        jobs, target = self.jobs, self.target
        self.destroyed.connect(lambda: jobs.cancel_target(target))

    def refresh(self, *, explicit=False):
        definitions = self.connector.get_optimization_operands()
        key = definition_key(definitions)
        if key != self._key:
            previous = defaultdict(deque)
            for definition, row in zip(self._definitions, self.rows, strict=True):
                previous[definition_key(definition)].append(row)
            self._definitions = pickle.loads(key)
            self._key = key
            self.rows = []
            for definition in self._definitions:
                matches = previous[definition_key(definition)]
                row = (
                    dict(matches.popleft())
                    if matches
                    else {"value": None, "error": "", "state": "stale"}
                )
                self.rows.append(row)
            self._dirty = True
            self._cancel()
            self._mark_stale()
        if explicit:
            self._dirty = True
            self._explicit = True
            self._cancel()
            self._mark_stale()
        if self._backend != BackendConfig.capture():
            self._dirty = True
        if self._request is not None:
            return
        if self._dirty and (self._visible or self._explicit):
            self.jobs.set_target_visible(self.target, True)
            self._timer.start()
            self.statusChanged.emit(
                "Waiting to update operand values", 0, len(self.rows)
            )

    @property
    def busy(self):
        return self._request is not None or self._timer.isActive()

    def set_visible(self, visible):
        self._visible = visible
        if not visible:
            self._explicit = False
            self._cancel()
            self.jobs.set_target_visible(self.target, False)
            if self._dirty:
                self._mark_stale()
                self.statusChanged.emit(
                    "Operand values need refresh", 0, len(self.rows)
                )
        else:
            self.jobs.set_target_visible(self.target, True)
            self.refresh()

    def stop(self):
        self._explicit = False
        self._cancel()
        self._mark_stale()
        self.statusChanged.emit("Operand update cancelled", 0, len(self.rows))

    def _cancel(self):
        self._timer.stop()
        if self._request is not None:
            self._dirty = True
        self._request = None
        self.jobs.cancel_target(self.target)

    def _mark_stale(self):
        for row in self.rows:
            row["state"] = "stale"
        self.rowsChanged.emit(list(range(len(self.rows))))

    @Slot(object)
    def _document_changed(self, token):
        replaced = token.document_id != self._token.document_id
        self._token = token
        self._cancel()
        self._dirty = True
        if replaced:
            for row in self.rows:
                row.update(value=None, error="")
        self._mark_stale()
        self.refresh()

    @Slot()
    def _submit(self):
        if not (self._visible or self._explicit) or not self._dirty:
            return
        if definition_key(self.connector.get_optimization_operands()) != self._key:
            self.refresh()
            return
        optic = self.connector.get_optic()
        if optic is None or not self.rows:
            self._dirty = False
            self.statusChanged.emit("No operand values to calculate", 0, 0)
            return
        try:
            if len(self.rows) > MAX_OPERAND_ROWS:
                raise ValueError(
                    f"Operand preview supports at most {MAX_OPERAND_ROWS} rows."
                )
            snapshot = OpticSnapshot.capture(optic)
            self._backend = snapshot.backend
            metadata = self.connector._optimization_service.OPERAND_METADATA
            self._request = self.jobs.submit(
                self.target,
                "optiland_gui.services.operand_preview_worker:prepare_operand_values",
                snapshot,
                {
                    "operands": self._definitions,
                    "operand_metadata": metadata,
                },
                context=self._key,
            )
        except Exception as error:
            self._fail(str(error))
            return
        for row in self.rows:
            row["state"] = "pending"
        self.rowsChanged.emit(list(range(len(self.rows))))
        self.statusChanged.emit("Operand values queued", 0, len(self.rows))

    def _accepts(self, request):
        return (
            self._request is not None
            and request.job_id == self._request.job_id
            and request.context == self._key
            and self.jobs.is_current(request)
            and definition_key(self.connector.get_optimization_operands()) == self._key
        )

    def _install(self, updates):
        changed = []
        for update in updates:
            index = update["index"]
            if not 0 <= index < len(self.rows):
                continue
            row = self.rows[index]
            error = update["error"]
            row["error"] = error
            row["state"] = "error" if error else "current"
            if not error:
                row["value"] = update["value"]
            changed.append(index)
        self.rowsChanged.emit(changed)

    @Slot(object, dict)
    def _on_progress(self, request, message):
        if self._accepts(request):
            self._install((message.get("details") or {}).get("rows", []))
            self.statusChanged.emit(
                message.get("stage", "Calculating operand values"),
                message.get("completed") or 0,
                message.get("total") or len(self.rows),
            )

    @Slot(object, str)
    def _on_state(self, request, state):
        if self._accepts(request) and state == "running":
            self.statusChanged.emit("Calculating operand values", 0, len(self.rows))

    @Slot(object)
    def _on_finished(self, result):
        if self._request is None or result.request.job_id != self._request.job_id:
            return
        accepted = result.current and self._accepts(result.request)
        self._request = None
        self._explicit = False
        if not accepted:
            self._dirty = True
            self._mark_stale()
            self.statusChanged.emit(
                "Operand values are out of date; refresh to update", 0, len(self.rows)
            )
            self.refresh()
            return
        if result.status != "succeeded":
            self._fail(
                "Update cancelled" if result.status == "cancelled" else result.error
            )
            return
        self._install(result.data["rows"])
        self._dirty = False
        errors = sum(bool(row["error"]) for row in self.rows)
        status = (
            f"Operand values updated; {errors} failed"
            if errors
            else "Operand values up to date"
        )
        self.statusChanged.emit(status, len(self.rows), len(self.rows))

    def _fail(self, error):
        error = error[:4000]
        for row in self.rows:
            row.update(error=error, state="error")
        self._dirty = False
        self._explicit = False
        self.rowsChanged.emit(list(range(len(self.rows))))
        summary = error.splitlines()[-1] if error else "Unknown error"
        self.statusChanged.emit(
            f"Operand update failed: {summary}",
            0,
            len(self.rows),
        )
