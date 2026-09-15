"""GUI-owned, bounded calculation queue with asynchronous process teardown."""

from __future__ import annotations

import pickle
import struct
import sys
import uuid
from collections import deque
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal, Slot

from optiland_gui.services.calculation_worker import MAX_MESSAGE_BYTES, encode_message
from optiland_gui.services.document_changes import (
    CHANGE_CATEGORIES,
    OPTICAL_CATEGORIES,
    DocumentChange,
)
from optiland_gui.services.job_records import (
    BackendConfig,
    DocumentToken,
    JobRequest,
    JobResult,
)

if TYPE_CHECKING:
    from optiland_gui.services.job_records import OpticSnapshot


class DocumentState(QObject):
    """Separate calculation validity from whole-document ownership.

    ``token`` changes for optical inputs; ``edit_token`` changes for every
    persisted edit, including metadata. Whole-document consumers must capture
    ``edit_token`` alongside their detached snapshot and compare it immediately
    before replacing the document or declaring a saved snapshot clean. Optical
    job currency alone cannot authorize overwriting intervening comment edits.
    """

    changed = Signal(object)
    committed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.token = DocumentToken(uuid.uuid4().hex, 0)
        self.edit_token = self.token
        self._transaction_depth = 0
        self._replacement_transaction = False
        self._invalidated = False
        self._edit_invalidated = False
        self._categories = set()
        self._surface_indices = set()
        self._columns = set()
        self._all_surfaces = False
        self._all_columns = False
        self._transaction_failed = False

    @Slot()
    def change(self) -> None:
        self.record("optical")

    @Slot()
    def replace(self) -> None:
        self.record("replacement")

    def record(self, category="optical", *, surface_indices=(), columns=()):
        """Invalidate immediately, then publish a classified committed change.

        Optical invalidation is never delayed until a repaint/coalescing timer.
        Metadata and presentation leave worker result permissions unchanged.
        """
        if category not in CHANGE_CATEGORIES:
            raise ValueError(f"Unknown document change category: {category}")
        replacement = category == "replacement" or self._replacement_transaction
        if (
            category == "replacement"
            and self._edit_invalidated
            and not self._replacement_transaction
        ):
            raise ValueError(
                "Declare replacement=True before a replacement transaction."
            )
        if category != "presentation" and not self._edit_invalidated:
            self.edit_token = (
                DocumentToken(uuid.uuid4().hex, 0)
                if replacement
                else DocumentToken(
                    self.edit_token.document_id, self.edit_token.revision + 1
                )
            )
            self._edit_invalidated = bool(self._transaction_depth)
        if category in OPTICAL_CATEGORIES and not self._invalidated:
            self.token = (
                DocumentToken(self.edit_token.document_id, 0)
                if replacement
                else DocumentToken(self.token.document_id, self.token.revision + 1)
            )
            self._invalidated = bool(self._transaction_depth)
            self.changed.emit(self.token)
        self._categories.add(category)
        surface_indices, columns = tuple(surface_indices), tuple(columns)
        self._surface_indices.update(surface_indices)
        self._columns.update(columns)
        if category not in {"presentation", "polarization"}:
            self._all_surfaces |= not surface_indices
            self._all_columns |= not columns
        if not self._transaction_depth:
            self._publish()

    @contextmanager
    def transaction(self, *, replacement=False):
        """Group successful notifications; callers own atomic mutation/rollback.

        A failed transaction publishes no successful change. Its already revoked
        worker permissions stay revoked. Callers must restore mutated state or
        publish a prepared replacement atomically before leaving this boundary.
        """
        if (
            self._transaction_depth
            and replacement
            and not self._replacement_transaction
        ):
            raise ValueError("A replacement must be declared on the outer transaction.")
        outer = not self._transaction_depth
        if outer:
            self._replacement_transaction = replacement
            self._transaction_failed = False
        self._transaction_depth += 1
        try:
            yield
        except Exception:
            self._transaction_failed = True
            if outer:
                self._reset_pending()
            raise
        finally:
            self._transaction_depth -= 1
            if outer:
                self._replacement_transaction = False
                self._invalidated = False
                self._edit_invalidated = False
                if self._transaction_failed:
                    self._reset_pending()
                else:
                    self._publish()

    def _publish(self):
        if self._categories:
            change = DocumentChange(
                self.token,
                self.edit_token,
                frozenset(self._categories),
                frozenset() if self._all_surfaces else frozenset(self._surface_indices),
                frozenset() if self._all_columns else frozenset(self._columns),
            )
            self._reset_pending()
            self.committed.emit(change)

    def _reset_pending(self):
        self._categories.clear()
        self._surface_indices.clear()
        self._columns.clear()
        self._all_surfaces = False
        self._all_columns = False


class CalculationJobs(QObject):
    """One persistent process and a finite queue of detached calculations.

    Signals are emitted by this GUI-owned QObject. Receivers that access widgets
    must be GUI-owned QObject slots. Every submitted job receives one terminal
    result, including replaced pending work. Recheck ``is_current(request)``
    immediately before presenting or committing; ``result.current`` is only the
    signal-time hint. Consumers may keep obsolete data separately.
    """

    state_changed = Signal(object, str)
    progress = Signal(object, dict)
    finished = Signal(object)
    stopped = Signal()

    def __init__(
        self,
        document: DocumentState,
        parent: QObject | None = None,
        *,
        max_pending: int = 16,
        cancel_grace_ms: int = 1000,
        worker_command: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.document = document
        self.max_pending = max_pending
        self._cancel_grace_ms = cancel_grace_ms
        self._command = worker_command or [
            sys.executable,
            "-u",
            "-m",
            "optiland_gui.services.calculation_worker",
        ]
        self._pending: deque[JobRequest] = deque()
        self._active: JobRequest | None = None
        self._process: QProcess | None = None
        self._ready = False
        self._buffer = bytearray()
        self._stderr = ""
        self._generations: dict[str, int] = {}
        self._visible: dict[str, bool] = {}
        self._serial = 0
        self._closed = False
        self._cancelling = False
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill_worker)
        document.changed.connect(self._document_changed)

    @property
    def running(self) -> bool:
        return self._active is not None or bool(self._pending)

    @property
    def active_request(self) -> JobRequest | None:
        return self._active

    def submit(
        self,
        target: str,
        handler: str,
        snapshot: OpticSnapshot | None,
        parameters: dict,
        *,
        replace: bool = True,
        cancel_on_document_change: bool = True,
        context: Any = None,
    ) -> JobRequest:
        """Queue detached inputs; replace previews while keeping explicit FIFO jobs."""
        if self._closed:
            raise RuntimeError("Calculation service is closed.")
        if replace:
            self.cancel_target(target)
        if len(self._pending) >= self.max_pending:
            raise RuntimeError(
                "Calculation queue is full; wait or cancel a pending job."
            )
        self._serial += 1
        generation = self._generations.setdefault(target, 0)
        request = JobRequest(
            self._serial,
            self.document.token,
            target,
            generation,
            handler,
            snapshot,
            pickle.loads(pickle.dumps(parameters, protocol=5)),
            cancel_on_document_change,
            context,
        )
        self._pending.append(request)
        self.state_changed.emit(request, "queued")
        QTimer.singleShot(0, self._dispatch)
        return request

    def is_current(self, request: JobRequest) -> bool:
        """Recheck immediately before presentation/commit, even after a result signal.

        Detached explicit jobs may finish against an older document; their data
        can be offered for review but never automatically overwrite current edits.
        """
        return (
            not self._closed
            and request.document == self.document.token
            and request.generation == self._generations.get(request.target, 0)
            and self._visible.get(request.target, True)
            and (
                request.snapshot is None
                or request.snapshot.backend == BackendConfig.capture()
            )
        )

    def set_target_visible(self, target: str, visible: bool) -> None:
        """Revoke a hidden target immediately; showing it allows fresh submissions."""
        self._visible[target] = visible
        if not visible:
            self.cancel_target(target)

    def cancel_target(self, target: str) -> None:
        self._generations[target] = self._generations.get(target, 0) + 1
        retained = deque()
        cancelled = []
        while self._pending:
            request = self._pending.popleft()
            (cancelled if request.target == target else retained).append(request)
        self._pending = retained
        for request in cancelled:
            self._terminal(request, "cancelled")
        if self._active is not None and self._active.target == target:
            self._request_cancel()

    @Slot(object)
    def _document_changed(self, token: DocumentToken) -> None:
        pending, self._pending = self._pending, deque()
        for request in pending:
            if self._closed or request.cancel_on_document_change:
                self._terminal(request, "cancelled")
            else:
                self._pending.append(request)
        if self._active is not None and (
            self._closed or self._active.cancel_on_document_change
        ):
            self._request_cancel()

    def _request_cancel(self) -> None:
        if self._active is None or self._cancelling:
            return
        self._cancelling = True
        self.state_changed.emit(self._active, "cancelling")
        self._send({"command": "cancel", "job_id": self._active.job_id})
        self._kill_timer.start(self._cancel_grace_ms)

    @Slot()
    def _dispatch(self) -> None:
        if self._closed or self._active is not None or not self._pending:
            return
        request = self._pending[0]
        eligible = (
            request.generation == self._generations.get(request.target, 0)
            and self._visible.get(request.target, True)
            and (not request.cancel_on_document_change or self.is_current(request))
        )
        if not eligible:
            self._terminal(self._pending.popleft(), "cancelled")
            QTimer.singleShot(0, self._dispatch)
            return
        if self._process is None:
            self._launch()
            return
        if not self._ready:
            return
        request = self._pending.popleft()
        self._active = request
        self._cancelling = False
        try:
            self._send(request.worker_message())
        except Exception as exc:
            self._active = None
            self._terminal(request, "failed", error=str(exc))
            QTimer.singleShot(0, self._dispatch)
            return
        self.state_changed.emit(request, "running")

    def _launch(self) -> None:
        process = self._process = QProcess(self)
        process.setProgram(self._command[0])
        process.setArguments(self._command[1:])
        environment = QProcessEnvironment.systemEnvironment()
        # Leave CPU capacity for Qt; each worker has independent backend state.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            if not environment.contains(name):
                environment.insert(name, "1")
        environment.insert("MPLBACKEND", "Agg")
        process.setProcessEnvironment(environment)
        self._buffer.clear()
        self._stderr = ""
        self._ready = False
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        process.start()
        self._kill_timer.start(30000)

    def _send(self, message: dict) -> None:
        if self._process is not None:
            self._process.write(encode_message(message))

    @Slot()
    def _read_stdout(self) -> None:
        process = self.sender()
        if process is not self._process:
            return
        chunk = process.readAllStandardOutput()
        self._buffer.extend(memoryview(chunk))
        try:
            while len(self._buffer) >= 4:
                size = struct.unpack_from("!I", self._buffer)[0]
                if size > MAX_MESSAGE_BYTES:
                    raise ValueError("Invalid calculation-worker message size.")
                if len(self._buffer) < 4 + size:
                    break
                # In-band protocol-5 decoding owns reconstructed array buffers.
                # Decode directly from the receive storage instead of making a
                # bytearray slice and a second full-size bytes copy on Qt.
                with memoryview(self._buffer) as frame, frame[4 : 4 + size] as payload:
                    message = pickle.loads(payload)
                del self._buffer[: 4 + size]
                self._message(message)
        except Exception as exc:
            self._stderr = str(exc)
            self._kill_worker()

    def _message(self, message: dict) -> None:
        if message["event"] == "ready":
            self._kill_timer.stop()
            self._ready = True
            if self._closed:
                self._send({"command": "shutdown"})
                self._kill_timer.start(self._cancel_grace_ms)
                return
            self._dispatch()
            return
        request = self._active
        if request is None or message.get("job_id") != request.job_id:
            return
        if message["event"] == "progress":
            if not self._cancelling and self.is_current(request):
                self.progress.emit(request, message)
        elif message["event"] == "result":
            self._kill_timer.stop()
            status = "cancelled" if self._cancelling else message["status"]
            self._active = None
            self._cancelling = False
            self._terminal(
                request, status, message.get("data"), message.get("error", "")
            )
            if self._closed:
                self._send({"command": "shutdown"})
                self._kill_timer.start(self._cancel_grace_ms)
            QTimer.singleShot(0, self._dispatch)

    def _terminal(
        self, request: JobRequest, status: str, data: Any = None, error: str = ""
    ) -> None:
        self.state_changed.emit(request, status)
        self.finished.emit(
            JobResult(
                request,
                status,
                data,
                error,
                status != "cancelled" and self.is_current(request),
            )
        )

    @Slot()
    def _read_stderr(self) -> None:
        process = self.sender()
        if process is self._process:
            self._stderr = (
                self._stderr
                + bytes(process.readAllStandardError()).decode(
                    "utf-8", errors="replace"
                )
            )[-16000:]

    @Slot()
    def _kill_worker(self) -> None:
        if self._process is not None:
            self._process.kill()

    @Slot(int, QProcess.ExitStatus)
    def _process_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        self._release_process()

    @Slot(QProcess.ProcessError)
    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._release_process()

    def _release_process(self) -> None:
        process = self.sender()
        if process is not self._process:
            return
        self._process = None
        self._ready = False
        self._kill_timer.stop()
        process.deleteLater()
        if self._active is not None:
            request, self._active = self._active, None
            self._terminal(
                request,
                "cancelled" if self._cancelling else "failed",
                error=self._stderr or "Calculation worker exited unexpectedly.",
            )
        elif not self._closed and self._pending:
            # A failed launch must not loop forever, or leave queued controls busy.
            pending, self._pending = self._pending, deque()
            for request in pending:
                self._terminal(
                    request,
                    "failed",
                    error=self._stderr or "Could not start the calculation worker.",
                )
        self._cancelling = False
        if self._closed:
            self.stopped.emit()
        else:
            QTimer.singleShot(0, self._dispatch)

    @Slot()
    def shutdown(self) -> None:
        """Revoke results and reap asynchronously; callers never join on the GUI."""
        if self._closed:
            return
        self._closed = True
        self._document_changed(self.document.token)
        if self._process is None:
            self.stopped.emit()
        elif self._active is None:
            self._send({"command": "shutdown"})
            self._kill_timer.start(self._cancel_grace_ms)
