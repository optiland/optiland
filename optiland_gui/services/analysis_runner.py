"""Analysis runner service for the Optiland GUI.

Discovers registered classes and schedules explicit analyses through the shared
calculation service. Neither construction nor plotting runs on the GUI thread.
"""

from __future__ import annotations

import copy
import importlib
import logging
from collections import deque

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from optiland_gui.services.analysis_worker import (
    normalized_parameters,
    validate_workload,
)
from optiland_gui.services.job_records import (
    DocumentToken,
    JobRequest,
    JobResult,
    OpticSnapshot,
)

logger = logging.getLogger(__name__)


class AnalysisRunner(QObject):
    """Manages analysis discovery and the execution lifecycle.

    Analysis classes are loaded lazily from
    :data:`optiland_gui.registry.ANALYSIS_REGISTRY` via
    :func:`importlib.import_module`.  The resolved
    ``(category, name, class)`` tuples are cached after the first call to
    :meth:`get_analysis_registry`.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`
            instance that owns this service.
    """

    state_changed = Signal(str, str)
    finished = Signal(str, object)
    busy_changed = Signal(bool)
    progress = Signal(str, str)

    def __init__(self, connector: object) -> None:
        super().__init__(connector if isinstance(connector, QObject) else None)
        self._connector = connector
        self._registry_cache: list[tuple[str, str, type]] | None = None
        self.jobs = connector.calculation_jobs
        self._requests = {}
        self._last_result = None
        self._batch = deque()
        self._batch_target = None
        self._batch_snapshot = None
        self._batch_token = None
        self._document_id = connector.document_state.token.document_id
        self.jobs.state_changed.connect(self._state_changed)
        self.jobs.finished.connect(self._finished)
        self.jobs.progress.connect(self._progress)
        connector.document_state.changed.connect(self._document_changed)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def get_analysis_registry(self) -> list[tuple[str, str, type]]:
        """Return the resolved analysis registry.

        Each entry is a ``(category, display_name, cls)`` tuple where *cls*
        is the live Python class loaded via :func:`importlib.import_module`.
        Entries whose class path cannot be imported are silently omitted and
        a warning is logged.

        The result is cached after the first call.

        Returns:
            A list of ``(category, display_name, cls)`` tuples.
        """
        if self._registry_cache is not None:
            return self._registry_cache

        from optiland_gui.registry import ANALYSIS_REGISTRY

        resolved: list[tuple[str, str, type]] = []
        for category, name, class_path in ANALYSIS_REGISTRY:
            try:
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                resolved.append((category, name, cls))
            except (ImportError, AttributeError) as exc:
                logger.warning(
                    "AnalysisRunner: could not load '%s' (%s): %s",
                    name,
                    class_path,
                    exc,
                )

        self._registry_cache = resolved
        return self._registry_cache

    # ------------------------------------------------------------------
    # Explicit jobs use the shared executor and immutable prescription snapshots.
    # ------------------------------------------------------------------

    def run(
        self,
        analysis_name: str,
        params: dict,
        optic: object,
        *,
        target: str = "analysis",
        view_args: dict | None = None,
        theme: str = "dark",
        snapshot: OpticSnapshot | None = None,
        document_token: DocumentToken | None = None,
    ) -> JobRequest:
        """Execute a named analysis with the given parameters.

        Args:
            analysis_name: The display name of the analysis as it appears in
                the registry.
            params: A dict of parameter name → value pairs to pass to the
                analysis class constructor.
            optic: The :class:`~optiland.optic.Optic` instance to analyse.
        """
        args = normalized_parameters(analysis_name, params)
        validate_workload(
            analysis_name,
            args,
            view_args or {},
            optic.surfaces.num_surfaces,
            optic.fields.num_fields,
            optic.wavelengths.num_wavelengths,
        )
        snapshot = snapshot or OpticSnapshot.capture(optic)
        # An explicit rerun supersedes this page's not-yet-dispatched batch entry.
        self._batch = deque(e for e in self._batch if e["target"] != target)
        request = self.jobs.submit(
            target,
            "optiland_gui.services.analysis_worker:prepare_analysis",
            snapshot,
            {
                "name": analysis_name,
                "constructor_args": args,
                "view_args": view_args or {},
                "theme": theme,
            },
            cancel_on_document_change=False,
            document_token=document_token,
        )
        self._requests[target] = request
        self.state_changed.emit(target, "queued")
        self.busy_changed.emit(True)
        return request

    @property
    def busy(self) -> bool:
        return bool(self._requests or self._batch)

    def run_batch(self, entries: list[dict], optic: object) -> None:
        """Freeze existing configured pages once, then submit one entry at a time."""
        if self.busy:
            raise ValueError(
                "Wait for analysis jobs to finish or stop them before Run All."
            )
        if not entries:
            return
        if len(entries) > 16:
            raise ValueError("Run All supports at most 16 configured analysis pages.")
        # Validate every entry before starting any work.
        frozen = []
        for entry in entries:
            args = normalized_parameters(entry["name"], entry["constructor_args"])
            validate_workload(
                entry["name"],
                args,
                entry["view_args"],
                optic.surfaces.num_surfaces,
                optic.fields.num_fields,
                optic.wavelengths.num_wavelengths,
            )
            frozen.append(copy.deepcopy({**entry, "constructor_args": args}))
        self._batch_snapshot = OpticSnapshot.capture(optic)
        self._batch_token = self._connector.document_state.token
        self._batch = deque(frozen)
        for entry in frozen:
            self.state_changed.emit(entry["target"], "queued")
        self.busy_changed.emit(True)
        QTimer.singleShot(0, self._dispatch_batch)

    @Slot()
    def _dispatch_batch(self) -> None:
        if self._batch_target is not None:
            return
        if not self._batch:
            self._batch_snapshot = None
            self._batch_token = None
            self.busy_changed.emit(self.busy)
            return
        entry = self._batch.popleft()
        target = entry["target"]
        self._batch_target = target
        try:
            # Validation already used the captured prescription. Do not inspect
            # a later edited optical model when dispatching the next batch entry.
            request = self.jobs.submit(
                target,
                "optiland_gui.services.analysis_worker:prepare_analysis",
                self._batch_snapshot,
                {
                    "name": entry["name"],
                    "constructor_args": entry["constructor_args"],
                    "view_args": entry["view_args"],
                    "theme": entry["theme"],
                },
                cancel_on_document_change=False,
                document_token=self._batch_token,
            )
            self._requests[target] = request
        except RuntimeError as exc:
            if "queue is full" in str(exc):
                self._batch.appendleft(entry)
                self._batch_target = None
                QTimer.singleShot(100, self._dispatch_batch)
            else:
                request = JobRequest(0, self._batch_token, target, 0, "", None, {})
                self.finished.emit(target, JobResult(request, "failed", error=str(exc)))
                self._batch_target = None
                self.stop()

    def cancel(self, target: str) -> None:
        """Revoke a page even if it is still waiting in the frozen batch."""
        self._batch = deque(e for e in self._batch if e["target"] != target)
        self.jobs.cancel_target(target)
        self.state_changed.emit(target, "cancelled")
        self.busy_changed.emit(self.busy)

    @Slot(object, str)
    def _state_changed(self, request: JobRequest, state: str) -> None:
        if self._requests.get(request.target) is request:
            self.state_changed.emit(request.target, state)

    @Slot(object, dict)
    def _progress(self, request: JobRequest, data: dict) -> None:
        if self._requests.get(request.target) is request:
            self.progress.emit(request.target, data["stage"])

    @Slot(object)
    def _finished(self, result: JobResult) -> None:
        target = result.request.target
        if self._requests.get(target) is not result.request:
            return
        del self._requests[target]
        self._last_result = result
        self.finished.emit(target, result)
        if result.status == "failed" and result.infrastructure_error:
            self.stop()
        if self._batch_target == target:
            self._batch_target = None
            QTimer.singleShot(0, self._dispatch_batch)
        self.busy_changed.emit(self.busy)

    @Slot(object)
    def _document_changed(self, token: DocumentToken) -> None:
        if token.document_id != self._document_id:
            self._document_id = token.document_id
            self.stop()

    def stop(self) -> None:
        """Request cancellation of an in-progress analysis run."""
        pending, self._batch = self._batch, deque()
        for entry in pending:
            self.state_changed.emit(entry["target"], "cancelled")
        for target in list(self._requests):
            self.jobs.cancel_target(target)
        self._batch_snapshot = None
        self._batch_token = None
        self.busy_changed.emit(self.busy)

    def get_result(self) -> object | None:
        """Return the result of the most recent analysis run.

        Returns:
            The most recent detached result, or ``None`` before a job completes.
        """
        return self._last_result
