"""Visible-view scheduling and contextual progress for numeric layout jobs."""

from __future__ import annotations

import pickle
import uuid
from dataclasses import replace

from PySide6.QtCore import QEvent, QObject, QTimer, Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget

from optiland_gui.services.job_records import BackendConfig, OpticSnapshot


class LayoutJobView(QObject):
    """Keep one view's last result while its detached replacement is calculated."""

    def __init__(self, viewer, kind, layout, parameters, present):
        super().__init__(viewer)
        self.viewer = viewer
        self.connector = viewer.connector
        self.jobs = self.connector.calculation_jobs
        self.kind = kind
        self.target = f"layout-{kind}-{uuid.uuid4().hex}"
        self.parameters = parameters
        self.present = present
        self.data = None
        self.context = None
        self._requested_key = None
        self._completed_key = None
        self._job_id = None
        self._active = False
        self._restyle_pending = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.request)
        self.status_widget = QWidget(viewer)
        row = QHBoxLayout(self.status_widget)
        row.setContentsMargins(5, 0, 5, 0)
        self.label = QLabel("Layout will update when shown.")
        self.label.setWordWrap(True)
        row.addWidget(self.label, 1)
        self.activity = QProgressBar()
        self.activity.setRange(0, 0)
        self.activity.setFixedSize(65, 12)
        self.activity.setAccessibleName("Layout calculation in progress")
        self.activity.hide()
        row.addWidget(self.activity)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.hide()
        row.addWidget(self.cancel_button)
        layout.addWidget(self.status_widget)
        self._busy_timer = QTimer(self)
        self._busy_timer.setSingleShot(True)
        self._busy_timer.timeout.connect(self._show_activity)
        self.jobs.state_changed.connect(self._state_changed)
        self.jobs.progress.connect(self._progress)
        self.jobs.finished.connect(self._finished)
        self.connector.document_state.changed.connect(self.invalidate)
        viewer.installEventFilter(self)

    def eventFilter(self, watched, event):
        # Qt may deliver final native events while Python is breaking the
        # parent/child reference cycle during widget destruction.
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return False
        if watched is viewer:
            if event.type() == QEvent.Type.Show:
                self.jobs.set_target_visible(self.target, True)
                self._refresh_timer.start(0)
            elif event.type() == QEvent.Type.Hide:
                self._refresh_timer.stop()
                self.jobs.set_target_visible(self.target, False)
                if self._active:
                    self._requested_key = None
        return super().eventFilter(watched, event)

    @Slot(object)
    def invalidate(self, token=None):
        self._requested_key = None
        self._completed_key = None
        self.label.setText("Previous layout is stale; update pending.")
        if self.viewer.isVisible():
            self._refresh_timer.start(0)

    @Slot()
    def request(self):
        if not self.viewer.isVisible():
            return
        parameters = self.parameters()
        key = (
            self.connector.document_state.token,
            BackendConfig.capture(),
            pickle.dumps(parameters, protocol=5),
        )
        if key in (self._requested_key, self._completed_key):
            if key == self._completed_key and self._restyle_pending:
                self.redraw()
            return
        try:
            optic = self.connector.get_optic()
            self.label.setText("Preparing layout update…")
            snapshot = OpticSnapshot.capture(optic)
            context = {
                "surface_identities": tuple(optic.surfaces),
                "document_id": self.connector.document_state.token.document_id,
                "parameters": parameters,
                "key": key,
            }
            request = self.jobs.submit(
                self.target,
                f"optiland_gui.services.layout_tasks:prepare_{self.kind}",
                snapshot,
                parameters,
                context=context,
            )
            self._job_id = request.job_id
            # Cancelling a previous queued preview emits its terminal result
            # synchronously inside submit. Install the new key afterwards.
            self._requested_key = key
        except Exception as exc:
            self._requested_key = None
            self._set_busy(False)
            self.label.setText(f"Unable to update layout: {exc}")

    @Slot()
    def cancel(self):
        self._refresh_timer.stop()
        self.jobs.cancel_target(self.target)
        self._requested_key = None

    def redraw(self):
        """Restyle retained data; no optical snapshot or calculation is requested."""
        if self.data is not None and self.viewer.isVisible():
            self.present(self.data, self.context, True)
            self._restyle_pending = False
        else:
            self._restyle_pending = True

    @Slot(object, str)
    def _state_changed(self, request, state):
        if request.target != self.target or (
            self._job_id is not None and request.job_id < self._job_id
        ):
            return
        if state in ("queued", "running", "cancelling"):
            self._set_busy(True)
            text = {
                "queued": "Update queued",
                "running": "Updating layout",
                "cancelling": "Cancelling layout update",
            }[state]
            self.label.setText(
                f"{text}… Previous result is stale." if self.data else f"{text}…"
            )
        else:
            self._set_busy(False)

    def _set_busy(self, busy):
        self._active = busy
        self.cancel_button.setVisible(busy)
        if busy:
            if not self._busy_timer.isActive() and not self.activity.isVisible():
                self._busy_timer.start(500)
        else:
            self._busy_timer.stop()
            self.activity.hide()

    @Slot()
    def _show_activity(self):
        if self._active:
            self.activity.show()

    @Slot(object, dict)
    def _progress(self, request, message):
        if request.target != self.target or not self.jobs.is_current(request):
            return
        text = message["stage"]
        if message.get("completed") is not None and message.get("total") is not None:
            text += f" ({message['completed']} of {message['total']})"
        self.label.setText(f"{text}…")

    @Slot(object)
    def _finished(self, result):
        request = result.request
        if request.target != self.target or request.job_id != self._job_id:
            return
        self._set_busy(False)
        self._requested_key = None
        if result.status == "cancelled":
            self.label.setText("Update cancelled; previous layout is stale.")
        elif result.status == "failed":
            if "Polarization must be set" in result.error:
                self.label.setText(
                    "Set incident polarization in System Properties > Polarization, "
                    "then click Apply Polarization."
                )
            else:
                self.label.setText(
                    "Error updating layout. Apply or reopen the view to retry."
                )
            self.label.setToolTip(result.error)
        elif result.current and self.jobs.is_current(request):
            try:
                self.present(result.data, request.context, False)
            except Exception as exc:
                self.label.setText(f"Error presenting layout: {exc}")
                return
            self.data = result.data
            self.context = request.context
            self._completed_key = request.context["key"]
            self._restyle_pending = False
            self.label.setToolTip("")
            self.label.setText("Layout is up to date.")
        else:
            self.label.setText("Previous layout is stale; apply to update.")
            backend_changed = (
                request.snapshot is not None
                and request.snapshot.backend != BackendConfig.capture()
            )
            # Recheck document, target generation, visibility and shutdown while
            # excluding only the obsolete backend. Cancellation must stay revoked.
            if backend_changed and self.jobs.is_current(
                replace(request, snapshot=None)
            ):
                self.invalidate()
