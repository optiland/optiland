"""Explicit scientific execution controls, separate from the public Qt console."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from optiland_gui.services.scientific_scripts import ScientificScripts


class ScientificScriptPanel(QWidget):
    """Bounded output and explicit result application for an isolated script run."""

    def __init__(self, connector, parent=None):
        super().__init__(parent)
        self.service = ScientificScripts(connector)
        self._busy = False
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Scientific Run uses an independent optic. Use optic, np, be and gui. "
            "The original Console retains connector/iface GUI access."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(2002)
        layout.addWidget(self.output, 1)
        row = QHBoxLayout()
        self.status = QLabel("Ready for Scientific Run.")
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        self.activity = QProgressBar()
        self.activity.setRange(0, 0)
        self.activity.setFixedSize(60, 12)
        self.activity.setAccessibleName("Scientific script running")
        self.activity.hide()
        row.addWidget(self.activity)
        self.cancel = QPushButton("Stop")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.service.cancel)
        row.addWidget(self.cancel)
        self.restart = QPushButton("Restart")
        self.restart.clicked.connect(self.service.restart)
        row.addWidget(self.restart)
        self.apply = QPushButton("Apply Script Result")
        self.apply.setEnabled(False)
        self.apply.clicked.connect(self.service.apply_candidate)
        row.addWidget(self.apply)
        layout.addLayout(row)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(lambda: self.activity.setVisible(self._busy))
        self.service.status_changed.connect(self._status_changed)
        self.service.output_changed.connect(self.output.setPlainText)
        self.service.candidate_changed.connect(self.apply.setEnabled)

    @Slot(str, bool)
    def _status_changed(self, text, busy):
        was_busy, self._busy = self._busy, busy
        self.status.setText(text)
        self.cancel.setEnabled(busy)
        if busy and not was_busy:
            self.timer.start(500)
        elif not busy:
            self.timer.stop()
            self.activity.hide()
