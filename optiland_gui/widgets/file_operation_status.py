"""Non-modal file-operation status and cancellation in the main status bar."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget


class FileOperationStatus(QWidget):
    """Show file activity locally while the rest of the window remains usable."""

    def __init__(self, operations, parent=None):
        super().__init__(parent)
        self._busy = False
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 0, 2, 0)
        self.label = QLabel()
        row.addWidget(self.label, 1)
        self.activity = QProgressBar()
        self.activity.setRange(0, 0)
        self.activity.setFixedSize(65, 12)
        self.activity.setAccessibleName("File operation in progress")
        self.activity.hide()
        row.addWidget(self.activity)
        self.cancel = QPushButton("Cancel file operations")
        self.cancel.clicked.connect(operations.cancel_pending)
        self.cancel.hide()
        row.addWidget(self.cancel)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._show_activity)
        operations.state_changed.connect(self.update_state)
        self.hide()

    @Slot(str, bool, bool)
    def update_state(self, text, busy, cancellable):
        self._busy = busy
        self.label.setText(text)
        self.cancel.setVisible(cancellable)
        self.show()
        if busy:
            if not self.timer.isActive() and not self.activity.isVisible():
                self.timer.start(500)
        else:
            self.timer.stop()
            self.activity.hide()

    @Slot()
    def _show_activity(self):
        if self._busy:
            self.activity.show()
