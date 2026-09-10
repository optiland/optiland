"""Explicit placement drafts beside the selected surface's shape properties."""

from __future__ import annotations

import math

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from optiland_gui.services.placement_tasks import (
    COMPONENTS,
    placement_values,
    validate_changes,
)


class SurfacePlacementWidget(QWidget):
    """Keep typed placement separate from the optic until explicit acceptance."""

    def __init__(self, row, connector, parent=None):
        super().__init__(parent)
        self.row, self.connector = row, connector
        self.service = connector.placement
        self.surface = connector.get_optic().surfaces[row]
        self._baseline = {}
        self._loading = False
        self._editable = False
        self.setObjectName("SurfacePlacementWidget")
        layout = QVBoxLayout(self)
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Coordinates:"))
        self.frame = QComboBox()
        self.frame.setObjectName("placement_frame")
        self.frame.addItem("Global", "global")
        self.frame.addItem("Reference frame", "reference")
        self.frame.setToolTip(
            "Reference uses the surface's existing parent coordinate frame."
        )
        frame_row.addWidget(self.frame)
        frame_row.addStretch()
        layout.addLayout(frame_row)
        grid = QGridLayout()
        grid.addWidget(QLabel("Position / decenter (lens units)"), 0, 0, 1, 2)
        grid.addWidget(QLabel("Tilt (degrees)"), 0, 2, 1, 2)
        self.inputs = {}
        for index, axis in enumerate("xyz", 1):
            for column, key in ((0, axis), (2, "r" + axis)):
                label = QLabel(axis.upper() if column == 0 else f"Tilt {axis.upper()}")
                edit = QLineEdit()
                edit.setObjectName(f"placement_{key}")
                edit.setAccessibleName(f"Surface {row} {label.text()}")
                edit.setMinimumWidth(75)
                label.setBuddy(edit)
                edit.textChanged.connect(self._draft_changed)
                edit.returnPressed.connect(self.apply)
                grid.addWidget(label, index, column)
                grid.addWidget(edit, index, column + 1)
                self.inputs[key] = edit
        layout.addLayout(grid)
        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        buttons = QHBoxLayout()
        self.apply_button = QPushButton("Apply Placement")
        self.apply_button.setObjectName("apply_placement")
        self.reload_button = QPushButton("Reload")
        self.reload_button.setToolTip(
            "Discard the draft and read the current surface placement."
        )
        self.cancel_button = QPushButton("Cancel")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumWidth(60)
        for widget in (
            self.apply_button,
            self.progress,
            self.reload_button,
            self.cancel_button,
        ):
            buttons.addWidget(widget)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setAccessibleName("Placement status")
        layout.addWidget(self.status)
        self.apply_button.clicked.connect(self.apply)
        self.reload_button.clicked.connect(lambda: self.refresh_values(force=True))
        self.cancel_button.clicked.connect(self.service.cancel)
        self.frame.currentIndexChanged.connect(lambda: self.refresh_values(force=True))
        self.service.state_changed.connect(self._service_changed)
        self.refresh_values(force=True)

    def changes(self):
        return {
            key: edit.text()
            for key, edit in self.inputs.items()
            if edit.text() != self._baseline.get(key)
        }

    def refresh_values(self, *, force=False):
        if not force and self.changes():
            if self._edit_token != self.connector.document_state.edit_token:
                self.status.setText(
                    "The design changed. Reload placement before applying."
                )
            return
        self._loading = True
        self._edit_token = self.connector.document_state.edit_token
        try:
            values = placement_values(self.surface, self.frame.currentData())
            self._editable = all(
                math.isfinite(value) for key, value in values.items() if key != "z"
            )
            self._baseline = {key: format(values[key], ".17g") for key in COMPONENTS}
            for key, edit in self.inputs.items():
                edit.setText(self._baseline[key])
            reference = self.frame.currentData() == "reference"
            parent = self.surface.geometry.cs.reference_cs
            frame_text = (
                "parent reference frame" if reference and parent else "global frame"
            )
            self.description.setText(
                f"Absolute values in the {frame_text}. X/Y are decenter components. "
                "Tilts rotate X, then Y, then Z about fixed frame axes. "
                "Only this surface moves; lengths are normally mm."
            )
            self.status.setText(
                self.service.message
                if math.isfinite(values["z"])
                else "Surface at infinity: enter a finite Z to set its position."
            )
        except Exception as exc:
            self._editable = False
            self.status.setText(f"Unable to read placement: {exc}")
        self._loading = False
        self._update_controls()

    @Slot()
    def _draft_changed(self):
        if not self._loading:
            self._update_controls()

    def _update_controls(self):
        busy = self.service.busy
        dirty = bool(self.changes())
        self.frame.setEnabled(not busy and not dirty)
        for edit in self.inputs.values():
            edit.setEnabled(self._editable and not busy)
        self.apply_button.setEnabled(self._editable and dirty and not busy)
        self.reload_button.setEnabled(not busy)
        self.cancel_button.setVisible(busy)
        self.progress.setVisible(busy)

    @Slot()
    def _service_changed(self):
        self.status.setText(self.service.message)
        self._update_controls()

    @Slot()
    def apply(self):
        if not self._editable or self.service.busy:
            return
        try:
            changes = validate_changes(self.changes())
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.service.submit(
            self.surface, self.frame.currentData(), changes, self._edit_token
        )
