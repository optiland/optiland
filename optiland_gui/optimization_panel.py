"""Full Optimization Panel for the Optiland GUI.

Provides a dockable panel for configuring and running optical-system
optimisations.  The panel is divided into three tabs (Variables, Operands,
Optimizer), a persistent live-updates section, Run/Stop buttons, and a
scrollable results log.

Author: Manuel Fragata Mendes / Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from .optiland_connector import OptilandConnector


# ---------------------------------------------------------------------------
# Helper dialogs
# ---------------------------------------------------------------------------


class AddVariableDialog(QDialog):
    """Small dialog for specifying a new optimization variable.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`.
        surface_index: Pre-selected surface number (default 1).
        suggested_type: Pre-selected variable type key (default ``"radius"``).
        parent: Parent widget.
    """

    def __init__(
        self,
        connector: OptilandConnector,
        surface_index: int = 1,
        suggested_type: str = "radius",
        initial_vd: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            "Add Optimization Variable"
            if not initial_vd
            else "Edit Optimization Variable"
        )
        self.setMinimumWidth(380)
        self._connector = connector
        self._suggested_surface = surface_index
        if initial_vd:
            self._suggested_surface = initial_vd.get("surface_number", surface_index)
            suggested_type = initial_vd.get("type", suggested_type)

        self._param_widgets: dict[str, QWidget] = {}
        self._dynamic_rows: list[int] = []

        layout = QVBoxLayout(self)
        self.top_form = QFormLayout()
        self.dynamic_form = QFormLayout()
        self.bottom_form = QFormLayout()
        for f in [self.top_form, self.dynamic_form, self.bottom_form]:
            f.setSpacing(8)

        layout.addLayout(self.top_form)
        layout.addLayout(self.dynamic_form)
        layout.addLayout(self.bottom_form)

        # Surface number (always required for variables)
        self.spnSurface = QSpinBox(self)
        self.spnSurface.setRange(0, 9999)
        self.spnSurface.setValue(self._suggested_surface)
        self.top_form.addRow("Surface #:", self.spnSurface)

        # Variable type
        self.cmbType = QComboBox(self)
        for display, key in connector.get_common_variable_types():
            self.cmbType.addItem(display, userData=key)
        self.top_form.addRow("Type:", self.cmbType)

        # Min / Max (always at the bottom)
        min_val = (
            str(initial_vd.get("min_val"))
            if initial_vd and initial_vd.get("min_val") is not None
            else "None"
        )
        max_val = (
            str(initial_vd.get("max_val"))
            if initial_vd and initial_vd.get("max_val") is not None
            else "None"
        )
        self.txtMin = QLineEdit(min_val, self)
        self.txtMax = QLineEdit(max_val, self)
        self.bottom_form.addRow("Min value:", self.txtMin)
        self.bottom_form.addRow("Max value:", self.txtMax)

        self.cmbType.currentIndexChanged.connect(self._rebuild_form)

        # Initial build
        idx = self.cmbType.findData(suggested_type)
        if idx >= 0:
            self.cmbType.setCurrentIndex(idx)

        self._rebuild_form()

        if initial_vd:
            self._apply_initial_vd(initial_vd)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _rebuild_form(self) -> None:
        """Clear and rebuild the dynamic parameters part of the form."""
        while self.dynamic_form.rowCount() > 0:
            self.dynamic_form.removeRow(0)

        self._param_widgets.clear()
        var_type = self.cmbType.currentData()
        meta = self._connector.get_variable_metadata(var_type)

        for name, info in meta.items():
            label = name.replace("_", " ").capitalize() + ":"
            widget = self._create_widget(name, info)
            self.dynamic_form.addRow(label, widget)
            self._param_widgets[name] = widget

    def _create_widget(self, name: str, info: dict) -> QWidget:
        """Create a widget based on metadata."""
        w_type = info.get("type", "int")

        if w_type == "int":
            w = QSpinBox()
            w.setRange(info.get("min", 0), info.get("max", 1000000))
            w.setValue(info.get("default", 0))
            return w

        if w_type == "float":
            w = QDoubleSpinBox()
            w.setRange(info.get("min", -1e9), info.get("max", 1e9))
            w.setDecimals(info.get("decimals", 4))
            w.setValue(info.get("default", 0.0))
            return w

        if w_type == "choice":
            w = QComboBox()
            w.addItems(info.get("options", []))
            idx = w.findText(info.get("default", ""))
            if idx >= 0:
                w.setCurrentIndex(idx)
            return w

        if w_type == "wavelength":
            w = QComboBox()
            w.addItem("Primary", userData="primary")
            for display, val in self._connector.get_wavelength_options():
                w.addItem(display, userData=val)
            return w

        return QLineEdit(str(info.get("default", "")))

    def _apply_initial_vd(self, vd: dict) -> None:
        """Apply parameters from vd to dynamic widgets."""
        for name, widget in self._param_widgets.items():
            if name in vd:
                val = vd[name]
                if isinstance(widget, QSpinBox | QDoubleSpinBox):
                    widget.setValue(val)
                elif isinstance(widget, QComboBox):
                    idx = widget.findData(val)
                    if idx < 0:
                        idx = widget.findText(str(val))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(val))

    def get_variable_dict(self) -> dict:
        """Return the variable descriptor dict from the current form values."""

        def _parse_val(text):
            text = text.strip()
            if text.lower() == "none" or not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None

        vd: dict = {
            "type": self.cmbType.currentData(),
            "surface_number": self.spnSurface.value(),
            "min_val": _parse_val(self.txtMin.text()),
            "max_val": _parse_val(self.txtMax.text()),
        }

        for name, widget in self._param_widgets.items():
            if isinstance(widget, QSpinBox | QDoubleSpinBox):
                vd[name] = widget.value()
            elif isinstance(widget, QComboBox):
                data = widget.currentData()
                vd[name] = data if data is not None else widget.currentText()
            elif isinstance(widget, QLineEdit):
                vd[name] = widget.text()

        return vd


class AddOperandDialog(QDialog):
    """Small dialog for specifying a new optimization operand.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`.
        parent: Parent widget.
    """

    def __init__(
        self,
        connector: OptilandConnector,
        initial_od: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            "Add Optimization Operand"
            if not initial_od
            else "Edit Optimization Operand"
        )
        self.setMinimumWidth(440)
        self._connector = connector
        self._param_widgets: dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        self.top_form = QFormLayout()
        self.dynamic_form = QFormLayout()
        self.bottom_form = QFormLayout()
        for f in [self.top_form, self.dynamic_form, self.bottom_form]:
            f.setSpacing(8)

        layout.addLayout(self.top_form)
        layout.addLayout(self.dynamic_form)
        layout.addLayout(self.bottom_form)

        # Category
        categories = list(connector.get_operand_categories().keys())
        self.cmbCategory = QComboBox(self)
        self.cmbCategory.addItems(categories)
        self.top_form.addRow("Category:", self.cmbCategory)

        # Type — filtered by category
        self.cmbType = QComboBox(self)
        self.top_form.addRow("Type:", self.cmbType)

        # Target / Weight (static)
        self.spnTarget = QDoubleSpinBox(self)
        self.spnTarget.setDecimals(6)
        self.spnTarget.setRange(-1e9, 1e9)
        self.spnTarget.setValue(0.0)

        self.spnWeight = QDoubleSpinBox(self)
        self.spnWeight.setDecimals(4)
        self.spnWeight.setRange(0.0, 1e9)
        self.spnWeight.setValue(1.0)

        self.bottom_form.addRow(QLabel("<hr>"))
        self.bottom_form.addRow("Target:", self.spnTarget)
        self.bottom_form.addRow("Weight:", self.spnWeight)

        self.cmbCategory.currentTextChanged.connect(self._on_category_changed)
        self.cmbType.currentTextChanged.connect(self._rebuild_form)

        if initial_od:
            cat_idx = self.cmbCategory.findText(initial_od.get("category", ""))
            if cat_idx >= 0:
                self.cmbCategory.setCurrentIndex(cat_idx)

            self._on_category_changed(self.cmbCategory.currentText())

            type_idx = self.cmbType.findText(initial_od.get("type", ""))
            if type_idx >= 0:
                self.cmbType.setCurrentIndex(type_idx)

            self.spnTarget.setValue(initial_od.get("target", 0.0))
            self.spnWeight.setValue(initial_od.get("weight", 1.0))
        else:
            # Initial build
            self._on_category_changed(self.cmbCategory.currentText())

        if initial_od:
            self._apply_initial_od(initial_od)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #e57373;")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @Slot(str)
    def _on_category_changed(self, category: str) -> None:
        """Repopulate the Type combo when the category changes."""
        self.cmbType.blockSignals(True)
        self.cmbType.clear()
        types = self._connector.get_operand_categories().get(category, [])
        self.cmbType.addItems(types)
        self.cmbType.blockSignals(False)
        self._rebuild_form()

    def _rebuild_form(self) -> None:
        """Clear and rebuild the dynamic parameters part of the form."""
        while self.dynamic_form.rowCount() > 0:
            self.dynamic_form.removeRow(0)

        self._param_widgets.clear()
        op_type = self.cmbType.currentText()
        if not op_type:
            return

        meta = self._connector.get_operand_metadata(op_type)

        if meta:
            # Header for extra parameters
            sep = QLabel("<b>Parameters</b>")
            sep.setAlignment(Qt.AlignCenter)
            self.dynamic_form.addRow(sep)

            for name, info in meta.items():
                label = (
                    name.replace("_", " ")
                    .replace("line ray", "Ray A")
                    .replace("point ray", "Ray B")
                    .capitalize()
                    + ":"
                )
                widget = self._create_widget(name, info)
                self.dynamic_form.addRow(label, widget)
                self._param_widgets[name] = widget

    def _create_widget(self, name: str, info: dict) -> QWidget:
        """Create a widget based on metadata."""
        w_type = info.get("type", "int")

        if w_type == "int":
            w = QSpinBox()
            w.setRange(info.get("min", 0), info.get("max", 9999))

            default = info.get("default", 1)
            if "surface" in name.lower():
                optic = self._connector.get_optic()
                if optic:
                    # Default to image surface (last surface)
                    default = max(0, optic.surface_group.num_surfaces - 1)

            w.setValue(default)
            return w

        if w_type == "float":
            w = QDoubleSpinBox()
            if name in ["Hx", "Hy", "Px", "Py"]:
                w.setRange(-1.0, 1.0)
            else:
                w.setRange(info.get("min", -1e9), info.get("max", 1e9))
            w.setDecimals(info.get("decimals", 4))
            w.setValue(info.get("default", 0.0))
            return w

        if w_type == "choice":
            w = QComboBox()
            w.addItems(info.get("options", []))
            idx = w.findText(info.get("default", ""))
            if idx >= 0:
                w.setCurrentIndex(idx)
            return w

        if w_type == "wavelength":
            w = QComboBox()
            w.addItem("Primary", userData="primary")
            for display, val in self._connector.get_wavelength_options():
                w.addItem(display, userData=val)
            return w

        if w_type == "tuple":
            # For tuples like (Hx, Hy), use two spin boxes in HBox
            container = QWidget()
            h_layout = QHBoxLayout(container)
            h_layout.setContentsMargins(0, 0, 0, 0)

            w1 = QDoubleSpinBox()
            w2 = QDoubleSpinBox()

            if "coords" in name.lower():
                w1.setRange(-1.0, 1.0)
                w2.setRange(-1.0, 1.0)
            else:
                w1.setRange(-1e9, 1e9)
                w2.setRange(-1e9, 1e9)

            w1.setDecimals(4)
            w1.setValue(info.get("default", [0.0, 0.0])[0])

            w2.setDecimals(4)
            w2.setValue(info.get("default", [0.0, 0.0])[1])

            h_layout.addWidget(w1)
            h_layout.addWidget(w2)

            # Monkey-patch a way to get the value
            container.get_value = lambda: [w1.value(), w2.value()]
            return container

        return QLineEdit(str(info.get("default", "")))

    def _apply_initial_od(self, od: dict) -> None:
        """Apply parameters from od to dynamic widgets."""
        input_data = od.get("input_data", {})
        for name, widget in self._param_widgets.items():
            if name in input_data:
                val = input_data[name]
                if hasattr(widget, "get_value"):  # Tuple container
                    spinboxes = widget.findChildren(QDoubleSpinBox)
                    if (
                        len(spinboxes) >= 2
                        and isinstance(val, list | tuple)
                        and len(val) >= 2
                    ):
                        spinboxes[0].setValue(val[0])
                        spinboxes[1].setValue(val[1])
                elif isinstance(widget, QSpinBox | QDoubleSpinBox):
                    widget.setValue(val)
                elif isinstance(widget, QComboBox):
                    idx = widget.findData(val)
                    if idx < 0:
                        idx = widget.findText(str(val))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(val))

    def accept(self) -> None:
        """Validate before accepting."""
        # For now, minimal validation as GUI is more restrictive than JSON
        super().accept()

    def get_operand_dict(self) -> dict:
        """Return the operand descriptor dict from the current form values."""
        input_data = {}
        for name, widget in self._param_widgets.items():
            if hasattr(widget, "get_value"):
                input_data[name] = widget.get_value()
            elif isinstance(widget, QSpinBox | QDoubleSpinBox):
                input_data[name] = widget.value()
            elif isinstance(widget, QComboBox):
                data = widget.currentData()
                input_data[name] = data if data is not None else widget.currentText()
            elif isinstance(widget, QLineEdit):
                input_data[name] = widget.text()

        return {
            "type": self.cmbType.currentText(),
            "category": self.cmbCategory.currentText(),
            "target": self.spnTarget.value(),
            "min_val": None,
            "max_val": None,
            "weight": self.spnWeight.value(),
            "input_data": input_data,
        }


class AutoGenerateOperandsDialog(QDialog):
    """Dialog for batch-generating image quality operands.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`.
        parent: Parent widget.
    """

    def __init__(
        self,
        connector: OptilandConnector,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto-Generate Image Quality Operands")
        self.setMinimumWidth(500)
        self._connector = connector

        layout = QVBoxLayout(self)

        # Selection Group (Fields & Wavelengths)
        selection_layout = QHBoxLayout()

        # Fields
        field_group = QGroupBox("Fields")
        field_vbox = QVBoxLayout(field_group)
        self.listFields = QListWidget()
        optic = connector.get_optic()
        if optic:
            field_coords = optic.fields.get_field_coords()
            for i, (hx, hy) in enumerate(field_coords):
                display_name = f"Field {i + 1}: ({hx:.3f}, {hy:.3f})"
                item = QListWidgetItem(display_name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                item.setData(Qt.UserRole, (hx, hy))
                self.listFields.addItem(item)
        field_vbox.addWidget(self.listFields)
        selection_layout.addWidget(field_group)

        # Wavelengths
        wave_group = QGroupBox("Wavelengths")
        wave_vbox = QVBoxLayout(wave_group)
        self.listWaves = QListWidget()
        if optic:
            wavelength_values = optic.wavelengths.get_wavelengths()
            for wl in wavelength_values:
                display_name = f"{wl:.4f} µm"
                item = QListWidgetItem(display_name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                item.setData(Qt.UserRole, wl)
                self.listWaves.addItem(item)
        wave_vbox.addWidget(self.listWaves)
        selection_layout.addWidget(wave_group)

        layout.addLayout(selection_layout)

        # Metric Group
        metric_group = QGroupBox("Target Metric")
        metric_hbox = QHBoxLayout(metric_group)
        self.radRMS = QRadioButton("RMS Spot Size")
        self.radOPD = QRadioButton("Optical Path Difference (OPD)")
        self.radRMS.setChecked(True)
        metric_hbox.addWidget(self.radRMS)
        metric_hbox.addWidget(self.radOPD)
        layout.addWidget(metric_group)

        # Parameters Form
        params_group = QGroupBox("Parameters")
        self.form_layout = QFormLayout(params_group)

        self.spnRays = QSpinBox()
        self.spnRays.setRange(1, 100)
        self.spnRays.setValue(3)
        self.form_layout.addRow("Number of Rays:", self.spnRays)

        self.cmbDist = QComboBox()
        self.form_layout.addRow("Distribution:", self.cmbDist)

        self.spnTarget = QDoubleSpinBox()
        self.spnTarget.setDecimals(6)
        self.spnTarget.setRange(-1e9, 1e9)
        self.spnTarget.setValue(0.0)
        self.form_layout.addRow("Target:", self.spnTarget)

        self.spnWeight = QDoubleSpinBox()
        self.spnWeight.setDecimals(4)
        self.spnWeight.setRange(0.0, 1e9)
        self.spnWeight.setValue(1.0)
        self.form_layout.addRow("Weight:", self.spnWeight)

        layout.addWidget(params_group)

        # Signals
        self.radRMS.toggled.connect(self._update_distributions)
        self.radOPD.toggled.connect(self._update_distributions)

        # Initial distributions
        self._update_distributions()

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.button(QDialogButtonBox.Ok).setText("Generate")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_distributions(self) -> None:
        self.cmbDist.clear()
        if self.radRMS.isChecked():
            self.cmbDist.addItems(["hexapolar", "grid", "uniform", "random"])
        else:
            self.cmbDist.addItems(["gaussian_quad", "hexapolar", "grid"])

    def get_selection(self) -> dict:
        fields = []
        for i in range(self.listFields.count()):
            item = self.listFields.item(i)
            if item.checkState() == Qt.Checked:
                fields.append(item.data(Qt.UserRole))

        waves = []
        for i in range(self.listWaves.count()):
            item = self.listWaves.item(i)
            if item.checkState() == Qt.Checked:
                waves.append(item.data(Qt.UserRole))

        return {
            "fields": fields,
            "wavelengths": waves,
            "metric": "rms_spot_size" if self.radRMS.isChecked() else "OPD_difference",
            "num_rays": self.spnRays.value(),
            "distribution": self.cmbDist.currentText(),
            "target": self.spnTarget.value(),
            "weight": self.spnWeight.value(),
        }


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------


class OptimizationPanel(QWidget):
    """Full optimization panel with Variables, Operands, and Optimizer tabs.

    Provides controls for defining variables and operands, selecting and
    configuring an optimizer algorithm, live-update options, Run/Stop buttons,
    and a scrollable results log.

    Signals:
        addVariableRequested (int, str): Emitted when the panel requests a new
            variable (surface_index, suggested_type).  Re-emitted from
            ``connector.requestAddOptimizationVariable``.
    """

    addVariableRequested = Signal(int, str)

    def __init__(self, connector: OptilandConnector, parent: QWidget | None = None):
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Optimization")

        self._iteration_count = 0

        self._init_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        """Build the full panel layout."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        splitter = QSplitter(Qt.Vertical)
        outer.addWidget(splitter, 1)

        # Top half: tabs
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_variables_tab(), "Variables")
        self._tabs.addTab(self._build_operands_tab(), "Operands")
        self._tabs.addTab(self._build_optimizer_tab(), "Optimizer")
        top_layout.addWidget(self._tabs)

        # Live updates
        top_layout.addWidget(self._build_live_updates_widget())

        # Run / Stop
        run_layout = QHBoxLayout()
        self.btnRun = QPushButton("▶  Run")
        self.btnStop = QPushButton("■  Stop")
        self.btnStop.setEnabled(False)
        run_layout.addWidget(self.btnRun)
        run_layout.addWidget(self.btnStop)
        top_layout.addLayout(run_layout)

        splitter.addWidget(top_widget)

        # Bottom half: log
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(2)
        log_layout.addWidget(QLabel("Results / Log:"))
        self.txtLog = QTextEdit()
        self.txtLog.setReadOnly(True)
        self.txtLog.setMinimumHeight(80)
        log_layout.addWidget(self.txtLog)
        splitter.addWidget(log_widget)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

    def _build_variables_tab(self) -> QWidget:
        """Build the Variables tab widget."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.tblVariables = QTableWidget()
        self.tblVariables.setColumnCount(5)
        self.tblVariables.setHorizontalHeaderLabels(
            ["Surface", "Type", "Current Value", "Min", "Max"]
        )
        self.tblVariables.horizontalHeader().setStretchLastSection(True)
        self.tblVariables.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.tblVariables.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.tblVariables)

        btn_layout = QHBoxLayout()
        self.btnAddVariable = QPushButton("+ Add Variable")
        self.btnRemoveVariable = QPushButton("- Remove Selected")
        self.btnRefreshVariables = QPushButton()
        self.btnRefreshVariables.setToolTip("Refresh current values")
        self.btnRefreshVariables.setFixedSize(25, 25)
        self.btnRefreshVariables.setIcon(QIcon(":/icons/dark/refresh.svg"))

        btn_layout.addWidget(self.btnAddVariable)
        btn_layout.addWidget(self.btnRemoveVariable)
        btn_layout.addWidget(self.btnRefreshVariables)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return w

    def _build_operands_tab(self) -> QWidget:
        """Build the Operands tab widget."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.tblOperands = QTableWidget()
        self.tblOperands.setColumnCount(6)
        self.tblOperands.setHorizontalHeaderLabels(
            ["Category", "Type", "Current Value", "Target", "Weight", "Parameters"]
        )
        self.tblOperands.horizontalHeader().setStretchLastSection(True)
        self.tblOperands.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tblOperands.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.tblOperands)

        btn_layout = QHBoxLayout()
        self.btnAddOperand = QPushButton("+ Add Operand")
        self.btnAutoGenerateOperands = QPushButton("Auto-Generate")
        self.btnRemoveOperand = QPushButton("- Remove Selected")
        self.btnRefreshOperands = QPushButton()
        self.btnRefreshOperands.setToolTip("Refresh current values")
        self.btnRefreshOperands.setFixedSize(25, 25)
        self.btnRefreshOperands.setIcon(QIcon(":/icons/dark/refresh.svg"))

        btn_layout.addWidget(self.btnAddOperand)
        btn_layout.addWidget(self.btnAutoGenerateOperands)
        btn_layout.addWidget(self.btnRemoveOperand)
        btn_layout.addWidget(self.btnRefreshOperands)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return w

    def _build_optimizer_tab(self) -> QWidget:
        """Build the Optimizer tab with algorithm selector and dynamic settings."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        form_top = QFormLayout()
        self.cmbAlgorithm = QComboBox()
        model = self.cmbAlgorithm.model()
        groups = self.connector.get_optimizer_groups()
        for group_name, entries in groups.items():
            # Non-selectable group header
            self.cmbAlgorithm.addItem(group_name)
            header_idx = self.cmbAlgorithm.count() - 1
            header_item = model.item(header_idx)
            if header_item:
                header_item.setEnabled(False)
                font = header_item.font()
                font.setBold(True)
                header_item.setFont(font)
            for name, cls, _bounds_mode in entries:
                self.cmbAlgorithm.addItem(f"  {name}", userData=cls)
        form_top.addRow("Algorithm:", self.cmbAlgorithm)
        layout.addLayout(form_top)

        # Scrollable dynamic settings area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        self._optimizer_settings_widget = QWidget()
        self._optimizer_settings_layout = QFormLayout(self._optimizer_settings_widget)
        self._optimizer_settings_layout.setSpacing(6)
        scroll.setWidget(self._optimizer_settings_widget)
        layout.addWidget(scroll, 1)

        self._optimizer_param_widgets: dict[str, QWidget] = {}
        self.cmbAlgorithm.currentIndexChanged.connect(self._rebuild_optimizer_settings)

        # Select first valid algorithm (usually index 1 as 0 is "Local" header)
        if self.cmbAlgorithm.count() > 1:
            self.cmbAlgorithm.setCurrentIndex(1)
            self._rebuild_optimizer_settings()

        return w

    def _build_live_updates_widget(self) -> QGroupBox:
        """Build the Live Updates section."""
        group = QGroupBox("Live Updates")
        layout = QHBoxLayout(group)
        layout.setSpacing(12)

        self.chkLiveViz = QCheckBox("Visualization")
        self.chkLiveVars = QCheckBox("Variable Values")
        layout.addWidget(self.chkLiveViz)
        layout.addWidget(self.chkLiveVars)

        layout.addWidget(QLabel("  Frequency:"))
        self.spnFrequency = QSpinBox()
        self.spnFrequency.setRange(1, 1000)
        self.spnFrequency.setValue(10)
        self.spnFrequency.setSuffix(" iters")
        layout.addWidget(self.spnFrequency)
        layout.addStretch()

        return group

    # ------------------------------------------------------------------
    # Optimizer settings form
    # ------------------------------------------------------------------

    def _rebuild_optimizer_settings(self) -> None:
        """Clear and re-populate the Optimizer settings form."""
        while self._optimizer_settings_layout.rowCount() > 0:
            self._optimizer_settings_layout.removeRow(0)

        self._optimizer_param_widgets.clear()
        cls = self.cmbAlgorithm.currentData()
        if not cls:
            return

        meta = self.connector.get_optimizer_metadata(cls)
        if not meta:
            return

        for name, info in meta.items():
            label = name.replace("_", " ").capitalize() + ":"
            widget = self._create_optimizer_widget(name, info)
            self._optimizer_settings_layout.addRow(label, widget)
            self._optimizer_param_widgets[name] = widget

    def _create_optimizer_widget(self, name: str, info: dict) -> QWidget:
        """Create a widget based on metadata for optimizer settings."""
        w_type = info.get("type", "float")

        if w_type == "int":
            w = QSpinBox()
            w.setRange(0, 1000000)
            w.setValue(info.get("default", 1000))
            return w

        if w_type == "float":
            w = QDoubleSpinBox()
            w.setRange(0, 1e9)
            w.setDecimals(info.get("decimals", 4))
            w.setValue(info.get("default", 0.001))
            return w

        if w_type == "choice":
            w = QComboBox()
            w.addItems(info.get("options", []))
            idx = w.findText(info.get("default", ""))
            if idx >= 0:
                w.setCurrentIndex(idx)
            return w

        if w_type == "bool":
            w = QCheckBox()
            w.setChecked(info.get("default", True))
            return w

        return QLineEdit(str(info.get("default", "")))

    def _collect_optimizer_kwargs(self) -> dict:
        """Collect all optimizer settings from the dynamic widgets.

        Returns:
            A dict of ``param_name → value``.
        """
        kwargs = {}
        for name, widget in self._optimizer_param_widgets.items():
            if isinstance(widget, QSpinBox | QDoubleSpinBox):
                kwargs[name] = widget.value()
            elif isinstance(widget, QComboBox):
                kwargs[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                kwargs[name] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                kwargs[name] = widget.text()
        return kwargs

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Wire all internal signals."""
        self.btnAddVariable.clicked.connect(self._on_add_variable)
        self.btnRemoveVariable.clicked.connect(self._on_remove_variable)
        self.btnRefreshVariables.clicked.connect(self._refresh_variables_current_values)
        self.tblVariables.itemDoubleClicked.connect(self._on_variable_double_clicked)

        self.btnAddOperand.clicked.connect(self._on_add_operand)
        self.btnAutoGenerateOperands.clicked.connect(self._on_auto_generate_operands)
        self.btnRemoveOperand.clicked.connect(self._on_remove_operand)
        self.btnRefreshOperands.clicked.connect(self._refresh_operands_current_values)
        self.tblOperands.itemDoubleClicked.connect(self._on_operand_double_clicked)

        self.btnRun.clicked.connect(self._on_run)
        self.btnStop.clicked.connect(self._on_stop)

        # Connector signal → open add-variable dialog pre-filled
        self.connector.requestAddOptimizationVariable.connect(
            self.open_add_variable_dialog
        )
        # Reflect variable changes in the lens editor highlighting
        self.connector.optimizationVariablesChanged.connect(
            self._refresh_variables_table
        )

    # ------------------------------------------------------------------
    # Slots — Variables
    # ------------------------------------------------------------------

    @Slot()
    def _on_add_variable(self) -> None:
        """Open the Add Variable dialog from the panel button."""
        self.open_add_variable_dialog(surface_index=0, suggested_type="radius")

    @Slot(int, str)
    def open_add_variable_dialog(
        self, surface_index: int = 0, suggested_type: str = "radius"
    ) -> None:
        """Open the AddVariableDialog pre-filled with the given values.

        Args:
            surface_index: Pre-selected surface number.
            suggested_type: Pre-selected variable type key.
        """
        dlg = AddVariableDialog(
            self.connector, surface_index, suggested_type, parent=self
        )
        if dlg.exec() == QDialog.Accepted:
            self.connector.add_optimization_variable(dlg.get_variable_dict())
            self._refresh_variables_table()

    @Slot()
    def _on_remove_variable(self) -> None:
        row = self.tblVariables.currentRow()
        if row < 0:
            return
        self.connector.remove_optimization_variable(row)
        self._refresh_variables_table()

    @Slot(QTableWidgetItem)
    def _on_variable_double_clicked(self, item: QTableWidgetItem) -> None:
        """Edit the double-clicked variable."""
        row = item.row()
        variables = self.connector.get_optimization_variables()
        if 0 <= row < len(variables):
            vd = variables[row]
            dlg = AddVariableDialog(self.connector, initial_vd=vd, parent=self)
            if dlg.exec() == QDialog.Accepted:
                self.connector.set_optimization_variable(row, dlg.get_variable_dict())
                self._refresh_variables_table()

    @Slot()
    def _refresh_variables_table(self) -> None:
        """Reload the Variables table from the connector."""
        variables = self.connector.get_optimization_variables()
        self.tblVariables.setRowCount(len(variables))
        for i, vd in enumerate(variables):
            self.tblVariables.setVerticalHeaderItem(i, QTableWidgetItem(str(i)))
            cur_val = self.connector.get_variable_current_value(vd)
            cur_str = f"{cur_val:.4f}" if cur_val is not None else "N/A"
            coeff_str = (
                f" [{vd['coeff_number']}]" if vd.get("coeff_number") is not None else ""
            )
            self.tblVariables.setItem(
                i, 0, QTableWidgetItem(str(vd.get("surface_number", "")))
            )
            self.tblVariables.setItem(
                i, 1, QTableWidgetItem(vd.get("type", "") + coeff_str)
            )
            self.tblVariables.setItem(i, 2, QTableWidgetItem(cur_str))
            self.tblVariables.setItem(
                i, 3, QTableWidgetItem(str(vd.get("min_val", "")))
            )
            self.tblVariables.setItem(
                i, 4, QTableWidgetItem(str(vd.get("max_val", "")))
            )

    def _refresh_variables_current_values(self) -> None:
        """Update only the Current Value column (cheaper than full refresh)."""
        variables = self.connector.get_optimization_variables()
        for i, vd in enumerate(variables):
            if i >= self.tblVariables.rowCount():
                break
            cur_val = self.connector.get_variable_current_value(vd)
            cur_str = f"{cur_val:.4f}" if cur_val is not None else "N/A"
            self.tblVariables.setItem(i, 2, QTableWidgetItem(cur_str))

    # ------------------------------------------------------------------
    # Slots — Operands
    # ------------------------------------------------------------------

    @Slot()
    def _on_add_operand(self) -> None:
        dlg = AddOperandDialog(self.connector, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.connector.add_optimization_operand(dlg.get_operand_dict())
            self._refresh_operands_table()

    @Slot()
    def _on_auto_generate_operands(self) -> None:
        """Open the auto-generate operands dialog and process results."""
        dialog = AutoGenerateOperandsDialog(self.connector, self)
        if dialog.exec() == QDialog.Accepted:
            selection = dialog.get_selection()
            fields = selection["fields"]
            waves = selection["wavelengths"]
            metric = selection["metric"]
            num_rays = selection["num_rays"]
            dist = selection["distribution"]
            target = selection["target"]
            weight = selection["weight"]

            last_surface = self.connector.get_surface_count() - 1

            for hx, hy in fields:
                for wl in waves:
                    input_data = {
                        "Hx": hx,
                        "Hy": hy,
                        "num_rays": num_rays,
                        "wavelength": wl,
                        "distribution": dist,
                    }
                    if metric == "rms_spot_size":
                        input_data["surface_number"] = last_surface

                    op_dict = {
                        "type": metric,
                        "category": "Ray",
                        "target": target,
                        "weight": weight,
                        "input_data": input_data,
                    }
                    self.connector.add_optimization_operand(op_dict)

            self._refresh_operands_table()

    @Slot()
    def _on_remove_operand(self) -> None:
        row = self.tblOperands.currentRow()
        if row < 0:
            return
        self.connector.remove_optimization_operand(row)
        self._refresh_operands_table()

    @Slot(QTableWidgetItem)
    def _on_operand_double_clicked(self, item: QTableWidgetItem) -> None:
        """Edit the double-clicked operand."""
        row = item.row()
        operands = self.connector.get_optimization_operands()
        if 0 <= row < len(operands):
            od = operands[row]
            dlg = AddOperandDialog(self.connector, initial_od=od, parent=self)
            if dlg.exec() == QDialog.Accepted:
                self.connector.set_optimization_operand(row, dlg.get_operand_dict())
                self._refresh_operands_table()

    @Slot()
    def _refresh_operands_table(self) -> None:
        """Reload the Operands table from the connector."""
        operands = self.connector.get_optimization_operands()
        self.tblOperands.setRowCount(len(operands))
        for i, od in enumerate(operands):
            self.tblOperands.setVerticalHeaderItem(i, QTableWidgetItem(str(i)))
            cur_val = self.connector.get_operand_current_value(od)
            cur_str = f"{cur_val:.6f}" if cur_val is not None else "N/A"
            target_str = f"{od['target']:.6f}" if od.get("target") is not None else "—"

            self.tblOperands.setItem(i, 0, QTableWidgetItem(od.get("category", "")))
            self.tblOperands.setItem(i, 1, QTableWidgetItem(od.get("type", "")))
            self.tblOperands.setItem(i, 2, QTableWidgetItem(cur_str))
            self.tblOperands.setItem(i, 3, QTableWidgetItem(target_str))
            self.tblOperands.setItem(i, 4, QTableWidgetItem(str(od.get("weight", 1.0))))

            # Format parameters for display
            params = od.get("input_data", {})
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            self.tblOperands.setItem(i, 5, QTableWidgetItem(param_str))

    def _refresh_operands_current_values(self) -> None:
        """Update only the Current Value column (cheaper than full refresh)."""
        operands = self.connector.get_optimization_operands()
        for i, od in enumerate(operands):
            if i >= self.tblOperands.rowCount():
                break
            cur_val = self.connector.get_operand_current_value(od)
            cur_str = f"{cur_val:.6f}" if cur_val is not None else "N/A"
            self.tblOperands.setItem(i, 2, QTableWidgetItem(cur_str))

    # ------------------------------------------------------------------
    # Slots — Run / Stop
    # ------------------------------------------------------------------

    @Slot()
    def _on_run(self) -> None:
        if self.connector.is_optimization_running():
            return

        self.txtLog.clear()
        self._iteration_count = 0
        self.txtLog.append("Starting optimization...")
        self.btnRun.setEnabled(False)
        self.btnStop.setEnabled(True)

        cls = self.cmbAlgorithm.currentData()
        if cls is None:
            self.txtLog.append("Error: no optimizer selected.")
            self.btnRun.setEnabled(True)
            self.btnStop.setEnabled(False)
            return

        # Validate bounds requirements for the selected optimizer
        bounds_err = self.connector.validate_bounds_for_optimizer(cls)
        if bounds_err:
            self.txtLog.append(f"Error: {bounds_err}")
            tm = getattr(self.connector, "toast_manager", None)
            if tm is not None:
                tm.notify(bounds_err, "error")
            self.btnRun.setEnabled(True)
            self.btnStop.setEnabled(False)
            return

        # Validate operand input_data before starting
        for od in self.connector.get_optimization_operands():
            input_val = od.get("input_data") or od.get("input_data_str", "{}")
            err = self.connector.validate_operand_input_data(
                od.get("type", ""), input_val
            )
            if err:
                self.txtLog.append(f"Error: {err}")
                tm = getattr(self.connector, "toast_manager", None)
                if tm is not None:
                    tm.notify(err, "error")
                self.btnRun.setEnabled(True)
                self.btnStop.setEnabled(False)
                return

        optimizer_kwargs = self._collect_optimizer_kwargs()
        freq = self.spnFrequency.value()
        live_viz = self.chkLiveViz.isChecked()
        live_vars = self.chkLiveVars.isChecked()

        def on_progress(n: int) -> None:
            self._iteration_count = n
            if n % freq == 0:
                if live_viz:
                    self.connector.opticChanged.emit()
                if live_vars:
                    self._refresh_variables_current_values()

        def on_finished(summary: str) -> None:
            self.txtLog.append(summary)
            self.btnRun.setEnabled(True)
            self.btnStop.setEnabled(False)
            self.connector.opticChanged.emit()
            self._refresh_variables_table()
            tm = getattr(self.connector, "toast_manager", None)
            if tm is not None:
                first_line = (
                    summary.splitlines()[0] if summary else "Optimization complete"
                )
                tm.notify(first_line, "success")

        def on_error(message: str) -> None:
            self.txtLog.append(f"Error: {message}")
            self.btnRun.setEnabled(True)
            self.btnStop.setEnabled(False)
            tm = getattr(self.connector, "toast_manager", None)
            if tm is not None:
                tm.notify(f"Optimization failed: {message}", "error")

        self.connector.run_optimization(
            cls, optimizer_kwargs, on_progress, on_finished, on_error
        )

    @Slot()
    def _on_stop(self) -> None:
        self.connector.stop_optimization()
        self.txtLog.append(
            f"Stop requested. Iterations so far: {self._iteration_count}"
        )

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def update_theme(self, theme_name: str) -> None:
        """Propagate theme changes.  Reserved for future embedded plots.

        Args:
            theme_name: ``"dark"`` or ``"light"``.
        """
