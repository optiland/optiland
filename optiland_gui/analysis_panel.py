"""Defines the main analysis panel for the Optiland GUI.

This module contains the `AnalysisPanel` widget, which is the primary interface
for performing and visualizing optical analyses such as spot diagrams, ray fans,
and MTF plots. It handles dynamic settings generation, plot display, and user
interactions for all supported analysis types.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import ast
import contextlib
import copy
import inspect
import json
from enum import Enum
from typing import TYPE_CHECKING, Literal, get_args, get_origin, get_type_hints

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRegularExpression,
    Qt,
    QTimer,
    Slot,
)
from PySide6.QtGui import QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from optiland.analysis import (
    Distortion,
    EncircledEnergy,
    FieldCurvature,
    FieldIncidentAngleVsHeight,
    GridDistortion,
    PupilAberration,
    PupilIncidentAngleVsHeight,
    RayFan,
    RmsSpotSizeVsField,
    RmsWavefrontErrorVsField,
    SpotDiagram,
    ThroughFocusSpotDiagram,
    YYbar,
)
from optiland.mtf import FFTMTF, GeometricMTF

from . import gui_plot_utils

if TYPE_CHECKING:
    from .optiland_connector import OptilandConnector


class CustomMatplotlibToolbar(NavigationToolbar):
    """A custom Matplotlib toolbar with styleable buttons.

    This toolbar assigns unique object names to its buttons, allowing them to be
    styled individually using Qt Style Sheets (QSS). This is useful for creating
    a consistent look and feel that matches the application's theme.

    Args:
        canvas: The Matplotlib canvas to which this toolbar is attached.
        parent: The parent widget.
    """

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent, coordinates=False)

        # Assign unique object names to each tool button
        for action in self.actions():
            tooltip = action.toolTip()
            if tooltip:
                action_id = tooltip.split(" ")[0].replace("(", "").replace(")", "")
                button_widget = self.widgetForAction(action)
                if button_widget:
                    button_widget.setObjectName(f"MPL{action_id}Button")


class AnalysisPanel(QWidget):
    """A comprehensive panel for running and displaying various optical analyses.

    This widget serves as the main user interface for all optical analysis tasks.
    It features a dropdown to select the analysis type, a central area for
    displaying plots and results, and a collapsible side panel for configuring
    analysis-specific settings. It also includes controls for running, stopping,
    and managing analysis results.

    Attributes:
        ANALYSIS_MAP (dict): A mapping from analysis names to their corresponding
                             classes in the `optiland.analysis` module.
        connector (OptilandConnector): An object that handles communication with
                                       the main Optiland backend.
        current_theme (str): The name of the current UI theme (e.g., "dark").
        analysis_results_pages (list): A cache for storing generated plot pages.
        current_plot_page_index (int): The index of the currently displayed plot page.
    """

    GEOMETRIC_MTF = "Geometric MTF"
    FFT_MTF = "FFT MTF"
    ANALYSIS_ERROR_TITLE = "Analysis Error"
    JSON_FILE_FILTER = "JSON Files (*.json);;All Files (*)"

    ANALYSIS_MAP = {
        "Spot Diagram": SpotDiagram,
        "Ray Fan": RayFan,
        "Angle vs Height (Scan Pupil)": PupilIncidentAngleVsHeight,
        "Angle vs Height (Scan Field)": FieldIncidentAngleVsHeight,
        "Distortion Plot": Distortion,
        "Grid Distortion": GridDistortion,
        "Field Curvature": FieldCurvature,
        "Encircled Energy": EncircledEnergy,
        "RMS Spot Size vs Field": RmsSpotSizeVsField,
        "RMS Wavefront Error vs Field": RmsWavefrontErrorVsField,
        "Through-Focus Spot Diagram": ThroughFocusSpotDiagram,
        # "Incoherent Irradiance": IncoherentIrradiance,
        "Pupil Aberration": PupilAberration,
        GEOMETRIC_MTF: GeometricMTF,
        FFT_MTF: FFTMTF,
        "YYbar": YYbar,
    }

    def __init__(self, connector: OptilandConnector, parent=None):
        """Initializes the AnalysisPanel."""
        super().__init__(parent)
        self._init_attributes(connector)
        self._setup_main_layout()

        self._setup_top_bar()
        self._setup_main_content_area()
        self._setup_log_area()

        self._connect_signals()
        self._set_initial_state()

    def _init_attributes(self, connector):
        """Initializes instance attributes."""
        self.current_theme = "dark"
        self.connector = connector
        self.setWindowTitle("Analysis")
        self.setObjectName("AnalysisPanel")
        gui_plot_utils.apply_gui_matplotlib_styles()

        self.analysis_results_pages = []
        self.current_plot_page_index = -1
        self.active_mpl_canvas_widget = None
        self.active_mpl_toolbar_widget = None
        self.motion_notify_cid = None
        self.current_settings_widgets = {}
        # Mapping of display name → class, built from the registry at init.
        self._analysis_class_map: dict[str, type] = {}

    def _setup_main_layout(self):
        """Sets up the main QVBoxLayout for the panel."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

    def _setup_top_bar(self):
        """Creates the top bar with analysis selection and control buttons."""
        top_bar_layout = QHBoxLayout()
        top_bar_layout.addWidget(QLabel("Analysis Type:"))
        self.analysisTypeCombo = QComboBox()
        self.analysisTypeCombo.setObjectName("AnalysisTypeCombo")
        self._build_analysis_class_map()
        self._populate_analysis_combo()
        top_bar_layout.addWidget(self.analysisTypeCombo)
        top_bar_layout.addSpacerItem(
            QSpacerItem(
                20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )

        self.btnRun = QPushButton()
        self.btnRun.setObjectName("RunAnalysisButton")
        self.btnRun.setToolTip("Run Selected Analysis")
        self.btnRun.setFixedSize(25, 25)
        self.btnRunAll = QPushButton()
        self.btnRunAll.setObjectName("RunAllAnalysisButton")
        self.btnRunAll.setFixedSize(25, 25)
        self.btnStop = QPushButton()
        self.btnStop.setObjectName("StopAnalysisButton")
        self.btnStop.setToolTip("Stop Analysis")
        self.btnStop.setFixedSize(25, 25)

        top_bar_layout.addWidget(self.btnRun)
        top_bar_layout.addWidget(self.btnRunAll)
        top_bar_layout.addWidget(self.btnStop)
        self.main_layout.addLayout(top_bar_layout)

    def _build_analysis_class_map(self) -> None:
        """Build ``_analysis_class_map`` from the analysis registry.

        Calls ``AnalysisRunner.get_analysis_registry`` via the connector and
        populates ``self._analysis_class_map`` with
        ``{display_name: cls}`` entries.  Falls back to ``ANALYSIS_MAP`` if
        the registry returns nothing (e.g. during unit tests with a stub
        connector).
        """
        registry = self.connector._analysis_runner.get_analysis_registry()
        if registry:
            self._analysis_class_map = {name: cls for _, name, cls in registry}
        else:
            self._analysis_class_map = dict(self.ANALYSIS_MAP)

    def _populate_analysis_combo(self) -> None:
        """Populate ``analysisTypeCombo`` with grouped entries from the registry.

        Category names are inserted as bold, disabled header items.
        Selectable analysis entries follow each category header.  If no
        registry entries are available the combo falls back to a flat list
        from ``ANALYSIS_MAP``.
        """
        self.analysisTypeCombo.clear()
        registry = self.connector._analysis_runner.get_analysis_registry()
        if not registry:
            self.analysisTypeCombo.addItems(list(self.ANALYSIS_MAP.keys()))
            return

        model = self.analysisTypeCombo.model()
        current_category: str | None = None
        for category, name, _cls in registry:
            if category != current_category:
                self.analysisTypeCombo.addItem(category)
                header_idx = self.analysisTypeCombo.count() - 1
                header_item = model.item(header_idx)
                header_item.setEnabled(False)
                font = header_item.font()
                font.setBold(True)
                header_item.setFont(font)
                current_category = category
            self.analysisTypeCombo.addItem(name)

        # Select the first selectable item
        for i in range(self.analysisTypeCombo.count()):
            if self.analysisTypeCombo.model().item(i).isEnabled():
                self.analysisTypeCombo.setCurrentIndex(i)
                break

    def _setup_main_content_area(self):
        """Sets up the central area containing the plot and settings panels."""
        main_separator_line = QFrame()
        main_separator_line.setObjectName("MainSeparatorLine")
        main_separator_line.setFrameShape(QFrame.Shape.HLine)
        main_separator_line.setFrameShadow(QFrame.Shadow.Sunken)
        self.main_layout.addWidget(main_separator_line)

        main_content_layout = QHBoxLayout()
        main_content_layout.setSpacing(10)

        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.setInterval(100)
        self.resize_timer.timeout.connect(self.handle_resize_finished)

        self._setup_plot_display_frame(main_content_layout)
        self._setup_settings_panel(main_content_layout)

        self.main_layout.addLayout(main_content_layout, 1)

    def _setup_plot_display_frame(self, parent_layout):
        """Creates the plot display frame, including its title bar and content area."""
        self.plot_display_frame = QFrame()
        self.plot_display_frame.setObjectName("PlotDisplayFrame")
        plot_display_frame_layout = QVBoxLayout(self.plot_display_frame)
        plot_display_frame_layout.setContentsMargins(5, 5, 5, 5)

        # Plot area title bar
        self._setup_plot_title_bar(plot_display_frame_layout)

        # Separator line
        title_plot_separator_line = QFrame()
        title_plot_separator_line.setFrameShape(QFrame.Shape.HLine)
        plot_display_frame_layout.addWidget(title_plot_separator_line)

        # Plot content and pagination
        self._setup_plot_content_area(plot_display_frame_layout)

        parent_layout.addWidget(self.plot_display_frame, 3)

    def _setup_plot_title_bar(self, parent_layout):
        """Creates the title bar for the plot area."""
        self.plot_area_title_bar_layout = QHBoxLayout()
        self.plotTitleLabel = QLabel("No Analysis Run")
        self.plotTitleLabel.setObjectName("PlotTitleLabel")
        self.plot_area_title_bar_layout.addWidget(self.plotTitleLabel)

        self.mpl_toolbar_in_titlebar_container = QWidget()
        self.mpl_toolbar_in_titlebar_container.setObjectName(
            "MPLToolbarInTitlebarContainer"
        )
        self.mpl_toolbar_in_titlebar_layout = QHBoxLayout(
            self.mpl_toolbar_in_titlebar_container
        )
        self.mpl_toolbar_in_titlebar_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_area_title_bar_layout.addWidget(
            self.mpl_toolbar_in_titlebar_container
        )
        self.mpl_toolbar_in_titlebar_container.setVisible(False)
        self.plot_area_title_bar_layout.addStretch()

        self.btnRefreshPlot = QPushButton()
        self.btnRefreshPlot.setObjectName("RefreshPlotButton")
        self.btnRefreshPlot.setFixedSize(25, 25)
        self.plot_area_title_bar_layout.addWidget(self.btnRefreshPlot)

        self.toggleSettingsButton = QPushButton()
        self.toggleSettingsButton.setObjectName("ToggleSettingsButton")
        self.toggleSettingsButton.setFixedSize(25, 25)
        self.plot_area_title_bar_layout.addWidget(self.toggleSettingsButton)

        parent_layout.addLayout(self.plot_area_title_bar_layout)

    def _setup_plot_content_area(self, parent_layout):
        """Creates the main plot content area, info labels, and page buttons."""
        plot_content_and_pages_layout = QHBoxLayout()
        plot_content_and_pages_layout.setContentsMargins(0, 0, 0, 0)

        plot_and_info_widget = QWidget()
        plot_and_info_layout = QVBoxLayout(plot_and_info_widget)
        plot_and_info_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_container_widget = QWidget()
        self.plot_container_widget.setObjectName("PlotContainerWidget")
        self.plot_container_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_and_info_layout.addWidget(self.plot_container_widget, 1)

        self.cursor_coord_label = QLabel("", self.plot_container_widget)
        self.cursor_coord_label.setObjectName("CursorCoordLabel")
        self.cursor_coord_label.setStyleSheet(
            "background-color:rgba(0,0,0,0.65);"
            "color:white;padding:2px 4px;border-radius:3px;"
        )
        self.cursor_coord_label.setVisible(False)
        self.cursor_coord_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )

        self.dataInfoLabel = QLabel("Data Analysis info will appear here.")
        self.dataInfoLabel.setObjectName("DataInfoLabel")
        plot_and_info_layout.addWidget(self.dataInfoLabel)

        plot_content_and_pages_layout.addWidget(plot_and_info_widget, 1)

        self._setup_pagination_controls(plot_content_and_pages_layout)

        parent_layout.addLayout(plot_content_and_pages_layout, 1)

    def _setup_pagination_controls(self, parent_layout):
        """Creates the vertical pagination buttons on the right of the plot."""
        self.page_buttons_scroll_area = QScrollArea()
        self.page_buttons_scroll_area.setObjectName("PageButtonsScrollArea")
        self.page_buttons_scroll_area.setWidgetResizable(True)
        self.page_buttons_scroll_area.setFixedWidth(30)
        self.page_buttons_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        page_buttons_container_widget = QWidget()
        self.vertical_page_buttons_layout = QVBoxLayout(page_buttons_container_widget)
        self.vertical_page_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.page_buttons_scroll_area.setWidget(page_buttons_container_widget)

        parent_layout.addWidget(self.page_buttons_scroll_area)

    def _setup_settings_panel(self, parent_layout):
        """Creates the collapsible settings panel on the right."""
        self.settings_area_widget = QWidget()
        self.settings_area_widget.setObjectName("SettingsArea")
        self.settings_area_widget.setFixedWidth(250)
        settings_layout = QVBoxLayout(self.settings_area_widget)
        settings_layout.setContentsMargins(5, 5, 5, 5)
        settings_layout.addWidget(QLabel("Analysis Settings"))

        self.settings_scroll_area = QScrollArea()
        self.settings_scroll_area.setWidgetResizable(True)
        self.settingsContentWidget = QWidget()
        self.settings_form_layout = QFormLayout(self.settingsContentWidget)
        self.settings_scroll_area.setWidget(self.settingsContentWidget)
        settings_layout.addWidget(self.settings_scroll_area, 1)

        settings_button_layout = QHBoxLayout()
        self.btnApplySettings = QPushButton()
        self.btnApplySettings.setObjectName("ApplySettingsButton")
        self.btnApplySettings.setToolTip("Apply current settings and rerun analysis")
        settings_button_layout.addWidget(self.btnApplySettings)

        self.btnSaveSettings = QPushButton()
        self.btnSaveSettings.setObjectName("SaveSettingsButton")
        self.btnSaveSettings.setToolTip("Save current analysis settings to a file")
        settings_button_layout.addWidget(self.btnSaveSettings)

        self.btnLoadSettings = QPushButton()
        self.btnLoadSettings.setObjectName("LoadSettingsButton")
        self.btnLoadSettings.setToolTip("Load analysis settings from a file")
        settings_button_layout.addWidget(self.btnLoadSettings)

        settings_layout.addLayout(settings_button_layout)
        parent_layout.addWidget(self.settings_area_widget, 1)

    def _setup_log_area(self):
        """Creates the log text area at the bottom."""
        self.logArea = QTextEdit()
        self.logArea.setObjectName("LogArea")
        self.logArea.setReadOnly(True)
        self.logArea.setFixedHeight(60)
        self.main_layout.addWidget(self.logArea)

    def _connect_signals(self):
        """Connects all widget signals to their corresponding slots."""
        self.btnRun.clicked.connect(self.run_analysis_slot)
        self.btnRunAll.clicked.connect(self.run_all_analysis_slot)
        self.btnStop.clicked.connect(self.stop_analysis_slot)
        self.analysisTypeCombo.currentTextChanged.connect(self.on_analysis_type_changed)
        self.toggleSettingsButton.clicked.connect(self.toggle_settings_panel_slot)
        self.btnRefreshPlot.clicked.connect(self._refresh_current_plot_page_slot)
        self.btnApplySettings.clicked.connect(
            self._apply_settings_and_rerun_analysis_slot
        )
        self.btnSaveSettings.clicked.connect(self._save_analysis_settings_slot)
        self.btnLoadSettings.clicked.connect(self._load_analysis_settings_slot)

    def _set_initial_state(self):
        """Sets the initial visibility and state of widgets."""
        self.update_theme()
        self.on_analysis_type_changed(self.analysisTypeCombo.currentText())
        self.update_pagination_ui()
        self.display_plot_page(self.current_plot_page_index)
        self.settings_area_widget.setVisible(False)

    # --- Layout and Widget Management ---
    def _cleanup_figure_canvas(self, canvas_widget: FigureCanvas):
        """Safely disconnects signals and closes a Matplotlib FigureCanvas."""
        if hasattr(canvas_widget, "_event_cids"):
            for cid in canvas_widget._event_cids:
                with contextlib.suppress(TypeError, RuntimeError):
                    canvas_widget.mpl_disconnect(cid)
            canvas_widget._event_cids = []
        plt.close(canvas_widget.figure)

    def _clear_layout(self, layout_to_clear):
        """Iteratively clears all widgets and sub-layouts from a given layout."""
        if layout_to_clear is None:
            return

        while (item := layout_to_clear.takeAt(0)) is not None:
            if (widget := item.widget()) is not None:
                if isinstance(widget, FigureCanvas):
                    self._cleanup_figure_canvas(widget)
                widget.setParent(None)
                widget.deleteLater()
            elif (sub_layout := item.layout()) is not None:
                self._clear_layout(sub_layout)

    # --- Settings Widget Creation ---
    def _create_combobox_for_parameter(self, param_name, default_value):
        """Creates and configures a QComboBox for a given parameter."""
        widget = QComboBox()
        if param_name == "fields":
            options = self.connector.get_field_options()
            for display_name, value_str in options:
                widget.addItem(display_name, userData=value_str)
            all_index = widget.findText("all")
            if all_index != -1:
                widget.setCurrentIndex(all_index)
        elif param_name in ["wavelengths", "wavelength"]:
            options = self.connector.get_wavelength_options()
            for display_name, value_str in options:
                widget.addItem(display_name, userData=value_str)
            default_index = widget.findText(str(default_value))
            if default_index != -1:
                widget.setCurrentIndex(default_index)
        elif param_name == "axis":
            widget.addItems(["Y-Axis (1)", "X-Axis (0)"])
            if default_value is not None:
                widget.setCurrentIndex(0 if default_value == 1 else 1)
        return widget

    def _create_spinbox_for_parameter(self, param_name, default_value, annotation):
        """Creates and configures a QSpinBox or QDoubleSpinBox."""
        if annotation is int:
            widget = QSpinBox()
            ranges = {
                "num_rays": (1, 10000000),
                "num_points": (1, 10000000),
                "num_rings": (1, 1024),
                "num_fields": (1, 1024),
                "num_steps": (1, 51),
                "surface_idx": (-100, 1000),
                "detector_surface": (-100, 1000),
                "grid_size": (32, 8192),
            }
            min_v, max_v = ranges.get(param_name, (-1000000, 1000000))
            step_v = 32 if param_name == "grid_size" else 1
            widget.setRange(min_v, max_v)
            widget.setSingleStep(step_v)
            widget.setValue(int(default_value) if default_value is not None else 0)
        else:  # float
            widget = QDoubleSpinBox()
            widget.setDecimals(4)

            # Normalized coordinate ranges
            norm_params = [
                "hx",
                "hy",
                "px",
                "py",
                "field_x",
                "field_y",
                "pupil_x",
                "pupil_y",
            ]
            if param_name.lower() in norm_params:
                widget.setRange(-1.0, 1.0)
            else:
                widget.setRange(-1e9, 1e11)

            widget.setSingleStep(0.01 if "delta_focus" in param_name else 0.1)
            widget.setValue(float(default_value) if default_value is not None else 0.0)
        return widget

    def _create_widget_for_string_parameter(self, param_name, default_value):
        """Creates a QComboBox or QLineEdit for a string parameter."""
        combo_options = {
            "distribution": [
                "hexapolar",
                "grid",
                "random",
                "ring",
                "line_x",
                "line_y",
                "gaussian",
                "uniform",
            ],
            "coordinates": ["local", "global"],
            "distortion_type": ["f-tan", "f-theta"],
            "cmap": ["inferno", "viridis", "plasma", "magma", "gray", "jet"],
            "strategy": ["chief_ray", "centroid", "best_fit"],
            "reference": ["chief_ray", "centroid"],
            "zernike_type": ["fringe", "standard", "noll"],
        }
        if param_name in combo_options:
            widget = QComboBox()
            widget.addItems(combo_options[param_name])

            # If default_value is an Enum, use its value or name for matching
            if isinstance(default_value, Enum):
                match_val = str(default_value.value)
            else:
                match_val = str(default_value)

            widget.setCurrentText(match_val if default_value else widget.itemText(0))
        else:
            if isinstance(default_value, Enum):
                text = str(default_value.value)
            else:
                text = str(default_value) if default_value is not None else ""
            widget = QLineEdit(text)
        return widget

    def _prepare_param_details(self, param_name, param_info, default_override=None):
        """Prepares the label, default value, and annotation for a parameter."""
        label_text = param_name.replace("_", " ").title() + ":"
        default_value = (
            default_override
            if default_override is not None
            else param_info.get("default")
        )

        annotation = param_info.get("annotation")
        if isinstance(annotation, str):
            # Forward-reference string — map common patterns to actual types.
            ann_lower = annotation.lower()
            if "tuple" in ann_lower:
                annotation = tuple
            elif annotation in ("int",):
                annotation = int
            elif annotation in ("float",):
                annotation = float
            elif annotation in ("bool",):
                annotation = bool
            elif annotation in ("str",):
                annotation = str
            else:
                # Unknown string annotation — infer from the default value.
                annotation = None

        if annotation in (inspect.Parameter.empty, None):
            if isinstance(default_value, bool):
                annotation = bool
            elif isinstance(default_value, int):
                annotation = int
            elif isinstance(default_value, float):
                annotation = float
            elif isinstance(default_value, str):
                annotation = str

        if param_name == "max_freq":
            annotation = str
        if param_name == "grid_size":
            annotation = int
            if default_value in [None, inspect.Parameter.empty]:
                default_value = 128
        if param_name in ["field", "pupil"] and default_value in [
            None,
            inspect.Parameter.empty,
            "",
        ]:
            annotation = tuple
            default_value = (0.0, 0.0)

        # wavelength (singular) is required — default to primary
        if param_name == "wavelength" and default_value in [
            None,
            inspect.Parameter.empty,
            "",
        ]:
            annotation = str
            default_value = "primary"
        # MMDFTPSF requires explicit image_size/pixel_pitch when they are None
        if param_name == "image_size" and default_value in [
            None,
            inspect.Parameter.empty,
        ]:
            annotation = int
            default_value = 128
        if param_name == "pixel_pitch" and default_value in [
            None,
            inspect.Parameter.empty,
        ]:
            annotation = float
            default_value = 5e-6

        return label_text, default_value, annotation

    def _create_widget_for_param(self, param_name, annotation, default_value):
        """Factory function to create the appropriate widget for a parameter."""
        widget = None
        if get_origin(annotation) is Literal:
            widget = self._create_literal_combobox(annotation, default_value)
        elif param_name in ["fields", "wavelengths", "wavelength", "axis"]:
            widget = self._create_combobox_for_parameter(param_name, default_value)
        elif annotation in [int, float]:
            widget = self._create_spinbox_for_parameter(
                param_name, default_value, annotation
            )
        elif annotation is bool:
            return self._create_checkbox(param_name, default_value)
        elif annotation is str:
            widget = self._create_widget_for_string_parameter(param_name, default_value)
        elif annotation is tuple or get_origin(annotation) is tuple:
            widget = self._create_tuple_line_edit(default_value)
        return widget

    def _create_literal_combobox(self, annotation, default_value):
        """Creates a QComboBox from a Literal type annotation."""
        widget = QComboBox()
        options = get_args(annotation)
        widget.addItems([str(opt) for opt in options])
        if str(default_value) in [str(o) for o in options]:
            widget.setCurrentText(str(default_value))
        return widget

    def _create_checkbox(self, param_name, default_value):
        """Creates a QCheckBox for a boolean parameter."""
        widget = QCheckBox(param_name.replace("_", " ").title())
        widget.setChecked(bool(default_value) if default_value is not None else False)
        return widget

    def _create_tuple_line_edit(self, default_value):
        """Creates a QLineEdit for a tuple parameter with validation."""
        widget = QLineEdit(", ".join(map(str, default_value)) if default_value else "")
        widget.setPlaceholderText("e.g., 0, 0.5 or 128, 128")

        # Regex for one or more numbers separated by commas/spaces
        regex = QRegularExpression(r"^[-+]?[\d\.]+\s*(,\s*[-+]?[\d\.]+\s*)*$")
        validator = QRegularExpressionValidator(regex, widget)
        widget.setValidator(validator)
        return widget

    def _add_setting_widget(
        self, param_name: str, param_info: dict, default_value_override=None
    ):
        """Adds a settings widget to the form layout for a given parameter."""
        if param_name == "kwargs":
            return

        label_text, default_value, annotation = self._prepare_param_details(
            param_name, param_info, default_value_override
        )
        widget = self._create_widget_for_param(param_name, annotation, default_value)

        if widget:
            if isinstance(widget, QCheckBox):
                self.settings_form_layout.addRow(widget)
            else:
                self.settings_form_layout.addRow(QLabel(label_text), widget)
            self.current_settings_widgets[param_name] = widget
        else:
            print(
                f"Warning: No widget created for '{param_name}' "
                f"(annotation: {annotation})"
            )

    def _get_analysis_params(self, analysis_class):
        """Gets constructor and view parameters for a given analysis class."""
        try:
            module = inspect.getmodule(analysis_class)
            resolved_hints = get_type_hints(
                analysis_class.__init__, globalns=getattr(module, "__dict__", None)
            )
        except (TypeError, NameError):
            resolved_hints = {}

        init_params = gui_plot_utils.get_analysis_parameters(analysis_class)
        for name, info in init_params.items():
            # Only overwrite with the resolved hint when it was actually resolved;
            # otherwise keep the raw annotation (may be a forward-ref string).
            resolved = resolved_hints.get(name)
            if resolved is not None:
                info["annotation"] = resolved

        if analysis_class.__name__ in [self.GEOMETRIC_MTF, self.FFT_MTF]:
            if "grid_size" not in init_params:
                init_params["grid_size"] = {"default": 128, "annotation": int}
            if (
                analysis_class.__name__ == self.FFT_MTF
                and "max_freq" not in init_params
            ):
                init_params["max_freq"] = {"default": "cutoff", "annotation": str}

        view_params = {}
        if hasattr(analysis_class, "view") and callable(analysis_class.view):
            view_sig = inspect.signature(analysis_class.view)
            view_arg_defaults = {
                "add_airy_disk": (bool, False),
                "cmap": (str, "inferno"),
                "normalize": (bool, True),
                "cross_section": (str, ""),
            }
            for arg, (arg_type, default) in view_arg_defaults.items():
                if arg in view_sig.parameters and arg not in init_params:
                    view_params[arg] = {"default": default, "annotation": arg_type}

        return init_params, view_params

    def _update_settings_ui(self, analysis_name: str):
        """Updates the settings panel with widgets for the selected analysis."""
        while self.settings_form_layout.rowCount() > 0:
            self.settings_form_layout.removeRow(0)
        self.current_settings_widgets.clear()

        analysis_class = self._analysis_class_map.get(analysis_name)
        if not analysis_class:
            self.settings_form_layout.addRow(QLabel("No settings available."))
            return

        init_params, view_params = self._get_analysis_params(analysis_class)

        for param_name, param_info in init_params.items():
            self._add_setting_widget(param_name, param_info)

        for param_name, param_info in view_params.items():
            self._add_setting_widget(param_name, param_info)

        self.settings_form_layout.addItem(
            QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

    def _set_line_edit_value(self, widget, value):
        """Sets the text of a QLineEdit, handling tuples."""
        text = ", ".join(map(str, value)) if isinstance(value, tuple) else str(value)
        widget.setText(text)

    def _set_combobox_value(self, widget, value, param_name):
        """Sets the value of a QComboBox, dispatching to a special handler if needed."""
        if param_name in ["fields", "wavelengths", "wavelength"]:
            self._set_special_combobox_value(widget, value)
        elif param_name == "axis":
            widget.setCurrentIndex(0 if value == 1 else 1)
        else:
            index = widget.findText(str(value))
            if index != -1:
                widget.setCurrentIndex(index)

    def _set_special_combobox_value(self, widget, value):
        """Sets the value for a QComboBox that uses itemData."""
        for i in range(widget.count()):
            try:
                item_data = ast.literal_eval(widget.itemData(i) or "None")
                if item_data == value:
                    widget.setCurrentIndex(i)
                    return
            except (ValueError, SyntaxError):
                continue

    def _set_widget_value(self, widget, value, param_name: str):
        """
        Sets the value of a widget by dispatching to the correct handler
        based on its type.
        """
        # A dictionary mapping widget types to their value-setting functions
        handler_map = {
            QSpinBox: lambda w, v, p: w.setValue(v),
            QDoubleSpinBox: lambda w, v, p: w.setValue(v),
            QCheckBox: lambda w, v, p: w.setChecked(bool(v)),
            QLineEdit: lambda w, v, p: self._set_line_edit_value(w, v),
            QComboBox: self._set_combobox_value,
        }

        # Get the correct handler function for the widget's type
        handler = handler_map.get(type(widget))

        # If a handler is found, call it with the required arguments
        if handler:
            handler(widget, value, param_name)

    def _set_combobox_value(self, widget, value, param_name):
        """Sets the value for a QComboBox based on parameter type."""
        if param_name in ["fields", "wavelengths", "wavelength"]:
            self._set_special_combobox_value(widget, value)
        elif param_name == "axis":
            widget.setCurrentIndex(0 if value == 1 else 1)
        else:
            index = widget.findText(str(value))
            if index != -1:
                widget.setCurrentIndex(index)

    @Slot(str)
    def on_analysis_type_changed(self, analysis_name: str):
        """Handles the change of the selected analysis type.

        This slot is connected to the ``currentTextChanged`` signal of the
        analysis type combo box.  Category header items are non-selectable,
        but if one is somehow reached this slot returns early without
        updating the UI.

        Args:
            analysis_name: The new analysis name selected in the combo box.
        """
        if analysis_name not in self._analysis_class_map:
            return  # category header or unrecognised item — skip
        self._update_settings_ui(analysis_name)
        if self.current_plot_page_index == -1 or not self.analysis_results_pages:
            self.plotTitleLabel.setText(analysis_name)

    def update_theme(self, theme="dark"):
        """Updates themes for icons AND all plots in the analysis panel."""

        theme_name = "dark" if "dark" in theme.lower() else "light"

        self.current_theme = theme_name
        refresh_icon_path = f":/icons/{self.current_theme}/refresh.svg"
        self.btnRefreshPlot.setIcon(QIcon(refresh_icon_path))
        settings_icon_path = f":/icons/{self.current_theme}/settings.svg"
        self.toggleSettingsButton.setIcon(QIcon(settings_icon_path))
        self.btnRun.setIcon(QIcon(f":/icons/{theme_name}/run.svg"))
        self.btnStop.setIcon(QIcon(f":/icons/{theme_name}/stop.svg"))
        self.btnRunAll.setIcon(QIcon(f":/icons/{theme_name}/run_all.svg"))
        self.btnApplySettings.setIcon(QIcon(f":/icons/{theme_name}/check_apply.svg"))
        self.btnSaveSettings.setIcon(QIcon(f":/icons/{theme_name}/save_settings.svg"))
        self.btnLoadSettings.setIcon(QIcon(f":/icons/{theme_name}/load_settings.svg"))

        # This new line will refresh the plot using the new theme
        self._refresh_current_plot_page_slot()

    def update_pagination_ui(self):
        self._clear_layout(self.vertical_page_buttons_layout)
        for i, _page_data in enumerate(self.analysis_results_pages):
            btn_page = QPushButton(str(i + 1))
            btn_page.setObjectName(f"PageButton_{i + 1}")
            btn_page.setCheckable(True)
            btn_page.setChecked(i == self.current_plot_page_index)
            btn_page.clicked.connect(
                lambda checked=False, index=i: self.switch_plot_page(index)
            )
            btn_page.setContextMenuPolicy(Qt.CustomContextMenu)
            btn_page.customContextMenuRequested.connect(
                lambda pos, index=i, btn=btn_page: self._show_page_button_context_menu(
                    pos, btn, index
                )
            )
            self.vertical_page_buttons_layout.addWidget(btn_page)
        self.vertical_page_buttons_layout.addStretch()

    def _show_page_button_context_menu(self, position, button, page_index):
        """Creates and shows the right-click menu for a page button."""
        menu = QMenu()
        clone_action = menu.addAction("Clone Analysis")
        remove_action = menu.addAction("Remove Analysis")
        undock_action = menu.addAction("Undock (WIP)")
        undock_action.setEnabled(False)

        action = menu.exec(button.mapToGlobal(position))

        if action == clone_action:
            self._clone_analysis_page(page_index)
        elif action == remove_action:
            self._remove_analysis_page(page_index)

    def _clone_analysis_page(self, page_index):
        """Clones an existing analysis page."""
        if not (0 <= page_index < len(self.analysis_results_pages)):
            return

        original_page_data = self.analysis_results_pages[page_index]
        cloned_page_data = {
            "name": original_page_data["name"],
            "analysis_instance": copy.deepcopy(original_page_data["analysis_instance"]),
            "plot_type": original_page_data["plot_type"],
            "view_args": copy.deepcopy(original_page_data["view_args"]),
            "constructor_args_used": copy.deepcopy(
                original_page_data["constructor_args_used"]
            ),
            "figsize": original_page_data.get("figsize"),
        }

        self.analysis_results_pages.append(cloned_page_data)
        self.update_pagination_ui()
        self.switch_plot_page(len(self.analysis_results_pages) - 1)
        self.logArea.append("Analysis cloned successfully.")

    def _remove_analysis_page(self, page_index):
        """Removes an analysis page from the analysis results."""
        if 0 <= page_index < len(self.analysis_results_pages):
            self.analysis_results_pages.pop(page_index)

            # If the current page was removed, switch to the nearest available page
            if self.current_plot_page_index == page_index:
                if not self.analysis_results_pages:
                    self.current_plot_page_index = -1
                elif self.current_plot_page_index >= len(self.analysis_results_pages):
                    self.current_plot_page_index = len(self.analysis_results_pages) - 1
            # If a later page was removed, current_plot_page_index stays the same
            # If an earlier page was removed, shift current_plot_page_index back 1
            elif self.current_plot_page_index > page_index:
                self.current_plot_page_index -= 1

            self.update_pagination_ui()
            self.display_plot_page(self.current_plot_page_index)
            self.logArea.append("Analysis removed.")

    def resizeEvent(self, event):
        """Restarts a timer every time the window is resized."""
        super().resizeEvent(event)
        self.resize_timer.start()

    def handle_resize_finished(self):
        """
        Called after the user has finished resizing the window.
        Applies tight_layout to the current plot.
        """
        if self.active_mpl_canvas_widget:
            try:
                self.active_mpl_canvas_widget.figure.tight_layout()
                self.active_mpl_canvas_widget.draw_idle()
            except Exception as e:
                print(f"Error applying tight_layout on resize: {e}")

    def switch_plot_page(self, page_index):
        if 0 <= page_index < len(self.analysis_results_pages):
            self.current_plot_page_index = page_index
            self.update_pagination_ui()
            self.display_plot_page(page_index)
            page_data = self.analysis_results_pages[page_index]
            self.logArea.append(
                f"Switched to page {page_index + 1}: "
                f"{page_data.get('name', 'Analysis')}"
            )
        else:
            self.current_plot_page_index = -1
            self.update_pagination_ui()
            self.display_plot_page(-1)
            self._update_settings_ui(self.analysisTypeCombo.currentText())

    def on_mouse_move_on_plot(self, event):
        if event.inaxes and self.active_mpl_canvas_widget:
            x_coord = f"{event.xdata:.6f}" if event.xdata is not None else "---"
            y_coord = f"{event.ydata:.6f}" if event.ydata is not None else "---"
            self.cursor_coord_label.setText(f"(x, y) = ({x_coord}, {y_coord})")
            self.cursor_coord_label.adjustSize()
            self.cursor_coord_label.move(5, 5)
            self.cursor_coord_label.setVisible(True)
            self.cursor_coord_label.raise_()
        elif self.active_mpl_canvas_widget:
            self.cursor_coord_label.setVisible(False)

    def _cleanup_plot_area(self):
        """Disconnects events, removes old widgets, and clears the plot layout."""
        # Disconnect any previously connected event handlers
        if self.active_mpl_canvas_widget and hasattr(
            self.active_mpl_canvas_widget, "_event_cids"
        ):
            for cid in self.active_mpl_canvas_widget._event_cids:
                with contextlib.suppress(TypeError, RuntimeError):
                    self.active_mpl_canvas_widget.mpl_disconnect(cid)
            self.active_mpl_canvas_widget._event_cids = []

        # Clean up old UI widgets
        if self.active_mpl_toolbar_widget:
            self.mpl_toolbar_in_titlebar_layout.removeWidget(
                self.active_mpl_toolbar_widget
            )
            self.active_mpl_toolbar_widget.deleteLater()
            self.active_mpl_toolbar_widget = None

        self.mpl_toolbar_in_titlebar_container.setVisible(False)
        self.cursor_coord_label.setVisible(False)

        # Clear the main plot container layout
        plot_content_area_layout = self.plot_container_widget.layout()
        if plot_content_area_layout:
            self._clear_layout(plot_content_area_layout)
        else:
            plot_content_area_layout = QVBoxLayout(self.plot_container_widget)
            self.plot_container_widget.setLayout(plot_content_area_layout)

        self.active_mpl_canvas_widget = None
        return plot_content_area_layout

    def _populate_settings_from_page_data(self, page_data):
        """Updates the settings UI widgets with values from a saved analysis page."""
        page_args = {
            **page_data.get("constructor_args_used", {}),
            **page_data.get("view_args", {}),
        }
        for param_name, widget in self.current_settings_widgets.items():
            if param_name in page_args:
                self._set_widget_value(widget, page_args[param_name], param_name)

    # --- Load/Save Settings ---
    def _apply_loaded_settings_to_ui(self, loaded_settings):
        """Applies settings loaded from a file to the current UI widgets."""
        analysis_name = loaded_settings.get("analysis_name")
        if not analysis_name:
            raise ValueError("Settings file does not contain an 'analysis_name'.")

        self.analysisTypeCombo.setCurrentText(analysis_name)
        self.on_analysis_type_changed(
            analysis_name
        )  # Rebuilds the UI for this analysis

        all_args = {
            **loaded_settings.get("constructor_args", {}),
            **loaded_settings.get("view_args", {}),
        }

        for param_name, value in all_args.items():
            if param_name in self.current_settings_widgets:
                widget = self.current_settings_widgets[param_name]
                self._set_widget_value(widget, value)

    @Slot()
    def _load_analysis_settings_slot(self):
        """Loads and applies settings for an analysis from a JSON file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Analysis Settings", "", self.JSON_FILE_FILTER
        )
        if not filepath:
            return

        try:
            with open(filepath) as f:
                loaded_settings = json.load(f)
            self._apply_loaded_settings_to_ui(loaded_settings)
            self.logArea.append(
                f"Settings loaded from {filepath}. Click 'Apply' or 'Run' "
                "to see results."
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Load Error", f"Could not load or apply settings:\n{e}"
            )

    def _create_new_plot_canvas(self, page_data):
        """Creates a new FigureCanvas and connects mouse interaction events."""
        fig = Figure(figsize=page_data.get("figsize", (7, 5)), dpi=100)
        canvas = FigureCanvas(fig)
        canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus | Qt.FocusPolicy.StrongFocus)
        canvas.setFocus()

        # Connect events and store their IDs for later disconnection
        cids = [
            canvas.mpl_connect("scroll_event", self.on_scroll_zoom),
            canvas.mpl_connect("motion_notify_event", self.on_mouse_move_on_plot),
            canvas.mpl_connect("button_press_event", self.on_plot_double_click),
        ]
        canvas._event_cids = cids
        return canvas

    def _draw_plot_on_canvas(self, analysis_instance, canvas, view_args):
        """Invokes the analysis's view method to draw the plot on the canvas."""
        axs = analysis_instance.view(fig_to_plot_on=canvas.figure, **view_args)

        # Add summary text overlay if available
        if hasattr(analysis_instance, "get_summary_text"):
            summary_text = analysis_instance.get_summary_text()
            ax_to_use = None
            if isinstance(axs, np.ndarray):
                ax_to_use = axs.flatten()[-1]
            elif isinstance(axs, Axes):
                ax_to_use = axs

            if ax_to_use:
                props = dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.6)
                ax_to_use.text(
                    0.97,
                    0.03,
                    summary_text,
                    transform=ax_to_use.transAxes,
                    fontsize=7,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    bbox=props,
                    color="white",
                )

        canvas.figure.tight_layout(rect=[0, 0.05, 1, 1])

    def _setup_plot_toolbar(self, canvas):
        """Creates and attaches a new custom Matplotlib toolbar."""
        self.active_mpl_toolbar_widget = CustomMatplotlibToolbar(
            canvas, self.mpl_toolbar_in_titlebar_container
        )
        self.mpl_toolbar_in_titlebar_layout.addWidget(self.active_mpl_toolbar_widget)
        self.mpl_toolbar_in_titlebar_container.setVisible(True)

    def _display_placeholder(self, layout):
        """Displays a placeholder message when no analysis is selected."""
        self.plotTitleLabel.setText("No Analysis Selected")
        self.dataInfoLabel.setText("Run an analysis to see results.")
        layout.addWidget(QLabel("Select or Run an Analysis"))
        self._update_settings_ui(self.analysisTypeCombo.currentText())

    def display_plot_page(self, page_index):
        """
        Displays a specific analysis result page, orchestrating
        UI cleanup and redrawing.
        """
        plot_layout = self._cleanup_plot_area()

        if not (0 <= page_index < len(self.analysis_results_pages)):
            self._display_placeholder(plot_layout)
            return

        page_data = self.analysis_results_pages[page_index]
        analysis_name = page_data.get("name", "Analysis")
        analysis_instance = page_data.get("analysis_instance")

        # Update UI text elements
        self.plotTitleLabel.setText(analysis_name)
        self.dataInfoLabel.setText(
            page_data.get("result_summary", f"Results for {analysis_name}")
        )

        # Update settings panel to reflect this analysis
        self._update_settings_ui(analysis_name)
        self._populate_settings_from_page_data(page_data)

        if page_data.get("plot_type") == "embedded_mpl" and analysis_instance:
            self.active_mpl_canvas_widget = self._create_new_plot_canvas(page_data)
            self._draw_plot_on_canvas(
                analysis_instance,
                self.active_mpl_canvas_widget,
                page_data.get("view_args", {}),
            )
            self._setup_plot_toolbar(self.active_mpl_canvas_widget)
            plot_layout.addWidget(self.active_mpl_canvas_widget)
            self._fade_in_canvas(self.active_mpl_canvas_widget)
        else:
            plot_layout.addWidget(QLabel(f"Cannot embed plot for {analysis_name}"))

    def _fade_in_canvas(self, canvas, duration_ms: int = 250) -> None:
        """Fade a newly-rendered canvas from transparent to fully opaque.

        Args:
            canvas: The :class:`FigureCanvas` widget to animate.
            duration_ms: Animation duration in milliseconds.
        """
        effect = QGraphicsOpacityEffect(canvas)
        canvas.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", canvas)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(duration_ms)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start(QPropertyAnimation.DeleteWhenStopped)

    def on_plot_double_click(self, event):
        """Handler for mouse events on the plot canvas."""
        if event.dblclick:
            print("Plot double-clicked, refreshing.")
            self._refresh_current_plot_page_slot()

    @Slot()
    def toggle_settings_panel_slot(self):
        is_visible = self.settings_area_widget.isVisible()
        self.settings_area_widget.setVisible(not is_visible)
        if not is_visible:
            self.display_plot_page(self.current_plot_page_index)

    def _parse_tuple_str(self, s, expected_type=float, expected_len=2):
        if not s or not isinstance(s, str):
            return None
        try:
            parts = tuple(map(expected_type, s.split(",")))
            return parts if len(parts) == expected_len else None
        except (ValueError, TypeError):
            return None

    def _get_value_from_spinbox(self, widget):
        """Extracts the value from a QSpinBox or QDoubleSpinBox."""
        return widget.value()

    def _get_value_from_checkbox(self, widget):
        """Extracts the value from a QCheckBox."""
        return widget.isChecked()

    def _get_value_from_combobox(self, widget, param_name):
        """Extracts the value from a QComboBox, handling special cases."""
        if param_name in ["fields", "wavelengths", "wavelength"]:
            value_str = widget.currentData()
            if not value_str:
                return None
            try:
                value = ast.literal_eval(value_str)
                if (
                    param_name == "wavelength"
                    and isinstance(value, list)
                    and len(value) == 1
                ):
                    return value[0]
                return value
            except (ValueError, SyntaxError):
                return value_str
        elif param_name == "axis":
            return 1 if "Y-Axis" in widget.currentText() else 0

        return widget.currentText()

    def _get_value_from_lineedit(self, widget, param_name):
        """Extracts and parses the value from a QLineEdit."""
        text = widget.text().strip()
        if not text:
            return None

        if param_name == "max_freq":
            try:
                return int(text)
            except (ValueError, TypeError):
                return text  # Return text on failure

        if param_name in ["field", "pupil"]:
            return self._parse_tuple_str(text, float, 2)
        if param_name in ["res", "px_size"]:
            return self._parse_tuple_str(text, int, 2)
        if param_name == "cross_section":
            return self._parse_cross_section(text)

        return text

    def _validate_all_inputs(self):
        """Validates all input widgets in the current settings panel.

        Returns:
            Tuple[bool, str]: (Is valid, Error message)
        """
        for param_name, widget in self.current_settings_widgets.items():
            if isinstance(widget, QLineEdit) and param_name in [
                "field",
                "pupil",
                "res",
                "px_size",
            ]:
                # Check tuple logic (field, pupil, res, etc.)
                val = self._get_value_from_lineedit(widget, param_name)
                if val is None:
                    return (
                        False,
                        f"Invalid '{param_name}'. Expected 'x, y' format.",
                    )

                # Range check for normalized coords
                if param_name in ["field", "pupil"]:
                    for v in val:
                        if not (-1.0001 <= v <= 1.0001):
                            return (
                                False,
                                f"'{param_name}' coord {v} outside [-1, 1].",
                            )
        return True, ""

    def _parse_cross_section(self, text):
        """Parses a cross-section string (e.g., 'cross-x, 128')."""
        parts = [p.strip() for p in text.split(",")]
        if len(parts) == 2 and parts[0].lower() in ["cross-x", "cross-y"]:
            try:
                return (parts[0].lower(), int(parts[1]))
            except ValueError:
                pass  # Fall through to return the original text
        return text

    def _validate_system_for_analysis(self, optic):
        """
        Checks if the optical system is valid for running an analysis.

        Args:
            optic: The Optic object to validate.

        Returns:
            True if the system is valid, False otherwise.
        """
        tm = getattr(self.connector, "toast_manager", None)
        if not optic or optic.surface_group.num_surfaces < 2:
            if tm:
                tm.notify(
                    "A minimal optical system (at least 2 surfaces) is required.",
                    "warning",
                )
            else:
                QMessageBox.warning(
                    self,
                    self.ANALYSIS_ERROR_TITLE,
                    "A minimal optical system (at least 2 surfaces) is required.",
                )
            return False
        if optic.wavelengths.num_wavelengths == 0:
            if tm:
                tm.notify("The optical system has no defined wavelengths.", "warning")
            else:
                QMessageBox.warning(
                    self,
                    self.ANALYSIS_ERROR_TITLE,
                    "The optical system has no defined wavelengths.",
                )
            return False
        return True

    def _run_and_package_analysis(
        self, analysis_class, analysis_name, constructor_args, view_args
    ):
        """
        Instantiates, runs, and packages the analysis results.

        Args:
            analysis_class: The class of the analysis to run.
            analysis_name (str): The display name of the analysis.
            constructor_args (dict): Arguments for the analysis class constructor.
            view_args (dict): Arguments for the analysis view method.

        Returns:
            A dictionary containing the packaged page data for display,
            or None on failure.
        """
        optic = self.connector.get_optic()
        final_args = {"optic": optic, **constructor_args}

        # Filter args to only those accepted by the constructor.
        # For factory dispatch classes (e.g. FFTPSF) whose __init__ is
        # *args/**kwargs, fall back to __new__ for the accepted-param list.
        init_sig = inspect.signature(analysis_class.__init__)
        init_params = init_sig.parameters
        _variadic = frozenset(
            {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        )
        if all(
            p.kind in _variadic for name, p in init_params.items() if name != "self"
        ) and hasattr(analysis_class, "__new__"):
            init_params = inspect.signature(analysis_class.__new__).parameters
        valid_init_params = init_params
        filtered_args = {k: v for k, v in final_args.items() if k in valid_init_params}

        # Inject defaults for required args that may be absent when the
        # settings panel has never been opened (e.g. field/wavelength for
        # wavefront and PSF analyses).
        _required_defaults = {"field": (0.0, 0.0), "wavelength": "primary"}
        for _key, _default in _required_defaults.items():
            if _key not in filtered_args and _key in valid_init_params:
                filtered_args[_key] = _default

        if (
            analysis_name in [self.GEOMETRIC_MTF, self.FFT_MTF]
            and "max_freq" in final_args
            and "max_freq" not in filtered_args
        ):
            filtered_args["max_freq"] = final_args["max_freq"]

        instance = analysis_class(**filtered_args)

        # Check if the analysis can be plotted directly on a Matplotlib figure
        can_embed = (
            hasattr(instance, "view")
            and "fig_to_plot_on" in inspect.signature(instance.view).parameters
        )
        if not can_embed:
            instance.view(**view_args)  # Open in a separate window

        page_data = {
            "name": analysis_name,
            "analysis_instance": instance,
            "plot_type": "embedded_mpl" if can_embed else "external_window",
            "view_args": view_args,
            "constructor_args_used": constructor_args,
        }

        # Special case for sizing the plot figure for certain analyses
        if analysis_name in ("Through-Focus Spot", "Through-Focus Spot Diagram"):
            num_f = optic.fields.num_fields
            num_s = final_args.get("num_steps", 5)
            page_data["figsize"] = (max(1, num_s) * 3, max(1, num_f) * 3)

        return page_data

    def _collect_current_settings(self):
        """
        Collects all analysis parameters from the current settings UI widgets.

        This method iterates through the UI widgets, extracts their values using
        type-specific helpers, and sorts them into constructor or view arguments.

        Returns:
            A tuple containing two dictionaries: (constructor_args, view_args).
        """
        constructor_args, view_args = {}, {}
        known_view_args = ["add_airy_disk", "cmap", "normalize", "cross_section"]

        for param_name, widget in self.current_settings_widgets.items():
            value = None
            if isinstance(widget, (QSpinBox | QDoubleSpinBox)):
                value = self._get_value_from_spinbox(widget)
            elif isinstance(widget, QCheckBox):
                value = self._get_value_from_checkbox(widget)
            elif isinstance(widget, QComboBox):
                value = self._get_value_from_combobox(widget, param_name)
            elif isinstance(widget, QLineEdit):
                value = self._get_value_from_lineedit(widget, param_name)

            if value is not None:
                if param_name in known_view_args:
                    view_args[param_name] = value
                else:
                    constructor_args[param_name] = value

        return constructor_args, view_args

    def _execute_analysis(
        self, analysis_class, analysis_name, constructor_args=None, view_args=None
    ):
        """
        Main entry point for executing an analysis.

        This method orchestrates the validation, settings collection, execution,
        and error handling for running a single analysis.

        Args:
            analysis_class: The analysis class to instantiate.
            analysis_name (str): The display name of the analysis.
            constructor_args (dict, optional): Pre-collected args, used for cloning.
            view_args (dict, optional): Pre-collected view args, used for cloning.

        Returns:
            A dictionary of page data, or None if the analysis fails.
        """
        optic = self.connector.get_optic()
        if not self._validate_system_for_analysis(optic):
            return None

        # Validate UI inputs (ranges, tuples, etc)
        valid, error_msg = self._validate_all_inputs()
        if not valid:
            tm = getattr(self.connector, "toast_manager", None)
            if tm:
                tm.notify(error_msg, "error")
            else:
                QMessageBox.warning(self, "Invalid Input", error_msg)
            return None

        try:
            # If no args are provided, get them from the UI (standard run)
            if constructor_args is None and view_args is None:
                constructor_args, view_args = self._collect_current_settings()

            return self._run_and_package_analysis(
                analysis_class, analysis_name, constructor_args, view_args
            )

        except Exception as e:
            msg = f"An error occurred during {analysis_name}:\n{e}"
            tm = getattr(self.connector, "toast_manager", None)
            if tm:
                tm.notify(msg, "error")
            else:
                QMessageBox.critical(
                    self,
                    self.ANALYSIS_ERROR_TITLE,
                    msg,
                )
            import traceback

            print(f"Analysis Panel Error: {e}\n{traceback.format_exc()}")
            return None

    @Slot()
    def _apply_settings_and_rerun_analysis_slot(self):
        if not (0 <= self.current_plot_page_index < len(self.analysis_results_pages)):
            return
        page_data = self.analysis_results_pages[self.current_plot_page_index]
        analysis_name = page_data.get("name")
        self.logArea.setText(f"Rerunning {analysis_name} with new settings...")
        new_page_data = self._execute_analysis(
            self._analysis_class_map.get(analysis_name), analysis_name
        )
        if new_page_data:
            self.analysis_results_pages[self.current_plot_page_index] = new_page_data
            self.display_plot_page(self.current_plot_page_index)
            self.logArea.append(f"{analysis_name} reran successfully.")

    @Slot()
    def _refresh_current_plot_page_slot(self):
        """Refreshes the currently displayed analysis plot."""
        if not (0 <= self.current_plot_page_index < len(self.analysis_results_pages)):
            self.logArea.append("No analysis page selected to refresh.")
            return

        self.logArea.setText("Refreshing current analysis...")
        self._apply_settings_and_rerun_analysis_slot()

    @Slot()
    def run_analysis_slot(self):
        analysis_name = self.analysisTypeCombo.currentText()
        analysis_class = self._analysis_class_map.get(analysis_name)
        if not analysis_class:
            return
        self.logArea.setText(f"Running {analysis_name}...")
        page_data = self._execute_analysis(analysis_class, analysis_name)
        if page_data:
            self.analysis_results_pages.append(page_data)
            self.switch_plot_page(len(self.analysis_results_pages) - 1)
            self.logArea.append(f"{analysis_name} run complete.")

    @Slot()
    def run_all_analysis_slot(self):
        self.logArea.append("Run All: Not yet implemented.")

    @Slot()
    def stop_analysis_slot(self):
        self.logArea.append("Stop: Not yet implemented.")

    @Slot()
    def _save_analysis_settings_slot(self):
        """Saves the current settings for the active analysis to a JSON file."""
        current_analysis_name = self.analysisTypeCombo.currentText()
        if not current_analysis_name:
            return

        constructor_args, view_args = self._collect_current_settings()
        settings_to_save = {
            "analysis_name": current_analysis_name,
            "constructor_args": constructor_args,
            "view_args": view_args,
        }

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {current_analysis_name} Settings",
            f"{current_analysis_name}_settings.json",
            "JSON Files (*.json);;All Files (*)",
        )

        if filepath:
            try:
                with open(filepath, "w") as f:
                    json.dump(settings_to_save, f, indent=4)
                self.logArea.append(
                    f"Settings for {current_analysis_name} saved to {filepath}"
                )
            except Exception as e:
                msg = f"Could not save settings:\n{e}"
                tm = getattr(self.connector, "toast_manager", None)
                if tm:
                    tm.notify(msg, "error")
                else:
                    QMessageBox.critical(self, "Save Error", msg)

    def on_scroll_zoom(self, event):
        gui_plot_utils.handle_matplotlib_scroll_zoom(event)

    @Slot()
    def _load_analysis_settings_slot(self):
        """Loads and applies settings for an analysis from a JSON file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Analysis Settings", "", "JSON Files (*.json);;All Files (*)"
        )

        if filepath:
            try:
                with open(filepath) as f:
                    loaded_settings = json.load(f)

                analysis_name = loaded_settings.get("analysis_name")
                self.analysisTypeCombo.setCurrentText(analysis_name)

                # Apply the loaded settings to the UI widgets
                self.on_analysis_type_changed(analysis_name)

                all_args = {
                    **loaded_settings.get("constructor_args", {}),
                    **loaded_settings.get("view_args", {}),
                }
                for param_name, value in all_args.items():
                    if param_name in self.current_settings_widgets:
                        widget = self.current_settings_widgets[param_name]
                        if isinstance(widget, QComboBox):
                            index = widget.findData(str(value))
                            if index != -1:
                                widget.setCurrentIndex(index)
                        elif isinstance(widget, QSpinBox | QDoubleSpinBox):
                            widget.setValue(value)
                        elif isinstance(widget, QCheckBox):
                            widget.setChecked(value)
                        elif isinstance(widget, QLineEdit):
                            widget.setText(str(value))

                self.logArea.append(
                    f"Settings loaded from {filepath}. "
                    "Click 'Apply' or 'Run' to see results."
                )

            except Exception as e:
                msg = f"Could not load or apply settings:\n{e}"
                tm = getattr(self.connector, "toast_manager", None)
                if tm:
                    tm.notify(msg, "error")
                else:
                    QMessageBox.critical(self, "Load Error", msg)
