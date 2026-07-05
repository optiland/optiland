"""
Provides the main viewing panel for optical systems.

This module defines the `ViewerPanel`, which contains tabs for 2D and 3D
visualizations of the optical system. It includes `MatplotlibViewer` for 2D
plots and `VTKViewer` for 3D rendering.

@author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

from typing import TYPE_CHECKING

from optiland.visualization.analysis.surface_sag import SurfaceSagViewer
from optiland.visualization.system.rays import Rays2D, Rays3D
from optiland.visualization.system.system import (
    OpticalSystem as OptilandOpticalSystemPlotter,
)

from . import gui_plot_utils
from .analysis_panel import CustomMatplotlibToolbar

if TYPE_CHECKING:
    from .optiland_connector import OptilandConnector


class SagViewer(QWidget):
    """A widget for displaying a 2D sag plot of a selected optical surface."""

    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.current_theme = "dark"

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(25)

        # Main Plotting Area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(2)
        main_layout.addWidget(plot_widget, 1)

        # --- Toolbar and Title ---
        toolbar_container = QWidget()
        toolbar_container.setObjectName("ViewerToolbarContainer")
        toolbar_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        toolbar_container.setMaximumHeight(60)
        toolbar_layout = QHBoxLayout(toolbar_container)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(toolbar_container)

        # --- Matplotlib Canvas ---
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas, 1)

        # --- Add Toolbar to container ---
        self.toolbar = CustomMatplotlibToolbar(self.canvas, toolbar_container)
        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addStretch()

        # Add settings toggle button to toolbar
        self.settings_toggle_btn = QToolButton()
        self.settings_toggle_btn.setToolTip("Toggle Sag Viewer Settings")
        self.settings_toggle_btn.setCheckable(True)
        self.settings_toggle_btn.setChecked(True)
        self.settings_toggle_btn.toggled.connect(self._toggle_settings)
        self.toolbar.addWidget(self.settings_toggle_btn)

        # Re-route the toolbar's home button to our full plot refresh
        for action in self.toolbar.actions():
            if action.toolTip() == "Reset original view":
                action.triggered.disconnect()
                action.triggered.connect(self.plot_sag)
                break

        # --- Cursor Coordinate Label ---
        self.cursor_coord_label = QLabel("", self.canvas)
        self.cursor_coord_label.setObjectName("CursorCoordLabel")
        self.cursor_coord_label.setStyleSheet(
            "background-color:rgba(0,0,0,0.65);color:white;padding:2px 4px;"
            "border-radius:3px;"
        )
        self.cursor_coord_label.setVisible(False)
        self.cursor_coord_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )

        # Connect mouse move event
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move_on_plot)

        # Settings Area
        self.settings_area = QWidget()
        self.settings_area.setFixedWidth(220)
        settings_layout = QVBoxLayout(self.settings_area)
        settings_layout.addWidget(QLabel("Sag Viewer Settings"))

        settings_form = QFormLayout()
        self.surface_selector = QSpinBox()
        self.surface_selector.setRange(0, 100)
        settings_form.addRow("Surface Index:", self.surface_selector)

        self.x_cross_section = QDoubleSpinBox()
        self.x_cross_section.setRange(-1000, 1000)
        self.x_cross_section.setValue(0.0)
        settings_form.addRow("X Cross-section (for Y-plot):", self.x_cross_section)

        self.y_cross_section = QDoubleSpinBox()
        self.y_cross_section.setRange(-1000, 1000)
        self.y_cross_section.setValue(0.0)
        settings_form.addRow("Y Cross-section (for X-plot):", self.y_cross_section)

        settings_layout.addLayout(settings_form)
        settings_layout.addStretch()

        self.maxExtentSpinBox = QDoubleSpinBox()
        self.maxExtentSpinBox.setRange(0.01, 1000.0)
        self.maxExtentSpinBox.setValue(20.0)  # Default value
        self.maxExtentSpinBox.setSuffix(" mm")
        self.maxExtentSpinBox.setToolTip("Set the viewing area extent (±mm)")
        self.maxExtentSpinBox.valueChanged.connect(self.plot_sag)

        settings_form.addRow("View Extent:", self.maxExtentSpinBox)

        apply_button = QPushButton("Plot Sag")
        apply_button.clicked.connect(self.plot_sag)
        settings_layout.addWidget(apply_button)
        main_layout.addWidget(self.settings_area)

        # Initial setup
        self.connector.opticChanged.connect(self.update_surface_range)
        self.update_surface_range()
        self.plot_sag()
        self.update_theme()

    def _toggle_settings(self, checked):
        """Toggle the visibility of the settings panel."""
        self.settings_area.setVisible(checked)

    def on_mouse_move_on_plot(self, event):
        """Displays the cursor's coordinates on the plot."""
        if event.inaxes:
            # Determine which axis the cursor is over for a more informative label
            axis_label = "Pos"
            if event.inaxes.get_xlabel() == "X-coordinate":
                axis_label = "(X, Sag)"
            elif event.inaxes.get_ylabel() == "Y-coordinate (mm)":
                axis_label = "(X, Y)"
            elif event.inaxes.get_xlabel() == "Sag (z)":
                axis_label = "(Sag, Y)"

            x_coord = f"{event.xdata:.3f}" if event.xdata is not None else "---"
            y_coord = f"{event.ydata:.3f}" if event.ydata is not None else "---"
            self.cursor_coord_label.setText(f"{axis_label} = ({x_coord}, {y_coord})")
            self.cursor_coord_label.adjustSize()
            # Position at the bottom-left of the canvas
            self.cursor_coord_label.move(
                5, self.canvas.height() - self.cursor_coord_label.height() - 5
            )
            self.cursor_coord_label.setVisible(True)
            self.cursor_coord_label.raise_()
        else:
            self.cursor_coord_label.setVisible(False)

    def update_surface_range(self):
        """Updates the range of the surface selector spinbox."""
        count = self.connector.get_surface_count()
        self.surface_selector.setRange(0, max(0, count - 1))

    def update_theme(self, theme="dark"):
        self.current_theme = theme
        self.settings_toggle_btn.setIcon(QIcon(f":/icons/{theme}/settings.svg"))
        self.plot_sag()

    @Slot()
    def plot_sag(self):
        gui_plot_utils.apply_gui_matplotlib_styles(theme=self.current_theme)
        optic = self.connector.get_optic()
        surface_index = self.surface_selector.value()
        self.figure.clear()

        if not optic or not (0 <= surface_index < optic.surface_group.num_surfaces):
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Invalid Surface Index: {surface_index}",
                ha="center",
                va="center",
            )
            self.canvas.draw()
            return

        # Use the existing backend SurfaceSagViewer class
        viewer = SurfaceSagViewer(optic)

        # Call its view method, passing our figure to be plotted on
        viewer.view(
            surface_index=surface_index,
            y_cross_section=self.y_cross_section.value(),
            x_cross_section=self.x_cross_section.value(),
            max_extent=self.maxExtentSpinBox.value(),
            fig_to_plot_on=self.figure,
        )

        # Redraw our canvas
        self.canvas.draw()


class ViewerPanel(QWidget):
    """
    A widget that contains multiple viewers for the optical system.

    This panel uses a QTabWidget to host different types of viewers, such as
    a 2D plot and a 3D rendering of the system.

    Attributes:
        connector (OptilandConnector): The connector to the main application logic.
        tabWidget (QTabWidget): The widget hosting the different viewer tabs.
        viewer2D (MatplotlibViewer): The 2D viewer widget.
        viewer3D (VTKViewer or QLabel): The 3D viewer widget, or a label if VTK
                                        is unavailable.
    """

    def __init__(self, connector: OptilandConnector, parent=None):
        """
        Initializes the ViewerPanel.

        Args:
            connector (OptilandConnector): The connector to the main application logic.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.connector = connector

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.tabWidget = QTabWidget()

        # Create 2D Viewer Tab
        self.viewer2D = MatplotlibViewer(self.connector)
        viewer2d_container = self._create_2d_viewer_tab()
        self.tabWidget.addTab(viewer2d_container, "2D Layout")

        # Create 3D Viewer Tab
        self.viewer3D = None
        if VTK_AVAILABLE:
            self.viewer3D = VTKViewer(self.connector)
            self.tabWidget.addTab(self.viewer3D, "3D Layout")

        # Create Sag Viewer Tab
        self.sagViewer = SagViewer(self.connector, self)
        self.tabWidget.addTab(self.sagViewer, "Sag")

        main_layout.addWidget(self.tabWidget)

        self.connector.opticLoaded.connect(self.update_viewers)
        self.connector.opticChanged.connect(self.update_viewers)

    def _create_2d_viewer_tab(self):
        """Creates the container widget for the 2D viewer, including its toolbar."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)

        # Add toolbar with 'Preserve Zoom' checkbox
        toolbar_layout = QHBoxLayout()
        self.preserve_zoom_checkbox = QCheckBox("Preserve Zoom")
        self.preserve_zoom_checkbox.setToolTip(
            "Lock the current zoom and pan level when the system updates."
        )
        toolbar_layout.addWidget(self.preserve_zoom_checkbox)
        toolbar_layout.addStretch()

        layout.addLayout(toolbar_layout)
        layout.addWidget(self.viewer2D)
        return container

    @Slot()
    def update_viewers(self):
        """Updates all active viewers with the current optic data."""
        if self.viewer2D:
            preserve = self.preserve_zoom_checkbox.isChecked()
            self.viewer2D.plot_optic(preserve_zoom=preserve)
        if self.viewer3D:
            self.viewer3D.render_optic()

    def update_theme(self, theme_name: str):
        """Updates the theme for all viewers in this panel."""
        if self.viewer2D:
            self.viewer2D.update_theme(theme_name)
        if self.viewer3D:
            self.viewer3D.update_theme(theme_name)
        if self.sagViewer:
            self.sagViewer.update_theme(theme_name)


class MatplotlibViewer(QWidget):
    """
    A widget for displaying a 2D plot of the optical system using Matplotlib.

    This viewer includes a Matplotlib canvas, a custom toolbar, and a settings
    panel for controlling the plot, such as the number of rays to trace.

    Attributes:
        figure (Figure): The Matplotlib figure object.
        canvas (FigureCanvas): The canvas widget that displays the figure.
        ax (Axes): The Matplotlib axes object for plotting.
        toolbar (CustomMatplotlibToolbar): The toolbar for plot navigation.
        settings_area (QWidget): The panel for viewer-specific settings.
    """

    def __init__(self, connector: OptilandConnector, parent=None):
        """
        Initializes the MatplotlibViewer.

        Args:
            connector (OptilandConnector): The connector to the main application logic.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.connector = connector
        self.current_theme = "dark"

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        viewer_widget = QWidget()
        self.layout = QVBoxLayout(viewer_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        main_layout.addWidget(viewer_widget, 1)

        self.toolbar_container = QWidget()
        self.toolbar_container.setObjectName("ViewerToolbarContainer")
        toolbar_layout = QHBoxLayout(self.toolbar_container)
        toolbar_layout.setContentsMargins(5, 0, 5, 0)
        self.layout.addWidget(self.toolbar_container)

        plot_container = QWidget()
        plot_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(plot_container, 1)

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)

        self._is_plotting = False
        self._user_initiated_view_change = False

        self.toolbar = CustomMatplotlibToolbar(self.canvas, self.toolbar_container)
        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addStretch()

        for action in self.toolbar.actions():
            if action.toolTip() == "Reset original view":
                # Disconnect the default trigger
                action.triggered.disconnect()
                # Connect our full plot refresh method
                action.triggered.connect(self.reset_view)
                break

        self.settings_area = QWidget()
        self.settings_area.setObjectName("ViewerSettingsArea")
        self.settings_area.setFixedWidth(200)
        self.settings_area.setVisible(False)
        settings_layout = QVBoxLayout(self.settings_area)
        self.settings_form_layout = QFormLayout()

        self.num_rays_spinbox = QSpinBox()
        self.num_rays_spinbox.setRange(1, 100)
        self.num_rays_spinbox.setValue(3)
        self.settings_form_layout.addRow("Num Rays:", self.num_rays_spinbox)

        self.dist_combo = QComboBox()
        self.dist_combo.addItems(["line_y", "line_x", "hexapolar", "random"])
        self.settings_form_layout.addRow("Distribution:", self.dist_combo)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.plot_optic)

        settings_layout.addLayout(self.settings_form_layout)
        settings_layout.addStretch()
        settings_layout.addWidget(apply_button)
        main_layout.addWidget(self.settings_area)

        self.settings_toggle_btn = QToolButton()
        self.settings_toggle_btn.setToolTip("Toggle Viewer Settings")
        self.settings_toggle_btn.setCheckable(True)
        self.settings_toggle_btn.toggled.connect(self.settings_area.setVisible)
        self.toolbar.addWidget(self.settings_toggle_btn)

        self.cursor_coord_label = QLabel("", self.canvas)
        self.cursor_coord_label.setObjectName("CursorCoordLabel")
        self.cursor_coord_label.setStyleSheet(
            "background-color:rgba(0,0,0,0.65);color:white;padding:2px 4px;"
            "border-radius:3px;"
        )
        self.cursor_coord_label.setVisible(False)
        self.cursor_coord_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move_on_plot)
        self.canvas.mpl_connect("scroll_event", self.on_scroll_zoom)

        # Add new event connections for panning
        self.canvas.mpl_connect("button_press_event", self.on_mouse_button_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_button_release)
        self.ax.callbacks.connect("xlim_changed", self.on_ax_limit_changed)
        self.ax.callbacks.connect("ylim_changed", self.on_ax_limit_changed)

        # Initialize panning state variables
        self._pan_start_x = None
        self._pan_start_y = None
        self._is_panning = False

        self.plot_optic()
        self.update_theme()

    def on_ax_limit_changed(self, ax):
        """Callback for when axis limits change, to detect user interaction."""
        if not self._is_plotting:
            self._user_initiated_view_change = True

    def reset_view(self):
        """Resets the view to the default 1:1 aspect ratio and zoom."""
        self._user_initiated_view_change = False
        self.plot_optic(preserve_zoom=False)

    def on_mouse_button_press(self, event):
        """
        Handles mouse button press events to initiate panning.

        Args:
            event: The Matplotlib mouse button press event.
        """
        if event.button == 1 and event.inaxes:  # Left mouse button
            self._pan_start_x = event.xdata
            self._pan_start_y = event.ydata
            self._is_panning = True
            self.canvas.setCursor(
                Qt.ClosedHandCursor
            )  # Change cursor to indicate panning

    def on_mouse_button_release(self, event):
        """
        Handles mouse button release events to stop panning.

        Args:
            event: The Matplotlib mouse button release event.
        """
        if event.button == 1:  # Left mouse button
            self._is_panning = False
            self._pan_start_x = None
            self._pan_start_y = None
            self.canvas.setCursor(Qt.ArrowCursor)  # Reset cursor

    def on_mouse_move_on_plot(self, event):
        """
        Displays the cursor's coordinates on the plot and handles panning.

        Args:
            event: The Matplotlib motion notify event.
        """
        if self._is_panning and event.inaxes and self._pan_start_x is not None:
            # Calculate the distance moved
            dx = self._pan_start_x - event.xdata
            dy = self._pan_start_y - event.ydata

            # Get current axis limits
            ax = event.inaxes
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Update the limits by the distance moved
            ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            ax.set_ylim(ylim[0] + dy, ylim[1] + dy)

            # Redraw the canvas
            self.canvas.draw_idle()
            return  # Skip the coordinate display when panning

        # Original coordinate display code
        if event.inaxes:
            x_coord = f"{event.xdata:.3f}"
            y_coord = f"{event.ydata:.3f}"
            self.cursor_coord_label.setText(f"(Z, Y) = ({x_coord}, {y_coord})")
            self.cursor_coord_label.adjustSize()
            self.cursor_coord_label.move(5, 5)
            self.cursor_coord_label.setVisible(True)
            self.cursor_coord_label.raise_()
        else:
            self.cursor_coord_label.setVisible(False)

    def on_scroll_zoom(self, event):
        """
        Implements zoom functionality using the mouse scroll wheel.

        Args:
            event: The Matplotlib scroll event.
        """
        if not event.inaxes:
            return

        ax = event.inaxes
        scale_factor = 1.1 if event.step < 0 else 1 / 1.1

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * rel_x])
        ax.set_ylim([ydata - new_height * (1 - rel_y), ydata + new_height * rel_y])
        ax.figure.canvas.draw_idle()

    def update_theme(self, theme="dark"):
        """
        Updates the theme of the Matplotlib plot.

        Args:
            theme (str, optional): The theme name ('dark' or 'light').
                                   Defaults to "dark".
        """
        if self.current_theme != theme:
            self.current_theme = theme
            gui_plot_utils.apply_gui_matplotlib_styles(theme=self.current_theme)
            self.plot_optic()
        self.settings_toggle_btn.setIcon(QIcon(f":/icons/{theme}/settings.svg"))

    def plot_optic(self, preserve_zoom=False):
        """
        Clears the current plot and redraws the optical system.

        This method retrieves the current optical system from the connector and
        uses Optiland's plotting utilities to generate a 2D layout.

        Args:
            preserve_zoom (bool): If True, maintains the current view
            limits after redrawing.
        """
        self._is_plotting = True
        try:
            gui_plot_utils.apply_gui_matplotlib_styles(theme=self.current_theme)

            should_preserve_limits = preserve_zoom or self._user_initiated_view_change
            xlim = self.ax.get_xlim() if should_preserve_limits else None
            ylim = self.ax.get_ylim() if should_preserve_limits else None

            self.ax.clear()
            face_color = matplotlib.rcParams["figure.facecolor"]
            self.figure.set_facecolor(face_color)
            self.ax.set_facecolor(face_color)

            optic = self.connector.get_optic()
            num_rays = self.num_rays_spinbox.value()
            distribution = self.dist_combo.currentText()
            if optic and optic.surface_group.num_surfaces > 0:
                try:
                    rays2d_plotter = Rays2D(optic)
                    system_plotter = OptilandOpticalSystemPlotter(
                        optic, rays2d_plotter, projection="2d"
                    )
                    from optiland.visualization.themes import get_active_theme

                    theme = get_active_theme()
                    rays2d_plotter.plot(
                        self.ax,
                        fields="all",
                        wavelengths="primary",
                        num_rays=num_rays,
                        distribution=distribution,
                        theme=theme,
                    )
                    system_plotter.plot(self.ax, theme=theme)
                    self.ax.set_title(
                        f"System: {optic.name} (2D)",
                        color=matplotlib.rcParams["text.color"],
                    )
                    self.ax.set_xlabel("Z-axis (mm)")
                    self.ax.set_ylabel("Y-axis (mm)")
                    self.ax.grid(True, linestyle="--", alpha=0.7)

                    if should_preserve_limits and xlim is not None and ylim is not None:
                        self.ax.set_xlim(xlim)
                        self.ax.set_ylim(ylim)
                        self.ax.set_aspect("auto")
                    else:
                        fig_width, fig_height = self.figure.get_size_inches()
                        widget_aspect = fig_height / fig_width

                        xlim_data = self.ax.get_xlim()
                        ylim_data = self.ax.get_ylim()
                        x_range = xlim_data[1] - xlim_data[0]
                        y_range = ylim_data[1] - ylim_data[0]

                        if x_range == 0:
                            x_range = 1e-6
                        if y_range == 0:
                            y_range = 1e-6

                        data_aspect = y_range / x_range

                        # determine which axis to expand to achieve equal aspect ratio
                        if data_aspect < widget_aspect:
                            # expand Y
                            y_center = (ylim_data[0] + ylim_data[1]) / 2
                            new_y_range = x_range * widget_aspect
                            self.ax.set_ylim(
                                y_center - new_y_range / 2, y_center + new_y_range / 2
                            )
                        else:
                            # expand X
                            x_center = (xlim_data[0] + xlim_data[1]) / 2
                            new_x_range = y_range / widget_aspect
                            self.ax.set_xlim(
                                x_center - new_x_range / 2, x_center + new_x_range / 2
                            )

                        self.ax.set_aspect("equal")

                except Exception:
                    self.ax.text(
                        0.5, 0.5, "Error plotting system", ha="center", va="center"
                    )
            else:
                self.ax.text(0.5, 0.5, "No system loaded", ha="center", va="center")

            self.canvas.draw()
        finally:
            self._is_plotting = False


class VTKViewer(QWidget):
    """
    A widget for displaying a 3D rendering of the optical system using VTK.

    This viewer embeds a QVTKRenderWindowInteractor to provide an interactive
    3D view of the optical system and traced rays.

    Attributes:
        vtkWidget (QVTKRenderWindowInteractor): The VTK render window interactor widget.
        renderer (vtkRenderer): The VTK renderer for the scene.
        iren (vtkRenderWindowInteractor): The interactor for camera manipulation.
    """

    def __init__(self, connector: OptilandConnector, parent=None):
        """
        Initializes the VTKViewer.

        Args:
            connector (OptilandConnector): The connector to the main application logic.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)

        self.connector = connector
        if not VTK_AVAILABLE:
            self.layout = QVBoxLayout(self)
            self.layout.addWidget(QLabel("VTK is not available."))
            return

        self.layout = QVBoxLayout(self)
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtkWidget)

        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.setup_default_camera()
        self.iren.Initialize()

    def setup_default_camera(self):
        """Sets up the default camera position and orientation for the 3D view."""
        self.renderer.SetBackground(0.1, 0.2, 0.4)
        camera = self.renderer.GetActiveCamera()
        if camera:
            camera.SetPosition(0.2, 0, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 1, 0)
            self.renderer.ResetCamera()
            camera.Elevation(0)
            camera.Azimuth(150)

    def update_theme(self, theme="dark"):
        """
        Updates the background color of the VTK renderer based on the theme.

        Args:
            theme (str, optional): The theme name ('dark' or 'light').
                                   Defaults to "dark".
        """
        from matplotlib.colors import to_rgb

        from optiland.visualization.themes import get_active_theme, set_theme

        set_theme(theme)
        theme = get_active_theme()
        background = to_rgb(theme.parameters["axes.facecolor"])
        self.renderer.SetBackground(*background)
        self.vtkWidget.GetRenderWindow().Render()

    def render_optic(self):
        """
        Clears the current scene and re-renders the optical system in 3D.

        This method retrieves the current optical system and uses Optiland's
        VTK plotting utilities to generate the 3D visualization.
        """
        if not VTK_AVAILABLE:
            return

        self.renderer.RemoveAllViewProps()
        optic = self.connector.get_optic()

        # Check if optic has surfaces and a valid aperture
        if (
            optic
            and optic.surface_group.num_surfaces > 0
            and hasattr(optic, "aperture")
            and optic.aperture is not None
        ):
            try:
                rays3d_plotter = Rays3D(optic)
                system_plotter = OptilandOpticalSystemPlotter(
                    optic, rays3d_plotter, projection="3d"
                )

                from optiland.visualization.themes import get_active_theme

                theme = get_active_theme()
                rays3d_plotter.plot(
                    self.renderer,
                    fields="all",
                    wavelengths="primary",
                    num_rays=24,
                    distribution="ring",
                    theme=theme,
                )

                system_plotter.plot(self.renderer, theme=theme)

                if not self.renderer.GetActiveCamera():
                    self.setup_default_camera()
                else:
                    self.renderer.ResetCameraClippingRange()
                    self.renderer.ResetCamera()

            except Exception as e:
                print(f"VTKViewer Error: {e}")
                textActor = vtk.vtkTextActor()
                textActor.SetInput(f"Error rendering 3D view:\n{e}")
                textActor.GetTextProperty().SetColor(1, 0, 0)
                self.renderer.AddActor2D(textActor)
        else:
            # Display a message if the optic doesn't have a valid aperture
            if (
                optic
                and optic.surface_group.num_surfaces > 0
                and (not hasattr(optic, "aperture") or optic.aperture is None)
            ):
                textActor = vtk.vtkTextActor()
                textActor.SetInput("Please set an aperture in System Properties.")
                textActor.GetTextProperty().SetColor(1, 0, 0)
                self.renderer.AddActor2D(textActor)

            # Add a default sphere for empty systems
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetRadius(0.1)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphereSource.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self.renderer.AddActor(actor)
            if not self.renderer.GetActiveCamera():
                self.setup_default_camera()
            else:
                self.renderer.ResetCamera()

        self.vtkWidget.GetRenderWindow().Render()
