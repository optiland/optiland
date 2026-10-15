"""Manages the creation, layout, and lifecycle of dockable panels.

This module provides :class:`PanelManager`, which is responsible for
instantiating, configuring, and arranging all the
:class:`~PySide6.QtWidgets.QDockWidget` panels used in the Optiland GUI.
Decoupling panel management from :class:`~optiland_gui.main_window.MainWindow`
keeps each class focused on a single responsibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

from .analysis_panel import AnalysisPanel
from .lens_editor import LensEditor
from .optimization_panel import OptimizationPanel
from .surface_interaction import SurfaceInteractionState
from .system_properties_panel import SystemPropertiesPanel
from .viewer_panel import ViewerPanel
from .widgets.custom_dock_widget import CustomDockWidget
from .widgets.python_terminal import PythonTerminalWidget
from .widgets.sidebar import SidebarWidget

if TYPE_CHECKING:
    from .optiland_connector import OptilandConnector

# Default theme applied before the user's preference is loaded.
_DEFAULT_THEME = "dark"


class PanelManager:
    """Manages the instantiation and layout of all GUI panels.

    Args:
        main_window: The application's main window.
        connector: The central
            :class:`~optiland_gui.optiland_connector.OptilandConnector` instance.
    """

    def __init__(self, main_window: QMainWindow, connector: OptilandConnector) -> None:
        self.main_window = main_window
        self.connector = connector
        self.all_docks: list[QDockWidget] = []
        self.sidebar: QDockWidget | None = None

    def create_all_panels(self, parent_for_iface: QWidget) -> None:
        """Create instances of all panels and their containing dock widgets.

        Args:
            parent_for_iface: The main window widget, whose ``iface`` attribute
                is injected into the scripting terminal's kernel namespace.
        """
        # Sidebar
        self.sidebar_content_widget = SidebarWidget(self.main_window)
        self.sidebar = self._create_dock(
            widget=self.sidebar_content_widget,
            name="SidebarDockWidget",
            title="NavigationSidebar",
            features=(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea,
            use_custom_title_bar=False,
        )
        self.sidebar.setTitleBarWidget(QWidget())

        # Main panels
        self.surface_interaction = SurfaceInteractionState(self.main_window)
        self.surface_interaction.sync_document(self.connector.get_optic())
        self.viewer_panel = ViewerPanel(self.connector)
        self.viewer_panel.viewer2D.set_interaction_state(self.surface_interaction)
        if self.viewer_panel.viewer3D is not None:
            self.viewer_panel.viewer3D.set_interaction_state(self.surface_interaction)
        self.viewer_dock = self._create_dock(
            self.viewer_panel, "ViewerDock", "System Viewer"
        )

        self.lens_editor = LensEditor(
            self.connector, interaction_state=self.surface_interaction
        )
        self.lens_editor_dock = self._create_dock(
            self.lens_editor, "LensEditorDock", "Lens Data Editor"
        )

        self.system_properties = SystemPropertiesPanel(self.connector)
        self.system_properties_dock = self._create_dock(
            self.system_properties, "SystemPropertiesDock", "System Properties"
        )

        self.analysis_panel = AnalysisPanel(self.connector)
        self.analysis_dock = self._create_dock(
            self.analysis_panel, "AnalysisPanelDock", "Analysis"
        )

        self.optimization_panel = OptimizationPanel(self.connector)
        self.optimization_dock = self._create_dock(
            self.optimization_panel, "OptimizationDock", "Optimization"
        )

        # Scripting terminal — inject connector and iface into the kernel
        self.python_terminal = PythonTerminalWidget(
            parent_for_iface,
            custom_variables={
                "connector": self.connector,
                "iface": parent_for_iface.iface,
            },
            theme=_DEFAULT_THEME,
        )
        self.terminal_dock = self._create_dock(
            self.python_terminal, "TerminalDock", "Scripts Terminal"
        )

        self.all_docks = [
            self.sidebar,
            self.viewer_dock,
            self.lens_editor_dock,
            self.system_properties_dock,
            self.analysis_dock,
            self.optimization_dock,
            self.terminal_dock,
        ]

    def _create_dock(
        self,
        widget: QWidget,
        name: str,
        title: str,
        features: QDockWidget.DockWidgetFeature | None = None,
        allowed_areas: Qt.DockWidgetArea = Qt.AllDockWidgetAreas,
        use_custom_title_bar: bool = True,
    ) -> QDockWidget:
        """Factory method for creating and configuring a :class:`QDockWidget`.

        Args:
            widget: The content widget to embed in the dock.
            name: Unique object name for state persistence.
            title: Window title shown in the title bar.
            features: Dock feature flags; defaults to Closable | Movable | Floatable.
            allowed_areas: Dock areas the widget may be placed in.
            use_custom_title_bar: If ``True``, use :class:`CustomDockWidget`;
                otherwise use a plain :class:`QDockWidget`.

        Returns:
            The configured dock widget.
        """
        if use_custom_title_bar:
            dock = CustomDockWidget(title, self.main_window)
            dock.setWidget(widget)
        else:
            dock = QDockWidget(title, self.main_window)
            dock.setWidget(widget)

        dock.setObjectName(name)
        if features is not None:
            dock.setFeatures(features)
        else:
            dock.setFeatures(
                QDockWidget.DockWidgetClosable
                | QDockWidget.DockWidgetMovable
                | QDockWidget.DockWidgetFloatable
            )
        dock.setAllowedAreas(allowed_areas)
        return dock

    def setup_default_layout(self) -> None:
        """Arrange the dock widgets into the application's default layout."""
        self.arrange_workspace("four_panes")

    def arrange_workspace(self, arrangement: str) -> None:
        """Arrange existing panels into two columns or four split panes.

        Alternative panels remain accessible as tabs. This changes only Qt's
        layout; panel contents, scientific results and size constraints remain
        owned by their existing widgets.
        """
        if arrangement not in {"two_panes", "four_panes"}:
            raise ValueError(f"Unknown workspace arrangement: {arrangement}")
        window = self.main_window
        updates_enabled = window.updatesEnabled()
        window.setUpdatesEnabled(False)
        try:
            for dock in self.all_docks:
                dock.setFloating(False)
                window.removeDockWidget(dock)
            window.addDockWidget(Qt.LeftDockWidgetArea, self.sidebar)
            window.addDockWidget(Qt.RightDockWidgetArea, self.lens_editor_dock)
            window.splitDockWidget(
                self.lens_editor_dock, self.analysis_dock, Qt.Horizontal
            )
            if arrangement == "four_panes":
                window.splitDockWidget(
                    self.lens_editor_dock, self.viewer_dock, Qt.Vertical
                )
                window.splitDockWidget(
                    self.analysis_dock, self.terminal_dock, Qt.Vertical
                )
            else:
                window.tabifyDockWidget(self.lens_editor_dock, self.viewer_dock)
                window.tabifyDockWidget(self.analysis_dock, self.terminal_dock)
            window.tabifyDockWidget(self.analysis_dock, self.system_properties_dock)
            window.tabifyDockWidget(self.analysis_dock, self.optimization_dock)
            for dock in self.all_docks:
                dock.show()
            self.lens_editor_dock.raise_()
            self.viewer_dock.raise_()
            self.analysis_dock.raise_()
            if arrangement == "four_panes":
                self.terminal_dock.raise_()
            terminal = self.python_terminal
            for dock in (terminal.editor_dock, terminal.console_dock):
                dock.setFloating(False)
                terminal.dock_area.removeDockWidget(dock)
            terminal.dock_area.addDockWidget(
                Qt.LeftDockWidgetArea, terminal.editor_dock
            )
            terminal.dock_area.addDockWidget(
                Qt.RightDockWidgetArea, terminal.console_dock
            )
            terminal.editor_dock.show()
            terminal.console_dock.show()
            window.layout().activate()
            self.equalize_workspace()
        finally:
            window.setUpdatesEnabled(updates_enabled)

    @staticmethod
    def _equalize_dock_area(window: QMainWindow, docks: list[QDockWidget]) -> None:
        """Ask Qt to balance visible splits, using one member of each tab group."""
        window.layout().activate()
        representatives = []
        visited = set()
        for dock in docks:
            if dock in visited or dock.isFloating() or dock.isHidden():
                continue
            group = [dock, *window.tabifiedDockWidgets(dock)]
            visited.update(group)
            visible = [
                member
                for member in group
                if not member.isHidden() and not member.isFloating()
            ]
            representative = next(
                (
                    member
                    for member in visible
                    if window.rect().intersects(
                        QRect(member.mapTo(window, QPoint()), member.size())
                    )
                ),
                None,
            )
            if representative:
                representatives.append(representative)
        if len(representatives) < 2:
            return

        rectangles = [
            QRect(dock.mapTo(window, QPoint()), dock.size()) for dock in representatives
        ]
        bounds = QRect()
        for rectangle in rectangles:
            bounds = bounds.united(rectangle)

        def simultaneous_splits(horizontal: bool) -> int:
            # A horizontal request balances columns sharing a vertical band;
            # a vertical request balances rows sharing a horizontal band.
            events = []
            for rectangle in rectangles:
                start = rectangle.top() if horizontal else rectangle.left()
                length = rectangle.height() if horizontal else rectangle.width()
                events.extend(((start, 1), (start + length, -1)))
            active = maximum = 0
            for _, change in sorted(events):
                active += change
                maximum = max(maximum, active)
            return max(1, maximum)

        for orientation, length, horizontal in (
            (Qt.Horizontal, bounds.width(), True),
            (Qt.Vertical, bounds.height(), False),
        ):
            size = max(1, length // simultaneous_splits(horizontal))
            window.resizeDocks(
                representatives, [size] * len(representatives), orientation
            )
        # resizeDocks preserves intrinsic min/max and the existing split tree.
        window.layout().activate()

    def equalize_workspace(self) -> None:
        """Balance outer and Script Terminal splits without changing topology."""
        self._equalize_dock_area(
            self.main_window,
            [dock for dock in self.all_docks if dock is not self.sidebar],
        )
        terminal = self.python_terminal
        if terminal.dock_area.isVisible():
            self._equalize_dock_area(
                terminal.dock_area, [terminal.editor_dock, terminal.console_dock]
            )

    def get_all_docks(self) -> list[QDockWidget]:
        """Return a list of all managed dock widgets."""
        return self.all_docks

    def connect_signals(self) -> None:
        """Connect signals for the managed panels."""
        self.sidebar_content_widget.menuSelected.connect(self.on_sidebar_menu_selected)
        self.sidebar_content_widget.showWipMessage.connect(self._on_sidebar_wip_message)
        self.python_terminal.commandExecuted.connect(self.connector.opticChanged.emit)

    def _on_sidebar_wip_message(self, message: str) -> None:
        """Forward a WIP message from the sidebar to the toast manager.

        Args:
            message: The notification text to display.
        """
        tm = getattr(self.main_window, "toast_manager", None)
        if tm is not None:
            tm.notify(message, "info")

    def on_sidebar_menu_selected(self, button_name: str) -> None:
        """Show and raise the dock widget corresponding to a sidebar button.

        Args:
            button_name: Internal name of the clicked sidebar button (e.g.
                ``"analysis"``, ``"design"``).
        """
        dock_map = {
            "analysis": self.analysis_dock,
            "optimization": self.optimization_dock,
            "scripts": self.terminal_dock,
            "design": self.lens_editor_dock,
        }
        dock = dock_map.get(button_name)
        if dock:
            dock.show()
            dock.raise_()

    def update_theme(self, theme_name: str) -> None:
        """Propagate a theme change to all relevant panels.

        Args:
            theme_name: Either ``"dark"`` or ``"light"``.
        """
        self.sidebar_content_widget.update_icons(theme_name)
        self.analysis_panel.update_theme(theme_name)
        self.viewer_panel.update_theme(theme_name)
        self.python_terminal.set_theme(theme_name)
        self.optimization_panel.update_theme(theme_name)
