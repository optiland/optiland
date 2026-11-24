"""Manages the creation and handling of QActions for the Optiland GUI.

This module provides :class:`ActionManager`, which is responsible for
instantiating and configuring all the :class:`~PySide6.QtGui.QAction` objects
used in the application's menus and toolbars.  Separating action definitions
from the main window keeps :class:`~optiland_gui.main_window.MainWindow` lean
and focused on layout / orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QProgressBar, QPushButton

from .config import THEME_DARK_PATH, THEME_LIGHT_PATH

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow

    from .optiland_connector import OptilandConnector


class ActionManager:
    """Creates and manages all :class:`QAction` objects for the application.

    Args:
        main_window: The application's main window.
        connector: Central :class:`~optiland_gui.optiland_connector.OptilandConnector`
            instance, used to wire undo/redo signal connections.
    """

    def __init__(self, main_window: QMainWindow, connector: OptilandConnector) -> None:
        self.main_window = main_window
        self.connector = connector
        self.actions: dict[str, QAction | QActionGroup] = {}

    def create_all_actions(self) -> None:
        """Create all actions and store them in the :attr:`actions` dictionary."""
        self._create_file_actions()
        self._create_edit_actions()
        self._create_view_actions()
        self._create_layout_actions()
        self._create_theme_actions()
        self._create_help_actions()

    def _create_action(
        self,
        name: str,
        text: str,
        shortcut: QKeySequence | str | None = None,
        triggered: object | None = None,
        tooltip: str | None = None,
        checkable: bool = False,
    ) -> QAction:
        """Factory method for creating and registering a single :class:`QAction`.

        Args:
            name: Registry key used to retrieve the action later.
            text: Display text (used in menus and tooltips).
            shortcut: Optional keyboard shortcut.
            triggered: Optional callable to connect to the ``triggered`` signal.
            tooltip: Optional tooltip text.
            checkable: Whether the action should be checkable.

        Returns:
            The newly created :class:`QAction`.
        """
        action = QAction(text, self.main_window, checkable=checkable)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if triggered:
            action.triggered.connect(triggered)
        if tooltip:
            action.setToolTip(tooltip)
        self.actions[name] = action
        return action

    def _create_file_actions(self) -> None:
        """Create all File-menu actions."""
        self._create_action(
            "new", "&New System", QKeySequence.New, self.main_window.new_system_action
        )
        self._create_action(
            "open",
            "&Open System...",
            QKeySequence.Open,
            self.main_window.open_system_action,
        )
        self._create_action(
            "save",
            "&Save System",
            QKeySequence.Save,
            self.main_window.save_system_action,
        )
        self._create_action(
            "save_as",
            "Save System &As...",
            QKeySequence.SaveAs,
            self.main_window.save_system_as_action,
        )
        self._create_action(
            "import_zemax",
            "From &Zemax (.zmx)...",
            triggered=self.main_window.import_zemax_action,
        )
        self._create_action(
            "import_codev",
            "From &CODE V (.seq)...",
            triggered=self.main_window.import_codev_action,
        )
        self._create_action(
            "export_zemax",
            "To &Zemax (.zmx)...",
            triggered=self.main_window.export_zemax_action,
        )
        self._create_action(
            "export_codev",
            "To &CODE V (.seq)...",
            triggered=self.main_window.export_codev_action,
        )
        self._create_action("exit", "E&xit", "Ctrl+Q", self.main_window.close)

    def _create_edit_actions(self) -> None:
        """Create Undo and Redo actions and wire their enabled
        state to the connector."""
        undo = self._create_action(
            "undo", "&Undo", QKeySequence.Undo, self.connector.request_undo
        )
        redo = self._create_action(
            "redo", "&Redo", QKeySequence.Redo, self.connector.request_redo
        )
        undo.setEnabled(False)
        redo.setEnabled(False)
        self.connector.undoStackAvailabilityChanged.connect(
            self._refresh_history_actions
        )
        self.connector.redoStackAvailabilityChanged.connect(
            self._refresh_history_actions
        )
        self.connector.history.state_changed.connect(self._history_state_changed)
        status = self.main_window.statusBar()
        self.history_progress = QProgressBar(status)
        self.history_progress.setRange(0, 0)
        self.history_progress.setFixedWidth(100)
        self.history_progress.setAccessibleName("History preparation progress")
        self.history_cancel = QPushButton("Cancel history", status)
        self.history_cancel.clicked.connect(self.connector.history.cancel)
        status.addPermanentWidget(self.history_progress)
        status.addPermanentWidget(self.history_cancel)
        self.history_progress.hide()
        self.history_cancel.hide()
        self._refresh_history_actions()

    def _refresh_history_actions(self, *args):
        manager = self.connector._undo_redo_manager
        available = not self.connector.history.busy
        self.actions["undo"].setEnabled(available and manager.can_undo())
        self.actions["redo"].setEnabled(available and manager.can_redo())

    def _history_state_changed(self, busy, message):
        self._refresh_history_actions()
        self.history_progress.setVisible(busy)
        self.history_cancel.setVisible(busy)
        self.main_window.statusBar().showMessage(message, 0 if busy else 10000)

    def _create_view_actions(self) -> None:
        """Create View-menu actions for docking and layout reset."""
        self._create_action(
            "dock_all",
            "Dock All Windows",
            triggered=self.main_window.reset_windows_action,
        )
        self._create_action(
            "reset_layout",
            "Reset Window Layout",
            triggered=self.main_window.reset_windows_action,
        )
        self._create_action(
            "arrange_two_panes",
            "Arrange &Two Panes",
            triggered=self.main_window.arrange_two_panes,
            tooltip="Design and viewer tabs beside analysis and scripting tabs",
        )
        self._create_action(
            "arrange_four_panes",
            "Arrange &Four Panes",
            triggered=self.main_window.arrange_four_panes,
            tooltip="Design, viewer, analysis and scripts in a two-by-two layout",
        )
        self._create_action(
            "equalize_panes",
            "&Equalize Split Panes",
            triggered=self.main_window.equalize_panes,
            tooltip="Balance visible splits within each panel's size limits",
        )

    def _create_layout_actions(self) -> None:
        """Create named controls for the two supported saved workspace slots."""
        self._create_action(
            "load_layout_1",
            "Restore Workspace 1",
            triggered=self.main_window.load_layout_1_slot,
        )
        self._create_action(
            "load_layout_2",
            "Restore Workspace 2",
            triggered=self.main_window.load_layout_2_slot,
        )
        for slot in (1, 2):
            self._create_action(
                f"save_layout_{slot}",
                f"Save to Workspace {slot}",
                triggered=lambda checked=False, number=slot: (
                    self.main_window.save_named_layout_slot(number)
                ),
            )
            self._create_action(
                f"rename_layout_{slot}",
                f"Rename Workspace {slot}...",
                triggered=lambda checked=False, number=slot: (
                    self.main_window.rename_layout_slot(number)
                ),
            )
        self._create_action(
            "save_layout",
            "Save to Next Workspace",
            triggered=self.main_window.save_layout_slot,
        )
        self.update_layout_actions()

    def update_layout_actions(self) -> None:
        """Keep workspace names, availability and the next-save destination current."""
        settings = self.main_window.settings
        for slot in (1, 2):
            name = self.main_window._workspace_name(slot)
            menu_name = name.replace("&", "&&")
            load = self.get_action(f"load_layout_{slot}")
            load.setText(f"Restore {menu_name}")
            load.setToolTip(f"Restore workspace {slot}: {name}")
            load.setEnabled(
                settings.contains(f"Layouts/Config{slot}Geometry")
                and settings.contains(f"Layouts/Config{slot}State")
            )
            self.get_action(f"save_layout_{slot}").setText(f"Save to {menu_name}")
            self.get_action(f"rename_layout_{slot}").setText(f"Rename {menu_name}...")
        target = self.main_window.next_save_slot_index
        self.get_action("save_layout").setText(f"Save to Next Workspace ({target})")
        self.get_action("save_layout").setToolTip(
            f"Save the current layout to workspace {target}: "
            f"{self.main_window._workspace_name(target)}"
        )

    def _create_theme_actions(self) -> None:
        """Create mutually exclusive Dark / Light theme actions."""
        group = QActionGroup(self.main_window)
        group.setExclusive(True)
        dark_action = self._create_action(
            "dark_theme",
            "Dark Theme",
            checkable=True,
            triggered=lambda: self.main_window.switch_theme(THEME_DARK_PATH),
        )
        light_action = self._create_action(
            "light_theme",
            "Light Theme",
            checkable=True,
            triggered=lambda: self.main_window.switch_theme(THEME_LIGHT_PATH),
        )
        group.addAction(dark_action)
        group.addAction(light_action)
        self.actions["theme_group"] = group

    def _create_help_actions(self) -> None:
        """Create Help-menu actions."""
        self._create_action(
            "about", "&About Optiland GUI", triggered=self.main_window.about_action
        )

    def get_action(self, name: str) -> QAction | None:
        """Return the registered action for *name*, or ``None``.

        Args:
            name: The registry key used when the action was created.

        Returns:
            The :class:`QAction` if found, or ``None``.
        """
        return self.actions.get(name)

    def get_actions(self, *names: str) -> list[QAction | None]:
        """Return multiple actions by their registry keys.

        Args:
            *names: Registry keys to look up.

        Returns:
            A list of :class:`QAction` objects (``None`` for unknown keys).
        """
        return [self.actions.get(name) for name in names]
