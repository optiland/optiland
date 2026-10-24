"""Panel visibility must not change the constraints used by Qt's dock layout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

from optiland_gui.main_window import MainWindow


class DockWindow(MainWindow):
    """Exercise main-window docking without optical panels or a Python kernel."""

    def __init__(self):
        QMainWindow.__init__(self)
        self.resize(1200, 800)
        self.setCentralWidget(QWidget())
        self.setDockNestingEnabled(True)

    # MainWindow's event handlers depend on the full application's services.
    showEvent = QMainWindow.showEvent
    resizeEvent = QMainWindow.resizeEvent
    closeEvent = QMainWindow.closeEvent


@pytest.fixture(params=["left", "top", "floating", "tabbed"])
def dock_layout(qapp, request):
    window = DockWindow()
    dock = QDockWidget("Test panel", window)
    dock.setObjectName("TestPanelDock")
    dock.setWidget(QWidget())
    dock.setMinimumSize(120, 90)
    dock.setMaximumSize(1500, 1000)
    placement = request.param
    area = (
        Qt.DockWidgetArea.TopDockWidgetArea
        if placement == "top"
        else Qt.DockWidgetArea.LeftDockWidgetArea
    )
    window.addDockWidget(area, dock)
    other = None
    if placement == "tabbed":
        other = QDockWidget("Other panel", window)
        other.setWidget(QWidget())
        window.addDockWidget(area, other)
        window.tabifyDockWidget(dock, other)
    if placement == "floating":
        dock.setFloating(True)
        dock.resize(240, 180)
    window.show()
    dock.raise_()
    qapp.processEvents()
    if placement != "floating":
        orientation = Qt.Vertical if placement == "top" else Qt.Horizontal
        window.resizeDocks([dock], [180 if placement == "top" else 240], orientation)
        qapp.processEvents()
    yield window, dock, other, placement
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def constraints(dock):
    return dock.minimumSize(), dock.maximumSize()


def test_scripted_toggle_changes_visibility_immediately(dock_layout):
    window, dock, _, _ = dock_layout
    before = constraints(dock)

    window.animate_dock_toggle(dock, False)
    assert dock.isHidden()
    assert not dock.toggleViewAction().isChecked()
    assert constraints(dock) == before

    window.animate_dock_toggle(dock, True)
    assert not dock.isHidden()
    assert dock.toggleViewAction().isChecked()
    assert constraints(dock) == before


def test_rapid_and_repeated_toggles_preserve_last_requested_state(dock_layout):
    window, dock, _, _ = dock_layout
    before = constraints(dock)

    # No event processing between calls: the final request must win immediately.
    for visible in (False, False, True, False, True, True, False):
        window.animate_dock_toggle(dock, visible)
        assert dock.isHidden() is not visible
        assert dock.toggleViewAction().isChecked() is visible
        assert constraints(dock) == before
    QTest.qWait(200)  # A completed or interrupted old animation must not intervene.
    assert dock.isHidden()

    # The old animation cache retained C++ objects deleted on completion.
    window.animate_dock_toggle(dock, True)
    QTest.qWait(200)
    window.animate_dock_toggle(dock, False)
    assert dock.isHidden()
    assert constraints(dock) == before


def test_native_view_action_preserves_constraints_and_stays_synchronized(dock_layout):
    window, dock, _, _ = dock_layout
    action = dock.toggleViewAction()
    before = constraints(dock)

    for _ in range(3):
        action.trigger()
        assert dock.isHidden()
        assert not action.isChecked()
        assert constraints(dock) == before
        action.trigger()
        assert not dock.isHidden()
        assert action.isChecked()
        assert constraints(dock) == before

    dock.close()
    assert not action.isChecked()
    window.animate_dock_toggle(dock, True)
    assert action.isChecked()


def test_panel_can_grow_after_being_shown_again(qapp, dock_layout):
    window, dock, _, placement = dock_layout
    orientation = Qt.Vertical if placement == "top" else Qt.Horizontal
    size_before = dock.height() if placement == "top" else dock.width()
    before = constraints(dock)
    window.animate_dock_toggle(dock, False)
    window.animate_dock_toggle(dock, True)
    QTest.qWait(200)

    requested_size = size_before + 100
    if placement == "floating":
        dock.resize(requested_size, dock.height() + 100)
    else:
        window.resizeDocks([dock], [requested_size], orientation)
    qapp.processEvents()
    size_after = dock.height() if placement == "top" else dock.width()
    assert size_after > size_before + 50
    assert constraints(dock) == before


@pytest.mark.parametrize("dock_layout", ["tabbed"], indirect=True)
def test_showing_a_tabbed_panel_brings_it_to_the_front(qapp, dock_layout):
    window, dock, other, _ = dock_layout
    other.raise_()
    qapp.processEvents()
    window.animate_dock_toggle(dock, True)
    qapp.processEvents()
    # Qt moves inactive dock tabs outside the window's content rectangle.
    assert dock.geometry().intersects(window.rect())
    assert not other.geometry().intersects(window.rect())


class ConnectorSignals(QObject):
    opticLoaded = Signal()
    opticChanged = Signal()
    modifiedStateChanged = Signal()


def test_final_setup_keeps_native_actions_and_existing_listeners(qapp, monkeypatch):
    import optiland_gui.main_window as module

    window = DockWindow()
    dock = QDockWidget("Test panel", window)
    dock.setWidget(QWidget())
    window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    window.show()
    qapp.processEvents()
    action = dock.toggleViewAction()
    observed = []
    action.triggered.connect(lambda checked: observed.append(checked))

    # Isolate finalization's action wiring from optical services and overlays.
    window.current_theme_path = ""
    window.action_manager = SimpleNamespace(get_action=lambda _: None)
    window.panel_manager = SimpleNamespace(
        get_all_docks=lambda: [dock], connect_signals=lambda: None
    )
    window.connector = ConnectorSignals(window)
    window.settings = object()
    window.command_registry = object()
    window.load_stylesheets = lambda: None
    window._register_palette_commands = lambda: None
    window._update_project_name_in_title_bar = lambda: None
    monkeypatch.setattr(module, "ToastManager", lambda _: SimpleNamespace())
    monkeypatch.setattr(module._log_handler, "install", lambda _: object())
    monkeypatch.setattr(
        module,
        "CommandPaletteWidget",
        lambda *_: SimpleNamespace(toggle=lambda: None),
    )
    try:
        window._finalize_setup()
        action.trigger()
        assert dock.isHidden()
        assert not action.isChecked()
        action.trigger()
        assert not dock.isHidden()
        assert action.isChecked()
        assert observed == [False, True]
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
