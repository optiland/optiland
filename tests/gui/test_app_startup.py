"""The GUI boots with its full panel set and releases resources on close."""

from __future__ import annotations

import logging
import time

import pytest
from PySide6.QtCore import QCoreApplication, QEvent

from optiland_gui.main_window import MainWindow

EXPECTED_DOCKS = {
    "SidebarDockWidget",
    "ViewerDock",
    "LensEditorDock",
    "SystemPropertiesDock",
    "AnalysisPanelDock",
    "OptimizationDock",
    "TerminalDock",
}

# Actions the window cannot function without.  ``about`` carries no shortcut.
CORE_ACTIONS = ("new", "open", "save", "save_as", "undo", "redo", "about")


def _wait_until(qapp, predicate, timeout=20.0):
    """Pump the event loop until *predicate* holds, or fail on timeout."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(
                "Timed out waiting for the application to shut down cleanly"
            )
        qapp.processEvents()
        time.sleep(0.002)


def _released(window):
    """True once the worker process and the embedded kernel are both gone."""
    return (
        getattr(window, "_calculation_shutdown_complete", False)
        and window.connector.calculation_jobs._process is None
        and not window.panel_manager.python_terminal.kernel_manager.is_alive()
    )


@pytest.fixture(scope="module")
def main_window(qapp):
    """A fully initialised window, torn down once for the whole module."""
    window = MainWindow()
    yield window

    window.close()
    _wait_until(qapp, lambda: _released(window))
    # The window installs a root-logger handler bound to its toast manager.
    # Drop it so later tests cannot reach into a deleted widget.
    logging.getLogger().removeHandler(window._gui_log_handler)
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_all_panels_are_docked(main_window):
    docks = main_window.panel_manager.all_docks
    assert {dock.objectName() for dock in docks} == EXPECTED_DOCKS
    assert all(dock.widget() is not None for dock in docks)


def test_default_system_is_ready_to_trace(main_window):
    optic = main_window.connector.get_optic()
    assert optic is not None
    assert len(optic.surfaces) >= 3
    assert len(optic.fields) >= 1
    assert len(optic.wavelengths) >= 1
    assert optic.aperture.ap_type == "EPD"
    assert optic.aperture.value > 0


def test_core_actions_are_available(main_window):
    for name in CORE_ACTIONS:
        action = main_window.action_manager.get_action(name)
        assert action is not None, f"missing action: {name}"
        if name != "about":
            assert not action.shortcut().isEmpty(), f"missing shortcut: {name}"


def test_menu_bar_exposes_primary_menus(main_window):
    titles = {
        action.text() for action in main_window._actual_menu_bar_instance.actions()
    }
    assert {"&File", "&Edit", "&View", "&Help"} <= titles


def test_scripting_interface_points_at_the_window(main_window):
    assert main_window.iface.get_main_window() is main_window
    assert main_window.iface.get_lens_editor() is main_window.panel_manager.lens_editor
