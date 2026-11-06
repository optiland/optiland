"""Panel controls retain their actions and remain usable in narrower docks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QRect, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QMainWindow

from optiland_gui.config import THEME_DARK_PATH, THEME_LIGHT_PATH
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.optimization_panel import OptimizationPanel
from optiland_gui.system_properties_panel import SystemPropertiesPanel


@pytest.fixture(params=[THEME_LIGHT_PATH, THEME_DARK_PATH], ids=["light", "dark"])
def panels(qapp, request):
    original_style, original_font = qapp.style().objectName(), qapp.font()
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    font_id = None
    if "Segoe UI" not in QFontDatabase.families() and font_path.is_file():
        # Windows' offscreen plugin may not enumerate installed system fonts.
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        assert font_id >= 0
    qapp.setStyle("Fusion")
    qapp.setFont(QFont("Segoe UI", 9))
    connector = OptilandConnector()
    properties = SystemPropertiesPanel(connector)
    properties.load_properties()
    optimization = OptimizationPanel(connector)
    stylesheet = Path(request.param).read_text(encoding="utf-8")
    for panel in (properties, optimization):
        panel.setStyleSheet(stylesheet)
        panel.ensurePolished()
    properties.resize(600, 600)
    optimization.resize(450, 650)
    yield properties, optimization, connector
    for panel in (properties, optimization):
        panel.close()
        panel.deleteLater()
    connector.calculation_jobs.shutdown()
    connector.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.setStyle(original_style)
    qapp.setFont(original_font)
    if font_id is not None:
        QFontDatabase.removeApplicationFont(font_id)


def wavelength_buttons(editor):
    return (
        editor.btnAddWavelength,
        editor.btnRemoveWavelength,
        editor.btnSetPrimary,
        editor.btnApplyWavelengths,
    )


def assert_control_fits(panel, control):
    rect = QRect(control.mapTo(panel, QPoint(0, 0)), control.size())
    assert panel.rect().contains(rect)
    assert control.width() >= control.minimumSizeHint().width()
    assert control.height() >= control.minimumSizeHint().height()


def test_inactive_wavelength_page_does_not_force_a_wide_properties_dock(qapp, panels):
    properties, _, _ = panels
    assert properties.stackedWidget.currentWidget() is properties.apertureEditor
    properties.show()
    qapp.processEvents()
    assert properties.width() == 600
    assert properties.minimumSizeHint().width() < 600
    assert properties.wavelengthsEditor.minimumSizeHint().width() < 450

    properties.stackedWidget.setCurrentWidget(properties.wavelengthsEditor)
    qapp.processEvents()
    for button in wavelength_buttons(properties.wavelengthsEditor):
        assert_control_fits(properties, button)


def test_wavelength_actions_follow_keyboard_order_and_edit_the_model_once(qapp, panels):
    properties, _, connector = panels
    editor = properties.wavelengthsEditor
    properties.stackedWidget.setCurrentWidget(editor)
    properties.show()
    properties.activateWindow()
    qapp.processEvents()
    buttons = wavelength_buttons(editor)
    buttons[0].setFocus()
    for button in buttons[1:]:
        QTest.keyClick(qapp.focusWidget(), Qt.Key_Tab)
        assert qapp.focusWidget() is button

    changes = []
    connector.document_state.committed.connect(changes.append)
    QTest.keyClick(buttons[0], Qt.Key_Space)
    assert editor.tableWavelengths.rowCount() == 2
    editor.tableWavelengths.setCurrentCell(1, 0)
    QTest.keyClick(buttons[2], Qt.Key_Space)
    assert connector.get_optic().wavelengths.primary_index == 1
    editor.tableWavelengths.item(1, 0).setText("0.6100")
    QTest.keyClick(buttons[3], Qt.Key_Space)
    assert connector.get_optic().wavelengths[1].value == pytest.approx(0.6100)
    editor.tableWavelengths.setCurrentCell(1, 0)
    QTest.keyClick(buttons[1], Qt.Key_Space)
    assert editor.tableWavelengths.rowCount() == 1
    assert connector.get_optic().wavelengths.primary_index == 0
    assert len(changes) == 4
    assert all(change.affects_optics for change in changes)


def test_live_updates_controls_fit_without_widening_optimization_tabs(qapp, panels):
    _, optimization, _ = panels
    optimization.show()
    qapp.processEvents()
    assert optimization.width() == 450
    assert optimization.minimumSizeHint().width() < 450
    group = optimization.chkLiveViz.parentWidget()
    assert group.minimumSizeHint().width() < 350
    assert (
        group.minimumSizeHint().width() < optimization._tabs.minimumSizeHint().width()
    )
    for control in (
        optimization.chkLiveViz,
        optimization.chkLiveVars,
        optimization.spnFrequency,
        *group.findChildren(QLabel),
    ):
        assert_control_fits(optimization, control)


def test_live_update_keyboard_controls_preserve_preview_and_variable_options(
    qapp, panels, monkeypatch
):
    _, optimization, connector = panels
    optimization.show()
    optimization.activateWindow()
    qapp.processEvents()
    optimization.chkLiveViz.setFocus()
    QTest.keyClick(optimization.chkLiveViz, Qt.Key_Space)
    assert optimization.chkLiveViz.isChecked()
    QTest.keyClick(optimization.chkLiveViz, Qt.Key_Tab)
    assert qapp.focusWidget() is optimization.chkLiveVars
    QTest.keyClick(optimization.chkLiveVars, Qt.Key_Space)
    assert optimization.chkLiveVars.isChecked()
    QTest.keyClick(optimization.chkLiveVars, Qt.Key_Tab)
    assert qapp.focusWidget() is optimization.spnFrequency
    optimization.spnFrequency.setValue(77)

    run = MagicMock()
    monkeypatch.setattr(connector, "run_optimization", run)
    QTest.keyClick(optimization.btnRun, Qt.Key_Space)
    run.assert_called_once()
    assert connector._optimization_service.preview_options == {
        "enabled": True,
        "frequency": 77,
    }
    optimization.tblVariables.setRowCount(1)
    optimization._on_optimization_progress(
        {"stage": "iteration", "evaluations": 1, "variables": [2.5]}
    )
    assert optimization.tblVariables.item(0, 2).text() == "2.5000"
    QTest.keyClick(optimization.chkLiveVars, Qt.Key_Space)
    optimization._on_optimization_progress(
        {"stage": "iteration", "evaluations": 2, "variables": [3.5]}
    )
    assert optimization.tblVariables.item(0, 2).text() == "2.5000"


def test_visible_optimization_panel_keeps_the_file_menu_keyboard_shortcut(qapp, panels):
    _, optimization, _ = panels
    window = QMainWindow()
    file_menu = window.menuBar().addMenu("&File")
    file_menu.addAction("Open")
    window.setCentralWidget(optimization)
    window.resize(900, 750)
    window.show()
    window.activateWindow()
    qapp.processEvents()
    try:
        optimization.spnFrequency.setFocus()
        QTest.keyClick(window, Qt.Key_F, Qt.AltModifier)
        qapp.processEvents()
        assert file_menu.isVisible()
        assert qapp.activePopupWidget() is file_menu
    finally:
        file_menu.hide()
        window.takeCentralWidget()
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
