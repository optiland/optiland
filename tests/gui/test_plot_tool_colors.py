"""Plot navigation controls retain legibility and behavior across themes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QEnterEvent, QFont, QFontDatabase, QIcon
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from optiland_gui.analysis_panel import AnalysisPanel, CustomMatplotlibToolbar
from optiland_gui.config import THEME_DARK_PATH, THEME_LIGHT_PATH
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.resources import resources_rc  # noqa: F401
from optiland_gui.viewer_panel import MatplotlibViewer, SagViewer


@pytest.fixture
def controls(qapp):
    previous_style = qapp.styleSheet()
    font = Path("C:/Windows/Fonts/segoeui.ttf")
    if not QFontDatabase.families() and font.exists():
        assert QFontDatabase.addApplicationFont(str(font)) >= 0
        qapp.setFont(QFont("Segoe UI", 9))
    window = QMainWindow()
    content = QWidget()
    layout = QVBoxLayout(content)
    container = QWidget()
    container.setObjectName("ViewerToolbarContainer")
    toolbar_layout = QVBoxLayout(container)
    canvas = FigureCanvasQTAgg(Figure(figsize=(6, 2)))
    canvas.figure.subplots().plot([0, 1, 2], [0, 1, 0])
    toolbar = CustomMatplotlibToolbar(canvas, container)
    toolbar_layout.addWidget(toolbar)
    layout.addWidget(container)
    layout.addWidget(canvas)
    window.setCentralWidget(content)
    window.resize(1050, 540)
    window.show()
    window.activateWindow()
    qapp.processEvents()
    yield window, canvas, toolbar
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    qapp.setStyleSheet(previous_style)


def set_theme(qapp, controls, theme):
    path = THEME_DARK_PATH if theme == "dark" else THEME_LIGHT_PATH
    qapp.setStyleSheet(Path(path).read_text())
    controls[2].update_theme(theme)
    qapp.processEvents()


def opaque_colors(icon, mode=QIcon.Mode.Normal, state=QIcon.State.Off):
    image = icon.pixmap(32, mode, state).toImage()
    return Counter(
        image.pixelColor(x, y).name()
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() == 255
    )


def hover(qapp, button, entered):
    """Route enter/leave events; offscreen has no physical cursor crossings."""
    event = (
        QEnterEvent(QPointF(button.rect().center()), QPointF(), QPointF())
        if entered
        else QEvent(QEvent.Type.Leave)
    )
    qapp.sendEvent(button, event)
    qapp.processEvents()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_navigation_modes_render_legibly_after_repeated_switches(qapp, controls, theme):
    _, canvas, toolbar = controls
    actions = list(toolbar.actions())
    sources = {
        action: (source.cacheKey(), source.devicePixelRatio())
        for action, source in toolbar._navigation_icons.items()
    }
    axes = canvas.figure.axes[0]
    axes.set_xlim(0.25, 1.5)
    toolbar.push_current()
    toolbar.pan()
    foreground = "#e0e0e0" if theme == "dark" else "#212529"
    disabled = "#8d969f" if theme == "dark" else "#6c757d"
    for selected_theme in (theme, "dark" if theme == "light" else "light", theme):
        set_theme(qapp, controls, selected_theme)
    for action in toolbar._navigation_icons:
        for mode in (QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected):
            for state in (QIcon.State.Off, QIcon.State.On):
                assert (
                    opaque_colors(action.icon(), mode, state).most_common(1)[0][0]
                    == foreground
                )
        assert (
            opaque_colors(action.icon(), QIcon.Mode.Disabled).most_common(1)[0][0]
            == disabled
        )
    assert toolbar.actions() == actions
    assert sources == {
        action: (source.cacheKey(), source.devicePixelRatio())
        for action, source in toolbar._navigation_icons.items()
    }
    assert toolbar.mode.name == "PAN"
    assert toolbar._actions["pan"].isChecked()
    assert axes.get_xlim() == (0.25, 1.5)
    # QAction connections stay live; keyboard activation toggles the same mode.
    button = toolbar.widgetForAction(toolbar._actions["pan"])
    button.setFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(button, Qt.Key.Key_Space)
    assert toolbar.mode.name == "NONE"
    assert not toolbar._actions["pan"].isChecked()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_navigation_checked_hover_and_focus_have_visible_non_color_cues(
    qapp, controls, theme
):
    window, _, toolbar = controls
    set_theme(qapp, controls, theme)
    button = toolbar.widgetForAction(toolbar._actions["pan"])
    window.setFocus()
    QTest.mouseMove(window, QPoint(window.width() - 2, window.height() - 2))
    hover(qapp, button, False)
    qapp.processEvents()
    normal = button.grab().toImage()
    toolbar.pan()
    qapp.processEvents()
    checked = button.grab().toImage()
    foreground = "#e0e0e0" if theme == "dark" else "#212529"
    assert normal != checked
    assert checked.pixelColor(button.width() // 2, 0).name() == foreground
    button.setFocus(Qt.FocusReason.TabFocusReason)
    qapp.processEvents()
    focused = button.grab().toImage()
    assert focused != checked  # Dashed border replaces the solid checked border.
    button.clearFocus()
    toolbar.pan()
    QTest.mouseMove(button, button.rect().center())
    hover(qapp, button, True)
    qapp.processEvents()
    hovered = button.grab().toImage()
    assert hovered != normal


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_disabled_and_history_buttons_keep_their_behavior_after_theme_changes(
    qapp, controls, theme
):
    _, canvas, toolbar = controls
    axes = canvas.figure.axes[0]
    original = axes.get_xlim()
    back = toolbar._actions["back"]
    forward = toolbar._actions["forward"]
    activations = []
    back.triggered.connect(lambda: activations.append("back"))
    set_theme(qapp, controls, theme)
    assert not back.isEnabled() and not forward.isEnabled()
    QTest.mouseClick(toolbar.widgetForAction(back), Qt.LeftButton)
    assert activations == [] and axes.get_xlim() == original

    toolbar.push_current()
    axes.set_xlim(0.25, 1.5)
    toolbar.push_current()
    set_theme(qapp, controls, "dark" if theme == "light" else "light")
    set_theme(qapp, controls, theme)
    assert back.isEnabled() and not forward.isEnabled()
    QTest.mouseClick(toolbar.widgetForAction(back), Qt.LeftButton)
    assert activations == ["back"] and axes.get_xlim() == original
    assert not back.isEnabled() and forward.isEnabled()
    QTest.mouseClick(toolbar.widgetForAction(forward), Qt.LeftButton)
    assert axes.get_xlim() == (0.25, 1.5)
    QTest.mouseClick(toolbar.widgetForAction(toolbar._actions["home"]), Qt.LeftButton)
    assert axes.get_xlim() == original

    zoom = toolbar.widgetForAction(toolbar._actions["zoom"])
    zoom.setFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(zoom, Qt.Key.Key_Space)
    assert toolbar.mode.name == "ZOOM" and toolbar._actions["zoom"].isChecked()
    QTest.keyClick(zoom, Qt.Key.Key_Space)
    assert toolbar.mode.name == "NONE" and not toolbar._actions["zoom"].isChecked()


@pytest.mark.parametrize("viewer_class", [MatplotlibViewer, SagViewer])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_viewers_theme_navigation_without_requests_or_mode_changes(
    qapp, viewer_class, theme
):
    connector = OptilandConnector()
    viewer = viewer_class(connector)
    serial = connector.calculation_jobs._serial
    actions = list(viewer.toolbar.actions())
    viewer.toolbar.pan()
    for selected_theme in (theme, "dark" if theme == "light" else "light", theme):
        viewer.update_theme(selected_theme)
    expected = "#e0e0e0" if theme == "dark" else "#212529"
    assert (
        opaque_colors(viewer.toolbar._actions["home"].icon()).most_common(1)[0][0]
        == expected
    )
    assert viewer.toolbar.actions() == actions
    assert viewer.toolbar.mode.name == "PAN"
    assert connector.calculation_jobs._serial == serial
    assert not viewer.settings_toggle_btn.icon().isNull()
    checked_before = viewer.settings_toggle_btn.isChecked()
    viewer.settings_toggle_btn.click()
    assert viewer.settings_toggle_btn.isChecked() != checked_before
    assert viewer.settings_area.isHidden() == checked_before
    viewer.settings_toggle_btn.click()
    assert viewer.settings_toggle_btn.isChecked() == checked_before
    assert viewer.settings_area.isHidden() != checked_before
    viewer.close()
    viewer.deleteLater()
    connector.calculation_jobs.shutdown()
    connector.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_analysis_toolbar_starts_with_current_theme(qapp, theme):
    connector = OptilandConnector()
    panel = AnalysisPanel(connector)
    panel.current_theme = theme
    canvas = FigureCanvasQTAgg(Figure(figsize=(6, 2)))
    panel._setup_plot_toolbar(canvas)
    toolbar = panel.active_mpl_toolbar_widget
    assert toolbar.canvas is canvas
    assert opaque_colors(toolbar._actions["home"].icon()).most_common(1)[0][0] == (
        "#e0e0e0" if theme == "dark" else "#212529"
    )
    panel.close()
    panel.deleteLater()
    canvas.close()
    canvas.deleteLater()
    connector.calculation_jobs.shutdown()
    connector.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
