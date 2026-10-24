"""Spin-box arrow clicks must reach the control rather than its text editor."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
    QStyle,
    QStyleFactory,
    QStyleOptionSpinBox,
    QVBoxLayout,
    QWidget,
)

from optiland_gui.widgets.spinbox_hit_areas import SpinBoxHitAreaGuard


@pytest.fixture(params=["windows11", "fusion"])
def styled_app(qapp, request):
    # QStyleFactory is Qt's factory class, not a mapping.
    names = {name.lower(): name for name in QStyleFactory.keys()}  # noqa: SIM118
    if request.param not in names:
        pytest.skip(f"Qt style {request.param} is unavailable")
    previous_style = qapp.style().objectName()
    previous_sheet = qapp.styleSheet()
    qapp.setStyle(names[request.param])
    yield qapp
    qapp.setStyleSheet(previous_sheet)
    qapp.setStyle(previous_style)


def button_rect(spin, control):
    option = QStyleOptionSpinBox()
    spin.initStyleOption(option)
    return spin.style().subControlRect(
        QStyle.ComplexControl.CC_SpinBox, option, control, spin
    )


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("spin_type", [QSpinBox, QDoubleSpinBox])
@pytest.mark.parametrize("width", [100, 180])
def test_arrows_are_separate_and_clickable(styled_app, theme, spin_type, width):
    theme_path = (
        Path(__file__).parents[2]
        / "optiland_gui/resources/styles"
        / f"{theme}_theme.qss"
    )
    styled_app.setStyleSheet(theme_path.read_text(encoding="utf-8"))
    container = QWidget()
    guard = SpinBoxHitAreaGuard(container)
    container.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    layout = QVBoxLayout(container)
    spin = spin_type()
    spin.setRange(0, 100)
    spin.setFixedWidth(width)
    spin.setValue(50)
    other = QSpinBox()
    layout.addWidget(spin)
    layout.addWidget(other)
    container.show()
    styled_app.processEvents()
    try:
        for focused in (False, True):
            (spin if focused else other).setFocus()
            styled_app.processEvents()
            for control, delta in (
                (QStyle.SubControl.SC_SpinBoxUp, 1),
                (QStyle.SubControl.SC_SpinBoxDown, -1),
            ):
                rect = button_rect(spin, control)
                assert not rect.isEmpty()
                assert not spin.lineEdit().geometry().intersects(rect)
                for point in (
                    rect.center(),
                    QPoint(rect.left(), rect.center().y()),
                    QPoint(rect.right(), rect.center().y()),
                    rect.topLeft() + QPoint(2, 2),
                    rect.bottomRight() - QPoint(2, 2),
                ):
                    spin.setValue(50)
                    target = spin.childAt(point) or spin
                    assert target is spin
                    QTest.mouseClick(target, Qt.MouseButton.LeftButton, pos=point)
                    assert spin.value() == 50 + delta
        spin.setValue(50)
        QTest.keyClick(spin.lineEdit(), Qt.Key.Key_Up)
        assert spin.value() == 51
        QTest.keyClick(spin.lineEdit(), Qt.Key.Key_Down)
        assert spin.value() == 50
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin.lineEdit(), "42")
        QTest.keyClick(spin.lineEdit(), Qt.Key.Key_Return)
        assert spin.value() == 42
        for value, control in (
            (100, QStyle.SubControl.SC_SpinBoxUp),
            (0, QStyle.SubControl.SC_SpinBoxDown),
        ):
            spin.setValue(value)
            QTest.mouseClick(
                spin, Qt.MouseButton.LeftButton, pos=button_rect(spin, control).center()
            )
            assert spin.value() == value
    finally:
        container.close()
        container.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("direction", [Qt.LeftToRight, Qt.RightToLeft])
def test_hit_areas_survive_theme_font_and_size_changes(styled_app, direction):
    container = QWidget()
    guard = SpinBoxHitAreaGuard(container)
    layout = QVBoxLayout(container)
    spin = QDoubleSpinBox()
    spin.setLayoutDirection(direction)
    layout.addWidget(spin)
    container.show()
    try:
        for theme in ("dark", "light", "dark"):
            path = Path(__file__).parents[2] / "optiland_gui/resources/styles"
            styled_app.setStyleSheet((path / f"{theme}_theme.qss").read_text())
            for width, points in ((180, 10), (100, 14), (240, 12)):
                font = spin.font()
                font.setPointSize(points)
                spin.setFont(font)
                spin.setFixedWidth(width)
                styled_app.processEvents()
                for control in (QStyle.SC_SpinBoxUp, QStyle.SC_SpinBoxDown):
                    rect = button_rect(spin, control)
                    assert not spin.lineEdit().geometry().intersects(rect)
                    assert spin.childAt(rect.left(), rect.center().y()) is None
                    assert spin.childAt(rect.right(), rect.center().y()) is None
    finally:
        container.close()
        container.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_guard_leaves_other_editors_and_buttonless_spinboxes_alone(styled_app):
    container = QWidget()
    guard = SpinBoxHitAreaGuard(container)
    external = QSpinBox()
    layout = QVBoxLayout(container)
    editor = QLineEdit()
    spin = QSpinBox()
    spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
    layout.addWidget(editor)
    layout.addWidget(spin)
    extra = QLineEdit(spin)
    container.show()
    external.show()
    styled_app.processEvents()
    try:
        # Exercise actual child geometry notifications, including an unrelated
        # editor parented to a spinbox and a spinbox in another window.
        for target in (editor, spin.lineEdit(), extra, external.lineEdit()):
            target.setGeometry(2, 2, 30, 20)
            assert target.geometry().getRect() == (2, 2, 30, 20)
    finally:
        external.close()
        external.deleteLater()
        container.close()
        container.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_guard_respects_arrows_positioned_on_the_left(styled_app):
    container = QWidget()
    guard = SpinBoxHitAreaGuard(container)
    layout = QVBoxLayout(container)
    spin = QSpinBox()
    layout.addWidget(spin)
    spin.setStyleSheet("""
        QSpinBox { border: 1px solid gray; padding: 5px; }
        QSpinBox::up-button {
            subcontrol-origin: border;
            subcontrol-position: top left;
        }
        QSpinBox::down-button {
            subcontrol-origin: border;
            subcontrol-position: bottom left;
        }
    """)
    container.show()
    styled_app.processEvents()
    try:
        # A custom editor or a style relayout can reassign the editor geometry.
        # Both left-hand arrow targets must remain protected after that change.
        spin.lineEdit().setGeometry(spin.rect())
        for control in (QStyle.SC_SpinBoxUp, QStyle.SC_SpinBoxDown):
            rect = button_rect(spin, control)
            assert rect.center().x() < spin.rect().center().x()
            assert not spin.lineEdit().geometry().intersects(rect)
            assert spin.childAt(rect.right(), rect.center().y()) is None
    finally:
        container.close()
        container.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
