"""Spin-box arrow clicks must reach the control rather than its text editor."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QSpinBox,
    QStyle,
    QStyleFactory,
    QStyleOptionSpinBox,
    QVBoxLayout,
    QWidget,
)


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
