"""Render the real theme resources: checked and mixed states must stay visible."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox, QStyle, QStyleFactory, QStyleOptionButton

from optiland_gui.resources import resources_rc  # noqa: F401


@pytest.fixture(params=["fusion", "windows11"])
def checkbox_style(qapp, request):
    names = {name.lower(): name for name in QStyleFactory.keys()}  # noqa: SIM118
    if request.param not in names:
        pytest.skip(f"Qt style {request.param} is unavailable")
    previous = qapp.style().objectName()
    qapp.setStyle(names[request.param])
    yield qapp
    qapp.setStyle(previous)


def _indicator_rect(box):
    option = QStyleOptionButton()
    box.initStyleOption(option)
    return box.style().subElementRect(QStyle.SE_CheckBoxIndicator, option, box)


def _indicator_pixels(box):
    pixmap = box.grab()
    image = pixmap.toImage()
    rect = _indicator_rect(box).adjusted(3, 3, -3, -3)
    scale = pixmap.devicePixelRatio()
    return [
        image.pixelColor(x, y).getRgb()[:3]
        for y in range(round(rect.top() * scale), round((rect.bottom() + 1) * scale))
        for x in range(round(rect.left() * scale), round((rect.right() + 1) * scale))
    ]


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("enabled", [True, False])
def test_checked_and_mixed_glyphs_render(checkbox_style, theme, enabled):
    box = QCheckBox("Example option")
    box.setAttribute(Qt.WA_ShowWithoutActivating)
    stylesheet = Path(__file__).parents[2] / "optiland_gui/resources/styles"
    box.setStyleSheet((stylesheet / f"{theme}_theme.qss").read_text(encoding="utf-8"))
    box.setTristate(True)
    box.setEnabled(enabled)
    box.resize(200, 40)
    box.show()
    try:
        masks = []
        for state in (Qt.Unchecked, Qt.Checked, Qt.PartiallyChecked):
            box.setCheckState(state)
            checkbox_style.processEvents()
            pixels = _indicator_pixels(box)
            white = tuple(min(pixel) > 240 for pixel in pixels)
            masks.append(white)
            if state != Qt.Unchecked:
                # A bundled SVG must paint a light symbol over the filled box.
                # This fails when the glyph resource is absent or transparent.
                assert sum(white) >= 8
                assert sum(max(pixel) < 210 for pixel in pixels) >= 8
        assert len(set(masks)) == 3, "Unchecked, tick and mixed states must differ"

        box.setTristate(False)
        box.setChecked(False)
        point = _indicator_rect(box).center()
        QTest.mouseClick(box, Qt.LeftButton, pos=point)
        assert box.isChecked() is enabled
        QTest.keyClick(box, Qt.Key_Space)
        assert not box.isChecked()
    finally:
        box.close()
