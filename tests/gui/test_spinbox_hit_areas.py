"""Spin-box arrow clicks must reach the control rather than its text editor."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from matplotlib import get_data_path
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPoint, QRect, Qt
from PySide6.QtGui import (
    QFont,
    QFontDatabase,
    QFontInfo,
    QFontMetrics,
    QRawFont,
    QStandardItemModel,
)
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox as NativeDoubleSpinBox,
    QLineEdit,
    QSpinBox as NativeSpinBox,
    QStyle,
    QStyleFactory,
    QStyleOptionSpinBox,
    QStyleOptionViewItem,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from optiland_gui.widgets.spinbox_hit_areas import (
    DoubleSpinBox,
    SpinBox,
    SpinBoxHitAreaGuard,
    protect_spinbox_editor,
)


@contextmanager
def spinbox_test_font(qapp, path):
    """Require the same usable, packaged font on every test platform."""
    assert path.is_file(), f"Required spinbox test font is missing: {path}"
    font_id = QFontDatabase.addApplicationFont(str(path))
    assert font_id >= 0, f"Qt could not load the spinbox test font: {path}"
    previous_font = QFont(qapp.font())
    try:
        families = QFontDatabase.applicationFontFamilies(font_id)
        assert families, f"No font family registered from {path}"
        font = QFont(families[0], 9)
        assert QFontInfo(font).family() == families[0], "Qt substituted the test font"
        raw_font = QRawFont.fromFont(font)
        assert raw_font.isValid(), f"Qt cannot render the test font: {path}"
        assert all(raw_font.supportsCharacter(ord(c)) for c in "0123456789.-"), (
            "Spinbox test font lacks numeric glyphs"
        )
        metrics = QFontMetrics(font)
        assert metrics.height() > 0 and metrics.horizontalAdvance("0123456789") > 0, (
            "Spinbox test font has unusable metrics"
        )
        qapp.setFont(font)
        yield
    finally:
        qapp.setFont(previous_font)
        assert QFontDatabase.removeApplicationFont(font_id)


@pytest.fixture(params=["windows11", "fusion"])
def styled_app(qapp, request):
    # QStyleFactory is Qt's factory class, not a mapping.
    names = {name.lower(): name for name in QStyleFactory.keys()}  # noqa: SIM118
    if request.param not in names:
        # Windows-specific cases do not run on Linux: Qt's Windows11 style is
        # unavailable there. Fusion still exercises the portable regressions.
        pytest.skip(f"Qt style {request.param} is unavailable")
    previous_style = qapp.style().objectName()
    previous_sheet = qapp.styleSheet()
    # Matplotlib is already a required GUI dependency and ships this font on
    # Linux and Windows. Do not fall back to machine-specific fonts or skip if
    # its font data is missing: that would weaken the geometry assertions.
    font_path = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    with spinbox_test_font(qapp, font_path):
        try:
            qapp.setStyle(names[request.param])
            yield qapp
        finally:
            qapp.setStyleSheet(previous_sheet)
            qapp.setStyle(previous_style)


def button_rect(spin, control):
    option = QStyleOptionSpinBox()
    spin.initStyleOption(option)
    return spin.style().subControlRect(
        QStyle.ComplexControl.CC_SpinBox, option, control, spin
    )


@pytest.mark.parametrize("missing", [True, False], ids=["missing", "corrupt"])
def test_required_font_cannot_be_silently_ignored(qapp, tmp_path, missing):
    path = tmp_path / "unusable.ttf"
    if not missing:
        path.write_bytes(b"not a font")
    previous_font = QFont(qapp.font())
    message = "font is missing" if missing else "could not load"
    with pytest.raises(AssertionError, match=message), spinbox_test_font(qapp, path):
        pytest.fail("An unavailable font must not reach the test body")
    assert qapp.font() == previous_font


def test_loaded_font_must_be_renderable(qapp, monkeypatch):
    path = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    previous_font = QFont(qapp.font())
    registered = []
    original_add = QFontDatabase.addApplicationFont

    def record_registration(path):
        font_id = original_add(path)
        registered.append(font_id)
        return font_id

    monkeypatch.setattr(QFontDatabase, "addApplicationFont", record_registration)
    monkeypatch.setattr(QRawFont, "fromFont", lambda font: QRawFont())
    with (
        pytest.raises(AssertionError, match="cannot render"),
        spinbox_test_font(qapp, path),
    ):
        pytest.fail("An unusable font must not reach the test body")
    assert qapp.font() == previous_font
    assert len(registered) == 1 and registered[0] >= 0
    assert not QFontDatabase.applicationFontFamilies(registered[0])


def test_wide_right_arrow_uses_editor_edge(styled_app):
    """Reproduce the midpoint defect on Fusion as well as native Windows."""
    container = QWidget()
    spin = SpinBox(container)
    spin.setFixedSize(100, 40)
    spin.setRange(0, 100)
    spin.setValue(50)
    spin.setStyleSheet("""
        QSpinBox { border: 1px solid gray; padding-right: 80px; }
        QSpinBox::up-button {
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 40px;
            right: 40px;
        }
        QSpinBox::down-button {
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 40px;
        }
    """)
    container.show()
    styled_app.processEvents()
    try:
        up = button_rect(spin, QStyle.SC_SpinBoxUp)
        assert up.left() > 0 and up.center().x() < spin.rect().center().x()
        # Force the one-column editor/arrow overlap observed in Qt 6.12.
        # This does not depend on an OS-specific font or native-style metrics.
        overlap = QRect(0, up.top(), up.left() + 1, up.height())
        assert overlap.intersects(up)
        spin.lineEdit().setGeometry(overlap)
        assert spin.lineEdit().geometry().right() == up.left() - 1
        point = QPoint(up.left(), up.center().y())
        assert spin.childAt(point) is None
        QTest.mouseClick(spin, Qt.LeftButton, pos=point)
        assert spin.value() == 51
    finally:
        container.close()
        container.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("spin_type", [SpinBox, DoubleSpinBox])
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
    other = SpinBox()
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
    layout = QVBoxLayout(container)
    spin = DoubleSpinBox()
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
    external = NativeSpinBox()
    layout = QVBoxLayout(container)
    editor = QLineEdit()
    spin = SpinBox()
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
    layout = QVBoxLayout(container)
    spin = SpinBox()
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


@pytest.mark.parametrize("spin_type", [SpinBox, DoubleSpinBox])
def test_filter_observes_only_its_editor(qapp, monkeypatch, spin_type):
    calls = []
    original = SpinBoxHitAreaGuard.eventFilter

    def observe(guard, watched, event):
        calls.append(watched)
        return original(guard, watched, event)

    monkeypatch.setattr(SpinBoxHitAreaGuard, "eventFilter", observe)
    window = QWidget()
    other = QWidget()
    spin = spin_type(window)
    unrelated_editor = QLineEdit(window)
    unrelated_object = QObject()
    window.show()
    other.show()
    qapp.processEvents()
    try:
        editor = spin.lineEdit()
        guard = editor._spinbox_hit_area_guard
        assert guard.parent() is editor
        protect_spinbox_editor(spin)
        assert editor.findChildren(SpinBoxHitAreaGuard) == [guard]
        calls.clear()
        for target in (qapp, window, other, unrelated_editor, unrelated_object, spin):
            for _ in range(20):
                qapp.sendEvent(target, QEvent(QEvent.Type.User))
        assert calls == []
        qapp.sendEvent(editor, QEvent(QEvent.Type.User))
        assert calls == [editor]
        spin.hide()
        spin.setParent(other)
        spin.show()
        qapp.processEvents()
        assert editor._spinbox_hit_area_guard is guard
        assert all(watched is editor for watched in calls)
    finally:
        window.close()
        other.close()
        window.deleteLater()
        other.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("spin_type", [SpinBox, DoubleSpinBox])
def test_replacement_editor_owns_guard_and_cleans_up(styled_app, spin_type):
    spin = spin_type()
    spin.resize(180, 40)
    path = Path(__file__).parents[2] / "optiland_gui/resources/styles/dark_theme.qss"
    spin.setStyleSheet(path.read_text(encoding="utf-8"))
    spin.show()
    styled_app.processEvents()
    old_guard = spin.lineEdit()._spinbox_hit_area_guard
    old_destroyed = QSignalSpy(old_guard.destroyed)
    replacement = QLineEdit()
    try:
        spin.setLineEdit(replacement)
        styled_app.processEvents()
        assert old_destroyed.count() == 1
        guard = replacement._spinbox_hit_area_guard
        assert guard.parent() is replacement
        replacement.setGeometry(spin.rect())
        for control in (QStyle.SC_SpinBoxUp, QStyle.SC_SpinBoxDown):
            assert not replacement.geometry().intersects(button_rect(spin, control))
        destroyed = QSignalSpy(guard.destroyed)
        spin.close()
        spin.show()
        styled_app.processEvents()
        assert replacement._spinbox_hit_area_guard is guard
    finally:
        spin.close()
        spin.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert destroyed.count() == 1


def test_reparented_editor_is_not_treated_as_another_spinbox_editor(qapp):
    spin = SpinBox()
    other = NativeSpinBox()
    plain = QWidget()
    editor = spin.lineEdit()
    try:
        for parent in (plain, other):
            editor.setParent(parent)
            editor.show()
            editor.setGeometry(2, 2, 30, 20)
            qapp.processEvents()
            assert editor.geometry() == QRect(2, 2, 30, 20)
    finally:
        editor.setParent(spin)
        for widget in (spin, other, plain):
            widget.close()
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [(4, NativeSpinBox), (1.25, NativeDoubleSpinBox), ("text", QLineEdit)],
)
def test_native_delegate_spinboxes_get_editor_local_guards(qapp, value, expected_type):
    from optiland_gui.lens_editor import _AccentFocusDelegate

    table = QTableWidget()
    registered = []
    owner = SimpleNamespace(
        tableWidget=table,
        hover_presentation=SimpleNamespace(
            register_editor=lambda widget, index: registered.append(widget),
            unregister_editor=lambda widget: registered.remove(widget),
        ),
    )
    delegate = _AccentFocusDelegate(owner)
    model = QStandardItemModel(1, 1)
    index = model.index(0, 0)
    model.setData(index, value, Qt.EditRole)
    try:
        editor = delegate.createEditor(table, QStyleOptionViewItem(), index)
        assert isinstance(editor, expected_type)
        assert registered == [editor]
        if isinstance(editor, QAbstractSpinBox):
            line_edit = editor.lineEdit()
            assert line_edit._spinbox_hit_area_guard.parent() is line_edit
        else:
            assert not hasattr(editor, "_spinbox_hit_area_guard")
        delegate.destroyEditor(editor, index)
        assert registered == []
    finally:
        table.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize(
    "dialog_kind", ["variable", "operand", "automatic", "optimizer"]
)
def test_optimization_numeric_controls_are_protected(
    styled_app, highlighting_connector, dialog_kind
):
    from optiland_gui.optimization_panel import (
        AddOperandDialog,
        AddVariableDialog,
        AutoGenerateOperandsDialog,
        OptimizationPanel,
    )

    classes = {
        "variable": AddVariableDialog,
        "operand": AddOperandDialog,
        "automatic": AutoGenerateOperandsDialog,
        "optimizer": OptimizationPanel,
    }
    window = classes[dialog_kind](highlighting_connector)
    dynamic = []
    try:
        window.show()
        styled_app.processEvents()
        if dialog_kind != "automatic":
            create = (
                window._create_optimizer_widget
                if dialog_kind == "optimizer"
                else window._create_widget
            )
            for kind, value in (("int", 4), ("float", 1.25)):
                widget = create("sample", {"type": kind, "default": value})
                dynamic.append(widget)
                assert widget.value() == value
            if dialog_kind == "operand":
                widget = create("coords", {"type": "tuple", "default": [0.25, 0.5]})
                dynamic.append(widget)
                assert widget.get_value() == [0.25, 0.5]
        controls = window.findChildren(QAbstractSpinBox)
        for widget in dynamic:
            controls.extend(
                [widget]
                if isinstance(widget, QAbstractSpinBox)
                else widget.findChildren(QAbstractSpinBox)
            )
        assert controls
        for spin in controls:
            editor = spin.lineEdit()
            assert editor._spinbox_hit_area_guard.parent() is editor
            # Dynamic settings and dialog controls must also correct later
            # geometry changes, independently of MainWindow's lifetime.
            # Inactive tabs and tuple containers also need to be visible:
            # Qt defers hidden widgets' geometry notifications until show.
            ancestor = spin
            while ancestor is not None:
                ancestor.show()
                ancestor = ancestor.parentWidget()
            styled_app.processEvents()
            editor.setGeometry(spin.rect())
            for control in (QStyle.SC_SpinBoxUp, QStyle.SC_SpinBoxDown):
                assert not editor.geometry().intersects(button_rect(spin, control))
    finally:
        for widget in [*dynamic, window]:
            widget.close()
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
