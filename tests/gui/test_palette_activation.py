"""A physical palette activation must dispatch only one command."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid

from optiland_gui.widgets.command_palette import (
    CommandPaletteWidget,
    CommandRegistry,
    PaletteCommand,
)


@pytest.fixture
def palette(qapp, tmp_path):
    window = QWidget()
    window.resize(800, 600)
    registry = CommandRegistry()
    settings = QSettings(str(tmp_path / "palette.ini"), QSettings.IniFormat)
    widget = CommandPaletteWidget(window, registry, settings)
    window.show()
    yield widget, registry
    if isValid(window):
        window.close()
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def open_command(palette, callback):
    widget, registry = palette
    registry.register(PaletteCommand("Run", "Test command", callback))
    widget.open_palette()
    QTest.keyClicks(widget._search_input, "Run")
    return widget


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
@pytest.mark.parametrize("focus", ["search", "results", "palette"])
def test_one_key_dispatches_once_and_records_recency(palette, key, focus):
    calls = []
    widget = open_command(palette, lambda: calls.append("run"))
    target = {"search": widget._search_input,
              "results": widget._results_list, "palette": widget}[focus]
    target.setFocus()
    QTest.keyClick(target, key)
    assert calls == ["run"]
    assert widget._settings.value("commandPalette/recentCommands") == ["Run"]
    # A second notification while the close animation is running is stale.
    widget._results_list.itemActivated.emit(widget._results_list.currentItem())
    QTest.keyClick(target, key)
    assert calls == ["run"]


@pytest.mark.parametrize("focus", ["search", "results", "palette"])
def test_held_enter_does_not_execute(palette, focus):
    calls = []
    widget = open_command(palette, lambda: calls.append("run"))
    target = {"search": widget._search_input,
              "results": widget._results_list, "palette": widget}[focus]
    event = QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier, "\r", True)
    QCoreApplication.sendEvent(target, event)
    assert calls == []
    assert widget._visible
    QTest.keyClick(target, Qt.Key_Return)
    assert calls == ["run"]


def test_command_reopening_palette_does_not_receive_propagated_enter(palette):
    calls = []
    widget, _ = palette

    def reopen():
        calls.append("run")
        widget.open_palette()

    open_command(palette, reopen)
    QTest.keyClick(widget._search_input, Qt.Key_Return)
    assert calls == ["run"]
    assert widget._visible


def test_new_opening_allows_a_new_activation_and_deduplicates_recency(palette):
    calls = []
    widget = open_command(palette, lambda: calls.append("run"))
    QTest.keyClick(widget._search_input, Qt.Key_Return)
    QTest.qWait(200)
    widget.open_palette()
    # Empty query selects the recent command and skips the non-command heading.
    assert widget._results_list.currentRow() == 1
    QTest.keyClick(widget._search_input, Qt.Key_Up)
    assert widget._results_list.currentRow() == 1
    QTest.keyClick(widget._search_input, Qt.Key_Down)
    assert widget._results_list.currentRow() == 1
    QTest.keyClick(widget._search_input, Qt.Key_Return)
    assert calls == ["run", "run"]
    assert widget._settings.value("commandPalette/recentCommands") == ["Run"]


def test_reentrant_activation_is_ignored(palette):
    calls = []
    widget, _ = palette

    def execute_again():
        calls.append("run")
        widget._execute_selected()

    open_command(palette, execute_again)
    QTest.keyClick(widget._search_input, Qt.Key_Return)
    assert calls == ["run"]


def test_result_activation_signal_dispatches_once(palette):
    calls = []
    widget = open_command(palette, lambda: calls.append("run"))
    item = widget._results_list.currentItem()
    widget._results_list.itemActivated.emit(item)
    widget._results_list.itemActivated.emit(item)
    assert calls == ["run"]


def test_result_double_click_dispatches_once(palette):
    calls = []
    widget = open_command(palette, lambda: calls.append("run"))
    QTest.qWait(200)
    results = widget._results_list
    position = results.visualItemRect(results.currentItem()).center()
    QTest.mouseClick(results.viewport(), Qt.LeftButton, pos=position)
    QTest.mouseDClick(results.viewport(), Qt.LeftButton, pos=position)
    QTest.mouseRelease(results.viewport(), Qt.LeftButton, pos=position)
    assert calls == ["run"]


def test_unhandled_palette_key_remains_ignored(palette):
    calls = []
    widget = open_command(palette, lambda: calls.append("run"))
    event = QKeyEvent(QEvent.KeyPress, Qt.Key_F6, Qt.NoModifier)
    QCoreApplication.sendEvent(widget, event)
    assert not event.isAccepted()
    assert calls == []
    assert widget._visible


@pytest.mark.parametrize("focus", ["search", "results"])
def test_navigation_escape_and_non_activation_keys(palette, focus):
    calls = []
    widget, registry = palette
    for name in ("Run Alpha", "Run Beta", "Run Gamma"):
        registry.register(PaletteCommand(name, "", lambda: calls.append("run")))
    widget.open_palette()
    QTest.keyClicks(widget._search_input, "Run")
    target = widget._search_input if focus == "search" else widget._results_list
    target.setFocus()
    QTest.keyClick(target, Qt.Key_Down)
    assert widget._results_list.currentRow() == 1
    QTest.keyClick(target, Qt.Key_Up)
    assert widget._results_list.currentRow() == 0
    if focus == "search":
        QTest.keyClick(target, Qt.Key_Backspace)
        assert target.text() == "Ru"
    else:
        QTest.keyClick(target, Qt.Key_End)
        assert target.currentRow() == 2
        QTest.keyClick(target, Qt.Key_Home)
        assert target.currentRow() == 0
    QTest.keyClick(target, Qt.Key_Escape)
    assert not widget._visible
    QTest.keyClick(target, Qt.Key_Return)
    assert calls == []


def test_no_results_cannot_execute(palette):
    calls = []
    widget = open_command(palette, lambda: calls.append("run"))
    widget._search_input.setText("missing")
    QTest.keyClick(widget._search_input, Qt.Key_Return)
    widget._execute_item(widget._results_list.item(0))
    assert calls == []
    assert widget._visible


def test_failed_command_is_not_retried_by_same_key(palette, capsys):
    calls = []

    def fail():
        calls.append("run")
        raise ValueError("command failed")

    widget = open_command(palette, fail)
    QTest.keyClick(widget._search_input, Qt.Key_Return)
    assert calls == ["run"]
    assert "command failed" in capsys.readouterr().out


def test_command_can_close_owner(palette):
    widget, _ = palette
    window = widget.parent()
    open_command(palette, window.close)
    QTest.keyClick(widget._search_input, Qt.Key_Return)
    assert not window.isVisible()
