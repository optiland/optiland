"""Command palette for the Optiland GUI.

Provides a keyboard-driven (Ctrl+K) command launcher similar to VS Code's
command palette.  Three public components:

- :class:`PaletteCommand` — dataclass for a single command entry.
- :class:`CommandRegistry` — singleton holding all registered commands.
- :class:`CommandPaletteWidget` — the floating UI panel.

Design notes (from SPEC §2):
- Opens/closes with Ctrl+K or Escape.
- Fuzzy substring matching; highlights matched chars.
- Empty state shows 5 most-recently-used commands.
- Max 20 results; rank: exact prefix > fuzzy score > recency.
- Recency persisted in QSettings under ``commandPalette/recentCommands``.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPoint,
    QPropertyAnimation,
    QSettings,
    Qt,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

_MAX_RESULTS = 20
_MAX_RECENT = 5
_PALETTE_WIDTH = 560
_PALETTE_TOP_OFFSET = 80
_OPEN_DURATION = 180
_CLOSE_DURATION = 130
_SETTINGS_KEY = "commandPalette/recentCommands"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PaletteCommand:
    """A single entry in the command palette.

    Args:
        name: Human-readable display name shown in the palette.
        description: Short description (shown as sub-text).
        callback: Zero-argument callable executed when the command is selected.
        keywords: Extra search terms that can match this command.
        shortcut: Optional keyboard shortcut hint (e.g. ``"Ctrl+S"``).
        category: Grouping category label (e.g. ``"File"``, ``"Analysis"``).
    """

    name: str
    description: str
    callback: Callable[[], None]
    keywords: list[str] = field(default_factory=list)
    shortcut: str = ""
    category: str = "General"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class CommandRegistry:
    """Singleton that holds all registered :class:`PaletteCommand` objects.

    Usage::

        registry = CommandRegistry.instance()
        registry.register(PaletteCommand("Save", "Save file", save_fn,
                                          shortcut="Ctrl+S", category="File"))
    """

    _instance: CommandRegistry | None = None

    @classmethod
    def instance(cls) -> CommandRegistry:
        """Return (and create on first call) the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._commands: list[PaletteCommand] = []

    def register(self, command: PaletteCommand) -> None:
        """Add *command* to the registry.

        Duplicate names are silently replaced.

        Args:
            command: The :class:`PaletteCommand` to register.
        """
        self._commands = [c for c in self._commands if c.name != command.name]
        self._commands.append(command)

    def all_commands(self) -> list[PaletteCommand]:
        """Return a copy of all registered commands."""
        return list(self._commands)

    def find(self, name: str) -> PaletteCommand | None:
        """Return the command with *name*, or ``None``."""
        for c in self._commands:
            if c.name == name:
                return c
        return None


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    return unicodedata.normalize("NFKD", s).casefold()


def _fuzzy_score(query: str, target: str) -> int | None:
    """Return an integer score ≥ 0 if *query* matches *target*, else ``None``.

    Higher is better.  Exact prefix match returns a very high score.
    """
    q = _normalize(query)
    t = _normalize(target)
    if not q:
        return 0
    if t.startswith(q):
        return 1000 + len(t) - len(q)  # prefer shorter targets
    # Subsequence / substring match
    qi = 0
    score = 0
    for char in t:
        if qi < len(q) and char == q[qi]:
            qi += 1
            score += 1
    if qi == len(q):
        return score
    return None


def _search(
    query: str,
    commands: list[PaletteCommand],
    recent_names: list[str],
) -> list[PaletteCommand]:
    """Return up to :data:`_MAX_RESULTS` commands ranked by relevance."""
    if not query.strip():
        # Empty: show recents
        recent = [c for name in recent_names for c in commands if c.name == name]
        return recent[:_MAX_RECENT]

    scored: list[tuple[int, int, PaletteCommand]] = []
    for cmd in commands:
        searchable = " ".join([cmd.name, cmd.description] + cmd.keywords)
        score = _fuzzy_score(query, searchable)
        if score is None:
            continue
        recency = recent_names.index(cmd.name) if cmd.name in recent_names else 999
        scored.append((score, recency, cmd))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [cmd for _, _, cmd in scored[:_MAX_RESULTS]]


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


class _ResultItem(QListWidgetItem):
    """List item carrying a :class:`PaletteCommand` reference."""

    def __init__(self, command: PaletteCommand, query: str) -> None:
        super().__init__()
        self.command = command
        self._query = query
        self.setSizeHint(self.sizeHint().__class__(0, 48))

    @property
    def display_text(self) -> str:
        shortcut = f"  {self.command.shortcut}" if self.command.shortcut else ""
        return f"{self.command.name}{shortcut}"


class CommandPaletteWidget(QWidget):
    """The floating command palette panel.

    Instantiate once in ``MainWindow`` and call :meth:`toggle` to open/close.

    Args:
        parent_window: The main window.  The palette is reparented to it.
        registry: The :class:`CommandRegistry` to search.
        settings: A :class:`QSettings` instance for persisting recency.
    """

    def __init__(
        self,
        parent_window: QWidget,
        registry: CommandRegistry,
        settings: QSettings,
    ) -> None:
        super().__init__(parent_window)
        self._registry = registry
        self._settings = settings
        self._visible = False
        self._recent: list[str] = self._load_recent()

        self._build_overlay()
        self._build_palette()
        self._apply_shadow()
        self.hide()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        """Open if closed, close if open."""
        if self._visible:
            self.close_palette()
        else:
            self.open_palette()

    def open_palette(self) -> None:
        """Show and animate the palette into view."""
        if self._visible:
            return
        self._visible = True
        self._reposition()
        self._overlay.setGeometry(self.parent().rect())
        self._overlay.show()
        self._overlay.raise_()
        self.show()
        self.raise_()
        self._search_input.clear()
        self._search_input.setFocus()
        self._populate_results("")
        self._animate_open()

    def close_palette(self) -> None:
        """Animate the palette out and hide it."""
        if not self._visible:
            return
        self._visible = False
        self._animate_close()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_overlay(self) -> None:
        self._overlay = QWidget(self.parent())
        self._overlay.setObjectName("CommandPaletteOverlay")
        self._overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self._overlay.setStyleSheet("background: rgba(0,0,0,120);")
        self._overlay.hide()
        self._overlay.mousePressEvent = lambda e: self.close_palette()  # type: ignore[method-assign]

    def _build_palette(self) -> None:
        self.setObjectName("CommandPaletteWidget")
        self.setFixedWidth(_PALETTE_WIDTH)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            QWidget#CommandPaletteWidget {
                background-color: #252526;
                border-radius: 10px;
                border: 1px solid #3C3C3C;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(0)

        # Search input
        self._search_input = QLineEdit()
        self._search_input.setObjectName("PaletteSearchInput")
        self._search_input.setPlaceholderText("Search commands…")
        self._search_input.setStyleSheet(
            """
            QLineEdit#PaletteSearchInput {
                background: transparent;
                border: none;
                border-bottom: 1px solid #3C3C3C;
                color: #E0E0E0;
                font-size: 14px;
                padding: 12px 16px;
            }
            """
        )
        layout.addWidget(self._search_input)

        # Results list
        self._results_list = QListWidget()
        self._results_list.setObjectName("PaletteResultsList")
        self._results_list.setStyleSheet(
            """
            QListWidget#PaletteResultsList {
                background: transparent;
                border: none;
                outline: none;
            }
            QListWidget#PaletteResultsList::item {
                color: #E0E0E0;
                padding: 8px 16px;
                border-left: 3px solid transparent;
            }
            QListWidget#PaletteResultsList::item:hover,
            QListWidget#PaletteResultsList::item:selected {
                background: rgba(0,122,204,0.15);
                border-left: 3px solid #007ACC;
                color: #FFFFFF;
            }
            """
        )
        self._results_list.setMaximumHeight(360)
        layout.addWidget(self._results_list)

        # Connect signals
        self._search_input.textChanged.connect(self._on_query_changed)
        self._search_input.returnPressed.connect(self._execute_selected)
        self._results_list.itemActivated.connect(self._execute_item)
        self._results_list.installEventFilter(self)

    def _apply_shadow(self) -> None:
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, int(0.7 * 255)))
        self.setGraphicsEffect(shadow)

    # ------------------------------------------------------------------
    # Positioning and animation
    # ------------------------------------------------------------------

    def _reposition(self) -> None:
        pw = self.parent().width()
        x = (pw - _PALETTE_WIDTH) // 2
        y = _PALETTE_TOP_OFFSET
        self.move(x, y)
        self.adjustSize()

    def _animate_open(self) -> None:
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)

        current_pos = self.pos()
        start_pos = QPoint(current_pos.x(), current_pos.y() - 12)

        pos_anim = QPropertyAnimation(self, b"pos", self)
        pos_anim.setStartValue(start_pos)
        pos_anim.setEndValue(current_pos)
        pos_anim.setDuration(_OPEN_DURATION)
        pos_anim.setEasingCurve(QEasingCurve.OutCubic)

        opacity_anim = QPropertyAnimation(effect, b"opacity", self)
        opacity_anim.setStartValue(0.0)
        opacity_anim.setEndValue(1.0)
        opacity_anim.setDuration(_OPEN_DURATION)
        opacity_anim.setEasingCurve(QEasingCurve.OutCubic)

        group = QParallelAnimationGroup(self)
        group.addAnimation(pos_anim)
        group.addAnimation(opacity_anim)
        group.start(QParallelAnimationGroup.DeleteWhenStopped)

    def _animate_close(self) -> None:
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)

        current_pos = self.pos()
        end_pos = QPoint(current_pos.x(), current_pos.y() - 12)

        pos_anim = QPropertyAnimation(self, b"pos", self)
        pos_anim.setStartValue(current_pos)
        pos_anim.setEndValue(end_pos)
        pos_anim.setDuration(_CLOSE_DURATION)
        pos_anim.setEasingCurve(QEasingCurve.InCubic)

        opacity_anim = QPropertyAnimation(effect, b"opacity", self)
        opacity_anim.setStartValue(1.0)
        opacity_anim.setEndValue(0.0)
        opacity_anim.setDuration(_CLOSE_DURATION)
        opacity_anim.setEasingCurve(QEasingCurve.InCubic)

        def _finish() -> None:
            self.hide()
            self._overlay.hide()

        group = QParallelAnimationGroup(self)
        group.addAnimation(pos_anim)
        group.addAnimation(opacity_anim)
        group.finished.connect(_finish)
        group.start(QParallelAnimationGroup.DeleteWhenStopped)

    # ------------------------------------------------------------------
    # Search and display
    # ------------------------------------------------------------------

    def _on_query_changed(self, query: str) -> None:
        self._populate_results(query)

    def _populate_results(self, query: str) -> None:
        self._results_list.clear()
        results = _search(query, self._registry.all_commands(), self._recent)

        if not results:
            item = QListWidgetItem("No results found")
            item.setFlags(Qt.NoItemFlags)
            item.setForeground(QColor("#888888"))
            self._results_list.addItem(item)
            return

        # Group by category when empty query (show recents) or when > 5 results
        if not query.strip():
            header = QListWidgetItem("  RECENT")
            header.setFlags(Qt.NoItemFlags)
            header.setForeground(QColor("#888888"))
            self._results_list.addItem(header)

        for cmd in results:
            item = _ResultItem(cmd, query)
            shortcut_suffix = f"   {cmd.shortcut}" if cmd.shortcut else ""
            item.setText(f"{cmd.name}{shortcut_suffix}")
            item.setToolTip(cmd.description)
            self._results_list.addItem(item)

        if self._results_list.count() > 0:
            for i in range(self._results_list.count()):
                it = self._results_list.item(i)
                if it and it.flags() & Qt.ItemIsEnabled:
                    self._results_list.setCurrentItem(it)
                    break

        self.adjustSize()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_selected(self) -> None:
        item = self._results_list.currentItem()
        if item:
            self._execute_item(item)

    def _execute_item(self, item: QListWidgetItem) -> None:
        if not isinstance(item, _ResultItem):
            return
        self.close_palette()
        self._record_recent(item.command.name)
        try:
            item.command.callback()
        except Exception as exc:  # noqa: BLE001
            print(f"CommandPalette: error executing '{item.command.name}': {exc}")

    def _record_recent(self, name: str) -> None:
        if name in self._recent:
            self._recent.remove(name)
        self._recent.insert(0, name)
        self._recent = self._recent[:_MAX_RECENT]
        self._settings.setValue(_SETTINGS_KEY, self._recent)

    def _load_recent(self) -> list[str]:
        val = self._settings.value(_SETTINGS_KEY, [])
        if isinstance(val, list):
            return val[:_MAX_RECENT]
        return []

    # ------------------------------------------------------------------
    # Keyboard navigation
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        key = event.key()
        if key == Qt.Key_Escape:
            self.close_palette()
        elif key == Qt.Key_Down:
            self._move_selection(1)
        elif key == Qt.Key_Up:
            self._move_selection(-1)
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            self._execute_selected()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, source, event) -> bool:  # noqa: ANN001
        from PySide6.QtCore import QEvent

        if source is self._results_list and event.type() == QEvent.KeyPress:
            self.keyPressEvent(event)
            return True
        return super().eventFilter(source, event)

    def _move_selection(self, delta: int) -> None:
        count = self._results_list.count()
        if count == 0:
            return
        current = self._results_list.currentRow()
        # Skip non-selectable items
        new_row = current + delta
        while 0 <= new_row < count:
            item = self._results_list.item(new_row)
            if item and (item.flags() & Qt.ItemIsEnabled):
                self._results_list.setCurrentRow(new_row)
                return
            new_row += delta
