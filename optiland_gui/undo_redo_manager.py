"""Provides a manager for handling undo and redo functionality.

This module defines :class:`UndoRedoManager`, which maintains stacks of
serialised optic states to enable undoing and redoing design changes within
the application.

Authors:
    Kramer Harrison, 2025
    Manuel Mendes, 2025
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class UndoRedoManager(QObject):
    """Manages undo and redo stacks for application state changes.

    Provides a simple stack-based mechanism for storing snapshots of an
    object (e.g. an optical system) before a change occurs.  The two
    signals below allow the UI to enable/disable undo and redo actions
    in response to stack changes.

    Signals:
        undoStackAvailabilityChanged (bool): Emitted when the availability
            of undo operations changes.
        redoStackAvailabilityChanged (bool): Emitted when the availability
            of redo operations changes.

    Args:
        parent: Optional parent :class:`~PySide6.QtCore.QObject`.
    """

    undoStackAvailabilityChanged = Signal(bool)
    redoStackAvailabilityChanged = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._undo_stack: list = []
        self._redo_stack: list = []

    def add_state(self, state_data: object) -> None:
        """Push *state_data* onto the undo stack.

        Call this with the state of the object **before** a change is made.
        Any existing redo history is discarded.

        Args:
            state_data: The state to save (must be serialisable via
                :meth:`~optiland.optic.Optic.to_dict`).
        """
        self._undo_stack.append(state_data)
        self._redo_stack.clear()
        self.undoStackAvailabilityChanged.emit(self.can_undo())
        self.redoStackAvailabilityChanged.emit(self.can_redo())
        logger.debug(
            "State added. Undo stack: %d, Redo stack: %d",
            len(self._undo_stack),
            len(self._redo_stack),
        )

    def undo(self, current_state_for_redo: object) -> object | None:
        """Perform an undo and return the state to restore.

        The caller's current state is pushed onto the redo stack before the
        previous state is returned.

        Args:
            current_state_for_redo: The current optic state *before* the
                undo is applied.  This will be added to the redo stack.

        Returns:
            The state to restore, or ``None`` if the undo stack is empty.
        """
        if not self.can_undo():
            return None

        restored_state = self._undo_stack.pop()
        self._redo_stack.append(current_state_for_redo)
        self.undoStackAvailabilityChanged.emit(self.can_undo())
        self.redoStackAvailabilityChanged.emit(self.can_redo())
        logger.debug(
            "Undo. Undo stack: %d, Redo stack: %d",
            len(self._undo_stack),
            len(self._redo_stack),
        )
        return restored_state

    def redo(self, current_state_for_undo: object) -> object | None:
        """Perform a redo and return the state to restore.

        The caller's current state is pushed onto the undo stack before the
        next state is returned.

        Args:
            current_state_for_undo: The current optic state *before* the
                redo is applied.  This will be added to the undo stack.

        Returns:
            The state to restore, or ``None`` if the redo stack is empty.
        """
        if not self.can_redo():
            return None

        restored_state = self._redo_stack.pop()
        self._undo_stack.append(current_state_for_undo)
        self.undoStackAvailabilityChanged.emit(self.can_undo())
        self.redoStackAvailabilityChanged.emit(self.can_redo())
        logger.debug(
            "Redo. Undo stack: %d, Redo stack: %d",
            len(self._undo_stack),
            len(self._redo_stack),
        )
        return restored_state

    def can_undo(self) -> bool:
        """Return ``True`` if there are states available to undo."""
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        """Return ``True`` if there are states available to redo."""
        return bool(self._redo_stack)

    def clear_stacks(self) -> None:
        """Clear both the undo and redo stacks and emit availability signals."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.undoStackAvailabilityChanged.emit(self.can_undo())
        self.redoStackAvailabilityChanged.emit(self.can_redo())
