"""GUI logging handler for the Optiland GUI.

Routes Python ``logging`` records at WARNING level and above to the
:class:`~optiland_gui.widgets.toast.ToastManager` so that backend warnings
are surfaced in the UI without coupling ``optiland/`` to GUI code.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal


class _LogSignalBridge(QObject):
    """Thread-safe bridge: emits a Qt signal from any thread."""

    record_received = Signal(int, str, str)  # levelno, message, name


_bridge = _LogSignalBridge()


class GuiLoggingHandler(logging.Handler):
    """A :class:`logging.Handler` that forwards records to :class:`ToastManager`.

    Instantiate once and pass a reference to the active
    :class:`~optiland_gui.widgets.toast.ToastManager`; the handler will
    call ``ToastManager.notify`` for every WARNING/ERROR/CRITICAL record.

    Args:
        toast_manager: The application's :class:`ToastManager` instance.
    """

    _LEVEL_TO_SEVERITY: dict[int, str] = {
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "error",
    }

    def __init__(self, toast_manager: object) -> None:
        super().__init__(level=logging.WARNING)
        self._toast_manager = toast_manager
        # Connect the bridge signal on the main thread so Qt updates are safe.
        _bridge.record_received.connect(self._on_record)

    def emit(self, record: logging.LogRecord) -> None:
        """Forward *record* to the toast manager via a Qt signal."""
        try:
            msg = self.format(record)
            _bridge.record_received.emit(record.levelno, msg, record.name)
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def _on_record(self, levelno: int, message: str, logger_name: str) -> None:
        severity = self._LEVEL_TO_SEVERITY.get(levelno, "warning")
        # Truncate very long messages
        display_msg = message if len(message) <= 120 else message[:117] + "…"
        self._toast_manager.notify(display_msg, severity, sub_message=logger_name)


def install(toast_manager: object, root_logger_name: str = "") -> GuiLoggingHandler:
    """Install a :class:`GuiLoggingHandler` on the named logger.

    Args:
        toast_manager: The active :class:`~optiland_gui.widgets.toast.ToastManager`.
        root_logger_name: Logger name to attach to (default: root logger).

    Returns:
        The installed handler (keep a reference to avoid garbage collection).
    """
    handler = GuiLoggingHandler(toast_manager)
    logging.getLogger(root_logger_name).addHandler(handler)
    return handler
