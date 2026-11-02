"""Exercise Qt singleton returns through normal interpreter shutdown."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("operation", ["void-return", "bool-signal"])
def test_qt_calls_and_interpreter_shutdown(operation):
    # In the affected PySide6 wheels, None/True lose references on each call.
    # Assertions in the main pytest process can pass before finalization aborts.
    # A subprocess observes the actual exit status and contains memory corruption.
    script = """
import sys
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget

class Emitter(QObject):
    changed = Signal(bool)

app = QApplication([])
widget = QWidget()
widget.resize(64, 64)
widget.show()
app.processEvents()
emitter = Emitter()
received = []
emitter.changed.connect(received.append)
for _ in range(20_000):
    if sys.argv[1] == "void-return":
        widget.repaint()
    else:
        emitter.changed.emit(True)
        assert received.pop() is True
widget.close()
app.processEvents()
print("Qt calls completed", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, operation],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Qt calls completed" in result.stdout
