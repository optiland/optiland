"""Pointer-monitor fallback maximization with real virtual Qt screens."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROBE = """
import json, sys
from PySide6.QtCore import QPoint, QRect, QSize
from PySide6.QtWidgets import QApplication, QLineEdit
from optiland_gui.widgets.frameless_window import FramelessWindow

app = QApplication([])
screens = app.screens()
assert len(screens) == 2
target = screens[int(sys.argv[1])]
origin = screens[1 - int(sys.argv[1])]
window = FramelessWindow()
editor = QLineEdit('unsaved optical notes', window)
window.setCentralWidget(editor)
window.setMinimumSize(300, 200)
window.setGeometry(QRect(origin.availableGeometry().center() - QPoint(379, 249),
                         QSize(760, 500)))
window.show()
app.processEvents()
original_size = window.size()
assert window.screen() is origin
bounds = target.availableGeometry()
pointer = QPoint(bounds.center().x(), bounds.top())
window._snap_fallback(pointer)
app.processEvents()
assert window.isMaximized()
assert window.screen() is target
assert window.geometry() == target.availableGeometry()
normal = window.normalGeometry()
assert normal.size() == original_size
assert target.availableGeometry().contains(normal.center())
window.showNormal()
app.processEvents()
assert window.geometry() == normal
assert window.screen() is target
assert window.centralWidget() is editor
assert editor.text() == 'unsaved optical notes'
window.close()
print(json.dumps({'target':target.name(), 'dpr':target.devicePixelRatio(),
                  'normal':normal.getRect(), 'content':'preserved'}))
"""


@pytest.mark.parametrize("target", [0, 1])
@pytest.mark.parametrize("right_dpr", [1.0, 1.5])
def test_top_fallback_maximizes_on_pointer_monitor_and_restores_content(
    tmp_path, target, right_dpr
):
    config = {
        "windowFrameMargins": False,
        "screens": [
            {
                "name": "Left",
                "x": -1200,
                "y": 40,
                "width": 1200,
                "height": 1000,
                "logicalDpi": 96,
                "logicalBaseDpi": 96,
                "dpr": 1.0,
            },
            {
                "name": "Right",
                "x": 0,
                "y": -80,
                "width": 1200,
                "height": 1000,
                "logicalDpi": 96,
                "logicalBaseDpi": 96,
                "dpr": right_dpr,
            },
        ],
    }
    (tmp_path / "screens.json").write_text(json.dumps(config), encoding="utf-8")
    environment = os.environ.copy()
    repo = str(Path(__file__).parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        [environment.get("PYTHONPATH", ""), repo]
    )
    # Relative config avoids Qt's platform-argument parsing of a Windows colon.
    environment["QT_QPA_PLATFORM"] = "offscreen:configfile=screens.json"
    # Screen DPRs are specified above; an inherited global scale multiplies them.
    environment["QT_SCALE_FACTOR"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(target)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    measured = json.loads(result.stdout)
    assert measured["target"] == ("Left" if target == 0 else "Right")
    assert measured["dpr"] == (1.0 if target == 0 else right_dpr)
    assert measured["content"] == "preserved"
