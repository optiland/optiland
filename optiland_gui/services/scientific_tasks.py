"""Owned scientific execution with bounded text and declarative GUI requests."""

from __future__ import annotations

import io
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout

from optiland_gui.services.job_records import OpticSnapshot, check_cancelled
from optiland_gui.services.model_initialization import initialize_loaded_optic

OUTPUT_BYTES = 64 * 1024
OUTPUT_LINES = 1000
LINE_CHARACTERS = 4096
SCRIPT_BYTES = 1024 * 1024
PANEL_NAMES = frozenset({"viewer", "analysis", "lens_editor"})


class BoundedOutput(io.TextIOBase):
    """Bound stored bytes, rendered lines and each line before Qt sees output."""

    def __init__(self, progress=None):
        super().__init__()
        self.text = ""
        self.truncated = False
        self._bytes = self._lines = self._column = 0
        self._progress = progress
        self._last_report = 0.0

    def write(self, text):
        original_length = len(text)
        if self.truncated:
            return original_length
        # Limit temporary encoding and iteration even for one enormous write.
        text = text[: OUTPUT_BYTES + 1]
        accepted = []
        for char in text:
            size = len(char.encode("utf-8", errors="replace"))
            if (
                self._bytes + size > OUTPUT_BYTES
                or self._lines >= OUTPUT_LINES
                or (char != "\n" and self._column >= LINE_CHARACTERS)
            ):
                self.truncated = True
                break
            accepted.append(char)
            self._bytes += size
            if char == "\n":
                self._lines += 1
                self._column = 0
            else:
                self._column += 1
        self.text += "".join(accepted)
        self.truncated |= len(accepted) < original_length
        now = time.monotonic()
        if self._progress and now - self._last_report >= 0.1:
            self._progress("Running scientific script", details={"output": self.value})
            self._last_report = now
        return original_length

    @property
    def value(self):
        return self.text + ("\n[Output truncated]\n" if self.truncated else "")


class GuiCommands:
    """Record a finite command vocabulary; no Qt object or callback is exposed."""

    def __init__(self):
        self.commands = []

    def _append(self, command):
        if len(self.commands) >= 32:
            raise ValueError("At most 32 GUI commands are allowed per scientific run.")
        self.commands.append(command)

    def show_panel(self, panel):
        if panel not in PANEL_NAMES:
            raise ValueError(f"Choose one of: {', '.join(sorted(PANEL_NAMES))}.")
        self._append(("show_panel", panel))

    def refresh_views(self):
        self._append(("refresh_views",))


def execute_scientific(snapshot, parameters, progress, cancelled):
    """Execute trusted user Python in the isolated scientific calculation process."""
    import numpy as np

    import optiland
    import optiland.backend as be

    code = parameters["code"]
    if len(code.encode("utf-8")) > SCRIPT_BYTES:
        raise ValueError("Scientific scripts are limited to 1 MiB of source text.")
    output = BoundedOutput(progress)
    commands = GuiCommands()
    error = ""
    candidate = None
    with redirect_stdout(output), redirect_stderr(output):
        try:
            optic = snapshot.restore()
            namespace = {
                "__name__": "__scientific_script__",
                "optic": optic,
                "optiland": optiland,
                "np": np,
                "be": be,
                "gui": commands,
            }
            progress("Running scientific script")
            exec(compile(code, "<scientific-script>", "exec"), namespace)
            check_cancelled(cancelled)
            progress("Preparing script result")
            initialize_loaded_optic(namespace["optic"])
            candidate = OpticSnapshot.capture(namespace["optic"])
        except BaseException as exc:
            check_cancelled(cancelled)
            error_output = BoundedOutput()
            for part in traceback.TracebackException.from_exception(exc).format():
                error_output.write(part)
            error = error_output.value
    return {
        "output": output.value,
        "error": error,
        "snapshot": candidate,
        "commands": tuple(commands.commands) if not error else (),
    }
