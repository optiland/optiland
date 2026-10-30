"""Actual file workers with deterministic stalls around transaction boundaries."""

from __future__ import annotations

import sys
import time

from optiland_gui.services import file_tasks
from optiland_gui.services.calculation_worker import main

if __name__ == "__main__":
    handler, when, seconds = sys.argv[1:]
    original = getattr(file_tasks, handler)

    def delayed(snapshot, parameters, progress, cancelled):
        progress("Fixture entered " + handler)
        if when == "before":
            time.sleep(float(seconds))
        result = original(snapshot, parameters, progress, cancelled)
        progress("Fixture completed " + handler)
        if when == "after":
            time.sleep(float(seconds))
        return result

    setattr(file_tasks, handler, delayed)
    main()
