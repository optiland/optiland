"""Lose one old Save As acknowledgement after its destination was published."""

from __future__ import annotations

import time
from pathlib import Path

from optiland_gui.services import file_tasks
from optiland_gui.services.calculation_worker import main

if __name__ == "__main__":
    original = file_tasks.publish_output

    def delayed_ack(snapshot, parameters, progress, cancelled):
        result = original(snapshot, parameters, progress, cancelled)
        if Path(parameters["path"]).name == "older.json":
            progress("Published older Save As; delaying acknowledgement")
            time.sleep(10)
        return result

    file_tasks.publish_output = delayed_ack
    main()
