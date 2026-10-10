"""Real worker with a deliberately slow scalar operand for lifecycle tests."""

from __future__ import annotations

import time

from optiland.optimization.operand.operand import operand_registry


def delayed_value(optic, delay=0.5):
    time.sleep(delay)
    return float(optic.surfaces[1].thickness)


if __name__ == "__main__":
    from optiland_gui.services.calculation_worker import main

    operand_registry.register("test_delayed_value", delayed_value)
    main()
