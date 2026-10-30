"""Native optimizer cores (opt v2 M4). Private until the facade lands.

Kramer Harrison, 2026
"""

from __future__ import annotations

from .least_squares import JACOBIAN_UPDATES, dls

__all__ = ["JACOBIAN_UPDATES", "dls"]
