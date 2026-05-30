"""Material warnings module.

Defines custom warning classes for the Optiland materials system.

Kramer Harrison, 2025
"""

from __future__ import annotations


class OptilandMaterialWarning(UserWarning):
    """Emitted when a material lookup resolves via fuzzy match or has ambiguity."""
