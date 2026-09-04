"""Did-you-mean suggestions for string-keyed lookups.

The implementation lives in :mod:`optiland._suggest` so that core modules
(materials, fields, aperture, ...) can use it without importing the
``optiland.diagnostics`` package, which would create a circular import.

Kramer Harrison, 2026
"""

from __future__ import annotations

from optiland._suggest import did_you_mean, options_hint

__all__ = ["did_you_mean", "options_hint"]
