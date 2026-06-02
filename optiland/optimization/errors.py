"""Optimization Errors Module

Kramer Harrison, 2026
"""

from __future__ import annotations


class ConfigurationError(ValueError):
    """Raised when an incompatible optimizer / backend / strategy combination
    is requested before any iteration begins."""
