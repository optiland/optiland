"""Ray Bundle Module

This module provides a simple container for ray bundle data.

Kramer Harrison, 2025
"""

from __future__ import annotations


class RayBundle:
    """A simple container for ray bundle data."""

    def __init__(self, x, y, z, field, bundle_id=None):
        self.x = x
        self.y = y
        self.z = z
        self.field = field
        self.bundle_id = bundle_id
