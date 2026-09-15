"""The materials package must expose a usable public import surface."""

from __future__ import annotations


def test_material_wildcard_import_exports_public_helpers():
    namespace = {}
    exec("from optiland.materials import *", namespace)
    assert callable(namespace["find_closest_glass"])
    assert callable(namespace["Material"])
