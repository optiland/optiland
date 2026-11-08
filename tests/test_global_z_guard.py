"""The cheap global-z predicate matches the full paraxial path (speed PR, item 3)."""

from __future__ import annotations

import pytest

from optiland.paraxial_path import (
    UnsupportedParaxialGeometryError,
    _positions_are_global_z,
    build_paraxial_path,
    require_global_z_geometry,
)

from .test_paraxial_path_host_math import SYSTEMS


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_predicate_matches_full_path(set_test_backend, name):
    optic = SYSTEMS[name]()
    expected = build_paraxial_path(list(optic.surfaces)).positions_are_global_z
    assert _positions_are_global_z(optic.surfaces) is expected


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_guard_raises_exactly_when_not_global_z(set_test_backend, name):
    optic = SYSTEMS[name]()
    if build_paraxial_path(list(optic.surfaces)).positions_are_global_z:
        require_global_z_geometry(optic.surfaces, "test")
    else:
        with pytest.raises(UnsupportedParaxialGeometryError):
            require_global_z_geometry(optic.surfaces, "test")


def test_systems_cover_both_outcomes(set_test_backend):
    outcomes = {
        _positions_are_global_z(factory().surfaces) for factory in SYSTEMS.values()
    }
    assert outcomes == {True, False}
    for name in ("folded", "fold60", "entered_along_neg_z"):
        assert _positions_are_global_z(SYSTEMS[name]().surfaces) is False
