"""The fast path build gives bit-identical paths (issue #934).

With NumPy in float64, ``build_paraxial_path`` does the arithmetic of its
walk on host floats. These tests build each path a second time with the
backend arithmetic and require every field to be equal, bit for bit. The
systems cover straight, decentered, tilted, folded, retro, periscope and
off-axis-entry paths, so the diagnostics and advisories are compared too.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import optiland.backend as be
import optiland.paraxial_path as paraxial_path
from optiland.coordinate_system import CoordinateSystem
from optiland.samples.objectives import CookeTriplet, DoubleGauss

from . import test_folded_paraxial as folded
from . import test_folded_paraxial_hardening as hardening


def _tilted_triplet():
    optic = CookeTriplet()
    optic.surfaces[3].geometry.cs.rx = be.array(0.01)
    return optic


def _decentered_triplet():
    optic = CookeTriplet()
    optic.surfaces[4].geometry.cs.y = be.array(0.2)
    return optic


SYSTEMS = {
    "cooke": CookeTriplet,
    "double_gauss": DoubleGauss,
    "tilted": _tilted_triplet,
    "decentered": _decentered_triplet,
    "straight": folded.straight,
    "retro": folded.retro,
    "folded": folded.folded,
    "periscope": folded.periscope,
    "fold60": hardening.fold60,
    "periscope_out_of_plane": hardening.periscope_out_of_plane,
    "entered_along_neg_z": hardening.entered_along_neg_z,
    "entered_along_x_finite": hardening.entered_along_x_finite,
    "retro_finite": hardening.retro_finite,
    "folded_finite": hardening.folded_finite,
    "trombone_mirror": hardening.trombone_mirror,
    "trombone_lens": hardening.trombone_lens,
}


def _assert_identical(a, b, where="path"):
    if dataclasses.is_dataclass(a):
        assert type(a) is type(b), where
        for field in dataclasses.fields(a):
            _assert_identical(
                getattr(a, field.name), getattr(b, field.name), f"{where}.{field.name}"
            )
    elif isinstance(a, tuple | list):
        assert isinstance(b, tuple | list) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_identical(x, y, f"{where}[{i}]")
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), where
        assert a.dtype == b.dtype and a.shape == b.shape, where
        assert np.array_equal(a, b, equal_nan=True), where
    else:
        assert type(a) is type(b), where
        assert a == b or (a != a and b != b), f"{where}: {a!r} != {b!r}"


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_host_math_is_bit_identical(name, monkeypatch):
    be.set_backend("numpy")
    surfaces = list(SYSTEMS[name]().surfaces)
    assert paraxial_path._host_math_available()
    fast = paraxial_path.build_paraxial_path(surfaces)

    monkeypatch.setattr(paraxial_path, "_host_math_available", lambda: False)
    reference = paraxial_path.build_paraxial_path(surfaces)

    _assert_identical(fast, reference)


def test_systems_cover_diagnostics_and_advisories():
    be.set_backend("numpy")
    paths = [
        paraxial_path.build_paraxial_path(list(build().surfaces))
        for build in SYSTEMS.values()
    ]
    assert any(path.advisories for path in paths)
    assert any(path.is_folded_or_off_axis for path in paths)
    assert any(not path.positions_are_global_z for path in paths)


def test_rotation_free_frame_matches_backend_calls():
    be.set_backend("numpy")
    parent = CoordinateSystem(x=1.0, y=-2.0, z=3.0)
    cs = CoordinateSystem(x=0.5, y=0.25, z=7.0, reference_cs=parent)
    (x, y, z), axis = cs.frame_in_gcs

    for value, expected in ((x, 1.5), (y, -1.75), (z, 10.0)):
        reference = be.atleast_1d(0.0 + expected)
        assert value.dtype == reference.dtype and value.shape == reference.shape
        assert np.array_equal(value, reference)
    zero = be.atleast_1d(be.array(0.0))
    one = be.atleast_1d(be.array(1.0))
    for value, reference in zip(axis, (zero, zero, one), strict=True):
        assert value.dtype == reference.dtype and value.shape == reference.shape
        assert np.array_equal(value, reference)
