"""Surfaces, points and helpers of the tests of the Forbes evaluation."""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.forbes import (
    ForbesQ2dGeometry,
    ForbesQNormalSlopeGeometry,
    ForbesSurfaceConfig,
)

NORM = 8.0
QBFS = {0: 1.0e-3, 1: -5.0e-4, 2: 2.0e-4, 3: -8.0e-5}
Q2D = {
    ("a", 0, 0): 1.0e-3,
    ("a", 0, 1): -4.0e-4,
    ("a", 1, 0): 2.0e-4,
    ("a", 1, 1): 1.0e-4,
    ("b", 1, 0): 3.0e-5,
    ("a", 2, 0): 1.0e-4,
    ("b", 2, 1): -5.0e-5,
    ("a", 3, 0): 4.0e-5,
}
# The vertex, points inside the normalization radius, the rim u = 1 (NORM is
# a power of two, so r**2 / NORM**2 is exactly 1) and points outside.
X = [0.0, 3.0, -2.0, 0.5, 1.0e-7, 8.0, 6.5, 9.0]
Y = [0.0, 1.0, 4.0, -0.2, -2.0e-7, 0.0, -3.0, 1.0]


def _geometry(terms, radius=-40.0, conic=-0.5):
    """A Q or Q-2D geometry (from the type of the keys) with these terms."""
    if isinstance(next(iter(terms)), tuple):
        cls = ForbesQ2dGeometry
    else:
        cls = ForbesQNormalSlopeGeometry
    config = ForbesSurfaceConfig(
        radius=radius, conic=conic, norm_radius=NORM, terms=dict(terms)
    )
    return cls(CoordinateSystem(), config)


def _bits(value):
    return np.asarray(be.to_numpy(value), dtype=np.float64).view(np.int64)


def _slopes(geometry, x, y):
    """The slopes d(sag)/dx and d(sag)/dy from the unit normal of the trace."""
    nx, ny, nz = geometry._surface_normal(x, y)
    return -nx / nz, -ny / nz


def _assert_bits(value, expected):
    np.testing.assert_array_equal(_bits(value), _bits(expected))
