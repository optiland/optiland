"""Cartesian Jones coatings have a native, format-independent contract."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import optiland.backend as be
from optiland.coatings import BaseCoating, CartesianJonesCoating
from optiland.rays import PolarizedRays, RealRays, create_polarization
from tests.utils import assert_allclose


@pytest.mark.parametrize("operation", ["reflect", "transmit"])
def test_cartesian_element_preserves_scalar_ray_state(operation, set_test_backend):
    rays = RealRays(*[be.asarray([v]) for v in (1, 2, 3, 0, 0, 1, 0.7, 0.55)])
    before = {
        name: be.to_numpy(getattr(rays, name)).copy()
        for name in ("x", "y", "z", "L", "M", "N", "i", "opd")
    }
    coating = CartesianJonesCoating(0.5, 0.2j, -0.2j, 0.25)
    assert getattr(coating, operation)(rays) is rays
    for name, value in before.items():
        assert_allclose(getattr(rays, name), value)


@pytest.mark.parametrize("operation", ["reflect", "transmit"])
def test_cartesian_element_follows_geometric_bend_and_native_reload(
    operation, set_test_backend
):
    theta = 0.35
    coating = CartesianJonesCoating(0.6j, -0.2j, -0.3, 0.4)
    restored = BaseCoating.from_dict(json.loads(json.dumps(coating.to_dict())))
    matrix = np.array([[0.6j, -0.2j, 0], [-0.3, 0.4, 0], [0, 0, 1]])
    beta = math.pi - theta if operation == "reflect" else math.asin(math.sin(theta) / 1.5)
    transport = Rotation.from_euler("y", beta - theta).as_matrix()
    for candidate in (coating, restored):
        rays = PolarizedRays(*[be.asarray([v]) for v in
                              (0, 0, 0, math.sin(theta), 0, math.cos(theta), 1, 0.55)])
        nx, ny, nz = be.zeros(1), be.zeros(1), be.ones(1)
        if operation == "reflect":
            rays.reflect(nx, ny, nz)
        else:
            rays.refract(nx, ny, nz, be.ones(1), be.full(1, 1.5))
        candidate.interact(rays, reflect=operation == "reflect", nx=nx, ny=ny, nz=nz)
        assert_allclose(rays.p[0], matrix @ transport)
        state = create_polarization("H")
        expected = matrix @ np.array([math.cos(beta), 0, -math.sin(beta)])
        assert_allclose(rays.get_exit_fields(state)[0][0], expected)
        rays.update_intensity(state)
        assert_allclose(rays.i, [np.vdot(expected, expected).real])


@pytest.mark.parametrize("entries", [[], [[1, 0]], [[1, 0]] * 5])
def test_native_coating_requires_four_entries(entries):
    with pytest.raises(ValueError, match="four"):
        CartesianJonesCoating.from_dict({"entries": entries})


@pytest.mark.parametrize("pair", [[], [1], [1, 0, 0]])
def test_native_coating_requires_real_imaginary_pairs(pair):
    with pytest.raises(ValueError, match="pairs"):
        CartesianJonesCoating.from_dict({"entries": [pair] * 4})


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("value", [complex(np.nan, 0), complex(np.inf, 0), complex(0, -np.inf)])
def test_native_coating_requires_finite_complex_entries(index, value):
    entries = [1, 0, 0, 1]
    entries[index] = value
    with pytest.raises(ValueError, match="finite"):
        CartesianJonesCoating(*entries)
