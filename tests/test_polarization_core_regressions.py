"""Behavior regressions using native APIs that already exist on master."""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coatings import PolarizerCoating, SimpleCoating
from optiland.optic import Optic
from optiland.rays import PolarizedRays, create_polarization
from tests.utils import assert_allclose


def axial_ray() -> PolarizedRays:
    """Unit-power ray with a fixed global source basis."""
    return PolarizedRays(*[be.array([v]) for v in (0, 0, 0, 0, 0, 1, 1, 0.55)])


@pytest.mark.parametrize("angle", [0.0, math.pi / 4, -math.pi / 4])
def test_rotated_native_polarizer_obeys_malus_law(angle, set_test_backend):
    """Local X rotates with the surface; it is not always global horizontal."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=1)
    optic.surfaces.add(index=1, rz=angle, coating=PolarizerCoating())
    rays = axial_ray()
    optic.surfaces[1].trace(rays)
    state = create_polarization("H")
    cosine, sine = math.cos(angle), math.sin(angle)
    assert_allclose(rays.get_exit_fields(state)[0][0], [cosine**2, cosine * sine, 0])
    rays.update_intensity(state)
    assert_allclose(rays.i, [cosine**2])
    assert_allclose([rays.L, rays.M, rays.N], [0, 0, 1])


@pytest.mark.parametrize("operation,factor", [("transmit", 0.36), ("reflect", 0.25)])
def test_scalar_coating_transports_field_after_geometric_bend(
    operation, factor, set_test_backend
):
    """Scalar loss changes power, but must still transport the field direction."""
    theta = 0.35
    rays = PolarizedRays(
        *[
            be.array([v])
            for v in (0, 0, 0, math.sin(theta), 0, math.cos(theta), 1, 0.55)
        ]
    )
    nx, ny, nz = be.zeros(1), be.zeros(1), be.ones(1)
    if operation == "reflect":
        rays.reflect(nx, ny, nz)
        beta = math.pi - theta
    else:
        rays.refract(nx, ny, nz, be.ones(1), be.full(1, 1.5))
        beta = math.asin(math.sin(theta) / 1.5)
    SimpleCoating(0.36, 0.25).interact(
        rays, reflect=operation == "reflect", nx=nx, ny=ny, nz=nz
    )
    state = create_polarization("H")
    expected = np.sqrt(factor) * np.array([math.cos(beta), 0, -math.sin(beta)])
    field = rays.get_exit_fields(state)[0]
    assert_allclose(field[0], expected)
    direction = be.stack([rays.L, rays.M, rays.N], axis=1)
    assert_allclose(be.sum(field * direction, axis=1), [0], atol=1e-14)
    rays.update_intensity(state)
    assert_allclose(rays.i, [factor])


@pytest.mark.parametrize("axis,state", [("x", "V"), ("y", "H"), ("z", "H")])
def test_rotating_ray_keeps_field_transverse_and_power_fixed(
    axis, state, set_test_backend
):
    rays = axial_ray()
    state = create_polarization(state)
    getattr(rays, f"rotate_{axis}")(0.4)
    field = rays.get_exit_fields(state)[0]
    direction = be.stack([rays.L, rays.M, rays.N], axis=1)
    assert_allclose(be.sum(field * direction, axis=1), [0], atol=1e-14)
    assert_allclose(be.sum(be.abs(field) ** 2, axis=1), [1])
