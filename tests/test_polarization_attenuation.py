"""Scalar losses must survive polarization-state evaluation."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coatings import SimpleCoating
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.phase import LinearGratingPhaseProfile
from optiland.propagation.homogeneous import HomogeneousPropagation
from optiland.rays import PolarizationState, PolarizedRays
from tests.utils import assert_allclose


def bundle():
    return PolarizedRays(*[be.array([v, v]) for v in (0, 0, 0, 0, 0, 1, 2, 0.5)])


def assert_power_and_fields(rays, expected):
    # Re-evaluation must neither restore lost power nor compound attenuation.
    for state in (
        PolarizationState(),
        PolarizationState(is_polarized=True, Ex=1, Ey=0, phase_x=0, phase_y=0),
        PolarizationState(),
    ):
        rays.update_intensity(state)
        assert_allclose(rays.i, expected)
        fields = rays.get_exit_fields(state)
        power = sum(be.sum(be.abs(field) ** 2, axis=1) for field in fields)
        assert_allclose(power, expected)


def test_clipped_polarized_rays_remain_dark(set_test_backend):
    rays = bundle()
    rays.clip(be.array([True, False]))
    assert_power_and_fields(rays, [0, 2])


def test_bulk_absorption_survives_polarization_evaluation(set_test_backend):
    rays = bundle()
    k, distance = 1e-6, 2.0
    propagation = HomogeneousPropagation(IdealMaterial(1.5, k))
    propagation.propagate(rays, distance)
    expected = 2 * np.exp(-4 * np.pi * k / 0.5 * distance * 1000)
    assert_power_and_fields(rays, [expected, expected])
    propagation.propagate(rays, distance)
    assert_power_and_fields(rays, [expected**2 / 2, expected**2 / 2])


@pytest.mark.parametrize("interaction,factor", [("transmit", 0.3), ("reflect", 0.2)])
def test_scalar_coating_and_clipping_compose(interaction, factor, set_test_backend):
    rays = bundle()
    getattr(SimpleCoating(0.3, 0.2), interaction)(rays)
    rays.clip(be.array([False, True]))
    assert_power_and_fields(rays, [2 * factor, 0])


def test_phase_efficiency_retains_polarized_power(set_test_backend):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=1)
    optic.surfaces.add(
        index=1, phase_profile=LinearGratingPhaseProfile(0.01, efficiency=0.6)
    )
    rays = bundle()
    optic.surfaces[1].trace(rays)
    assert_power_and_fields(rays, [1.2, 1.2])
