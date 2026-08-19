from __future__ import annotations

from unittest.mock import Mock

import pytest

from optiland import backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.plane import Plane
from optiland.interactions.phase_interaction_model import PhaseInteractionModel
from optiland.materials.ideal import IdealMaterial
from optiland.phase.linear_grating import LinearGratingPhaseProfile
from optiland.rays.real_rays import RealRays
from optiland.surfaces.standard_surface import Surface

from .utils import assert_allclose


@pytest.fixture
def mock_surface(set_test_backend):
    surface = Mock(spec=Surface)
    surface.geometry = Plane(coordinate_system=CoordinateSystem())
    surface.material_pre = IdealMaterial(n=1.0)
    surface.material_post = IdealMaterial(n=1.5)
    surface.geometry.surface_normal = Mock(
        return_value=(be.zeros(1), be.zeros(1), be.ones(1))
    )
    return surface


@pytest.mark.parametrize("order", [1, -1, 2])
def test_linear_grating_phase_profile(order):
    period = 0.5
    angle = be.pi / 4
    lg = LinearGratingPhaseProfile(period=period, angle=angle, order=order)
    x, y = be.array([1.0]), be.array([2.0])

    K = order * 2 * be.pi / period
    K_x = K * be.cos(angle)
    K_y = K * be.sin(angle)

    expected_phase = K_x * x + K_y * y
    phase = lg.get_phase(x, y)
    assert_allclose(phase, expected_phase)

    grad_x, grad_y, grad_z = lg.get_gradient(x, y)
    assert_allclose(grad_x, K_x)
    assert_allclose(grad_y, K_y)
    assert_allclose(grad_z, 0)


def test_linear_grating_to_from_dict():
    lg = LinearGratingPhaseProfile(period=0.5, angle=be.pi / 4, order=2, efficiency=0.8)
    data = lg.to_dict()
    new_lg = LinearGratingPhaseProfile.from_dict(data)
    assert lg.period == new_lg.period
    assert lg.angle == new_lg.angle
    assert lg.order == new_lg.order
    assert lg.efficiency == new_lg.efficiency


def test_linear_grating_efficiency(mock_surface):
    efficiency = 0.5
    phase_profile = LinearGratingPhaseProfile(period=1.0, efficiency=efficiency)
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)

    initial_intensity = be.array([0.8])
    rays = RealRays(
        x=be.array([0.0]),
        y=be.array([0.0]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        wavelength=0.5e-3,
        intensity=initial_intensity,
    )

    interacted_rays = model.interact_real_rays(rays)

    expected_intensity = initial_intensity * efficiency
    assert_allclose(interacted_rays.i, expected_intensity)


@pytest.mark.parametrize("order", [1, -1])
def test_linear_grating_transmission(mock_surface, order):
    period = 1.0  # mm
    wavelength = 0.5  # microns (0.0005 mm)
    angle = 0.0

    phase_profile = LinearGratingPhaseProfile(period=period, angle=angle, order=order)
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)

    rays = RealRays(
        x=be.array([0.0]),
        y=be.array([0.0]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        wavelength=wavelength,
        intensity=be.array([1.0]),
    )

    interacted_rays = model.interact_real_rays(rays)

    # Grating equation: n2 * sin(theta_out) - n1 * sin(theta_in) = m * lambda / d
    # For normal incidence, theta_in = 0.
    # n2 * sin(theta_out) = order * lambda / period
    # L_out = sin(theta_out)
    n2 = mock_surface.material_post.n(wavelength)
    expected_L = order * (wavelength * 1e-3) / (period * n2)

    assert_allclose(interacted_rays.L, be.array([expected_L]), atol=1e-9)


@pytest.mark.parametrize("order", [1, -1])
def test_linear_grating_reflection(mock_surface, order):
    period = 1.0
    wavelength = 0.5  # microns
    angle = be.pi / 2

    phase_profile = LinearGratingPhaseProfile(period=period, angle=angle, order=order)
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=True)

    rays = RealRays(
        x=be.array([0.0]),
        y=be.array([0.0]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        wavelength=wavelength,
        intensity=be.array([1.0]),
    )

    interacted_rays = model.interact_real_rays(rays)

    # Grating equation for reflection: sin(theta_out) - sin(theta_in) = m * lambda / d
    # n1=n2=1, but handled by the model. The model uses the vector grating equation.
    # We expect the ray to be deflected in the M direction.
    n1 = mock_surface.material_pre.n(wavelength)
    expected_M = order * (wavelength * 1e-3) / (period * n1)

    assert_allclose(interacted_rays.M, be.array([expected_M]), atol=1e-9)
    assert be.all(interacted_rays.N < 0)
