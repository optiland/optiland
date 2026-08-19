from __future__ import annotations

from unittest.mock import Mock

import pytest

from optiland import backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.plane import Plane
from optiland.interactions.phase_interaction_model import PhaseInteractionModel
from optiland.materials.ideal import IdealMaterial
from optiland.phase.radial import RadialPhaseProfile
from optiland.rays.paraxial_rays import ParaxialRays
from optiland.rays.real_rays import RealRays
from optiland.surfaces.standard_surface import Surface

from .utils import assert_allclose


@pytest.fixture
def mock_surface(set_test_backend):
    # Create a mock surface with the necessary attributes
    surface = Mock(spec=Surface)
    surface.geometry = Plane(CoordinateSystem())
    surface.material_pre = IdealMaterial(n=1.0)
    surface.material_post = IdealMaterial(n=1.5)
    # Mock methods that might be called
    surface.geometry.surface_normal = Mock(
        return_value=(be.zeros(1), be.zeros(1), be.ones(1))
    )
    return surface


def test_phase_interaction_model_init(mock_surface):
    phase_profile = RadialPhaseProfile(coefficients=[0.1])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)
    assert model.phase_profile == phase_profile
    assert model.parent_surface == mock_surface


def test_interact_real_rays_transmission(mock_surface):
    # Metalens focusing at f=100mm
    # phi = -k0/(2f) * r^2
    f = 100.0
    w = 0.5
    k0 = 2 * be.pi / (w * 1e-3)  # w is in um, convert to mm for k0
    coeff = -k0 / (2 * f)
    phase_profile = RadialPhaseProfile(coefficients=[coeff])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)

    rays = RealRays(
        x=be.array([1.0]),
        y=be.array([0.0]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        wavelength=w,
        intensity=be.array([1.0]),
    )

    # Expected outgoing angle: L = -x / (n2 * f)
    expected_L = -1.0 / (mock_surface.material_post.n(w) * f)

    interacted_rays = model.interact_real_rays(rays)

    assert_allclose(interacted_rays.L, be.array([expected_L]), atol=1e-6)
    assert_allclose(interacted_rays.M, be.array([0.0]), atol=1e-6)


def test_interact_real_rays_reflection(mock_surface):
    f = 100.0
    w = 0.5
    k0 = 2 * be.pi / (w * 1e-3)
    coeff = -k0 / (2 * f)  # Same phase profile, different interaction
    phase_profile = RadialPhaseProfile(coefficients=[coeff])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=True)

    rays = RealRays(
        x=be.array([1.0]),
        y=be.array([0.0]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        wavelength=w,
        intensity=be.array([1.0]),
    )

    # Expected outgoing angle for reflection is more complex, but N should be negative
    interacted_rays = model.interact_real_rays(rays)
    assert be.all(interacted_rays.N < 0)


def test_interact_paraxial_rays(mock_surface):
    f = 100.0
    w = 0.5
    k0 = 2 * be.pi / (w * 1e-3)
    coeff = -k0 / (2 * f)
    phase_profile = RadialPhaseProfile(coefficients=[coeff])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)

    rays = ParaxialRays(
        y=be.array([1.0]), u=be.array([0.0]), z=be.array([0.0]), wavelength=w
    )

    # Expected paraxial angle: u_out = u_in - y/f (for n2=1)
    # With n2 != 1, u_out = u_in/n2 - y/(n2*f)
    # The sign convention was changed to match the diffractive model.
    n2 = mock_surface.material_post.n(w)
    expected_u = 1.0 / (n2 * f)

    interacted_rays = model.interact_paraxial_rays(rays)

    assert_allclose(interacted_rays.u, be.array([expected_u]), atol=1e-6)


def test_tir_case(mock_surface):
    # Create a gradient so large it must cause TIR
    phase_profile = RadialPhaseProfile(coefficients=[1e6])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)

    rays = RealRays(
        x=be.array([1.0]),
        y=be.array([0.0]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        wavelength=0.5,
        intensity=be.array([1.0]),
    )

    interacted_rays = model.interact_real_rays(rays)
    # The ray should be clipped (intensity becomes zero)
    assert be.all(interacted_rays.i == 0)


def test_serialization(mock_surface):
    phase_profile = RadialPhaseProfile(coefficients=[0.1, 0.2])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)

    data = model.to_dict()
    assert data["type"] == "PhaseInteractionModel"
    assert data["phase_profile"]["phase_type"] == "radial"

    new_model = PhaseInteractionModel.from_dict(data, mock_surface)
    assert isinstance(new_model, PhaseInteractionModel)
    assert isinstance(new_model.phase_profile, RadialPhaseProfile)
    assert new_model.phase_profile.coefficients == [0.1, 0.2]


def test_unit_conversion_mm_microns(mock_surface):
    # Explicit bug reproduction scenario
    # focal_distance = 24 mm
    # wavelength = 0.521 um = 0.000521 mm
    # k0 = 2*pi / 0.000521 rad/mm ~ 12060
    # coeff = -k0 / (2 * 24) ~ -251
    # RadialPhaseProfile with this coeff should produce expected focus

    f = 24.0
    w = 0.521  # microns
    k0 = 2 * be.pi / (w * 1e-3)  # rad/mm
    coeff = -k0 / (2 * f)

    phase_profile = RadialPhaseProfile(coefficients=[coeff])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)

    # Ray at x=1.0 mm, parallel incoming
    rays = RealRays(
        x=be.array([1.0]),
        y=be.array([0.0]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        wavelength=w,
        intensity=be.array([1.0]),
    )

    # Expected L = -x / (n2 * f)
    n2 = mock_surface.material_post.n(w)  # 1.5
    expected_L = -1.0 / (n2 * f)  # -1.0 / 36.0 ~ -0.0277

    interacted_rays = model.interact_real_rays(rays)

    assert_allclose(interacted_rays.L, be.array([expected_L]), atol=1e-6)
