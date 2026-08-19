"""Unit tests for the HomogeneousPropagation model."""

from __future__ import annotations

from optiland import backend as be
from optiland.materials.ideal import IdealMaterial
from optiland.propagation.homogeneous import HomogeneousPropagation
from optiland.rays.real_rays import RealRays

from ..utils import assert_allclose


def test_homogeneous_propagation_position_update(set_test_backend):
    """Verify that ray coordinates are updated correctly."""
    basic_rays = RealRays(
        x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1.0], wavelength=[0.5]
    )
    material = IdealMaterial(n=1.0)  # k=0 by default
    model = HomogeneousPropagation(material)
    t = 10.0

    model.propagate(basic_rays, t)

    assert_allclose(basic_rays.x, be.array([0.0]))
    assert_allclose(basic_rays.y, be.array([0.0]))
    assert_allclose(basic_rays.z, be.array([10.0]))


def test_homogeneous_propagation_no_attenuation_with_k0(set_test_backend):
    """Verify ray intensity is unchanged when k=0."""
    basic_rays = RealRays(
        x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1.0], wavelength=[0.5]
    )
    material = IdealMaterial(n=1.0, k=0.0)
    model = HomogeneousPropagation(material)
    t = 10.0

    initial_intensity = be.copy(basic_rays.i)
    model.propagate(basic_rays, t)

    assert_allclose(basic_rays.i, initial_intensity)


def test_homogeneous_propagation_attenuation_with_k_gt_0(set_test_backend):
    """Verify ray intensity is correctly attenuated when k > 0."""
    basic_rays = RealRays(
        x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1.0], wavelength=[0.5]
    )
    k_val = 0.1
    wavelength = be.to_numpy(basic_rays.w)[0]
    t = 10.0  # distance in mm

    material = IdealMaterial(n=1.0, k=k_val)
    model = HomogeneousPropagation(material)

    initial_intensity = be.to_numpy(basic_rays.i)[0]
    model.propagate(basic_rays, t)

    # Calculate expected intensity based on Beer-Lambert law
    # alpha = 4 * pi * k / lambda
    # I = I_0 * exp(-alpha * z)
    # distance t is in mm, wavelength w is in um. Convert t to um.
    alpha = 4 * be.pi * k_val / wavelength
    expected_intensity = initial_intensity * be.exp(-alpha * t * 1e3)

    assert_allclose(basic_rays.i, be.array([expected_intensity]))


def test_homogeneous_propagation_normalizes_rays(set_test_backend):
    """Verify that unnormalized rays are normalized after propagation."""
    basic_rays = RealRays(
        x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1.0], wavelength=[0.5]
    )
    material = IdealMaterial(n=1.0)
    model = HomogeneousPropagation(material)

    # Manually un-normalize the rays
    basic_rays.L = basic_rays.L * 2
    basic_rays.is_normalized = False

    model.propagate(basic_rays, t=10.0)

    assert basic_rays.is_normalized is True
    magnitude = be.sqrt(basic_rays.L**2 + basic_rays.M**2 + basic_rays.N**2)
    assert_allclose(magnitude, be.array([1.0]))
