from __future__ import annotations

import pytest

from optiland import backend as be
from optiland.materials.ideal import IdealMaterial
from optiland.phase.height_profile import HeightProfile

from .utils import assert_allclose


class DummySurface:
    def __init__(self, material_pre, material_post):
        self.material_pre = material_pre
        self.material_post = material_post


@pytest.fixture
def height_data():
    x = be.linspace(-1, 1, 25)
    y = be.linspace(-1, 1, 25)
    height_map = be.array([[i + j for i in x] for j in y])
    material_pre = IdealMaterial(n=1.0)
    material_post = IdealMaterial(n=1.5)
    return x, y, height_map, material_pre, material_post


@pytest.fixture
def height_profile(height_data):
    x, y, height_map, material_pre, material_post = height_data
    profile = HeightProfile(x, y, height_map)
    profile.parent_surface = DummySurface(
        material_pre=material_pre,
        material_post=material_post,
    )
    return profile


def test_height_profile_init(height_data):
    x, y, height_map, _, _ = height_data
    profile = HeightProfile(x, y, height_map)

    assert profile.x_coords.shape[0] == len(x)
    assert profile.y_coords.shape[0] == len(y)
    assert profile.height_map.shape == (len(y), len(x))


def test_height_profile_get_phase(height_profile):
    wavelength = be.array([1.0])
    px = be.array([0.0])
    py = be.array([0.0])

    phase = height_profile.get_phase(px, py, wavelength)

    assert phase.shape == (1,)
    assert isinstance(phase.item(), float)


def test_height_profile_phase_matches_height(height_data, height_profile):
    _, _, _, material_pre, material_post = height_data

    wavelength = be.array([1.0])
    px = be.array([0.5])
    py = be.array([0.25])

    h = height_profile._interpolate_height(px, py)
    phi = height_profile.get_phase(px, py, wavelength)

    n_pre = material_pre.n(wavelength)
    n_post = material_post.n(wavelength)
    factor = 2 * be.pi / (wavelength * 1e-3) * (n_post - n_pre)

    assert_allclose(phi, factor * h, atol=1e-6)


def test_height_profile_get_gradient(height_profile):
    wavelength = be.array([1.0])
    grad_x, grad_y, grad_z = height_profile.get_gradient(
        be.array([0.0]), be.array([0.0]), wavelength
    )

    assert grad_x.shape == grad_y.shape == grad_z.shape
    assert_allclose(grad_z, be.zeros_like(grad_z), atol=1e-6)


def test_height_profile_gradient_values(height_data, height_profile):
    _, _, _, material_pre, material_post = height_data

    wavelength = be.array([1.0])
    grad_x, grad_y, grad_z = height_profile.get_gradient(
        be.array([0.0]), be.array([0.0]), wavelength
    )

    n_pre = material_pre.n(wavelength)
    n_post = material_post.n(wavelength)
    factor = 2 * be.pi / (wavelength * 1e-3) * (n_post - n_pre)

    assert_allclose(grad_x, factor, atol=1e-6)
    assert_allclose(grad_y, factor, atol=1e-6)
    assert_allclose(grad_z, be.zeros_like(grad_z), atol=1e-6)


def test_height_profile_get_paraxial_gradient(height_profile):
    wavelength = be.array([1.0])
    y_vals = be.array([0.0, 0.5])

    paraxial = height_profile.get_paraxial_gradient(y_vals, wavelength)

    assert paraxial.shape[0] == 2


def test_height_profile_paraxial_value(height_data, height_profile):
    _, _, _, material_pre, material_post = height_data

    wavelength = be.array([1.0])
    y_vals = be.array([0.0, 0.5, 1.0])

    paraxial = height_profile.get_paraxial_gradient(y_vals, wavelength)

    n_pre = material_pre.n(wavelength)
    n_post = material_post.n(wavelength)
    factor = 2 * be.pi / (wavelength * 1e-3) * (n_post - n_pre)

    assert_allclose(paraxial, factor * be.ones_like(y_vals), atol=1e-6)


def test_height_profile_to_from_dict(height_data):
    x, y, height_map, _, _ = height_data
    profile = HeightProfile(x, y, height_map)

    data = profile.to_dict()

    assert data["phase_type"] == "height_profile"
    assert "x_coords" in data
    assert "y_coords" in data
    assert "height_map" in data

    new_profile = HeightProfile(
        x_coords=be.array(data["x_coords"]),
        y_coords=be.array(data["y_coords"]),
        height_map=be.array(data["height_map"]),
    )

    assert isinstance(new_profile, HeightProfile)
    assert_allclose(new_profile.x_coords, x, atol=1e-6)
    assert_allclose(new_profile.y_coords, y, atol=1e-6)
    assert_allclose(new_profile.height_map, height_map, atol=1e-6)