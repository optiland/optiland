from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.rays import (
    BaseRays,
    ParaxialRays,
    PolarizationState,
    PolarizedRays,
    RayGenerator,
    RealRays,
    create_polarization,
)
from optiland.samples.lithography import UVProjectionLens
from optiland.samples.objectives import TessarLens
from tests.utils import assert_allclose


class _ConcreteRays(BaseRays):
    def trace_on_surface(self, surface):
        pass

    def record_on_surface(self, surface):
        pass


def test_translate(set_test_backend):
    rays = _ConcreteRays()
    rays.x = 1.0
    rays.y = 2.0
    rays.z = 3.0

    rays.translate(0.5, -1.0, 2.5)

    assert_allclose(rays.x, 1.5)
    assert_allclose(rays.y, 1.0)
    assert_allclose(rays.z, 5.5)

    rays.translate(-1.5, 0.0, -5.5)

    assert_allclose(rays.x, 0.0)
    assert_allclose(rays.y, 1.0)
    assert_allclose(rays.z, 0.0)

    rays.translate(0.0, 0.0, 0.0)

    assert_allclose(rays.x, 0.0)
    assert_allclose(rays.y, 1.0)
    assert_allclose(rays.z, 0.0)

    rays.translate(2.0, -1.0, 3.0)

    assert_allclose(rays.x, 2.0)
    assert_allclose(rays.y, 0.0)
    assert_allclose(rays.z, 3.0)


def test__process_input(set_test_backend):
    # Test scalar input
    data = 1
    processed_data = be.as_array_1d(data)
    assert isinstance(processed_data, be.ndarray)
    assert processed_data.shape == (1,)
    assert processed_data.dtype == be.array(data).dtype
    assert_allclose(processed_data[0], 1.0)

    # Test float input
    data = 2.5
    processed_data = be.as_array_1d(data)
    assert isinstance(processed_data, be.ndarray)
    assert processed_data.shape == (1,)
    assert processed_data.dtype == be.array(data).dtype
    assert_allclose(processed_data[0], 2.5)

    # Test numpy array input
    data = be.array([3, 4, 5])
    processed_data = be.as_array_1d(data)
    assert isinstance(processed_data, be.ndarray)
    assert processed_data.shape == (3,)
    assert processed_data.dtype == be.array(data).dtype
    assert be.array_equal(processed_data, be.array([3.0, 4.0, 5.0]))

    # Test unsupported input type
    data = "invalid"
    with pytest.raises(ValueError):
        be.as_array_1d(data)


def test_real_rays_init(set_test_backend):
    x = 1
    y = 2
    z = 3
    L = 4
    M = 5
    N = 6
    intensity = 7
    wavelength = 8

    rays = RealRays(x, y, z, L, M, N, intensity, wavelength)

    assert isinstance(rays.x, be.ndarray)
    assert rays.x.shape == (1,)
    assert rays.x.dtype == be.array(1.0).dtype
    assert_allclose(rays.x[0], 1.0)

    assert isinstance(rays.y, be.ndarray)
    assert rays.y.shape == (1,)
    assert rays.y.dtype == be.array(1.0).dtype
    assert_allclose(rays.y[0], 2.0)

    assert isinstance(rays.z, be.ndarray)
    assert rays.z.shape == (1,)
    assert rays.z.dtype == be.array(1.0).dtype
    assert_allclose(rays.z[0], 3.0)

    assert isinstance(rays.L, be.ndarray)
    assert rays.L.shape == (1,)
    assert rays.L.dtype == be.array(1.0).dtype
    assert_allclose(rays.L[0], 4.0)

    assert isinstance(rays.M, be.ndarray)
    assert rays.M.shape == (1,)
    assert rays.M.dtype == be.array(1.0).dtype
    assert_allclose(rays.M[0], 5.0)

    assert isinstance(rays.N, be.ndarray)
    assert rays.N.shape == (1,)
    assert rays.N.dtype == be.array(1.0).dtype
    assert_allclose(rays.N[0], 6.0)

    assert isinstance(rays.i, be.ndarray)
    assert rays.i.shape == (1,)
    assert rays.i.dtype == be.array(1.0).dtype
    assert_allclose(rays.i[0], 7.0)

    assert isinstance(rays.w, be.ndarray)
    assert rays.w.shape == (1,)
    assert rays.w.dtype == be.array(1.0).dtype
    assert_allclose(rays.w[0], 8.0)

    assert isinstance(rays.opd, be.ndarray)
    assert rays.opd.shape == (1,)
    assert rays.opd.dtype == be.array(0.0).dtype
    assert_allclose(rays.opd[0], 0.0)


def test_rotate_x(set_test_backend):
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    rays.rotate_x(be.pi / 2)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], -3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], -1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 0.0, rtol=0, atol=1e-10)

    rays.rotate_x(-be.pi / 2)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.rotate_x(be.pi)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], -2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], -3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], -1.0, rtol=0, atol=1e-10)

    rays.rotate_x(0.0)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], -2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], -3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], -1.0, rtol=0, atol=1e-10)


def test_rotate_y(set_test_backend):
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    rays.rotate_y(be.pi / 2)

    assert_allclose(rays.x[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], -1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 0.0, rtol=0, atol=1e-10)

    rays.rotate_y(-be.pi / 2)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.rotate_y(be.pi)

    assert_allclose(rays.x[0], -1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], -3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], -1.0, rtol=0, atol=1e-10)

    rays.rotate_y(0.0)

    assert_allclose(rays.x[0], -1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], -3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], -1.0, rtol=0, atol=1e-10)


def test_rotate_z(set_test_backend):
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    rays.rotate_z(be.pi / 2)

    assert_allclose(rays.x[0], -2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.rotate_z(-be.pi / 2)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.rotate_z(be.pi)

    assert_allclose(rays.x[0], -1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], -2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.rotate_z(0.0)

    assert_allclose(rays.x[0], -1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], -2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)


def test_normalize(set_test_backend):
    """Test the normalization of ray direction cosines."""
    # Create rays with unnormalized direction cosines
    rays = RealRays(x=0, y=0, z=0, L=1, M=1, N=1, intensity=1, wavelength=0.5)
    rays.is_normalized = False

    rays.normalize()

    # Check that the flag is updated
    assert rays.is_normalized is True

    # Check that the direction cosines are correctly normalized
    expected_L = 1 / be.sqrt(3)
    expected_M = 1 / be.sqrt(3)
    expected_N = 1 / be.sqrt(3)
    assert_allclose(rays.L[0], expected_L, atol=1e-10)
    assert_allclose(rays.M[0], expected_M, atol=1e-10)
    assert_allclose(rays.N[0], expected_N, atol=1e-10)

    # Check that the magnitude is 1
    magnitude = be.sqrt(rays.L[0] ** 2 + rays.M[0] ** 2 + rays.N[0] ** 2)
    assert_allclose(magnitude, 1.0, atol=1e-10)


def test_clip(set_test_backend):
    # Test clipping with condition True
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    condition = True
    rays.clip(condition)
    assert_allclose(rays.i[0], 0.0, atol=1e-10)

    # Test clipping with condition False
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    condition = False
    rays.clip(condition)
    assert_allclose(rays.i[0], 1.0, atol=1e-10)


def test_paraxial_rays_init(set_test_backend):
    y = 1.0
    u = 0.1
    z = -10.0
    wavelength = 2.0

    rays = ParaxialRays(y, u, z, wavelength)

    assert isinstance(rays.x, be.ndarray)
    assert rays.x.shape == (1,)
    assert rays.x.dtype == be.array(0.0).dtype
    assert_allclose(rays.x[0], 0.0, atol=1e-10)

    assert isinstance(rays.y, be.ndarray)
    assert rays.y.shape == (1,)
    assert rays.y.dtype == be.array(y).dtype
    assert_allclose(rays.y[0], 1.0, atol=1e-10)

    assert isinstance(rays.z, be.ndarray)
    assert rays.z.shape == (1,)
    assert rays.z.dtype == be.array(z).dtype
    assert_allclose(rays.z[0], -10.0, atol=1e-10)

    assert isinstance(rays.u, be.ndarray)
    assert rays.u.shape == (1,)
    assert rays.u.dtype == be.array(u).dtype
    assert_allclose(rays.u[0], 0.1, atol=1e-10)

    assert isinstance(rays.i, be.ndarray)
    assert rays.i.shape == (1,)
    assert rays.i.dtype == be.array(1.0).dtype
    assert_allclose(rays.i[0], 1.0, atol=1e-10)

    assert isinstance(rays.w, be.ndarray)
    assert rays.w.shape == (1,)
    assert rays.w.dtype == be.array(wavelength).dtype
    assert_allclose(rays.w[0], 2.0, atol=1e-10)


def test_paraxial_propagate(set_test_backend):
    y = 1.0
    u = 0.1
    z = -10.0
    wavelength = 2.0

    rays = ParaxialRays(y, u, z, wavelength)

    rays.propagate(t=0.0)

    assert_allclose(rays.x[0], 0.0, atol=1e-10)
    assert_allclose(rays.y[0], 1.0, atol=1e-10)
    assert_allclose(rays.z[0], -10.0, atol=1e-10)
    assert_allclose(rays.u[0], 0.1, atol=1e-10)

    rays.propagate(t=5.0)

    assert_allclose(rays.x[0], 0.0, atol=1e-10)
    assert_allclose(rays.y[0], 1.5, atol=1e-10)
    assert_allclose(rays.z[0], -5.0, atol=1e-10)
    assert_allclose(rays.u[0], 0.1, atol=1e-10)

    rays.propagate(t=-5.0)

    assert_allclose(rays.x[0], 0.0, atol=1e-10)
    assert_allclose(rays.y[0], 1.0, atol=1e-10)
    assert_allclose(rays.z[0], -10.0, atol=1e-10)
    assert_allclose(rays.u[0], 0.1, atol=1e-10)


def test_reflect(set_test_backend):
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    # Reflect rays on a surface with normal (0, 0, 1)
    rays.reflect(0.0, 0.0, 1.0)

    assert_allclose(rays.L[0], 0.0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, atol=1e-10)
    assert_allclose(rays.N[0], -1.0, atol=1e-10)

    # Reflect rays on a surface with normal (1, 0, 0)
    rays = RealRays(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 1.0)
    rays.reflect(1.0, 0.0, 0.0)

    assert_allclose(rays.L[0], -1.0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, atol=1e-10)
    assert_allclose(rays.N[0], 0.0, atol=1e-10)

    # Reflect rays on a surface with normal (0, 1, 0)
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 1.0)
    rays.reflect(0.0, 1.0, 0.0)

    assert_allclose(rays.L[0], 0.0, atol=1e-10)
    assert_allclose(rays.M[0], -1.0, atol=1e-10)
    assert_allclose(rays.N[0], 0.0, atol=1e-10)


def test_real_rays_str(set_test_backend):
    """Tests the __str__ method of the RealRays class."""

    rays_empty = RealRays([], [], [], [], [], [], [], [])
    rays_empty.x = []
    assert str(rays_empty) == "RealRays object (No rays)"

    x = be.array([1.0, 1.1])
    y = be.array([2.0, 2.1])
    z = be.array([3.0, 3.1])
    L = be.array([0.0, 0.1])
    M = be.array([0.0, 0.2])
    N = be.array([1.0, 0.9])
    i = be.array([1.0, 0.5])
    w = be.array([0.55, 0.65])
    rays_few = RealRays(x, y, z, L, M, N, i, w)

    h = " Ray # |          x |          y |          z |          L |          M |          N |  Intensity |   Wavelength\n"
    s = "----------------------------------------------------------------------------------------------------------------------\n"
    r0 = "     0 |     1.0000 |     2.0000 |     3.0000 |   0.000000 |   0.000000 |   1.000000 |     1.0000 |       0.5500\n"
    r1 = "     1 |     1.1000 |     2.1000 |     3.1000 |   0.100000 |   0.200000 |   0.900000 |     0.5000 |       0.6500\n"
    f = "Showing 2 of 2 rays.\n"
    expected_few = h + s + r0 + r1 + s + f
    actual_output = str(rays_few)
    print("\n--- Actual Output (repr) ---")
    print(repr(actual_output))
    print("--- Expected Output (repr) ---")
    print(repr(expected_few))
    print("--- End ---")
    assert str(rays_few) == expected_few

    x = be.array([1.0, 1.1, 1.2, 1.3, 1.4])
    y = be.array([2.0, 2.1, 2.2, 2.3, 2.4])
    z = be.array([3.0, 3.1, 3.2, 3.3, 3.4])
    L = be.array([0.0, 0.1, 0.2, 0.3, 0.4])
    M = be.array([0.0, 0.1, 0.2, 0.3, 0.4])
    N = be.array([1.0, 0.9, 0.8, 0.7, 0.6])
    i = be.array([1.0, 0.9, 0.8, 0.7, 0.6])
    w = be.array([0.55, 0.55, 0.55, 0.55, 0.55])
    rays_many = RealRays(x, y, z, L, M, N, i, w)

    # Build expected_many programmatically
    r0 = "     0 |     1.0000 |     2.0000 |     3.0000 |   0.000000 |   0.000000 |   1.000000 |     1.0000 |       0.5500\n"
    r2 = "     2 |     1.2000 |     2.2000 |     3.2000 |   0.200000 |   0.200000 |   0.800000 |     0.8000 |       0.5500\n"
    r4 = "     4 |     1.4000 |     2.4000 |     3.4000 |   0.400000 |   0.400000 |   0.600000 |     0.6000 |       0.5500\n"
    f = "Showing 3 of 5 rays.\n"
    expected_many = h + s + r0 + r2 + r4 + s + f
    assert str(rays_many) == expected_many


class TestPolarizationState:
    def test_constructor(self, set_test_backend):
        state = PolarizationState(is_polarized=True, Ex=1, Ey=2, phase_x=0, phase_y=1)
        assert state.is_polarized is True
        assert_allclose(state.Ex, 1 / be.sqrt(5))
        assert_allclose(state.Ey, 2 / be.sqrt(5))
        assert state.phase_x == 0
        assert state.phase_y == 1

    def test_constructor_invalid(self, set_test_backend):
        with pytest.raises(ValueError):
            PolarizationState(is_polarized=True, Ex=1, Ey=2, phase_x=0, phase_y=None)
        with pytest.raises(ValueError):
            PolarizationState(is_polarized=False, Ex=1, Ey=2, phase_x=None, phase_y=1)

    def test_str(self, set_test_backend):
        state = PolarizationState(is_polarized=True, Ex=1, Ey=0, phase_x=0, phase_y=1)
        val = "Polarized Light: Ex: 1.0, Ey: 0.0, Phase x: 0.0, Phase y: 1.0"
        assert str(state) == val

    def test_str_unpolarized(self, set_test_backend):
        state = PolarizationState(is_polarized=False)
        assert str(state) == "Unpolarized Light"

    def test_repr(self, set_test_backend):
        state = PolarizationState(is_polarized=True, Ex=0, Ey=1, phase_x=0, phase_y=1)
        val = "Polarized Light: Ex: 0.0, Ey: 1.0, Phase x: 0.0, Phase y: 1.0"
        assert repr(state) == val


class TestCreatePolarization:
    def test_create_polarization_unpolarized(self, set_test_backend):
        state = create_polarization("unpolarized")
        assert state.is_polarized is False
        assert state.Ex is None
        assert state.Ey is None
        assert state.phase_x is None
        assert state.phase_y is None

    def test_create_polarization_horizontal(self, set_test_backend):
        state = create_polarization("H")
        assert state.is_polarized is True
        assert_allclose(state.Ex, 1.0, atol=1e-10)
        assert_allclose(state.Ey, 0.0, atol=1e-10)
        assert_allclose(state.phase_x, 0.0, atol=1e-10)
        assert_allclose(state.phase_y, 0.0, atol=1e-10)

    def test_create_polarization_vertical(self, set_test_backend):
        state = create_polarization("V")
        assert state.is_polarized is True
        assert_allclose(state.Ex, 0.0, atol=1e-10)
        assert_allclose(state.Ey, 1.0, atol=1e-10)
        assert_allclose(state.phase_x, 0.0, atol=1e-10)
        assert_allclose(state.phase_y, 0.0, atol=1e-10)

    def test_create_polarization_linear_45(self, set_test_backend):
        state = create_polarization("L+45")
        assert state.is_polarized is True
        assert_allclose(state.Ex, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.Ey, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.phase_x, 0.0, atol=1e-10)
        assert_allclose(state.phase_y, 0.0, atol=1e-10)

    def test_create_polarization_linear_minus_45(self, set_test_backend):
        state = create_polarization("L-45")
        assert state.is_polarized is True
        assert_allclose(state.Ex, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.Ey, -be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.phase_x, 0.0, atol=1e-10)
        assert_allclose(state.phase_y, 0.0, atol=1e-10)

    def test_create_polarization_right_circular(self, set_test_backend):
        state = create_polarization("RCP")
        assert state.is_polarized is True
        assert_allclose(state.Ex, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.Ey, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.phase_x, 0.0, atol=1e-10)
        assert_allclose(state.phase_y, -be.pi / 2, atol=1e-10)

    def test_create_polarization_left_circular(self, set_test_backend):
        state = create_polarization("LCP")
        assert state.is_polarized is True
        assert_allclose(state.Ex, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.Ey, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.phase_x, 0.0, atol=1e-10)
        assert_allclose(state.phase_y, be.pi / 2, atol=1e-10)

    def test_create_polarization_invalid(self, set_test_backend):
        with pytest.raises(ValueError):
            create_polarization("invalid")


class TestPolarizedRays:
    def test_init(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L = be.array([0.0])
        M = be.array([0.0])
        N = be.array([1.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L, M, N, intensity, wavelength)

        assert isinstance(rays.p, be.ndarray)
        assert rays.p.shape == (1, 3, 3)
        assert_allclose(rays.p[0], be.eye(3))

    def test_get_output_field(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L = be.array([0.0])
        M = be.array([0.0])
        N = be.array([1.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L, M, N, intensity, wavelength)
        E = be.array([[1.0, 0.0, 0.0]])

        output_field = rays.get_output_field(E)
        assert output_field.shape == (1, 3)
        assert_allclose(output_field, E)

    def test_update_intensity(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L = be.array([0.0])
        M = be.array([0.0])
        N = be.array([1.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L, M, N, intensity, wavelength)
        state = PolarizationState(
            is_polarized=True,
            Ex=1.0,
            Ey=0.0,
            phase_x=0.0,
            phase_y=0.0,
        )

        rays.update_intensity(state)
        assert rays.i.shape == (1,)
        assert_allclose(rays.i[0], 1.0, atol=1e-10)

        # test case for unpolarized light
        state = PolarizationState(is_polarized=False)
        rays.update_intensity(state)
        assert rays.i.shape == (1,)
        assert_allclose(rays.i[0], 1.0, atol=1e-10)

    def test_update(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L = be.array([0.0])
        M = be.array([0.0])
        N = be.array([1.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L, M, N, intensity, wavelength)
        rays.L0 = be.array([0.0])
        rays.M0 = be.array([0.0])
        rays.N0 = be.array([1.0])
        jones_matrix = be.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

        rays.update(jones_matrix)
        assert rays.p.shape == (1, 3, 3)
        assert_allclose(rays.p, jones_matrix)

        # test case when k not orthogonal to N0
        rays.L0 = be.array([0.0])
        rays.M0 = be.array([0.1])
        rays.N0 = be.sqrt(be.array([1 - 0.1**2]))
        rays.update(jones_matrix)
        assert rays.p.shape == (1, 3, 3)
        expected_jones_matrix = be.array(
            [[[1.0, 0.0, 0.0], [0.0, 0.99498744, -0.1], [0.0, 0.1, 0.99498744]]],
        )
        assert_allclose(
            rays.p, expected_jones_matrix, atol=1e-8
        )  # Reduced tolerance slightly for potential backend differences

        # test case when jones = None
        rays.L0 = be.array([0.0])
        rays.M0 = be.array([0.0])
        rays.N0 = be.array([1.0])
        rays.update(None)
        assert rays.p.shape == (1, 3, 3)
        # The state should not change if jones_matrix is None, so compare to previous state
        assert_allclose(rays.p, expected_jones_matrix, atol=1e-8)

    def test_get_3d_electric_field(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L = be.array([0.0])
        M = be.array([0.0])
        N = be.array([1.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L, M, N, intensity, wavelength)
        state = PolarizationState(
            is_polarized=True,
            Ex=1.0,
            Ey=0.0,
            phase_x=0.0,
            phase_y=0.0,
        )

        E = rays._get_3d_electric_field(state)
        assert E.shape == (1, 3)
        assert_allclose(E, be.array([[1.0, 0.0, 0.0]]), atol=1e-10)

    def test_get_3d_electric_field_error(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L = be.array([1.0])  # k-vector propagates in x-direction
        M = be.array([0.0])
        N = be.array([0.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L, M, N, intensity, wavelength)
        rays._L0 = be.array([1.0])
        rays._M0 = be.array([0.0])
        rays._N0 = be.array([0.0])
        state = PolarizationState(is_polarized=True, Ex=1, Ey=0, phase_x=0, phase_y=0)

        with pytest.raises(ValueError):
            rays._get_3d_electric_field(state)


@pytest.mark.usefixtures("set_test_backend")
class TestRayGenerator:
    def test_generate_rays(self):
        Hx = 0.5
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        wavelength = 0.55

        lens = TessarLens()
        generator = RayGenerator(lens)
        rays = generator.generate_rays(Hx, Hy, Px, Py, wavelength)

        assert isinstance(rays, RealRays)
        assert rays.x.shape == (2,)
        assert rays.y.shape == (2,)
        assert rays.z.shape == (2,)
        assert rays.L.shape == (2,)
        assert rays.M.shape == (2,)
        assert rays.N.shape == (2,)
        assert rays.i.shape == (2,)
        assert rays.w.shape == (2,)

        assert_allclose(rays.x, be.array([-0.23535066, -0.1909309]), atol=1e-8)
        assert_allclose(rays.y, be.array([-0.23535066, -0.1909309]), atol=1e-8)
        assert_allclose(rays.z, be.array([-0.88839505, -0.88839505]), atol=1e-8)
        assert_allclose(rays.L, be.array([0.17519154, 0.17519154]), atol=1e-8)
        assert_allclose(rays.M, be.array([0.17519154, 0.17519154]), atol=1e-8)
        assert_allclose(rays.N, be.array([0.96882189, 0.96882189]), atol=1e-8)
        assert_allclose(rays.i, be.array([1.0, 1.0]), atol=1e-8)
        assert_allclose(rays.w, be.array([0.55, 0.55]), atol=1e-8)

    def test_generate_rays_telecentric(self):
        lens = UVProjectionLens()
        generator = RayGenerator(lens)

        Hx = 0.0
        Hy = 1.0
        Px = 0.8
        Py = 0.0
        wavelength = 0.248

        rays = generator.generate_rays(Hx, Hy, Px, Py, wavelength)
        assert_allclose(rays.x[0], 0.0, atol=1e-8)
        assert_allclose(rays.y[0], 48.0, atol=1e-8)
        assert_allclose(rays.z[0], -110.85883544, atol=1e-8)
        assert_allclose(rays.L[0], 0.10674041, atol=1e-8)
        assert_allclose(rays.M[0], 0.0, atol=1e-8)
        assert_allclose(rays.N[0], 0.99428692, atol=1e-8)
        assert_allclose(rays.i[0], 1.0, atol=1e-8)
        assert_allclose(rays.w[0], 0.248, atol=1e-8)

    def test_generate_rays_invalid_field_type(self):
        lens = UVProjectionLens()
        generator = RayGenerator(lens)

        Hx = 0.5
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        wavelength = 0.55

        lens.set_aperture("EPD", 1.0)
        with pytest.raises(ValueError):
            generator.generate_rays(Hx, Hy, Px, Py, wavelength)

        lens.set_aperture("imageFNO", 10.0)
        with pytest.raises(ValueError):
            generator.generate_rays(Hx, Hy, Px, Py, wavelength)

        lens.fields.set_type("angle")
        with pytest.raises(ValueError):
            generator.generate_rays(Hx, Hy, Px, Py, wavelength)

    def test_invalid_polarization(self):
        lens = TessarLens()
        lens.surfaces.set_fresnel_coatings()
        generator = RayGenerator(lens)

        Hx = 0.5
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        wavelength = 0.55

        # polarization type is 'ignore' but lens has coatings
        with pytest.raises(ValueError):
            generator.generate_rays(Hx, Hy, Px, Py, wavelength)

    def test_generate_polarized_rays(self):
        Hx = 0.5
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        wavelength = 0.55

        lens = TessarLens()
        state = PolarizationState(is_polarized=False)
        lens.updater.set_polarization(state)
        generator = RayGenerator(lens)
        rays = generator.generate_rays(Hx, Hy, Px, Py, wavelength)

        assert isinstance(rays, PolarizedRays)
        assert rays.x.shape == (2,)
        assert rays.y.shape == (2,)
        assert rays.z.shape == (2,)
        assert rays.L.shape == (2,)
        assert rays.M.shape == (2,)
        assert rays.N.shape == (2,)
        assert rays.i.shape == (2,)
        assert rays.w.shape == (2,)
        assert rays.p.shape == (2, 3, 3)

        assert_allclose(rays.x, be.array([-0.23535066, -0.1909309]), atol=1e-8)
        assert_allclose(rays.y, be.array([-0.23535066, -0.1909309]), atol=1e-8)
        assert_allclose(rays.z, be.array([-0.88839505, -0.88839505]), atol=1e-8)
        assert_allclose(rays.L, be.array([0.17519154, 0.17519154]), atol=1e-8)
        assert_allclose(rays.M, be.array([0.17519154, 0.17519154]), atol=1e-8)
        assert_allclose(rays.N, be.array([0.96882189, 0.96882189]), atol=1e-8)
        assert_allclose(rays.i, be.array([1.0, 1.0]), atol=1e-8)
        assert_allclose(rays.w, be.array([0.55, 0.55]), atol=1e-8)

    def test_get_ray_origins_infinite_object(self):
        lens = TessarLens()
        generator = RayGenerator(lens)

        Hx = 0.5
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        wavelength = 0.55

        rays = generator.generate_rays(Hx, Hy, Px, Py, wavelength)

        assert rays.x.shape == (2,)
        assert rays.y.shape == (2,)
        assert rays.z.shape == (2,)

    def test_get_ray_origins_invalid_field_type(self):
        lens = TessarLens()
        lens.fields.set_type("object_height")
        generator = RayGenerator(lens)

        Hx = 0.5
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        wavelength = 0.55

        with pytest.raises(ValueError):
            generator.generate_rays(Hx, Hy, Px, Py, wavelength)

    def test_invalid_ray_origin_telecentric(self):
        lens = TessarLens()
        lens.obj_space_telecentric = True
        generator = RayGenerator(lens)

        Hx = 0.0
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        wavelength = 0.55

        with pytest.raises(ValueError):
            generator.generate_rays(Hx, Hy, Px, Py, wavelength)


@pytest.mark.usefixtures("set_test_backend")
class TestOpticTrace:
    @pytest.fixture
    def sample_optic(self):
        """Provides a configured TessarLens instance for tracing tests."""
        optic = TessarLens()
        optic.fields.add(y=0.7)
        optic.fields.add(y=1.0)
        return optic

    def test_trace_single_field_scalar_input(self, sample_optic):
        """Tests .trace() with a single field point and scalar Hx, Hy."""
        num_rays = 50
        rays = sample_optic.trace(
            Hx=0.0,
            Hy=0.7,
            wavelength=sample_optic.primary_wavelength,
            num_rays=num_rays,
            distribution="random",
        )
        assert isinstance(rays, RealRays)
        assert len(rays.x) == num_rays

    def test_trace_multiple_fields_array_input(self, sample_optic):
        """Tests .trace() with multiple field points using array inputs for Hx, Hy."""
        num_rays_grid_size = 20
        num_fields = 2
        Hx_all = be.array([0.0, 0.0])
        Hy_all = be.array([0.7, 1.0])

        from optiland.distribution import create_distribution

        dist = create_distribution("uniform")
        dist.generate_points(num_rays_grid_size)
        num_pupil_points = len(dist.x)

        rays = sample_optic.trace(
            Hx=Hx_all,
            Hy=Hy_all,
            wavelength=sample_optic.primary_wavelength,
            num_rays=num_rays_grid_size,
            distribution="uniform",
        )
        assert isinstance(rays, RealRays)
        assert len(rays.x) == num_pupil_points * num_fields

    def test_trace_generic_single_ray(self, sample_optic):
        """Tests .trace_generic() with scalar inputs for a single ray."""
        rays = sample_optic.trace_generic(
            Hx=0.0, Hy=1.0, Px=0.5, Py=0.5, wavelength=sample_optic.primary_wavelength
        )
        assert isinstance(rays, RealRays)
        assert len(rays.x) == 1

    def test_trace_generic_multiple_rays_mixed_inputs(self, sample_optic):
        """Tests .trace_generic() with a mix of scalar and array inputs."""
        num_pupil_points = 10
        Px = be.linspace(-1, 1, num_pupil_points)
        Py = be.zeros(num_pupil_points)

        rays = sample_optic.trace_generic(
            Hx=0.0, Hy=0.7, Px=Px, Py=Py, wavelength=sample_optic.primary_wavelength
        )
        assert isinstance(rays, RealRays)
        assert len(rays.x) == num_pupil_points

    def test_trace_generic_multiple_fields_and_pupil_points(self, sample_optic):
        """Tests .trace_generic() with manually expanded arrays for all coordinates."""

        Hx_in = be.array([0.0, 0.5])
        Hy_in = be.array([0.7, 0.5])
        Px_in = be.array([-0.5, 0.5])
        Py_in = be.array([0.0, 0.0])

        num_fields = len(Hx_in)
        num_pupils = len(Px_in)

        Hx = be.repeat(Hx_in, num_pupils)
        Hy = be.repeat(Hy_in, num_pupils)
        Px = be.tile(Px_in, num_fields)
        Py = be.tile(Py_in, num_fields)

        # results in: 4 rays (2 fields x 2 pupil points)
        rays = sample_optic.trace_generic(
            Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=sample_optic.primary_wavelength
        )
        assert isinstance(rays, RealRays)
        assert len(rays.x) == 4
