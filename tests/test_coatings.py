from __future__ import annotations

import pytest

import optiland.backend as be
from optiland import coatings, materials, rays

from .utils import assert_allclose


@pytest.fixture
def rays_parallel():
    num = 10
    x = be.linspace(0, 10, num)
    y = be.linspace(10, 20, num)
    z = be.linspace(-5, 10, num)
    L = be.zeros(num)
    M = be.zeros(num)
    N = be.ones(num)
    intensity = be.ones(num)
    wavelength = be.ones(num)

    r = rays.RealRays(x, y, z, L, M, N, intensity, wavelength)
    r.L0 = r.L
    r.M0 = r.M
    r.N0 = r.N
    return r


@pytest.fixture
def rays_parallel_polarized():
    num = 10
    x = be.linspace(0, 10, num)
    y = be.linspace(10, 20, num)
    z = be.linspace(-5, 10, num)
    L = be.zeros(num)
    M = be.zeros(num)
    N = be.ones(num)
    intensity = be.ones(num)
    wavelength = be.ones(num)

    r = rays.PolarizedRays(x, y, z, L, M, N, intensity, wavelength)
    r.L0 = r.L
    r.M0 = r.M
    r.N0 = r.N
    return r


@pytest.fixture
def rays_non_parallel():
    num = 10
    x = be.linspace(0, 10, num)
    y = be.linspace(10, 20, num)
    z = be.linspace(-5, 10, num)
    L = be.linspace(0, 0.5, num)
    M = be.linspace(-0.3, 0.4, num)
    N = be.sqrt(1 - L**2 - M**2)
    intensity = be.ones(num)
    wavelength = be.ones(num)

    r = rays.RealRays(x, y, z, L, M, N, intensity, wavelength)
    r.L0 = r.L
    r.M0 = r.M
    r.N0 = r.N
    return r


class TestSimpleCoating:
    def test_interact_reflect(self, set_test_backend, rays_parallel):
        coating = coatings.SimpleCoating(transmittance=0.8, reflectance=0.1)

        x_before = be.copy(rays_parallel.x)
        y_before = be.copy(rays_parallel.y)
        z_before = be.copy(rays_parallel.z)
        L_before = be.copy(rays_parallel.L)
        M_before = be.copy(rays_parallel.M)
        N_before = be.copy(rays_parallel.N)
        i_before = be.copy(rays_parallel.i)
        w_before = be.copy(rays_parallel.w)
        rays_after = coating.interact(rays_parallel, reflect=True)

        assert_allclose(rays_after.x, x_before)
        assert_allclose(rays_after.y, y_before)
        assert_allclose(rays_after.z, z_before)
        assert_allclose(rays_after.L, L_before)
        assert_allclose(rays_after.M, M_before)
        assert_allclose(rays_after.N, N_before)
        assert_allclose(rays_after.i, 0.1 * i_before)
        assert_allclose(rays_after.w, w_before)

    def test_interact_transmit(self, set_test_backend, rays_parallel):
        coating = coatings.SimpleCoating(transmittance=0.3, reflectance=0.5)

        x_before = be.copy(rays_parallel.x)
        y_before = be.copy(rays_parallel.y)
        z_before = be.copy(rays_parallel.z)
        L_before = be.copy(rays_parallel.L)
        M_before = be.copy(rays_parallel.M)
        N_before = be.copy(rays_parallel.N)
        i_before = be.copy(rays_parallel.i)
        w_before = be.copy(rays_parallel.w)
        rays_after = coating.interact(rays_parallel, reflect=False)

        assert_allclose(rays_after.x, x_before)
        assert_allclose(rays_after.y, y_before)
        assert_allclose(rays_after.z, z_before)
        assert_allclose(rays_after.L, L_before)
        assert_allclose(rays_after.M, M_before)
        assert_allclose(rays_after.N, N_before)
        assert_allclose(rays_after.i, 0.3 * i_before)
        assert_allclose(rays_after.w, w_before)

    def test_compute_aoi(self, set_test_backend, rays_parallel):
        coating = coatings.SimpleCoating(transmittance=0.3, reflectance=0.5)

        nx = be.zeros_like(rays_parallel.x)
        ny = be.zeros_like(rays_parallel.y)
        nz = be.ones_like(rays_parallel.z)

        aoi = coating._compute_aoi(rays_parallel, nx, ny, nz)

        assert be.all(aoi == 0)

    def test_to_dict(self, set_test_backend):
        coating = coatings.SimpleCoating(transmittance=0.3, reflectance=0.5)
        assert coating.to_dict() == {
            "type": "SimpleCoating",
            "transmittance": 0.3,
            "reflectance": 0.5,
        }

    def test_from_dict(self, set_test_backend):
        coating_dict = {
            "type": "SimpleCoating",
            "transmittance": 0.3,
            "reflectance": 0.5,
        }
        coating = coatings.SimpleCoating.from_dict(coating_dict)
        assert coating.transmittance == 0.3
        assert coating.reflectance == 0.5

        # Also test BaseCoating.from_dict
        base_coating = coatings.BaseCoating.from_dict(coating_dict)
        assert base_coating.transmittance == 0.3
        assert base_coating.reflectance == 0.5


class TestFresnelCoating:
    def test_reflect(self, set_test_backend, rays_parallel_polarized):
        mat1 = materials.IdealMaterial(n=1.0)
        mat2 = materials.IdealMaterial(n=1.5)
        coating = coatings.FresnelCoating(mat1, mat2)
        state = rays.PolarizationState(is_polarized=False)

        nx = be.zeros_like(rays_parallel_polarized.x)
        ny = be.zeros_like(rays_parallel_polarized.y)
        nz = be.ones_like(rays_parallel_polarized.z)

        i_before = be.copy(rays_parallel_polarized.i)
        rays_after = coating.reflect(rays_parallel_polarized, nx, ny, nz)
        rays_after.update_intensity(state)

        R = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2
        assert be.all(rays_after.i == R * i_before)

    def test_transmit(self, set_test_backend, rays_parallel_polarized):
        mat1 = materials.IdealMaterial(n=1.0)
        mat2 = materials.IdealMaterial(n=1.5)
        coating = coatings.FresnelCoating(mat1, mat2)
        state = rays.PolarizationState(is_polarized=False)

        nx = be.zeros_like(rays_parallel_polarized.x)
        ny = be.zeros_like(rays_parallel_polarized.y)
        nz = be.ones_like(rays_parallel_polarized.z)

        i_before = be.copy(rays_parallel_polarized.i)
        rays_after = coating.transmit(rays_parallel_polarized, nx, ny, nz)
        rays_after.update_intensity(state)

        R = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2
        assert be.allclose(rays_after.i * 1.5, (1 - R) * i_before)

    def test_to_dict(self, set_test_backend):
        mat1 = materials.IdealMaterial(n=1.0)
        mat2 = materials.IdealMaterial(n=1.5)
        coating = coatings.FresnelCoating(mat1, mat2)
        assert coating.to_dict() == {
            "type": "FresnelCoating",
            "material_pre": mat1.to_dict(),
            "material_post": mat2.to_dict(),
        }

    def test_from_dict(self, set_test_backend):
        mat1 = materials.IdealMaterial(n=1.0)
        mat2 = materials.IdealMaterial(n=1.5)
        coating = coatings.FresnelCoating(mat1, mat2)
        coating_dict = coating.to_dict()
        coating2 = coatings.FresnelCoating.from_dict(coating_dict)
        assert coating2.to_dict() == coating.to_dict()


class TestPolarizerCoating:
    def test_init(self, set_test_backend):
        coating = coatings.PolarizerCoating(axis=(0.0, 1.0, 0.0))
        assert coating.axis == (0.0, 1.0, 0.0)
        assert coating.jones is not None

    def test_to_dict(self, set_test_backend):
        coating = coatings.PolarizerCoating(axis=(0.0, 1.0, 0.0))
        assert coating.to_dict() == {
            "type": "PolarizerCoating",
            "axis": [0.0, 1.0, 0.0],
        }

    def test_from_dict(self, set_test_backend):
        coating_dict = {
            "type": "PolarizerCoating",
            "axis": (0.0, 1.0, 0.0),
        }
        coating = coatings.PolarizerCoating.from_dict(coating_dict)
        assert coating.axis == (0.0, 1.0, 0.0)

        # test fallback
        coating_dict_2 = {"type": "PolarizerCoating"}
        coating2 = coatings.PolarizerCoating.from_dict(coating_dict_2)
        assert coating2.axis == (1.0, 0.0, 0.0)


class TestRetarderCoating:
    def test_init(self, set_test_backend):
        coating = coatings.RetarderCoating(retardance=1.57, axis=(0.0, 1.0, 0.0))
        assert coating.retardance == 1.57
        assert coating.axis == (0.0, 1.0, 0.0)
        assert coating.jones is not None

    def test_to_dict(self, set_test_backend):
        coating = coatings.RetarderCoating(retardance=1.57, axis=(0.0, 1.0, 0.0))
        assert coating.to_dict() == {
            "type": "RetarderCoating",
            "retardance": 1.57,
            "axis": [0.0, 1.0, 0.0],
        }

    def test_from_dict(self, set_test_backend):
        coating_dict = {
            "type": "RetarderCoating",
            "retardance": 1.57,
            "axis": (0.0, 1.0, 0.0),
        }
        coating = coatings.RetarderCoating.from_dict(coating_dict)
        assert coating.retardance == 1.57
        assert coating.axis == (0.0, 1.0, 0.0)

        # test fallback
        coating_dict_2 = {"type": "RetarderCoating", "retardance": 1.57}
        coating2 = coatings.RetarderCoating.from_dict(coating_dict_2)
        assert coating2.axis == (1.0, 0.0, 0.0)


class TestThinFilmCoatings:
    def test_jones_thin_film_matrix_transmit_and_reflect(
        self, set_test_backend, rays_parallel_polarized
    ):
        air = materials.IdealMaterial(n=1.0)
        glass = materials.IdealMaterial(n=1.5)
        stack = coatings.ThinFilmStack(air, glass)
        stack.add_layer_nm(materials.IdealMaterial(n=1.3), 120.0, name="L1")

        jones_tf = coatings.JonesThinFilm(stack)

        j_transmit = jones_tf.calculate_matrix(rays_parallel_polarized, reflect=False)
        j_reflect = jones_tf.calculate_matrix(rays_parallel_polarized, reflect=True)

        assert j_transmit.shape == (10, 3, 3)
        assert j_reflect.shape == (10, 3, 3)

    def test_thin_film_coating_init_and_interaction(
        self, set_test_backend, rays_parallel_polarized
    ):
        air = materials.IdealMaterial(n=1.0)
        glass = materials.IdealMaterial(n=1.5)
        layer_mat = materials.IdealMaterial(n=1.3)

        coating = coatings.ThinFilmCoating(
            material_pre=air,
            material_post=glass,
            layers=[(layer_mat, 95.0, "L1")],
        )

        assert len(coating.stack.layers) == 1
        assert coating.jones is not None

        nx = be.zeros_like(rays_parallel_polarized.x)
        ny = be.zeros_like(rays_parallel_polarized.y)
        nz = be.ones_like(rays_parallel_polarized.z)

        state = rays.PolarizationState(is_polarized=False)
        rays_r = coating.reflect(rays_parallel_polarized, nx, ny, nz)
        rays_r.update_intensity(state)

        rays_t = coating.transmit(rays_parallel_polarized, nx, ny, nz)
        rays_t.update_intensity(state)

        assert be.all(rays_r.i >= 0)
        assert be.all(rays_t.i >= 0)
