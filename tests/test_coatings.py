from __future__ import annotations

import numpy as np
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

    @staticmethod
    def _field_after(coating, k, reflect):
        """One ray through a plane z = 0: Optiland bends it, then the coating
        acts. Returns the polarization matrix and the outgoing direction."""
        ray = rays.PolarizedRays(
            *(be.zeros(1),) * 3, *(be.array([c]) for c in k), be.ones(1), be.ones(1)
        )
        nx, ny, nz = be.zeros(1), be.zeros(1), be.ones(1)
        if reflect:
            ray.reflect(nx, ny, nz)
            coating.reflect(ray, nx, ny, nz)
        else:
            ray.refract(nx, ny, nz, 1.0, 1.5)
            coating.transmit(ray, nx, ny, nz)
        k_out = np.array([be.to_numpy(v)[0] for v in (ray.L, ray.M, ray.N)])
        return be.to_numpy(ray.p)[0], k_out

    @pytest.mark.parametrize("reflect", [True, False])
    def test_field_matches_closed_form(self, set_test_backend, reflect):
        """The 3D field after the surface is the closed-form Fresnel field, with
        p along s x k for every wave, for any incident polarization."""
        n1, n2 = 1.0, 1.5
        coating = coatings.FresnelCoating(
            materials.IdealMaterial(n=n1), materials.IdealMaterial(n=n2)
        )
        for deg, az in [(0, 0), (20, 30), (45, 100), (70, -60)]:
            th, b = np.radians(deg), np.radians(az)
            k_in = np.array(
                [np.sin(th) * np.cos(b), np.sin(th) * np.sin(b), np.cos(th)]
            )
            c1 = np.cos(th)
            c2 = np.sqrt(1 - (n1 * np.sin(th) / n2) ** 2)
            if reflect:
                ts = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
                tp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
            else:
                ts = 2 * n1 * c1 / (n1 * c1 + n2 * c2)
                tp = 2 * n1 * c1 / (n2 * c1 + n1 * c2)
            s = np.array([-np.sin(b), np.cos(b), 0.0])
            p_in = np.cross(s, k_in)

            pol, k_out = self._field_after(coating, k_in, reflect)
            p_out = np.cross(s, k_out)
            for E in (s, p_in, 0.6 * s + (0.2 + 0.75j) * p_in):
                expected = ts * np.dot(E, s) * s + tp * np.dot(E, p_in) * p_out
                np.testing.assert_allclose(pol @ E, expected, atol=1e-12)

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

    @staticmethod
    def _airy(n0, n1, d, n2, wl, theta, pol, tangential):
        """r and t of one layer on a substrate by the Airy sum, exp(-iwt) (n + ik).

        With ``tangential``, the coefficients are ratios of the tangential
        fields, from the tilted admittances n cos(theta) (s) and n / cos(theta)
        (p). Otherwise they are ratios of the full fields, with p along s x k
        for every wave.
        """
        s0 = n0 * np.sin(theta)
        nc0, nc1, nc2 = (np.sqrt(complex(n**2 - s0**2)) for n in (n0, n1, n2))
        e = np.exp(2j * 2 * np.pi * d * nc1 / wl)

        def interface(na, nca, nb, ncb):
            if pol == "s":
                return (nca - ncb) / (nca + ncb), 2 * nca / (nca + ncb)
            if tangential:
                ya, yb = na**2 / nca, nb**2 / ncb
                return (ya - yb) / (ya + yb), 2 * ya / (ya + yb)
            ca, cb = nca / na, ncb / nb
            den = nb * ca + na * cb
            return (nb * ca - na * cb) / den, 2 * na * ca / den

        r01, t01 = interface(n0, nc0, n1, nc1)
        r12, t12 = interface(n1, nc1, n2, nc2)
        den = 1 + r01 * r12 * e
        return (r01 + r12 * e) / den, t01 * t12 * np.sqrt(e) / den

    @pytest.mark.parametrize("pol", ["s", "p"])
    @pytest.mark.parametrize("n_layer", [1.8, 1.8 + 0.05j])
    def test_stack_r_t_match_airy(self, set_test_backend, pol, n_layer):
        """The stack's r and t are the Airy coefficients of the tangential
        fields, in the n + ik convention, for lossless and absorbing layers."""
        n0, n2, d, wl = 1.0, 1.5 + 0.02j, 0.37, 0.6
        stack = coatings.ThinFilmStack(
            materials.IdealMaterial(n=n0),
            materials.IdealMaterial(n=n2.real, k=n2.imag),
        )
        stack.add_layer_nm(
            materials.IdealMaterial(n=n_layer.real, k=n_layer.imag), d * 1000
        )
        thetas = np.radians([0.0, 20.0, 45.0, 70.0])
        out = stack.compute_rtRTA_elementwise(
            be.array([wl] * 4), be.array(thetas), pol
        )
        expected = [
            self._airy(n0, n_layer, d, n2, wl, th, pol, tangential=True)
            for th in thetas
        ]
        assert_allclose(out["r"], np.array([r for r, _ in expected]))
        assert_allclose(out["t"], np.array([t for _, t in expected]))

    @pytest.mark.parametrize("reflect", [True, False])
    @pytest.mark.parametrize("n_layer", [1.8, 1.8 + 0.05j])
    def test_field_matches_closed_form(self, set_test_backend, reflect, n_layer):
        """The 3D field after the coating is the Airy field, with p along s x k
        for every wave, for any incident polarization."""
        n0, n2, d, wl = 1.0, 1.5, 0.37, 0.6
        coating = coatings.ThinFilmCoating(
            materials.IdealMaterial(n=n0),
            materials.IdealMaterial(n=n2),
            [(materials.IdealMaterial(n=n_layer.real, k=n_layer.imag), d * 1000, "L")],
        )
        for deg, az in [(0, 0), (20, 30), (45, 100), (70, -60)]:
            th, b = np.radians(deg), np.radians(az)
            k_in = np.array(
                [np.sin(th) * np.cos(b), np.sin(th) * np.sin(b), np.cos(th)]
            )
            (r_s, t_s), (r_p, t_p) = (
                self._airy(n0, n_layer, d, n2, wl, th, pol, tangential=False)
                for pol in "sp"
            )
            ts, tp = (r_s, r_p) if reflect else (t_s, t_p)
            s = np.array([-np.sin(b), np.cos(b), 0.0])
            p_in = np.cross(s, k_in)

            ray = rays.PolarizedRays(
                *(be.zeros(1),) * 3,
                *(be.array([c]) for c in k_in),
                be.ones(1),
                be.array([wl]),
            )
            nx, ny, nz = be.zeros(1), be.zeros(1), be.ones(1)
            if reflect:
                ray.reflect(nx, ny, nz)
                coating.reflect(ray, nx, ny, nz)
            else:
                ray.refract(nx, ny, nz, n0, n2)
                coating.transmit(ray, nx, ny, nz)
            jones = be.to_numpy(ray.p)[0]
            k_out = np.array([be.to_numpy(v)[0] for v in (ray.L, ray.M, ray.N)])

            p_out = np.cross(s, k_out)
            for E in (s, p_in, 0.6 * s + (0.2 + 0.75j) * p_in):
                expected = ts * np.dot(E, s) * s + tp * np.dot(E, p_in) * p_out
                np.testing.assert_allclose(jones @ E, expected, atol=1e-12)

    def test_transmit_p_with_absorbing_substrate(self, set_test_backend):
        """Into an absorbing substrate, the p transmission is the Airy
        full-field coefficient, with the complex cosine of the substrate."""
        n0, n_layer, n2, d, wl = 1.0, 1.8, 1.5 + 0.02j, 0.37, 0.6
        stack = coatings.ThinFilmStack(
            materials.IdealMaterial(n=n0),
            materials.IdealMaterial(n=n2.real, k=n2.imag),
        )
        stack.add_layer_nm(materials.IdealMaterial(n=n_layer), d * 1000)
        thetas = np.radians([0.0, 20.0, 45.0, 70.0])
        ray = rays.PolarizedRays(
            *(be.zeros(4),) * 6, be.ones(4), be.array([wl] * 4)
        )
        jones = coatings.JonesThinFilm(stack).calculate_matrix(
            ray, reflect=False, aoi=be.array(thetas)
        )
        expected = [
            self._airy(n0, n_layer, d, n2, wl, th, "p", tangential=False)[1]
            for th in thetas
        ]
        assert_allclose(jones[:, 1, 1], np.array(expected))
