from __future__ import annotations

import optiland.backend as be
from optiland import scatter
from optiland.rays import RealRays


class TestGetPointLambertian:
    def test_point_within_unit_disk(self):
        for _ in range(1000):
            x, y = scatter.get_point_lambertian()
            assert x**2 + y**2 <= 1

    def test_randomness(self):
        points = [scatter.get_point_lambertian() for _ in range(1000)]
        unique_points = set(points)
        assert len(unique_points) == 1000

    def test_output_type(self):
        x, y = scatter.get_point_lambertian()
        assert isinstance(x, float)
        assert isinstance(y, float)


class TestGetPointGaussian:
    def test_output_type(self):
        x, y = scatter.get_point_gaussian(1)
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_randomness(self):
        points = [scatter.get_point_gaussian(1) for _ in range(1000)]
        unique_points = set(points)
        assert len(unique_points) == 1000


class TestScatter:
    def test_output_type(self):
        L, M, N, nx, ny, nz = be.random.rand(6)
        s = scatter.scatter(L, M, N, nx, ny, nz, scatter.get_point_lambertian)
        assert isinstance(s, be.ndarray)

    def test_randomness(self):
        L, M, N, nx, ny, nz = be.random.rand(6)
        points = [
            scatter.scatter(L, M, N, nx, ny, nz, scatter.get_point_lambertian)
            for _ in range(1000)
        ]
        unique_points = list(set(tuple(point) for point in points))
        assert len(unique_points) == 1000

    def test_large_L(self):
        nx = 0.1
        ny = 0.3
        nz = be.sqrt(1 - nx**2 - ny**2)
        L = 0.99999
        M = 0.0
        N = be.sqrt(1 - L**2)
        s = scatter.scatter(L, M, N, nx, ny, nz, scatter.get_point_lambertian)
        assert be.isclose(be.linalg.norm(s), 1)

    def test_output_unit_vector(self):
        for _ in range(1000):
            L, M, N, nx, ny, nz = be.random.rand(6)

            mag = be.sqrt(L**2 + M**2 + N**2)
            L /= mag
            M /= mag
            N /= mag

            mag = be.sqrt(nx**2 + ny**2 + nz**2)
            nx /= mag
            ny /= mag
            nz /= mag

            s = scatter.scatter(L, M, N, nx, ny, nz, scatter.get_point_lambertian)
            assert be.isclose(be.linalg.norm(s), 1)


class TestFuncWrapper:
    def test_output_type(self):
        func = scatter.get_point_lambertian
        wrapped_func = scatter.func_wrapper(func)
        assert callable(wrapped_func)

    def test_output(self):
        func = scatter.get_point_lambertian
        wrapped_func = scatter.func_wrapper(func)
        out = wrapped_func()
        assert isinstance(out, tuple)


class TestScatterParallel:
    def test_output_type(self):
        L, M, N, nx, ny, nz = be.random.rand(6, 10)
        s = scatter.scatter_parallel(L, M, N, nx, ny, nz, scatter.get_point_lambertian)
        assert isinstance(s, be.ndarray)

    def test_randomness(self):
        L, M, N, nx, ny, nz = be.random.rand(6, 1000)
        points = scatter.scatter_parallel(
            L,
            M,
            N,
            nx,
            ny,
            nz,
            scatter.get_point_lambertian,
        )
        unique_points = list(set(tuple(point) for point in points))
        assert len(unique_points) == 1000

    def test_output_unit_vector(self):
        L, M, N, nx, ny, nz = be.random.rand(6, 1000)

        mag = be.sqrt(L**2 + M**2 + N**2)
        L /= mag
        M /= mag
        N /= mag

        mag = be.sqrt(nx**2 + ny**2 + nz**2)
        nx /= mag
        ny /= mag
        nz /= mag

        s = scatter.scatter_parallel(L, M, N, nx, ny, nz, scatter.get_point_lambertian)
        assert be.allclose(be.linalg.norm(s, axis=1), 1)


class TestLamberianBSDF:
    def test_output_type(self):
        scatterer = scatter.LambertianBSDF()
        x = be.random.rand(100)
        L, M, N, nx, ny, nz = be.random.rand(6, 100)
        rays = RealRays(x, x, x, L, M, N, x, x)
        s = scatterer.scatter(rays, nx, ny, nz)
        assert isinstance(s, RealRays)

    def test_output_unit_vector(self):
        scatterer = scatter.LambertianBSDF()
        x = be.random.rand(100)
        L, M, N, nx, ny, nz = be.random.rand(6, 100)

        mag = be.sqrt(L**2 + M**2 + N**2)
        L /= mag
        M /= mag
        N /= mag

        mag = be.sqrt(nx**2 + ny**2 + nz**2)
        nx /= mag
        ny /= mag
        nz /= mag

        rays = RealRays(x, x, x, L, M, N, x, x)
        rays_out = scatterer.scatter(rays, nx, ny, nz)

        mag = rays_out.L**2 + rays_out.M**2 + rays_out.N**2
        assert be.allclose(mag, 1)


class TestGaussianBSDF:
    def test_output(self):
        scatterer = scatter.GaussianBSDF(sigma=0.01)
        x = be.random.rand(100)
        L, M, N, nx, ny, nz = be.random.rand(6, 100)

        mag = be.sqrt(L**2 + M**2 + N**2)
        L /= mag
        M /= mag
        N /= mag

        mag = be.sqrt(nx**2 + ny**2 + nz**2)
        nx /= mag
        ny /= mag
        nz /= mag

        rays = RealRays(x, x, x, L, M, N, x, x)
        rays_out = scatterer.scatter(rays, nx, ny, nz)

        mag = rays_out.L**2 + rays_out.M**2 + rays_out.N**2
        assert be.allclose(mag, 1)
        assert isinstance(rays_out, RealRays)
