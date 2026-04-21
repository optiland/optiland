"""Regression tests for the NumPy path of Newton-Raphson distance/tracing."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.rays import RealRays
from optiland.samples.simple import AsphericSinglet

from .nr_implicit_test_utils import backend_state, build_reference_even_asphere
from .utils import assert_allclose


def _build_distance_regression_bundle() -> RealRays:
    x = np.array([0.0, 0.35, -0.22])
    y = np.array([0.0, -0.27, 0.18])
    z = np.array([-5.0, -5.0, -4.5])
    L = np.array([0.02, 0.06, -0.04])
    M = np.array([-0.015, -0.035, 0.025])
    N = np.sqrt(1.0 - L**2 - M**2)
    return RealRays(x=x, y=y, z=z, L=L, M=M, N=N, intensity=1.0, wavelength=0.587)


def test_numpy_distance_matches_reference_values():
    with backend_state("numpy", precision="float64"):
        geometry = build_reference_even_asphere()
        rays = _build_distance_regression_bundle()

        t = geometry.distance(rays)
        x_int = rays.x + t * rays.L
        y_int = rays.y + t * rays.M
        z_int = rays.z + t * rays.N

        assert_allclose(
            t,
            np.array([5.0019507685589124, 5.0275941535358291, 4.5111182791860802]),
            rtol=0.0,
            atol=1e-11,
        )
        assert_allclose(
            x_int,
            np.array([0.10003901537117825, 0.65165564921214969, -0.40044473116744322]),
            rtol=0.0,
            atol=1e-11,
        )
        assert_allclose(
            y_int,
            np.array([-0.075029261528383684, -0.44596579537375403, 0.29277795697965203]),
            rtol=0.0,
            atol=1e-11,
        )
        assert_allclose(
            z_int,
            np.array([0.00038741463150682165, 0.015450416545310652, 0.0060968653836308562]),
            rtol=0.0,
            atol=1e-11,
        )


def test_numpy_off_axis_trace_matches_reference_metrics():
    with backend_state("numpy", precision="float64"):
        lens = AsphericSinglet()
        lens.fields.add(y=5.0)
        lens.trace(0.2, 0.3, 0.587, 24, "hexapolar")

        img = lens.surfaces.num_surfaces - 1
        x = be.to_numpy(lens.surfaces.x[img, :])
        y = be.to_numpy(lens.surfaces.y[img, :])
        rms = np.sqrt(((x - x.mean()) ** 2 + (y - y.mean()) ** 2).mean())

        assert_allclose(x.mean(), 0.4533279377925549, rtol=0.0, atol=1e-12)
        assert_allclose(y.mean(), 0.6800782393520015, rtol=0.0, atol=1e-12)
        assert_allclose(rms, 0.0283767065911844, rtol=0.0, atol=1e-12)


def test_numpy_distance_matches_torch_no_grad():
    torch = pytest.importorskip("torch")

    with backend_state("numpy", precision="float64"):
        geometry = build_reference_even_asphere()
        rays = _build_distance_regression_bundle()
        t_numpy = be.to_numpy(geometry.distance(rays))

    with backend_state("torch", precision="float64"):
        geometry = build_reference_even_asphere()
        rays = _build_distance_regression_bundle()
        with torch.no_grad():
            t_torch = be.to_numpy(geometry.distance(rays))

    assert_allclose(t_numpy, t_torch, rtol=0.0, atol=1e-12)


def test_numpy_off_axis_trace_matches_torch_no_grad():
    torch = pytest.importorskip("torch")

    with backend_state("numpy", precision="float64"):
        lens_np = AsphericSinglet()
        lens_np.fields.add(y=5.0)
        lens_np.trace(0.2, 0.3, 0.587, 24, "hexapolar")
        img_np = lens_np.surfaces.num_surfaces - 1
        x_np = be.to_numpy(lens_np.surfaces.x[img_np, :])
        y_np = be.to_numpy(lens_np.surfaces.y[img_np, :])

    with backend_state("torch", precision="float64"):
        lens_t = AsphericSinglet()
        lens_t.fields.add(y=5.0)
        with torch.no_grad():
            lens_t.trace(0.2, 0.3, 0.587, 24, "hexapolar")
        img_t = lens_t.surfaces.num_surfaces - 1
        x_t = be.to_numpy(lens_t.surfaces.x[img_t, :])
        y_t = be.to_numpy(lens_t.surfaces.y[img_t, :])

    assert_allclose(x_np, x_t, rtol=0.0, atol=3e-7)
    assert_allclose(y_np, y_t, rtol=0.0, atol=3e-7)
