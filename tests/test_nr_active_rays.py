"""The Newton-Raphson loop goes on with the active rays only.

At the steps of ``_ACTIVE_CHECKS``, the loop counts its active rays. If at
most the part ``_ACTIVE_ONLY_FRACTION`` of at least ``_ACTIVE_ONLY_MIN_RAYS``
rays is active, it goes on with these rays only, so a few rays that converge
slowly, or not at all, do not hold the whole bundle. Each ray takes the same
steps, so the results keep their bits. The tests force the continuation at
early steps and compare it with the loop over all rays.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    BiconicGeometry,
    ChebyshevPolynomialGeometry,
    EvenAsphere,
    OddAsphere,
    PolynomialGeometry,
    ToroidalGeometry,
    ZernikePolynomialGeometry,
)
from optiland.geometries import newton_raphson as nr
from optiland.geometries.forbes import (
    ForbesQ2dGeometry,
    ForbesQNormalSlopeGeometry,
    ForbesSurfaceConfig,
)
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry
from optiland.rays import RealRays

from .utils import assert_bits_equal


def _cs():
    return CoordinateSystem()


GEOMETRIES = {
    "even asphere": lambda: EvenAsphere(
        _cs(), radius=15.0, coefficients=[0.0, 1.5e-4, 0.0], tol=1e-10
    ),
    "odd asphere": lambda: OddAsphere(
        _cs(), radius=20.0, conic=-0.5, coefficients=[2e-3, -1e-3, 3e-4]
    ),
    "zernike": lambda: ZernikePolynomialGeometry(
        _cs(),
        radius=30.0,
        coefficients=[0.0, 0.0, 0.0, 0.0, 1e-2, 5e-3, -4e-3],
        norm_radius=10.0,
    ),
    "xy polynomial": lambda: PolynomialGeometry(
        _cs(),
        radius=-45.0,
        coefficients=[[0.0, 0.0, 2e-4], [0.0, 1e-5, 0.0], [3e-4, 0.0, -2e-6]],
    ),
    "chebyshev": lambda: ChebyshevPolynomialGeometry(
        _cs(),
        radius=-45.0,
        coefficients=[[0.0, 8e-3, -2e-3], [1.5e-2, 0.0, 1e-3]],
        norm_x=10.0,
        norm_y=10.0,
    ),
    "toroidal": lambda: ToroidalGeometry(
        _cs(), radius_x=20.0, radius_y=30.0, coeffs_poly_y=[1e-4]
    ),
    "biconic": lambda: BiconicGeometry(
        _cs(), radius_x=20.0, radius_y=30.0, conic_x=-0.5
    ),
    "forbes qbfs": lambda: ForbesQNormalSlopeGeometry(
        _cs(),
        ForbesSurfaceConfig(
            radius=40.0, conic=-0.5, terms={0: 2e-2, 1: -1e-2}, norm_radius=8.0
        ),
    ),
    "forbes q2d": lambda: ForbesQ2dGeometry(
        _cs(),
        ForbesSurfaceConfig(
            radius=40.0,
            terms={("a", 0, 0): 2e-2, ("a", 1, 1): 1e-2, ("b", 2, 0): -5e-3},
            norm_radius=8.0,
        ),
    ),
}


def _bundle(n=301):
    """Rays from z = -5 towards the surface, off the y-z plane."""
    y = np.linspace(-6.0, 6.0, n)
    x = 2.0 * np.sin(5.0 * y)
    L = np.full(n, 0.03)
    M = np.full(n, -0.06)
    N = np.sqrt(1.0 - L**2 - M**2)
    return RealRays(x, y, np.full(n, -5.0), L, M, N, np.ones(n), np.full(n, 0.55))


def _fan(n=401):
    """Parallel rays at 20 deg across the strong even asphere: some stall."""
    y = np.linspace(-9.0, 9.0, n)
    angle = np.deg2rad(20.0)
    return RealRays(
        np.zeros(n),
        y,
        np.full(n, -5.0),
        np.zeros(n),
        np.full(n, np.sin(angle)),
        np.full(n, np.cos(angle)),
        np.ones(n),
        np.full(n, 0.5876),
    )


def _np(value):
    return np.asarray(be.to_numpy(value))


def _no_grad():
    if be.get_backend() == "torch":
        import torch

        return torch.no_grad()
    return contextlib.nullcontext()


@pytest.fixture
def continuations(monkeypatch):
    """Count the continuations on the active rays."""
    calls = []
    original = NewtonRaphsonGeometry._iterate_active_rays

    def counted(self, *args, **kwargs):
        calls.append(type(self).__name__)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NewtonRaphsonGeometry, "_iterate_active_rays", counted)
    return calls


def _force(monkeypatch):
    """Go on with the active rays from step 1 on, for any number of rays."""
    monkeypatch.setattr(nr, "_ACTIVE_CHECKS", frozenset({1, 2, 4, 8, 16, 32, 64}))
    monkeypatch.setattr(nr, "_ACTIVE_ONLY_FRACTION", 1.0)
    monkeypatch.setattr(nr, "_ACTIVE_ONLY_MIN_RAYS", 1)


def _compare(geometry, rays, monkeypatch):
    monkeypatch.setattr(nr, "_ACTIVE_CHECKS", frozenset())
    with _no_grad():
        everyone = geometry._solve_distance_primal(rays)
    _force(monkeypatch)
    with _no_grad():
        active = geometry._solve_distance_primal(rays)
    assert_bits_equal(_np(active.t), _np(everyone.t))
    np.testing.assert_array_equal(_np(active.converged), _np(everyone.converged))
    np.testing.assert_array_equal(_np(active.reached), _np(everyone.reached))
    finite = np.isfinite(_np(everyone.residual))
    np.testing.assert_array_equal(np.isfinite(_np(active.residual)), finite)
    assert_bits_equal(_np(active.residual)[finite], _np(everyone.residual)[finite])
    assert active.iterations == everyone.iterations
    return everyone, active


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_the_active_rays_take_the_same_steps(
    set_test_backend, monkeypatch, continuations, name
):
    everyone, _ = _compare(GEOMETRIES[name](), _bundle(), monkeypatch)
    assert everyone.iterations >= 2
    assert continuations


def test_stalled_rays_go_on_alone(set_test_backend, monkeypatch, continuations):
    """On the strong asphere, a few rays stall; the rest of the bundle stops."""
    everyone, _ = _compare(GEOMETRIES["even asphere"](), _fan(), monkeypatch)
    assert not bool(np.all(_np(everyone.reached)))
    assert continuations


def test_the_default_checks_select_few_active_rays(monkeypatch, continuations):
    """With the default settings, the stalled rays of the fan go on alone."""
    be.set_backend("numpy")
    geometry = GEOMETRIES["even asphere"]()
    result = geometry._solve_distance_primal(_fan())
    stalled = int(np.sum(~_np(result.reached)))
    assert 0 < stalled <= nr._ACTIVE_ONLY_FRACTION * 401
    assert continuations == ["EvenAsphere"]


def test_the_selection_needs_enough_rays_on_the_cpu():
    be.set_backend("numpy")
    enough = np.zeros(nr._ACTIVE_ONLY_MIN_RAYS, dtype=bool)
    assert nr._active_only_pays(enough)
    assert not nr._active_only_pays(enough[:-1])


def test_no_checks_on_a_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from .nr_implicit_test_utils import backend_state

    with backend_state("torch", "float64"):
        done = torch.zeros(nr._ACTIVE_ONLY_MIN_RAYS, dtype=torch.bool, device="cuda")
        assert not nr._active_only_pays(done)
        assert nr._active_only_pays(done.cpu())


class _PlainAsphere(EvenAsphere):
    """An even asphere that runs the loop with sag and normal."""

    def sag(self, x=0, y=0):
        return EvenAsphere.sag(self, x, y)

    def _surface_normal(self, x, y):
        return EvenAsphere._surface_normal(self, x, y)


GRADIENT_GEOMETRIES = {
    "even asphere": GEOMETRIES["even asphere"],
    "odd asphere": GEOMETRIES["odd asphere"],
    "xy polynomial": GEOMETRIES["xy polynomial"],
    "loop with sag and normal": lambda: _PlainAsphere(
        _cs(), radius=15.0, coefficients=[0.0, 1.5e-4, 0.0], tol=1e-10
    ),
}


@pytest.mark.parametrize("name", sorted(GRADIENT_GEOMETRIES))
def test_gradients_keep_their_bits(monkeypatch, name):
    """With gradients, the root classification sees the same values."""
    torch = pytest.importorskip("torch")
    from .nr_implicit_test_utils import backend_state

    with backend_state("torch", "float64"):
        results = []
        for force in (False, True):
            geometry = GRADIENT_GEOMETRIES[name]()
            geometry.radius = torch.tensor(
                float(geometry.radius), dtype=torch.float64, requires_grad=True
            )
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(nr, "_ACTIVE_CHECKS", frozenset())
                if force:
                    _force(patch)
                t = geometry.distance(_bundle())
                (grad,) = torch.autograd.grad(torch.nansum(t), geometry.radius)
            results.append((t.detach(), grad))
        assert_bits_equal(_np(results[0][0]), _np(results[1][0]))
        assert_bits_equal(_np(results[0][1]), _np(results[1][1]))
