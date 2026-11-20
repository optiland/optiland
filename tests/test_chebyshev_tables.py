"""Tests for the Chebyshev geometry with tables of T_0 to T_n.

``ChebyshevPolynomialGeometry`` takes T_0 to T_n of each coordinate from one
pass of the recurrence per call, and the sag and the slopes share these
values in the Newton-Raphson loop. Before, each term started its own
recurrence. The tests compare the new values bit for bit with a copy of the
former evaluation (``_former_sag`` and ``_former_slopes``).
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import ChebyshevPolynomialGeometry
from optiland.geometries import newton_raphson as nr
from optiland.rays import RealRays

from .utils import assert_bits_equal

NORM = 10.0
_RNG = np.random.default_rng(7)
ARRAYS = {
    "sparse 6 x 6": np.where(
        _RNG.random((6, 6)) < 0.3, _RNG.normal(0.0, 1e-3, (6, 6)), 0.0
    ),
    "dense 4 x 4": _RNG.normal(0.0, 1e-3, (4, 4)),
    "row 1 x 5": np.array([[1e-3, 0.0, -2e-4, 0.0, 5e-5]]),
    "column 3 x 1": np.array([[0.0], [2e-3], [-1e-4]]),
    "all zero": np.zeros((3, 3)),
}


def _geometry(coefficients, tol=1e-10):
    return ChebyshevPolynomialGeometry(
        CoordinateSystem(),
        radius=-45.0,
        conic=-0.3,
        coefficients=be.array(coefficients),
        norm_x=NORM,
        norm_y=NORM,
        tol=tol,
    )


def _t(n, x):
    t_prev = be.ones_like(x)
    if n == 0:
        return t_prev
    t_curr = x
    for _ in range(n - 1):
        t_prev, t_curr = t_curr, 2.0 * x * t_curr - t_prev
    return t_curr


def _dt(n, x):
    if n == 0:
        return be.zeros_like(x)
    u_prev = be.ones_like(x)
    if n == 1:
        return u_prev
    u_curr = 2.0 * x
    for _ in range(n - 2):
        u_prev, u_curr = u_curr, 2.0 * x * u_curr - u_prev
    return n * u_curr


def _terms(geometry):
    values = be.to_numpy(geometry.coefficients).tolist()
    return [(i, j) for i, row in enumerate(values) for j, v in enumerate(row) if v]


def _former_sag(geometry, x, y):
    """The sag of the former code: one recurrence for each term."""
    xn, yn = x / geometry.norm_x, y / geometry.norm_y
    r2 = x**2 + y**2
    root = be.sqrt(1 - (1 + geometry.k) * r2 / geometry.radius**2)
    z = r2 / (geometry.radius * (1 + root))
    for i, j in _terms(geometry):
        z = z + geometry.coefficients[i, j] * _t(i, xn) * _t(j, yn)
    return z


def _former_slopes(geometry, x, y):
    """The slopes of the former code: one recurrence for each term."""
    xn, yn = x / geometry.norm_x, y / geometry.norm_y
    r2 = x**2 + y**2
    denom = geometry.radius * be.sqrt(1 - (1 + geometry.k) * r2 / geometry.radius**2)
    dzdx, dzdy = x / denom, y / denom
    for i, j in _terms(geometry):
        c = geometry.coefficients[i, j]
        dzdx = dzdx + (_dt(i, xn) / geometry.norm_x * c * _t(j, yn))
        dzdy = dzdy + (_dt(j, yn) / geometry.norm_y * c * _t(i, xn))
    return dzdx, dzdy


def _points(n=400, seed=11):
    rng = np.random.default_rng(seed)
    return be.array(rng.uniform(-9.0, 9.0, n)), be.array(rng.uniform(-9.0, 9.0, n))


def _np(value):
    return np.asarray(be.to_numpy(value))


def _no_grad():
    if be.get_backend() == "torch":
        import torch

        return torch.no_grad()
    return contextlib.nullcontext()


@pytest.mark.parametrize("name", sorted(ARRAYS))
def test_sag_and_slopes_keep_the_bits_of_the_former_code(set_test_backend, name):
    geometry = _geometry(ARRAYS[name])
    x, y = _points()
    with _no_grad():
        assert_bits_equal(_np(geometry.sag(x, y)), _np(_former_sag(geometry, x, y)))
        slopes = geometry._slopes_from_terms(x, y, geometry._terms(x, y))
        for new, old in zip(slopes, _former_slopes(geometry, x, y), strict=True):
            assert_bits_equal(_np(new), _np(old))
        normal = geometry._surface_normal(x, y)
        dzdx, dzdy = _former_slopes(geometry, x, y)
        norm = be.sqrt(dzdx**2 + dzdy**2 + 1)
        for new, old in zip(normal, (dzdx / norm, dzdy / norm, -1 / norm), strict=True):
            assert_bits_equal(_np(new), _np(old))


def test_one_pass_of_the_recurrence_per_coordinate(set_test_backend, monkeypatch):
    """No term starts its own recurrence."""

    def fail(self, n, x):
        raise AssertionError("a term started its own recurrence")

    monkeypatch.setattr(ChebyshevPolynomialGeometry, "_chebyshev", fail)
    monkeypatch.setattr(ChebyshevPolynomialGeometry, "_chebyshev_derivative", fail)
    calls = []
    original = ChebyshevPolynomialGeometry._chebyshev_values

    def counted(n_max, x):
        calls.append(n_max)
        return original(n_max, x)

    monkeypatch.setattr(
        ChebyshevPolynomialGeometry, "_chebyshev_values", staticmethod(counted)
    )
    geometry = _geometry(ARRAYS["sparse 6 x 6"])
    x, y = _points()
    with _no_grad():
        geometry.sag(x, y)
        assert len(calls) == 2
        geometry._surface_normal(x, y)
        assert len(calls) == 4


def _bundle(n=301):
    y = np.linspace(-6.0, 6.0, n)
    x = 2.0 * np.sin(5.0 * y)
    L = np.full(n, 0.03)
    M = np.full(n, -0.06)
    N = np.sqrt(1.0 - L**2 - M**2)
    return RealRays(x, y, np.full(n, -5.0), L, M, N, np.ones(n), np.full(n, 0.55))


@pytest.mark.parametrize("name", ["sparse 6 x 6", "dense 4 x 4"])
def test_the_newton_loop_shares_the_tables(set_test_backend, monkeypatch, name):
    """The loop with terms finds the roots of the loop with sag and normal."""
    geometry = _geometry(ARRAYS[name], tol=1e-12)
    assert nr._shares_terms(geometry)
    with _no_grad():
        shared = geometry._solve_distance_primal(_bundle())
        monkeypatch.setattr(nr, "_shares_terms", lambda geometry: False)
        plain = geometry._solve_distance_primal(_bundle())
    assert shared.final is not None
    assert plain.final is None
    assert bool(np.all(_np(shared.converged)))
    assert bool(np.all(_np(plain.converged)))
    np.testing.assert_allclose(_np(shared.t), _np(plain.t), rtol=0, atol=1e-12)


def test_gradients_of_the_coefficients_through_the_loop():
    """The shared loop keeps the gradients of the Chebyshev coefficients."""
    torch = pytest.importorskip("torch")
    from .nr_implicit_test_utils import backend_state

    with backend_state("torch", "float64"):
        values = torch.tensor(ARRAYS["dense 4 x 4"], dtype=torch.float64)
        grads = []
        for shares in (True, False):
            leaf = values.clone().requires_grad_(True)
            geometry = _geometry(leaf, tol=1e-12)
            geometry.coefficients = leaf
            with pytest.MonkeyPatch.context() as patch:
                if not shares:
                    patch.setattr(nr, "_shares_terms", lambda geometry: False)
                t = geometry.distance(_bundle())
                (grad,) = torch.autograd.grad(t.sum(), leaf)
            grads.append(grad)
        torch.testing.assert_close(grads[0], grads[1], rtol=1e-12, atol=1e-15)
