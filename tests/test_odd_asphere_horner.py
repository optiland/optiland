"""Tests for the odd asphere with Horner's method in r.

``OddAsphere`` computes the sum of C_i r^i and its slope with Horner's
method in r, and the sag and the slopes share r**2, r and the conic root in
the Newton-Raphson loop. Before, each term computed its own power of r, and
the normal set the value 0 / 0 of the C1 slope at the vertex to 0 in place.
The tests compare the values with a copy of the former evaluation and with
a reference with 50 digits (``decimal``).
"""

from __future__ import annotations

import contextlib
from decimal import Decimal, localcontext

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import OddAsphere
from optiland.geometries import newton_raphson as nr
from optiland.rays import RealRays

EPS = float(np.finfo(np.float64).eps)
# (radius, conic, coefficients C1 to Cn)
CASES = {
    "C1 to C5": (20.0, -0.5, [2e-3, -1e-3, 3e-4, -2e-5, 1e-6]),
    "C1 to C9": (
        -35.0,
        0.3,
        [1e-3, 5e-4, -2e-4, 3e-5, -4e-6, 2e-7, -1e-8, 3e-10, -4e-12],
    ),
    "C3 and C5": (50.0, 0.0, [0.0, 0.0, 4e-5, 0.0, -2e-7]),
    "C1 only": (25.0, 0.0, [3e-3]),
}


def _geometry(name, tol=1e-10):
    radius, conic, coefficients = CASES[name]
    return OddAsphere(
        CoordinateSystem(),
        radius=radius,
        conic=conic,
        coefficients=list(coefficients),
        tol=tol,
    )


def _points(n=300, seed=5):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-8.0, 8.0, n), rng.uniform(-8.0, 8.0, n)
    x[0] = y[0] = 0.0  # the vertex
    return x, y


def _np(value):
    return np.asarray(be.to_numpy(value))


def _no_grad():
    if be.get_backend() == "torch":
        import torch

        return torch.no_grad()
    return contextlib.nullcontext()


def _reference(name, x, y):
    """Sag and slopes with 50 digits."""
    radius, conic, coefficients = CASES[name]
    sags, slopes_x, slopes_y = [], [], []
    with localcontext() as context:
        context.prec = 50
        big_r, k = Decimal(radius), Decimal(conic)
        c = [Decimal(value) for value in coefficients]
        for xi, yi in zip(x, y, strict=True):
            px, py = Decimal(float(xi)), Decimal(float(yi))
            r2 = px * px + py * py
            r = r2.sqrt()
            root = (1 - (1 + k) * r2 / (big_r * big_r)).sqrt()
            sags.append(
                r2 / (big_r * (1 + root))
                + sum(ci * r ** (i + 1) for i, ci in enumerate(c))
            )
            # g = (dz/dr) / r of the polynomial; the C1 part is 0 at the vertex.
            g = sum(
                (i + 1) * ci * (r ** (i - 1) if i > 1 else 1)
                for i, ci in enumerate(c)
                if i > 0
            )
            if r > 0:
                g += c[0] / r
            slopes_x.append(px / (big_r * root) + px * g)
            slopes_y.append(py / (big_r * root) + py * g)
    return sags, slopes_x, slopes_y


def _errors(values, reference):
    return np.array(
        [
            abs(float(Decimal(float(v)) - ref))
            for v, ref in zip(values, reference, strict=True)
        ]
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_values_against_a_reference_with_50_digits(name):
    be.set_backend("numpy")
    geometry = _geometry(name)
    x, y = _points()
    sag = geometry.sag(x, y)
    slopes = geometry._slopes_from_terms(x, y, geometry._odd_terms(x, y))
    reference = _reference(name, x, y)
    scale = np.maximum(1.0, np.abs(sag))
    assert np.all(_errors(sag, reference[0]) <= 4 * EPS * scale)
    for value, ref in zip(slopes, reference[1:], strict=True):
        assert np.all(_errors(value, ref) <= 4 * EPS * np.maximum(1.0, np.abs(value)))


def test_the_slope_of_the_c1_term_is_zero_at_the_vertex(set_test_backend):
    geometry = _geometry("C1 only")
    zero = be.array([0.0])
    with _no_grad():
        nx, ny, nz = geometry._surface_normal(zero, zero)
    np.testing.assert_array_equal(_np(nx), [0.0])
    np.testing.assert_array_equal(_np(ny), [0.0])
    np.testing.assert_array_equal(_np(nz), [-1.0])
    with _no_grad():
        assert float(_np(geometry.sag())) == 0.0


def test_the_gradients_at_the_vertex_are_finite():
    """A ray at the vertex gets finite gradients, also for C1 and the position.

    The former code gave NaN for both: x C1 / r is 0 / 0 at the vertex, and
    sqrt(r**2) has an infinite derivative there.
    """
    torch = pytest.importorskip("torch")
    from .nr_implicit_test_utils import backend_state

    with backend_state("torch", "float64"):
        c1 = torch.tensor(3e-3, dtype=torch.float64, requires_grad=True)
        geometry = OddAsphere(CoordinateSystem(), radius=25.0, coefficients=[c1, 1e-4])
        x = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0.0, 2.0], dtype=torch.float64, requires_grad=True)
        nx, ny, nz = geometry._surface_normal(x, y)
        grads = torch.autograd.grad((nx + ny + nz).sum(), (c1, x, y))
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        # vmap needs code without in-place writes.
        mapped = torch.func.vmap(lambda a, b: geometry._surface_normal(a, b)[0])(
            x.detach(), y.detach()
        )
        torch.testing.assert_close(mapped, nx.detach(), rtol=0, atol=0)


def _bundle(n=301):
    y = np.linspace(-6.0, 6.0, n)
    x = 2.0 * np.sin(5.0 * y)
    L = np.full(n, 0.03)
    M = np.full(n, -0.06)
    N = np.sqrt(1.0 - L**2 - M**2)
    return RealRays(x, y, np.full(n, -5.0), L, M, N, np.ones(n), np.full(n, 0.55))


@pytest.mark.parametrize("name", ["C1 to C5", "C3 and C5"])
def test_the_newton_loop_shares_the_terms(set_test_backend, monkeypatch, name):
    """The loop with terms finds the roots of the loop with sag and normal."""
    geometry = _geometry(name, tol=1e-12)
    assert nr._shares_terms(geometry)
    with _no_grad():
        shared = geometry._solve_distance_primal(_bundle())
        monkeypatch.setattr(nr, "_shares_terms", lambda geometry: False)
        plain = geometry._solve_distance_primal(_bundle())
    assert shared.final is not None
    assert bool(np.all(_np(shared.converged)))
    assert bool(np.all(_np(plain.converged)))
    np.testing.assert_allclose(_np(shared.t), _np(plain.t), rtol=0, atol=1e-12)


def test_gradients_of_the_coefficients_through_the_loop():
    torch = pytest.importorskip("torch")
    from .nr_implicit_test_utils import backend_state

    with backend_state("torch", "float64"):
        values = torch.tensor(CASES["C1 to C5"][2], dtype=torch.float64)
        grads = []
        for shares in (True, False):
            leaf = values.clone().requires_grad_(True)
            geometry = OddAsphere(
                CoordinateSystem(), radius=20.0, conic=-0.5, coefficients=leaf
            )
            geometry.tol = 1e-12
            with pytest.MonkeyPatch.context() as patch:
                if not shares:
                    patch.setattr(nr, "_shares_terms", lambda geometry: False)
                t = geometry.distance(_bundle())
                (grad,) = torch.autograd.grad(t.sum(), leaf)
            grads.append(grad)
        torch.testing.assert_close(grads[0], grads[1], rtol=1e-12, atol=1e-15)
