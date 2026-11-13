"""The Newton-Raphson loop with shared sag and slope terms.

A geometry that overrides ``_sag_and_terms`` and ``_slopes_from_terms``, such
as ``EvenAsphere``, gets the sag and the slopes at an iterate from one set of
terms, and its slopes enter dF/dt directly. Other geometries keep the loop
with ``sag`` and ``_surface_normal``.
"""

from __future__ import annotations

import contextlib
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import EvenAsphere
from optiland.geometries import newton_raphson as nr
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry
from optiland.rays import RealRays

from .nr_implicit_test_utils import backend_state

COEFFICIENTS = [-2.2e-4, -4.7e-6, 1e-8]


def _asphere(coefficients=COEFFICIENTS, **kwargs):
    return EvenAsphere(
        CoordinateSystem(),
        radius=20.0,
        conic=-0.35,
        coefficients=list(coefficients),
        **kwargs,
    )


def _bundle(n=401, height=4.0, miss=False):
    """Rays from z = -5, off the y-z plane. With ``miss``, some miss the conic."""
    y = np.linspace(-height, height, n)
    if miss:
        y[::50] = 30.0  # outside the base conic: no conic seed
    x = 0.3 * np.sin(7.0 * y)
    L = np.full(n, 0.02)
    M = np.full(n, 0.08)
    N = np.sqrt(1.0 - L**2 - M**2)
    return RealRays(x, y, np.full(n, -5.0), L, M, N, np.ones(n), np.full(n, 0.55))


def _no_grad():
    if be.get_backend() == "torch":
        import torch

        return torch.no_grad()
    return contextlib.nullcontext()


def _np(value):
    return np.asarray(be.to_numpy(value))


def _same_bits(a, b):
    a, b = _np(a), _np(b)
    assert a.dtype == b.dtype
    np.testing.assert_array_equal(a.view(np.int64), b.view(np.int64))


class _PlainSurface(NewtonRaphsonGeometry):
    """A plugin geometry with only ``sag`` and ``_surface_normal``."""

    def __init__(self):
        super().__init__(CoordinateSystem(), radius=be.inf, tol=1e-10)

    def sag(self, x=0, y=0):
        return 1e-3 * (x**2 + y**2)

    def _surface_normal(self, x, y):
        dfdx, dfdy = 2e-3 * x, 2e-3 * y
        mag = be.sqrt(dfdx**2 + dfdy**2 + 1)
        return dfdx / mag, dfdy / mag, -1 / mag


class _TermsWithoutSlopes(_PlainSurface):
    """Overrides only _sag_and_terms: there are no slopes from terms."""

    def _sag_and_terms(self, x, y):
        return self.sag(x, y), None


class _NormalOnly(EvenAsphere):
    """Overrides the normal, not the terms: another surface than its terms."""

    def _surface_normal(self, x, y):
        return super()._surface_normal(x, y)


class _OwnSagAndNormal(EvenAsphere):
    """Overrides sag and the normal, but not the two methods of the terms.

    The terms of EvenAsphere would then describe another surface, so the
    loop must call sag and the normal.
    """

    def sag(self, x=0, y=0):
        return EvenAsphere.sag(self, x, y)

    def _surface_normal(self, x, y):
        return EvenAsphere._surface_normal(self, x, y)


class _Inherits(EvenAsphere):
    """Overrides nothing: the terms of EvenAsphere describe it."""


@pytest.mark.parametrize("coefficients", [[], [1e-3], COEFFICIENTS])
def test_sag_and_normal_keep_the_expressions_of_master(set_test_backend, coefficients):
    """The sag and the normal of EvenAsphere are the expressions of before."""
    geometry = _asphere(coefficients)
    coefficients = geometry.coefficients
    x = be.linspace(-5.0, 5.0, 201)
    y = be.linspace(3.0, -4.0, 201)
    with _no_grad():
        r2 = x**2 + y**2
        root = be.sqrt(1 - (1 + geometry.k) * r2 / geometry.radius**2)
        sag = r2 / (
            geometry.radius
            * (1 + be.sqrt(1 - (1 + geometry.k) * r2 / geometry.radius**2))
        )
        dfdx = x / (geometry.radius * root)
        dfdy = y / (geometry.radius * root)
        n = len(coefficients)
        if n:
            polynomial = coefficients[n - 1]
            for i in range(n - 2, -1, -1):
                polynomial = coefficients[i] + r2 * polynomial
            sag = sag + r2 * polynomial
            derivative = n * coefficients[n - 1]
            for i in range(n - 1, 0, -1):
                derivative = i * coefficients[i - 1] + r2 * derivative
            dfdx = dfdx + 2 * x * derivative
            dfdy = dfdy + 2 * y * derivative
        mag = be.sqrt(dfdx**2 + dfdy**2 + 1)

        _same_bits(geometry.sag(x, y), sag)
        terms_sag, (terms_r2, terms_root) = geometry._sag_and_terms(x, y)
        _same_bits(terms_sag, sag)
        _same_bits(terms_r2, r2)
        _same_bits(terms_root, root)
        slopes = geometry._slopes_from_terms(x, y, (terms_r2, terms_root))
        _same_bits(slopes[0], dfdx)
        _same_bits(slopes[1], dfdy)
        for value, expected in zip(
            geometry._surface_normal(x, y),
            (dfdx / mag, dfdy / mag, -1 / mag),
            strict=True,
        ):
            _same_bits(value, expected)


def test_slopes_of_the_terms_are_the_derivatives_of_the_sag(set_test_backend):
    geometry = _asphere()
    x = be.linspace(-5.0, 5.0, 101)
    y = be.linspace(3.0, -4.0, 101)
    h = 1e-6
    with _no_grad():
        _, terms = geometry._sag_and_terms(x, y)
        dfdx, dfdy = geometry._slopes_from_terms(x, y, terms)
        fd_x = (geometry.sag(x + h, y) - geometry.sag(x - h, y)) / (2 * h)
        fd_y = (geometry.sag(x, y + h) - geometry.sag(x, y - h)) / (2 * h)
    np.testing.assert_allclose(_np(dfdx), _np(fd_x), rtol=0, atol=1e-9)
    np.testing.assert_allclose(_np(dfdy), _np(fd_y), rtol=0, atol=1e-9)


def test_which_geometries_share_their_terms():
    assert nr._shares_terms(_asphere())
    assert nr._shares_terms(_Inherits(CoordinateSystem(), radius=20.0))
    # A subclass of EvenAsphere with its own sag and normal.
    assert not nr._shares_terms(_OwnSagAndNormal(CoordinateSystem(), radius=20.0))
    assert not nr._shares_terms(_NormalOnly(CoordinateSystem(), radius=20.0))
    assert not nr._shares_terms(_PlainSurface())
    assert not nr._shares_terms(_TermsWithoutSlopes())
    mocked = _asphere()
    mocked.sag = lambda x=0, y=0: x * 0.0
    assert not nr._shares_terms(mocked)


def test_the_decision_follows_a_method_patched_on_a_class(monkeypatch):
    """The decision of a class is kept, but a patched method renews it."""
    geometry = _Inherits(CoordinateSystem(), radius=20.0)
    assert nr._shares_terms(geometry)
    monkeypatch.setattr(_Inherits, "sag", lambda self, x=0, y=0: x * 0.0)
    assert not nr._shares_terms(geometry)
    monkeypatch.undo()
    assert nr._shares_terms(geometry)


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
@pytest.mark.parametrize("miss", [False, True], ids=["hits", "with_misses"])
def test_the_loop_with_terms_finds_the_roots_of_the_loop_without(
    set_test_backend, monkeypatch, miss
):
    geometry = _asphere(tol=1e-10)
    with _no_grad():
        shared = geometry._solve_distance_primal(_bundle(miss=miss))
        monkeypatch.setattr(nr, "_shares_terms", lambda geometry: False)
        plain = geometry._solve_distance_primal(_bundle(miss=miss))
    assert shared.final is not None
    assert plain.final is None
    assert shared.iterations == plain.iterations
    np.testing.assert_array_equal(_np(shared.converged), _np(plain.converged))
    t_shared, t_plain = _np(shared.t), _np(plain.t)
    np.testing.assert_array_equal(np.isnan(t_shared), np.isnan(t_plain))
    np.testing.assert_allclose(t_shared, t_plain, rtol=0, atol=1e-12, equal_nan=True)
    assert np.all(np.abs(_np(shared.residual))[_np(shared.converged)] < 1e-10)


def test_the_slopes_are_evaluated_only_while_a_ray_is_not_converged(
    set_test_backend, monkeypatch
):
    calls = {"sag": 0, "slopes": 0, "normal": 0}

    def counted(name, method):
        def wrapper(self, *args):
            calls[name] += 1
            return method(self, *args)

        return wrapper

    monkeypatch.setattr(
        EvenAsphere, "_sag_and_terms", counted("sag", EvenAsphere._sag_and_terms)
    )
    monkeypatch.setattr(
        EvenAsphere,
        "_slopes_from_terms",
        counted("slopes", EvenAsphere._slopes_from_terms),
    )
    monkeypatch.setattr(
        EvenAsphere, "_surface_normal", counted("normal", EvenAsphere._surface_normal)
    )
    with _no_grad():
        result = _asphere(tol=1e-10)._solve_distance_primal(_bundle())
        assert result.iterations > 0
        assert calls == {
            "sag": result.iterations + 1,
            "slopes": result.iterations,
            "normal": 0,
        }

        # Without coefficients the conic seed is the root: no step, no slopes.
        for name in calls:
            calls[name] = 0
        result = _asphere([], tol=1e-10)._solve_distance_primal(_bundle())
        assert result.iterations == 0
        assert calls == {"sag": 1, "slopes": 0, "normal": 0}


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
def test_root_classification_from_the_final_terms(set_test_backend):
    """The masks are those of a new normal evaluation at the root."""
    geometry = _asphere(tol=1e-10)
    rays = _bundle(miss=True)
    with _no_grad():
        result = geometry._solve_distance_primal(rays)
        from_terms = geometry._classify_final_roots(result, rays)
        plain = dataclasses.replace(result, final=None)
        from_normal = geometry._classify_final_roots(plain, rays)
    for name in (
        "regular",
        "nonfinite",
        "nonconverged",
        "nz_singular",
        "near_singular",
    ):
        np.testing.assert_array_equal(
            _np(getattr(from_terms, name)), _np(getattr(from_normal, name)), name
        )
    regular = _np(from_terms.regular)
    assert regular.any() and not regular.all()
    np.testing.assert_allclose(
        _np(from_terms.df_dt)[regular], _np(from_normal.df_dt)[regular], rtol=1e-13
    )


@pytest.mark.parametrize(
    ("dtype", "regular", "singular"),
    [("float64", 1e13, 1e15), ("float32", 1e5, 1e6)],
)
def test_slope_validity_is_the_test_of_the_unit_normal(dtype, regular, singular):
    """|n_z| > tau of the unit normal, as a test on the slopes."""
    fx = np.array([0.0, 0.3, -2.0, regular, -regular, singular, np.inf, np.nan], dtype)
    fy = np.array([0.0, -0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype)
    one = np.ones_like(fx)
    rays = SimpleNamespace(L=0.1 * one, M=-0.2 * one, N=np.sqrt(one - 0.05))
    df_dt, scale, nz_regular = nr._residual_dt_from_slopes(fx, fy, rays)

    with np.errstate(invalid="ignore"):
        mag = np.sqrt(fx**2 + fy**2 + 1)
        nz = -1 / mag
    expected = np.isfinite(nz) & (np.abs(nz) > nr._nz_threshold(nz))
    np.testing.assert_array_equal(nz_regular, expected)
    assert expected[:5].all() and not expected[5:].any()
    with np.errstate(invalid="ignore"):
        np.testing.assert_array_equal(df_dt, fx * rays.L + fy * rays.M - rays.N)
        np.testing.assert_array_equal(
            scale, np.abs(fx * rays.L) + np.abs(fy * rays.M) + np.abs(rays.N)
        )


def test_geometries_without_terms_keep_the_loop_with_sag_and_normal(monkeypatch):
    def fail(*args):
        raise AssertionError("the loop used the terms")

    monkeypatch.setattr(NewtonRaphsonGeometry, "_iterate_with_terms", fail)
    for geometry in (
        _PlainSurface(),
        _OwnSagAndNormal(CoordinateSystem(), radius=20.0, coefficients=[1e-4, 1e-6]),
    ):
        result = geometry._solve_distance_primal(_bundle())
        assert result.final is None
        assert bool(np.all(_np(result.converged)))


class TestTorchGradients:
    @pytest.fixture(autouse=True)
    def _torch(self):
        pytest.importorskip("torch")
        with backend_state("torch", "float64"):
            yield

    def _geometry(self):
        import torch

        geometry = _asphere(tol=1e-12)
        geometry.radius = torch.nn.Parameter(torch.tensor(20.0, dtype=torch.float64))
        geometry.coefficients = [
            torch.nn.Parameter(torch.tensor(c, dtype=torch.float64))
            for c in COEFFICIENTS
        ]
        return geometry

    def _gradients(self, geometry):
        t = geometry.distance(_bundle(n=61))
        t.sum().backward()
        return [geometry.radius.grad] + [c.grad for c in geometry.coefficients]

    def test_gradients_match_the_loop_without_terms(self, monkeypatch):
        """The implicit gradients of both loops agree to 1e-12."""
        shared = self._gradients(self._geometry())
        monkeypatch.setattr(nr, "_shares_terms", lambda geometry: False)
        plain = self._gradients(self._geometry())
        for a, b in zip(shared, plain, strict=True):
            assert float(abs(a - b)) <= 1e-12 * float(abs(b))

    def test_distance_with_gradients_evaluates_no_normal(self, monkeypatch):
        """The root classification takes the slopes from the final terms."""
        calls = []
        normal = EvenAsphere._surface_normal

        def counted(self, x, y):
            calls.append(1)
            return normal(self, x, y)

        monkeypatch.setattr(EvenAsphere, "_surface_normal", counted)
        gradients = self._gradients(self._geometry())
        assert calls == []
        assert all(float(abs(g)) > 0 for g in gradients)
