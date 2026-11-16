"""Tests for the nested Horner evaluation of the XY polynomial geometry.

PolynomialGeometry computes its sag and its slopes with nested Horner sums
(see ``optiland/geometries/polynomial.py``). The tests compare the new values
with:

- the former evaluation of master 4f4f514f (``_master_sag`` and
  ``_master_slopes`` below, a copy with powers of x and y);
- finite differences of the sag;
- a reference with 50 digits that needs only the standard library:
  ``fractions.Fraction`` for the polynomial and ``decimal`` for the square
  root of the conic. The new error must not be larger than the error of the
  former code.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

import optiland.backend as be
import optiland.geometries.polynomial as polynomial_module
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import EvenAsphere, PolynomialGeometry
from optiland.geometries import newton_raphson as nr
from optiland.rays import RealRays

from .utils import assert_bits_equal

EPS = float(np.finfo(np.float64).eps)

# The XY polynomial of the performance gate and of the equivalence corpus:
# a sparse 5 x 5 array.
SPARSE = [
    [0.0, 0.0, 2e-4, 0.0, -1e-6],
    [0.0, 1e-5, 0.0, 2e-7, 0.0],
    [3e-4, 0.0, -2e-6, 0.0, 0.0],
    [0.0, 4e-7, 0.0, 0.0, 0.0],
    [-1e-6, 0.0, 0.0, 0.0, 0.0],
]
# A dense 6 x 6 array with alternating signs.
DENSE = [[(-1) ** (i + j) * 1e-3 / 10 ** (i + j) for j in range(6)] for i in range(6)]
# Rows of different lengths. The constructor needs a rectangular array, so
# the tests assign this list after the construction.
RAGGED = [[1e-3, 0.0, -2e-4], [5e-4], [0.0, 3e-5, 0.0, 1e-6]]

# (coefficients, radius, conic) of the surfaces under test.
SURFACES = {
    "sparse": (SPARSE, -45.0, 0.0),
    "sparse-flat": (SPARSE, math.inf, 0.0),
    "dense-conic": (DENSE, 30.0, -0.7),
    "dense-flat": (DENSE, math.inf, 0.0),
    "ragged": (RAGGED, -60.0, 0.2),
}
FLAT = [name for name, spec in SURFACES.items() if math.isinf(spec[1])]
CURVED = [name for name, spec in SURFACES.items() if math.isfinite(spec[1])]


def _geometry(name, tol=1e-10):
    coefficients, radius, conic = SURFACES[name]
    geometry = PolynomialGeometry(
        CoordinateSystem(), radius=radius, conic=conic, tol=tol
    )
    if name == "ragged":
        geometry.coefficients = [list(row) for row in coefficients]
    else:
        geometry.coefficients = be.atleast_2d(coefficients)
    return geometry


def _points(count, seed=0, radius=9.0):
    """Points in a disk, with the center, the axes and the rim."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(count))
    angle = 2 * np.pi * rng.random(count)
    x = np.concatenate([r * np.cos(angle), [0.0, 0.0, radius, -radius, 1e-7]])
    y = np.concatenate([r * np.sin(angle), [0.0, -radius, 0.0, 0.0, -1e-7]])
    return x, y


def _numpy(value):
    return np.asarray(be.to_numpy(value), dtype=float)


def _no_grad():
    if be.get_backend() == "torch":
        import torch

        return torch.no_grad()
    return contextlib.nullcontext()


# --- The former evaluation (master 4f4f514f) -------------------------------


def _master_sag(geometry, x, y):
    coefficients, radius, k = geometry.coefficients, geometry.radius, geometry.k
    r2 = x**2 + y**2
    z = r2 / (radius * (1 + be.sqrt(1 - (1 + k) * r2 / radius**2)))
    for i in range(len(coefficients)):
        for j in range(len(coefficients[i])):
            z = z + coefficients[i][j] * (x**i) * (y**j)
    return z


def _master_slopes(geometry, x, y):
    coefficients, radius, k = geometry.coefficients, geometry.radius, geometry.k
    r2 = x**2 + y**2
    denom = radius * be.sqrt(1 - (1 + k) * r2 / radius**2)
    dzdx = x / denom
    dzdy = y / denom
    for i in range(1, len(coefficients)):
        for j in range(len(coefficients[i])):
            dzdx = dzdx + i * coefficients[i][j] * (x ** (i - 1)) * (y**j)
    for i in range(len(coefficients)):
        for j in range(1, len(coefficients[i])):
            dzdy = dzdy + j * coefficients[i][j] * (x**i) * (y ** (j - 1))
    return dzdx, dzdy


def _unit_normal(dzdx, dzdy):
    norm = be.sqrt(dzdx**2 + dzdy**2 + 1)
    return dzdx / norm, dzdy / norm, -1 / norm


def _slopes(geometry, x, y):
    _, terms = geometry._sag_and_terms(x, y)
    return geometry._slopes_from_terms(x, y, terms)


# --- Rounding scales and the reference with 50 digits ------------------------


def _scales(name, x, y):
    """Per point: the sums of the magnitudes of the terms of the sag and slopes.

    A sum of these terms has a rounding error of a few EPS times these sums.
    """
    coefficients, radius, conic = SURFACES[name]
    ax, ay = np.abs(x), np.abs(y)
    if math.isfinite(radius):
        r2 = x * x + y * y
        root = np.sqrt(1 - (1 + conic) * r2 / radius**2)
        sag = np.abs(r2 / (radius * (1 + root)))
        slope_x, slope_y = np.abs(x / (radius * root)), np.abs(y / (radius * root))
    else:
        sag, slope_x, slope_y = (np.zeros_like(x) for _ in range(3))
    for i, row in enumerate(coefficients):
        for j, c in enumerate(row):
            sag = sag + abs(c) * ax**i * ay**j
            if i:
                slope_x = slope_x + i * abs(c) * ax ** (i - 1) * ay**j
            if j:
                slope_y = slope_y + j * abs(c) * ax**i * ay ** (j - 1)
    return sag, slope_x, slope_y


def _depth(name):
    """The number of rows plus the length of the longest row."""
    coefficients = SURFACES[name][0]
    return len(coefficients) + max(len(row) for row in coefficients)


def _decimal(value):
    value = Fraction(value)
    return Decimal(value.numerator) / Decimal(value.denominator)


def _reference(name, x, y):
    """Sag, slopes and unit normal at the float points, with 50 digits.

    Columns: sag, dz/dx, dz/dy, nx, ny, nz.
    """
    coefficients, radius, conic = SURFACES[name]
    out = np.empty((len(x), 6))
    with localcontext() as context:
        context.prec = 50
        for n, (xf, yf) in enumerate(zip(x, y, strict=True)):
            px, py = Fraction(float(xf)), Fraction(float(yf))
            value = slope_x = slope_y = Fraction(0)
            for i, row in enumerate(coefficients):
                for j, c in enumerate(row):
                    c = Fraction(c)
                    value += c * px**i * py**j
                    if i:
                        slope_x += i * c * px ** (i - 1) * py**j
                    if j:
                        slope_y += j * c * px**i * py ** (j - 1)
            sag, dzdx, dzdy = _decimal(value), _decimal(slope_x), _decimal(slope_y)
            if math.isfinite(radius):
                r2 = px * px + py * py
                big_r = _decimal(radius)
                root = _decimal(1 - (1 + Fraction(conic)) * r2 / Fraction(radius) ** 2)
                root = root.sqrt()
                sag += _decimal(r2) / (big_r * (1 + root))
                dzdx += _decimal(px) / (big_r * root)
                dzdy += _decimal(py) / (big_r * root)
            norm = (dzdx * dzdx + dzdy * dzdy + 1).sqrt()
            out[n] = [
                float(sag),
                float(dzdx),
                float(dzdy),
                float(dzdx / norm),
                float(dzdy / norm),
                float(-1 / norm),
            ]
    return out


def _errors(name, x_np, y_np):
    """Absolute errors of the new and the former code against the reference."""
    geometry = _geometry(name)
    x, y = be.array(x_np), be.array(y_np)
    reference = _reference(name, x_np, y_np)
    with _no_grad():
        slopes = _slopes(geometry, x, y)
        new = np.column_stack(
            [_numpy(geometry.sag(x, y))]
            + [_numpy(v) for v in slopes]
            + [_numpy(v) for v in geometry._surface_normal(x, y)]
        )
        slopes = _master_slopes(geometry, x, y)
        old = np.column_stack(
            [_numpy(_master_sag(geometry, x, y))]
            + [_numpy(v) for v in slopes]
            + [_numpy(v) for v in _unit_normal(*slopes)]
        )
    return np.abs(new - reference), np.abs(old - reference), reference


# --- Against master, finite differences and the reference -------------------


@pytest.mark.parametrize("name", list(SURFACES))
def test_sag_and_slopes_match_master(set_test_backend, name):
    """New and former values agree to a few units of the rounding scale."""
    geometry = _geometry(name)
    x_np, y_np = _points(400, seed=1)
    x, y = be.array(x_np), be.array(y_np)
    scale_sag, scale_x, scale_y = _scales(name, x_np, y_np)
    factor = _depth(name) * EPS
    with _no_grad():
        new_sag = _numpy(geometry.sag(x, y))
        new_x, new_y = (_numpy(v) for v in _slopes(geometry, x, y))
        old_sag = _numpy(_master_sag(geometry, x, y))
        old_x, old_y = (_numpy(v) for v in _master_slopes(geometry, x, y))
        new_normal = [_numpy(v) for v in geometry._surface_normal(x, y)]
        old_normal = [_numpy(v) for v in _unit_normal(*_master_slopes(geometry, x, y))]
    assert np.all(np.abs(new_sag - old_sag) <= factor * scale_sag)
    assert np.all(np.abs(new_x - old_x) <= factor * scale_x)
    assert np.all(np.abs(new_y - old_y) <= factor * scale_y)
    for new, old in zip(new_normal, old_normal, strict=True):
        assert np.max(np.abs(new - old)) <= 4 * EPS


@pytest.mark.parametrize("name", ["sparse", "dense-conic", "ragged"])
def test_slopes_match_finite_differences(set_test_backend, name):
    geometry = _geometry(name)
    x, y = (be.array(v) for v in _points(200, seed=2))
    h = 1e-6
    with _no_grad():
        dzdx, dzdy = _slopes(geometry, x, y)
        fd_x = (geometry.sag(x + h, y) - geometry.sag(x - h, y)) / (2 * h)
        fd_y = (geometry.sag(x, y + h) - geometry.sag(x, y - h)) / (2 * h)
    np.testing.assert_allclose(_numpy(dzdx), _numpy(fd_x), rtol=0, atol=1e-9)
    np.testing.assert_allclose(_numpy(dzdy), _numpy(fd_y), rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", FLAT)
def test_polynomial_error_is_not_larger_than_master(set_test_backend, name):
    """The largest and the RMS error of the polynomial part: new <= former.

    On a flat base, the sag is 0 + the polynomial, so the errors are those of
    the changed code only.
    """
    new, old, _ = _errors(name, *_points(120, seed=3))
    for columns in (slice(0, 1), slice(1, 3)):
        assert new[:, columns].max() <= old[:, columns].max()
        assert np.sqrt(np.mean(new[:, columns] ** 2)) <= np.sqrt(
            np.mean(old[:, columns] ** 2)
        )
    # The unit normal adds the same rounding in both: 1 ulp or 2.
    assert new[:, 3:].max() <= 2 * EPS


@pytest.mark.parametrize("name", CURVED)
def test_surface_error_is_a_few_ulp(set_test_backend, name):
    """With the conic, the values are within a few ulp of the reference."""
    new, old, reference = _errors(name, *_points(120, seed=4))
    for column in range(6):
        scale = EPS * max(1.0, np.max(np.abs(reference[:, column])))
        assert new[:, column].max() <= 4 * scale
    # Not larger than the error of the former code.
    for columns in (slice(0, 1), slice(1, 3)):
        assert new[:, columns].max() <= old[:, columns].max()


# --- Structure and behavior --------------------------------------------------


@pytest.mark.parametrize("name", list(SURFACES))
def test_sag_and_terms_give_the_bits_of_sag_and_normal(set_test_backend, name):
    geometry = _geometry(name)
    x, y = (be.array(v) for v in _points(200, seed=5))
    with _no_grad():
        sag, terms = geometry._sag_and_terms(x, y)
        assert_bits_equal(sag, geometry.sag(x, y))
        slopes = geometry._slopes_from_terms(x, y, terms)
        expected = _unit_normal(*slopes)
        for value, other in zip(geometry._surface_normal(x, y), expected, strict=True):
            assert_bits_equal(value, other)
        r2 = x**2 + y**2
        assert_bits_equal(terms.r2, r2)
        root = be.sqrt(1 - (1 + geometry.k) * r2 / geometry.radius**2)
        assert_bits_equal(terms.root, root)


def test_conic_part_keeps_the_bits_of_master(set_test_backend):
    """Without terms, sag and normal are the conic of master, bit for bit.

    With zero coefficients, the values are those of master. Only the sign of
    a zero can differ: master adds the zero terms, and numbers leave them out.
    """
    x, y = be.array([0.0, 1.0, -2.0, 3.5]), be.array([0.0, 0.5, 3.0, -1.0])
    for coefficients in ([], [[0.0]], [[0.0, 0.0], [0.0, 0.0]]):
        geometry = PolynomialGeometry(
            CoordinateSystem(), radius=-30.0, conic=0.2, coefficients=coefficients
        )
        with _no_grad():
            r2 = x**2 + y**2
            root = be.sqrt(1 - (1 + geometry.k) * r2 / geometry.radius**2)
            sag = r2 / (geometry.radius * (1 + root))
            normal = _unit_normal(*(v / (geometry.radius * root) for v in (x, y)))
            new_normal = geometry._surface_normal(x, y)
            old_normal = _unit_normal(*_master_slopes(geometry, x, y))
            if not coefficients:
                assert_bits_equal(geometry.sag(x, y), sag)
                assert_bits_equal(geometry.sag(x, y), _master_sag(geometry, x, y))
                for a, b in zip(new_normal, normal, strict=True):
                    assert_bits_equal(a, b)
            np.testing.assert_array_equal(
                _numpy(geometry.sag(x, y)), _numpy(_master_sag(geometry, x, y))
            )
            for a, b in zip(new_normal, old_normal, strict=True):
                np.testing.assert_array_equal(_numpy(a), _numpy(b))


def test_scalar_inputs(set_test_backend):
    """Python numbers as coordinates work as in master, also the defaults."""
    geometry = _geometry("dense-conic")
    with _no_grad():
        for x, y in ((0, 0), (1, 1), (-2, 0.5), (0, 3.0)):
            new = float(_numpy(geometry.sag(x, y)))
            old = float(_numpy(_master_sag(geometry, x, y)))
            assert abs(new - old) <= 8 * EPS * max(1.0, abs(old))
        # At the vertex, the sag is C00.
        assert float(_numpy(geometry.sag())) == DENSE[0][0]
        assert float(_numpy(geometry.sag())) == float(
            _numpy(_master_sag(geometry, 0, 0))
        )


def test_the_newton_loop_takes_sag_and_slopes_from_the_terms(
    set_test_backend, monkeypatch
):
    geometry = _geometry("sparse")
    assert nr._shares_terms(geometry)
    calls = {"terms": 0, "slopes": 0}
    original_terms = PolynomialGeometry._sag_and_terms
    original_slopes = PolynomialGeometry._slopes_from_terms

    def sag_and_terms(self, x, y):
        calls["terms"] += 1
        return original_terms(self, x, y)

    def slopes_from_terms(self, x, y, terms):
        calls["slopes"] += 1
        return original_slopes(self, x, y, terms)

    def fail(self, x, y):
        raise AssertionError("the loop evaluated the normal")

    monkeypatch.setattr(PolynomialGeometry, "_sag_and_terms", sag_and_terms)
    monkeypatch.setattr(PolynomialGeometry, "_slopes_from_terms", slopes_from_terms)
    monkeypatch.setattr(PolynomialGeometry, "_surface_normal", fail)
    rays = _bundle()
    with _no_grad():
        result = geometry._solve_distance_primal(rays)
    assert result.final is not None
    assert result.iterations > 0
    assert calls == {"terms": result.iterations + 1, "slopes": result.iterations}


def test_a_polynomial_without_terms_runs_the_loop_of_sag_and_normal(
    set_test_backend,
):
    """Without a term, the shared loop has nothing to share (``_has_terms``)."""
    empty = PolynomialGeometry(CoordinateSystem(), radius=-45.0)
    zero = PolynomialGeometry(CoordinateSystem(), radius=-45.0, coefficients=[[0.0]])
    constant = PolynomialGeometry(
        CoordinateSystem(), radius=-45.0, coefficients=[[1e-3]]
    )
    zeros = PolynomialGeometry(
        CoordinateSystem(), radius=-45.0, coefficients=[[0.0, 0.0], [0.0, 0.0]]
    )
    assert be.size(empty.coefficients) == 0
    assert not nr._shares_terms(empty)
    # A tensor keeps a zero term, so that its coefficient keeps its derivative.
    assert nr._shares_terms(zero) is (be.get_backend() == "torch")
    assert nr._shares_terms(constant)
    assert nr._shares_terms(zeros)
    with _no_grad():
        result = empty._solve_distance_primal(_bundle())
    assert result.final is None
    assert bool(np.all(_numpy(result.converged)))


def test_the_decision_follows_the_coefficients_of_the_call():
    """A coefficient written in place applies to the next decision."""
    be.set_backend("numpy")
    geometry = PolynomialGeometry(
        CoordinateSystem(), radius=-45.0, coefficients=[[0.0]]
    )
    assert not nr._shares_terms(geometry)
    geometry.coefficients[0][0] = 1e-3
    assert nr._shares_terms(geometry)
    geometry.coefficients[0][0] = 0.0
    assert not nr._shares_terms(geometry)


def test_only_a_class_that_overrides_has_terms_is_asked(monkeypatch):
    """The other classes keep the decision for their class: no call per instance."""
    be.set_backend("numpy")

    def fail(self):
        raise AssertionError("the loop asked an instance of a class without terms")

    monkeypatch.setattr(nr.NewtonRaphsonGeometry, "_has_terms", fail)
    asphere = EvenAsphere(CoordinateSystem(), radius=20.0, coefficients=[1e-4])
    assert nr._shares_terms(asphere)
    assert not nr._shares_terms(PolynomialGeometry(CoordinateSystem(), radius=-45.0))
    with _no_grad():
        result = asphere._solve_distance_primal(_bundle())
    assert result.final is not None


def _bundle(n=301):
    """Rays from z = -5 towards the surface, off the y-z plane."""
    y = np.linspace(-6.0, 6.0, n)
    x = 2.0 * np.sin(5.0 * y)
    L = np.full(n, 0.03)
    M = np.full(n, -0.06)
    N = np.sqrt(1.0 - L**2 - M**2)
    return RealRays(x, y, np.full(n, -5.0), L, M, N, np.ones(n), np.full(n, 0.55))


@pytest.mark.parametrize("name", ["sparse", "dense-conic", "ragged"])
def test_the_loop_with_terms_finds_the_roots_of_the_loop_without(
    set_test_backend, monkeypatch, name
):
    geometry = _geometry(name, tol=1e-12)
    with _no_grad():
        shared = geometry._solve_distance_primal(_bundle())
        monkeypatch.setattr(nr, "_shares_terms", lambda geometry: False)
        plain = geometry._solve_distance_primal(_bundle())
    assert shared.final is not None
    assert plain.final is None
    assert bool(np.all(_numpy(shared.converged)))
    assert bool(np.all(_numpy(plain.converged)))
    np.testing.assert_allclose(_numpy(shared.t), _numpy(plain.t), rtol=0, atol=1e-12)


def test_root_classification_from_the_final_terms(set_test_backend):
    geometry = _geometry("dense-conic", tol=1e-12)
    rays = _bundle()
    with _no_grad():
        result = geometry._solve_distance_primal(rays)
        from_terms = geometry._classify_final_roots(result, rays)
        plain = dataclasses.replace(result, final=None)
        from_normal = geometry._classify_final_roots(plain, rays)
    np.testing.assert_array_equal(
        _numpy(from_terms.regular), _numpy(from_normal.regular)
    )
    np.testing.assert_allclose(
        _numpy(from_terms.df_dt), _numpy(from_normal.df_dt), rtol=1e-13
    )


def test_coefficient_change_applies_at_the_next_call(set_test_backend):
    geometry = _geometry("sparse")
    x, y = (be.array(v) for v in _points(50, seed=6))
    with _no_grad():
        before = _numpy(geometry.sag(x, y))
        geometry.coefficients[1][2] = 3e-6  # in place, as the optimizer does
        changed = _numpy(geometry.sag(x, y))
        changed_normal = [_numpy(v) for v in geometry._surface_normal(x, y)]
    fresh = PolynomialGeometry(
        CoordinateSystem(),
        radius=-45.0,
        coefficients=_numpy(geometry.coefficients).tolist(),
    )
    with _no_grad():
        assert not np.array_equal(before, changed)
        assert_bits_equal(changed, _numpy(fresh.sag(x, y)))
        for a, b in zip(changed_normal, fresh._surface_normal(x, y), strict=True):
            assert_bits_equal(a, _numpy(b))
        # A new, smaller array.
        geometry.coefficients = be.atleast_2d([[0.0, 1e-3], [2e-3, 0.0]])
        small = PolynomialGeometry(
            CoordinateSystem(), radius=-45.0, coefficients=[[0.0, 1e-3], [2e-3, 0.0]]
        )
        assert_bits_equal(geometry.sag(x, y), small.sag(x, y))


@pytest.mark.parametrize("name", FLAT)
def test_numbers_and_tensors_give_the_same_values(name):
    """NumPy numbers leave out zero terms; tensors keep them. Same values.

    The surfaces are flat, and the test compares the sag and the slopes: the
    conic root and the unit normal of NumPy and of Torch can differ in the
    last bit (also in master), and the polynomial part does not.
    """
    pytest.importorskip("torch")
    from .nr_implicit_test_utils import backend_state

    x_np, y_np = _points(300, seed=7)
    results = {}
    for backend in ("numpy", "torch"):
        with backend_state(backend, "float64"), _no_grad():
            geometry = _geometry(name)
            x, y = be.array(x_np), be.array(y_np)
            _, terms = geometry._sag_and_terms(x, y)
            results[backend] = [
                _numpy(geometry.sag(x, y)),
                *(_numpy(v) for v in geometry._slopes_from_terms(x, y, terms)),
            ]
    for a, b in zip(results["numpy"], results["torch"], strict=True):
        np.testing.assert_array_equal(a, b)  # -0.0 == 0.0


# --- Torch -------------------------------------------------------------------

torch_only = pytest.mark.skipif(
    "torch" not in be.list_available_backends(), reason="Requires the Torch backend"
)


@pytest.fixture
def torch_backend():
    import torch

    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.grad_mode.disable()
    yield torch
    be.set_precision("float64")
    be.set_backend("numpy")


def _torch_geometry(torch, name, device="cpu", tol=1e-10):
    geometry = _geometry(name, tol=tol)
    geometry.coefficients = torch.tensor(
        SURFACES[name][0], dtype=torch.float64, device=device
    ).requires_grad_(True)
    geometry.radius = geometry.radius.to(device)
    geometry.k = geometry.k.to(device)
    return geometry


@contextlib.contextmanager
def _form(form):
    """Force the Horner pass for all rows at once ("all") or for each row."""
    limits = (
        polynomial_module._ALL_ROWS_LIMIT_CPU,
        polynomial_module._ALL_ROWS_LIMIT_GPU,
    )
    value = 2**62 if form == "all" else 0
    polynomial_module._ALL_ROWS_LIMIT_CPU = polynomial_module._ALL_ROWS_LIMIT_GPU = (
        value
    )
    try:
        yield
    finally:
        polynomial_module._ALL_ROWS_LIMIT_CPU, polynomial_module._ALL_ROWS_LIMIT_GPU = (
            limits
        )


@torch_only
@pytest.mark.parametrize("name", ["sparse", "dense-conic"])
def test_both_torch_forms_give_the_same_bits(torch_backend, name):
    torch = torch_backend
    geometry = _torch_geometry(torch, name)
    x, y = (torch.tensor(v) for v in _points(500, seed=8))
    values = {}
    for form in ("all", "rows"):
        with _form(form), torch.no_grad():
            sag, terms = geometry._sag_and_terms(x, y)
            values[form] = [sag, *geometry._slopes_from_terms(x, y, terms)]
            values[form] += list(geometry._surface_normal(x, y))
    for a, b in zip(values["all"], values["rows"], strict=True):
        assert_bits_equal(a, b)


@torch_only
@pytest.mark.parametrize("form", ["all", "rows"])
@pytest.mark.parametrize("name", ["sparse", "dense-conic"])
def test_gradient_of_the_sag_with_respect_to_coefficients(torch_backend, name, form):
    """d sag / d Cij = x^i y^j, also for the zero coefficients."""
    torch = torch_backend
    geometry = _torch_geometry(torch, name)
    x_np, y_np = _points(100, seed=9)
    x, y = torch.tensor(x_np), torch.tensor(y_np)
    weights = np.linspace(0.5, 1.5, len(x_np))
    with _form(form):
        (geometry.sag(x, y) * torch.tensor(weights)).sum().backward()
    gradient = geometry.coefficients.grad.numpy()
    rows, columns = gradient.shape
    expected = np.array(
        [
            [np.sum(weights * x_np**i * y_np**j) for j in range(columns)]
            for i in range(rows)
        ]
    )
    np.testing.assert_allclose(gradient, expected, rtol=1e-12, atol=0)
    zero = np.array(SURFACES[name][0]) == 0
    if name == "sparse":
        assert zero.sum() == 17
    assert np.all(gradient[zero] != 0)  # x^i y^j, not 0


@torch_only
@pytest.mark.parametrize("name", ["sparse", "dense-conic"])
def test_gradient_of_the_normal_with_respect_to_coefficients(torch_backend, name):
    torch = torch_backend
    geometry = _torch_geometry(torch, name)
    x, y = (torch.tensor(v) for v in _points(60, seed=10))
    weights = torch.linspace(0.5, 1.5, len(x), dtype=torch.float64)

    def loss(coefficients, slopes=None):
        geometry.coefficients = coefficients
        if slopes is None:
            nx, ny, nz = geometry._surface_normal(x, y)
        else:
            nx, ny, nz = _unit_normal(*slopes(geometry, x, y))
        return ((nx + 2 * ny + 3 * nz) * weights).sum()

    coefficients = geometry.coefficients
    loss(coefficients).backward()
    gradient = coefficients.grad.numpy()
    step = 1e-7
    with torch.no_grad():
        for i, j in ((0, 2), (1, 1), (2, 0), (3, 1), (0, 1)):
            plus = coefficients.detach().clone()
            minus = coefficients.detach().clone()
            plus[i, j] += step
            minus[i, j] -= step
            difference = (loss(plus) - loss(minus)) / (2 * step)
            assert abs(gradient[i, j] - float(difference)) <= 1e-7 * max(
                1.0, abs(gradient[i, j])
            )
    # The gradients of the former evaluation, to rounding.
    former = coefficients.detach().clone().requires_grad_(True)
    loss(former, slopes=_master_slopes).backward()
    np.testing.assert_allclose(gradient, former.grad.numpy(), rtol=1e-12, atol=1e-14)


@torch_only
@pytest.mark.parametrize("form", ["all", "rows"])
def test_gradient_with_respect_to_positions_is_the_slope(torch_backend, form):
    torch = torch_backend
    geometry = _torch_geometry(torch, "dense-conic")
    x_np, y_np = _points(100, seed=11)
    x = torch.tensor(x_np).requires_grad_(True)
    y = torch.tensor(y_np).requires_grad_(True)
    with _form(form):
        geometry.sag(x, y).sum().backward()
        dzdx, dzdy = _slopes(geometry, x.detach(), y.detach())
    _, scale_x, scale_y = _scales("dense-conic", x_np, y_np)
    factor = 2 * _depth("dense-conic") * EPS
    assert np.all(np.abs(x.grad.numpy() - dzdx.detach().numpy()) <= factor * scale_x)
    assert np.all(np.abs(y.grad.numpy() - dzdy.detach().numpy()) <= factor * scale_y)


@torch_only
def test_torch_func_transforms(torch_backend):
    """jacrev and jvp over positions, vmap of the normal, jvp at a zero Cij."""
    torch = torch_backend
    geometry = _torch_geometry(torch, "sparse")
    x_np, y_np = _points(8, seed=12)
    x, y = torch.tensor(x_np), torch.tensor(y_np)
    with torch.no_grad():
        dzdx, dzdy = _slopes(geometry, x, y)
    _, scale_x, scale_y = _scales("sparse", x_np, y_np)
    factor = 2 * _depth("sparse") * EPS
    jacobian = torch.func.jacrev(lambda xx: geometry.sag(xx, y))(x)
    assert np.all(
        np.abs(torch.diagonal(jacobian).detach().numpy() - dzdx.numpy())
        <= factor * scale_x
    )
    _, tangent = torch.func.jvp(
        lambda yy: geometry.sag(x, yy), (y,), (torch.ones_like(y),)
    )
    assert np.all(np.abs(tangent.detach().numpy() - dzdy.numpy()) <= factor * scale_y)
    mapped = torch.func.vmap(lambda a, b: geometry._surface_normal(a, b)[0])(x, y)
    assert_bits_equal(mapped.detach(), geometry._surface_normal(x, y)[0].detach())

    # Forward mode at the zero coefficient C00: the tangent is x^0 y^0 = 1.
    base = geometry.coefficients.detach().clone()
    direction = torch.zeros_like(base)
    direction[0, 0] = 1.0

    def sag_of(coefficients):
        geometry.coefficients = coefficients
        return geometry.sag(x, y)

    _, tangent = torch.func.jvp(sag_of, (base,), (direction,))
    np.testing.assert_array_equal(tangent.detach().numpy(), np.ones(len(x_np)))


@torch_only
def test_list_of_tensors_keeps_the_gradients(torch_backend):
    """Rows of tensors (for example parameters) keep all terms."""
    torch = torch_backend
    geometry = _geometry("ragged")
    leaves = [
        [torch.tensor(c, dtype=torch.float64, requires_grad=True) for c in row]
        for row in RAGGED
    ]
    geometry.coefficients = leaves
    x_np, y_np = _points(40, seed=13)
    x, y = torch.tensor(x_np), torch.tensor(y_np)
    geometry.sag(x, y).sum().backward()
    for i, row in enumerate(leaves):
        for j, leaf in enumerate(row):
            expected = np.sum(x_np**i * y_np**j)
            assert float(leaf.grad) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@torch_only
def test_a_single_tensor_coefficient_keeps_its_gradient(torch_backend):
    """A 1 x 1 tensor: the sag adds C00, also at zero, and d(sag)/dC00 = 1."""
    torch = torch_backend
    x_np, y_np = _points(30, seed=15)
    x, y = torch.tensor(x_np), torch.tensor(y_np)
    for value in (0.0, 1e-3):
        leaf = torch.tensor([[value]], dtype=torch.float64, requires_grad=True)
        geometry = PolynomialGeometry(
            CoordinateSystem(), radius=-45.0, coefficients=leaf
        )
        sag = geometry.sag(x, y)
        assert_bits_equal(sag.detach(), _master_sag(geometry, x, y).detach())
        sag.sum().backward()
        assert float(leaf.grad[0, 0]) == len(x_np)
        normal = geometry._surface_normal(x, y)
        expected = _unit_normal(*_master_slopes(geometry, x, y))
        for a, b in zip(normal, expected, strict=True):
            assert_bits_equal(a.detach(), b.detach())


@torch_only
def test_numpy_coefficients_in_torch_mode(torch_backend):
    """A NumPy array (as the padding of PolynomialCoeffVariable makes it)."""
    torch = torch_backend
    geometry = _geometry("sparse")
    x, y = (torch.tensor(v) for v in _points(50, seed=14))
    expected = geometry.sag(x, y).detach()
    geometry.coefficients = np.array(SPARSE)
    value = geometry.sag(x, y)
    assert isinstance(value, torch.Tensor)
    np.testing.assert_array_equal(value.numpy(), expected.numpy())


@torch_only
def test_float32(torch_backend):
    torch = torch_backend
    x64, y64 = (torch.tensor(v) for v in _points(200, seed=15))
    expected_sag = _torch_geometry(torch, "dense-conic").sag(x64, y64).detach()
    expected_normal = [
        v.detach()
        for v in _torch_geometry(torch, "dense-conic")._surface_normal(x64, y64)
    ]
    be.set_precision("float32")
    for form in ("all", "rows"):
        geometry = _geometry("dense-conic")
        x, y = x64.float(), y64.float()
        with _form(form):
            sag = geometry.sag(x, y)
            normal = geometry._surface_normal(x, y)
        assert sag.dtype == torch.float32
        assert all(v.dtype == torch.float32 for v in normal)
        torch.testing.assert_close(sag.double(), expected_sag, rtol=0, atol=2e-6)
        for a, b in zip(normal, expected_normal, strict=True):
            torch.testing.assert_close(a.double(), b, rtol=0, atol=1e-6)


@torch_only
def test_distance_gradients_match_the_loop_without_terms(torch_backend, monkeypatch):
    torch = torch_backend

    def gradients():
        geometry = _torch_geometry(torch, "sparse", tol=1e-12)
        geometry.radius = torch.nn.Parameter(torch.tensor(-45.0, dtype=torch.float64))
        t = geometry.distance(_bundle(61))
        t.sum().backward()
        return [geometry.radius.grad.numpy(), geometry.coefficients.grad.numpy()]

    shared = gradients()
    monkeypatch.setattr(nr, "_shares_terms", lambda geometry: False)
    plain = gradients()
    for a, b in zip(shared, plain, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@torch_only
@pytest.mark.skipif(not _cuda_available(), reason="Requires CUDA")
@pytest.mark.parametrize("form", ["all", "rows"])
@pytest.mark.parametrize("name", ["sparse-flat", "sparse"])
def test_cuda_matches_cpu_without_host_reads(torch_backend, form, name):
    """The polynomial part has the bits of the CPU; no value goes to the host.

    The conic root and the unit normal of CUDA and of the CPU can differ in
    the last bit (also in master), so they agree to a few ulp.
    """
    torch = torch_backend
    x_np, y_np = _points(1000, seed=16)
    values = {}
    for device in ("cpu", "cuda"):
        be.set_device(device)
        try:
            geometry = _torch_geometry(torch, name, device=device)
            x = torch.tensor(x_np, device=device)
            y = torch.tensor(y_np, device=device)
            with _form(form), torch.no_grad():
                sag, terms = geometry._sag_and_terms(x, y)
                slopes = geometry._slopes_from_terms(x, y, terms)
                normal = geometry._surface_normal(x, y)
            values[device] = [v.cpu() for v in (sag, *slopes, *normal)]
        finally:
            be.set_device("cpu")
    for a, b in zip(values["cuda"][:3], values["cpu"][:3], strict=True):
        if math.isinf(SURFACES[name][1]):
            assert_bits_equal(a, b)
        torch.testing.assert_close(a, b, rtol=4 * EPS, atol=4 * EPS)
    for a, b in zip(values["cuda"][3:], values["cpu"][3:], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=2 * EPS)

    be.set_device("cuda")
    try:
        geometry = _torch_geometry(torch, name, device="cuda")
        x = torch.tensor(x_np, device="cuda")
        y = torch.tensor(y_np, device="cuda")
        with _form(form):
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                with torch.no_grad():
                    sag, terms = geometry._sag_and_terms(x, y)
                    geometry._slopes_from_terms(x, y, terms)
                    geometry._surface_normal(x, y)
                loss = geometry.sag(x, y).sum()
                loss = loss + sum(geometry._surface_normal(x, y)).sum()
                loss.backward()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        assert torch.all(torch.isfinite(geometry.coefficients.grad))
    finally:
        be.set_device("cpu")
