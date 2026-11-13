"""Tests for the table evaluation of the Zernike geometry.

ZernikePolynomialGeometry computes its sag and slopes from tables of exact
rational numbers, without arctan2, trigonometric functions or powers of rho
(see ``optiland/geometries/zernike.py``). The tests compare the new values
with:

- the Zernike basis of ``optiland.zernike``, term by term;
- the former evaluation of master 4f4f514f (``_master_sag`` and
  ``_master_slopes`` below, a copy with arctan2, cos, sin and powers of rho);
- a reference with 50 digits that needs only the standard library:
  ``fractions.Fraction`` for the polynomials and ``decimal`` for the square
  roots. The new error must not be larger than the error of the former code.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

import optiland._host_values as host_values
import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import ZernikePolynomialGeometry
from optiland.geometries.zernike import _zernike_layout
from optiland.zernike import ZernikeFringe, ZernikeNoll, ZernikeStandard

from .utils import assert_bits_equal

EPS = float(np.finfo(np.float64).eps)
TYPES = {"fringe": ZernikeFringe, "standard": ZernikeStandard, "noll": ZernikeNoll}

# The Zernike Fringe 37 surface of the equivalence corpus (about 2e-3 mm).
FRINGE_37 = [
    0.0, 0.0, 0.0, 0.0, 1.6e-3, -2.1e-3, 0.9e-3, 1.4e-3, -0.7e-3, 2.4e-3,
    -1.2e-3, 0.5e-3, 1.9e-3, -1.6e-3, 0.8e-3, -0.4e-3, 1.1e-3, -2.2e-3, 0.6e-3,
    1.3e-3, -0.9e-3, 0.3e-3, -1.5e-3, 1.0e-3, -0.6e-3, 0.7e-3, -1.1e-3, 0.4e-3,
    0.9e-3, -0.8e-3, 0.5e-3, -0.3e-3, 0.6e-3, -0.5e-3, 0.2e-3, 0.4e-3, -0.2e-3,
]  # fmt: skip


def _coefficients(num_terms, seed):
    return list(np.random.default_rng(seed).normal(0.0, 2e-3, num_terms))


# (Zernike type, coefficients, radius, conic) of the surfaces under test.
SURFACES = {
    "fringe37": ("fringe", FRINGE_37, -50.0, -0.3),
    "fringe37-flat": ("fringe", FRINGE_37, math.inf, 0.0),
    "standard45": ("standard", _coefficients(45, 1), 40.0, 0.5),
    "noll45-flat": ("noll", _coefficients(45, 2), math.inf, 0.0),
    "standard66-flat": ("standard", _coefficients(66, 3), math.inf, 0.0),
}
NORM_RADIUS = 12.0


def _geometry(name):
    zernike_type, coefficients, radius, conic = SURFACES[name]
    return ZernikePolynomialGeometry(
        CoordinateSystem(),
        radius=radius,
        conic=conic,
        coefficients=coefficients,
        zernike_type=zernike_type,
        norm_radius=NORM_RADIUS,
    )


def _points(count, seed=0, radius=0.98 * NORM_RADIUS):
    """Points in the normalization disk, with the center and the axes."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(count))
    angle = 2 * np.pi * rng.random(count)
    x = np.concatenate([r * np.cos(angle), [0.0, 0.0, 1e-7, 0.5 * radius, radius]])
    y = np.concatenate([r * np.sin(angle), [0.0, 0.5 * radius, 0.0, 0.0, 0.0]])
    return x, y


def _numpy(value):
    return np.asarray(be.to_numpy(value), dtype=float)


# --- The former evaluation (master 4f4f514f) -------------------------------


def _master_polar(geometry, x, y):
    x_norm = x / geometry.norm_radius
    y_norm = y / geometry.norm_radius
    rho2 = x_norm**2 + y_norm**2
    center = rho2 == 0
    safe_rho2 = be.where(center, be.ones_like(rho2), rho2)
    safe_rho = be.sqrt(safe_rho2)
    rho = be.where(center, be.zeros_like(safe_rho), safe_rho)
    x_phi = be.where(center, be.ones_like(x_norm), x_norm)
    y_phi = be.where(center, be.zeros_like(y_norm), y_norm)
    return x_norm, y_norm, center, safe_rho2, safe_rho, rho, be.arctan2(y_phi, x_phi)


def _master_sag(geometry, x, y):
    _, _, _, _, _, rho, phi = _master_polar(geometry, x, y)
    r2 = x**2 + y**2
    if bool(be.all(be.isinf(geometry.radius))):
        z = be.zeros_like(x)
    else:
        radius, k = geometry.radius, geometry.k
        z = r2 / (radius * (1 + be.sqrt(1 - (1 + k) * r2 / radius**2)))
    return z + geometry.zernike.poly(rho, phi)


def _master_slopes(geometry, x, y):
    r2 = x**2 + y**2
    if bool(be.all(be.isinf(geometry.radius))):
        dzdx, dzdy = be.zeros_like(x), be.zeros_like(y)
    else:
        radius, k = geometry.radius, geometry.k
        denominator = radius * be.sqrt(1 - (1 + k) * r2 / radius**2)
        valid = be.abs(denominator) > 0
        safe = be.where(valid, denominator, be.ones_like(denominator))
        dzdx = be.where(valid, x / safe, be.zeros_like(x))
        dzdy = be.where(valid, y / safe, be.zeros_like(y))
    x_norm, y_norm, center, safe_rho2, safe_rho, rho, phi = _master_polar(
        geometry, x, y
    )
    norm_radius = geometry.norm_radius
    zero_x, zero_y = be.zeros_like(x), be.zeros_like(y)
    drho_dx = be.where(center, zero_x, (x / norm_radius**2) / safe_rho)
    drho_dy = be.where(center, zero_y, (y / norm_radius**2) / safe_rho)
    dphi_dx = be.where(center, zero_x, -y_norm / safe_rho2 * (1.0 / norm_radius))
    dphi_dy = be.where(center, zero_y, x_norm / safe_rho2 * (1.0 / norm_radius))
    zernike = geometry.zernike
    for (n, m), c in zip(zernike.indices, zernike.coeffs, strict=True):
        dz_drho, dz_dphi = zernike.get_derivative(n, m, rho, phi)
        derivative_x = dz_drho * drho_dx + dz_dphi * dphi_dx
        derivative_y = dz_drho * drho_dy + dz_dphi * dphi_dy
        if abs(m) == 1:
            slope = zernike._radial_derivative(n, m, be.zeros_like(rho)) / norm_radius
            if m > 0:
                derivative_x = be.where(center, slope, derivative_x)
            else:
                derivative_y = be.where(center, slope, derivative_y)
        dzdx = dzdx + zernike._norm_constant(n, m) * c * derivative_x
        dzdy = dzdy + zernike._norm_constant(n, m) * c * derivative_y
    return dzdx, dzdy


def _unit_normal(dzdx, dzdy):
    norm = be.sqrt(dzdx**2 + dzdy**2 + 1)
    return dzdx / norm, dzdy / norm, -be.ones_like(dzdx) / norm


# --- The reference with 50 digits ------------------------------------------


def _exact_radial(n, m):
    """The coefficients b_j of R_n^m / rho^|m| in u = rho^2 (master's sum)."""
    m = abs(m)
    half = (n - m) // 2
    b = [0] * (half + 1)
    for k in range(half + 1):
        b[half - k] = (-1) ** k * Fraction(
            math.factorial(n - k),
            math.factorial(k)
            * math.factorial((n + m) // 2 - k)
            * math.factorial(half - k),
        )
    return b


def _exact_term(n, m, x_norm, y_norm):
    """Z_n^m and its derivatives in X and Y, as exact fractions."""
    am = abs(m)
    u = x_norm * x_norm + y_norm * y_norm
    re, im = [Fraction(1)], [Fraction(0)]  # (X + iY)^k
    for _ in range(am):
        re, im = (
            re + [re[-1] * x_norm - im[-1] * y_norm],
            im + [re[-1] * y_norm + im[-1] * x_norm],
        )
    b = _exact_radial(n, m)
    radial = sum((bj * u**j for j, bj in enumerate(b)), Fraction(0))
    d_radial = sum((j * bj * u ** (j - 1) for j, bj in enumerate(b) if j), Fraction(0))
    if m >= 0:
        h, hx, hy = re[am], am * re[am - 1] if am else 0, -am * im[am - 1] if am else 0
    else:
        h, hx, hy = im[am], am * im[am - 1], am * re[am - 1]
    value = radial * h
    return (
        value,
        2 * x_norm * d_radial * h + radial * hx,
        2 * y_norm * d_radial * h + radial * hy,
    )


def _decimal(value):
    value = Fraction(value)
    return Decimal(value.numerator) / Decimal(value.denominator)


def _reference(geometry, x, y):
    """Sag, slopes and unit normal at the float points, with 50 digits."""
    zernike_type, coefficients, radius, conic = (
        type(geometry.zernike),
        [float(c) for c in _numpy(geometry.zernike.coeffs)],
        float(_numpy(geometry.radius)),
        float(_numpy(geometry.k)),
    )
    terms = [(int(n), int(m)) for n, m in geometry.zernike.indices]
    out = np.empty((len(x), 6))
    with localcontext() as context:
        context.prec = 50
        scales = [
            _decimal(c) * Decimal(zernike_type._norm_squared(n, m)).sqrt()
            for (n, m), c in zip(terms, coefficients, strict=True)
        ]
        norm_radius = _decimal(geometry.norm_radius)
        for i, (xi, yi) in enumerate(zip(x, y, strict=True)):
            x_norm = Fraction(float(xi)) / Fraction(geometry.norm_radius)
            y_norm = Fraction(float(yi)) / Fraction(geometry.norm_radius)
            sag = slope_x = slope_y = Decimal(0)
            for (n, m), scale in zip(terms, scales, strict=True):
                value, d_x, d_y = _exact_term(n, m, x_norm, y_norm)
                sag += scale * _decimal(value)
                slope_x += scale * _decimal(d_x) / norm_radius
                slope_y += scale * _decimal(d_y) / norm_radius
            if math.isfinite(radius):
                r2 = Fraction(float(xi)) ** 2 + Fraction(float(yi)) ** 2
                root = _decimal(1 - (1 + Fraction(conic)) * r2 / Fraction(radius) ** 2)
                root = root.sqrt()
                sag += _decimal(r2) / (_decimal(radius) * (1 + root))
                slope_x += _decimal(float(xi)) / (_decimal(radius) * root)
                slope_y += _decimal(float(yi)) / (_decimal(radius) * root)
            norm = (slope_x * slope_x + slope_y * slope_y + 1).sqrt()
            out[i] = [
                float(sag),
                float(slope_x),
                float(slope_y),
                float(slope_x / norm),
                float(slope_y / norm),
                float(-1 / norm),
            ]
    return out


# --- Tables ------------------------------------------------------------------


@pytest.mark.parametrize("zernike_type", list(TYPES))
def test_radial_coefficients_are_master_radial_sum(zernike_type):
    zernike = TYPES[zernike_type]
    for n in range(15):
        for m in range(-n, n + 1, 2):
            exact = _exact_radial(n, m)
            assert list(zernike._radial_coefficients(n, m)) == exact
            assert sum(exact) == 1  # R_n^m(1) = 1


@pytest.mark.parametrize("zernike_type", list(TYPES))
def test_norm_squared_gives_master_norm_constant(zernike_type):
    zernike = TYPES[zernike_type]
    for n in range(15):
        for m in range(-n, n + 1, 2):
            expected = float(np.asarray(zernike._norm_constant(n, m)))
            assert math.sqrt(zernike._norm_squared(n, m)) == expected


@pytest.mark.parametrize("zernike_type", list(TYPES))
def test_table_entries_are_exact(zernike_type):
    """Each entry is the exact coefficient in t = 2 rho^2 - 1 (or of dQ/du)."""
    zernike = TYPES[zernike_type]
    layout = _zernike_layout(zernike, 91)
    for row, (n, m) in zip(layout.table, zernike._generate_indices(91), strict=True):
        harmonic = next(
            h for h in layout.harmonics if (h.m, h.sine) == (abs(int(m)), m < 0)
        )
        b = _exact_radial(int(n), int(m))
        for power in range(len(b)):
            a = sum(
                (b[j] * math.comb(j, power) / 2**j for j in range(power, len(b))),
                Fraction(0),
            )
            assert Fraction(row[harmonic.offset + power]) == a
            if power:
                assert Fraction(row[harmonic.slope_offset + power - 1]) == 2 * power * a


@pytest.mark.parametrize("zernike_type", list(TYPES))
def test_each_term_matches_the_basis(set_test_backend, zernike_type):
    """A unit coefficient gives N_i Z_i, as the basis of optiland.zernike."""
    num_terms = 45
    geometry = ZernikePolynomialGeometry(
        CoordinateSystem(),
        radius=be.inf,
        coefficients=[0.0] * num_terms,
        zernike_type=zernike_type,
        norm_radius=NORM_RADIUS,
    )
    x, y = _points(40, seed=4)
    scale = Fraction(NORM_RADIUS)
    _, _, _, _, _, rho, phi = _master_polar(geometry, be.array(x), be.array(y))
    for i, (n, m) in enumerate(geometry.zernike.indices):
        coefficients = [0.0] * num_terms
        coefficients[i] = 1.0
        geometry.coefficients = coefficients
        new = _numpy(geometry.sag(be.array(x), be.array(y)))
        basis = _numpy(geometry.zernike.get_term(1.0, n, m, rho, phi))
        norm = math.sqrt(type(geometry.zernike)._norm_squared(n, m))
        exact = np.array(
            [
                norm
                * float(
                    _exact_term(
                        int(n), int(m), Fraction(a) / scale, Fraction(b) / scale
                    )[0]
                )
                for a, b in zip(x, y, strict=True)
            ]
        )
        assert np.max(np.abs(new - exact)) <= 32 * EPS * norm
        assert np.allclose(basis, exact, rtol=0.0, atol=1e-11 * norm)


# --- Against master and against the reference -------------------------------


def _bound(geometry, slopes=False):
    """The rounding scale of a sum of powers of rho for |rho| <= 1.

    1 + sum over terms and powers of |c_i N_i b_ij|, with the factor
    (2 j + |m|) / normalization of the derivative for the slopes.
    """
    zernike = geometry.zernike
    total = 1.0
    for (n, m), c in zip(zernike.indices, _numpy(zernike.coeffs), strict=True):
        norm = math.sqrt(type(zernike)._norm_squared(int(n), int(m)))
        b = _exact_radial(int(n), int(m))
        if slopes:
            weights = [(2 * j + abs(m)) / NORM_RADIUS for j in range(len(b))]
        else:
            weights = [1] * len(b)
        total += (
            abs(c)
            * norm
            * float(sum(abs(v) * w for v, w in zip(b, weights, strict=True)))
        )
    return total


@pytest.mark.parametrize("surface", list(SURFACES))
def test_sag_and_normal_match_master(set_test_backend, surface):
    """New and former values agree to a few units of the rounding scale."""
    geometry = _geometry(surface)
    x, y = _points(500, seed=5)
    x, y = be.array(x), be.array(y)
    new = _numpy(geometry.sag(x, y))
    old = _numpy(_master_sag(geometry, x, y))
    assert np.max(np.abs(new - old)) <= 4 * EPS * _bound(geometry)
    new = np.stack([_numpy(v) for v in geometry._surface_normal(x, y)])
    old = np.stack([_numpy(v) for v in _unit_normal(*_master_slopes(geometry, x, y))])
    assert np.max(np.abs(new - old)) <= 4 * EPS * _bound(geometry, slopes=True)


def _errors(geometry, x_np, y_np):
    """Absolute errors of the new and the former code against the reference.

    Columns: sag, dz/dx, dz/dy, nx, ny, nz.
    """
    x, y = be.array(x_np), be.array(y_np)
    reference = _reference(geometry, x_np, y_np)
    slopes = geometry._slopes_from_tables(geometry._point_tables(x, y))
    new = np.column_stack(
        [_numpy(geometry.sag(x, y))]
        + [_numpy(v) for v in slopes]
        + [_numpy(v) for v in _unit_normal(*slopes)]
    )
    slopes = _master_slopes(geometry, x, y)
    old = np.column_stack(
        [_numpy(_master_sag(geometry, x, y))]
        + [_numpy(v) for v in slopes]
        + [_numpy(v) for v in _unit_normal(*slopes)]
    )
    return np.abs(new - reference), np.abs(old - reference), reference


@pytest.mark.parametrize("surface", list(SURFACES))
def test_zernike_error_is_not_larger_than_master(set_test_backend, surface):
    """The largest and the RMS error of the Zernike part: new <= former.

    On a flat base, the sag is 0 + the Zernike part, so the errors are those
    of the changed code only. (With a conic, both codes add the same conic
    value. At one point, its rounding error can cancel or add to the Zernike
    error, so a comparison there measures chance; see the next test.)
    """
    geometry = _geometry(surface)
    geometry.radius = be.array(be.inf)
    new, old, _ = _errors(geometry, *_points(80, seed=6))
    for columns in (slice(0, 1), slice(1, 3)):
        assert new[:, columns].max() <= old[:, columns].max()
        assert np.sqrt(np.mean(new[:, columns] ** 2)) <= np.sqrt(
            np.mean(old[:, columns] ** 2)
        )
    # The unit normal adds the same rounding in both: 1 ulp or 2.
    assert new[:, 3:].max() <= 2 * EPS


@pytest.mark.parametrize(
    "surface", [name for name, spec in SURFACES.items() if math.isfinite(spec[2])]
)
def test_surface_error_is_a_few_ulp(set_test_backend, surface):
    """With the conic, the values are within a few ulp of the reference."""
    new, _, reference = _errors(_geometry(surface), *_points(80, seed=6))
    for column in range(6):
        scale = EPS * max(1.0, np.max(np.abs(reference[:, column])))
        assert new[:, column].max() <= 4 * scale


# --- Structure and behavior --------------------------------------------------


def test_shared_tables_give_the_bits_of_sag_and_normal(set_test_backend):
    geometry = _geometry("fringe37")
    x, y = (be.array(v) for v in _points(200, seed=7))
    tables = geometry._point_tables(x, y, validate=True)
    assert_bits_equal(geometry._sag_from_tables(tables), geometry.sag(x, y))
    slopes = geometry._slopes_from_tables(tables)
    for a, b in zip(_unit_normal(*slopes), geometry._surface_normal(x, y), strict=True):
        assert_bits_equal(a, b)


def test_the_newton_loop_takes_the_point_tables(set_test_backend, monkeypatch):
    """The Newton loop evaluates the tables once per iterate (_sag_and_terms)."""
    from optiland.geometries import newton_raphson
    from optiland.rays import RealRays

    geometry = _geometry("fringe37")
    assert newton_raphson._shares_terms(geometry)
    x, y = (be.array(v) for v in _points(200, seed=7))
    sag, tables = geometry._sag_and_terms(x, y)
    assert_bits_equal(sag, geometry.sag(x, y))
    expected = geometry._slopes_from_tables(geometry._point_tables(x, y))
    for a, b in zip(geometry._slopes_from_terms(x, y, tables), expected, strict=True):
        assert_bits_equal(a, b)

    def rays():
        n = 301
        y0 = np.linspace(-9.0, 9.0, n)
        return RealRays(
            0.4 * np.sin(5.0 * y0),
            y0,
            np.full(n, -5.0),
            np.full(n, 0.02),
            np.full(n, 0.05),
            np.full(n, math.sqrt(1.0 - 0.02**2 - 0.05**2)),
            np.ones(n),
            np.full(n, 0.55),
        )

    shared = geometry._solve_distance_primal(rays())
    monkeypatch.setattr(newton_raphson, "_shares_terms", lambda geometry: False)
    plain = geometry._solve_distance_primal(rays())
    assert shared.final is not None
    assert plain.final is None
    assert shared.iterations == plain.iterations
    np.testing.assert_array_equal(_numpy(shared.converged), _numpy(plain.converged))
    np.testing.assert_allclose(_numpy(shared.t), _numpy(plain.t), rtol=0, atol=1e-12)


def test_validation_is_as_master(set_test_backend):
    geometry = _geometry("fringe37")
    x, y = be.array([0.0, 12.5]), be.array([0.0, 0.0])
    with pytest.raises(ValueError, match="Zernike coordinates must be normalized"):
        geometry.sag(x, y)
    with pytest.raises(ValueError, match="Zernike coordinates must be normalized"):
        geometry._point_tables(x, y, validate=True)
    geometry._surface_normal(x, y)  # the normal does not check, as before


def test_center_slope_is_the_tilt_coefficient(set_test_backend):
    """The harmonic form needs no special case at rho = 0."""
    geometry = ZernikePolynomialGeometry(
        CoordinateSystem(),
        radius=be.inf,
        coefficients=[0.0, 0.25, 0.5],
        zernike_type="standard",
        norm_radius=2.0,
    )
    dzdx, dzdy = geometry._slopes_from_tables(
        geometry._point_tables(be.array([0.0]), be.array([0.0]))
    )
    # Z_1^1 = 2 X and Z_1^-1 = 2 Y with N = 2, so dz/dx = 0.5 * 2 / 2.
    assert float(_numpy(dzdx)[0]) == 0.5
    assert float(_numpy(dzdy)[0]) == 0.25


def test_no_terms_give_the_conic(set_test_backend):
    geometry = ZernikePolynomialGeometry(
        CoordinateSystem(), radius=-30.0, conic=0.2, coefficients=[], norm_radius=5.0
    )
    x, y = be.array([0.0, 1.0, -2.0]), be.array([0.0, 0.5, 3.0])
    r2 = x**2 + y**2
    root = be.sqrt(1 - (1 + geometry.k) * r2 / geometry.radius**2)
    assert_bits_equal(geometry.sag(x, y), r2 / (geometry.radius * (1 + root)) + 0)
    nx, ny, _ = geometry._surface_normal(x, y)
    expected = _unit_normal(x / (geometry.radius * root), y / (geometry.radius * root))
    assert_bits_equal(nx, expected[0])
    assert_bits_equal(ny, expected[1])


def test_coefficient_change_applies_at_the_next_call(set_test_backend):
    geometry = _geometry("standard45")
    x, y = (be.array(v) for v in _points(50, seed=8))
    before = _numpy(geometry.sag(x, y))
    coefficients = geometry.zernike.coeffs
    if be.get_backend() == "torch":
        import torch

        with torch.no_grad():
            coefficients[7] += 1e-3
    else:
        coefficients[7] += 1e-3
    changed = _numpy(geometry.sag(x, y))
    fresh = ZernikePolynomialGeometry(
        CoordinateSystem(),
        radius=40.0,
        conic=0.5,
        coefficients=_numpy(coefficients).tolist(),
        zernike_type="standard",
        norm_radius=NORM_RADIUS,
    )
    assert not np.array_equal(before, changed)
    assert_bits_equal(changed, _numpy(fresh.sag(x, y)))
    geometry.coefficients = _numpy(coefficients)[:10].tolist()
    assert_bits_equal(_numpy(geometry.sag(x, y)), _numpy(_shorter(fresh, 10).sag(x, y)))


def _shorter(geometry, num_terms):
    return ZernikePolynomialGeometry(
        CoordinateSystem(),
        radius=float(_numpy(geometry.radius)),
        conic=float(_numpy(geometry.k)),
        coefficients=_numpy(geometry.zernike.coeffs)[:num_terms].tolist(),
        zernike_type=geometry.zernike_type,
        norm_radius=geometry.norm_radius,
    )


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


def _torch_geometry(torch, surface, device="cpu"):
    geometry = _geometry(surface)
    geometry.zernike.coeffs = geometry.zernike.coeffs.to(device).requires_grad_(True)
    geometry.radius = geometry.radius.to(device)
    geometry.k = geometry.k.to(device)
    return geometry


@torch_only
@pytest.mark.parametrize("surface", ["fringe37", "standard45", "noll45-flat"])
def test_gradient_of_the_sag_with_respect_to_coefficients(torch_backend, surface):
    torch = torch_backend
    geometry = _torch_geometry(torch, surface)
    x, y = (torch.tensor(v) for v in _points(100, seed=9))
    weights = torch.linspace(0.5, 1.5, len(x), dtype=torch.float64)
    (geometry.sag(x, y) * weights).sum().backward()
    gradient = geometry.zernike.coeffs.grad.numpy()
    # The sag is linear in the coefficients: d sag / d c_i = N_i Z_i.
    flat = _shorter(geometry, len(gradient))
    flat.radius = torch.tensor(math.inf, dtype=torch.float64)
    expected = []
    for i in range(len(gradient)):
        unit = torch.zeros(len(gradient), dtype=torch.float64)
        unit[i] = 1.0
        flat.zernike.coeffs = unit
        expected.append(float((flat.sag(x, y) * weights).sum()))
    np.testing.assert_allclose(gradient, expected, rtol=1e-13, atol=1e-13)


@torch_only
@pytest.mark.parametrize("surface", ["fringe37", "standard45", "noll45-flat"])
def test_gradient_of_the_normal_with_respect_to_coefficients(torch_backend, surface):
    torch = torch_backend
    geometry = _torch_geometry(torch, surface)
    x, y = (torch.tensor(v) for v in _points(60, seed=10))
    weights = torch.linspace(0.5, 1.5, len(x), dtype=torch.float64)

    def loss(coefficients):
        geometry.zernike.coeffs = coefficients
        nx, ny, nz = geometry._surface_normal(x, y)
        return ((nx + 2 * ny + 3 * nz) * weights).sum()

    coefficients = geometry.zernike.coeffs
    loss(coefficients).backward()
    gradient = coefficients.grad.numpy()
    step = 1e-6
    with torch.no_grad():
        for i in (4, 9, len(gradient) - 1):
            plus, minus = coefficients.detach().clone(), coefficients.detach().clone()
            plus[i] += step
            minus[i] -= step
            difference = (loss(plus) - loss(minus)) / (2 * step)
            assert abs(gradient[i] - float(difference)) <= 1e-7 * max(
                1.0, abs(gradient[i])
            )
    # The same gradients as the former evaluation, to rounding.
    geometry.zernike.coeffs = coefficients.detach().clone().requires_grad_(True)
    nx, ny, nz = _unit_normal(*_master_slopes(geometry, x, y))
    ((nx + 2 * ny + 3 * nz) * weights).sum().backward()
    np.testing.assert_allclose(
        gradient, geometry.zernike.coeffs.grad.numpy(), rtol=1e-10, atol=1e-12
    )


@torch_only
@pytest.mark.parametrize("surface", ["fringe37", "noll45-flat"])
def test_gradient_with_respect_to_positions_is_the_slope(torch_backend, surface):
    torch = torch_backend
    geometry = _torch_geometry(torch, surface)
    x, y = (torch.tensor(v).requires_grad_(True) for v in _points(100, seed=11))
    geometry.sag(x, y).sum().backward()
    dzdx, dzdy = geometry._slopes_from_tables(
        geometry._point_tables(x.detach(), y.detach())
    )
    scale = 4 * EPS * _bound(geometry, slopes=True)
    assert torch.max(torch.abs(x.grad - dzdx)) <= scale
    assert torch.max(torch.abs(y.grad - dzdy)) <= scale
    old_x, old_y = _master_slopes(geometry, x.detach(), y.detach())
    assert torch.max(torch.abs(dzdx - old_x)) <= scale
    assert torch.max(torch.abs(dzdy - old_y)) <= scale


@torch_only
def test_torch_func_transforms_over_positions(torch_backend):
    """jacrev of the sag and the normal, vmap of the normal, jvp of the sag."""
    torch = torch_backend
    geometry = _torch_geometry(torch, "fringe37")
    x, y = (torch.tensor(v) for v in _points(8, seed=12))
    dzdx, dzdy = geometry._slopes_from_tables(geometry._point_tables(x, y))
    scale = 4 * EPS * _bound(geometry, slopes=True)
    jacobian = torch.func.jacrev(lambda xx: geometry.sag(xx, y))(x)
    assert torch.max(torch.abs(torch.diagonal(jacobian) - dzdx)) <= scale
    jacobian = torch.func.jacrev(lambda yy: geometry._surface_normal(x, yy)[1])(y)
    assert torch.all(torch.isfinite(jacobian))
    mapped = torch.func.vmap(lambda a, b: geometry._surface_normal(a, b)[0])(x, y)
    assert_bits_equal(mapped, geometry._surface_normal(x, y)[0])
    _, tangent = torch.func.jvp(
        lambda yy: geometry.sag(x, yy), (y,), (torch.ones_like(y),)
    )
    assert torch.max(torch.abs(tangent - dzdy)) <= scale


@torch_only
def test_float32(torch_backend):
    torch = torch_backend
    x64, y64 = (torch.tensor(v) for v in _points(200, seed=13))
    expected = _geometry("fringe37")._surface_normal(x64, y64)
    sag64 = _geometry("fringe37").sag(x64, y64)
    be.set_precision("float32")
    geometry = _geometry("fringe37")
    x, y = x64.float(), y64.float()
    sag = geometry.sag(x, y)
    normal = geometry._surface_normal(x, y)
    assert sag.dtype == torch.float32
    assert all(v.dtype == torch.float32 for v in normal)
    torch.testing.assert_close(sag.double(), sag64, rtol=0, atol=1e-6)
    for a, b in zip(normal, expected, strict=True):
        torch.testing.assert_close(a.double(), b, rtol=0, atol=1e-6)


@torch_only
def test_tables_made_in_inference_mode_work_with_autograd(torch_backend):
    torch = torch_backend
    geometry = _torch_geometry(torch, "fringe37")
    x, y = (torch.tensor(v) for v in _points(20, seed=14))
    with torch.inference_mode():
        geometry.sag(x, y)
    geometry.sag(x, y).sum().backward()
    assert torch.all(torch.isfinite(geometry.zernike.coeffs.grad))


@torch_only
def test_radius_check_reads_the_host_copy(torch_backend, monkeypatch):
    """In the host-value scope, only the input check of sag() reads a value."""
    torch = torch_backend
    monkeypatch.setattr(host_values, "_DEVICE_TYPES", frozenset({"cuda", "cpu"}))
    geometry = _torch_geometry(torch, "fringe37")
    x, y = (torch.tensor(v) for v in _points(20, seed=15))
    geometry.sag(x, y)  # make the tables
    reads = []
    original = torch.Tensor.item

    def item(tensor):
        reads.append(tensor.shape)
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "item", item)
    with host_values.host_scope([geometry.radius]):
        geometry._surface_normal(x, y)
        assert reads == []
        geometry.sag(x, y)
        assert len(reads) == 2  # be.any of |X| > 1 and of |Y| > 1


@torch_only
@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch")
    or not __import__("torch").cuda.is_available(),
    reason="Requires CUDA",
)
def test_cuda_matches_cpu_without_host_reads(torch_backend, monkeypatch):
    torch = torch_backend
    x, y = _points(1000, seed=16)
    cpu = _torch_geometry(torch, "fringe37")
    expected = [cpu.sag(torch.tensor(x), torch.tensor(y))]
    expected += list(cpu._surface_normal(torch.tensor(x), torch.tensor(y)))
    be.set_device("cuda")
    try:
        geometry = _torch_geometry(torch, "fringe37", device="cuda")
        xc, yc = torch.tensor(x, device="cuda"), torch.tensor(y, device="cuda")
        result = [geometry.sag(xc, yc), *geometry._surface_normal(xc, yc)]
        for a, b in zip(result, expected, strict=True):
            torch.testing.assert_close(a.cpu(), b, rtol=0, atol=1e-14)
        with host_values.host_scope([geometry.radius]):
            torch.cuda.set_sync_debug_mode("error")
            try:
                geometry._surface_normal(xc, yc)
            finally:
                torch.cuda.set_sync_debug_mode("default")
    finally:
        be.set_device("cpu")
