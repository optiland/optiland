"""A coefficient with the value zero still gets its derivative.

Two geometries left out the terms of zero coefficients:
``ChebyshevPolynomialGeometry`` skipped them, and
``EvenAsphere.paraxial_radius`` returned the base radius when the r^2
coefficient was zero. The value does not change when
a zero term is left out, but the derivative with respect to a zero
coefficient is not zero. Without it, a gradient-based optimizer cannot move
a coefficient that starts at zero.

The rule: a zero term is left out only if no derivative with respect to its
coefficient can be requested. Reverse mode, ``torch.func`` transforms and
forward mode (``torch.autograd.forward_ad``) can all request one. Without a
derivative to request (NumPy, or Torch inside ``torch.no_grad``), the code
takes the path of before, with the same operations, values and speed. In
reverse mode, Chebyshev computes the derivative of its zero terms in the
backward pass only.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    ChebyshevPolynomialGeometry,
    EvenAsphere,
    OddAsphere,
)
from optiland.rays import RealRays

from .nr_implicit_test_utils import backend_state

torch = pytest.importorskip("torch")

NORM = 10.0
# Zero coefficients at (0, 0), (1, 1) and (2, 2).
CHEBYSHEV = [
    [0.0, 8.0e-3, -2.0e-3],
    [1.5e-2, 0.0, 1.0e-3],
    [5.0e-3, 2.0e-3, 0.0],
]
NONZERO = sum(1 for row in CHEBYSHEV for value in row if value != 0)
X = [1.2, -3.9, 0.0, 4.5]
Y = [-0.8, 2.4, 0.0, -2.7]


def _chebyshev_t(n, value):
    """T_n(value) from NumPy's Chebyshev module, as an independent reference."""
    return float(np.polynomial.chebyshev.chebval(value, [0.0] * n + [1.0]))


def _chebyshev(coefficients):
    return ChebyshevPolynomialGeometry(
        CoordinateSystem(),
        radius=-45.0,
        conic=0.0,
        coefficients=coefficients,
        norm_x=NORM,
        norm_y=NORM,
    )


@contextlib.contextmanager
def _grad_mode_off():
    """Switch Optiland's Torch grad mode off; other tests can leave it on."""
    previous = be.grad_mode.requires_grad
    be.grad_mode.disable()
    try:
        yield
    finally:
        if previous:
            be.grad_mode.enable()


def _bits(tensor):
    """The bits of a float64 tensor, so that -0.0 and 0.0 differ."""
    return tensor.detach().contiguous().view(torch.int64)


def _t22():
    """T_2(x) T_2(y) at the test points: the derivative along c_22."""
    return [
        _chebyshev_t(2, x / NORM) * _chebyshev_t(2, y / NORM)
        for x, y in zip(X, Y, strict=True)
    ]


class TestChebyshev:
    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_sag_gradient_includes_zero_coefficients(self, precision):
        """The sag gradient is T_i(x) T_j(y) for every coefficient, also zero."""
        with backend_state("torch", precision):
            dtype = torch.float64 if precision == "float64" else torch.float32
            coefficients = torch.tensor(CHEBYSHEV, dtype=dtype, requires_grad=True)
            geometry = _chebyshev(coefficients)
            sag = geometry.sag(be.array(X), be.array(Y)).sum()
            (grad,) = torch.autograd.grad(sag, coefficients)

            expected = [
                [
                    sum(
                        _chebyshev_t(i, x / NORM) * _chebyshev_t(j, y / NORM)
                        for x, y in zip(X, Y, strict=True)
                    )
                    for j in range(3)
                ]
                for i in range(3)
            ]
            tol = 1e-12 if precision == "float64" else 1e-5
            np.testing.assert_allclose(
                grad.detach().numpy(), expected, rtol=tol, atol=tol
            )
            # The zero coefficients themselves have a non-zero derivative.
            assert float(grad[0, 0]) != 0.0
            assert float(grad[2, 2]) != 0.0

    def test_func_jvp_tangent_of_zero_coefficient(self):
        """torch.func.jvp gives T_2(x) T_2(y) along a zero coefficient."""
        with backend_state("torch", "float64"):
            coefficients = torch.tensor(CHEBYSHEV, dtype=torch.float64)
            tangent = torch.zeros_like(coefficients)
            tangent[2, 2] = 1.0  # direction of a zero coefficient

            def sag(c):
                return _chebyshev(c).sag(be.array(X), be.array(Y))

            _, sag_tangent = torch.func.jvp(sag, (coefficients,), (tangent,))
            np.testing.assert_allclose(
                sag_tangent.detach().numpy(), _t22(), rtol=1e-12, atol=1e-12
            )

    def test_forward_ad_tangent_of_zero_coefficient(self):
        """A torch.autograd.forward_ad dual gives the same tangent."""
        forward_ad = torch.autograd.forward_ad
        with backend_state("torch", "float64"), _grad_mode_off():
            coefficients = torch.tensor(CHEBYSHEV, dtype=torch.float64)
            tangent = torch.zeros_like(coefficients)
            tangent[2, 2] = 1.0
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(coefficients, tangent)
                sag = _chebyshev(dual).sag(be.array(X), be.array(Y))
                sag_tangent = forward_ad.unpack_dual(sag).tangent
            np.testing.assert_allclose(
                sag_tangent.detach().numpy(), _t22(), rtol=1e-12, atol=1e-12
            )

    def test_vmap_over_coefficient_arrays(self):
        """torch.func.vmap over coefficient arrays gives the per-array sags."""
        with backend_state("torch", "float64"):
            doubled = np.multiply(CHEBYSHEV, 2.0).tolist()
            batch = torch.tensor([CHEBYSHEV, doubled], dtype=torch.float64)
            x, y = be.array(X), be.array(Y)

            def sag(c):
                return _chebyshev(c).sag(x, y)

            sags = torch.func.vmap(sag)(batch)
            with torch.no_grad():
                for k in range(2):
                    np.testing.assert_allclose(
                        sags[k].numpy(), sag(batch[k]).numpy(), rtol=1e-15, atol=0
                    )

    def test_normal_gradient_of_zero_coefficient_matches_fd(self):
        """n_x has the central-difference derivative at a zero coefficient."""
        with backend_state("torch", "float64"):
            x, y = be.array([2.0]), be.array([-1.5])

            def nx(c11):
                coefficients = torch.tensor(CHEBYSHEV, dtype=torch.float64)
                mask = torch.zeros_like(coefficients)
                mask[1, 1] = 1.0
                return _chebyshev(coefficients + mask * c11)._surface_normal(x, y)[0]

            param = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            (ad,) = torch.autograd.grad(nx(param).sum(), param)
            h = 1e-6
            with torch.no_grad():
                plus = float(nx(torch.tensor(h)).sum())
                minus = float(nx(torch.tensor(-h)).sum())
            fd = (plus - minus) / (2 * h)
            assert float(ad) != 0.0
            assert float(ad) == pytest.approx(fd, rel=1e-6, abs=1e-12)

    def test_distance_gradient_of_zero_coefficient_matches_fd(self):
        """The implicit distance gradient at a zero coefficient matches FD."""
        with backend_state("torch", "float64"):
            L, M = 0.08, -0.05
            rays = RealRays(
                x=0.9,
                y=-0.6,
                z=-5.0,
                L=L,
                M=M,
                N=(1.0 - L * L - M * M) ** 0.5,
                intensity=1.0,
                wavelength=0.587,
            )

            def distance(c22):
                coefficients = torch.tensor(CHEBYSHEV, dtype=torch.float64)
                mask = torch.zeros_like(coefficients)
                mask[2, 2] = 1.0
                geometry = _chebyshev(coefficients + mask * c22)
                geometry.tol = 1e-12
                return geometry.distance(rays)[0]

            param = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            (ad,) = torch.autograd.grad(distance(param), param)
            fd = []
            for h in (2e-6, 1e-6):
                with torch.no_grad():
                    plus = float(distance(torch.tensor(h, dtype=torch.float64)))
                    minus = float(distance(torch.tensor(-h, dtype=torch.float64)))
                fd.append((plus - minus) / (2 * h))
            assert fd[0] == pytest.approx(fd[1], rel=1e-6, abs=1e-12)
            assert float(ad) != 0.0
            assert float(ad) == pytest.approx(fd[1], rel=1e-6, abs=1e-12)

    @pytest.mark.parametrize("mode", ["reverse", "forward"])
    def test_derivative_path_keeps_the_bits(self, mode):
        """Sag and normal keep their bits with a derivative, also a -0.0."""
        forward_ad = torch.autograd.forward_ad
        x = torch.cat([torch.zeros(1), torch.linspace(-9.0, 9.0, 37)]).double()
        y = torch.cat([torch.zeros(1), torch.linspace(7.5, -8.5, 37)]).double()
        # With only these two odd terms, the sag at the vertex is -0.0.
        odd = [[0.0, -1.0e-3], [-2.0e-3, 0.0]]
        with backend_state("torch", "float64"), _grad_mode_off():
            for array in (CHEBYSHEV, odd):
                coefficients = torch.tensor(array, dtype=torch.float64)
                plain = _chebyshev(coefficients)
                with torch.no_grad():
                    expected = (plain.sag(x, y), *plain._surface_normal(x, y))
                if mode == "reverse":
                    tracked = _chebyshev(coefficients.clone().requires_grad_())
                    values = (tracked.sag(x, y), *tracked._surface_normal(x, y))
                else:
                    with forward_ad.dual_level():
                        tangent = torch.ones_like(coefficients)
                        dual = forward_ad.make_dual(coefficients, tangent)
                        tracked = _chebyshev(dual)
                        values = tuple(
                            forward_ad.unpack_dual(value).primal
                            for value in (
                                tracked.sag(x, y),
                                *tracked._surface_normal(x, y),
                            )
                        )
                for value, reference in zip(values, expected, strict=True):
                    assert torch.equal(_bits(value), _bits(reference))
            assert bool(torch.signbit(expected[0][0]))  # the -0.0 was checked

    def test_reverse_mode_evaluates_zero_terms_only_in_backward(self, monkeypatch):
        """In reverse mode, the forward pass evaluates no zero term."""
        with backend_state("torch", "float64"):
            coefficients = torch.tensor(
                CHEBYSHEV, dtype=torch.float64, requires_grad=True
            )
            geometry = _chebyshev(coefficients)
            calls = []
            original = ChebyshevPolynomialGeometry._zero_term_sum

            def counted(*args):
                calls.append(1)
                return original(*args)

            monkeypatch.setattr(
                ChebyshevPolynomialGeometry, "_zero_term_sum", staticmethod(counted)
            )
            sag = geometry.sag(be.array(X), be.array(Y))
            normal = geometry._surface_normal(be.array(X), be.array(Y))
            assert not calls
            assert "ZeroTermDerivative" in type(sag.grad_fn).__name__
            (grad,) = torch.autograd.grad(sag.sum() + normal[0].sum(), coefficients)
            assert float(grad[1, 1]) != 0.0

    def test_second_derivative_through_the_surface(self):
        """create_graph=True works; d2z/dx2 is that of untracked coefficients."""
        with backend_state("torch", "float64"):

            def second_derivative(coefficients):
                x = torch.tensor(X, dtype=torch.float64, requires_grad=True)
                geometry = _chebyshev(coefficients)
                total = geometry.sag(x, be.array(Y)).sum()
                total = total + geometry._surface_normal(x, be.array(Y))[0].sum()
                (first,) = torch.autograd.grad(total, x, create_graph=True)
                (second,) = torch.autograd.grad(first.sum(), x)
                return second

            expected = second_derivative(torch.tensor(CHEBYSHEV, dtype=torch.float64))
            tracked = torch.tensor(CHEBYSHEV, dtype=torch.float64, requires_grad=True)
            assert torch.equal(second_derivative(tracked), expected)

    def test_transform_over_positions_with_tracked_coefficients(self):
        """torch.func.jacrev over the positions, with and without tracked c_ij."""
        with backend_state("torch", "float64"):
            x, y = be.array(X), be.array(Y)
            plain = _chebyshev(torch.tensor(CHEBYSHEV, dtype=torch.float64))
            tracked = _chebyshev(
                torch.tensor(CHEBYSHEV, dtype=torch.float64, requires_grad=True)
            )
            for name in ("sag", "normal"):

                def value(geometry, v, name=name):
                    if name == "sag":
                        return geometry.sag(v, y)
                    return geometry._surface_normal(v, y)[0]

                expected = torch.func.jacrev(lambda v: value(plain, v))(x)
                result = torch.func.jacrev(lambda v: value(tracked, v))(x)
                assert torch.equal(result, expected)

    @pytest.mark.parametrize("context", ["numpy", "torch", "torch_no_grad"])
    def test_without_derivative_only_nonzero_terms_are_evaluated(
        self, context, monkeypatch
    ):
        """No derivative to request: the zero terms stay out, as before."""
        backend = "numpy" if context == "numpy" else "torch"
        with backend_state(backend, "float64"):
            if context == "torch_no_grad":
                coefficients = torch.tensor(
                    CHEBYSHEV, dtype=torch.float64, requires_grad=True
                )
            else:
                coefficients = be.array(CHEBYSHEV)
            geometry = _chebyshev(coefficients)
            calls = []
            original = ChebyshevPolynomialGeometry._chebyshev

            def counted(self, n, x):
                calls.append(n)
                return original(self, n, x)

            monkeypatch.setattr(ChebyshevPolynomialGeometry, "_chebyshev", counted)
            with torch.no_grad():
                geometry.sag(be.array(X), be.array(Y))
            assert len(calls) == 2 * NONZERO

    def test_numpy_values_are_the_sum_of_the_nonzero_terms(self):
        """On NumPy the zero terms stay out, so the bits are those of before."""
        with backend_state("numpy", "float64"):
            geometry = _chebyshev(be.array(CHEBYSHEV))
            x, y = be.array(X), be.array(Y)
            xn, yn = x / NORM, y / NORM
            r2 = x**2 + y**2
            expected = r2 / (
                geometry.radius
                * (1 + be.sqrt(1 - (1 + geometry.k) * r2 / geometry.radius**2))
            )
            for i in range(3):
                for j in range(3):
                    if CHEBYSHEV[i][j] != 0:
                        expected = expected + geometry.coefficients[
                            i, j
                        ] * geometry._chebyshev(i, xn) * geometry._chebyshev(j, yn)
            np.testing.assert_array_equal(geometry.sag(x, y), expected)


class TestParaxialRadius:
    @pytest.mark.parametrize("cls", [EvenAsphere, OddAsphere])
    def test_zero_r2_coefficient_keeps_radius_and_gets_gradient(self, cls):
        """A zero tensor r^2 coefficient gives R exactly and d/dc2 = -2 R^2."""
        with backend_state("torch", "float64"):
            coefficients = [0.0] * (cls._r2_index + 2)
            leaf = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            coefficients[cls._r2_index] = leaf
            geometry = cls(
                CoordinateSystem(), radius=-35.0, conic=0.0, coefficients=coefficients
            )
            radius = geometry.paraxial_radius
            assert float(radius.detach()) == -35.0  # exactly the base radius
            (grad,) = torch.autograd.grad(radius, leaf)
            assert float(grad) == pytest.approx(-2 * 35.0**2, rel=1e-15)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_zero_r2_coefficient_without_derivative_returns_the_radius(self, backend):
        """No derivative to request: the radius object itself, as before."""
        with backend_state(backend, "float64"):
            grad_mode = _grad_mode_off if backend == "torch" else contextlib.nullcontext
            with grad_mode():
                geometry = EvenAsphere(
                    CoordinateSystem(),
                    radius=-35.0,
                    conic=0.0,
                    coefficients=[be.array(0.0), 1e-5],
                )
                assert geometry.paraxial_radius is geometry.radius

    def test_flat_base_keeps_infinite_radius(self):
        """A flat base keeps its infinite paraxial radius."""
        with backend_state("torch", "float64"):
            leaf = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            geometry = EvenAsphere(
                CoordinateSystem(), radius=be.inf, conic=0.0, coefficients=[leaf]
            )
            assert bool(be.isinf(geometry.paraxial_radius))
