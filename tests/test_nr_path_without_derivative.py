"""Torch takes the differentiable Newton-Raphson path only for a derivative.

Torch records gradients by default. Before, every Torch trace outside
``torch.no_grad()`` ran the root classification, the implicit correction and
the rejection check of ``NewtonRaphsonGeometry.distance()``, also when no
tensor required grad. Now these run only if a ray position or direction, or
a parameter of the surface, requires grad, or inside a ``torch.func``
transform or a forward-mode level. Else Torch returns the result of the
primal solve, as NumPy does.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import EvenAsphere, ZernikePolynomialGeometry
from optiland.geometries import newton_raphson as nr
from optiland.geometries._derivatives import requires_grad_in
from optiland.geometries.forbes import ForbesQNormalSlopeGeometry, ForbesSurfaceConfig
from optiland.rays import RealRays

from .nr_implicit_test_utils import backend_state

torch = pytest.importorskip("torch")

Y = np.linspace(-3.0, 3.0, 7)


@contextlib.contextmanager
def _torch():
    """Torch in float64, with Optiland's grad mode off.

    Other tests can leave the grad mode on. Then new tensors require grad,
    and each geometry has a parameter that can carry a derivative.
    """
    with backend_state("torch", "float64"):
        previous = be.grad_mode.requires_grad
        be.grad_mode.disable()
        try:
            yield
        finally:
            if previous:
                be.grad_mode.enable()


def _rays(requires_grad=False):
    """Rays parallel to the axis, from z = -5, at the heights Y."""
    n = Y.size
    rays = RealRays(
        np.zeros(n),
        Y,
        np.full(n, -5.0),
        np.zeros(n),
        np.zeros(n),
        np.ones(n),
        np.ones(n),
        np.full(n, 0.55),
    )
    if requires_grad:
        rays.y.requires_grad_(True)
    return rays


def _asphere(c2=1e-4):
    """An even asphere whose only coefficient multiplies r^2."""
    return EvenAsphere(CoordinateSystem(), radius=20.0, coefficients=[c2])


def _count_classifications(monkeypatch):
    """Count the calls of the root classification of the differentiable path."""
    calls = []
    original = nr.NewtonRaphsonGeometry._classify_final_roots

    def spy(self, *args, **kwargs):
        calls.append(type(self).__name__)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(nr.NewtonRaphsonGeometry, "_classify_final_roots", spy)
    return calls


def test_default_grad_mode_without_a_tracked_tensor_takes_the_primal_path(
    monkeypatch,
):
    """The result has the bits of the result under torch.no_grad()."""
    with _torch():
        assert torch.is_grad_enabled()
        geometry = _asphere()
        with torch.no_grad():
            expected = geometry.distance(_rays())
        calls = _count_classifications(monkeypatch)
        t = geometry.distance(_rays())
    assert calls == []
    assert not t.requires_grad
    assert torch.equal(t, expected)


@pytest.mark.parametrize("tracked", ["ray", "coefficient", "radius"])
def test_a_tracked_tensor_selects_the_differentiable_path(tracked, monkeypatch):
    """A ray field or a parameter that requires grad needs the derivative."""
    with _torch():
        leaf = torch.tensor(1e-4, dtype=torch.float64, requires_grad=True)
        geometry = _asphere(leaf if tracked == "coefficient" else 1e-4)
        if tracked == "radius":
            geometry.radius = torch.tensor(
                20.0, dtype=torch.float64, requires_grad=True
            )
        calls = _count_classifications(monkeypatch)
        t = geometry.distance(_rays(requires_grad=tracked == "ray"))
    assert calls == ["EvenAsphere"]
    assert t.requires_grad


def test_gradient_of_a_coefficient():
    """t = 5 + sag(y), so dt/dc2 = sum of y^2 over the rays."""
    with _torch():
        leaf = torch.tensor(1e-4, dtype=torch.float64, requires_grad=True)
        t = _asphere(leaf).distance(_rays())
        (grad,) = torch.autograd.grad(t.sum(), leaf)
    assert float(grad) == pytest.approx(float(np.sum(Y**2)), rel=1e-12)


def test_parameters_in_lists_dicts_and_objects_are_found():
    """The search looks into the containers that the geometries use."""
    with _torch():

        def leaf(value):
            return torch.tensor(value, dtype=torch.float64, requires_grad=True)

        cs = CoordinateSystem()
        # A list of coefficients.
        assert _asphere(leaf(1e-4))._derivative_requested(_rays())
        # The coefficients of the Zernike object of the geometry.
        zernike = ZernikePolynomialGeometry(
            cs, radius=20.0, coefficients=leaf([0.0, 1e-3]), norm_radius=5.0
        )
        assert zernike._derivative_requested(_rays())
        # A dict of Forbes terms.
        forbes = ForbesQNormalSlopeGeometry(
            cs,
            ForbesSurfaceConfig(radius=20.0, terms={0: leaf(1e-3)}, norm_radius=5.0),
        )
        assert forbes._derivative_requested(_rays())
        # The same geometries without a tracked tensor.
        assert not _asphere()._derivative_requested(_rays())
        assert not ZernikePolynomialGeometry(
            cs, radius=20.0, coefficients=[0.0, 1e-3], norm_radius=5.0
        )._derivative_requested(_rays())


def test_the_coordinate_system_does_not_count():
    """The rays are already in the local frame of the surface."""
    with _torch():
        geometry = _asphere()
        geometry.cs.x = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        assert not geometry._derivative_requested(_rays())


def test_transforms_and_forward_mode_keep_the_differentiable_path():
    """Inside a transform or a dual level, the decision is always True."""
    with _torch():
        geometry = _asphere()
        with torch.autograd.forward_ad.dual_level():
            assert geometry._derivative_requested(_rays())
        decisions = []

        def f(x):
            decisions.append(geometry._derivative_requested(_rays()))
            return 2.0 * x

        torch.func.grad(f)(torch.tensor(1.0, dtype=torch.float64))
    assert decisions == [True]


def test_a_parameter_that_the_search_does_not_find_keeps_its_gradient():
    """The primal solve records its iterations: the unrolled loop gives it."""

    class _Box:
        def __init__(self, value):
            self.value = value

    class _Deep(EvenAsphere):
        """Keeps its r^2 coefficient 4 levels deep, below the search."""

        def __init__(self, c2):
            super().__init__(CoordinateSystem(), radius=20.0, coefficients=[0.0])
            self.box = _Box(_Box(_Box(_Box(c2))))

        @property
        def coefficients(self):
            return [self.box.value.value.value.value]

        @coefficients.setter
        def coefficients(self, value):
            pass

    with _torch():
        leaf = torch.tensor(1e-4, dtype=torch.float64, requires_grad=True)
        geometry = _Deep(leaf)
        assert not geometry._derivative_requested(_rays())
        t = geometry.distance(_rays())
        (grad,) = torch.autograd.grad(t.sum(), leaf)
    assert float(grad) == pytest.approx(float(np.sum(Y**2)), rel=1e-9)


def test_requires_grad_in():
    leaf = torch.zeros((), requires_grad=True)
    plain = torch.zeros(())

    class _Holder:
        pass

    holder = _Holder()
    holder.inner = {"a": [leaf]}
    assert requires_grad_in({"holder": holder})
    assert not requires_grad_in({"holder": holder}, depth=2)
    assert not requires_grad_in({"cs": leaf}, skip=("cs",))
    assert not requires_grad_in({"x": np.zeros(3), "s": "text", "n": 3, "t": plain})
    assert requires_grad_in({"pair": (plain, leaf)})
