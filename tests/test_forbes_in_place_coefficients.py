"""A coefficient tensor that changes in place reaches the next evaluation.

The Forbes geometries prepare their coefficients once per coefficient
change: they stack the coefficient tensors into one new tensor. A dict
assignment marks the preparation stale, but a change of a tensor in place
did not: a torch.optim step on a coefficient tensor, or a later
requires_grad_(), left the stack of the former values, so the sag did not
move (issue #1175).

Now each preparation keeps the version counter and the requires_grad flag
of each coefficient tensor, and the next evaluation prepares again when one
of them changed.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.forbes import (
    ForbesQ2dGeometry,
    ForbesQNormalSlopeGeometry,
    ForbesSurfaceConfig,
)

from .nr_implicit_test_utils import backend_state

torch = pytest.importorskip("torch")

# One zero coefficient each: it moves only with its zero-coefficient gradient.
QBFS = {0: 1.0e-3, 1: -5.0e-4, 2: 0.0}
Q2D = {("a", 0, 0): 1.0e-3, ("a", 1, 1): -4.0e-4, ("b", 2, 0): 0.0}
X = [3.0, -2.0, 0.5, 6.0]
Y = [1.0, 4.0, -0.2, -3.0]


def _geometry(terms):
    """A Q or Q-2D geometry (from the type of the keys) with these terms."""
    if isinstance(next(iter(terms)), tuple):
        cls = ForbesQ2dGeometry
    else:
        cls = ForbesQNormalSlopeGeometry
    config = ForbesSurfaceConfig(
        radius=-40.0, conic=-0.5, norm_radius=8.0, terms=dict(terms)
    )
    return cls(CoordinateSystem(), config)


def _tensors(base, requires_grad=False):
    return {
        key: torch.tensor(value, dtype=torch.float64, requires_grad=requires_grad)
        for key, value in base.items()
    }


def _points():
    return (torch.tensor(v, dtype=torch.float64) for v in (X, Y))


def _values(geometry):
    """The sag and the normal at the test points, without a graph."""
    x, y = _points()
    with torch.no_grad():
        return (geometry.sag(x, y), *geometry._surface_normal(x, y))


def _fresh_values(terms):
    """The values of a new geometry with the current values of ``terms``."""
    return _values(_geometry({k: v.detach().clone() for k, v in terms.items()}))


def _bits(tensor):
    return tensor.detach().contiguous().view(torch.int64)


def _assert_same(values, expected):
    for value, reference in zip(values, expected, strict=True):
        assert torch.equal(_bits(value), _bits(reference))


@pytest.mark.parametrize("base", [QBFS, Q2D], ids=["qbfs", "q2d"])
def test_change_in_place_reaches_the_next_evaluation(base):
    """The issue reproduction: change a coefficient tensor in place."""
    with backend_state("torch", "float64"):
        terms = _tensors(base, requires_grad=True)
        geometry = _geometry(terms)
        before = _values(geometry)
        key = next(iter(terms))
        with torch.no_grad():
            terms[key].add_(2.0e-3)
        after = _values(geometry)
        assert not torch.equal(after[0], before[0])
        _assert_same(after, _fresh_values(terms))


@pytest.mark.parametrize("base", [QBFS, Q2D], ids=["qbfs", "q2d"])
def test_change_of_a_view_in_place_reaches_the_next_evaluation(base):
    """A coefficient that is a view of a parameter vector shares its version."""
    with backend_state("torch", "float64"):
        vector = torch.tensor(list(base.values()), dtype=torch.float64)
        terms = dict(zip(base, vector.unbind(), strict=True))
        geometry = _geometry(terms)
        _values(geometry)
        with torch.no_grad():
            vector[0] += 2.0e-3
        _assert_same(_values(geometry), _fresh_values(terms))


@pytest.mark.parametrize("base", [QBFS, Q2D], ids=["qbfs", "q2d"])
def test_requires_grad_after_preparation_gives_the_gradient(base):
    """requires_grad_() after an evaluation: the next evaluation has a graph."""
    with backend_state("torch", "float64"):
        terms = _tensors(base)
        geometry = _geometry(terms)
        _values(geometry)
        for value in terms.values():
            value.requires_grad_()
        x, y = _points()
        grads = torch.autograd.grad(geometry.sag(x, y).sum(), list(terms.values()))
        assert all(float(g) != 0.0 for g in grads)


@pytest.mark.parametrize("base", [QBFS, Q2D], ids=["qbfs", "q2d"])
def test_sgd_on_coefficient_tensors_moves_the_sag(base):
    """A torch.optim.SGD loop on the coefficient tensors moves the surface."""
    with backend_state("torch", "float64"):
        terms = _tensors(base, requires_grad=True)
        geometry = _geometry(terms)
        optimizer = torch.optim.SGD(list(terms.values()), lr=1.0e-3)
        x, y = _points()
        target = torch.full_like(x, 0.5)
        sags = []
        for _ in range(3):
            optimizer.zero_grad()
            sag = geometry.sag(x, y)
            sags.append(sag.detach().clone())
            loss = ((sag - target) ** 2).sum()
            loss.backward()
            optimizer.step()
        assert not torch.equal(sags[1], sags[0])
        assert not torch.equal(sags[2], sags[1])
        # The coefficient that started at zero moved too.
        assert float(list(terms.values())[-1].detach()) != 0.0
        _assert_same(_values(geometry), _fresh_values(terms))


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("base", [QBFS, Q2D], ids=["qbfs", "q2d"])
def test_no_new_preparation_without_a_change(base, backend, monkeypatch):
    """Evaluations without a change use the preparation of before."""
    with backend_state(backend, "float64"):
        terms = dict(base) if backend == "numpy" else _tensors(base, True)
        geometry = _geometry(terms)
        calls = []
        original = type(geometry)._prepare_coeffs

        def counted(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(type(geometry), "_prepare_coeffs", counted)
        x, y = be.array(X), be.array(Y)
        for _ in range(3):
            geometry.sag(x, y)
            geometry._surface_normal(x, y)
        # Q prepares at the first evaluation, Q-2D in __init__.
        assert len(calls) == (1 if base is QBFS else 0)
        if backend == "numpy":
            assert geometry._watched == ()
