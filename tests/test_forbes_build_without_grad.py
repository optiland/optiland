"""A Forbes surface built or scaled inside torch.no_grad() keeps its gradients.

The Forbes geometries prepare their coefficients once per coefficient
change: they stack the coefficient tensors into one tensor. A stack made
while gradient recording is off has no graph, so the sag and the normal of
a later evaluation with gradients did not depend on the coefficients.
ForbesQ2dGeometry prepared its coefficients in __init__ and in scale() in
the gradient mode of the caller. A Q-2D surface built inside torch.no_grad()
therefore lost the gradients of its coefficients (issue #1174). The Q
surface prepares at the first evaluation, under torch.enable_grad().

Now the Q-2D surface prepares under torch.enable_grad() in __init__, and
scale() lets the next evaluation prepare, as the Q surface does.
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

Q2D = {("a", 0, 0): 1.0e-3, ("a", 1, 1): -4.0e-4, ("b", 2, 0): 2.0e-4}
QBFS = {0: 1.0e-3, 1: -5.0e-4, 2: 2.0e-4}
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


def _leaves(base):
    return {
        key: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for key, value in base.items()
    }


def _value(geometry, output):
    x = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(Y, dtype=torch.float64)
    if output == "sag":
        return geometry.sag(x, y).sum()
    nx, ny, nz = geometry._surface_normal(x, y)
    return (0.7 * nx - 1.3 * ny + 0.4 * nz).sum()


def _gradients(geometry, leaves, output):
    value = _value(geometry, output)
    assert value.requires_grad
    return torch.autograd.grad(value, list(leaves.values()))


def _bits(tensor):
    return tensor.detach().contiguous().view(torch.int64)


@pytest.mark.parametrize("output", ["sag", "normal"])
@pytest.mark.parametrize("base", [Q2D, QBFS], ids=["q2d", "qbfs"])
def test_surface_built_in_no_grad_keeps_coefficient_gradients(base, output):
    """The issue reproduction: build in no_grad, evaluate with gradients."""
    with backend_state("torch", "float64"):
        reference_leaves = _leaves(base)
        expected = _gradients(_geometry(reference_leaves), reference_leaves, output)
        leaves = _leaves(base)
        with torch.no_grad():
            geometry = _geometry(leaves)
        result = _gradients(geometry, leaves, output)
        for value, reference in zip(result, expected, strict=True):
            assert torch.equal(_bits(value), _bits(reference))
        assert any(float(value) != 0.0 for value in result)


@pytest.mark.parametrize("output", ["sag", "normal"])
@pytest.mark.parametrize("built_in_no_grad", [False, True])
def test_q2d_scaled_in_no_grad_keeps_coefficient_gradients(built_in_no_grad, output):
    """scale() inside no_grad: the scaled coefficients can get gradients later.

    The scaled coefficients are new tensors without a graph. A caller that
    then lets them require grad must get their gradients: the preparation
    may not happen before, without a graph.
    """
    with backend_state("torch", "float64"):
        leaves = _leaves(Q2D)
        if built_in_no_grad:
            with torch.no_grad():
                geometry = _geometry(leaves)
        else:
            geometry = _geometry(leaves)
        with torch.no_grad():
            geometry.scale(2.0)
        scaled = geometry.freeform_coeffs
        for value in scaled.values():
            value.requires_grad_()
        result = _gradients(geometry, scaled, output)

        # A surface built with the scaled values and the scaled parameters.
        reference = {
            key: value.detach().clone().requires_grad_()
            for key, value in scaled.items()
        }
        config = ForbesSurfaceConfig(
            radius=-80.0, conic=-0.5, norm_radius=16.0, terms=dict(reference)
        )
        expected = _gradients(
            ForbesQ2dGeometry(CoordinateSystem(), config), reference, output
        )
        for value, reference_value in zip(result, expected, strict=True):
            assert torch.equal(_bits(value), _bits(reference_value))
        assert any(float(value) != 0.0 for value in result)


def test_q2d_prepares_in_init_and_lazily_after_scale():
    """The coefficient lists exist after __init__; scale() marks them stale."""
    with backend_state("torch", "float64"):
        geometry = _geometry(_leaves(Q2D))
        assert not geometry._coeffs_dirty
        assert len(geometry.ams_coeffs) == 2  # families m = 1 and m = 2
        with torch.no_grad():
            geometry.scale(2.0)
        assert geometry._coeffs_dirty
        x, y = be.array([1.0]), be.array([2.0])
        geometry.sag(x, y)
        assert not geometry._coeffs_dirty
