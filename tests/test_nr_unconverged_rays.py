"""A ray that does not converge within max_iter is a ray failure (#1171).

Before, such a ray kept its last iterate, a point off the surface, and the
trace went on from there as if the ray had hit the surface. Now its distance
is NaN, as for a ray that misses a conic. The generic loop and the compiled
trace plan apply the same rule.
"""

from __future__ import annotations

import contextlib
import re

import numpy as np
import pytest

import optiland.backend as be
import optiland.surfaces._chunked_trace as chunked
import optiland.surfaces._trace_plan as plan
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import EvenAsphere
from optiland.geometries.newton_raphson import _DistanceSolveResult
from optiland.optic import Optic
from optiland.rays import RealRays

from .nr_implicit_test_utils import backend_state
from .utils import assert_bits_equal

# The strong asphere of #1171.
COEFFICIENTS = [0.0, 1.5e-4, 0.0]


def _asphere(max_iter=100, coefficients=None):
    return EvenAsphere(
        CoordinateSystem(),
        radius=15.0,
        conic=0.0,
        coefficients=COEFFICIENTS if coefficients is None else coefficients,
        tol=1e-6,
        max_iter=max_iter,
    )


def _fan(n=41):
    """Parallel rays at 20 deg from z = -5, across the asphere."""
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


def test_a_ray_that_does_not_converge_gets_the_distance_nan(set_test_backend):
    """After max_iter steps, a ray above the tolerance gets NaN."""
    rays = _fan()
    with _no_grad():
        stopped = _asphere(max_iter=2)._solve_distance_primal(rays)
        full = _asphere()._solve_distance_primal(rays)
    t, reached = _np(stopped.t), _np(stopped.reached)
    assert reached.any() and not reached.all()
    np.testing.assert_array_equal(np.isnan(t), ~reached)
    np.testing.assert_array_equal(_np(stopped.converged), reached)
    # A converged ray takes no more steps, so it keeps its distance.
    assert_bits_equal(t[reached], _np(full.t)[reached])
    # The residual of a stopped ray is its last one: above the tolerance, or
    # NaN for a ray that misses the asphere.
    residual = _np(stopped.residual)[~reached]
    assert np.all(np.isnan(residual) | (np.abs(residual) >= 1e-6))
    assert np.isfinite(residual).any()


def _lens(max_iter=100):
    """The system of #1171: a strong asphere at a 20 deg field."""
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(index=1, radius=be.inf, thickness=3.0, is_stop=True)
    lens.surfaces.add(
        index=2,
        surface_type="even_asphere",
        radius=15.0,
        conic=0.0,
        coefficients=COEFFICIENTS,
        material="N-BK7",
        thickness=8.0,
    )
    lens.surfaces.add(index=3, radius=-40.0, thickness=30.0)
    lens.surfaces.add(index=4)
    lens.set_aperture(aperture_type="EPD", value=18.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.fields.add(y=20.0)
    lens.wavelengths.add(value=0.5876, is_primary=True)
    lens.surfaces[2].geometry.max_iter = max_iter
    return lens


@pytest.mark.parametrize("enabled", [True, False], ids=["plan", "generic"])
def test_no_lit_ray_comes_from_a_point_off_the_asphere(monkeypatch, enabled):
    """The reproduction of #1171, with and without the compiled trace plan."""
    monkeypatch.setattr(plan, "_ENABLED", enabled)
    monkeypatch.setattr(chunked, "_MIN_RAYS", 10**12)
    lens = _lens()
    lens.trace(0.0, 1.0, 0.5876, num_rays=64, distribution="hexapolar")
    surfaces = lens.surfaces
    geometry = surfaces[2].geometry
    x, y, z = surfaces.x[2], surfaces.y[2], surfaces.z[2] - 3.0
    residual = np.abs(geometry.sag(x, y) - z)
    lit = np.isfinite(residual) & (surfaces.intensity[-1] > 0)
    assert lit.sum() > 11000
    assert not np.any(lit & (residual > geometry.tol))


def test_plan_and_generic_trace_give_the_same_failures(monkeypatch):
    """The kernel sets NaN for the same rays as the generic loop."""
    monkeypatch.setattr(chunked, "_MIN_RAYS", 10**12)
    traced = {}
    for enabled in (True, False):
        monkeypatch.setattr(plan, "_ENABLED", enabled)
        lens = _lens(max_iter=3)
        lens.trace(0.0, 1.0, 0.5876, num_rays=32, distribution="hexapolar")
        traced[enabled] = np.array(lens.surfaces.z[2], copy=True)
    assert np.isnan(traced[True]).any()
    assert_bits_equal(traced[True], traced[False])


def test_the_warning_counts_the_rays_that_did_not_converge():
    """With gradients, such a ray is rejected and reported as non-converged.

    A ray with a NaN residual (it misses the asphere) stays non-finite.
    """
    torch = pytest.importorskip("torch")
    with backend_state("torch", "float64"):
        c = torch.tensor(1.5e-4, dtype=torch.float64, requires_grad=True)
        geometry = _asphere(max_iter=2, coefficients=[0.0, c, 0.0])
        rays = _fan()
        with pytest.warns(RuntimeWarning, match="non-converged") as record:
            t = geometry.distance(rays)
        with torch.no_grad():
            solved = geometry._solve_distance_primal(rays)
        reached = solved.reached
        finite = torch.isfinite(solved.residual)
        message = str(record[0].message)
        stalled = int(re.search(r"(\d+) non-converged \(max residual", message)[1])
        missed = int(re.search(r"(\d+) with non-finite state", message)[1])
        assert stalled == int((~reached & finite).sum()) > 0
        assert missed == int((~finite).sum()) > 0
        assert torch.isnan(t[~reached]).all()
        assert torch.isfinite(t[reached]).all()
        (grad,) = torch.autograd.grad(t[reached].sum(), c)
        assert torch.isfinite(grad)


def test_a_result_without_the_reached_mask_keeps_the_old_categories():
    """A result built without ``reached`` (for example in a test) still works."""
    with backend_state("numpy", "float64"):
        geometry = _asphere()
        rays = _fan(3)
        with _no_grad():
            solved = geometry._solve_distance_primal(rays)
        t = _np(solved.t)
        result = _DistanceSolveResult(
            t=t,
            residual=np.array([0.0, 0.5, np.nan]),
            converged=np.array([True, False, False]),
            iterations=1,
        )
        state = geometry._classify_final_roots(result, rays)
        np.testing.assert_array_equal(state.nonconverged, [False, True, False])
        np.testing.assert_array_equal(state.nonfinite, [False, False, True])
