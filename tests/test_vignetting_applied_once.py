"""Field vignetting must be applied exactly once, for every aimer and entry point.

Before the fix, ``trace_generic`` pre-scaled the pupil and the paraxial aimer
scaled it again (the pupil was compressed by ``(1 - v)**2``), while
distribution traces under iterative / robust aiming were not vignetted at all.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.apodization import GaussianApodization
from optiland.distribution import create_distribution
from optiland.samples.objectives import CookeTriplet

from .utils import assert_allclose

VY = 0.3
PY = np.array([-1.0, 0.0, 1.0])
AIMING_MODES = ["paraxial", "iterative", "robust"]


def _cooke(mode, vy=0.0):
    optic = CookeTriplet()
    optic.fields.fields[-1].vy = vy
    optic.ray_tracer.set_aiming(mode)
    return optic


def _stop_y(optic):
    return be.to_numpy(optic.surfaces[optic.surfaces.stop_index].y).ravel()


def _generic_stop_y(optic, py):
    n = len(py)
    optic.trace_generic(
        be.zeros(n), be.ones(n), be.zeros(n), be.array(py), optic.primary_wavelength
    )
    return _stop_y(optic)


def _distribution_stop_y(optic):
    d = create_distribution("line_y")
    d.generate_points(3)
    optic.trace(0.0, 1.0, optic.primary_wavelength, distribution=d)
    order = np.argsort(be.to_numpy(d.y))
    return _stop_y(optic)[order]


@pytest.mark.parametrize("mode", AIMING_MODES)
def test_generic_trace_vignettes_once(set_test_backend, mode):
    """A vignetted field traces like an unvignetted one with a (1 - v) pupil."""
    vignetted = _generic_stop_y(_cooke(mode, VY), PY)
    shrunk = _generic_stop_y(_cooke(mode), PY * (1 - VY))
    assert_allclose(vignetted, shrunk, rtol=0, atol=1e-9)


@pytest.mark.parametrize("mode", AIMING_MODES)
def test_distribution_trace_vignettes_once(set_test_backend, mode):
    """Distribution traces apply the same vignetting under every aimer."""
    vignetted = _distribution_stop_y(_cooke(mode, VY))
    shrunk = _generic_stop_y(_cooke(mode), PY * (1 - VY))
    assert_allclose(vignetted, shrunk, rtol=0, atol=1e-9)


@pytest.mark.parametrize("mode", AIMING_MODES)
def test_generic_and_distribution_traces_agree(set_test_backend, mode):
    """No entry point may apply the vignetting twice or skip it."""
    optic = _cooke(mode, VY)
    generic = _generic_stop_y(optic, PY)
    distribution = _distribution_stop_y(optic)
    assert_allclose(generic, distribution, rtol=0, atol=1e-9)


@pytest.mark.parametrize("mode", AIMING_MODES)
def test_apodization_is_evaluated_on_the_vignetted_pupil(set_test_backend, mode):
    """A ray at Py = 1 of a field with vy crosses the pupil at 1 - vy."""

    def launch_intensity(vy, py):
        optic = _cooke(mode, vy)
        optic.updater.set_apodization(GaussianApodization(sigma=0.5))
        n = len(py)
        rays = optic.ray_tracer.ray_generator.generate_rays(
            be.zeros(n),
            be.ones(n),
            be.zeros(n),
            be.array(py),
            optic.primary_wavelength,
        )
        return be.to_numpy(rays.i)

    vignetted = launch_intensity(VY, PY)
    shrunk = launch_intensity(0.0, PY * (1 - VY))
    assert_allclose(vignetted, shrunk, rtol=0, atol=1e-12)
    # the edge ray is brighter than the unvignetted pupil edge
    assert vignetted[2] > launch_intensity(0.0, PY)[2]


def test_apply_vignetting_uses_nearest_field_factor(set_test_backend):
    optic = CookeTriplet()
    optic.fields.fields[-1].vx = 0.2
    optic.fields.fields[-1].vy = 0.4
    Px, Py = optic.fields.apply_vignetting(
        be.array([0.0, 1.0]), be.array([1.0, 1.0]), be.array([1.0, 1.0]),
        be.array([1.0, 1.0]),
    )
    assert_allclose(Px, [0.8, 0.8])
    assert_allclose(Py, [0.6, 0.6])
    Px, Py = optic.fields.apply_vignetting(0.0, 0.0, 1.0, 1.0)
    assert_allclose(Px, 1.0)
    assert_allclose(Py, 1.0)


def test_apply_vignetting_is_identity_without_vignetting(set_test_backend):
    optic = CookeTriplet()
    Px, Py = be.array([0.5, -1.0]), be.array([0.25, 1.0])
    out_x, out_y = optic.fields.apply_vignetting(0.0, 1.0, Px, Py)
    assert out_x is Px
    assert out_y is Py
