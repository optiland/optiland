"""A direct SurfaceGroup.trace runs inside an evaluation epoch.

The epoch is a fast path, so a trace in an epoch must give the same bits as
the trace without an epoch (``SurfaceGroup._trace_surfaces``). The tests
compare bits with ``assert_bits_equal``.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
import optiland.surfaces.surface_group as surface_group
from optiland import _epoch, _instrument
from optiland.rays import RealRays
from optiland.samples.objectives import CookeTriplet, DoubleGauss
from optiland.samples.telescopes import HubbleTelescope

from .utils import assert_bits_equal

RAY_FIELDS = ("x", "y", "z", "L", "M", "N", "i", "opd")
RECORD_FIELDS = ("x", "y", "z", "L", "M", "N", "intensity", "aoi", "opd")

# The largest bundle that a direct trace traces without an epoch.
MAX_RAYS = surface_group._MAX_RAYS_WITHOUT_EPOCH


def _fan(n=MAX_RAYS + 1, angle=0.08, wavelength=0.55, height=4.0):
    """A fan of ``n`` rays in the y-z plane at one wavelength.

    The rays start at z = 0, at heights from -height to height, with angles
    from -angle to angle (radians) to the z axis.
    """
    ang = np.linspace(-angle, angle, n)
    return RealRays(
        be.array(np.zeros(n)),
        be.array(np.linspace(-height, height, n)),
        be.array(np.zeros(n)),
        be.array(np.zeros(n)),
        be.array(np.sin(ang)),
        be.array(np.cos(ang)),
        be.array(np.ones(n)),
        be.array(np.full(n, wavelength)),
    )


def _spy_epochs(monkeypatch, optic):
    """Record the open epoch at each trace of the surfaces of ``optic``."""
    seen = []
    group = optic.surfaces
    original = group._trace_surfaces

    def spy(*args, **kwargs):
        seen.append(_epoch.current())
        return original(*args, **kwargs)

    monkeypatch.setattr(group, "_trace_surfaces", spy)
    return seen


def test_direct_trace_runs_inside_an_epoch(set_test_backend, monkeypatch):
    """A bundle above the limit is traced in an epoch that closes after."""
    optic = CookeTriplet()
    seen = _spy_epochs(monkeypatch, optic)
    optic.surfaces.trace(_fan())
    assert len(seen) == 1
    assert seen[0] is not None
    assert _epoch.current() is None


def test_outer_epoch_is_reused(set_test_backend, monkeypatch):
    """A trace inside an open epoch uses that epoch and opens no other."""
    optic = CookeTriplet()
    seen = _spy_epochs(monkeypatch, optic)
    with _epoch.evaluation_epoch() as outer:
        optic.surfaces.trace(_fan())
        optic.surfaces.trace(_fan())
    assert seen == [outer, outer]


def test_small_direct_trace_opens_no_epoch(set_test_backend, monkeypatch):
    """A bundle at the limit is traced without an epoch."""
    optic = CookeTriplet()
    seen = _spy_epochs(monkeypatch, optic)
    optic.surfaces.trace(_fan(n=MAX_RAYS))
    assert seen == [None]


def test_each_direct_trace_gets_a_new_epoch(set_test_backend, monkeypatch):
    """No memoized value of one direct trace is kept for the next one."""
    optic = CookeTriplet()
    seen = _spy_epochs(monkeypatch, optic)
    optic.surfaces.trace(_fan())
    optic.surfaces.trace(_fan())
    assert len(seen) == 2
    assert seen[0] is not seen[1]


def test_direct_trace_memoizes_materials():
    """The epoch runs the regular material evaluation less often.

    NumPy only: in the Torch grad mode of the tests, ``be.array`` gives
    tensors that require gradients, and the material memo does not key such
    wavelengths, so the counts are equal.
    """
    be.set_backend("numpy")
    optic = CookeTriplet()
    with _instrument.counting() as plain:
        optic.surfaces._trace_surfaces(_fan(), 0, True)
    with _instrument.counting() as memoized:
        optic.surfaces.trace(_fan())
    assert 0 < memoized["material_n"] < plain["material_n"]


@pytest.mark.parametrize(
    "factory, rays",
    [
        (CookeTriplet, lambda: _fan()),
        (CookeTriplet, lambda: _fan(n=MAX_RAYS)),  # no epoch
        (CookeTriplet, lambda: _fan(angle=0.6, height=30.0)),  # clipped and missed
        (DoubleGauss, lambda: _fan(wavelength=0.4861)),
        (HubbleTelescope, lambda: _fan(angle=0.001, height=1000.0)),
    ],
    ids=["cooke", "cooke_small", "cooke_vignetted", "double_gauss", "hubble_mirrors"],
)
@pytest.mark.parametrize("record", [True, False])
def test_results_are_bit_identical_to_the_plain_trace(
    set_test_backend, factory, rays, record
):
    """The epoch is a fast path, so it must give the same bits as the trace
    without an epoch.
    """
    plain_optic = factory()
    plain = plain_optic.surfaces._trace_surfaces(rays(), 0, record)
    optic = factory()
    traced = optic.surfaces.trace(rays(), record=record)
    for name in RAY_FIELDS:
        assert_bits_equal(getattr(traced, name), getattr(plain, name))
    for surface, plain_surface in zip(
        optic.surfaces.surfaces, plain_optic.surfaces.surfaces, strict=True
    ):
        for name in RECORD_FIELDS:
            assert_bits_equal(getattr(surface, name), getattr(plain_surface, name))


def test_state_change_between_direct_traces_is_seen(set_test_backend):
    """A change of the optic between two direct traces applies to the second."""
    optic = CookeTriplet()
    before = be.to_numpy(optic.surfaces.trace(_fan()).y).copy()
    radius = 1.05 * float(be.to_numpy(optic.surfaces[1].geometry.radius))
    optic.updater.set_radius(radius, 1)
    after = optic.surfaces.trace(_fan()).y

    fresh = CookeTriplet()
    fresh.updater.set_radius(radius, 1)
    assert_bits_equal(after, fresh.surfaces._trace_surfaces(_fan(), 0, True).y)
    assert not np.array_equal(be.to_numpy(after), before, equal_nan=True)


def test_direct_trace_does_not_change_the_optic(set_test_backend, monkeypatch):
    """The debug guard finds no change of the optic in a direct trace.

    A direct trace opens its epoch without optics, so the guard cannot check
    that epoch. Inside an outer epoch that names the optic, the trace reuses
    the outer epoch, and the guard compares the optic before and after.
    """
    monkeypatch.setenv(_epoch.CHECK_ENV, "1")
    optic = CookeTriplet()
    with _epoch.evaluation_epoch((optic,)):
        optic.surfaces.trace(_fan())
        optic.surfaces.trace(_fan(), skip=1, record=False)
