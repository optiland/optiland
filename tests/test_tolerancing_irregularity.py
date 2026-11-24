"""Tests for the irregularity tolerance type (ISO 10110-5 B and C)."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.geometries import SagErrorGeometry
from optiland.samples.objectives import CookeTriplet
from optiland.tolerancing import SensitivityAnalysis, Tolerancing, tolerance_registry
from optiland.tolerancing.base import Phase
from optiland.tolerancing.monte_carlo import MonteCarlo
from optiland.tolerancing.types import IrregularityTolerance
from tests.utils import assert_allclose

SURFACE = 3


def _np(value):
    return np.asarray(be.to_numpy(value))


@pytest.fixture
def tolerancing(set_test_backend):
    """Tolerancing on a Cooke triplet with an RMS spot size operand."""
    optic = CookeTriplet()
    tol = Tolerancing(optic)
    tol.add_operand(
        "rms_spot_size",
        {
            "optic": optic,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 0.0,
            "num_rays": 5,
            "wavelength": 0.55,
            "distribution": "hexapolar",
        },
    )
    return tol


def _sag_error_pv(geometry, original, h):
    """Peak-to-valley sag error on a polar grid that holds the extrema."""
    # u = rho^2 on a grid that contains 0, 1/2 and 1; phi in steps of 1 deg
    r = h * np.sqrt(np.linspace(0.0, 1.0, 401))
    phi = np.radians(np.arange(0.0, 360.0, 1.0))
    rr, pp = np.meshgrid(r, phi)
    x, y = be.array((rr * np.cos(pp)).ravel()), be.array((rr * np.sin(pp)).ravel())
    error = _np(geometry.sag(x, y) - original.sag(x, y))
    return error.max() - error.min()


def _trace(optic):
    rays = optic.trace(0.0, 1.0, 0.55, 64, "hexapolar")
    return [_np(getattr(rays, a)).copy() for a in "xyzLMN"]


def test_registered():
    assert "irregularity" in tolerance_registry
    assert tolerance_registry.get("irregularity") is IrregularityTolerance


def test_attributes(tolerancing):
    tol = tolerancing.add_tolerance("irregularity", surface_number=SURFACE, limit=1)
    assert tol.phase is Phase.FORM
    assert tol.type.units == "fringes"
    assert tol.label == f"Irregularity S{SURFACE}"
    tol = tolerancing.add_tolerance(
        "irregularity", surface_number=SURFACE, limit=1, kind="C"
    )
    assert tol.label == f"Rot. irregularity S{SURFACE}"


@pytest.mark.parametrize("kind", ["B", "C"])
@pytest.mark.parametrize("fringes", [-2.0, 0.5, 3.0])
def test_peak_to_valley_is_half_wave_per_fringe(tolerancing, kind, fringes):
    optic = tolerancing.optic
    tol = tolerancing.add_tolerance(
        "irregularity", surface_number=SURFACE, limit=5, kind=kind
    )
    original = optic.surfaces[SURFACE].geometry
    h = tol.type.semi_aperture
    tol.type.apply(fringes)
    geometry = optic.surfaces[SURFACE].geometry
    assert isinstance(geometry, SagErrorGeometry)
    assert geometry.base is original
    pv = _sag_error_pv(geometry, original, h)
    assert_allclose(pv, abs(fringes) * 0.6328e-3 / 2, rtol=1e-9, atol=0)
    # no vertex offset
    assert_allclose(geometry.sag(0.0, 0.0), original.sag(0.0, 0.0), atol=1e-15)
    tol.reset()


def test_astigmatism_angle_and_wavelength(tolerancing):
    tol = IrregularityTolerance(
        tolerancing.optic,
        SURFACE,
        test_wavelength=0.5461,
        semi_aperture=4.0,
        angle=45.0,
    )
    c = tol.coefficients(2.0)
    # 2 fringes = 0.5461 um of sag PV; Z6 amplitude is half of it
    assert_allclose(c[5], 0.5461e-3 / 2, rtol=1e-12)
    assert abs(c[4]) < 1e-18
    assert sum(abs(v) for i, v in enumerate(c) if i != 5) < 1e-18


def test_rotational_kind_is_symmetric_spherical(tolerancing):
    tol = IrregularityTolerance(tolerancing.optic, SURFACE, kind="C")
    c = tol.coefficients(1.5)
    assert_allclose(c[8], 1.5 * 0.6328e-3 / 2 / 1.5, rtol=1e-12)
    assert_allclose(c[0], -c[8], rtol=1e-12)
    tol.apply(1.5)
    assert tolerancing.optic.surfaces[SURFACE].geometry.is_symmetric
    tol.reset()


def test_default_semi_aperture_matches_fringes(tolerancing):
    optic = tolerancing.optic
    irregularity = tolerancing.add_tolerance(
        "irregularity", surface_number=SURFACE, limit=1
    )
    fringes = tolerancing.add_tolerance("fringes", surface_number=SURFACE, limit=1)
    assert irregularity.type.semi_aperture == fringes.type.semi_aperture
    optic.surfaces[SURFACE].set_semi_aperture(4.25)
    assert IrregularityTolerance(optic, SURFACE).semi_aperture == 4.25


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"kind": "A"}, "kind"),
        ({"kind": "C", "angle": 10.0}, "angle"),
        ({"semi_aperture": 0.0}, "semi_aperture"),
    ],
)
def test_invalid_arguments(tolerancing, kwargs, match):
    with pytest.raises(ValueError, match=match):
        IrregularityTolerance(tolerancing.optic, SURFACE, **kwargs)


def test_apply_and_reset_restore_the_geometry_object(tolerancing):
    optic = tolerancing.optic
    original = optic.surfaces[SURFACE].geometry
    nominal = _trace(optic)
    tol = tolerancing.add_tolerance("irregularity", surface_number=SURFACE, limit=5)
    tol.type.apply(4.0)
    perturbed = _trace(optic)
    assert np.nanmax(np.abs(perturbed[0] - nominal[0])) > 1e-6
    # a second apply wraps the original, not the previous sag error
    tol.type.apply(-4.0)
    assert optic.surfaces[SURFACE].geometry.base is original
    tol.reset()
    assert optic.surfaces[SURFACE].geometry is original
    for a, b in zip(_trace(optic), nominal, strict=True):
        np.testing.assert_array_equal(a, b)
    tol.reset()  # a second reset changes nothing
    assert optic.surfaces[SURFACE].geometry is original


def test_together_with_fringes_on_the_same_surface(tolerancing):
    optic = tolerancing.optic
    original = optic.surfaces[SURFACE].geometry
    nominal_radius = _np(original.radius).item()
    tolerancing.add_tolerance("irregularity", surface_number=SURFACE, limit=2)
    tolerancing.add_tolerance("fringes", surface_number=SURFACE, limit=2)
    for perturbation in tolerancing.ordered_perturbations():
        perturbation.apply()
    assert isinstance(optic.surfaces[SURFACE].geometry, SagErrorGeometry)
    tolerancing.reset()
    assert optic.surfaces[SURFACE].geometry is original
    assert _np(original.radius).item() == nominal_radius


@pytest.mark.parametrize("kind", ["B", "C"])
def test_sensitivity_analysis(tolerancing, kind):
    optic = tolerancing.optic
    original = optic.surfaces[SURFACE].geometry
    nominal = tolerancing.evaluate()[0]
    tolerancing.add_tolerance(
        "irregularity", surface_number=SURFACE, limit=10, kind=kind
    )
    sa = SensitivityAnalysis(tolerancing)
    sa.run()
    df = sa.get_results()
    assert_allclose(df["perturbation_value"].to_numpy(), np.array([-10.0, 10.0]))
    spots = df["0: rms spot size"].to_numpy()
    assert np.all(np.isfinite(spots))
    assert np.all(spots > nominal * 1.01)
    assert optic.surfaces[SURFACE].geometry is original


def test_monte_carlo(tolerancing):
    optic = tolerancing.optic
    original = optic.surfaces[SURFACE].geometry
    nominal = tolerancing.evaluate()[0]
    tolerancing.add_tolerance(
        "irregularity", surface_number=SURFACE, limit=10, distribution="end_point"
    )
    mc = MonteCarlo(tolerancing)
    mc.run(4)
    df = mc.get_results()
    assert np.all(np.abs(df[f"Irregularity S{SURFACE}"].to_numpy()) == 10.0)
    spots = df["0: rms spot size"].to_numpy()
    assert np.all(np.isfinite(spots))
    assert np.all(spots > nominal * 1.01)
    assert optic.surfaces[SURFACE].geometry is original


def test_failure_restores_the_geometry(tolerancing, monkeypatch):
    optic = tolerancing.optic
    original = optic.surfaces[SURFACE].geometry
    tolerancing.add_tolerance("irregularity", surface_number=SURFACE, limit=3)
    seen = []

    def fail():
        seen.append(optic.surfaces[SURFACE].geometry)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(tolerancing, "evaluate", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        MonteCarlo(tolerancing).run(2)
    assert isinstance(seen[0], SagErrorGeometry)
    assert optic.surfaces[SURFACE].geometry is original
