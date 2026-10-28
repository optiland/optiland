"""Deterministic evaluation-count tests (decision V15).

Counts come from :mod:`optiland._instrument` and involve no timing, so they
catch regressions of the engine speed work without wall-clock noise. When a
speed PR legitimately lowers a count, it updates the expected value here.

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from benchmarks.optimization.problems import PROBLEMS
from optiland import _epoch, _instrument

pytestmark = pytest.mark.filterwarnings("ignore")

# Steady-state counts of one ``sum_squared`` evaluation (after a warm-up one),
# identical on numpy and torch. One ``optic.trace`` per rms_spot_size operand.
EXPECTED = {
    "cooke": {"trace": 9, "trace_generic": 0, "paraxial_trace": 28, "material_n": 413},
    "double_gauss": {
        "trace": 9,
        "trace_generic": 0,
        "paraxial_trace": 55,
        "material_n": 1039,
    },
    "freeform": {
        "trace": 9,
        "trace_generic": 0,
        "paraxial_trace": 1,
        "material_n": 141,
    },
    "scaling_10": {
        "trace": 3,
        "trace_generic": 0,
        "paraxial_trace": 1,
        "material_n": 111,
    },
}


def _evaluation_counts(problem) -> dict[str, int]:
    problem.sum_squared()  # warm caches
    with _instrument.counting() as counts:
        problem.sum_squared()
    return dict(counts)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_merit_evaluation_counts(set_test_backend, name):
    _, problem = PROBLEMS[name]()
    assert _evaluation_counts(problem) == EXPECTED[name]


def test_wide_asphere_trace_count(set_test_backend):
    # Iterative ray aiming differs between backends, so only traces are pinned.
    _, problem = PROBLEMS["wide_asphere"]()
    counts = _evaluation_counts(problem)
    assert counts["trace"] == 15
    assert counts["trace_generic"] == 0


def test_double_gauss_one_trace_per_field_wavelength(set_test_backend):
    # 3 fields x 3 wavelengths (the EFL operand traces no real rays).
    _, problem = PROBLEMS["double_gauss"]()
    assert _evaluation_counts(problem)["trace"] == 9


def test_fd_jacobian_cost_is_linear_in_variables(set_test_backend):
    from optiland.optimization import LeastSquares

    _, problem = PROBLEMS["scaling_10"]()
    optimizer = LeastSquares(problem)
    x0 = be.to_numpy([var.value for var in problem.variables]).astype(float)
    optimizer._compute_residuals_vector(x0)  # warm caches
    with _instrument.counting() as one:
        optimizer._compute_residuals_vector(x0)
    with _instrument.counting() as jac:
        optimizer._compute_residuals_vector(x0)
        for j in range(len(x0)):
            x = x0.copy()
            x[j] += 1e-6
            optimizer._compute_residuals_vector(x)
    assert jac["trace"] == (len(x0) + 1) * one["trace"]


def _epoch_evaluation_counts(problem) -> dict[str, int]:
    """Counts of one ``LeastSquares`` residual evaluation, which opens an epoch.

    Variable writes and ``update_optics`` run before the epoch opens, so only
    the residual read is counted.
    """
    from optiland.optimization import LeastSquares

    optimizer = LeastSquares(problem)
    x0 = be.to_numpy([var.value for var in problem.variables]).astype(float)
    optimizer._compute_residuals_vector(x0)  # warm caches
    problem.update_optics()
    with _instrument.counting() as counts, _epoch.evaluation_epoch():
        problem.residual_vector()
    return dict(counts)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_epoch_material_evaluations_bounded(name):
    # Numpy backend. One evaluation per (material, wavelength) for the ray
    # bundles, plus one per material for the scalar paraxial primary-wavelength
    # query (scalars and arrays are memoized apart to keep result types).
    optic, problem = PROBLEMS[name]()
    materials = {id(s.material_post) for s in optic.surfaces}
    num_wavelengths = len(optic.wavelengths.get_wavelengths())
    counts = _epoch_evaluation_counts(problem)
    assert counts["material_n"] <= len(materials) * (num_wavelengths + 1)
    assert counts["material_n"] < EXPECTED[name]["material_n"]
    assert counts["trace"] == EXPECTED[name]["trace"]


# Paraxial solves of one residual evaluation inside an epoch: f2 (the focal
# length operand, and the EPD of an F-number aperture) and, where the stop is
# not surface 1, the entrance pupil location (ray aiming), each solved once.
EXPECTED_EPOCH_PARAXIAL = {
    "cooke": 2,
    "double_gauss": 2,
    "freeform": 1,
    "scaling_10": 1,
}

PARAXIAL_QUANTITIES = (
    "f1",
    "f2",
    "EPL",
    "XPL",
    "EPD",
    "XPD",
    "FNO",
    "marginal_ray",
    "chief_ray",
    "entrance_pupil_axial_position",
)


@pytest.mark.parametrize("name", sorted(EXPECTED_EPOCH_PARAXIAL))
def test_epoch_one_paraxial_solve_per_quantity(set_test_backend, name):
    _, problem = PROBLEMS[name]()
    problem.sum_squared()  # warm caches
    problem.update_optics()
    with _instrument.counting() as counts, _epoch.evaluation_epoch():
        problem.residual_vector()
    assert counts["paraxial_trace"] == EXPECTED_EPOCH_PARAXIAL[name]
    assert counts["trace"] == EXPECTED[name]["trace"]


@pytest.mark.parametrize("name", ["cooke", "double_gauss", "freeform"])
def test_epoch_paraxial_quantities_solved_once(set_test_backend, name):
    # One trace each for f1, f2, EPL, XPL and the marginal ray, and three for
    # the chief ray (unit forward, unit reverse, scaled). Every other quantity
    # derives from these. EPL needs no trace when the stop is surface 1.
    optic, _ = PROBLEMS[name]()
    paraxial = optic.paraxial
    with _epoch.evaluation_epoch():
        with _instrument.counting() as first:
            for quantity in PARAXIAL_QUANTITIES:
                getattr(paraxial, quantity)()
        with _instrument.counting() as again:
            for _ in range(3):
                for quantity in PARAXIAL_QUANTITIES:
                    getattr(paraxial, quantity)()
    assert first["paraxial_trace"] == (8 if optic.surfaces.stop_index != 1 else 7)
    assert again["paraxial_trace"] == 0


def test_epoch_aperture_solved_once_per_wavelength(set_test_backend):
    # A float-by-stop aperture traces one paraxial ray per wavelength; the
    # default wavelength is the primary one and shares its entry.
    from optiland.samples.objectives import CookeTriplet

    optic = CookeTriplet()
    optic.set_aperture(aperture_type="float_by_stop_size", value=10.0)
    wavelengths = optic.wavelengths.get_wavelengths()
    with _instrument.counting() as counts, _epoch.evaluation_epoch():
        for _ in range(3):
            for wavelength in wavelengths:
                optic.aperture.compute_epd(optic.paraxial, wavelength)
            optic.aperture.compute_epd(optic.paraxial)
            optic.paraxial.EPD()
    assert len(wavelengths) == 3
    assert counts["paraxial_trace"] == len(wavelengths)


def test_epoch_does_not_change_residuals(set_test_backend):
    import numpy as np

    _, problem = PROBLEMS["double_gauss"]()
    without = be.to_numpy(problem.residual_vector())
    with _epoch.evaluation_epoch():
        within = be.to_numpy(problem.residual_vector())
    np.testing.assert_array_equal(within, without)


def test_counting_is_off_by_default():
    before = _instrument.snapshot()
    _, problem = PROBLEMS["scaling_10"]()
    problem.sum_squared()
    assert _instrument.snapshot() == before


def test_counting_resets_and_is_not_reentrant():
    with _instrument.counting() as outer:
        _instrument.count("trace")
        assert outer["trace"] == 1
        with pytest.raises(RuntimeError), _instrument.counting():
            pass
    with _instrument.counting() as again:
        assert again["trace"] == 0
    _instrument.count("trace")  # disabled again: no effect, no error
