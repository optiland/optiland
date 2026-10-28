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

# Steady-state counts of one ``sum_squared`` evaluation (after a warm-up one).
# The batched evaluator traces one super-bundle per (optic, wavelength) with
# ``optic.trace_generic`` inside an evaluation epoch. Material evaluations
# depend on the backend (epoch memo hits on wavelength arrays differ), so they
# are pinned per backend; all other counts are identical on numpy and torch.
EXPECTED = {
    "cooke": {
        "trace": 0,
        "trace_generic": 3,
        "paraxial_trace": 10,
        "material_n": {"numpy": 32, "torch": 71},
    },
    "double_gauss": {
        "trace": 0,
        "trace_generic": 3,
        "paraxial_trace": 19,
        "material_n": {"numpy": 52, "torch": 121},
    },
    "freeform": {
        "trace": 0,
        "trace_generic": 3,
        "paraxial_trace": 1,
        "material_n": {"numpy": 24, "torch": 51},
    },
    "scaling_10": {
        "trace": 0,
        "trace_generic": 1,
        "paraxial_trace": 1,
        "material_n": {"numpy": 24, "torch": 45},
    },
}


def _expected(name: str) -> dict[str, int]:
    """Expected counts of a problem on the active backend."""
    expected = dict(EXPECTED[name])
    expected["material_n"] = expected["material_n"][be.get_backend()]
    return expected


def _evaluation_counts(problem) -> dict[str, int]:
    problem.sum_squared()  # warm caches
    with _instrument.counting() as counts:
        problem.sum_squared()
    return dict(counts)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_merit_evaluation_counts(set_test_backend, name):
    _, problem = PROBLEMS[name]()
    assert _evaluation_counts(problem) == _expected(name)


def test_wide_asphere_trace_count(set_test_backend):
    # Iterative ray aiming differs between backends, so only traces are pinned.
    _, problem = PROBLEMS["wide_asphere"]()
    counts = _evaluation_counts(problem)
    assert counts["trace"] == 0
    assert counts["trace_generic"] == 3


def test_double_gauss_one_trace_per_wavelength(set_test_backend):
    # 3 fields x 3 wavelengths of spots merge into one bundle per wavelength
    # (the EFL operand traces no real rays).
    _, problem = PROBLEMS["double_gauss"]()
    counts = _evaluation_counts(problem)
    assert counts["trace_generic"] == 3
    assert counts["trace"] == 0


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
    assert jac["trace_generic"] == (len(x0) + 1) * one["trace_generic"]


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
    assert counts["trace_generic"] == EXPECTED[name]["trace_generic"]


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
