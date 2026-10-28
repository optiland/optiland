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
from optiland import _instrument

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
