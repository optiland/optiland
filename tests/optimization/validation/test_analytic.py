"""Native DLS on MGH test functions and NIST StRD certified problems.

Every start of every problem must reach the known (certified) minimum with
both Jacobian-update modes that may serve as the default. The data must
reproduce the certified merit at the certified parameters, which guards the
transcribed datasets.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.optimization.native import dls

from .problems import PROBLEMS, CallableEvaluator

CASES = [
    (name, k) for name, problem in PROBLEMS.items() for k in range(len(problem.starts))
]


def _reached(problem, merit: float) -> bool:
    if problem.merit == 0.0:
        return merit < 1e-10
    return abs(merit - problem.merit) <= 1e-6 * problem.merit


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_certified_data(name):
    problem = PROBLEMS[name]
    r = problem.residuals(np.array(problem.solution))
    assert float(r @ r) == pytest.approx(problem.merit, rel=1e-8, abs=1e-30)


@pytest.mark.parametrize("mode", ["exact", "auto"])
@pytest.mark.parametrize(("name", "start"), CASES)
def test_reaches_certified_minimum(name, start, mode):
    problem = PROBLEMS[name]
    ev = CallableEvaluator(problem.residuals, np.array(problem.starts[start]))
    res = dls(
        ev,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-10,
        max_nfev=5000,
        jacobian_update=mode,
    )
    merit = 2.0 * res.cost
    assert _reached(problem, merit), (res.status, res.message, merit)
    if problem.certified:
        np.testing.assert_allclose(res.x, problem.solution, rtol=1e-4)
