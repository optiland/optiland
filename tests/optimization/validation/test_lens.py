"""Native DLS on the fast lens benchmarks versus SciPy ``trf``.

Each problem runs ``LeastSquares`` with ``method_choice="trf"`` and ``"dls"``
from the same fresh design, both at their default budgets and the default
``tol``. DLS must never crash and must reach a final merit within 1% of
``trf``'s.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math

import pytest

import optiland.backend as be
from benchmarks.optimization.problems import PROBLEMS
from optiland.optimization import LeastSquares

pytestmark = pytest.mark.filterwarnings("ignore")

CASES = ["cooke", "double_gauss", "scaling_10"]


def _final_merit(name: str, method: str) -> float:
    _, problem = PROBLEMS[name]()
    LeastSquares(problem).optimize(method_choice=method)
    return float(be.to_numpy(problem.sum_squared()))


@pytest.mark.parametrize("name", CASES)
def test_dls_within_one_percent_of_trf(name):
    trf = _final_merit(name, "trf")
    dls = _final_merit(name, "dls")
    assert math.isfinite(dls)
    assert dls <= 1.01 * trf, (dls, trf)
