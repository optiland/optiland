"""Partial-trace Jacobians equal full Jacobians (opt v2 PR4.1 / PR4.3).

Each benchmark optic is brought into the partial-trace eligible configuration
(object-side aperture, upstream-only aiming). The resumed traces are bitwise
equal to full traces (PR3.1), so the Jacobians must agree far below the
finite-difference noise level, at the start and at a moved design.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from benchmarks.optimization.problems import PROBLEMS
from optiland.optimization.evaluators import FiniteDiffEvaluator

pytestmark = pytest.mark.filterwarnings("ignore")

NAMES = ["cooke", "double_gauss", "wide_asphere", "freeform", "scaling_10"]


def _eligible(name: str):
    optic, problem = PROBLEMS[name]()
    if optic.aperture.ap_type not in ("EPD", "float_by_stop_size", "objectNA"):
        optic.set_aperture("EPD", float(be.to_numpy(optic.paraxial.EPD())))
    optic.ray_tracer.set_aiming("paraxial")
    return optic, problem


@pytest.mark.parametrize("scheme", ["forward", "central"])
@pytest.mark.parametrize("name", NAMES)
def test_partial_jacobian_equals_full(name, scheme):
    _, problem = _eligible(name)
    full = FiniteDiffEvaluator(problem, scheme=scheme, partial_trace=False)
    part = FiniteDiffEvaluator(problem, scheme=scheme)
    x0 = full.read_x()
    for x in (x0, x0 + 1e-3 * np.sign(x0 + 0.5)):
        r = part.residuals(x)
        J_part = part.jacobian(x, r0=r)
        J_full = full.jacobian(x)
        # Resumed and full traces agree to round-off, so the Jacobians agree
        # to the finite-difference noise level (eps * |r| / h), far below the
        # truncation error of the differences.
        noise = (
            1e2
            * np.finfo(float).eps
            * max(1.0, np.abs(r[np.isfinite(r)]).max())
            / part.rel_step
        )
        np.testing.assert_allclose(J_part, J_full, rtol=1e-6, atol=noise)
    starts = part.partial_trace_starts()
    if any(s is not None for s in starts):
        assert part.resumed_traces > 0
