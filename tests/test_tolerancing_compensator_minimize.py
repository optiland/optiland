"""Tests for CompensatorOptimizer.run through optimization.minimize."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.optimization import LeastSquares, OptimizerGeneric
from optiland.samples.objectives import CookeTriplet
from optiland.tolerancing import Tolerancing
from optiland.tolerancing.compensator import CompensatorOptimizer
from tests.utils import assert_allclose


def _compensated(method: str) -> Tolerancing:
    """Cooke triplet with a radius error and an image-gap compensator."""
    lens = CookeTriplet()
    tol = Tolerancing(lens, method=method)
    tol.add_operand(
        "rms_spot_size",
        {
            "optic": lens,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 0.0,
            "num_rays": 5,
            "wavelength": 0.55,
        },
    )
    tol.add_tolerance("radius", surface_number=1, limit=0.5)
    tol.add_compensator("thickness", surface_number=6)
    tol.perturbations[0].apply_value(0.5)
    tol.compensator.operands = tol.operands
    return tol


def _state(tol: Tolerancing) -> tuple[float, float]:
    variable = tol.compensator.variables[0]
    return float(be.to_numpy(variable.value)), float(tol.compensator.sum_squared())


@pytest.mark.parametrize(
    "method, legacy, kwargs",
    [
        ("generic", OptimizerGeneric, {"disp": False}),
        ("least_squares", LeastSquares, {}),
    ],
)
def test_matches_legacy_optimizer(set_test_backend, method, legacy, kwargs):
    new = _compensated(method)
    new.compensator.run()
    x_new, merit_new = _state(new)

    old = _compensated(method)
    legacy(old.compensator).optimize(tol=old.compensator.tol, **kwargs)
    x_old, merit_old = _state(old)

    if method == "generic":
        # same SciPy call: L-BFGS-B, maxiter 1000, same tol and bounds
        assert x_new == x_old
        assert merit_new == merit_old
    else:
        # minimize keeps the best evaluated point, never a worse one
        assert merit_new <= merit_old
        assert_allclose(x_new, x_old, rtol=1e-6)


def test_run_prints_nothing(set_test_backend, capsys):
    _compensated("generic").compensator.run()
    assert capsys.readouterr().out == ""


def test_untraceable_start_is_left_unchanged(set_test_backend, monkeypatch):
    tol = _compensated("generic")
    before = _state(tol)[0]
    monkeypatch.setattr(
        tol.compensator, "fun_array", lambda: be.array([np.nan]), raising=False
    )
    monkeypatch.setattr(
        tol.compensator,
        "residual_vector",
        lambda: be.array([np.nan]),
        raising=False,
    )
    assert tol.compensator.run() is None
    assert _state(tol)[0] == before


def test_invalid_method_raises():
    compensator = CompensatorOptimizer(method="bogus")
    with pytest.raises(ValueError, match="Invalid optimizer method"):
        compensator.run()
