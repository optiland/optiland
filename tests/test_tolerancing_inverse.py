"""Tests for InverseSensitivity."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.optimization.scaling.identity import IdentityScaler
from optiland.samples.objectives import CookeTriplet
from optiland.tolerancing import (
    InverseSensitivity,
    Perturbation,
    RangeSampler,
    SensitivityAnalysis,
    Tolerance,
    Tolerancing,
)
from optiland.tolerancing.registry import ToleranceType
from optiland.tolerancing.types import ThicknessTolerance
from tests.utils import assert_allclose

RTOL = 1e-4  # default search tolerance, relative to the start limit


@pytest.fixture
def track_system(set_test_backend):
    """Two finite spacings, 10 + 15 mm; total track is linear in each."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(index=1, thickness=10.0, is_stop=True)
    optic.surfaces.add(index=2, thickness=15.0)
    optic.surfaces.add(index=3)
    optic.set_aperture("EPD", 1.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def track_tolerancing(optic, n_operands=1):
    tolerancing = Tolerancing(optic, tol=1e-8)
    for _ in range(n_operands):
        tolerancing.add_operand("total_track", input_data={"optic": optic})
    return tolerancing


class QuadraticThickness(ToleranceType):
    """Thickness change of ``delta + delta**2``: an asymmetric response."""

    label = "Quadratic S2"

    def __init__(self, optic):
        super().__init__(optic)
        self._inner = ThicknessTolerance(optic, surface_number=2)

    def apply(self, delta):
        self._inner.apply(delta + delta**2)

    def reset(self):
        self._inner.reset()


def assert_nominal(optic):
    assert_allclose(optic.surfaces.get_thickness(1), [10.0], rtol=0, atol=1e-12)
    assert_allclose(optic.surfaces.get_thickness(2), [15.0], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("limit", "allowed"), [(0.1, 0.25), (1.0, 0.3)], ids=["expand", "shrink"]
)
def test_analytic_linear_recovery(track_system, limit, allowed):
    tolerancing = track_tolerancing(track_system, n_operands=2)
    tol = tolerancing.add_tolerance("thickness", surface_number=2, limit=limit)
    analysis = InverseSensitivity(tolerancing, {0: 2 * allowed, 1: allowed})
    analysis.run()
    row = analysis.get_results().iloc[0]

    atol = RTOL * limit
    assert_allclose(row["inverse_minus"], -allowed, rtol=0, atol=atol)
    assert_allclose(row["inverse_plus"], allowed, rtol=0, atol=atol)
    assert row["tolerance"] == tol.label
    assert (row["minus"], row["plus"]) == (-limit, limit)
    assert row["limiting_minus"] == row["limiting_plus"] == analysis.operand_names[1]
    assert not row["capped_minus"] and not row["capped_plus"]
    assert_allclose(analysis.nominal, [25.0, 25.0])
    assert_nominal(track_system)
    assert tol.value == 0.0


@pytest.mark.parametrize("side", ["minus", "plus"])
def test_recovers_sensitivity_analysis_limit(set_test_backend, side):
    optic = CookeTriplet()
    tolerancing = Tolerancing(optic)
    tolerancing.add_operand("f2", input_data={"optic": optic})
    tol = tolerancing.add_tolerance("radius", surface_number=1, minus=-0.4, plus=0.6)

    nominal = tolerancing.evaluate()[0]
    sensitivity = SensitivityAnalysis(tolerancing)
    sensitivity.run()
    df = sensitivity.get_results()
    index = 0 if side == "minus" else 1
    change = abs(df[sensitivity.operand_names[0]].iloc[index] - nominal)

    analysis = InverseSensitivity(tolerancing, {0: change})
    analysis.run()
    row = analysis.get_results().iloc[0]
    limit = getattr(tol, side)
    assert_allclose(row[f"inverse_{side}"], limit, rtol=0, atol=RTOL * abs(limit))
    assert not row[f"capped_{side}"]


def test_asymmetric_response(track_system):
    tolerancing = track_tolerancing(track_system)
    tolerancing.perturbations.append(
        Tolerance(QuadraticThickness(track_system), -0.05, 0.05)
    )
    analysis = InverseSensitivity(tolerancing, {0: 0.1})
    analysis.run()
    row = analysis.get_results().iloc[0]

    # |x + x^2| = 0.1 on each side
    expected_plus = (-1 + 1.4**0.5) / 2
    expected_minus = (-1 + 0.6**0.5) / 2
    assert_allclose(row["inverse_plus"], expected_plus, rtol=0, atol=RTOL * 0.05)
    assert_allclose(row["inverse_minus"], expected_minus, rtol=0, atol=RTOL * 0.05)
    assert abs(row["inverse_minus"]) > row["inverse_plus"]
    assert_nominal(track_system)


def test_cap_reported_when_operand_insensitive(track_system):
    tolerancing = track_tolerancing(track_system)
    # the adjust spacing absorbs the change, so total track never moves
    tolerancing.add_tolerance("thickness", surface_number=2, adjust=1, limit=0.1)
    analysis = InverseSensitivity(tolerancing, {0: 0.01}, max_factor=10)
    analysis.run()
    row = analysis.get_results().iloc[0]
    assert_allclose(row["inverse_minus"], -1.0)
    assert_allclose(row["inverse_plus"], 1.0)
    assert row["capped_minus"] and row["capped_plus"]
    assert row["limiting_minus"] is None and row["limiting_plus"] is None
    assert_nominal(track_system)


def test_compensators_apply_in_every_trial(track_system):
    tolerancing = Tolerancing(track_system, tol=1e-8)
    tolerancing.add_operand(
        "total_track", input_data={"optic": track_system}, target=25.0
    )
    tolerancing.add_tolerance("thickness", surface_number=2, limit=0.1)
    tolerancing.add_compensator("thickness", surface_number=1, scaler=IdentityScaler())
    analysis = InverseSensitivity(tolerancing, {0: 0.01}, max_factor=4)
    analysis.run()
    row = analysis.get_results().iloc[0]
    assert row["capped_minus"] and row["capped_plus"]
    assert_nominal(track_system)


def test_legacy_perturbations_ignored(track_system):
    tolerancing = track_tolerancing(track_system)
    legacy = Perturbation(
        track_system, "thickness", RangeSampler(11.0, 12.0, 2), surface_number=1
    )
    tolerancing.perturbations.append(legacy)
    tolerancing.add_tolerance("thickness", surface_number=2, limit=0.1)
    analysis = InverseSensitivity(tolerancing, {0: 0.2})
    analysis.run()
    df = analysis.get_results()
    assert len(df) == 1
    assert_allclose(df["inverse_plus"].iloc[0], 0.2, rtol=0, atol=RTOL * 0.1)
    assert_nominal(track_system)


def test_no_tolerances_raises(track_system):
    tolerancing = track_tolerancing(track_system)
    tolerancing.add_perturbation(
        "thickness", RangeSampler(11.0, 12.0, 2), surface_number=1
    )
    with pytest.raises(ValueError, match="No tolerances"):
        InverseSensitivity(tolerancing, {0: 0.1})


@pytest.mark.parametrize(
    ("max_change", "kwargs", "match"),
    [
        ({}, {}, "at least one"),
        ({1: 0.1}, {}, "out of range"),
        ({0: 0.0}, {}, "positive"),
        ({0: 0.1}, {"max_factor": 1.0}, "max_factor"),
    ],
)
def test_invalid_arguments(track_system, max_change, kwargs, match):
    tolerancing = track_tolerancing(track_system)
    tolerancing.add_tolerance("thickness", surface_number=2, limit=0.1)
    with pytest.raises(ValueError, match=match):
        InverseSensitivity(tolerancing, max_change, **kwargs)


def test_zero_limits_raise(track_system):
    tolerancing = track_tolerancing(track_system)
    tolerancing.add_tolerance("thickness", surface_number=2, limit=0.0)
    with pytest.raises(ValueError, match="zero limits"):
        InverseSensitivity(tolerancing, {0: 0.1})


def test_one_sided_tolerance_uses_other_limit(track_system):
    tolerancing = track_tolerancing(track_system)
    tolerancing.add_tolerance("thickness", surface_number=2, minus=0.0, plus=0.1)
    analysis = InverseSensitivity(tolerancing, {0: 0.3})
    analysis.run()
    row = analysis.get_results().iloc[0]
    assert_allclose(row["inverse_minus"], -0.3, rtol=0, atol=RTOL * 0.1)


def test_failure_restores_nominal_and_keeps_previous_results(
    track_system, monkeypatch
):
    tolerancing = track_tolerancing(track_system)
    tol = tolerancing.add_tolerance("thickness", surface_number=2, limit=0.1)
    analysis = InverseSensitivity(tolerancing, {0: 0.25})
    analysis.run()
    previous = analysis.get_results().copy()
    previous_nominal = list(analysis.nominal)

    calls = []
    original = tolerancing.evaluate

    def failing_evaluate():
        calls.append(None)
        if len(calls) == 3:
            assert tol.value != 0.0  # fail mid-trial, with a deviation applied
            raise RuntimeError("boom")
        return original()

    monkeypatch.setattr(tolerancing, "evaluate", failing_evaluate)
    with pytest.raises(RuntimeError, match="boom"):
        analysis.run()

    assert analysis.get_results().equals(previous)
    assert analysis.nominal == previous_nominal
    assert tol.value == 0.0
    assert_nominal(track_system)


def test_apply_limits(track_system):
    tolerancing = track_tolerancing(track_system)
    tol = tolerancing.add_tolerance("thickness", surface_number=2, limit=0.1)
    analysis = InverseSensitivity(tolerancing, {0: 0.25})
    with pytest.raises(RuntimeError, match="Run"):
        analysis.apply_limits()
    analysis.run()
    analysis.apply_limits()
    assert_allclose(tol.limits, (-0.25, 0.25), rtol=0, atol=RTOL * 0.1)


def test_view(track_system):
    tolerancing = track_tolerancing(track_system)
    tolerancing.add_tolerance("thickness", surface_number=2, limit=0.1)
    tolerancing.add_tolerance("thickness", surface_number=1, limit=0.2)
    analysis = InverseSensitivity(tolerancing, {0: 0.25})
    analysis.run()
    fig, ax = analysis.view()
    assert len(ax.patches) == 8
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "0: Thickness S2",
        "1: Thickness S1",
    ]
    plt.close(fig)
