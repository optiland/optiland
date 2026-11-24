"""Tests for tolerancing result summaries (RSS estimate, tornado plot, Monte
Carlo seeds, yield and percentiles) and the quick-focus compensator."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import optiland.backend as be
from optiland.samples.objectives import CookeTriplet
from optiland.tolerancing import (
    RangeSampler,
    SensitivityAnalysis,
    Tolerancing,
)
from optiland.tolerancing.compensator import QuickFocusCompensator
from optiland.tolerancing.monte_carlo import MonteCarlo
from tests.utils import assert_allclose

SPOT = "0: rms spot size"


def _f(value) -> float:
    return float(np.ravel(be.to_numpy(value))[0])


def _image_gap(optic) -> int:
    """Return the surface number of the gap in front of the image."""
    return optic.surfaces.num_surfaces - 2


def _positions(optic) -> list[float]:
    return [_f(surface.geometry.cs.z) for surface in optic.surfaces]


@pytest.fixture
def tolerancing(set_test_backend):
    """Tolerancing on a Cooke triplet, built after the backend is set."""
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


@pytest.fixture
def two_tolerances(tolerancing):
    tolerancing.add_tolerance("radius", surface_number=1, limit=0.5)
    tolerancing.add_tolerance("thickness", surface_number=2, minus=-0.05, plus=0.1)
    return tolerancing


# --- SensitivityAnalysis: rss_estimate ---


def test_rss_estimate_matches_hand_calculation(two_tolerances):
    nominal = two_tolerances.evaluate()[0]
    sa = SensitivityAnalysis(two_tolerances)
    sa.run()

    df = sa.get_results()
    delta = (df[SPOT] - nominal).abs()
    worst_radius = delta.iloc[0:2].max()
    worst_thickness = delta.iloc[2:4].max()
    expected = np.hypot(worst_radius, worst_thickness)

    rss = sa.rss_estimate()
    assert list(rss.columns) == ["nominal", "rss_change", "estimate"]
    assert list(rss.index) == [SPOT]
    assert_allclose(rss.loc[SPOT, "nominal"], nominal)
    assert_allclose(rss.loc[SPOT, "rss_change"], expected)
    assert_allclose(rss.loc[SPOT, "estimate"], nominal + expected)


def test_rss_estimate_includes_legacy_perturbations(tolerancing):
    tolerancing.add_tolerance("radius", surface_number=1, limit=0.5)
    tolerancing.add_perturbation(
        "thickness", RangeSampler(start=4.0, end=4.2, steps=3), surface_number=2
    )
    sa = SensitivityAnalysis(tolerancing)
    sa.run()

    df = sa.get_results()
    nominal = sa.rss_estimate().loc[SPOT, "nominal"]
    delta = (df[SPOT] - nominal).abs()
    expected = np.hypot(delta.iloc[0:2].max(), delta.iloc[2:5].max())
    assert_allclose(sa.rss_estimate().loc[SPOT, "rss_change"], expected)


def test_rss_estimate_propagates_nan(two_tolerances):
    sa = SensitivityAnalysis(two_tolerances)
    sa.run()
    sa._results.loc[1, SPOT] = np.nan
    assert np.isnan(sa.rss_estimate().loc[SPOT, "rss_change"])


def test_rss_estimate_before_run_raises(two_tolerances):
    sa = SensitivityAnalysis(two_tolerances)
    with pytest.raises(RuntimeError, match="run"):
        sa.rss_estimate()


def test_run_keeps_result_columns(two_tolerances):
    sa = SensitivityAnalysis(two_tolerances)
    sa.run()
    assert list(sa.get_results().columns) == [
        "perturbation_type",
        "perturbation_value",
        SPOT,
    ]


# --- SensitivityAnalysis: view_tornado ---


def test_view_tornado_sorts_largest_on_top(two_tolerances):
    sa = SensitivityAnalysis(two_tolerances)
    sa.run()
    fig, ax = sa.view_tornado()
    try:
        worst = sa._worst_changes()[SPOT]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # largest on top = last tick
        assert labels == list(worst.sort_values().index)
        assert len(ax.patches) == 4
    finally:
        plt.close(fig)


def test_view_tornado_top(two_tolerances):
    sa = SensitivityAnalysis(two_tolerances)
    sa.run()
    fig, ax = sa.view_tornado(top=1)
    try:
        worst = sa._worst_changes()[SPOT]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == [worst.idxmax()]
    finally:
        plt.close(fig)


def test_view_tornado_before_run_raises(two_tolerances):
    sa = SensitivityAnalysis(two_tolerances)
    with pytest.raises(RuntimeError, match="run"):
        sa.view_tornado()


# --- MonteCarlo: seed ---


def test_monte_carlo_seed_is_reproducible(two_tolerances):
    mc = MonteCarlo(two_tolerances)
    mc.run(5, seed=7)
    first = mc.get_results().copy()
    mc.run(5, seed=7)
    pd.testing.assert_frame_equal(first, mc.get_results())

    mc.run(5, seed=8)
    assert not first.equals(mc.get_results())


def test_monte_carlo_seed_restores_nominal(two_tolerances):
    optic = two_tolerances.optic
    before = _positions(optic)
    MonteCarlo(two_tolerances).run(3, seed=1)
    assert_allclose(_positions(optic), before)


# --- MonteCarlo: yield_fraction and percentiles ---


@pytest.fixture
def synthetic_mc(two_tolerances):
    """A MonteCarlo with known results."""
    mc = MonteCarlo(two_tolerances)
    columns = {p.label: np.zeros(5) for p in two_tolerances.perturbations}
    columns[SPOT] = [1.0, 2.0, 3.0, 4.0, np.nan]
    columns["C0: comp"] = [10.0, 20.0, 30.0, 40.0, 50.0]
    mc._results = pd.DataFrame(columns)
    return mc


def test_yield_fraction_known_answer(synthetic_mc):
    assert synthetic_mc.yield_fraction({0: ("<=", 2.0)}) == pytest.approx(0.4)
    assert synthetic_mc.yield_fraction({0: ("<", 2.0)}) == pytest.approx(0.2)
    # NaN trial fails ">=" as well
    assert synthetic_mc.yield_fraction({0: (">=", 0.0)}) == pytest.approx(0.8)
    both = {0: (">", 1.0), "C0: comp": ("<=", 30.0)}
    assert synthetic_mc.yield_fraction(both) == pytest.approx(0.4)


def test_yield_fraction_invalid_input(synthetic_mc):
    with pytest.raises(ValueError, match="at least one"):
        synthetic_mc.yield_fraction({})
    with pytest.raises(ValueError, match="Unknown operator"):
        synthetic_mc.yield_fraction({0: ("==", 1.0)})
    with pytest.raises(KeyError):
        synthetic_mc.yield_fraction({"missing": ("<", 1.0)})


def test_percentiles_skip_nan_and_tolerance_columns(synthetic_mc):
    p = synthetic_mc.percentiles([0, 50, 100])
    assert list(p.columns) == [SPOT, "C0: comp"]
    assert list(p.index) == [0.0, 50.0, 100.0]
    assert_allclose(p[SPOT].to_numpy(), [1.0, 2.5, 4.0])
    assert_allclose(p["C0: comp"].to_numpy(), [10.0, 30.0, 50.0])


def test_percentiles_invalid_input(synthetic_mc):
    with pytest.raises(ValueError, match="between 0 and 100"):
        synthetic_mc.percentiles([101])


def test_summaries_before_run_raise(two_tolerances):
    mc = MonteCarlo(two_tolerances)
    with pytest.raises(RuntimeError, match="run"):
        mc.yield_fraction({0: ("<", 1.0)})
    with pytest.raises(RuntimeError, match="run"):
        mc.percentiles()


def test_percentiles_on_real_run(two_tolerances):
    mc = MonteCarlo(two_tolerances)
    mc.run(6, seed=3)
    p = mc.percentiles([50])
    assert list(p.columns) == [SPOT]
    assert_allclose(p.loc[50.0, SPOT], np.median(mc.get_results()[SPOT]))


# --- quick_focus compensator ---


def test_quick_focus_refocuses_image_gap_error(tolerancing):
    optic = tolerancing.optic
    nominal = tolerancing.evaluate()[0]
    tolerancing.add_tolerance("thickness", surface_number=_image_gap(optic), limit=0.5)
    sa = SensitivityAnalysis(tolerancing)
    sa.run()
    defocused = sa.get_results()[SPOT].to_numpy()
    assert np.all(defocused > 2 * nominal)

    tolerancing.add_compensator("quick_focus")
    sa = SensitivityAnalysis(tolerancing)
    sa.run()
    df = sa.get_results()
    refocused = df[SPOT].to_numpy()
    compensated_nominal = sa.rss_estimate().loc[SPOT, "nominal"]
    assert_allclose(refocused, [compensated_nominal] * 2, atol=1e-9)
    assert QuickFocusCompensator.label in df.columns


def test_quick_focus_reset_restores_positions(tolerancing):
    optic = tolerancing.optic
    before = _positions(optic)
    tolerancing.add_tolerance("thickness", surface_number=_image_gap(optic), limit=0.5)
    tolerancing.add_tolerance("thickness", surface_number=2, limit=0.1)
    tolerancing.add_compensator("quick_focus")

    MonteCarlo(tolerancing).run(3, seed=0)
    assert_allclose(_positions(optic), before)

    SensitivityAnalysis(tolerancing).run()
    assert_allclose(_positions(optic), before)


def test_quick_focus_records_image_distance(tolerancing):
    optic = tolerancing.optic
    tolerancing.add_compensator("quick_focus", wavelength=0.55, num_rays=4)
    result = tolerancing.apply_compensators()
    gap = _f(optic.surfaces[-1].geometry.cs.z) - _f(optic.surfaces[-2].geometry.cs.z)
    assert result[QuickFocusCompensator.label] == pytest.approx(gap)
    tolerancing.reset()


def test_quick_focus_reset_without_apply_is_noop(tolerancing):
    optic = tolerancing.optic
    before = _positions(optic)
    tolerancing.add_compensator("quick_focus")
    tolerancing.reset()
    assert_allclose(_positions(optic), before)
