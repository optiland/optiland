from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from optiland.samples.objectives import ReverseTelephoto
from optiland.tolerancing.core import Tolerancing
from optiland.tolerancing.perturbation import DistributionSampler, RangeSampler
from optiland.tolerancing.sensitivity_analysis import SensitivityAnalysis

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def tolerancing():
    optic = ReverseTelephoto()
    tolerancing = Tolerancing(optic)
    tolerancing.add_operand(operand_type="f1", input_data={"optic": optic})
    tolerancing.add_operand(operand_type="f2", input_data={"optic": optic})
    sampler = RangeSampler(start=90, end=110, steps=10)
    tolerancing.add_perturbation("radius", sampler, surface_number=1)
    tolerancing.add_compensator("thickness", surface_number=2)
    return tolerancing


def test_sensitivity_analysis_initialization(tolerancing):
    sa = SensitivityAnalysis(tolerancing)
    assert sa.tolerancing == tolerancing
    assert sa.operand_names == ["0: f1", "1: f2"]
    assert isinstance(sa._results, pd.DataFrame)


def test_sensitivity_analysis_run(tolerancing):
    sa = SensitivityAnalysis(tolerancing)
    sa.run()
    assert not sa._results.empty
    assert len(sa._results) == 10
    assert "perturbation_type" in sa._results.columns
    assert "perturbation_value" in sa._results.columns
    assert "0: f1" in sa._results.columns
    assert "1: f2" in sa._results.columns


def test_sensitivity_analysis_get_results(tolerancing):
    sa = SensitivityAnalysis(tolerancing)
    sa.run()
    results = sa.get_results()
    assert isinstance(results, pd.DataFrame)
    assert not results.empty


def test_sensitivity_analysis_validation_no_operands(tolerancing):
    tolerancing.operands = []
    msg = "No operands found in the tolerancing system."
    with pytest.raises(ValueError, match=msg):
        SensitivityAnalysis(tolerancing)


def test_sensitivity_analysis_validation_no_perturbations(tolerancing):
    tolerancing.perturbations = []
    msg = "No perturbations found in the tolerancing system."
    with pytest.raises(ValueError, match=msg):
        SensitivityAnalysis(tolerancing)


def test_sensitivity_analysis_validation_too_many_operands(tolerancing):
    tolerancing.operands = ["op1", "op2", "op3", "op4", "op5", "op6", "op7"]
    msg = "Sensitivity analysis is limited to 6 operands."
    with pytest.raises(ValueError, match=msg):
        SensitivityAnalysis(tolerancing)


def test_sensitivity_analysis_validation_too_many_perturbations(tolerancing):
    tolerancing.perturbations = [0 for _ in range(11)]
    msg = "Sensitivity analysis is limited to 6 perturbations."
    with pytest.raises(ValueError, match=msg):
        SensitivityAnalysis(tolerancing)


def test_sensitivity_analysis_view(tolerancing):
    sa = SensitivityAnalysis(tolerancing)
    sa.run()
    fig, axes = sa.view()
    assert fig is not None
    assert axes is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_invalid_sampler(tolerancing):
    sampler = DistributionSampler(distribution="uniform", low=0, high=1)
    tolerancing.add_perturbation("radius", sampler, surface_number=1)
    sa = SensitivityAnalysis(tolerancing)
    with pytest.raises(ValueError):
        sa.run()
