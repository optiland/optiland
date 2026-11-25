"""Tests for MultiConfiguration.analyze() and MultiConfiguration.summary()."""

from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import optiland.backend as be
from optiland.analysis import Distortion, SpotDiagram
from optiland.materials import Material
from optiland.multiconfig import MultiConfigAnalysis, MultiConfiguration
from optiland.samples.objectives import CookeTriplet
from tests.utils import assert_allclose

matplotlib.use("Agg")

FIRST_ORDER_ROWS = ["EFL", "F/#", "Image height", "Total track", "EPD"]


def make_zoom() -> MultiConfiguration:
    """Three-configuration 'zoom' of the Cooke triplet.

    Config 1 and 2 change the air gap after surface 2 and the image
    distance; config 2 also has a smaller EPD and a different glass on
    surface 3.
    """
    mc = MultiConfiguration(CookeTriplet())
    mc.add_configuration()
    mc.add_configuration()
    mc.set_thickness(2, 3.0, configurations=[1])
    mc.set_thickness(2, 1.5, configurations=[2])
    mc.set_thickness(6, 44.0, configurations=[1])
    mc.set_thickness(6, 45.0, configurations=[2])
    mc.set_material(3, Material("N-SF5"), configurations=[2])
    mc.configurations[2].set_aperture("EPD", 8.0)
    for optic in mc.configurations:
        optic.updater.update()
    return mc


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_summary_first_order_matches_paraxial(set_test_backend):
    mc = make_zoom()
    frame = mc.summary()

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["Config 0", "Config 1", "Config 2"]
    assert list(frame.index[:5]) == FIRST_ORDER_ROWS
    for i, optic in enumerate(mc.configurations):
        column = frame[f"Config {i}"]
        y_chief, _ = optic.paraxial.chief_ray()
        assert_allclose(column["EFL"], optic.paraxial.f2())
        assert_allclose(column["F/#"], optic.paraxial.FNO())
        assert_allclose(column["EPD"], optic.paraxial.EPD())
        assert_allclose(column["Total track"], optic.total_track)
        assert_allclose(column["Image height"], be.abs(be.ravel(y_chief)[-1]))


def test_summary_lists_only_varying_properties(set_test_backend):
    frame = make_zoom().summary()

    varying = list(frame.index[len(FIRST_ORDER_ROWS) :])
    assert sorted(varying) == sorted(
        [
            "Aperture value",
            "Surface 2 thickness",
            "Surface 3 material",
            "Surface 6 thickness",
        ]
    )
    assert_allclose(
        frame.loc["Surface 2 thickness"].to_numpy(dtype=float), [6.00755, 3.0, 1.5]
    )
    assert_allclose(
        frame.loc["Aperture value"].to_numpy(dtype=float), [10.0, 10.0, 8.0]
    )
    materials = list(frame.loc["Surface 3 material"])
    assert materials[0] == materials[1] == "F2"
    assert materials[2] == "N-SF5"


def test_summary_single_configuration_has_only_first_order_rows():
    frame = MultiConfiguration(CookeTriplet()).summary()
    assert list(frame.index) == FIRST_ORDER_ROWS
    assert list(frame.columns) == ["Config 0"]


def test_summary_failing_quantity_is_nan(monkeypatch):
    mc = make_zoom()

    def _raise():
        raise ValueError("no paraxial EFL for this system")

    monkeypatch.setattr(mc.configurations[1].paraxial, "f2", _raise)
    frame = mc.summary()

    assert math.isnan(frame.loc["EFL", "Config 1"])
    assert not math.isnan(frame.loc["EFL", "Config 0"])
    assert not math.isnan(frame.loc["EPD", "Config 1"])


def test_analyze_returns_one_result_per_configuration(set_test_backend):
    mc = make_zoom()
    result = mc.analyze(SpotDiagram, num_rings=3)

    assert isinstance(result, MultiConfigAnalysis)
    assert len(result) == 3
    assert result.configurations == [0, 1, 2]
    for optic, spot in zip(mc.configurations, result, strict=True):
        assert isinstance(spot, SpotDiagram)
        assert spot.optic is optic
        reference = SpotDiagram(optic, num_rings=3)
        assert_allclose(
            be.stack([r for f in spot.rms_spot_radius() for r in f]),
            be.stack([r for f in reference.rms_spot_radius() for r in f]),
        )


def test_analyze_subset_of_configurations():
    mc = make_zoom()
    result = mc.analyze(Distortion, "primary", configurations=[2, 0])

    assert result.configurations == [2, 0]
    assert result.results[0].optic is mc.configurations[2]
    assert result.results[1].optic is mc.configurations[0]


def test_view_one_titled_figure_per_configuration(monkeypatch):
    mc = make_zoom()
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))

    outputs = mc.analyze(SpotDiagram, num_rings=3).view(show=False)

    assert shown == []
    figures = [out[0] for out in outputs]
    assert len({id(fig) for fig in figures}) == 3
    titles = [fig.get_suptitle() for fig in figures]
    assert [t.split(":")[0] for t in titles] == [
        "Configuration 0",
        "Configuration 1",
        "Configuration 2",
    ]


def test_view_show_calls_plt_show_once(monkeypatch):
    mc = make_zoom()
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))

    mc.analyze(Distortion, configurations=[0, 1]).view(show=True)

    assert shown == [True]


def test_multiconfig_analysis_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one result per configuration"):
        MultiConfigAnalysis([object()], [0, 1])
