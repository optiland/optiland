from __future__ import annotations

from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.analysis import SeidelDiagram
from optiland.optic import Optic
from optiland.samples.objectives import CookeTriplet, DoubleGauss, ReverseTelephoto
from optiland.samples.simple import CementedAchromat, Edmund_49_847
from optiland.samples.telescopes import HubbleTelescope
from tests.utils import assert_allclose

matplotlib.use("Agg")

SAMPLES = [
    CookeTriplet,
    DoubleGauss,
    ReverseTelephoto,
    Edmund_49_847,
    CementedAchromat,
    HubbleTelescope,
]


def stop_at_center_of_curvature():
    """A single refracting sphere whose center of curvature is at the stop.

    The chief ray meets the sphere along its normal, so the sphere adds no
    coma, astigmatism or distortion.
    """
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(index=1, thickness=50.0, is_stop=True)
    lens.surfaces.add(index=2, radius=-50.0, thickness=100.0, material="N-BK7")
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=10.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.fields.add(y=5.0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    return lens


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.__name__)
def test_per_surface_sums_to_seidels(set_test_backend, sample):
    lens = sample()
    per_surface = lens.aberrations.seidels_per_surface()
    assert tuple(per_surface.shape) == (5, lens.surfaces.num_surfaces - 2)
    assert_allclose(
        be.sum(per_surface, axis=1), lens.aberrations.seidels(), rtol=0, atol=1e-12
    )


def test_petzval_per_surface_matches_closed_form(set_test_backend):
    """SIV of each surface is -H^2 c (n' - n) / (n n')."""
    lens = CookeTriplet()
    per_surface = lens.aberrations.seidels_per_surface()
    H = lens.paraxial.invariant()
    n = lens.surfaces.n(lens.primary_wavelength)
    c = 1 / lens.surfaces.radii
    expected = be.stack(
        [
            -(H**2) * c[k] * (n[k] - n[k - 1]) / (n[k] * n[k - 1])
            for k in range(1, lens.surfaces.num_surfaces - 1)
        ]
    )
    assert_allclose(per_surface[3], be.ravel(expected), rtol=1e-10, atol=1e-15)


def test_stop_at_center_of_curvature(set_test_backend):
    per_surface = be.to_numpy(
        stop_at_center_of_curvature().aberrations.seidels_per_surface()
    )
    # the stop surface is a flat surface in air and contributes nothing
    assert np.allclose(per_surface[:, 0], 0.0, atol=1e-15)
    sphere = per_surface[:, 1]
    assert abs(sphere[0]) > 1e-6  # SI
    assert abs(sphere[3]) > 1e-6  # SIV
    assert np.allclose(sphere[[1, 2, 4]], 0.0, atol=1e-15)  # SII, SIII, SV


def test_data(set_test_backend):
    lens = CookeTriplet()
    diagram = SeidelDiagram(lens)
    assert_allclose(diagram.data["per_surface"], lens.aberrations.seidels_per_surface())
    assert_allclose(diagram.data["sum"], lens.aberrations.seidels(), rtol=0, atol=1e-12)


def test_view(set_test_backend):
    diagram = SeidelDiagram(CookeTriplet())
    fig, ax = diagram.view(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    per_surface = be.to_numpy(diagram.data["per_surface"])
    totals = be.to_numpy(diagram.data["sum"])
    num_surfaces = per_surface.shape[1]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == [str(i) for i in range(1, num_surfaces + 1)] + ["Sum"]

    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == [
        "SI Spherical",
        "SII Coma",
        "SIII Astigmatism",
        "SIV Petzval",
        "SV Distortion",
    ]
    # one bar container per Seidel term, one bar per surface plus the sum
    assert len(ax.containers) == 5
    for k, bars in enumerate(ax.containers):
        heights = [bar.get_height() for bar in bars]
        assert np.allclose(heights, [*per_surface[k], totals[k]])
    plt.close(fig)


def test_view_on_existing_figure(set_test_backend):
    diagram = SeidelDiagram(CookeTriplet())
    existing = plt.figure()
    fig, ax = diagram.view(fig_to_plot_on=existing)
    assert fig is existing
    assert ax in fig.axes
    plt.close(fig)


@patch("matplotlib.pyplot.show")
def test_view_shows_by_default(mock_show, set_test_backend):
    diagram = SeidelDiagram(CookeTriplet())
    fig, _ = diagram.view()
    mock_show.assert_called_once()
    plt.close(fig)
