from __future__ import annotations

from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.optic import Optic
from optiland.psf import FFTPSF, MMDFTPSF, HuygensPSF
from tests.utils import assert_allclose

matplotlib.use("Agg")

WAVELENGTH = 0.55


def ideal_lens():
    """An aberration-free F/5 lens: a single paraxial surface at the stop."""
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(
        index=1, surface_type="paraxial", f=50.0, thickness=50.0, is_stop=True
    )
    lens.surfaces.add(index=2)
    lens.set_aperture(aperture_type="EPD", value=10.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=WAVELENGTH, is_primary=True)
    return lens


def fine_fftpsf():
    """FFT PSF of the ideal lens, sampled with ~10 pixels per Airy radius."""
    return FFTPSF(ideal_lens(), (0, 0), WAVELENGTH, num_rays=128, grid_size=1024)


PSF_CLASSES = {
    "FFTPSF": lambda lens: FFTPSF(lens, (0, 0), WAVELENGTH, num_rays=64),
    "HuygensPSF": lambda lens: HuygensPSF(
        lens, (0, 0), WAVELENGTH, num_rays=32, image_size=32
    ),
    "MMDFTPSF": lambda lens: MMDFTPSF(
        lens, (0, 0), WAVELENGTH, num_rays=64, image_size=64
    ),
}


def test_first_zero_matches_airy_radius(set_test_backend):
    """The first dark ring of an aberration-free PSF is at 1.22 λN."""
    psf = fine_fftpsf()
    coords, values = (be.to_numpy(a) for a in psf.cross_section(axis="x"))

    peak = int(np.argmax(values))
    assert abs(coords[peak]) < 1e-9  # on-axis peak sits at the origin

    first_zero = peak
    while values[first_zero + 1] < values[first_zero]:
        first_zero += 1
    radius = coords[first_zero] - coords[peak]
    airy_radius = 1.22 * WAVELENGTH * psf._get_working_FNO()
    assert abs(radius - airy_radius) < coords[1] - coords[0]


def test_x_and_y_cross_sections_agree_on_axis(set_test_backend):
    """An on-axis PSF of a rotationally symmetric lens is the same along x and y."""
    psf = fine_fftpsf()
    _, x_values = psf.cross_section(axis="x")
    _, y_values = psf.cross_section(axis="y")
    assert_allclose(x_values, y_values, rtol=1e-6, atol=0)


@pytest.mark.parametrize("make_psf", list(PSF_CLASSES.values()), ids=list(PSF_CLASSES))
def test_each_psf_class(set_test_backend, make_psf):
    """Pixel pitch, peak value and LSF total for every PSF implementation."""
    psf = make_psf(ideal_lens())
    x_extent, y_extent = psf._get_psf_units(psf.psf)
    dx = float(x_extent) / psf.psf.shape[1]
    dy = float(y_extent) / psf.psf.shape[0]

    coords, values = psf.cross_section(axis="x")
    assert tuple(values.shape) == (psf.psf.shape[1],)
    assert_allclose(coords[1] - coords[0], dx)
    assert_allclose(be.max(values), be.max(psf.psf))

    psf_total = be.sum(psf.psf) * dx * dy
    _, lsf_x = psf.line_spread_function(axis="x")
    _, lsf_y = psf.line_spread_function(axis="y")
    assert_allclose(be.sum(lsf_x) * dx, psf_total, rtol=1e-9, atol=0)
    assert_allclose(be.sum(lsf_y) * dy, psf_total, rtol=1e-9, atol=0)


def test_slices_and_sums_follow_rows_and_columns(set_test_backend):
    psf = FFTPSF(ideal_lens(), (0, 0), WAVELENGTH, num_rays=64)
    image = np.arange(64, dtype=float).reshape(8, 8)
    image[2, 5] = 1000.0  # peak at row 2 (y), column 5 (x)
    psf.psf = be.array(image)
    dx, dy = (float(e) / 8 for e in psf._get_psf_units(psf.psf))

    assert_allclose(psf.cross_section(axis="x", through="peak")[1], image[2, :])
    assert_allclose(psf.cross_section(axis="y", through="peak")[1], image[:, 5])
    assert_allclose(psf.cross_section(axis="x", through="center")[1], image[4, :])
    assert_allclose(psf.cross_section(axis="y", through="center")[1], image[:, 4])
    assert_allclose(psf.line_spread_function(axis="x")[1], image.sum(axis=0) * dy)
    assert_allclose(psf.line_spread_function(axis="y")[1], image.sum(axis=1) * dx)


def test_invalid_arguments_raise(set_test_backend):
    psf = FFTPSF(ideal_lens(), (0, 0), WAVELENGTH, num_rays=64)
    with pytest.raises(ValueError, match="Axis must be"):
        psf.cross_section(axis="z")
    with pytest.raises(ValueError, match="Through must be"):
        psf.cross_section(through="edge")
    with pytest.raises(ValueError, match="Axis must be"):
        psf.line_spread_function(axis="both")
    with pytest.raises(ValueError, match="Axis must be"):
        psf.view_cross_section(axis="z", show=False)


def test_uncomputed_psf_raises(set_test_backend):
    psf = FFTPSF(ideal_lens(), (0, 0), WAVELENGTH, num_rays=64)
    psf.psf = None
    with pytest.raises(RuntimeError, match="PSF has not been computed"):
        psf.cross_section()
    with pytest.raises(RuntimeError, match="PSF has not been computed"):
        psf.line_spread_function()
    with pytest.raises(RuntimeError, match="PSF has not been computed"):
        psf.view_cross_section(show=False)


@pytest.mark.parametrize("axis, log", [("both", False), ("x", True)])
def test_view(set_test_backend, axis, log):
    psf = FFTPSF(ideal_lens(), (0, 0), WAVELENGTH, num_rays=64)
    fig, ax = psf.view_cross_section(axis=axis, log=log, show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == (2 if axis == "both" else 1)
    assert ax.get_yscale() == ("log" if log else "linear")
    assert (ax.get_legend() is not None) == (axis == "both")
    assert ax.get_xlabel() == ("Position (µm)" if axis == "both" else "X (µm)")
    plt.close(fig)


def test_view_on_existing_figure(set_test_backend):
    psf = FFTPSF(ideal_lens(), (0, 0), WAVELENGTH, num_rays=64)
    existing = plt.figure()
    fig, ax = psf.view_cross_section(fig_to_plot_on=existing)
    assert fig is existing
    assert ax in fig.axes
    plt.close(fig)


@patch("matplotlib.pyplot.show")
def test_view_shows_by_default(mock_show, set_test_backend):
    psf = FFTPSF(ideal_lens(), (0, 0), WAVELENGTH, num_rays=64)
    fig, _ = psf.view_cross_section()
    mock_show.assert_called_once()
    plt.close(fig)
