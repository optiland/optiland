from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np
import optiland.backend as be
from optiland.mtf.fft import ScalarFFTMTF
from optiland.psf import FFTPSF, HuygensPSF
from optiland.psf.fft import calculate_grid_size
from optiland.samples.objectives import CookeTriplet
from optiland.utils import get_working_FNO

from .utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def make_fftpsf(set_test_backend):
    def _factory(
        field=(0, 0),
        wavelength=0.55,
        num_rays=128,
        grid_size=128,
        tweak_optic=None,
    ):
        optic = CookeTriplet()
        if tweak_optic:
            tweak_optic(optic)
        return FFTPSF(optic, field, wavelength, num_rays, grid_size)

    return _factory


def test_initialization(make_fftpsf):
    fftpsf = make_fftpsf(grid_size=1024)
    assert fftpsf.grid_size == 1024
    assert len(fftpsf.pupils) == 1
    assert fftpsf.psf.shape == (1024, 1024)


@pytest.mark.parametrize(
    "num_rays,expected_pupil_sampling",
    [
        (32, 32),
        (64, 45),
        (128, 64),
        (256, 90),
        (512, 128),
        (1024, 181),
        (2048, 256),
        (4096, 362),
        (8192, 512),
    ],
)
def test_calculate_grid_size(num_rays, expected_pupil_sampling):
    assert calculate_grid_size(num_rays) == (expected_pupil_sampling, 2 * num_rays)


@pytest.mark.parametrize(
    "num_rays,expected_pupil_sampling",
    [
        (32, 32),
        (64, 45),
        (128, 64),
        (256, 90),
        (1024, 181),
    ],
)
def test_num_rays_and_grid_size(make_fftpsf, num_rays, expected_pupil_sampling):
    fftpsf = make_fftpsf(num_rays=num_rays, grid_size=None)

    assert fftpsf.num_rays == expected_pupil_sampling
    assert fftpsf.grid_size == 2 * num_rays


@pytest.mark.parametrize(
    "num_rays,grid_size,expectation",
    [
        (32, None, does_not_raise()),
        (64, None, does_not_raise()),
        (12, 16, does_not_raise()),
        (
            16,
            None,
            pytest.raises(
                ValueError,
                match="num_rays must be at least 32 if grid_size is not specified",
            ),
        ),
    ],
)
def test_num_rays_below_32(make_fftpsf, num_rays, grid_size, expectation):
    with expectation:
        make_fftpsf(num_rays=num_rays, grid_size=grid_size)


@pytest.mark.parametrize(
    "num_rays, grid_size",
    [
        (64, 128),
        (65, 256),
        (64, 257),
    ],
)
def test_grid_size(make_fftpsf, num_rays, grid_size):
    fftpsf = make_fftpsf(num_rays=num_rays, grid_size=grid_size)

    assert fftpsf.psf.shape == (grid_size, grid_size)


def test_invalid_grid_size(make_fftpsf):
    with pytest.raises(
        ValueError,
        match=r"Grid size \(\d+\) must be greater than or equal to the number of rays \(\d+\)",
    ):
        make_fftpsf(grid_size=63, num_rays=64)


def test_strehl_ratio(make_fftpsf):
    fftpsf = make_fftpsf(grid_size=256)
    strehl_ratio = fftpsf.strehl_ratio()
    assert 0 <= strehl_ratio <= 1


@pytest.mark.parametrize(
    "projection, log",
    [
        ("2d", False),
        ("3d", False),
        ("2d", True),
        ("3d", True),
    ],
)
def test_view(projection, log, make_fftpsf, set_test_backend):
    # Skip for torch since view isn't implemented there
    fftpsf = make_fftpsf(field=(0, 1))
    fig, ax = fftpsf.view(projection=projection, log=log)
    assert fig is not None
    assert ax is not None
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_find_bounds(make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))

    psf = be.to_numpy(fftpsf.psf)
    min_x, min_y, max_x, max_y = fftpsf._find_bounds(psf, threshold=0.05)
    assert min_x >= 0
    assert min_y >= 0
    assert max_x <= 128
    assert max_y <= 128


def test_view_invalid_projection(make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    with pytest.raises(ValueError):
        fftpsf.view(projection="invalid", log=True)


@pytest.mark.parametrize(
    "projection",
    [
        "2d",
        "3d",
    ],
)
@patch("matplotlib.figure.Figure.text")
def test_view_annotate_sampling(mock_text, projection, make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    fftpsf.view(projection=projection, num_points=32)

    mock_text.assert_called_once()

    plt.close("all")


@pytest.mark.parametrize(
    "projection",
    [
        "2d",
        "3d",
    ],
)
def test_view_oversampling(projection, make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))

    with pytest.warns(UserWarning, match="The PSF view has a high oversampling factor"):
        fftpsf.view(projection=projection, log=False, num_points=128)


def test_get_units_finite_obj(make_fftpsf):
    def tweak(optic):
        optic.surfaces[0].geometry.cs.z = -be.array(1e6)

    fftpsf = make_fftpsf(field=(0, 1), tweak_optic=tweak)
    image = be.zeros((128, 128))
    x, y = fftpsf._get_psf_units(image)
    # Off-axis in y: the tangential (y) pitch is coarser than the sagittal.
    assert_allclose(x, 369.12999618)
    assert_allclose(y, 421.24612411)


def test_psf_log_tick_formatter(make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    assert fftpsf._log_tick_formatter(10) == "$10^{10}$"
    assert fftpsf._log_tick_formatter(1) == "$10^{1}$"
    assert fftpsf._log_tick_formatter(0) == "$10^{0}$"
    assert fftpsf._log_tick_formatter(-1) == "$10^{-1}$"
    assert fftpsf._log_tick_formatter(-10) == "$10^{-10}$"


def test_invalid_working_FNO(make_fftpsf):
    def tweak(optic):
        optic.surfaces[0].geometry.cs.z = -be.array(1e100)

    # Error now surfaces at construction, not at .view().
    with pytest.raises(ValueError):
        make_fftpsf(field=(0, 1), tweak_optic=tweak)


def test_interpolate_zoom_factor_one(make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    assert fftpsf._interpolate_psf(fftpsf.psf) is fftpsf.psf


def test_large_threshold(make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    psf = be.to_numpy(fftpsf.psf)
    min_x, min_y, max_x, max_y = fftpsf._find_bounds(psf, threshold=100)
    assert min_x == 0
    assert min_y == 0
    assert max_x == 128
    assert max_y == 128


def _isotropic_pitch(fftpsf):
    """The former isotropic FFT PSF pitch, wavelength * FNO_w / Q, in µm."""
    fno = be.to_numpy(fftpsf._get_working_FNO())
    q = fftpsf.grid_size / (fftpsf.num_rays - 1)
    return fftpsf.wavelengths[0].value * fno / q


def test_pixel_pitch_on_axis_matches_working_fno(make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 0), num_rays=64, grid_size=256)
    dx, dy, unit = fftpsf._get_pixel_pitch()
    assert unit == "µm"
    expected = _isotropic_pitch(fftpsf)
    assert_allclose(dx, expected, rtol=1e-10)
    assert_allclose(dy, expected, rtol=1e-10)


def test_pixel_pitch_matches_fft_mtf_frequency_steps(make_fftpsf):
    """PSF pitch along an axis is 1 / (grid_size * MTF step) along that axis."""
    fftpsf = make_fftpsf(field=(0, 1), num_rays=64, grid_size=256)
    dx, dy, _ = fftpsf._get_pixel_pitch()

    mtf = ScalarFFTMTF(CookeTriplet(), fields=[(0, 1)], num_rays=64, grid_size=256)
    step_tang = be.to_numpy(mtf.freq_tang[0][1])  # cycles/mm, pupil Y
    step_sag = be.to_numpy(mtf.freq_sag[0][1])  # cycles/mm, pupil X
    assert_allclose(dx, 1e3 / (256 * step_sag), rtol=1e-10)
    assert_allclose(dy, 1e3 / (256 * step_tang), rtol=1e-10)

    # The y field compresses the tangential (y) direction-cosine span, so the
    # pitch grows along y and shrinks along x relative to the mean F/#.
    iso = _isotropic_pitch(fftpsf)
    assert dy > 1.05 * iso
    assert dx < 0.98 * iso


def test_pixel_pitch_follows_field_direction(make_fftpsf):
    """A field along x changes the x pitch the way a field along y changes y."""
    dx_y, dy_y, _ = make_fftpsf(field=(0, 1), num_rays=64)._get_pixel_pitch()
    dx_x, dy_x, _ = make_fftpsf(field=(1, 0), num_rays=64)._get_pixel_pitch()
    assert dy_y > dx_y
    assert dx_x > dy_x
    assert_allclose(dx_x, dy_y, rtol=1e-8)
    assert_allclose(dy_x, dx_y, rtol=1e-8)


def _fft_psf_on_huygens_grid(fftpsf, huygens, dx, dy):
    """Sample an FFT PSF, with pitches dx, dy in µm, on a HuygensPSF grid.

    The Huygens image points are `linspace(min, max, image_size)` in mm on
    the (flat, untilted) image plane. The FFT PSF is centred, at pixel
    ``n // 2``, on the chief-ray intercept.
    """
    from scipy.ndimage import map_coordinates

    xmin, xmax, ymin, ymax = (
        float(be.to_numpy(v)) for v in huygens._get_image_extent()
    )
    size = huygens.image_size
    x, y = np.meshgrid(np.linspace(xmin, xmax, size), np.linspace(ymin, ymax, size))
    chief = huygens.optic.trace_generic(*huygens.fields[0].coord, 0.0, 0.0, 0.55)
    xc = float(be.to_numpy(chief.x).ravel()[0])
    yc = float(be.to_numpy(chief.y).ravel()[0])
    fft = np.asarray(be.to_numpy(fftpsf.psf), dtype=float)
    center = fft.shape[0] // 2
    rows = (y - yc) * 1e3 / dy + center
    cols = (x - xc) * 1e3 / dx + center
    return map_coordinates(fft, [rows, cols], order=3, mode="constant")


def _relative_residual(reference, model):
    """Least-squares residual of `model` scaled onto `reference`."""
    gain = (model.ravel() @ reference.ravel()) / (model.ravel() @ model.ravel())
    return np.sum((reference - gain * model) ** 2) / np.sum(reference**2)


def test_off_axis_fft_psf_matches_huygens_psf(set_test_backend):
    """The FFT PSF at its per-axis pitch matches the Huygens PSF off axis.

    HuygensPSF evaluates the diffraction integral at physical image-plane
    points, so it is an independent reference for the FFT PSF pitch.
    Observed residuals (CookeTriplet, field (0, 1)): ~4e-4 with the per-axis
    pitch, ~0.1 with the former isotropic pitch.
    """
    optic = CookeTriplet()
    fftpsf = FFTPSF(optic, (0, 1), 0.55, num_rays=64, grid_size=512)
    huygens = HuygensPSF(optic, (0, 1), 0.55, num_rays=64, image_size=33)
    reference = np.asarray(be.to_numpy(huygens.psf), dtype=float)

    dx, dy, _ = fftpsf._get_pixel_pitch()
    per_axis = _fft_psf_on_huygens_grid(fftpsf, huygens, dx, dy)
    assert _relative_residual(reference, per_axis) < 2e-3

    fno = float(be.to_numpy(get_working_FNO(optic, (0, 1), 0.55)))
    iso = 0.55 * fno * (fftpsf.num_rays - 1) / fftpsf.grid_size
    isotropic = _fft_psf_on_huygens_grid(fftpsf, huygens, iso, iso)
    assert _relative_residual(reference, isotropic) > 5e-2
