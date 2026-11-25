from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.psf import MMDFTPSF, ScalarFFTPSF
from optiland.samples.objectives import CookeTriplet

from .utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def make_mmdftpsf(set_test_backend):
    def _factory(
        field=(0, 0),
        wavelength=0.55,
        num_rays=128,
        image_size=128,
        pixel_pitch=None,
        tweak_optic=None,
    ):
        optic = CookeTriplet()
        if tweak_optic:
            tweak_optic(optic)
        return MMDFTPSF(
            optic,
            field,
            wavelength,
            num_rays=num_rays,
            image_size=image_size,
            pixel_pitch=pixel_pitch,
        )

    return _factory


@pytest.fixture
def make_mmdftpsf_and_fftpsf(set_test_backend):
    def _factory(
        field=(0, 0),
        wavelength=0.55,
        num_rays=128,
        image_size=128,
        pixel_pitch=None,
        tweak_optic=None,
    ):
        optic = CookeTriplet()
        if tweak_optic:
            tweak_optic(optic)
        fftpsf = ScalarFFTPSF(
            optic, field, wavelength, num_rays=num_rays, grid_size=image_size
        )
        dx = (
            fftpsf.wavelengths[0].value
            * fftpsf._get_working_FNO()
            * (fftpsf.num_rays - 1)
            / fftpsf.grid_size
        )
        mmdftpsf = MMDFTPSF(
            optic,
            field,
            wavelength,
            num_rays=fftpsf.num_rays,
            image_size=fftpsf.grid_size,
            pixel_pitch=dx,
        )
        return fftpsf, mmdftpsf

    return _factory


def test_initialization(make_mmdftpsf):
    mmdftpsf = make_mmdftpsf(image_size=1024)
    assert mmdftpsf.image_size == 1024
    assert mmdftpsf.psf.shape == (1024, 1024)


@pytest.mark.parametrize(
    "num_rays,expected_pupil_sampling, expected_pixel_pitch",
    [
        (32, 32, 1.32622273171),
        (64, 45, 0.94119032573),
        (128, 64, 0.67380671047),
        (256, 90, 0.47594283517),
        (1024, 181, 0.24064525374),
    ],
)
def test_calcs_from_num_rays(
    make_mmdftpsf, num_rays, expected_pupil_sampling, expected_pixel_pitch
):
    mmdftpsf = make_mmdftpsf(num_rays=num_rays, image_size=None)

    assert mmdftpsf.num_rays == expected_pupil_sampling
    assert mmdftpsf.image_size == 2 * num_rays
    assert_allclose(mmdftpsf.pixel_pitch, expected_pixel_pitch)


@pytest.mark.parametrize(
    "pixel_pitch, expected_image_size",
    [(0.25, 1390), (0.50, 695), (0.75, 463), (1.00, 347), (1.50, 231), (2.00, 173)],
)
def test_calcs_from_pixel_pitch(make_mmdftpsf, pixel_pitch, expected_image_size):
    mmdftpsf = make_mmdftpsf(pixel_pitch=pixel_pitch, image_size=None)

    assert mmdftpsf.image_size == expected_image_size


@pytest.mark.parametrize(
    "image_size, expected_pixel_pitch",
    [
        (128, 2.71661753109),
        (256, 1.35830876554),
        (512, 0.67915438277),
        (1024, 0.33957719139),
        (2048, 0.16978859569),
        (4096, 0.08489429785),
    ],
)
def test_calcs_from_image_size(make_mmdftpsf, image_size, expected_pixel_pitch):
    mmdftpsf = make_mmdftpsf(image_size=image_size, pixel_pitch=None)

    assert_allclose(expected_pixel_pitch, mmdftpsf.pixel_pitch)


@pytest.mark.parametrize(
    "num_rays,image_size,expectation",
    [
        (32, None, does_not_raise()),
        (64, None, does_not_raise()),
        (12, 16, does_not_raise()),
        (
            16,
            None,
            pytest.raises(
                ValueError,
                match="num_rays must be at least 32 if image_size and pixel_pitch are "
                "not specified.",
            ),
        ),
    ],
)
def test_num_rays_below_32(make_mmdftpsf, num_rays, image_size, expectation):
    with expectation:
        make_mmdftpsf(num_rays=num_rays, image_size=image_size)


@pytest.mark.parametrize(
    "num_rays, image_size",
    [
        (64, 128),
        (65, 256),
        (64, 257),
    ],
)
def test_image_size(make_mmdftpsf, num_rays, image_size):
    mmdftpsf = make_mmdftpsf(num_rays=num_rays, image_size=image_size)

    assert mmdftpsf.psf.shape == (image_size, image_size)


def test_invalid_image_size(make_mmdftpsf):
    with pytest.raises(
        ValueError,
        match=r"Supplied image_size of \d+ not less than or equal to calculated "
        r"pad size of \d+",
    ):
        make_mmdftpsf(image_size=400, pixel_pitch=1)


def test_strehl_ratio(make_mmdftpsf):
    mmdftpsf = make_mmdftpsf(image_size=256)
    strehl_ratio = mmdftpsf.strehl_ratio()
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
def test_view(projection, log, make_mmdftpsf, set_test_backend):
    # Skip for torch since view isn't implemented there
    mmdftpsf = make_mmdftpsf(field=(0, 1))
    fig, ax = mmdftpsf.view(projection=projection, log=log)
    assert fig is not None
    assert ax is not None
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_find_bounds(make_mmdftpsf):
    mmdftpsf = make_mmdftpsf(field=(0, 1))

    psf = be.to_numpy(mmdftpsf.psf)
    min_x, min_y, max_x, max_y = mmdftpsf._find_bounds(psf, threshold=0.05)
    assert min_x >= 0
    assert min_y >= 0
    assert max_x <= 128
    assert max_y <= 128


def test_view_invalid_projection(make_mmdftpsf):
    mmdftpsf = make_mmdftpsf(field=(0, 1))
    with pytest.raises(ValueError):
        mmdftpsf.view(projection="invalid", log=True)


@pytest.mark.parametrize(
    "projection",
    [
        "2d",
        "3d",
    ],
)
@patch("matplotlib.figure.Figure.text")
def test_view_annotate_sampling(mock_text, projection, make_mmdftpsf):
    mmdftpsf = make_mmdftpsf(field=(0, 1))
    mmdftpsf.view(projection=projection, num_points=32)

    mock_text.assert_called_once()

    plt.close("all")


@pytest.mark.parametrize(
    "projection",
    [
        "2d",
        "3d",
    ],
)
def test_view_oversampling(projection, make_mmdftpsf):
    mmdftpsf = make_mmdftpsf(field=(0, 1))

    with pytest.warns(UserWarning, match="The PSF view has a high oversampling factor"):
        mmdftpsf.view(projection=projection, log=False, num_points=128)


def test_get_units_finite_obj(make_mmdftpsf):
    def tweak(optic):
        optic.surfaces[0].geometry.cs.z = -be.array(1e6)

    mmdftpsf = make_mmdftpsf(field=(0, 1), tweak_optic=tweak)
    image = be.zeros((128, 128))
    x, y = mmdftpsf._get_psf_units(image)
    # The default pitch is the largest alias-free pitch for image_size: off
    # axis it is set by the larger per-axis frequency step (here pupil X), so
    # it equals the FFT PSF x-pitch of the same case, not lambda * FNO_w / Q.
    assert_allclose(x, 369.12999618)
    assert_allclose(y, 369.12999618)


def test_psf_log_tick_formatter(make_mmdftpsf):
    mmdftpsf = make_mmdftpsf(field=(0, 1))
    assert mmdftpsf._log_tick_formatter(10) == "$10^{10}$"
    assert mmdftpsf._log_tick_formatter(1) == "$10^{1}$"
    assert mmdftpsf._log_tick_formatter(0) == "$10^{0}$"
    assert mmdftpsf._log_tick_formatter(-1) == "$10^{-1}$"
    assert mmdftpsf._log_tick_formatter(-10) == "$10^{-10}$"


def test_invalid_working_FNO(make_mmdftpsf):
    def tweak(optic):
        optic.surfaces[0].geometry.cs.z = -be.array(1e100)

    with pytest.raises(ValueError):
        mmdftpsf = make_mmdftpsf(field=(0, 1), tweak_optic=tweak)
        fig, ax = mmdftpsf.view()
        plt.close(fig)


def test_interpolate_zoom_factor_one(make_mmdftpsf):
    mmdftpsf = make_mmdftpsf(field=(0, 1))
    assert mmdftpsf._interpolate_psf(mmdftpsf.psf) is mmdftpsf.psf


def test_large_threshold(make_mmdftpsf):
    mmdftpsf = make_mmdftpsf(field=(0, 1))
    psf = be.to_numpy(mmdftpsf.psf)
    min_x, min_y, max_x, max_y = mmdftpsf._find_bounds(psf, threshold=100)
    assert min_x == 0
    assert min_y == 0
    assert max_x == 128
    assert max_y == 128


@pytest.mark.parametrize(
    "num_rays, image_size",
    [
        (32, 64),
        (45, 128),
        (64, 256),
        (90, 512),
        (181, 2048),
        (128, 128),
        (256, 256),
        (512, 512),
    ],
)
def test_fft_agreement(make_mmdftpsf_and_fftpsf, num_rays, image_size):
    fftpsf, mmdftpsf = make_mmdftpsf_and_fftpsf(
        num_rays=num_rays, image_size=image_size
    )
    assert_allclose(fftpsf.psf, mmdftpsf.psf)
    assert_allclose(fftpsf.strehl_ratio(), mmdftpsf.strehl_ratio())


def test_off_axis_matches_fft_psf_per_axis(set_test_backend):
    """Off axis, MMDFT samples at `pixel_pitch` along both axes.

    With the pitch set to the FFT PSF pitch along one axis, the MMDFT row (or
    column) through the chief ray reproduces the FFT PSF samples on that axis.
    A single isotropic pitch fails one of the two.
    """
    optic = CookeTriplet()
    fft = ScalarFFTPSF(optic, (0, 1), 0.55, num_rays=64, grid_size=512)
    dx, dy, _ = fft._get_pixel_pitch()
    assert dy > 1.05 * dx  # anamorphic field

    size, centre = 33, 512 // 2
    window = slice(centre - size // 2, centre + size // 2 + 1)
    for pitch, axis in ((dx, "x"), (dy, "y")):
        mmdft = MMDFTPSF(
            optic, (0, 1), 0.55, num_rays=64, image_size=size, pixel_pitch=pitch
        )
        if axis == "x":
            expected, actual = fft.psf[centre, window], mmdft.psf[size // 2, :]
        else:
            expected, actual = fft.psf[window, centre], mmdft.psf[:, size // 2]
        assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)


def _rotated_field_optic(angle_deg, azimuth_deg):
    """CookeTriplet with fields at (0, angle) and the same angle rotated.

    Angle fields compose tan(x) and tan(y), so the rotated field is
    (atan(t sin(az)), atan(t cos(az))) with t = tan(angle).
    """
    optic = CookeTriplet()
    t = np.tan(np.radians(angle_deg))
    az = np.radians(azimuth_deg)
    fx = float(np.degrees(np.arctan(t * np.sin(az))))
    fy = float(np.degrees(np.arctan(t * np.cos(az))))
    optic.fields.add(x=fx, y=fy)
    max_field = float(be.to_numpy(optic.fields.max_field))
    return optic, (0.0, angle_deg / max_field), (fx / max_field, fy / max_field)


def _rotation_error(optic, meridional, skew, azimuth, **kwargs):
    """Relative error between a skew-field PSF and the rotated meridional PSF."""
    from scipy.ndimage import map_coordinates

    psf_m = be.to_numpy(MMDFTPSF(optic, meridional, 0.55, **kwargs).psf)
    psf_s = be.to_numpy(MMDFTPSF(optic, skew, 0.55, **kwargs).psf)
    size = psf_m.shape[0]
    c, s = np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))
    # psf_s(r) = psf_m(R^T r), with R taking +y to the skew field direction.
    offsets = np.arange(size) - size // 2
    x, y = np.meshgrid(offsets, offsets)
    rotated = map_coordinates(
        psf_m,
        [s * x + c * y + size // 2, c * x - s * y + size // 2],
        order=3,
        mode="constant",
    )
    inner = np.hypot(x, y) < size // 2 - 2  # stay clear of the grid corners
    return np.sum((psf_s - rotated)[inner] ** 2) / np.sum(psf_s[inner] ** 2)


def test_skew_field_is_rotated_meridional_psf(set_test_backend, monkeypatch):
    """Rotational symmetry: a skew-field PSF is the meridional one, rotated.

    For a rotationally symmetric lens, the PSF of a field rotated by an angle
    about the axis is the PSF of the unrotated field, rotated by the same
    angle about the chief ray. Together with the per-axis scale checked in
    `test_off_axis_matches_fft_psf_per_axis`, this fixes the skew-field PSF.
    A skew field couples the pupil axes: dropping the off-diagonal terms of
    the pupil-to-direction map (separable per-axis kernels) breaks the
    symmetry. The remaining error is pupil-grid discretization and decreases
    with num_rays (about 3e-4, 6e-5 and 2e-5 at 64, 128 and 256 rays).
    """
    import optiland.psf.mmdft as mmdft_module

    azimuth = 36.87
    optic, meridional, skew = _rotated_field_optic(18.0, azimuth)

    # The two chief rays are rotations of each other.
    rays = optic.trace_generic(
        Hx=be.array([meridional[0], skew[0]]),
        Hy=be.array([meridional[1], skew[1]]),
        Px=0.0,
        Py=0.0,
        wavelength=0.55,
    )
    L, M = be.to_numpy(rays.L), be.to_numpy(rays.M)
    c, s = np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))
    np.testing.assert_allclose([L[1], M[1]], [s * M[0], c * M[0]], atol=1e-9)

    kwargs = {"num_rays": 128, "image_size": 61, "pixel_pitch": 0.3}
    assert _rotation_error(optic, meridional, skew, azimuth, **kwargs) < 1e-4

    # Control: without the off-diagonal coupling the symmetry is broken.
    full_matrix = mmdft_module.pupil_direction_matrix

    def diagonal_only(*args):
        A = full_matrix(*args)
        return A * be.array([[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(mmdft_module, "pupil_direction_matrix", diagonal_only)
    assert _rotation_error(optic, meridional, skew, azimuth, **kwargs) > 1e-3
