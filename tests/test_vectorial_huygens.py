"""Tests for VectorialHuygensPSF and HuygensPSF factory dispatch.

Tests are parametrized over all available backends via the ``set_test_backend``
fixture defined in conftest.py.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import optiland.backend as be
from optiland.psf.huygens_fresnel import HuygensPSF, ScalarHuygensPSF
from optiland.psf.vectorial_huygens import VectorialHuygensPSF
from optiland.rays import PolarizationState
from tests.utils import assert_allclose

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def generic_optic(set_test_backend):
    """Fresh CookeTriplet optic, created after the backend is configured."""
    from optiland.samples.objectives import CookeTriplet

    return CookeTriplet()


@pytest.fixture
def polarized_optic(set_test_backend):
    """CookeTriplet with horizontal linear polarization."""
    from optiland.samples.objectives import CookeTriplet

    optic = CookeTriplet()
    state = PolarizationState(
        is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0
    )
    optic.updater.set_polarization(state)
    return optic


@pytest.fixture
def unpolarized_optic(set_test_backend):
    """CookeTriplet with unpolarized (two incoherent states) polarization."""
    from optiland.samples.objectives import CookeTriplet

    optic = CookeTriplet()
    state = PolarizationState(is_polarized=False)
    optic.updater.set_polarization(state)
    return optic


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_factory_returns_scalar_when_no_polarization(generic_optic, set_test_backend):
    """HuygensPSF factory should return ScalarHuygensPSF with no polarization."""
    psf = HuygensPSF(
        generic_optic,
        field=(0.0, 0.0),
        wavelength="primary",
        num_rays=32,
        image_size=32,
    )
    assert isinstance(psf, ScalarHuygensPSF)
    assert not isinstance(psf, VectorialHuygensPSF)


def test_factory_returns_vectorial_when_polarized(generic_optic, set_test_backend):
    """HuygensPSF factory should return VectorialHuygensPSF when polarization is set."""
    state = PolarizationState(
        is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0
    )
    generic_optic.updater.set_polarization(state)
    psf = HuygensPSF(
        generic_optic,
        field=(0.0, 0.0),
        wavelength="primary",
        num_rays=32,
        image_size=32,
    )
    assert isinstance(psf, VectorialHuygensPSF)


def test_factory_returns_vectorial_when_unpolarized_state(
    generic_optic, set_test_backend
):
    """An unpolarized PolarizationState should still dispatch to VectorialHuygensPSF."""
    state = PolarizationState(is_polarized=False)
    generic_optic.updater.set_polarization(state)
    psf = HuygensPSF(
        generic_optic,
        field=(0.0, 0.0),
        wavelength="primary",
        num_rays=32,
        image_size=32,
    )
    assert isinstance(psf, VectorialHuygensPSF)


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


def test_vectorial_psf_shape(polarized_optic, set_test_backend):
    """PSF array should have shape (image_size, image_size)."""
    image_size = 32
    psf = VectorialHuygensPSF(
        polarized_optic,
        field=(0.0, 0.0),
        wavelength="primary",
        num_rays=32,
        image_size=image_size,
    )
    assert psf.psf.shape == (image_size, image_size)


def test_vectorial_psf_non_negative(polarized_optic, set_test_backend):
    """PSF values must be non-negative (intensities)."""
    psf = VectorialHuygensPSF(
        polarized_optic,
        field=(0.0, 0.0),
        wavelength="primary",
        num_rays=32,
        image_size=32,
    )
    assert np.all(be.to_numpy(psf.psf) >= 0)


def test_vectorial_psf_normalization_positive(polarized_optic, set_test_backend):
    """Normalization factor must be strictly positive."""
    psf = VectorialHuygensPSF(
        polarized_optic,
        field=(0.0, 0.0),
        wavelength="primary",
        num_rays=32,
        image_size=32,
    )
    norm = psf._get_normalization()
    assert be.to_numpy(norm).item() > 0


# ---------------------------------------------------------------------------
# Strehl ratio
# ---------------------------------------------------------------------------


def test_vectorial_strehl_positive(polarized_optic, set_test_backend):
    """Vectorial Strehl ratio must be a positive finite value in (0, 1]."""
    psf = VectorialHuygensPSF(
        polarized_optic,
        field=(0.0, 0.0),
        wavelength="primary",
        num_rays=64,
        image_size=64,
    )
    strehl = psf.strehl_ratio()
    assert 0.0 < strehl <= 1.0


def test_vectorial_strehl_matches_scalar_on_axis(generic_optic, set_test_backend):
    """Vectorial and scalar Strehl agree within 1% for an uncoated lens on axis."""
    psf_scalar = ScalarHuygensPSF(
        generic_optic,
        field=(0.0, 0.0),
        wavelength="primary",
        num_rays=64,
        image_size=64,
    )

    state = PolarizationState(
        is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0
    )
    generic_optic.updater.set_polarization(state)
    psf_vectorial = VectorialHuygensPSF(
        generic_optic,
        field=(0.0, 0.0),
        wavelength="primary",
        num_rays=64,
        image_size=64,
    )

    assert_allclose(psf_vectorial.strehl_ratio(), psf_scalar.strehl_ratio(), atol=0.01)


# ---------------------------------------------------------------------------
# Off-axis field
# ---------------------------------------------------------------------------


def test_vectorial_off_axis_psf_valid(polarized_optic, set_test_backend):
    """VectorialHuygensPSF computes a valid PSF for an off-axis field."""
    psf = VectorialHuygensPSF(
        polarized_optic,
        field=(0.0, 0.7),
        wavelength="primary",
        num_rays=32,
        image_size=32,
    )
    arr = be.to_numpy(psf.psf)
    assert arr.shape == (32, 32)
    assert np.all(arr >= 0)
    assert arr.max() > 0


# ---------------------------------------------------------------------------
# Unpolarized (two incoherent states)
# ---------------------------------------------------------------------------


def test_vectorial_unpolarized_psf_valid(unpolarized_optic, set_test_backend):
    """An unpolarized source produces a non-negative PSF with a positive peak."""
    psf = VectorialHuygensPSF(
        unpolarized_optic,
        field=(0.0, 0.0),
        wavelength="primary",
        num_rays=32,
        image_size=32,
    )
    arr = be.to_numpy(psf.psf)
    assert np.all(arr >= 0)
    assert arr.max() > 0


# ---------------------------------------------------------------------------
# View method (matplotlib; works on both backends)
# ---------------------------------------------------------------------------


def test_vectorial_view_runs(polarized_optic, set_test_backend):
    """view() should produce a figure without error on all backends."""
    psf = VectorialHuygensPSF(
        polarized_optic,
        field=(0.0, 0.0),
        wavelength="primary",
        num_rays=32,
        image_size=32,
    )
    fig, _ = psf.view()
    plt.close(fig)


# ---------------------------------------------------------------------------
# pixel_pitch pass-through
# ---------------------------------------------------------------------------


def test_vectorial_pixel_pitch_override(polarized_optic, set_test_backend):
    """pixel_pitch kwarg must be stored correctly and not affect PSF shape."""
    psf = VectorialHuygensPSF(
        polarized_optic,
        field=(0.0, 0.0),
        wavelength="primary",
        num_rays=32,
        image_size=32,
        pixel_pitch=0.005,
    )
    assert psf.pixel_pitch == pytest.approx(0.005)
    assert psf.psf.shape == (32, 32)
