from __future__ import annotations

import pytest

from optiland.psf.fft import FFTPSF, ScalarFFTPSF
from optiland.psf.vectorial_fft import VectorialFFTPSF
from optiland.rays import PolarizationState
from tests.utils import assert_allclose


@pytest.fixture
def generic_optic(set_test_backend):
    from optiland.samples.objectives import CookeTriplet

    return CookeTriplet()


def test_fftpsf_factory_dispatch(generic_optic, set_test_backend):
    # Default is unpolarized ("ignore")
    psf_scalar = FFTPSF(
        generic_optic, field=(0.0, 0.0), wavelength="primary", num_rays=32
    )
    assert isinstance(psf_scalar, ScalarFFTPSF)

    # Enable polarization
    state = PolarizationState(
        is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0
    )
    generic_optic.updater.set_polarization(state)
    psf_vectorial = FFTPSF(
        generic_optic, field=(0.0, 0.0), wavelength="primary", num_rays=32
    )
    assert isinstance(psf_vectorial, VectorialFFTPSF)


def test_vectorial_fft_strehl_matches_scalar(generic_optic, set_test_backend):
    # For a simple paraxial/uncoated lens, the scalar and vectorial Strehl
    # should be roughly identical since PRT matrices are mostly identity and
    # the unpolarized sum roughly matches the uniform pupil.

    psf_scalar = FFTPSF(
        generic_optic, field=(0.0, 0.0), wavelength="primary", num_rays=64
    )
    strehl_scalar = psf_scalar.strehl_ratio()

    # Enable polarization (Horizontal)
    state = PolarizationState(
        is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0
    )
    generic_optic.updater.set_polarization(state)
    psf_vectorial_h = FFTPSF(
        generic_optic, field=(0.0, 0.0), wavelength="primary", num_rays=64
    )
    strehl_vectorial_h = psf_vectorial_h.strehl_ratio()

    # Enable polarization (unpolarized source -> 2 incoherent states)
    state_unpol = PolarizationState(is_polarized=False)
    generic_optic.updater.set_polarization(state_unpol)
    psf_vectorial_unpol = FFTPSF(
        generic_optic, field=(0.0, 0.0), wavelength="primary", num_rays=64
    )
    strehl_vectorial_unpol = psf_vectorial_unpol.strehl_ratio()

    # Strehl ratio is heavily dependent on definition and normalization, but
    # should identically match the scalar counterpart for 0 aberration cases and
    # simple standard cases.
    assert_allclose(strehl_vectorial_h, strehl_scalar, atol=1e-2)
    assert_allclose(strehl_vectorial_unpol, strehl_scalar, atol=1e-2)
