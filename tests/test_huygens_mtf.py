"""Tests for ScalarHuygensMTF, VectorialHuygensMTF, and the HuygensMTF factory.

Tests are parametrized over all available backends via the ``set_test_backend``
fixture defined in conftest.py.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import optiland.backend as be
from optiland.mtf import HuygensMTF, ScalarHuygensMTF, VectorialHuygensMTF
from optiland.rays import PolarizationState
from tests.utils import assert_allclose

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def real_optic(set_test_backend):
    """DoubleGauss optic, created after the backend is configured."""
    from optiland.samples.objectives import DoubleGauss

    return DoubleGauss()


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
    """CookeTriplet with an unpolarized polarization state."""
    from optiland.samples.objectives import CookeTriplet

    optic = CookeTriplet()
    optic.updater.set_polarization(PolarizationState(is_polarized=False))
    return optic


# ---------------------------------------------------------------------------
# ScalarHuygensMTF
# ---------------------------------------------------------------------------


class TestScalarHuygensMTF:
    def test_init_sets_attributes_cutoff(self, real_optic):
        mtf = ScalarHuygensMTF(real_optic, max_freq="cutoff", image_size=16)
        assert mtf.num_rays == 128
        assert mtf.image_size == 16
        # max_freq is calibrated to the on-axis working F/#, per OpticStudio convention.
        on_axis_fno = mtf._get_fno()
        assert_allclose(mtf.max_freq, 1 / (0.5876e-3 * on_axis_fno), rtol=1e-4)
        # freq is a list of per-field arrays; check the first field.
        assert len(be.to_numpy(mtf.freq[0])) == 16 // 2

    def test_init_non_primary_wavelength(self, real_optic):
        mtf = ScalarHuygensMTF(
            real_optic, max_freq="cutoff", image_size=16, wavelength=0.55
        )
        on_axis_fno = mtf._get_fno()
        assert_allclose(mtf.max_freq, 1 / (0.55e-3 * on_axis_fno), rtol=1e-4)

    def test_init_with_numeric_max_freq(self, real_optic):
        mtf = ScalarHuygensMTF(real_optic, max_freq=200.0, image_size=8)
        assert mtf.max_freq == 200.0

    def test_get_fno(self, real_optic):
        mtf = ScalarHuygensMTF(real_optic, fields=[(0, 0)], image_size=8)
        # FNO is now a per-field list; check the on-axis (first) field.
        assert_allclose(mtf.FNO[0], 4.992598838013766, rtol=1e-4)

    def test_calculate_psf_stores_data(self, real_optic):
        mtf = ScalarHuygensMTF(real_optic, image_size=8)
        mtf.resolved_fields = [(0, 0)]
        mtf.resolved_wavelength = 0.55
        mtf._calculate_psf()
        assert len(mtf.psf_data) == 1
        assert mtf.psf_data[0].shape == (8, 8)
        assert len(mtf.psf_instances) == 1

    def test_generate_mtf_data_normalization(self, real_optic):
        mtf = ScalarHuygensMTF(real_optic, image_size=8)
        mtf.psf_data = [be.ones((8, 8))]
        results = mtf._generate_mtf_data()
        tangential, sagittal = results[0]
        assert np.all(be.to_numpy(tangential) <= 1.0)
        assert np.all(be.to_numpy(sagittal) <= 1.0)
        assert_allclose(tangential[0], 1.0)
        assert_allclose(sagittal[0], 1.0)

    def test_generate_mtf_data_divide_by_zero(self, real_optic):
        mtf = ScalarHuygensMTF(real_optic, image_size=8)
        mtf.psf_data = [be.zeros((8, 8))]
        results = mtf._generate_mtf_data()
        tangential, sagittal = results[0]
        assert np.allclose(be.to_numpy(tangential), 0)
        assert np.allclose(be.to_numpy(sagittal), 0)

    def test_plot_field_mtf_runs(self, real_optic):
        mtf = ScalarHuygensMTF(real_optic, image_size=8)
        mtf.resolved_fields = [(1.0, 2.0)]
        # freq is a list of per-field arrays
        mtf.freq = [be.arange(4)]
        mtf_field_data = [be.linspace(0, 1, 4), be.linspace(1, 0, 4)]
        fig, ax = plt.subplots()
        mtf._plot_field_mtf(ax, 0, mtf_field_data, color="blue")
        plt.close(fig)

    def test_get_mtf_units_valid_and_invalid(self, real_optic):
        mtf = ScalarHuygensMTF(real_optic, image_size=8)
        mtf.resolved_fields = [(0, 0)]
        mtf.resolved_wavelength = 0.55
        mtf._calculate_psf()
        # _get_mtf_units was split into tang and sag versions
        df_tang = mtf._get_mtf_units_tang(0)
        df_sag = mtf._get_mtf_units_sag(0)
        assert df_tang > 0
        assert df_sag > 0

        mtf.psf_instances[0].pixel_pitch = 0
        with pytest.raises(ValueError, match="Pixel pitch"):
            mtf._get_mtf_units_tang(0)

    def test_integration(self, real_optic):
        """Full PSF + MTF calculation (low resolution for speed)."""
        mtf = ScalarHuygensMTF(real_optic, fields=[(0, 0)], image_size=16)
        mtf._calculate_psf()
        mtf_data = mtf._generate_mtf_data()
        assert len(mtf_data) == 1
        tang_np = be.to_numpy(mtf_data[0][0])
        sag_np = be.to_numpy(mtf_data[0][1])
        assert np.all((tang_np >= 0) & (tang_np <= 1))
        assert np.all((sag_np >= 0) & (sag_np <= 1))


# ---------------------------------------------------------------------------
# HuygensMTF factory dispatch
# ---------------------------------------------------------------------------


class TestHuygensMTFFactory:
    def test_no_polarization_returns_scalar(self, real_optic):
        """HuygensMTF returns ScalarHuygensMTF when polarization is not set."""
        mtf = HuygensMTF(real_optic, image_size=8)
        assert isinstance(mtf, ScalarHuygensMTF)
        assert not isinstance(mtf, VectorialHuygensMTF)

    def test_polarization_returns_vectorial(self, polarized_optic):
        """HuygensMTF returns VectorialHuygensMTF when polarization is set."""
        mtf = HuygensMTF(polarized_optic, image_size=8)
        assert isinstance(mtf, VectorialHuygensMTF)

    def test_unpolarized_state_returns_vectorial(self, unpolarized_optic):
        """An unpolarized PolarizationState still dispatches to VectorialHuygensMTF."""
        mtf = HuygensMTF(unpolarized_optic, image_size=8)
        assert isinstance(mtf, VectorialHuygensMTF)


# ---------------------------------------------------------------------------
# VectorialHuygensMTF
# ---------------------------------------------------------------------------


class TestVectorialHuygensMTF:
    def test_psf_data_shape(self, polarized_optic):
        mtf = VectorialHuygensMTF(polarized_optic, fields=[(0, 0)], image_size=16)
        assert len(mtf.psf_data) == 1
        assert mtf.psf_data[0].shape == (16, 16)
        assert len(mtf.psf_instances) == 1

    def test_mtf_values_in_unit_interval(self, polarized_optic):
        mtf = VectorialHuygensMTF(polarized_optic, fields=[(0, 0)], image_size=16)
        tangential, sagittal = mtf.mtf[0]
        assert np.all(be.to_numpy(tangential) >= 0)
        assert np.all(be.to_numpy(tangential) <= 1)
        assert np.all(be.to_numpy(sagittal) >= 0)
        assert np.all(be.to_numpy(sagittal) <= 1)

    def test_dc_value_is_one(self, polarized_optic):
        mtf = VectorialHuygensMTF(polarized_optic, fields=[(0, 0)], image_size=16)
        tangential, sagittal = mtf.mtf[0]
        assert_allclose(tangential[0], 1.0, atol=1e-6)
        assert_allclose(sagittal[0], 1.0, atol=1e-6)

    def test_frequency_array_length(self, polarized_optic):
        image_size = 16
        mtf = VectorialHuygensMTF(
            polarized_optic, fields=[(0, 0)], image_size=image_size
        )
        # freq is now a list of per-field arrays; check the first field.
        assert len(be.to_numpy(mtf.freq[0])) == image_size // 2

    def test_unpolarized_mtf_valid(self, unpolarized_optic):
        """An unpolarized source produces a valid MTF."""
        mtf = VectorialHuygensMTF(unpolarized_optic, fields=[(0, 0)], image_size=16)
        tangential, sagittal = mtf.mtf[0]
        assert np.all(be.to_numpy(tangential) >= 0)
        assert np.all(be.to_numpy(tangential) <= 1)
        assert np.all(be.to_numpy(sagittal) >= 0)
        assert np.all(be.to_numpy(sagittal) <= 1)

    def test_scalar_and_vectorial_agree_uncoated(self, set_test_backend):
        """Scalar and vectorial MTF agree within 5% for an uncoated lens on axis."""
        from optiland.samples.objectives import CookeTriplet

        optic_scalar = CookeTriplet()
        optic_vectorial = CookeTriplet()
        state = PolarizationState(
            is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0
        )
        optic_vectorial.updater.set_polarization(state)

        mtf_scalar = ScalarHuygensMTF(optic_scalar, fields=[(0, 0)], image_size=16)
        mtf_vectorial = VectorialHuygensMTF(
            optic_vectorial, fields=[(0, 0)], image_size=16
        )

        tang_s, sag_s = mtf_scalar.mtf[0]
        tang_v, sag_v = mtf_vectorial.mtf[0]
        assert_allclose(tang_s, tang_v, atol=0.05)
        assert_allclose(sag_s, sag_v, atol=0.05)


# ---------------------------------------------------------------------------
# Frequency axis
# ---------------------------------------------------------------------------


def _perfect_parabola():
    """F/10 paraboloidal mirror (f = 100 mm): perfect on-axis focus."""
    from optiland.optic import Optic

    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        radius=-200,
        conic=-1.0,
        thickness=-100,
        material="mirror",
        is_stop=True,
    )
    optic.surfaces.add(index=2)
    optic.set_aperture("EPD", 10)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _first_crossing(freq, mtf, level):
    """Frequency at which the MTF first falls below ``level`` (linear interp)."""
    i = int(np.argmax(mtf < level))
    return np.interp(level, [mtf[i], mtf[i - 1]], [freq[i], freq[i - 1]])


@pytest.fixture
def numpy_backend():
    """NumPy backend for the 257 x 257 Huygens grids below.

    With the torch test settings (autograd enabled) the Huygens summation
    keeps its graph for every image point, tens of GB at this grid size.
    """
    original_backend = be.get_backend()
    be.set_backend("numpy")
    yield
    be.set_backend(original_backend)


def test_on_axis_mtf_matches_diffraction_limit(numpy_backend):
    """A perfect mirror follows (2/pi)(acos(v) - v sqrt(1 - v^2)), v = nu/nu_c.

    The Huygens MTF is about 1 % high at every frequency because the PSF
    window (image_size * pixel_pitch) misses about 1.1 % of the Airy energy,
    which lowers the DC term. Hence the tolerance of 0.02.
    """
    mtf = ScalarHuygensMTF(
        _perfect_parabola(), fields=[(0, 0)], num_rays=64, image_size=257
    )
    cutoff = 1 / (0.55e-3 * be.to_numpy(mtf._get_fno()))
    v = np.linspace(0.05, 0.95, 19)
    expected = 2 / np.pi * (np.arccos(v) - v * np.sqrt(1 - v**2))
    for axis, freq in ((0, mtf.freq_tang[0]), (1, mtf.freq_sag[0])):
        actual = np.interp(v * cutoff, be.to_numpy(freq), be.to_numpy(mtf.mtf[0][axis]))
        assert_allclose(actual, expected, rtol=0, atol=0.02)


def test_off_axis_tangential_mtf_matches_fft_mtf(numpy_backend):
    """CookeTriplet full field: tangential frequency scale agrees with FFTMTF.

    The frequency where the tangential MTF falls to 0.5 differs from FFTMTF by
    1.1 % (8.2 % with the former FNO_on / FNO_off factor), and the largest
    difference up to 60 cycles/mm is 0.031 (0.078 with the factor).
    """
    from optiland.mtf import FFTMTF
    from optiland.samples.objectives import CookeTriplet

    optic = CookeTriplet()
    fft = FFTMTF(
        optic, fields=[(0, 1.0)], wavelength=0.55, num_rays=128, grid_size=1024
    )
    huygens = ScalarHuygensMTF(
        optic, fields=[(0, 1.0)], wavelength=0.55, num_rays=48, image_size=257
    )
    f_fft, m_fft = (be.to_numpy(a) for a in (fft.freq_tang[0], fft.mtf[0][0]))
    f_huy, m_huy = (be.to_numpy(a) for a in (huygens.freq_tang[0], huygens.mtf[0][0]))

    mtf50_fft = _first_crossing(f_fft, m_fft, 0.5)
    mtf50_huy = _first_crossing(f_huy, m_huy, 0.5)
    assert abs(mtf50_huy / mtf50_fft - 1) < 0.03

    nu = np.linspace(0, 60, 61)
    difference = np.interp(nu, f_huy, m_huy) - np.interp(nu, f_fft, m_fft)
    assert np.max(np.abs(difference)) < 0.045


def test_tangential_and_sagittal_steps_are_equal(set_test_backend):
    """Both axes use the DFT step 1 / (image_size * pixel_pitch) off axis."""
    from optiland.samples.objectives import CookeTriplet

    mtf = ScalarHuygensMTF(CookeTriplet(), fields=[(0, 1.0)], image_size=16)
    expected = 1 / (16 * mtf.psf_instances[0].pixel_pitch)
    assert_allclose(mtf._get_mtf_units_tang(0), expected, rtol=1e-12, atol=0)
    assert_allclose(mtf._get_mtf_units_sag(0), expected, rtol=1e-12, atol=0)
