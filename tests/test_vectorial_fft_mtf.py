"""Tests for VectorialFFTMTF and the FFTMTF factory dispatch."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # ensure non-interactive backend for testing

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.mtf import FFTMTF, ScalarFFTMTF, VectorialFFTMTF
from optiland.psf.vectorial_fft import VectorialFFTPSF
from optiland.rays import PolarizationState
from optiland.samples.objectives import CookeTriplet

from .utils import assert_allclose

# Run every test with every available backend
pytestmark = pytest.mark.parametrize(
    "set_test_backend",
    be.list_available_backends(),
    indirect=True,
    ids=[f"backend={b}" for b in be.list_available_backends()],
)

_NUM_RAYS = 32
_GRID_SIZE = 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def optic():
    """Unpolarized CookeTriplet."""
    return CookeTriplet()


@pytest.fixture
def polarized_optic():
    """CookeTriplet with horizontal (H) linear polarization enabled."""
    lens = CookeTriplet()
    state = PolarizationState(
        is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0
    )
    lens.updater.set_polarization(state)
    return lens


@pytest.fixture
def unpolarized_state_optic():
    """CookeTriplet with unpolarized (incoherent sum) polarization enabled."""
    lens = CookeTriplet()
    lens.updater.set_polarization(PolarizationState(is_polarized=False))
    return lens


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


class TestFFTMTFFactoryDispatch:
    def test_no_polarization_returns_scalar(self, set_test_backend, optic):
        mtf = FFTMTF(optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        assert isinstance(mtf, ScalarFFTMTF)
        assert not isinstance(mtf, VectorialFFTMTF)

    def test_polarized_returns_vectorial(self, set_test_backend, polarized_optic):
        mtf = FFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        assert isinstance(mtf, VectorialFFTMTF)

    def test_unpolarized_state_returns_vectorial(
        self, set_test_backend, unpolarized_state_optic
    ):
        mtf = FFTMTF(unpolarized_state_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        assert isinstance(mtf, VectorialFFTMTF)

    def test_vectorial_is_subclass_of_scalar(self, set_test_backend):
        assert issubclass(VectorialFFTMTF, ScalarFFTMTF)


# ---------------------------------------------------------------------------
# VectorialFFTMTF initialization
# ---------------------------------------------------------------------------


class TestVectorialFFTMTFInit:
    def test_explicit_grid_size(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        assert mtf.num_rays == _NUM_RAYS
        assert mtf.grid_size == _GRID_SIZE

    def test_auto_grid_size(self, set_test_backend, polarized_optic):
        num_rays = 64
        mtf = VectorialFFTMTF(polarized_optic, num_rays=num_rays, grid_size=None)
        # calculate_grid_size(64) -> (45, 128)
        assert mtf.num_rays == 45
        assert mtf.grid_size == 128

    def test_cutoff_max_freq(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        expected = 1.0 / (mtf.resolved_wavelength * 1e-3 * mtf.FNO[0])
        assert_allclose(mtf.max_freq, expected)

    def test_custom_max_freq(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(
            polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE, max_freq=100.0
        )
        assert be.to_numpy(mtf.max_freq) == pytest.approx(100.0)

    def test_fno_attribute_is_set(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        assert mtf.FNO is not None
        assert np.all(be.to_numpy(mtf.FNO) > 0)

    def test_freq_array_length(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        freq_np = be.to_numpy(mtf.freq[0])
        assert len(freq_np) == _GRID_SIZE // 2

    def test_freq_array_starts_at_zero(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        assert_allclose(mtf.freq[0][0], 0.0)

    def test_psf_list_length_matches_fields(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(
            polarized_optic, fields="all", num_rays=_NUM_RAYS, grid_size=_GRID_SIZE
        )
        assert len(mtf.psf) == len(mtf.resolved_fields)


# ---------------------------------------------------------------------------
# _calculate_psf uses VectorialFFTPSF
# ---------------------------------------------------------------------------


class TestVectorialFFTMTFUsesPSF:
    def test_psf_shape(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        for psf in mtf.psf:
            assert psf.shape == (_GRID_SIZE, _GRID_SIZE)

    def test_psf_values_nonnegative(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        for psf in mtf.psf:
            assert be.to_numpy(be.min(psf)) >= 0.0

    def test_calculate_psf_uses_vectorial_fft_psf(
        self, set_test_backend, polarized_optic
    ):
        """Verify _calculate_psf produces output from VectorialFFTPSF directly."""
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)

        # Reproduce a single-field VectorialFFTPSF and compare the PSF values
        field = mtf.resolved_fields[0]
        expected_psf = VectorialFFTPSF(
            polarized_optic,
            field,
            mtf.resolved_wavelength,
            _NUM_RAYS,
            _GRID_SIZE,
        ).psf
        assert_allclose(mtf.psf[0], expected_psf)


# ---------------------------------------------------------------------------
# MTF data
# ---------------------------------------------------------------------------


class TestVectorialFFTMTFData:
    def test_mtf_list_length_matches_fields(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(
            polarized_optic, fields="all", num_rays=_NUM_RAYS, grid_size=_GRID_SIZE
        )
        assert len(mtf.mtf) == len(mtf.resolved_fields)

    def test_mtf_has_tangential_and_sagittal(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        for field_mtf in mtf.mtf:
            tang, sag = field_mtf
            assert tang is not None
            assert sag is not None

    def test_mtf_values_in_range(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        for tang, sag in mtf.mtf:
            tang_np = be.to_numpy(tang)
            sag_np = be.to_numpy(sag)
            assert tang_np.min() >= -1e-6  # numerical noise tolerance
            assert tang_np.max() <= 1.0 + 1e-6
            assert sag_np.min() >= -1e-6
            assert sag_np.max() <= 1.0 + 1e-6

    def test_mtf_tangential_sagittal_lengths(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        for tang, sag in mtf.mtf:
            assert len(be.to_numpy(tang)) == _GRID_SIZE // 2
            assert len(be.to_numpy(sag)) == _GRID_SIZE // 2

    def test_generate_mtf_data_returns_list(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        result = mtf._generate_mtf_data()
        assert isinstance(result, list)
        assert len(result) == len(mtf.resolved_fields)

    def test_mtf_dc_value_is_one(self, set_test_backend, polarized_optic):
        """The MTF at zero frequency (DC) should be normalized to 1."""
        mtf = VectorialFFTMTF(
            polarized_optic, fields=[(0, 0)], num_rays=_NUM_RAYS, grid_size=_GRID_SIZE
        )
        tang, sag = mtf.mtf[0]
        assert_allclose(tang[0], 1.0, atol=1e-5)
        assert_allclose(sag[0], 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Polarization state variations
# ---------------------------------------------------------------------------


class TestVectorialFFTMTFPolarizations:
    def test_horizontal_polarization(self, set_test_backend, optic):
        state = PolarizationState(
            is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0
        )
        optic.updater.set_polarization(state)
        mtf = VectorialFFTMTF(optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        assert mtf.mtf is not None
        assert len(mtf.mtf) > 0

    def test_vertical_polarization(self, set_test_backend, optic):
        state = PolarizationState(
            is_polarized=True, Ex=0.0, Ey=1.0, phase_x=0.0, phase_y=0.0
        )
        optic.updater.set_polarization(state)
        mtf = VectorialFFTMTF(optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        assert mtf.mtf is not None

    def test_unpolarized_state(self, set_test_backend, unpolarized_state_optic):
        mtf = VectorialFFTMTF(
            unpolarized_state_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE
        )
        assert mtf.mtf is not None

    def test_h_and_v_produce_same_on_axis_mtf(self, set_test_backend, optic):
        """Horizontal and vertical polarizations yield identical on-axis MTF
        for a rotationally symmetric uncoated lens."""
        state_h = PolarizationState(
            is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0
        )
        optic.updater.set_polarization(state_h)
        mtf_h = VectorialFFTMTF(
            optic, fields=[(0, 0)], num_rays=_NUM_RAYS, grid_size=_GRID_SIZE
        )

        state_v = PolarizationState(
            is_polarized=True, Ex=0.0, Ey=1.0, phase_x=0.0, phase_y=0.0
        )
        optic.updater.set_polarization(state_v)
        mtf_v = VectorialFFTMTF(
            optic, fields=[(0, 0)], num_rays=_NUM_RAYS, grid_size=_GRID_SIZE
        )

        assert_allclose(mtf_h.mtf[0][0], mtf_v.mtf[0][0], atol=1e-2)
        assert_allclose(mtf_h.mtf[0][1], mtf_v.mtf[0][1], atol=1e-2)


# ---------------------------------------------------------------------------
# Consistency with ScalarFFTMTF
# ---------------------------------------------------------------------------


class TestVectorialVsScalarConsistency:
    def test_on_axis_mtf_close_to_scalar(self, set_test_backend, optic):
        """For an uncoated lens the vectorial MTF should closely match the
        scalar MTF on axis."""
        mtf_scalar = ScalarFFTMTF(
            optic, fields=[(0, 0)], num_rays=_NUM_RAYS, grid_size=_GRID_SIZE
        )

        state = PolarizationState(
            is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0
        )
        optic.updater.set_polarization(state)
        mtf_vec = VectorialFFTMTF(
            optic, fields=[(0, 0)], num_rays=_NUM_RAYS, grid_size=_GRID_SIZE
        )

        assert_allclose(mtf_scalar.mtf[0][0], mtf_vec.mtf[0][0], atol=5e-2)
        assert_allclose(mtf_scalar.mtf[0][1], mtf_vec.mtf[0][1], atol=5e-2)


# ---------------------------------------------------------------------------
# Multiple fields
# ---------------------------------------------------------------------------


class TestVectorialFFTMTFMultiField:
    def test_multiple_fields_all(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(
            polarized_optic, fields="all", num_rays=_NUM_RAYS, grid_size=_GRID_SIZE
        )
        n_fields = len(polarized_optic.fields.get_field_coords())
        assert len(mtf.mtf) == n_fields
        assert len(mtf.psf) == n_fields

    def test_single_off_axis_field(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(
            polarized_optic, fields=[(0, 0.7)], num_rays=_NUM_RAYS, grid_size=_GRID_SIZE
        )
        assert len(mtf.mtf) == 1

    def test_explicit_field_list(self, set_test_backend, polarized_optic):
        fields = [(0, 0), (0, 0.5), (0, 1.0)]
        mtf = VectorialFFTMTF(
            polarized_optic, fields=fields, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE
        )
        assert len(mtf.mtf) == 3


# ---------------------------------------------------------------------------
# Strategy options
# ---------------------------------------------------------------------------


class TestVectorialFFTMTFStrategies:
    @pytest.mark.parametrize(
        "strategy", ["chief_ray", "centroid_sphere", "best_fit_sphere"]
    )
    def test_strategy(self, set_test_backend, polarized_optic, strategy):
        mtf = VectorialFFTMTF(
            polarized_optic,
            fields=[(0, 0)],
            num_rays=_NUM_RAYS,
            grid_size=_GRID_SIZE,
            strategy=strategy,
        )
        assert mtf.mtf is not None
        assert len(mtf.mtf) == 1

    def test_remove_tilt_true(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(
            polarized_optic,
            fields=[(0, 0)],
            num_rays=_NUM_RAYS,
            grid_size=_GRID_SIZE,
            remove_tilt=True,
        )
        assert mtf.mtf is not None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


class TestVectorialFFTMTFView:
    def test_view_returns_figure_and_axes(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        fig, ax = mtf.view()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_add_reference(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        fig, ax = mtf.view(add_reference=True)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_custom_figsize(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        fig, ax = mtf.view(figsize=(16, 6))
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_on_existing_figure(self, set_test_backend, polarized_optic):
        """Embedding in an existing figure (GUI mode) should not raise."""
        mtf = VectorialFFTMTF(polarized_optic, num_rays=_NUM_RAYS, grid_size=_GRID_SIZE)
        existing_fig = plt.figure()
        fig, ax = mtf.view(fig_to_plot_on=existing_fig)
        assert fig is existing_fig
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_multiple_fields(self, set_test_backend, polarized_optic):
        mtf = VectorialFFTMTF(
            polarized_optic, fields="all", num_rays=_NUM_RAYS, grid_size=_GRID_SIZE
        )
        fig, ax = mtf.view()
        assert isinstance(fig, Figure)
        plt.close(fig)
