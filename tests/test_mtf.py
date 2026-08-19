from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")  # ensure non-interactive backend for testing

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.mtf import FFTMTF, GeometricMTF
from optiland.samples.objectives import CookeTriplet

# Parametrize every test over the available backends
pytestmark = pytest.mark.parametrize(
    "set_test_backend",
    be.list_available_backends(),
    indirect=True,
    ids=[f"backend={b}" for b in be.list_available_backends()],
)


@pytest.fixture
def optic():
    """A fresh CookeTriplet for each test."""
    return CookeTriplet()


class TestGeometricMTF:
    def test_view_mtf_defaults(self, set_test_backend, optic):
        m = GeometricMTF(optic)
        fig, ax = m.view()  # default figsize, no reference overlay
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_mtf_custom_fig(self, set_test_backend, optic):
        m = GeometricMTF(optic)
        fig, ax = m.view(figsize=(20, 20), add_reference=True)
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_generate_data_scaled(self, set_test_backend, optic):
        m = GeometricMTF(optic, scale=True)
        m._generate_mtf_data()
        assert m.data is not None, "Scaled MTF data should be generated"

    def test_generate_data_unscaled(self, set_test_backend, optic):
        m = GeometricMTF(optic, scale=False)
        m._generate_mtf_data()
        assert m.data is not None, "Unscaled MTF data should be generated"

    def test_max_freq_specification(self, set_test_backend, optic):
        m1 = GeometricMTF(optic)

        wavelength = optic.primary_wavelength
        expected_cutoff = 1 / (wavelength * 1e-3 * optic.paraxial.FNO())
        assert be.to_numpy(m1.max_freq) == pytest.approx(be.to_numpy(expected_cutoff))

        custom_freq = 50.0
        m2 = GeometricMTF(optic, max_freq=custom_freq)
        assert be.to_numpy(m2.max_freq) == pytest.approx(custom_freq)


class TestFFTMTF:
    def test_view_mtf_defaults(self, set_test_backend, optic):
        m = FFTMTF(optic)
        fig, ax = m.view()
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_mtf_custom_fig(self, set_test_backend, optic):
        m = FFTMTF(optic)
        fig, ax = m.view(figsize=(20, 20), add_reference=True)
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_generate_data_infinite_object(self, set_test_backend, optic):
        """Default (infinite object distance) should produce an MTF array."""
        m = FFTMTF(optic)
        m._generate_mtf_data()
        assert hasattr(m, "mtf") and m.mtf is not None

    def test_generate_data_finite_object(self, set_test_backend, optic):
        """With a finite object distance, MTF still gets generated."""
        # Push the first surface very far away to mimic a finite-object scenario
        optic.surfaces[0].geometry.cs.z = be.array(1e6)
        m = FFTMTF(optic)
        m._generate_mtf_data()
        assert hasattr(m, "mtf") and m.mtf is not None

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
    def test_num_rays_and_grid_size(
        self, set_test_backend, num_rays, expected_pupil_sampling, optic
    ):
        m = FFTMTF(optic, num_rays=num_rays, grid_size=None)

        assert m.num_rays == expected_pupil_sampling
        assert m.grid_size == 2 * num_rays

    def test_freq_step_image_plane_correction(self, set_test_backend, optic):
        """Frequency step applies image-plane cos(θ) correction for off-axis fields.

        The per-field working F/# is measured in the chief-ray frame.  For an
        off-axis field the convergent cone projects onto the flat image plane
        at an oblique angle, reducing the frequency step by FNO_on / FNO_off.
        Concretely:  df_tang[k] = df_chief[k] * (FNO_on / FNO_off[k])

        There is no tilt in the sagittal plane, so the sagittal step uses the
        uncorrected per-field df:  df_sag[k] = df_chief[k]

        On-axis the correction is unity; off-axis it is < 1 (smaller df,
        lower cutoff labelled frequency), removing the ~10 % stretch vs
        OpticStudio observed when using the raw per-field df.
        """
        from optiland.utils import get_working_FNO

        m = FFTMTF(optic)
        wl = m.resolved_wavelength
        N = m.num_rays
        fno_on = be.to_numpy(get_working_FNO(optic, (0.0, 0.0), wl)).item()

        for k, field in enumerate(m.resolved_fields):
            fno_off = be.to_numpy(get_working_FNO(optic, field, wl)).item()
            df_chief = 1.0 / ((N - 1) * wl * 1e-3 * fno_off)

            # Tangential: corrected by image-plane projection factor
            expected_df_tang = df_chief * (fno_on / fno_off)
            actual_df_tang = (
                be.to_numpy(m.freq_tang[k][1]).item()
                - be.to_numpy(m.freq_tang[k][0]).item()
            )
            assert actual_df_tang == pytest.approx(expected_df_tang, rel=1e-5), (
                f"field {field}: tang df={actual_df_tang:.5f} "
                f"expected={expected_df_tang:.5f}"
            )

            # Sagittal: no tilt correction, uses plain per-field df
            actual_df_sag = (
                be.to_numpy(m.freq_sag[k][1]).item()
                - be.to_numpy(m.freq_sag[k][0]).item()
            )
            assert actual_df_sag == pytest.approx(df_chief, rel=1e-5), (
                f"field {field}: sag df={actual_df_sag:.5f} expected={df_chief:.5f}"
            )

        # Off-axis tangential df must be strictly less than on-axis
        # (larger effective FNO due to image-plane correction)
        if len(m.resolved_fields) > 1:
            df_tang_off = (
                be.to_numpy(m.freq_tang[-1][1]).item()
                - be.to_numpy(m.freq_tang[-1][0]).item()
            )
            df_tang_on = (
                be.to_numpy(m.freq_tang[0][1]).item()
                - be.to_numpy(m.freq_tang[0][0]).item()
            )
            assert df_tang_off < df_tang_on
