"""Tests for PR14: photometric conversion layer (D13).

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQScene,
    PointSourceConfig,
    SpectralDetectorConfig,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.sources.base import Spectrum as SpectrumCls
from optiland.nonsequential.units import (
    KM_PHOTOPIC,
    KM_SCOTOPIC,
    VISIBLE_BAND_UM,
    PhotometricMap,
    PhotometricScalar,
    lumens_to_watts,
    luminous_efficacy_of_spectrum,
    to_photometric,
    v_lambda,
)


def _lens_free_scene(detector_config, source_config=None) -> NSQScene:
    spec = Spectrum.monochromatic(0.555)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        source_config
        or CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=5.0),
    )
    scene.add_detector("D1", CoordinateSystem(z=10.0), detector_config)
    return scene


# ---------------------------------------------------------------------------
# v_lambda / luminous_efficacy_of_spectrum -- pure function tests
# ---------------------------------------------------------------------------


class TestVLambda:
    def test_photopic_peak_near_555nm(self):
        assert v_lambda(0.555) == pytest.approx(0.995, abs=0.01)

    def test_scotopic_peak_near_507nm(self):
        assert v_lambda(0.507, weighting="scotopic") == pytest.approx(1.0, abs=0.05)

    def test_zero_outside_visible_band(self):
        assert v_lambda(1.5) == 0.0
        assert v_lambda(0.2) == 0.0

    def test_zero_at_band_edges(self):
        lo, hi = VISIBLE_BAND_UM
        assert v_lambda(lo - 0.05) == 0.0
        assert v_lambda(hi + 0.05) == 0.0

    def test_vectorized_input(self):
        result = v_lambda(np.array([0.555, 1.5, 0.507]))
        assert result.shape == (3,)
        assert result[0] > result[2] > 0
        assert result[1] == 0.0

    def test_unknown_weighting_raises(self):
        with pytest.raises(ValueError):
            v_lambda(0.555, weighting="bogus")


class TestLuminousEfficacyOfSpectrum:
    def test_monochromatic_at_peak_gives_km(self):
        eff = luminous_efficacy_of_spectrum(
            np.array([0.555]), np.array([1.0]), "photopic"
        )
        # The 10 nm tabulation flanks the true 555 nm peak (550/560 are both
        # 0.9950 in this table, not the exact 1.0 CIE peak), so allow 1%.
        assert eff == pytest.approx(KM_PHOTOPIC, rel=0.01)

    def test_monochromatic_out_of_band_gives_zero(self):
        eff = luminous_efficacy_of_spectrum(
            np.array([1.5]), np.array([1.0]), "photopic"
        )
        assert eff == 0.0

    def test_scotopic_uses_different_table(self):
        eff_photopic = luminous_efficacy_of_spectrum(
            np.array([0.507]), np.array([1.0]), "photopic"
        )
        eff_scotopic = luminous_efficacy_of_spectrum(
            np.array([0.507]), np.array([1.0]), "scotopic"
        )
        assert eff_scotopic > eff_photopic
        assert eff_scotopic == pytest.approx(KM_SCOTOPIC, rel=0.05)


# ---------------------------------------------------------------------------
# lumens_to_watts
# ---------------------------------------------------------------------------


class TestLumensToWatts:
    def test_matches_km_at_peak_wavelength(self):
        spec = SpectrumCls.monochromatic(0.555)
        watts = lumens_to_watts(KM_PHOTOPIC, spec)
        assert watts == pytest.approx(1.0, rel=0.01)

    def test_scales_linearly_with_lumens(self):
        spec = SpectrumCls.monochromatic(0.555)
        w1 = lumens_to_watts(100.0, spec)
        w2 = lumens_to_watts(200.0, spec)
        assert w2 == pytest.approx(2 * w1)

    def test_dimmer_wavelength_needs_more_watts_for_same_lumens(self):
        spec_peak = SpectrumCls.monochromatic(0.555)
        spec_edge = SpectrumCls.monochromatic(0.450)
        w_peak = lumens_to_watts(10.0, spec_peak)
        w_edge = lumens_to_watts(10.0, spec_edge)
        assert w_edge > w_peak

    def test_out_of_band_spectrum_raises(self):
        spec_ir = SpectrumCls.monochromatic(1.5)
        with pytest.raises(ValueError, match="negligible overlap"):
            lumens_to_watts(100.0, spec_ir)

    def test_scotopic_weighting(self):
        spec = SpectrumCls.monochromatic(0.507)
        watts = lumens_to_watts(KM_SCOTOPIC, spec, weighting="scotopic")
        assert watts == pytest.approx(1.0, rel=0.05)


# ---------------------------------------------------------------------------
# to_photometric -- end to end against real trace results
# ---------------------------------------------------------------------------


class TestToPhotometricIrradianceMap:
    def test_requires_explicit_wavelength(self):
        scene = _lens_free_scene(
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=16, num_pixels_y=16
            )
        )
        result = scene.trace(num_rays=5000, seed=1, backend=NumpyBackend(seed=1))
        with pytest.raises(ValueError, match="wavelength_um"):
            to_photometric(result.detectors["D1"], quantity="illuminance")

    def test_out_of_band_wavelength_raises(self):
        scene = _lens_free_scene(
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=16, num_pixels_y=16
            )
        )
        result = scene.trace(num_rays=5000, seed=1, backend=NumpyBackend(seed=1))
        with pytest.raises(ValueError, match="negligible"):
            to_photometric(
                result.detectors["D1"], quantity="illuminance", wavelength_um=1.5
            )

    def test_illuminance_map_matches_manual_conversion(self):
        scene = _lens_free_scene(
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=16, num_pixels_y=16
            )
        )
        result = scene.trace(num_rays=50_000, seed=1, backend=NumpyBackend(seed=1))
        irr_map = result.detectors["D1"]
        photometric = to_photometric(
            irr_map, quantity="illuminance", wavelength_um=0.555
        )
        assert isinstance(photometric, PhotometricMap)
        expected_scale = KM_PHOTOPIC * v_lambda(0.555) * 1.0e6
        np.testing.assert_allclose(
            photometric.data, irr_map.irradiance * expected_scale, rtol=1e-9
        )

    def test_luminous_flux_scalar(self):
        scene = _lens_free_scene(
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=16, num_pixels_y=16
            )
        )
        result = scene.trace(num_rays=50_000, seed=1, backend=NumpyBackend(seed=1))
        lm = to_photometric(
            result.detectors["D1"], quantity="luminous_flux", wavelength_um=0.555
        )
        assert isinstance(lm, PhotometricScalar)
        expected = (
            result.detectors["D1"].total_flux_float * KM_PHOTOPIC * v_lambda(0.555)
        )
        assert float(lm) == pytest.approx(expected, rel=1e-6)

    def test_unknown_quantity_raises(self):
        scene = _lens_free_scene(
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=16, num_pixels_y=16
            )
        )
        result = scene.trace(num_rays=2000, seed=1, backend=NumpyBackend(seed=1))
        with pytest.raises(TypeError):
            to_photometric(
                result.detectors["D1"], quantity="bogus", wavelength_um=0.555
            )


class TestToPhotometricSpectralResult:
    def test_wavelength_auto_detected_from_bins(self):
        scene = _lens_free_scene(
            SpectralDetectorConfig(
                width=20,
                height=20,
                num_pixels_x=16,
                num_pixels_y=16,
                wl_min=0.4,
                wl_max=0.7,
                num_bins=30,
            )
        )
        result = scene.trace(num_rays=50_000, seed=1, backend=NumpyBackend(seed=1))
        photometric = to_photometric(result.detectors["D1"], quantity="illuminance")
        assert isinstance(photometric, PhotometricMap)
        assert photometric.total > 0

    def test_matches_irradiance_map_total_for_same_monochromatic_scene(self):
        irr_scene = _lens_free_scene(
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=16, num_pixels_y=16
            )
        )
        spec_scene = _lens_free_scene(
            SpectralDetectorConfig(
                width=20,
                height=20,
                num_pixels_x=16,
                num_pixels_y=16,
                wl_min=0.4,
                wl_max=0.7,
                num_bins=30,
            )
        )
        irr_result = irr_scene.trace(
            num_rays=50_000, seed=1, backend=NumpyBackend(seed=1)
        )
        spec_result = spec_scene.trace(
            num_rays=50_000, seed=1, backend=NumpyBackend(seed=1)
        )
        lux_from_irr = to_photometric(
            irr_result.detectors["D1"], quantity="illuminance", wavelength_um=0.555
        )
        lux_from_spec = to_photometric(
            spec_result.detectors["D1"], quantity="illuminance"
        )
        assert lux_from_spec.total == pytest.approx(lux_from_irr.total, rel=0.02)

    def test_out_of_band_bins_raise(self):
        scene = _lens_free_scene(
            SpectralDetectorConfig(
                width=20,
                height=20,
                num_pixels_x=16,
                num_pixels_y=16,
                wl_min=0.4,
                wl_max=0.7,
                num_bins=10,
            ),
            source_config=CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.555),
                total_flux=1.0,
                aperture_radius=5.0,
            ),
        )
        result = scene.trace(num_rays=2000, seed=1, backend=NumpyBackend(seed=1))
        spectral_result = result.detectors["D1"]
        # Fabricate an out-of-band copy to exercise the guardrail directly,
        # without needing an IR source (Spectrum forbids > 20 um already).
        spectral_result.wavelengths = np.array([1.0, 1.2, 1.4])
        with pytest.raises(ValueError, match="negligible|zero overlap"):
            to_photometric(spectral_result, quantity="illuminance")


class TestPhotometricMapPlot:
    def test_plot_smoke(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scene = _lens_free_scene(
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=16, num_pixels_y=16
            )
        )
        result = scene.trace(num_rays=5000, seed=1, backend=NumpyBackend(seed=1))
        photometric = to_photometric(
            result.detectors["D1"], quantity="illuminance", wavelength_um=0.555
        )
        fig, ax = plt.subplots()
        photometric.plot(ax=ax)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Sources specified in lumens
# ---------------------------------------------------------------------------


class TestSourceLumens:
    def test_point_source_lumens_converts_to_watts(self):
        spec = Spectrum.monochromatic(0.555)
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(),
            PointSourceConfig(spectrum=spec, total_flux_lumens=KM_PHOTOPIC),
        )
        scene.add_detector(
            "D1",
            CoordinateSystem(z=1.0),
            IrradianceDetectorConfig(
                width=200, height=200, num_pixels_x=8, num_pixels_y=8
            ),
        )
        result = scene.trace(num_rays=5000, seed=1, backend=NumpyBackend(seed=1))
        assert result.total_flux_in == pytest.approx(1.0, rel=0.01)

    def test_collimated_source_lumens_converts_to_watts(self):
        spec = Spectrum.monochromatic(0.555)
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(
                spectrum=spec, total_flux_lumens=2 * KM_PHOTOPIC, aperture_radius=5.0
            ),
        )
        scene.add_detector(
            "D1",
            CoordinateSystem(z=10.0),
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=8, num_pixels_y=8
            ),
        )
        result = scene.trace(num_rays=5000, seed=1, backend=NumpyBackend(seed=1))
        assert result.total_flux_in == pytest.approx(2.0, rel=0.01)

    def test_lumens_takes_precedence_and_warns_if_total_flux_also_set(self):
        spec = Spectrum.monochromatic(0.555)
        scene = NSQScene()
        # The precedence warning fires at add_source() time (that's where
        # _build_source/_resolve_total_flux run), not at trace() time.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scene.add_source(
                "S",
                CoordinateSystem(),
                CollimatedSourceConfig(
                    spectrum=spec,
                    total_flux=999.0,
                    total_flux_lumens=KM_PHOTOPIC,
                    aperture_radius=5.0,
                ),
            )
        assert any("total_flux_lumens" in str(w.message) for w in caught)

        scene.add_detector(
            "D1",
            CoordinateSystem(z=10.0),
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=8, num_pixels_y=8
            ),
        )
        result = scene.trace(num_rays=2000, seed=1, backend=NumpyBackend(seed=1))
        assert result.total_flux_in == pytest.approx(1.0, rel=0.01)

    def test_out_of_band_source_spectrum_raises_at_construction(self):
        spec_ir = Spectrum.monochromatic(1.5)
        scene = NSQScene()
        with pytest.raises(ValueError, match="negligible overlap"):
            scene.add_source(
                "S",
                CoordinateSystem(),
                PointSourceConfig(spectrum=spec_ir, total_flux_lumens=100.0),
            )

    def test_default_lumens_none_is_unchanged_behavior(self):
        spec = Spectrum.monochromatic(0.555)
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(),
            PointSourceConfig(spectrum=spec, total_flux=3.0),
        )
        scene.add_detector(
            "D1",
            CoordinateSystem(z=1.0),
            IrradianceDetectorConfig(
                width=200, height=200, num_pixels_x=8, num_pixels_y=8
            ),
        )
        result = scene.trace(num_rays=2000, seed=1, backend=NumpyBackend(seed=1))
        assert result.total_flux_in == pytest.approx(3.0, rel=1e-6)
