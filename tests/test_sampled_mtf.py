"""Unit tests for the SampledMTF class."""

from __future__ import annotations

from optiland.mtf import SampledMTF
from optiland.samples.objectives import CookeTriplet
from tests.utils import assert_allclose


class TestSampledMTF:
    """Tests for the SampledMTF class."""

    def test_sampled_mtf_instantiation(self, set_test_backend):
        """Test that SampledMTF can be instantiated without errors."""
        optic = CookeTriplet()

        # On-axis field, primary wavelength from optic
        field = (0, 0)
        wavelength = optic.primary_wavelength

        sampled_mtf_instance = SampledMTF(
            optic=optic,
            field=field,
            wavelength=wavelength,
            num_rays=32,
            distribution="uniform",
            zernike_terms=37,
            zernike_type="fringe",
        )
        assert sampled_mtf_instance is not None
        assert sampled_mtf_instance.optic == optic
        assert sampled_mtf_instance.field == field
        assert sampled_mtf_instance.wavelength == wavelength
        assert sampled_mtf_instance.num_rays == 32

    def test_mtf_at_zero_frequency(self, set_test_backend):
        """Test that MTF at zero frequency is 1.0."""
        optic = CookeTriplet()
        field = optic.fields.get_field_coords()[0]

        sampled_mtf_instance = SampledMTF(
            optic=optic,
            field=field,
            wavelength=0.55,
        )

        frequencies = [(0.0, 0.0)]
        mtf_values = sampled_mtf_instance.calculate_mtf(frequencies)

        assert len(mtf_values) == 1
        assert_allclose(mtf_values[0], 1.0)

    def test_triplet_system_behavior(self, set_test_backend):
        """Test MTF properties for a well-behaved system."""
        optic = CookeTriplet()
        field = (0, 0)

        sampled_mtf_instance = SampledMTF(
            optic=optic, field=field, wavelength=0.55, num_rays=32, zernike_terms=37
        )

        freqs = [(30.0, 0.0), (0.0, 30.0), (20.0, 20.0)]
        mtf_values = sampled_mtf_instance.calculate_mtf(frequencies=freqs)

        assert len(mtf_values) == len(freqs)
        assert_allclose(mtf_values, [0.77596046, 0.77596046, 0.79466559])

    def test_defocused_system_behavior(self, set_test_backend):
        """Test that defocus lowers the MTF compared to a focused system."""
        optic = CookeTriplet()
        field = (0, 1)

        # Defocus
        optic.image_surface.geometry.cs.z = optic.image_surface.geometry.cs.z + 0.5

        sampled_mtf_focused = SampledMTF(
            optic=optic,
            field=field,
            wavelength=0.6,
            num_rays=32,
        )

        freq = (25.0, 0.0)  # A single non-zero frequency
        mtf_values = sampled_mtf_focused.calculate_mtf(frequencies=[freq])
        assert_allclose(mtf_values, 0.04456888)

    def test_calculate_mtf_multiple_calls(self, set_test_backend):
        """Test consistency of calculate_mtf with multiple calls."""
        optic = CookeTriplet()
        field = (0.5, 0.5)

        sampled_mtf_instance = SampledMTF(
            optic=optic,
            field=field,
            wavelength=0.6,
        )

        freq1 = (5.0, 0.0)
        freq2 = (10.0, 0.0)

        mtf_val1_list = sampled_mtf_instance.calculate_mtf(frequencies=[freq1])
        mtf_val2_list = sampled_mtf_instance.calculate_mtf(frequencies=[freq2])

        mtf_vals_combined_list = sampled_mtf_instance.calculate_mtf(
            frequencies=[freq1, freq2]
        )

        assert_allclose(mtf_val1_list + mtf_val2_list, mtf_vals_combined_list)

    def test_zero_xpd_handling(self, set_test_backend):
        """Test MTF calculation when Exit Pupil Diameter (XPD) is zero."""
        optic = CookeTriplet()
        field = (0, 0)

        sampled_mtf_instance = SampledMTF(
            optic=optic,
            field=field,
            wavelength=0.56,
        )

        sampled_mtf_instance.xpd = 0.0

        frequencies = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (5.0, 5.0)]
        expected_mtfs = [1.0, 0.0, 0.0, 0.0]

        mtf_values = sampled_mtf_instance.calculate_mtf(frequencies=frequencies)

        assert_allclose(mtf_values, expected_mtfs)
