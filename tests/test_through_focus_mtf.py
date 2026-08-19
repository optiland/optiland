from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from optiland.analysis import ThroughFocusMTF
from optiland.samples.objectives import CookeTriplet


class TestThroughFocusMTF:
    """Test suite for the ThroughFocusMTF class."""

    def test_init_defaults(self, set_test_backend):
        """Test initialization with default parameters."""
        optic = CookeTriplet()
        tfm = ThroughFocusMTF(optic, spatial_frequency=10.0)

        assert tfm.spatial_frequency == 10.0
        assert tfm.num_rays == 128
        assert tfm.wavelength == optic.primary_wavelength
        assert tfm.delta_focus == 0.1  # Default
        assert tfm.num_steps == 5  # Default
        assert len(tfm.fields) == len(optic.fields.fields)  # Default "all"
        assert [wp.value for wp in tfm.wavelengths] == [optic.primary_wavelength]

        # Check results structure (analysis is done in __init__ via parent)
        assert len(tfm.results) == tfm.num_steps
        assert len(tfm.results[0]) == len(tfm.fields)

    def test_init_custom_params(self, set_test_backend):
        """Test initialization with custom parameters."""
        optic = CookeTriplet()
        custom_fields = [(0.0, 0.0), (0.1, 0.7)]
        custom_wl = 0.550
        tfm = ThroughFocusMTF(
            optic=optic,
            spatial_frequency=25.0,
            delta_focus=0.05,
            num_steps=3,
            fields=custom_fields,
            wavelength=custom_wl,
            num_rays=23,
        )

        assert tfm.spatial_frequency == 25.0
        assert tfm.num_rays == 23
        assert tfm.wavelength == custom_wl
        assert tfm.delta_focus == 0.05
        assert tfm.num_steps == 3
        assert [fp.coord for fp in tfm.fields] == custom_fields
        assert [wp.value for wp in tfm.wavelengths] == [custom_wl]

        assert len(tfm.results) == 3
        assert len(tfm.results[0]) == len(custom_fields)

    def test_init_wavelength_direct_float(self, set_test_backend):
        """Test initialization with a specific float wavelength."""
        optic = CookeTriplet()
        wavelength_val = 0.6328
        tfm = ThroughFocusMTF(
            optic,
            spatial_frequency=5.0,
            wavelength=wavelength_val,
            num_rays=23,
            num_steps=3,
        )
        assert tfm.wavelength == wavelength_val
        assert [wp.value for wp in tfm.wavelengths] == [wavelength_val]

    def test_init_invalid_num_steps(self, set_test_backend):
        """Test initialization with invalid num_steps values."""
        optic = CookeTriplet()

        # Test MIN_STEPS boundary
        with pytest.raises(ValueError):
            ThroughFocusMTF(optic, 10.0, num_steps=0)

        # Test MAX_STEPS boundary
        with pytest.raises(ValueError):
            ThroughFocusMTF(optic, 10.0, num_steps=17)

        # Test even number of steps
        with pytest.raises(ValueError):
            ThroughFocusMTF(optic, 10.0, num_steps=4)

        # Test non-integer number of steps
        with pytest.raises(ValueError):
            ThroughFocusMTF(optic, 10.0, num_steps=3.5)

    def test_analysis_results_structure_and_values(self, set_test_backend):
        """Test the structure and basic validity of the analysis results."""
        optic = CookeTriplet()
        tfm = ThroughFocusMTF(
            optic=optic,
            spatial_frequency=25.0,
            delta_focus=0.05,
            num_steps=1,
            num_rays=23,
        )
        assert len(tfm.results) == tfm.num_steps

        for step_result in tfm.results:
            assert len(step_result) == len(tfm.fields)
            for field_result in step_result:
                assert "tangential" in field_result
                assert "sagittal" in field_result
                # MTF values should be between 0 and 1 (inclusive)
                assert 0.0 <= field_result["tangential"] <= 1.0
                assert 0.0 <= field_result["sagittal"] <= 1.0

    def test_view_min_steps(self, set_test_backend):
        """Test view method with num_steps=1 (k=0 spline order)."""
        optic = CookeTriplet()
        tfm = ThroughFocusMTF(optic, spatial_frequency=10.0, num_steps=1)
        fig, ax = tfm.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)  # Close the figure to free resources

    def test_view_few_steps(self, set_test_backend):
        """Test view method with num_steps=3 (k=1 spline order)."""
        optic = CookeTriplet()
        tfm = ThroughFocusMTF(optic, spatial_frequency=10.0, num_steps=3)
        fig, ax = tfm.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_default_steps(self, set_test_backend):
        """Test view method with default num_steps=5 (k=3 spline order)."""
        optic = CookeTriplet()
        tfm = ThroughFocusMTF(optic, spatial_frequency=152.0, num_rays=53)
        fig, ax = tfm.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_single_field(self, set_test_backend):
        """Test view method with a single specified field."""
        optic = CookeTriplet()
        tfm = ThroughFocusMTF(
            optic,
            spatial_frequency=10.0,
            num_steps=5,
            num_rays=23,
            fields=[(0.0, 0.0)],  # Single field
        )
        assert len(tfm.fields) == 1
        fig, ax = tfm.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)
