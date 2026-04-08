from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.analysis import MTFvsField
from optiland.samples.objectives import CookeTriplet


class TestMTFvsField:
    """Test suite for the MTFvsField class."""

    @patch("optiland.analysis.MTFvsField._generate_data")
    def test_init_defaults(self, mock_generate_data, set_test_backend):
        """Test initialization with default parameters."""
        mock_generate_data.return_value = [
            [
                {"tangential": np.zeros(32), "sagittal": np.zeros(32)},
                {"tangential": np.zeros(32), "sagittal": np.zeros(32)},
            ]
            for _ in range(3)
        ]
        optic = CookeTriplet()
        mtf_vf = MTFvsField(optic, frequencies=[10.0, 20.0])

        assert mtf_vf.frequencies == [10.0, 20.0]
        assert mtf_vf.num_fields == 32
        assert mtf_vf.num_rays == 128
        assert [wp.value for wp in mtf_vf.wavelengths] == [
            w.value for w in optic.wavelengths.wavelengths
        ]

        # Check results structure
        assert len(mtf_vf.data) == len(mtf_vf.wavelengths)
        assert len(mtf_vf.data[0]) == len(mtf_vf.frequencies)

        # Check tangential and sagittal arrays
        res = mtf_vf.data[0][0]
        assert "tangential" in res
        assert "sagittal" in res
        assert len(res["tangential"]) == mtf_vf.num_fields
        assert len(res["sagittal"]) == mtf_vf.num_fields

    def test_init_limits_frequencies(self, set_test_backend):
        """Test the frequency limit checks."""
        optic = CookeTriplet()
        freqs = [10, 20, 30, 40, 50, 60]  # length 6 > MAX=5

        with pytest.raises(ValueError, match="Number of frequencies"):
            MTFvsField(optic, frequencies=freqs)

        # Try with override
        with patch("optiland.analysis.MTFvsField._generate_data"):
            mtf_vf = MTFvsField(
                optic,
                frequencies=freqs,
                num_rays=16,
                num_fields=2,
                override_limits=True,
            )
        assert len(mtf_vf.frequencies) == 6

    def test_init_limits_wavelengths(self, set_test_backend):
        """Test the frequency limit checks."""
        optic = CookeTriplet()
        optic.wavelengths.add(0.5, weight=1.0)
        optic.wavelengths.add(0.6, weight=1.0)
        optic.wavelengths.add(0.7, weight=1.0)
        # Optic now has 4 wavelengths (>3 MAX)

        with pytest.raises(ValueError, match="Number of wavelengths"):
            MTFvsField(optic, frequencies=[10], wavelengths="all")

        # Try with override
        with patch("optiland.analysis.MTFvsField._generate_data"):
            mtf_vf = MTFvsField(
                optic,
                frequencies=[10],
                wavelengths="all",
                num_rays=16,
                num_fields=2,
                override_limits=True,
            )
        assert len(mtf_vf.wavelengths) == 6

    def test_custom_params(self, set_test_backend):
        """Test initialization with custom parameters."""
        optic = CookeTriplet()
        mtf_vf = MTFvsField(
            optic, frequencies=[15.0], num_fields=3, wavelengths=[0.55], num_rays=16
        )

        assert mtf_vf.frequencies == [15.0]
        assert mtf_vf.num_fields == 3
        assert mtf_vf.num_rays == 16
        assert [wp.value for wp in mtf_vf.wavelengths] == [0.55]

        # Check data
        res = mtf_vf.data[0][0]
        assert len(res["tangential"]) == 3

        # Check values
        assert be.all(res["tangential"] >= 0.0)
        assert be.all(res["tangential"] <= 1.0)
        assert be.all(res["sagittal"] >= 0.0)
        assert be.all(res["sagittal"] <= 1.0)

    def test_view(self, set_test_backend):
        """Test view method to ensure it creates plots properly."""
        optic = CookeTriplet()
        mtf_vf = MTFvsField(optic, frequencies=[10.0], num_fields=2, num_rays=16)
        fig, ax = mtf_vf.view()

        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check label is present based on field definition in CookeTriplet
        assert ax.get_xlabel() in ["Angle (deg)", "Height (mm)", "Field Coordinate"]
        assert ax.get_ylabel() == "Modulus of the OTF"

        plt.close(fig)
