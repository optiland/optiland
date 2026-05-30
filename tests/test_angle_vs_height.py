"""Tests for the Incident Angle vs. Height Plot Analysis module."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.analysis.angle_vs_height import (
    BaseAngleVsHeightAnalysis,
    FieldIncidentAngleVsHeight,
    PupilIncidentAngleVsHeight,
)
from optiland.samples.objectives import CookeTriplet

from .utils import assert_allclose

# Use a non-interactive backend for testing to prevent plots from being displayed.
matplotlib.use("Agg")


@pytest.fixture
def cooke_triplet():
    """Provides a CookeTriplet instance for testing."""
    return CookeTriplet()


class TestPupilIncidentAngleVsHeight:
    """Tests for the PupilIncidentAngleVsHeight analysis class."""

    def test_initialization(self, set_test_backend, cooke_triplet):
        """Verify that the class is initialized with the correct attributes."""
        analysis = PupilIncidentAngleVsHeight(
            optic=cooke_triplet,
            surface_idx=-2,
            axis=0,
            wavelength=0.55,
            field=(0.1, 0.2),
            num_points=64,
        )
        assert analysis.optic is cooke_triplet
        assert analysis.surface_idx == -2
        assert analysis.axis == 0
        assert [wp.value for wp in analysis.wavelengths] == [0.55]
        assert analysis.field == (0.1, 0.2)
        assert analysis.num_points == 64

    def test_get_trace_coordinates_axis_y(self, set_test_backend, cooke_triplet):
        """
        Verify the generated trace coordinates when scanning the pupil in the
        Y direction.
        """
        analysis = PupilIncidentAngleVsHeight(
            optic=cooke_triplet, axis=1, field=(0.5, 0.5), num_points=10
        )
        scan_range = be.linspace(-1, 1, 10)
        Hx, Hy, Px, Py, coord_label = analysis._get_trace_coordinates(scan_range)

        assert coord_label == "Field"
        assert_allclose(Hx, be.full_like(scan_range, 0.5))
        assert_allclose(Hy, be.full_like(scan_range, 0.5))
        assert_allclose(Px, be.zeros_like(scan_range))
        assert_allclose(Py, scan_range)

    def test_get_trace_coordinates_axis_x(self, set_test_backend, cooke_triplet):
        """
        Verify the generated trace coordinates when scanning the pupil in the
        X direction.
        """
        analysis = PupilIncidentAngleVsHeight(
            optic=cooke_triplet, axis=0, field=(0.3, 0.6), num_points=10
        )
        scan_range = be.linspace(-1, 1, 10)
        Hx, Hy, Px, Py, coord_label = analysis._get_trace_coordinates(scan_range)

        assert coord_label == "Field"
        assert_allclose(Hx, be.full_like(scan_range, 0.3))
        assert_allclose(Hy, be.full_like(scan_range, 0.6))
        assert_allclose(Px, scan_range)
        assert_allclose(Py, be.zeros_like(scan_range))

    def test_generate_data(self, set_test_backend, cooke_triplet):
        """Test the data generation process and the structure of the output."""
        analysis = PupilIncidentAngleVsHeight(
            optic=cooke_triplet,
            field=(0.0, 0.7),
            wavelength="primary",
            num_points=16,
        )
        # Data dictionary keys are (Hx, Hy, wavelength)
        data_key = (0.0, 0.7, cooke_triplet.primary_wavelength)
        assert data_key in analysis.data

        result = analysis.data[data_key]
        assert result["fixed_coordinates"] == "Field"
        assert "height" in result
        assert "angle" in result
        assert "scan_range" in result
        assert len(result["height"]) == 16
        assert len(result["angle"]) == 16
        assert len(result["scan_range"]) == 16

        # Check some values
        assert_allclose(result["height"][0], 12.3949823, atol=1e-5)
        assert_allclose(np.rad2deg(result["angle"][0]), 18.96203899, atol=1e-5)
        assert_allclose(result["height"][-1], 12.46054762, atol=1e-5)
        assert_allclose(np.rad2deg(result["angle"][-1]), 8.20666509, atol=1e-5)

    def test_view(self, set_test_backend, cooke_triplet):
        """Test that the view method executes and calls plt.show()."""
        analysis = PupilIncidentAngleVsHeight(cooke_triplet)
        fig, ax = analysis.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(fig)


class TestFieldIncidentAngleVsHeight:
    """Tests for the FieldIncidentAngleVsHeight analysis class."""

    def test_initialization(self, set_test_backend, cooke_triplet):
        """Verify that the class is initialized with the correct attributes."""
        analysis = FieldIncidentAngleVsHeight(
            optic=cooke_triplet,
            surface_idx=-1,
            axis=1,
            wavelength="primary",
            pupil=(0.0, 0.0),
            num_points=128,
        )
        assert analysis.optic is cooke_triplet
        assert analysis.surface_idx == -1
        assert analysis.axis == 1
        assert [wp.value for wp in analysis.wavelengths] == [
            cooke_triplet.primary_wavelength
        ]
        assert analysis.pupil == (0.0, 0.0)
        assert analysis.num_points == 128

    def test_get_trace_coordinates_axis_y(self, set_test_backend, cooke_triplet):
        """
        Verify the generated trace coordinates when scanning the field in the
        Y direction.
        """
        analysis = FieldIncidentAngleVsHeight(
            optic=cooke_triplet, axis=1, pupil=(0.1, 0.8), num_points=10
        )
        scan_range = be.linspace(-1, 1, 10)
        Hx, Hy, Px, Py, coord_label = analysis._get_trace_coordinates(scan_range)

        assert coord_label == "Pupil"
        assert_allclose(Hx, be.zeros_like(scan_range))
        assert_allclose(Hy, scan_range)
        assert_allclose(Px, be.full_like(scan_range, 0.1))
        assert_allclose(Py, be.full_like(scan_range, 0.8))

    def test_get_trace_coordinates_axis_x(self, set_test_backend, cooke_triplet):
        """
        Verify the generated trace coordinates when scanning the field in the
        X direction.
        """
        analysis = FieldIncidentAngleVsHeight(
            optic=cooke_triplet, axis=0, pupil=(0.4, 0.2), num_points=10
        )
        scan_range = be.linspace(-1, 1, 10)
        Hx, Hy, Px, Py, coord_label = analysis._get_trace_coordinates(scan_range)

        assert coord_label == "Pupil"
        assert_allclose(Hx, scan_range)
        assert_allclose(Hy, be.zeros_like(scan_range))
        assert_allclose(Px, be.full_like(scan_range, 0.4))
        assert_allclose(Py, be.full_like(scan_range, 0.2))

    def test_generate_data(self, set_test_backend, cooke_triplet):
        """Test the data generation process and the structure of the output."""
        analysis = FieldIncidentAngleVsHeight(
            optic=cooke_triplet,
            pupil=(0.0, 0.0),
            wavelength="primary",
            num_points=16,
        )
        # Data dictionary keys are (Px, Py, wavelength)
        data_key = (0.0, 0.0, cooke_triplet.primary_wavelength)
        assert data_key in analysis.data

        result = analysis.data[data_key]
        assert result["fixed_coordinates"] == "Pupil"
        assert "height" in result
        assert "angle" in result
        assert "scan_range" in result
        assert len(result["height"]) == 16
        assert len(result["angle"]) == 16
        assert len(result["scan_range"]) == 16

        # Check some values (on-axis chief ray)
        assert_allclose(result["height"][8], 1.15905111, atol=1e-5)
        assert_allclose(np.rad2deg(result["angle"][8]), 1.3027495, atol=1e-5)
        assert_allclose(result["height"][-1], 18.13602593, atol=1e-5)
        assert_allclose(np.rad2deg(result["angle"][-1]), 18.98548523, atol=1e-5)

    def test_view(self, set_test_backend, cooke_triplet):
        """Test that the view method executes and calls plt.show()."""
        analysis = FieldIncidentAngleVsHeight(cooke_triplet)
        fig, ax = analysis.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestAngleVsHeightErrors:
    """Tests for error handling in the angle vs. height analysis module."""

    def test_invalid_coord_label_raises_error(self, set_test_backend, cooke_triplet):
        """
        Verify that an invalid coordinate label from _get_trace_coordinates
        raises a ValueError during data generation.
        """

        # Create a dummy class that returns a bad coordinate label
        class BadAnalysis(BaseAngleVsHeightAnalysis):
            def _get_trace_coordinates(self, scan_range):
                return (
                    be.array([0]),
                    be.array([0]),
                    be.array([0]),
                    be.array([0]),
                    "InvalidLabel",
                )

        with pytest.raises(ValueError) as excinfo:
            BadAnalysis(cooke_triplet)
        assert "Coord. label must be 'Pupil' or 'Field'." in str(excinfo.value)

    @patch("matplotlib.pyplot.show")
    def test_view_with_different_color_labels(
        self, mock_show, set_test_backend, cooke_triplet
    ):
        """
        Ensure the colorbar label changes correctly based on whether the
        pupil or field is being scanned.
        """
        # Mock the helper plot function to inspect its arguments
        with patch(
            "optiland.analysis.angle_vs_height._plot_angle_vs_height"
        ) as mock_plot:
            # Case 1: Field is scanned (PupilIncidentAngleVsHeight)
            pupil_scan_analysis = PupilIncidentAngleVsHeight(cooke_triplet, axis=0)
            pupil_scan_analysis.view()
            # Get the keyword arguments from the last call to the mock
            kwargs = mock_plot.call_args.kwargs
            assert "Normalized Pupil Coordinate (Px)" in kwargs["color_label"]

            mock_plot.reset_mock()

            # Case 2: Pupil is scanned (FieldIncidentAngleVsHeight)
            field_scan_analysis = FieldIncidentAngleVsHeight(cooke_triplet, axis=1)
            field_scan_analysis.view()
            kwargs = mock_plot.call_args.kwargs
            assert "Normalized Field Coordinate (Hy)" in kwargs["color_label"]

        plt.close("all")
