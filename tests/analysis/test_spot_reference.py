from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.analysis.spot_diagram import SpotDiagram
from optiland.analysis.spot_diagram.reference import (
    CentroidReference,
    ChiefRayReference,
)
from optiland.samples.objectives import CookeTriplet


@pytest.fixture
def cooke_triplet():
    return CookeTriplet()


class TestSpotReference:
    def test_default_reference(self, set_test_backend, cooke_triplet):
        """SpotDiagram should default to ChiefRayReference."""
        spot = SpotDiagram(cooke_triplet)
        assert isinstance(spot._reference_strategy, ChiefRayReference)

    def test_centroid_reference(self, set_test_backend, cooke_triplet):
        """SpotDiagram with centroid reference should use CentroidReference."""
        spot = SpotDiagram(cooke_triplet, reference="centroid")
        assert isinstance(spot._reference_strategy, CentroidReference)

    def test_invalid_reference(self, cooke_triplet):
        """SpotDiagram with invalid reference should raise ValueError."""
        with pytest.raises(ValueError):
            SpotDiagram(cooke_triplet, reference="invalid_ref")

    def test_centroid_centering_logic(self, set_test_backend, cooke_triplet):
        """When using centroid reference, the mean of the centered rays at the ref wavelength should be ~zero."""
        spot = SpotDiagram(cooke_triplet, reference="centroid")
        centered_data = spot._center_spots(spot.data)

        ref_idx = spot._analysis_ref_wavelength_index
        for field_data in centered_data:
            ref_spot = field_data[ref_idx]
            mean_x = be.to_numpy(be.mean(ref_spot.x)).item()
            mean_y = be.to_numpy(be.mean(ref_spot.y)).item()

            # The centered spots should have a centroid of (0, 0)
            assert np.isclose(mean_x, 0.0, atol=1e-7)
            assert np.isclose(mean_y, 0.0, atol=1e-7)

    def test_chief_ray_centering_logic(self, set_test_backend, cooke_triplet):
        """When using chief ray reference, the chief ray coordinates should be (0, 0) in the centered data."""
        spot = SpotDiagram(cooke_triplet, reference="chief_ray")

        ref_idx = spot._analysis_ref_wavelength_index
        ref_wl = spot.wavelengths[ref_idx]

        # Center the data using the chief ray
        centered_data = spot._center_spots(spot.data)

        # Also compute the uncentered chief ray centers
        chief_centers = spot.generate_chief_rays_centers(ref_wl.value)
        chief_centers_np = be.to_numpy(chief_centers)

        # Original centroid of the data (not centered on chief ray yet)
        raw_centroid = spot.centroid()

        for i, field_data in enumerate(centered_data):
            # The chief ray should be at (0, 0) in the centered data plot.
            # But the spots are just dots. The difference between the uncentered chief ray
            # and the uncentered centroid should match the centered centroid.

            ref_spot = field_data[ref_idx]
            centered_mean_x = be.to_numpy(be.mean(ref_spot.x)).item()
            centered_mean_y = be.to_numpy(be.mean(ref_spot.y)).item()

            raw_cx, raw_cy = raw_centroid[i]
            chief_x, chief_y = chief_centers_np[i]

            # The centered data centroid should equal (raw centroid - chief ray)
            expected_mean_x = be.to_numpy(raw_cx).item() - chief_x
            expected_mean_y = be.to_numpy(raw_cy).item() - chief_y

            assert np.isclose(centered_mean_x, expected_mean_x, atol=1e-7)
            assert np.isclose(centered_mean_y, expected_mean_y, atol=1e-7)

    def test_different_radii(self, set_test_backend, cooke_triplet):
        """The RMS radii should generally be different under different reference choices for off-axis fields."""
        spot_chief = SpotDiagram(cooke_triplet, reference="chief_ray")
        spot_centroid = SpotDiagram(cooke_triplet, reference="centroid")

        rms_chief = spot_chief.rms_spot_radius()
        rms_centroid = spot_centroid.rms_spot_radius()

        # Field 1 and 2 are off-axis (Cooke triplet has fields at Y=0, Y=10, Y=20 roughly)
        # Check that field 1 or 2 at reference wavelength are different

        f1_chief = be.to_numpy(rms_chief[1][1]).item()
        f1_centroid = be.to_numpy(rms_centroid[1][1]).item()

        assert not np.isclose(f1_chief, f1_centroid, atol=1e-5), (
            "RMS should differ between chief ray and centroid for off-axis fields."
        )
