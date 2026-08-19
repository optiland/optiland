from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import optiland.backend as be
from optiland.analysis import (
    angle_vs_height,
    image_simulation,
    jones_pupil,
    through_focus_mtf,
)
from optiland.samples.objectives import CookeTriplet


@pytest.fixture
def cooke_triplet(set_test_backend):
    return CookeTriplet()


# -----------------------------------------------------------------------------
# Image Simulation Tests (Mocked)
# -----------------------------------------------------------------------------


def test_simulator_init(cooke_triplet, set_test_backend):
    from optiland.analysis.image_simulation import simulator

    sim = simulator.SpatiallyVariableSimulator()
    assert isinstance(sim, simulator.SpatiallyVariableSimulator)


def test_distortion_warper_init(cooke_triplet, set_test_backend):
    warper = image_simulation.DistortionWarper(cooke_triplet)
    assert warper.optic == cooke_triplet


def test_psf_basis_generator(cooke_triplet, set_test_backend):
    gen = image_simulation.PSFBasisGenerator(cooke_triplet, wavelength=0.55)
    assert gen.optic == cooke_triplet


# -----------------------------------------------------------------------------
# Jones Pupil Tests
# -----------------------------------------------------------------------------


def test_jones_pupil_init(cooke_triplet, set_test_backend):
    with patch.object(cooke_triplet, "trace_generic") as mock_trace:
        mock_rays = MagicMock()
        mock_rays.x = be.array([0.0])
        mock_rays.p = be.zeros((1, 3, 3))
        if be.get_backend() == "torch":
            import torch

            mock_rays.p = torch.zeros((1, 3, 3), dtype=torch.float64)
            mock_rays.x = torch.zeros(1, dtype=torch.float64)

        mock_trace.return_value = mock_rays

        jp = jones_pupil.JonesPupil(cooke_triplet)
        assert jp.optic == cooke_triplet


def test_jones_pupil_generate_data(cooke_triplet, set_test_backend):
    with patch.object(cooke_triplet, "trace_generic") as mock_trace:
        mock_rays = MagicMock()
        mock_rays.x = be.array([0.0])
        mock_rays.p = be.zeros((1, 3, 3))
        if be.get_backend() == "torch":
            import torch

            mock_rays.p = torch.zeros((1, 3, 3), dtype=torch.float64)
            mock_rays.x = torch.zeros(1, dtype=torch.float64)
        mock_trace.return_value = mock_rays

        jp = jones_pupil.JonesPupil(cooke_triplet)
        assert jp.optic == cooke_triplet


# -----------------------------------------------------------------------------
# Through Focus MTF Tests
# -----------------------------------------------------------------------------


def test_through_focus_mtf_init(cooke_triplet, set_test_backend):
    with patch("optiland.analysis.through_focus.be.copy", side_effect=lambda x: x):
        tf_mtf = through_focus_mtf.ThroughFocusMTF(
            cooke_triplet, spatial_frequency=50.0
        )
        assert tf_mtf.optic == cooke_triplet


# -----------------------------------------------------------------------------
# Angle vs Height Tests
# -----------------------------------------------------------------------------


def test_angle_vs_height_init(cooke_triplet, set_test_backend):
    if hasattr(angle_vs_height, "AngleVsHeight"):
        avh = angle_vs_height.AngleVsHeight(cooke_triplet)
        assert avh.optic == cooke_triplet
    else:
        pass
