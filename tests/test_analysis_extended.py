from __future__ import annotations

from unittest.mock import patch

import pytest

import optiland.backend as be
from optiland.analysis import (
    angle_vs_height,
    image_simulation,
    jones_pupil,
    through_focus_mtf,
)
from optiland.rays import PolarizedRays
from optiland.samples.objectives import CookeTriplet
from tests.utils import assert_allclose


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
    with patch.object(
        cooke_triplet, "trace_generic", wraps=cooke_triplet.trace_generic
    ) as trace:
        jp = jones_pupil.JonesPupil(cooke_triplet, grid_size=3)
        assert jp.optic == cooke_triplet
        assert trace.call_count == len(jp.wavelengths)
        assert cooke_triplet.polarization == "ignore"
        for data in jp.data:
            assert data["J"].shape == (9, 2, 2)
            assert be.all(be.isfinite(data["J"][4]))


def test_jones_pupil_generate_data(cooke_triplet, set_test_backend):
    def transmitted_rays(*, Hx, Hy, Px, Py, wavelength):
        # Supply real ray-owned source fields and power for every pupil sample.
        zero, one = be.zeros_like(Px), be.ones_like(Px)
        return PolarizedRays(Px, Py, zero, zero, zero, one, 0.25 * one, wavelength)

    with patch.object(
        cooke_triplet, "trace_generic", side_effect=transmitted_rays
    ) as trace:
        jp = jones_pupil.JonesPupil(cooke_triplet, field=(0, 1), grid_size=3)

    assert len(jp.data) == len(jp.wavelengths) == trace.call_count
    assert cooke_triplet.polarization == "ignore"
    for data, wavelength, call in zip(
        jp.data, jp.wavelengths, trace.call_args_list, strict=True
    ):
        assert call.kwargs["Hx"] == 0
        assert call.kwargs["Hy"] == 1
        assert call.kwargs["wavelength"] == wavelength.value
        assert data["Px"].shape == data["Py"].shape == (9,)
        assert_allclose(data["Px"], call.kwargs["Px"])
        assert_allclose(data["Py"], call.kwargs["Py"])
        assert_allclose(data["J"], 0.5 * be.tile(be.eye(2), (9, 1, 1)))


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
