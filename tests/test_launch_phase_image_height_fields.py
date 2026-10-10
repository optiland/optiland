"""Image-height fields of an object at infinity: the launch plane's phase is restored.

For an object at infinity, an angle field and the paraxial and real image-height fields
all launch a collimated beam from a common plane, the image-height fields at the angle
that reaches their image height. An image-height field is therefore the same beam as the
angle field aimed at the same chief ray, and must give the same wavefront.

Before the fix only angle fields had the incident phase across that plane restored, so
an off-axis image-height field kept the launch plane's tilt: about 2,700 waves on the
double Gauss at full field, which corrupts every PSF computed from it (issue #750).
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.samples.objectives import DoubleGauss
from optiland.wavefront import Wavefront

from .utils import assert_allclose


def _single_field(field_type: str, value: float) -> DoubleGauss:
    optic = DoubleGauss()
    while optic.fields.num_fields:
        optic.fields.remove(0)
    optic.fields.set_type(field_type)
    optic.fields.add(y=value)
    return optic


def _chief_angle(optic) -> float:
    """The field angle, in degrees, of the chief ray the optic launches at Hy = 1."""
    rays = optic.ray_tracer.ray_generator.generate_rays(
        0.0, 1.0, 0.0, 0.0, optic.primary_wavelength
    )
    M = float(be.to_numpy(rays.M).reshape(-1)[0])
    N = float(be.to_numpy(rays.N).reshape(-1)[0])
    return float(np.degrees(np.arctan2(M, N)))


def _opd(optic) -> np.ndarray:
    wavefront = Wavefront(
        optic,
        fields=[(0.0, 1.0)],
        wavelengths=[optic.primary_wavelength],
        num_rays=12,
        distribution="hexapolar",
        strategy="chief_ray",
    )
    data = next(iter(wavefront.data.values()))
    return be.to_numpy(data.opd)


@pytest.mark.parametrize("field_type", ["paraxial_image_height", "real_image_height"])
def test_an_image_height_field_has_the_wavefront_of_its_angle(
    set_test_backend, field_type
):
    """The same chief ray, named by its image height or by its angle."""
    height = _single_field(field_type, 20.0)
    angle = _single_field("angle", _chief_angle(height))

    assert_allclose(_opd(height), _opd(angle), rtol=0, atol=1e-6)


def test_an_off_axis_image_height_field_keeps_no_launch_plane_tilt(set_test_backend):
    """A tilt of thousands of waves is the launch plane's, not the lens's."""
    opd = _opd(_single_field("real_image_height", 20.0))
    assert np.ptp(opd) < 20.0
