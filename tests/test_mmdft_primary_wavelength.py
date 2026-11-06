"""Regressions for resolved primary-wavelength MMDFT image sampling."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.psf import MMDFTPSF
from optiland.samples.objectives import CookeTriplet

from .utils import assert_allclose, assert_array_equal


@pytest.fixture
def optic(set_test_backend) -> CookeTriplet:
    """Use the same on-axis, rotationally symmetric lens for both inputs."""
    return CookeTriplet()


@pytest.mark.parametrize("primary_index", [1, 2], ids=["green", "red"])
@pytest.mark.parametrize(
    "image_size,pixel_pitch,expected_num_rays",
    [
        (None, None, 45),
        (48, None, 64),
        (None, 2.0, 64),
        (48, 2.0, 64),
    ],
    ids=["default", "image_size_only", "pixel_pitch_only", "both_supplied"],
)
def test_primary_wavelength_sampling(
    optic: CookeTriplet,
    primary_index: int,
    image_size: int | None,
    pixel_pitch: float | None,
    expected_num_rays: int,
) -> None:
    """Resolve the alias before deriving image sampling, in micrometers."""
    optic.wavelengths.primary_index = primary_index
    wavelength_um = optic.primary_wavelength

    # Independent on-axis reference: NA = n * sin(theta) for a marginal ray,
    # and working F/# = 1 / (2 * NA). The chief ray lies along the optical axis.
    marginal = optic.trace_generic(Hx=0, Hy=0, Px=0, Py=1, wavelength=wavelength_um)
    n_image = optic.image_surface.material_post.n(wavelength_um)
    numerical_aperture = float(
        be.to_numpy(n_image * be.sqrt(marginal.L**2 + marginal.M**2)).item()
    )
    working_fno = 1 / (2 * numerical_aperture)
    # Wavelength is in um; F/# and the pupil interval count are dimensionless.
    full_extent_um = wavelength_um * working_fno * (expected_num_rays - 1)
    expected_image_size = image_size
    if expected_image_size is None:
        expected_image_size = (
            128 if pixel_pitch is None else int(full_extent_um / pixel_pitch)
        )
    expected_pixel_pitch_um = (
        full_extent_um / expected_image_size if pixel_pitch is None else pixel_pitch
    )

    sampling = dict(num_rays=64, image_size=image_size, pixel_pitch=pixel_pitch)
    numeric = MMDFTPSF(optic, (0, 0), wavelength_um, **sampling)
    primary = MMDFTPSF(optic, (0, 0), "primary", **sampling)

    for psf in (numeric, primary):
        assert psf.wavelengths[0].value == wavelength_um
        assert psf.num_rays == expected_num_rays
        assert psf.image_size == expected_image_size
        assert psf.pupil.shape == (expected_num_rays, expected_num_rays)
        assert psf.psf.shape == (expected_image_size, expected_image_size)
        assert_allclose(
            psf.pixel_pitch, expected_pixel_pitch_um, rtol=1e-12, atol=1e-12
        )
        expected_extent_um = expected_image_size * expected_pixel_pitch_um
        assert_allclose(
            psf._get_psf_units(psf.psf),
            (expected_extent_um, expected_extent_um),
            rtol=1e-12,
            atol=1e-12,
        )
        # Extents must use actual image dimensions, including rectangular crops.
        assert_allclose(
            psf._get_psf_units(psf.psf[:7, :11]),
            (11 * expected_pixel_pitch_um, 7 * expected_pixel_pitch_um),
            rtol=1e-12,
            atol=1e-12,
        )
        intensity = be.to_numpy(psf.psf)
        assert np.isfinite(intensity).all()
        assert (intensity >= 0).all()
        assert 0 <= float(be.to_numpy(psf.strehl_ratio())) <= 1
        if image_size is None and pixel_pitch is not None:
            # Derivation must truncate fractional pixel counts, not round up.
            assert expected_image_size <= full_extent_um / pixel_pitch
            assert full_extent_um / pixel_pitch < expected_image_size + 1

    assert_array_equal(primary.pixel_pitch, numeric.pixel_pitch)
    assert_array_equal(primary.pupil, numeric.pupil)
    assert_array_equal(primary.psf, numeric.psf)
    assert_array_equal(primary.strehl_ratio(), numeric.strehl_ratio())
