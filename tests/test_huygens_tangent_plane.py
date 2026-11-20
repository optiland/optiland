"""Huygens PSF image grid: tangent plane at the chief-ray intercept.

OpticStudio computes the Huygens PSF on the plane tangent to the image surface
at the chief-ray intercept, centered on the chief ray, with the grid spacing
measured in that plane. These tests pin that convention, and the
``image_solve`` focus for an image surface whose material differs from the
last gap.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.psf import HuygensPSF

from .utils import assert_allclose


def _monocentric_mirror(field_deg, curved_image=True, fno=20.0, radius=-100.0):
    """Spherical mirror with the stop at its center of curvature.

    With a concentric image surface, every field sees the same geometry,
    except that the flat stop is foreshortened by cos(theta) in y.
    """
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, thickness=-radius, is_stop=True)
    optic.surfaces.add(index=2, radius=radius, thickness=radius / 2, material="mirror")
    optic.surfaces.add(index=3, radius=(radius / 2 if curved_image else be.inf))
    optic.set_aperture("EPD", -radius / 2 / fno)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.fields.add(y=field_deg)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _plate_system(image_index, plate_index=1.5):
    """Paraxial lens focusing through a plate, image surface material varied."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, surface_type="paraxial", f=50.0, thickness=20.0, is_stop=True
    )
    optic.surfaces.add(
        index=2,
        radius=be.inf,
        thickness=30.0,
        material=IdealMaterial(n=plate_index),
    )
    optic.surfaces.add(index=3, material=IdealMaterial(n=image_index))
    optic.set_aperture("EPD", 2.5)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _fwhm(profile, pitch):
    """Full width at half maximum of a 1-D profile, linearly interpolated."""
    profile = np.asarray(profile)
    half = profile.max() / 2
    above = np.where(profile >= half)[0]
    i0, i1 = above[0], above[-1]
    left = i0 - (profile[i0] - half) / (profile[i0] - profile[i0 - 1])
    right = i1 + (profile[i1] - half) / (profile[i1] - profile[i1 + 1])
    return (right - left) * pitch


def _grid_step(psf):
    """Spacing of the image grid, which spans ``image_size * pixel_pitch``.

    The grid includes both end points, so adjacent samples are
    ``image_size / (image_size - 1)`` pixel pitches apart.
    """
    return psf.image_size * psf.pixel_pitch / (psf.image_size - 1)


def _local_grid(psf):
    """Image grid of ``psf`` in the image surface's local frame."""
    from optiland.visualization.system.utils import transform

    gx, gy, gz = psf._get_image_coordinates()
    x, y, z = transform(gx, gy, gz, psf.optic.image_surface, is_global=True)
    shape = (psf.image_size, psf.image_size)
    return (be.to_numpy(be.reshape(c, shape)) for c in (x, y, z))


def test_flat_image_grid_is_the_image_plane(set_test_backend):
    optic = _monocentric_mirror(20.0, curved_image=False)
    psf = HuygensPSF(optic, (0, 1), 0.55, num_rays=16, image_size=9, pixel_pitch=1e-3)
    x, y, z = _local_grid(psf)
    step = _grid_step(psf)
    assert_allclose(z, np.zeros_like(z), atol=1e-12)
    assert_allclose(np.diff(x, axis=1), np.full((9, 8), step))
    assert_allclose(np.diff(y, axis=0), np.full((8, 9), step))


def test_grid_is_centered_on_chief_ray(set_test_backend):
    optic = _monocentric_mirror(20.0)
    psf = HuygensPSF(optic, (0, 1), 0.55, num_rays=16, image_size=9, pixel_pitch=1e-3)
    chief = optic.trace_generic(0, 1, 0, 0, 0.55)
    assert_allclose(psf.cx, be.ravel(chief.x)[0], atol=1e-9)
    assert_allclose(psf.cy, be.ravel(chief.y)[0], atol=1e-9)


def test_curved_image_grid_lies_on_tangent_plane(set_test_backend):
    optic = _monocentric_mirror(20.0)
    pitch = 1e-3
    psf = HuygensPSF(optic, (0, 1), 0.55, num_rays=16, image_size=9, pixel_pitch=pitch)
    x, y, z = _local_grid(psf)
    center = np.array([x[4, 4], y[4, 4], z[4, 4]])

    # The center point sits on the image surface.
    sag = be.to_numpy(optic.image_surface.geometry.sag(center[0], center[1]))
    assert_allclose(center[2], sag, atol=1e-12)

    # Every grid point lies in the plane normal to the surface normal there.
    # For a sphere of radius R about its center of curvature, that normal is
    # along (x, y, z - R).
    radius = -50.0
    normal = center - np.array([0.0, 0.0, radius])
    normal /= np.linalg.norm(normal)
    offsets = np.stack([x, y, z], axis=-1) - center
    assert_allclose(offsets @ normal, np.zeros((9, 9)), atol=1e-12)

    # Spacing is measured in the tangent plane, not in projected local y.
    step_y = np.linalg.norm(offsets[1:, :] - offsets[:-1, :], axis=-1)
    step_x = np.linalg.norm(offsets[:, 1:] - offsets[:, :-1], axis=-1)
    step = _grid_step(psf)
    assert_allclose(step_y, np.full((8, 9), step))
    assert_allclose(step_x, np.full((9, 8), step))


@pytest.mark.parametrize("field_deg", [20.0, 35.0])
def test_monocentric_psf_width_follows_stop_foreshortening(field_deg):
    """On the tangent plane, only the stop's cos(theta) foreshortening remains.

    The tangential width grows by 1/cos(theta) and the sagittal width does
    not change. On the old projected grid the tangential width came out
    unchanged instead.
    """
    be.set_backend("numpy")
    optic = _monocentric_mirror(field_deg)
    pitch = 5e-4
    widths = {}
    for hy in (0, 1):
        psf = HuygensPSF(
            optic, (0, hy), 0.55, num_rays=96, image_size=81, pixel_pitch=pitch
        )
        image = be.to_numpy(psf.psf)
        iy, ix = np.unravel_index(image.argmax(), image.shape)
        widths[hy] = (_fwhm(image[iy, :], pitch), _fwhm(image[:, ix], pitch))

    cos_t = np.cos(np.radians(field_deg))
    assert widths[1][0] / widths[0][0] == pytest.approx(1.0, rel=2e-3)
    assert widths[1][1] / widths[0][1] == pytest.approx(1 / cos_t, rel=2e-3)


@pytest.mark.parametrize("image_index", [1.5, 1.0])
def test_image_solve_focuses_in_last_gap(set_test_backend, image_index):
    """The solve zeroes the marginal height even if the image material differs."""
    optic = _plate_system(image_index)
    optic.updater.image_solve()
    ya, _ = optic.paraxial.marginal_ray()
    assert_allclose(ya[-1, 0], 0.0, atol=1e-10)
    # Paraxial focus: 30 mm of air remains after the plate face, i.e. 45 mm
    # inside n = 1.5, and the plate starts at z = 20.
    assert_allclose(optic.surfaces[-1].geometry.cs.z, 65.0)
    assert_allclose(optic.surfaces[-2].thickness, 45.0)
