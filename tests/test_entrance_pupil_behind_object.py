"""Paraxial aiming when a finite object is closer to the lens than its entrance pupil.

A virtual entrance pupil can lie behind the object, on the far side from the lens:
here a singlet with its stop beyond its focal length images the stop about 300 mm in
front of the lens, and the object is 100 mm away. Every ray must still leave the
object toward the lens, along the line from the object point through its point of
the entrance pupil, and the entrance pupil diameter must be positive.

Before the fix the aimer sent each ray from the object toward that pupil point, away
from the lens: the ray ran backwards (N < 0), its optical path came out negative, and
the wavefront had the wrong sign. The object-NA aperture also gave a negative
diameter, which mirrored the pupil, (Px, Py) -> (Px, -Py).
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic

from .utils import assert_allclose

OBJECT_DISTANCE = 100.0
NA = 0.05


def _pupil_behind_object() -> Optic:
    """A biconvex singlet (f about 51 mm) with its stop 60 mm behind it."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=OBJECT_DISTANCE, material=IdealMaterial(1.0))
    optic.surfaces.add(index=1, radius=50.0, thickness=5.0, material=IdealMaterial(1.5))
    optic.surfaces.add(
        index=2, radius=-50.0, thickness=60.0, material=IdealMaterial(1.0)
    )
    optic.surfaces.add(
        index=3, thickness=45.0, material=IdealMaterial(1.0), is_stop=True
    )
    optic.surfaces.add(index=4, material=IdealMaterial(1.0))
    optic.set_aperture("objectNA", NA)
    optic.fields.set_type("object_height")
    optic.fields.add(y=0.0)
    optic.fields.add(y=2.0)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _pupil_grid():
    px, py = np.meshgrid(np.linspace(-1.0, 1.0, 5), np.linspace(-1.0, 1.0, 5))
    keep = px**2 + py**2 <= 1.0
    return be.array(px[keep]), be.array(py[keep])


def _object_z(optic: Optic) -> float:
    return float(be.to_numpy(optic.object_surface.geometry.cs.z))


def test_the_entrance_pupil_lies_behind_the_object(set_test_backend):
    """The premise of the other tests: the pupil is on the far side of the object."""
    optic = _pupil_behind_object()
    pupil_z = float(be.to_numpy(optic.paraxial.entrance_pupil_axial_position()))
    assert pupil_z < _object_z(optic) - 100.0


def test_the_entrance_pupil_diameter_is_positive(set_test_backend):
    optic = _pupil_behind_object()
    epd = float(be.to_numpy(optic.paraxial.EPD()))
    pupil_z = float(be.to_numpy(optic.paraxial.entrance_pupil_axial_position()))
    distance = _object_z(optic) - pupil_z
    expected = 2.0 * distance * np.tan(np.arcsin(NA))
    assert epd > 0.0
    assert_allclose(epd, expected, rtol=1e-12)


@pytest.mark.parametrize("hy", [0.0, 1.0])
def test_every_ray_leaves_the_object_toward_the_lens(set_test_backend, hy):
    optic = _pupil_behind_object()
    px, py = _pupil_grid()
    rays = optic.ray_tracer.ray_generator.generate_rays(0.0, hy, px, py, 0.55)
    assert np.all(be.to_numpy(rays.N) > 0.0)


@pytest.mark.parametrize("hy", [0.0, 1.0])
def test_each_ray_lies_on_the_line_through_its_pupil_point(set_test_backend, hy):
    """(Px, Py) names the point (Px, Py) * EPD / 2 of the entrance pupil's plane."""
    optic = _pupil_behind_object()
    px, py = _pupil_grid()
    rays = optic.ray_tracer.ray_generator.generate_rays(0.0, hy, px, py, 0.55)
    x, y, z = (be.to_numpy(v) for v in (rays.x, rays.y, rays.z))
    L, M, N = (be.to_numpy(v) for v in (rays.L, rays.M, rays.N))
    pupil_z = float(be.to_numpy(optic.paraxial.entrance_pupil_axial_position()))
    t = (pupil_z - z) / N
    radius = float(be.to_numpy(optic.paraxial.EPD())) / 2.0
    assert_allclose(x + t * L, be.to_numpy(px) * radius, rtol=0, atol=1e-9)
    assert_allclose(y + t * M, be.to_numpy(py) * radius, rtol=0, atol=1e-9)


@pytest.mark.parametrize("hy", [0.0, 1.0])
def test_the_optical_path_to_the_image_is_positive(set_test_backend, hy):
    """A ray traced backwards had its whole path counted negative."""
    optic = _pupil_behind_object()
    px, py = _pupil_grid()
    rays = optic.trace_generic(0.0, hy, px, py, 0.55)
    opd = be.to_numpy(rays.opd)
    axial = (
        OBJECT_DISTANCE + 5.0 * 1.5 + 60.0 + 45.0
    )  # the object to the image along the axis, with the glass's index
    assert np.all(opd > 0.0)
    assert_allclose(opd, axial, rtol=0.02)
