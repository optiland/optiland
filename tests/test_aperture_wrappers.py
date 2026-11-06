"""Native aperture transforms and drawing boundaries independent of file import."""

from __future__ import annotations

import json
import math

import pytest

import optiland.backend as be
from optiland.physical_apertures import (
    BaseAperture,
    EllipticalAperture,
    RadialAperture,
    RectangularAperture,
    RotatedAperture,
    UnclippedAperture,
)
from optiland.rays import RealRays
from tests.utils import assert_allclose


def make_rays(x, y):
    count = len(x)
    return RealRays(
        x, y, be.zeros(count), be.zeros(count), be.zeros(count),
        be.ones(count), be.asarray([0.25, 0.75]), be.full(count, 0.55),
    )


@pytest.mark.parametrize("angle", [math.pi / 2, -math.pi / 2])
def test_rotated_rectangle_keeps_pivot_and_clips_in_rotated_frame(angle, set_test_backend):
    aperture = RotatedAperture(RectangularAperture(2, 6, 1, 3), angle, 4, 2)
    assert_allclose(aperture.extent, [3, 5, 0, 4])
    rays = make_rays(be.asarray([4.0, 5.1]), be.asarray([3.5, 2.0]))
    assert_allclose(aperture.contains(rays.x, rays.y), [True, False])
    aperture.clip(rays)
    assert_allclose(rays.i, [0.25, 0])


def test_nonright_rotation_has_expected_asymmetric_bounds_and_membership(set_test_backend):
    aperture = RotatedAperture(RectangularAperture(0, 2, 0, 1), math.pi / 4)
    root = math.sqrt(0.5)
    assert_allclose(aperture.extent, [-root, 2 * root, 0, 3 * root])
    assert_allclose(
        aperture.contains(be.asarray([0.0, 1.5]), be.asarray([root, 0.0])),
        [True, False],
    )


def test_rotation_native_reload_and_scaling_preserve_pivot(set_test_backend):
    original = RotatedAperture(RectangularAperture(2, 6, 1, 3), math.pi / 2, 4, 2)
    restored = BaseAperture.from_dict(json.loads(json.dumps(original.to_dict())))
    assert isinstance(restored, RotatedAperture)
    restored.scale(2)
    assert_allclose(restored.extent, [6, 10, 0, 8])
    assert (restored.center_x, restored.center_y) == (8, 4)
    assert_allclose(
        restored.contains(be.asarray([8, 10.2]), be.asarray([7, 4])), [True, False]
    )
    assert_allclose(original.extent, [3, 5, 0, 4])


def test_offset_ellipse_rotates_about_surface_origin(set_test_backend):
    aperture = RotatedAperture(EllipticalAperture(2, 1, 3, 4), math.pi / 2)
    assert_allclose(aperture.extent, [-5, -3, 1, 5])
    assert_allclose(aperture.contains(be.array([-4.0]), be.array([3.0])), [True])


def test_unclipped_boundary_retains_display_shape_and_all_ray_state(set_test_backend):
    aperture = UnclippedAperture(RadialAperture(r_max=1))
    rays = make_rays(be.asarray([0.0, 2.0]), be.asarray([0.0, 0.0]))
    before = {
        name: be.to_numpy(getattr(rays, name)).copy()
        for name in ("x", "y", "z", "L", "M", "N", "i", "w")
    }
    assert_allclose(aperture.extent, [-1, 1, -1, 1])
    assert_allclose(aperture.contains(rays.x, rays.y), [True, False])
    aperture.clip(rays)
    for name, expected in before.items():
        assert_allclose(getattr(rays, name), expected)


def test_nested_drawing_boundary_native_reload_and_scale(set_test_backend):
    original = UnclippedAperture(
        RotatedAperture(RectangularAperture(2, 6, 1, 3), math.pi / 2, 4, 2)
    )
    restored = BaseAperture.from_dict(json.loads(json.dumps(original.to_dict())))
    assert isinstance(restored, UnclippedAperture)
    assert isinstance(restored.aperture, RotatedAperture)
    restored.scale(0.5)
    assert_allclose(restored.extent, [1.5, 2.5, 0, 2])
    rays = make_rays(be.asarray([2.0, 20.0]), be.asarray([1.0, 1.0]))
    restored.clip(rays)
    assert_allclose(rays.i, [0.25, 0.75])


@pytest.mark.parametrize("drawing", [False, True])
def test_native_optic_reload_retains_surface_clipping_mode(drawing, set_test_backend):
    from optiland.optic import Optic

    aperture = RotatedAperture(RectangularAperture(2, 6, 1, 3), math.pi / 2, 4, 2)
    optic = Optic()
    optic.surfaces.add(index=0, thickness=math.inf)
    optic.surfaces.add(
        index=1, thickness=5, is_stop=True,
        aperture=UnclippedAperture(aperture) if drawing else aperture,
    )
    optic.surfaces.add(index=2)
    optic.set_aperture("EPD", 2)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    restored = Optic.from_dict(json.loads(json.dumps(optic.to_dict())))
    assert_allclose(restored.surfaces[1].aperture.extent, [3, 5, 0, 4])
    rays = make_rays(be.asarray([4.0, 5.1]), be.asarray([3.5, 2.0]))
    rays.z = be.full(2, -1.0)
    restored.surfaces[1].trace(rays)
    assert_allclose(rays.i, [0.25, 0.75 if drawing else 0.0])


@pytest.mark.parametrize("angle", [0.0, math.pi / 4, -math.pi / 2])
def test_rotation_of_unbounded_aperture_keeps_valid_extent(angle, set_test_backend):
    aperture = RotatedAperture(RadialAperture(r_max=math.inf), angle)
    assert aperture.extent == (-math.inf, math.inf, -math.inf, math.inf)
    assert_allclose(
        aperture.contains(be.asarray([0.0, 100.0]), be.asarray([0.0, -100.0])),
        [True, True],
    )


@pytest.mark.parametrize("field", ["angle", "center_x", "center_y"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rotation_requires_finite_parameters(field, value):
    options = {"angle": 0.0, "center_x": 0.0, "center_y": 0.0, field: value}
    with pytest.raises(ValueError, match="finite"):
        RotatedAperture(RectangularAperture(-1, 1, -1, 1), **options)
