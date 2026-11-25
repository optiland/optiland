"""Tests for optic_at_temperature, a deep copy of an optic at another temperature."""

from __future__ import annotations

import math
from copy import deepcopy

import pytest

import optiland.backend as be
from optiland.environment import optic_at_temperature
from optiland.environment.thermal import _thermal_scale_factors
from optiland.materials import IdealMaterial, MaterialAtConditions
from optiland.optic import Optic
from optiland.paraxial_path import UnsupportedParaxialGeometryError

from ..test_folded_paraxial import folded
from ..utils import assert_allclose

WAVELENGTH = 0.5876
REFERENCE = 20.0


def _scalar(value) -> float:
    """Convert a backend scalar to a Python float."""
    return float(be.to_numpy(value).reshape(-1)[0])


def _bk7_singlet(thickness=4.0, image_gap=100.0, *, solve_image=True):
    """Biconvex N-BK7 singlet, R = +/-102 mm, object at infinity."""
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=102.0,
        thickness=thickness,
        material="N-BK7",
        is_stop=True,
    )
    lens.surfaces.add(index=2, radius=-102.0, thickness=image_gap)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=20.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=WAVELENGTH, is_primary=True)
    lens.updater.update_paraxial()
    if solve_image:
        lens.updater.image_solve()
    return lens


def _ideal_singlet(index=1.5, radius=102.0, thickness=4.0, image_gap=100.0):
    """Biconvex singlet of a temperature-insensitive ideal glass."""
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=radius,
        thickness=thickness,
        material=IdealMaterial(index),
        is_stop=True,
    )
    lens.surfaces.add(index=2, radius=-radius, thickness=image_gap)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=20.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=WAVELENGTH, is_primary=True)
    lens.updater.update_paraxial()
    return lens


def _plane_gap():
    """Two plane surfaces with a finite air gap: a flat window in air."""
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1, radius=be.inf, thickness=10.0, material="N-BK7", is_stop=True
    )
    lens.surfaces.add(index=2, radius=be.inf, thickness=25.0)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=10.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=WAVELENGTH, is_primary=True)
    lens.updater.update_paraxial()
    return lens


def _curved_mirror(radius=-2000.0, mirror_thickness=6.0, image_gap=-50.0):
    """Concave mirror with a substrate thickness, focusing collimated light."""
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=radius,
        thickness=image_gap,
        material="mirror",
        mirror_thickness=mirror_thickness,
        is_stop=True,
    )
    lens.surfaces.add(index=2)
    lens.set_aperture(aperture_type="EPD", value=10.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=WAVELENGTH, is_primary=True)
    lens.updater.update_paraxial()
    return lens


def _trace(optic):
    return optic.trace(0.0, 0.7, WAVELENGTH, num_rays=32)


def test_reference_temperature_reproduces_the_original_trace(set_test_backend):
    original = _bk7_singlet()
    same = optic_at_temperature(original, REFERENCE, pressure=1.0)

    expected = _trace(original)
    got = _trace(same)
    for name in ("x", "y", "z", "L", "M", "N", "opd"):
        assert_allclose(getattr(got, name), getattr(expected, name))


def test_pure_expansion_scales_the_focal_length(set_test_backend):
    alpha = 8.0e-6
    delta_t = 40.0
    lens = _ideal_singlet()
    hot = optic_at_temperature(
        lens,
        REFERENCE + delta_t,
        mount_cte=alpha,
        medium_cte={1: alpha, 2: alpha},
    )

    f_nominal = float(be.to_numpy(lens.paraxial.f2()).reshape(-1)[0])
    f_hot = float(be.to_numpy(hot.paraxial.f2()).reshape(-1)[0])
    assert f_hot == pytest.approx(f_nominal * (1.0 + alpha * delta_t), rel=1e-10)


def test_flat_gap_thickness_scales_with_the_mount(set_test_backend):
    alpha = 23.6e-6
    delta_t = 40.0
    lens = _plane_gap()
    hot = optic_at_temperature(lens, REFERENCE + delta_t, mount_cte=alpha)

    # The solid 10 mm N-BK7 plate expands with its own CTE; the 25 mm gap
    # behind it expands with the mount.
    cte_glass = lens.surfaces[1].material_post.thermal_expansion
    assert float(hot.surfaces[1].thickness) == pytest.approx(
        10.0 * (1.0 + cte_glass * delta_t)
    )
    assert float(hot.surfaces[2].thickness) == pytest.approx(
        25.0 * (1.0 + alpha * delta_t)
    )


def test_curved_gap_with_rigid_mount_and_expanding_glass(set_test_backend):
    """Hand calculation of the centre thickness of a curved air gap.

    The 100 mm image gap sits between the curved back surface (R = -102 mm)
    and a flat image plane. ``mount_cte = 0``, so the edge thickness does
    not change and the contact heights stay at the nominal semi-apertures.
    The back surface scales with the glass CTE; the image plane is
    unchanged.

        e  = t + sag_image(h_image) - sag_back(h_back)
        e' = e
        t' = e' - sag_image'(h_image') + sag_back'(h_back')

    sag' is the sag of a deepcopy of the geometry after
    ``geometry.scale(k)``, evaluated at the contact height. The image is
    a plane, so its sag is zero before and after scaling.
    """
    alpha = 7.1e-6
    delta_t = 40.0
    lens = _bk7_singlet(thickness=4.0, image_gap=100.0, solve_image=False)
    hot = optic_at_temperature(lens, REFERENCE + delta_t, mount_cte=0.0)

    back, image = lens.surfaces[2], lens.surfaces[3]
    t = float(be.to_numpy(back.thickness).reshape(-1)[0])
    h_back = float(be.to_numpy(back.semi_aperture).reshape(-1)[0])
    h_image = float(be.to_numpy(image.semi_aperture).reshape(-1)[0])
    sag_back = float(be.to_numpy(back.geometry.sag(x=0, y=h_back)).reshape(-1)[0])
    sag_image = float(be.to_numpy(image.geometry.sag(x=0, y=h_image)).reshape(-1)[0])
    k = 1.0 + alpha * delta_t
    back_hot = deepcopy(back.geometry)
    back_hot.scale(k)
    image_hot = deepcopy(image.geometry)
    image_hot.scale(1.0)
    sag_back_hot = float(be.to_numpy(back_hot.sag(x=0, y=h_back)).reshape(-1)[0])
    sag_image_hot = float(be.to_numpy(image_hot.sag(x=0, y=h_image)).reshape(-1)[0])
    t_new = t + sag_image - sag_back - sag_image_hot + sag_back_hot

    assert float(
        be.to_numpy(hot.surfaces[2].thickness).reshape(-1)[0]
    ) == pytest.approx(t_new, abs=1e-12)
    # The gap is not simply scaled: the back radius changes at a fixed edge.
    assert not math.isclose(t_new, t * k)


def test_thin_lens_thermal_focal_shift(set_test_backend):
    """An N-BK7 singlet in an aluminium mount heated from 20 to 60 °C.

    Thin-lens estimate of the focal shift, for glass and mount expanding
    together (a uniform expansion of the whole lens scales both radii by k, so
    its focal length scales by k as well):

        df = f (alpha_glass - dn/dT / (n - 1)) dT

    with n(T) the relative index from ``MaterialAtConditions`` at the primary
    wavelength. Separately, ``optic_at_temperature`` moves the image surface
    by about -87 um, and the back focal point is then about -0.087 mm behind
    the thermally moved image surface.
    """
    alpha_mount = 23.6e-6
    delta_t = 40.0
    lens = _bk7_singlet()
    f_nominal = float(be.to_numpy(lens.paraxial.f2()).reshape(-1)[0])

    hot = optic_at_temperature(lens, REFERENCE + delta_t, mount_cte=alpha_mount)

    f_hot = float(be.to_numpy(hot.paraxial.f2()).reshape(-1)[0])
    df = f_hot - f_nominal

    glass = lens.surfaces[1].material_post
    n_ref = float(
        be.to_numpy(MaterialAtConditions(glass, REFERENCE).n(WAVELENGTH)).reshape(-1)[0]
    )
    n_hot = float(
        be.to_numpy(
            MaterialAtConditions(glass, REFERENCE + delta_t).n(WAVELENGTH)
        ).reshape(-1)[0]
    )
    dndt = (n_hot - n_ref) / delta_t
    alpha_glass = glass.thermal_expansion
    df_thin = f_nominal * (alpha_glass - dndt / (n_ref - 1.0)) * delta_t

    assert df == pytest.approx(df_thin, rel=0.05)
    assert df == pytest.approx(0.0060, abs=0.002)

    back_focus = float(be.to_numpy(hot.paraxial.F2()).reshape(-1)[0])
    assert back_focus == pytest.approx(-0.087, abs=0.01)


def test_missing_cte_raises_unless_overridden(set_test_backend):
    lens = _ideal_singlet(radius=be.inf)
    with pytest.raises(ValueError, match=r"surface 1"):
        optic_at_temperature(lens, REFERENCE + 40.0)

    alpha = 1.0e-5
    hot = optic_at_temperature(
        lens, REFERENCE + 40.0, medium_cte={1: alpha}, surface_cte={1: alpha}
    )
    assert float(hot.surfaces[1].thickness) == pytest.approx(4.0 * (1.0 + alpha * 40.0))


def test_input_optic_is_not_modified(set_test_backend):
    lens = _bk7_singlet()
    before = _trace(lens)
    radius = float(lens.surfaces[2].geometry.radius)
    thickness = float(lens.surfaces[1].thickness)

    hot = optic_at_temperature(lens, REFERENCE + 40.0, mount_cte=23.6e-6)
    assert hot is not lens

    after = _trace(lens)
    for name in ("x", "y", "z", "L", "M", "N", "opd"):
        assert_allclose(getattr(after, name), getattr(before, name))
    assert float(lens.surfaces[2].geometry.radius) == radius
    assert float(lens.surfaces[1].thickness) == thickness


def test_user_semi_aperture_is_preserved_across_backends(set_test_backend):
    """A user-set semi-aperture reaches the hot copy on every backend.

    ``copy.deepcopy`` fails on the torch backend once a paraxial update has
    left non-leaf tensors, so the copy is made through the ``to_dict`` /
    ``from_dict`` serializers, which drop ``semi_aperture``. The hot copy
    then recomputed it from paraxial ray heights, and both the preserved
    semi-aperture and the resulting thicknesses silently diverged from
    the numpy result.
    """
    if len(be.list_available_backends()) < 2:
        pytest.skip("requires both backends to compare")

    def build():
        lens = _bk7_singlet()
        lens.surfaces[2].set_semi_aperture(15.0)
        return lens

    # Reference: the same system brought to temperature under numpy.
    active_backend = be.get_backend()
    try:
        be.set_backend("numpy")
        reference = optic_at_temperature(build(), REFERENCE + 40.0, mount_cte=23.6e-6)
        expected_semi_apertures = [_scalar(s.semi_aperture) for s in reference.surfaces]
        expected_thicknesses = [_scalar(s.thickness) for s in reference.surfaces]
    finally:
        be.set_backend(active_backend)

    hot = optic_at_temperature(build(), REFERENCE + 40.0, mount_cte=23.6e-6)

    assert _scalar(hot.surfaces[2].semi_aperture) == pytest.approx(15.0)
    for surface, expected in zip(hot.surfaces, expected_semi_apertures):
        assert _scalar(surface.semi_aperture) == pytest.approx(expected, abs=1e-9)
    for surface, expected in zip(hot.surfaces, expected_thicknesses):
        got = _scalar(surface.thickness)
        if math.isinf(expected):
            assert math.isinf(got)
        else:
            assert got == pytest.approx(expected, abs=1e-9)


def test_mirror_thickness_follows_the_surface_cte(set_test_backend):
    """The mirror substrate thickness scales with its geometry CTE.

    A mirror in air takes the mount CTE unless ``surface_cte`` overrides
    it. The substrate must also survive the copy itself: the torch
    fallback copies through the serializers, which drop
    ``mirror_thickness``.
    """
    mount_cte = 23.6e-6
    substrate_cte = 5.0e-6
    delta_t = 40.0
    lens = _curved_mirror()
    assert _scalar(lens.surfaces[1].mirror_thickness) == 6.0

    hot = optic_at_temperature(lens, REFERENCE + delta_t, mount_cte=mount_cte)
    assert hot.surfaces[1].mirror_thickness == pytest.approx(
        6.0 * (1.0 + mount_cte * delta_t), abs=1e-12
    )

    hot = optic_at_temperature(
        lens, REFERENCE + delta_t, surface_cte={1: substrate_cte}
    )
    assert hot.surfaces[1].mirror_thickness == pytest.approx(
        6.0 * (1.0 + substrate_cte * delta_t), abs=1e-12
    )
    assert _scalar(lens.surfaces[1].mirror_thickness) == 6.0


def test_folded_systems_raise(set_test_backend):
    with pytest.raises(UnsupportedParaxialGeometryError):
        optic_at_temperature(folded(), REFERENCE + 10.0, mount_cte=23.6e-6)


def test_helper_has_no_side_effects_and_keeps_infinite_thickness(set_test_backend):
    lens = _bk7_singlet()
    radii = [float(s.geometry.radius) for s in lens.surfaces]
    thicknesses = [float(s.thickness) for s in lens.surfaces]

    factors, new_thicknesses = _thermal_scale_factors(
        lens.surfaces, 40.0, wavelength=WAVELENGTH, mount_cte=23.6e-6
    )

    assert [float(s.geometry.radius) for s in lens.surfaces] == radii
    assert [float(s.thickness) for s in lens.surfaces] == thicknesses
    assert len(factors) == len(new_thicknesses) == lens.surfaces.num_surfaces
    assert math.isinf(new_thicknesses[0])
    assert new_thicknesses[-1] is None
    assert all(factor > 1.0 for factor in factors)
