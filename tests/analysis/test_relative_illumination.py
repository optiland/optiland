"""Tests for the RelativeIllumination analysis.

The reference values come from ideal (paraxial) lenses, where the image-space
direction tangents of a collimated beam fill a disk exactly. For a lens of
focal length f with the stop a distance d in front of it, the beam at field
angle theta leaves the lens with tangents in a disk of radius D / (2 f)
centered on tan(theta) * (1 - d / f). The projected solid angle is the
integral of (1 + p^2 + q^2)^-2 over that disk.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pytest
from scipy import integrate

import optiland.backend as be
from optiland.analysis import RelativeIllumination
from optiland.distribution import UniformDistribution
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.samples.objectives import CookeTriplet
from tests.utils import assert_allclose

matplotlib.use("Agg")


def _ideal_lens(f=100.0, fno=10.0, max_field=20.0, stop_distance=0.0):
    """Ideal lens, object at infinity, image at the focal plane.

    Args:
        f: Focal length in mm.
        fno: F-number (EPD = f / fno).
        max_field: Maximum field angle in degrees.
        stop_distance: Distance of the stop in front of the lens, in mm.
            0 puts the stop at the lens, f makes image space telecentric.
    """
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    if stop_distance == 0:
        optic.surfaces.add(
            index=1, surface_type="paraxial", f=f, thickness=f, is_stop=True
        )
        optic.surfaces.add(index=2)
    else:
        optic.surfaces.add(index=1, thickness=stop_distance, is_stop=True)
        optic.surfaces.add(index=2, surface_type="paraxial", f=f, thickness=f)
        optic.surfaces.add(index=3)
    optic.set_aperture("EPD", f / fno)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.fields.add(y=max_field)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _tangent_disk_area(center, radius):
    """Integral of (1 + p^2 + q^2)^-2 over a disk centered at (0, center)."""

    def integrand(r, phi):
        p = r * np.cos(phi)
        q = center + r * np.sin(phi)
        return r / (1 + p * p + q * q) ** 2

    value, _ = integrate.dblquad(integrand, 0, 2 * np.pi, 0, radius, epsabs=1e-14)
    return value


def _exact_relative_illumination(hy, f, fno, max_field, stop_distance):
    theta = np.radians(be.to_numpy(hy) * max_field)
    center = np.tan(theta) * (1 - stop_distance / f)
    area = np.array([_tangent_disk_area(c, 0.5 / fno) for c in center])
    return area / area.max()


def _caustic_lens():
    """A plate whose rear asphere slope rises, then falls, across the pupil.

    The image-space directions of the outer pupil zone fold back over those
    of the inner zone, a caustic in direction space.
    """
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, thickness=5, is_stop=True, material=IdealMaterial(n=1.5)
    )
    optic.surfaces.add(
        index=2,
        surface_type="even_asphere",
        radius=-1e9,
        conic=0,
        coefficients=[-0.02, 0.02 / 54],
        thickness=50,
    )
    optic.surfaces.add(index=3)
    optic.set_aperture("EPD", 10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.fields.add(y=5)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def test_stop_at_lens_matches_exact_and_cos4(set_test_backend):
    optic = _ideal_lens(f=100.0, fno=10.0, max_field=20.0)
    ri = RelativeIllumination(optic, num_fields=9)
    got = ri.data["relative_illumination"][0]
    exact = _exact_relative_illumination(ri.data["Hy"], 100.0, 10.0, 20.0, 0.0)
    assert_allclose(got, exact, rtol=0, atol=1e-6)

    # The acceptance criterion of the issue: cos^4 within 0.5 % for f/10.
    theta = np.radians(be.to_numpy(ri.data["Hy"]) * 20.0)
    assert_allclose(got, np.cos(theta) ** 4, rtol=5e-3, atol=0)


def test_shifted_stop_is_not_cos4(set_test_backend):
    optic = _ideal_lens(f=100.0, fno=10.0, max_field=25.0, stop_distance=50.0)
    ri = RelativeIllumination(optic, num_fields=6)
    got = ri.data["relative_illumination"][0]
    exact = _exact_relative_illumination(ri.data["Hy"], 100.0, 10.0, 25.0, 50.0)
    assert_allclose(got, exact, rtol=0, atol=1e-6)
    # Far from cos^4 (0.675 at 25 degrees).
    assert float(be.to_numpy(got)[-1]) > 0.89


def test_image_space_telecentric_is_flat(set_test_backend):
    optic = _ideal_lens(f=100.0, fno=10.0, max_field=20.0, stop_distance=100.0)
    ri = RelativeIllumination(optic, num_fields=5)
    assert_allclose(ri.data["relative_illumination"][0], np.ones(5), atol=1e-9)


@pytest.mark.parametrize("num_pupil", [5, 64])
def test_on_axis_projected_solid_angle(set_test_backend, num_pupil):
    # On axis the cone is circular with marginal tangent 0.05, so it covers
    # pi * sin(u)^2 in direction-cosine space, sin(u) = 0.05 / sqrt(1.0025).
    optic = _ideal_lens(f=100.0, fno=10.0)
    ri = RelativeIllumination(optic, num_fields=3, num_pupil=num_pupil)
    expected = np.pi * 0.05**2 / (1 + 0.05**2)
    assert_allclose(ri.data["irradiance"][0][0], expected, rtol=1e-9)


def test_vignetting_reduces_illumination(set_test_backend):
    nominal = RelativeIllumination(CookeTriplet(), num_fields=5)
    optic = CookeTriplet()
    optic.surfaces[1].aperture = RadialAperture(r_max=6.0)
    clipped = RelativeIllumination(optic, num_fields=5)

    ratio = be.to_numpy(clipped.data["irradiance"][0]) / be.to_numpy(
        nominal.data["irradiance"][0]
    )
    # The on-axis beam passes; off-axis beams are clipped more and more.
    assert_allclose(ratio[:2], np.ones(2), rtol=1e-12)
    assert np.all(np.diff(ratio[1:]) < 0)

    # The loss tracks the fraction of pupil rays that the aperture blocks.
    # It is not equal to it: the pupil-to-direction Jacobian varies across
    # the pupil, so the clipped direction-space area differs from the clipped
    # ray fraction. Converged (num_pupil=256), the gap at full field is
    # 0.0043; the test allows 0.01.
    pupil = UniformDistribution()
    pupil.generate_points(201)
    px, py = pupil.x, pupil.y
    hy = be.to_numpy(clipped.data["Hy"])
    passed = []
    for h in hy:
        optic.trace_generic(be.zeros_like(px), be.full_like(px, float(h)), px, py, 0.55)
        passed.append(float(np.mean(be.to_numpy(optic.surfaces.intensity[-1]) > 0)))
    assert_allclose(ratio, passed, rtol=0, atol=1e-2)


def test_vignetted_sampling_converges(set_test_backend):
    # The convergence criterion of the issue (64 vs 128 within 0.5 %), on a
    # vignetted lens. Partially vignetted cells are weighted by their lit
    # corners, so the vignetting edge converges more slowly than the pupil
    # circle (about 2e-3 from the converged value at num_pupil=64).
    optic = CookeTriplet()
    optic.surfaces[1].aperture = RadialAperture(r_max=6.0)
    coarse = RelativeIllumination(optic, num_fields=5, num_pupil=64)
    fine = RelativeIllumination(optic, num_fields=5, num_pupil=128)
    assert_allclose(
        coarse.data["relative_illumination"][0],
        fine.data["relative_illumination"][0],
        rtol=5e-3,
        atol=0,
    )


def test_pupil_sampling_converges(set_test_backend):
    optic = CookeTriplet()
    coarse = RelativeIllumination(optic, num_fields=8, num_pupil=64)
    fine = RelativeIllumination(optic, num_fields=8, num_pupil=128)
    assert_allclose(
        coarse.data["relative_illumination"][0],
        fine.data["relative_illumination"][0],
        rtol=5e-3,
        atol=0,
    )


def test_real_lens_is_reliable_and_normalized(set_test_backend):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ri = RelativeIllumination(CookeTriplet(), wavelengths="all", num_fields=6)
    assert len(ri.data["relative_illumination"]) == 3
    for values, reliable in zip(
        ri.data["relative_illumination"], ri.data["reliable"], strict=True
    ):
        values = be.to_numpy(values)
        assert np.isclose(values.max(), 1.0)
        assert np.all((values > 0) & (values <= 1.0))
        assert be.to_numpy(reliable).all()


def test_caustic_is_flagged(set_test_backend):
    with pytest.warns(RuntimeWarning, match="caustic"):
        ri = RelativeIllumination(_caustic_lens(), num_fields=3, num_pupil=32)
    assert not be.to_numpy(ri.data["reliable"][0]).any()


def test_tilted_flat_image_uses_local_frame(set_test_backend):
    # Tilting the image plane changes the projected solid angle on axis by
    # the cosine-projection of the cone; the local frame captures it.
    flat = RelativeIllumination(_ideal_lens(), num_fields=3)
    optic = _ideal_lens()
    optic.surfaces[-1].geometry.cs.rx = be.array(np.radians(10.0))
    tilted = RelativeIllumination(optic, num_fields=3)
    e_flat = be.to_numpy(flat.data["irradiance"][0])[0]
    e_tilt = be.to_numpy(tilted.data["irradiance"][0])[0]
    # A small cone along the axis, seen from a plane tilted by 10 degrees.
    assert_allclose(e_tilt / e_flat, np.cos(np.radians(10.0)), rtol=1e-3)


def test_curved_image_surface_raises(set_test_backend):
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, surface_type="paraxial", f=100.0, thickness=100.0, is_stop=True
    )
    optic.surfaces.add(index=2, radius=-100.0)
    optic.set_aperture("EPD", 10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(0.55, is_primary=True)
    with pytest.raises(NotImplementedError, match="planar"):
        RelativeIllumination(optic)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"num_fields": 0}, "num_fields"), ({"num_pupil": 4}, "num_pupil")],
)
def test_invalid_sampling_raises(set_test_backend, kwargs, match):
    with pytest.raises(ValueError, match=match):
        RelativeIllumination(CookeTriplet(), **kwargs)


def test_view(set_test_backend):
    ri = RelativeIllumination(CookeTriplet(), num_fields=5, num_pupil=16)
    fig, ax = ri.view(show=False)
    assert fig is not None
    assert ax.get_title() == "Relative Illumination"
    assert len(ax.get_lines()) == 1
    matplotlib.pyplot.close(fig)


def test_view_marks_caustic_fields(set_test_backend):
    with pytest.warns(RuntimeWarning):
        ri = RelativeIllumination(_caustic_lens(), num_fields=3, num_pupil=32)
    fig, ax = ri.view(show=False)
    assert len(ax.get_lines()) == 2
    matplotlib.pyplot.close(fig)


def test_view_on_existing_figure(set_test_backend):
    ri = RelativeIllumination(CookeTriplet(), num_fields=5, num_pupil=16)
    figure = matplotlib.pyplot.figure()
    fig, _ = ri.view(fig_to_plot_on=figure, show=False)
    assert fig is figure
    matplotlib.pyplot.close(fig)
