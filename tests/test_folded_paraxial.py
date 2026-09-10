"""First-order behaviour of systems folded off the global z axis (issue #726).

The same plano-convex singlet is built four ways: straight, retro (a flat
mirror at normal incidence, a fold that global z can express), folded 90 deg by
a 45 deg mirror, and a two-fold periscope. Each folded build unfolds across its
mirrors onto the straight system, so every first-order quantity has a reference
value to be checked against.
"""

from __future__ import annotations

import math

import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.rays import RealRays

from .utils import assert_allclose, assert_array_equal


def _finish(optic):
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    return optic


def straight():
    """Plano-convex singlet, image plane 46 mm behind the flat face."""
    optic = Optic(name="straight")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=25.84, thickness=4.0, material="N-BK7", is_stop=True
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=46.0)
    optic.surfaces.add(index=3)
    return _finish(optic)


def retro():
    """Same singlet, folded back on itself by a mirror at normal incidence."""
    optic = Optic(name="retro")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=25.84, thickness=4.0, material="N-BK7", is_stop=True
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=20.0)
    optic.surfaces.add(index=3, radius=be.inf, material="mirror", thickness=-26.0)
    optic.surfaces.add(index=4)
    return _finish(optic)


def folded():
    """Same singlet, folded 90 deg into +y by a 45 deg mirror 20 mm back."""
    optic = Optic(name="folded")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=25.84, thickness=4.0, material="N-BK7", is_stop=True
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=20.0)
    optic.surfaces.add(index=3, x=0.0, y=0.0, z=24.0, rx=math.pi / 4, material="mirror")
    # Image plane 26 mm along the folded axis (equivalent unfolded z = 50).
    optic.surfaces.add(index=4, x=0.0, y=26.0, z=24.0, rx=-math.pi / 2)
    return _finish(optic)


def periscope():
    """Same singlet, offset 13 mm in +y by two 45 deg mirrors.

    The beam runs 20 mm to the first mirror, 13 mm across to the second and
    13 mm on to the image: 46 mm of post-lens propagation, as in ``straight``.
    Reflection parity is back to positive after the second mirror.
    """
    optic = Optic(name="periscope")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=25.84, thickness=4.0, material="N-BK7", is_stop=True
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=20.0)
    optic.surfaces.add(index=3, x=0.0, y=0.0, z=24.0, rx=math.pi / 4, material="mirror")
    optic.surfaces.add(
        index=4, x=0.0, y=13.0, z=24.0, rx=math.pi / 4, material="mirror"
    )
    optic.surfaces.add(index=5, x=0.0, y=13.0, z=37.0)
    return _finish(optic)


def _trace_axial_marginal(optic):
    """Trace an explicit y = +3 mm ray parallel to the axis, no pupil aiming."""
    rays = RealRays(
        be.array([0.0]),
        be.array([3.0]),
        be.array([-10.0]),
        be.array([0.0]),
        be.array([0.0]),
        be.array([1.0]),
        be.array([1.0]),
        be.array([0.55]),
    )
    optic.surfaces.trace(rays)
    return rays


class TestPositionsOnAxis:
    """Systems whose legs all run along +/-z keep reporting global z."""

    @pytest.mark.parametrize(
        ("builder", "expected"),
        [
            (straight, [-be.inf, 0.0, 4.0, 50.0]),
            (retro, [-be.inf, 0.0, 4.0, 24.0, -2.0]),
        ],
        ids=["straight", "retro"],
    )
    def test_positions_unchanged(self, builder, expected, set_test_backend):
        optic = builder()
        assert_array_equal(be.ravel(optic.surfaces.positions), be.array(expected))

    @pytest.mark.parametrize("builder", [straight, retro], ids=["straight", "retro"])
    def test_positions_are_global_z(self, builder, set_test_backend):
        optic = builder()
        assert_array_equal(optic.surfaces.positions, optic.surfaces.global_z_positions)


class TestSingleFold:
    """A 90 deg fold must read exactly like the retro system it unfolds onto."""

    def test_positions_continue_across_the_fold(self, set_test_backend):
        optic = folded()
        assert_allclose(
            be.ravel(optic.surfaces.positions),
            be.array([-be.inf, 0.0, 4.0, 24.0, -2.0]),
        )

    def test_global_z_positions_stay_in_the_global_frame(self, set_test_backend):
        optic = folded()
        assert_allclose(
            be.ravel(optic.surfaces.global_z_positions),
            be.array([-be.inf, 0.0, 4.0, 24.0, 24.0]),
        )

    def test_first_order_matches_retro(self, set_test_backend):
        fold = folded()
        reference = retro()

        assert_allclose(fold.paraxial.f2(), reference.paraxial.f2())
        assert_allclose(fold.paraxial.XPL(), reference.paraxial.XPL())
        assert_allclose(fold.paraxial.EPD(), reference.paraxial.EPD())
        assert_allclose(fold.paraxial.FNO(), reference.paraxial.FNO())

        y_fold, u_fold = fold.paraxial.marginal_ray()
        y_ref, u_ref = reference.paraxial.marginal_ray()
        assert_allclose(y_fold, y_ref)
        assert_allclose(u_fold, u_ref)

    def test_real_ray_agrees_with_the_paraxial_axis(self, set_test_backend):
        """The real marginal ray lands the same distance off the folded axis."""
        rays = _trace_axial_marginal(folded())
        # The folded axis runs along +y at z = 24, so the transverse residual
        # is the ray's z offset from the mirror vertex.
        residual = be.ravel(rays.z)[0] - 24.0

        straight_rays = _trace_axial_marginal(straight())
        assert_allclose(residual, be.ravel(straight_rays.y)[0])


class TestTwoFolds:
    """Reflection parity returns to positive after a second mirror."""

    def test_positions_reverse_then_resume(self, set_test_backend):
        optic = periscope()
        assert_allclose(
            be.ravel(optic.surfaces.positions),
            be.array([-be.inf, 0.0, 4.0, 24.0, 11.0, 24.0]),
        )

    def test_first_order_matches_straight(self, set_test_backend):
        scope = periscope()
        reference = straight()

        assert_allclose(scope.paraxial.f2(), reference.paraxial.f2())
        assert_allclose(scope.paraxial.XPL(), reference.paraxial.XPL())
        assert_allclose(scope.paraxial.FNO(), reference.paraxial.FNO())

        y_scope, _ = scope.paraxial.marginal_ray()
        y_ref, _ = reference.paraxial.marginal_ray()
        assert_allclose(be.ravel(y_scope)[-1], be.ravel(y_ref)[-1])

    def test_real_ray_agrees_with_the_paraxial_axis(self, set_test_backend):
        rays = _trace_axial_marginal(periscope())
        # After two folds the beam runs along +z again, offset 13 mm in +y.
        residual = be.ravel(rays.y)[0] - 13.0

        straight_rays = _trace_axial_marginal(straight())
        assert_allclose(residual, be.ravel(straight_rays.y)[0])


def entered_along_x():
    """Same singlet, but the whole system is entered along +x.

    A collimated source posed to fire along +x into a 45 deg mirror that turns
    the beam onto -z. Nothing about the optics changes -- the lens is 20 mm
    ahead of the mirror and the image 26 mm past it, as in ``retro`` -- so
    every first-order quantity has to come back the same. What changes is that
    not one leg of the path, not even the first, is a z interval.
    """
    optic = Optic(name="entered_along_x")
    # Object at infinity down the -x arm; a flat object plane has no axis of
    # its own to read, and the first leg says which way the light travels.
    optic.surfaces.add(index=0, x=-be.inf, y=0.0, z=0.0, radius=be.inf)
    # Lens axis along +x: local +z is the surface normal.
    optic.surfaces.add(
        index=1,
        x=0.0,
        y=0.0,
        z=0.0,
        ry=math.pi / 2,
        radius=25.84,
        material="N-BK7",
        is_stop=True,
    )
    optic.surfaces.add(index=2, x=4.0, y=0.0, z=0.0, ry=math.pi / 2, radius=be.inf)
    # 45 deg fold 20 mm on: normal bisects +x in and -z out.
    optic.surfaces.add(
        index=3, x=24.0, y=0.0, z=0.0, ry=-3 * math.pi / 4, material="mirror"
    )
    optic.surfaces.add(index=4, x=24.0, y=0.0, z=-26.0)
    return _finish(optic)


class TestOffAxisEntry:
    """A system entered off the z axis, not merely folded part way through."""

    def test_positions_run_along_the_entry_axis(self, set_test_backend):
        optic = entered_along_x()
        assert_allclose(
            be.ravel(optic.surfaces.positions),
            be.array([-be.inf, 0.0, 4.0, 24.0, -2.0]),
        )

    def test_global_z_positions_stay_in_the_global_frame(self, set_test_backend):
        """Every vertex but the image shares z = 0: the arm is invisible to z."""
        optic = entered_along_x()
        assert_allclose(
            be.ravel(optic.surfaces.global_z_positions),
            be.array([0.0, 0.0, 0.0, 0.0, -26.0]),
        )

    def test_first_order_matches_retro(self, set_test_backend):
        entered = entered_along_x()
        reference = retro()

        assert_allclose(entered.paraxial.f2(), reference.paraxial.f2())
        assert_allclose(entered.paraxial.XPL(), reference.paraxial.XPL())
        assert_allclose(entered.paraxial.EPD(), reference.paraxial.EPD())
        assert_allclose(entered.paraxial.FNO(), reference.paraxial.FNO())

        y_entered, u_entered = entered.paraxial.marginal_ray()
        y_ref, u_ref = reference.paraxial.marginal_ray()
        assert_allclose(y_entered, y_ref)
        assert_allclose(u_entered, u_ref)

    def test_object_at_infinity_off_the_z_axis_is_recognised(self, set_test_backend):
        """The infinity sits in x here, and means what it means in z."""
        assert entered_along_x().object_surface.is_infinite

    def test_real_ray_agrees_with_the_paraxial_axis(self, set_test_backend):
        """The real marginal ray lands the same distance off the folded axis."""
        rays = RealRays(
            be.array([-10.0]),
            be.array([3.0]),
            be.array([0.0]),
            be.array([1.0]),
            be.array([0.0]),
            be.array([0.0]),
            be.array([1.0]),
            be.array([0.55]),
        )
        entered_along_x().surfaces.trace(rays)
        # The sagittal offset is authored in y, which survives a fold in x-z.
        straight_rays = _trace_axial_marginal(straight())
        assert_allclose(be.ravel(rays.y)[0], be.ravel(straight_rays.y)[0])


class TestTiltedObjectIsNotAFold:
    """A tilted object plane does not steer the beam, so it is not a fold.

    The axis is read off the vertices for exactly this reason: taking it from
    the object surface's own orientation would send every system with a tilted
    object down the unfolding path and change positions that must not move.
    """

    def test_positions_stay_global_z(self, set_test_backend):
        optic = straight()
        optic.surfaces.surfaces[0].geometry.cs.rx = be.array(math.pi / 8)
        assert_array_equal(optic.surfaces.positions, optic.surfaces.global_z_positions)


class TestPublicTraceThroughFolds:
    """optic.trace() aims in the entry frame, so folded systems trace (#728)."""

    def test_entered_along_x_matches_an_explicit_ray(self, set_test_backend):
        optic = entered_along_x()
        traced = optic.trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=3, distribution="line_y"
        )
        assert bool(be.all(be.isfinite(traced.y)))
        # The trace's marginal ray, built by hand: parallel to the entry
        # axis at the pupil edge, offset in y, the sagittal direction of
        # this fold.
        rays = RealRays(
            be.array([-10.0]),
            be.array([5.0]),
            be.array([0.0]),
            be.array([1.0]),
            be.array([0.0]),
            be.array([0.0]),
            be.array([1.0]),
            be.array([0.55]),
        )
        optic.surfaces.trace(rays)
        assert_allclose(be.ravel(traced.y)[-1], be.ravel(rays.y)[0])

    def test_folded_matches_straight_through_the_public_api(self, set_test_backend):
        """The fold turns z into y, so the in-plane image coordinate is z - 24."""
        t_fold = folded().trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        t_ref = straight().trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        assert_allclose(be.ravel(t_fold.z) - 24.0, be.ravel(t_ref.y))

    def test_periscope_matches_straight_through_the_public_api(self, set_test_backend):
        """After two folds the beam runs along +z again, offset 13 mm in y."""
        t_scope = periscope().trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        t_ref = straight().trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        assert_allclose(be.ravel(t_scope.y) - 13.0, be.ravel(t_ref.y))

    def test_field_angle_entered_along_x_matches_straight(self, set_test_backend):
        """A sagittal field angle lands where the straight reference puts it.

        Field angles measure against the entry axis; y is untouched by the
        x-z fold, so the y field of the entered-along-x system must
        reproduce the straight system's y behaviour exactly.
        """
        entered = entered_along_x()
        reference = straight()
        for optic in (entered, reference):
            optic.fields.add(y=2.0)
        t_entered = entered.trace(
            Hx=0, Hy=1, wavelength=0.55, num_rays=3, distribution="line_y"
        )
        t_ref = reference.trace(
            Hx=0, Hy=1, wavelength=0.55, num_rays=3, distribution="line_y"
        )
        assert bool(be.all(be.isfinite(t_entered.y)))
        assert_allclose(be.ravel(t_entered.y), be.ravel(t_ref.y))
