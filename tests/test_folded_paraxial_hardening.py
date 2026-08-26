"""Hardening tests for folded-system paraxial optics and entry-frame aiming.

Ports the independent audit's strongest experiments (audit of PRs #727/#729)
into permanent coverage: orientation-aware powered surfaces on odd-parity
legs in both local-axis authorings, non-45-degree and out-of-plane folds,
translated and -z-entered systems, stop placement around folds, entry-frame
iterative/robust aiming, 3-D pupil points, real-space references, and the
explicit rejection of geometries outside the supported scalar domain.

Every folded assertion compares against an exact unfolding reference (the
classic z-authored retro/trombone chain), never a hard-coded number.
"""

from __future__ import annotations

import math
import warnings

import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.paraxial_path import (
    ParaxialDomainWarning,
    UnsupportedParaxialGeometryError,
)
from optiland.rays import RealRays
from optiland.rays.ray_aiming.iterative import IterativeRayAimer
from optiland.rays.ray_aiming.parameterization import LaunchParameterization
from optiland.rays.ray_aiming.robust import RobustRayAimer
from optiland.solves.thickness import MarginalRayHeightThicknessSolve

from .test_folded_paraxial import (
    _finish,
    entered_along_x,
    folded,
    retro,
    straight,
)
from .utils import assert_allclose, assert_array_equal

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def fold60():
    """Singlet folded by 60 degrees (mirror tilt pi/3), image on folded axis.

    Unfolds onto the same retro chain as the 90-degree fold: positions must
    be [-inf, 0, 4, 24, -2].
    """
    optic = Optic(name="fold60")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=25.84, thickness=4.0, material="N-BK7", is_stop=True
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=20.0)
    optic.surfaces.add(
        index=3, x=0.0, y=0.0, z=24.0, rx=math.pi / 3, material="mirror"
    )
    # Beam leaves along (0, sin60, cos60) rotated by 2*60 deg from +z:
    # d' = (0, sin(2*pi/3)? ...) -- computed: reflection of +z about the
    # mirror normal turns the beam by 120 deg here; place the image 26 mm
    # along the actual outgoing direction, oriented normal to it.
    n = (0.0, -math.sin(math.pi / 3), math.cos(math.pi / 3))
    d = (0.0, -2 * n[2] * n[1], 1 - 2 * n[2] * n[2])
    optic.surfaces.add(
        index=4,
        x=0.0,
        y=26.0 * d[1],
        z=24.0 + 26.0 * d[2],
        rx=-math.atan2(d[1], d[2]),
    )
    return _finish(optic)


def periscope_out_of_plane():
    """Out-of-plane periscope +z -> +y -> +x (audit E2).

    Unfolds onto the straight singlet's trombone positions
    [-inf, 0, 4, 24, 11, 24].
    """
    optic = Optic(name="periscope-oop")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=25.84, thickness=4.0, material="N-BK7", is_stop=True
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=20.0)
    # Mirror 1 at z=24 folds +z into +y.
    optic.surfaces.add(
        index=3, x=0.0, y=0.0, z=24.0, rx=math.pi / 4, material="mirror"
    )
    # Mirror 2 at y=13 folds +y into +x (normal (-1, 1, 0)/sqrt(2)).
    optic.surfaces.add(
        index=4,
        x=0.0,
        y=13.0,
        z=24.0,
        rx=math.pi / 2,
        rz=-3 * math.pi / 4,
        material="mirror",
    )
    # Image 13 mm along +x, normal to the beam (local +z = +x).
    optic.surfaces.add(index=5, x=13.0, y=13.0, z=24.0, ry=math.pi / 2)
    return _finish(optic)


def entered_along_neg_z():
    """The straight singlet rigidly rotated by pi about the x axis.

    The beam enters along -z; every first-order quantity must equal the
    straight reference's, and positions must be the unfolded axis
    [-inf, 0, 4, 50] -- not the descending global z.
    """
    optic = Optic(name="entered-neg-z")
    optic.surfaces.add(index=0, x=0.0, y=0.0, z=be.inf, radius=be.inf)
    optic.surfaces.add(
        index=1,
        x=0.0,
        y=0.0,
        z=0.0,
        rx=math.pi,
        radius=25.84,
        material="N-BK7",
        is_stop=True,
    )
    optic.surfaces.add(index=2, x=0.0, y=0.0, z=-4.0, rx=math.pi, radius=be.inf)
    optic.surfaces.add(index=3, x=0.0, y=0.0, z=-50.0, rx=math.pi)
    return _finish(optic)


def entered_along_x_finite():
    """Finite-conjugate variant of the +x-entered singlet.

    Reference: the same singlet on the z axis with the object at z=-60.
    """
    optic = Optic(name="entered-x-finite")
    optic.surfaces.add(index=0, x=-60.0, y=0.0, z=0.0, radius=be.inf)
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
    optic.surfaces.add(
        index=3, x=24.0, y=0.0, z=0.0, ry=-3 * math.pi / 4, material="mirror"
    )
    optic.surfaces.add(index=4, x=24.0, y=0.0, z=-26.0)
    return _finish(optic)


def straight_finite():
    """Finite-conjugate straight singlet (no fold)."""
    optic = Optic(name="straight-finite")
    optic.surfaces.add(index=0, radius=be.inf, thickness=60.0)
    optic.surfaces.add(
        index=1, radius=25.84, thickness=4.0, material="N-BK7", is_stop=True
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=46.0)
    optic.surfaces.add(index=3)
    return _finish(optic)


def retro_finite():
    """Finite-conjugate retro reference for ``entered_along_x_finite``:
    the exact unfolding of its 45-degree mirror onto the z axis."""
    optic = Optic(name="retro-finite")
    optic.surfaces.add(index=0, radius=be.inf, thickness=60.0)
    optic.surfaces.add(
        index=1, radius=25.84, thickness=4.0, material="N-BK7", is_stop=True
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=20.0)
    optic.surfaces.add(index=3, radius=be.inf, material="mirror", thickness=-26.0)
    optic.surfaces.add(index=4)
    return _finish(optic)


def folded_finite():
    """Finite-conjugate folded singlet entered along +z: 45-degree fold at
    z = 24, image on the +y arm. Exact trombone reference: ``retro_finite``."""
    optic = Optic(name="folded-finite")
    optic.surfaces.add(index=0, radius=be.inf, thickness=60.0)
    optic.surfaces.add(
        index=1, radius=25.84, thickness=4.0, material="N-BK7", is_stop=True
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=20.0)
    optic.surfaces.add(index=3, x=0.0, y=0.0, z=24.0, rx=math.pi / 4, material="mirror")
    optic.surfaces.add(index=4, x=0.0, y=26.0, z=24.0, rx=-math.pi / 2)
    return _finish(optic)


def _translate(optic, dx=0.0, dy=0.0, dz=0.0):
    """Rigidly translate every surface of an optic (absolute coordinates)."""
    for surf in optic.surfaces:
        cs = surf.geometry.cs
        for name, delta in (("x", dx), ("y", dy), ("z", dz)):
            value = getattr(cs, name)
            if bool(be.all(be.isfinite(value))):
                setattr(cs, name, value + delta)
    return optic


# Powered-surface authoring fixtures (audit E7): a concave f=40 mirror at
# normal incidence on the odd-parity leg after a 90-degree fold, plus the
# classic z-authored trombone chain it unfolds onto.


def trombone_mirror():
    optic = Optic(name="trombone-mirror")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, is_stop=True, thickness=24.0)
    optic.surfaces.add(index=2, radius=be.inf, material="mirror", thickness=-20.0)
    optic.surfaces.add(index=3, radius=80.0, material="mirror", thickness=40.0)
    optic.surfaces.add(index=4)
    return _finish(optic)


def folded_powered_mirror(authoring):
    """90-deg fold into +y, concave f=40 mirror at normal incidence on arm."""
    optic = Optic(name=f"folded-mirror-{authoring}")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, is_stop=True)
    optic.surfaces.add(index=2, x=0.0, y=0.0, z=24.0, rx=math.pi / 4, material="mirror")
    if authoring == "along":
        # Local +z along the incoming +y beam.
        optic.surfaces.add(
            index=3, x=0.0, y=20.0, z=24.0, rx=-math.pi / 2, radius=-80.0,
            material="mirror",
        )
    else:
        # Local +z against the incoming beam.
        optic.surfaces.add(
            index=3, x=0.0, y=20.0, z=24.0, rx=math.pi / 2, radius=80.0,
            material="mirror",
        )
    optic.surfaces.add(index=4, x=0.0, y=-20.0, z=24.0, rx=math.pi / 2)
    return _finish(optic)


def trombone_lens():
    """Classic z-authored retro chain with a plano-convex lens on the return
    leg (beam travels -z through it)."""
    optic = Optic(name="trombone-lens")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, is_stop=True, thickness=24.0)
    optic.surfaces.add(index=2, radius=be.inf, material="mirror", thickness=-20.0)
    optic.surfaces.add(index=3, radius=-25.84, material="N-BK7", thickness=-4.0)
    optic.surfaces.add(index=4, radius=be.inf, thickness=-22.0)
    optic.surfaces.add(index=5)
    return _finish(optic)


def folded_powered_lens(authoring):
    """90-deg fold into +y, plano-convex lens at normal incidence on the arm.

    Physically identical glass in both authorings: the front surface is the
    same sphere (center of curvature at global y = 45.84).
    """
    optic = Optic(name=f"folded-lens-{authoring}")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, is_stop=True)
    optic.surfaces.add(index=2, x=0.0, y=0.0, z=24.0, rx=math.pi / 4, material="mirror")
    if authoring == "along":
        optic.surfaces.add(
            index=3, x=0.0, y=20.0, z=24.0, rx=-math.pi / 2, radius=25.84,
            material="N-BK7",
        )
        optic.surfaces.add(
            index=4, x=0.0, y=24.0, z=24.0, rx=-math.pi / 2, radius=be.inf
        )
    else:
        optic.surfaces.add(
            index=3, x=0.0, y=20.0, z=24.0, rx=math.pi / 2, radius=-25.84,
            material="N-BK7",
        )
        optic.surfaces.add(
            index=4, x=0.0, y=24.0, z=24.0, rx=math.pi / 2, radius=be.inf
        )
    optic.surfaces.add(index=5, x=0.0, y=46.0, z=24.0, rx=math.pi / 2)
    return _finish(optic)


def oblique_powered_mirror():
    """Concave mirror used as a 45-degree fold: astigmatic, out of domain."""
    optic = Optic(name="oblique-powered")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, is_stop=True)
    optic.surfaces.add(
        index=2, x=0.0, y=0.0, z=24.0, rx=math.pi / 4, radius=-200.0,
        material="mirror",
    )
    optic.surfaces.add(index=3, x=0.0, y=100.0, z=24.0, rx=-math.pi / 2)
    return _finish(optic)


def folded_tilted_lens(tilt=0.3):
    """Lens on a folded arm, tilted relative to the local beam segment."""
    optic = Optic(name="folded-tilted-lens")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, is_stop=True)
    optic.surfaces.add(index=2, x=0.0, y=0.0, z=24.0, rx=math.pi / 4, material="mirror")
    optic.surfaces.add(
        index=3, x=0.0, y=20.0, z=24.0, rx=-math.pi / 2 + tilt, radius=25.84,
        material="N-BK7",
    )
    optic.surfaces.add(
        index=4, x=0.0, y=24.0, z=24.0, rx=-math.pi / 2 + tilt, radius=be.inf
    )
    optic.surfaces.add(index=5, x=0.0, y=46.0, z=24.0, rx=math.pi / 2)
    return _finish(optic)


def folded_steering_plate(tilt=0.3):
    """Tilted plane refractive interface on a folded arm (Snell steering)."""
    optic = Optic(name="folded-steering-plate")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, is_stop=True)
    optic.surfaces.add(index=2, x=0.0, y=0.0, z=24.0, rx=math.pi / 4, material="mirror")
    optic.surfaces.add(
        index=3, x=0.0, y=20.0, z=24.0, rx=-math.pi / 2 + tilt, radius=be.inf,
        material="N-BK7",
    )
    optic.surfaces.add(
        index=4, x=0.0, y=24.0, z=24.0, rx=-math.pi / 2, radius=be.inf
    )
    optic.surfaces.add(index=5, x=0.0, y=46.0, z=24.0, rx=math.pi / 2)
    return _finish(optic)


def stop_after_fold():
    """Singlet, 90-degree fold, stop on the folded arm."""
    optic = Optic(name="stop-after-fold")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=25.84, thickness=4.0, material="N-BK7")
    optic.surfaces.add(index=2, radius=be.inf, thickness=20.0)
    optic.surfaces.add(index=3, x=0.0, y=0.0, z=24.0, rx=math.pi / 4, material="mirror")
    optic.surfaces.add(
        index=4, x=0.0, y=10.0, z=24.0, rx=-math.pi / 2, radius=be.inf, is_stop=True
    )
    optic.surfaces.add(index=5, x=0.0, y=26.0, z=24.0, rx=-math.pi / 2)
    return _finish(optic)


def stop_after_mirror_straight():
    """Retro reference for ``stop_after_fold`` (stop at q = 14)."""
    optic = Optic(name="stop-after-mirror-straight")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=25.84, thickness=4.0, material="N-BK7")
    optic.surfaces.add(index=2, radius=be.inf, thickness=20.0)
    optic.surfaces.add(index=3, radius=be.inf, material="mirror", thickness=-10.0)
    optic.surfaces.add(index=4, radius=be.inf, is_stop=True, thickness=-16.0)
    optic.surfaces.add(index=5)
    return _finish(optic)


def stop_between_folds():
    """Periscope with the stop on the middle leg."""
    optic = Optic(name="stop-between-folds")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=25.84, thickness=4.0, material="N-BK7")
    optic.surfaces.add(index=2, radius=be.inf, thickness=20.0)
    optic.surfaces.add(index=3, x=0.0, y=0.0, z=24.0, rx=math.pi / 4, material="mirror")
    optic.surfaces.add(
        index=4, x=0.0, y=6.0, z=24.0, rx=-math.pi / 2, radius=be.inf, is_stop=True
    )
    optic.surfaces.add(
        index=5, x=0.0, y=13.0, z=24.0, rx=math.pi / 4, material="mirror"
    )
    optic.surfaces.add(index=6, x=0.0, y=13.0, z=37.0)
    return _finish(optic)


def stop_mid_straight():
    """Straight reference for ``stop_between_folds`` (stop at z = 30)."""
    optic = Optic(name="stop-mid-straight")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=25.84, thickness=4.0, material="N-BK7")
    optic.surfaces.add(index=2, radius=be.inf, thickness=26.0)
    optic.surfaces.add(index=3, radius=be.inf, is_stop=True, thickness=20.0)
    optic.surfaces.add(index=4)
    return _finish(optic)


# ---------------------------------------------------------------------------
# 16.1 Path and sign tests
# ---------------------------------------------------------------------------


class TestPathAndSigns:
    def test_60_degree_fold_matches_retro_positions(self, set_test_backend):
        assert_allclose(
            be.ravel(fold60().surfaces.positions),
            be.array([-be.inf, 0.0, 4.0, 24.0, -2.0]),
        )

    def test_60_degree_fold_first_order_matches_retro(self, set_test_backend):
        optic, reference = fold60(), retro()
        assert_allclose(optic.paraxial.f2(), reference.paraxial.f2())
        assert_allclose(optic.paraxial.EPL(), reference.paraxial.EPL())
        assert_allclose(optic.paraxial.XPL(), reference.paraxial.XPL())

    def test_out_of_plane_periscope_matches_trombone(self, set_test_backend):
        optic = periscope_out_of_plane()
        assert_allclose(
            be.ravel(optic.surfaces.positions),
            be.array([-be.inf, 0.0, 4.0, 24.0, 11.0, 24.0]),
        )
        reference = straight()
        assert_allclose(optic.paraxial.f2(), reference.paraxial.f2())
        assert_allclose(optic.paraxial.XPL(), reference.paraxial.XPL())

    def test_negative_leg_stays_negative(self, set_test_backend):
        """A surface authored behind its predecessor on an off-axis entry
        keeps its negative spacing via the signed projection."""
        optic = Optic(name="negative-leg")
        optic.surfaces.add(index=0, x=-be.inf, y=0.0, z=0.0, radius=be.inf)
        optic.surfaces.add(
            index=1, x=0.0, y=0.0, z=0.0, ry=math.pi / 2, radius=be.inf,
            is_stop=True,
        )
        optic.surfaces.add(
            index=2, x=-3.0, y=0.0, z=0.0, ry=math.pi / 2, radius=be.inf
        )
        optic.surfaces.add(index=3, x=21.0, y=0.0, z=0.0, ry=math.pi / 2)
        _finish(optic)
        assert_allclose(
            be.ravel(optic.surfaces.positions),
            be.array([-be.inf, 0.0, -3.0, 21.0]),
        )

    def test_translated_system_keeps_global_z_positions(self, set_test_backend):
        optic = _translate(straight(), dx=5.0, dy=3.0)
        assert_array_equal(
            optic.surfaces.positions, optic.surfaces.global_z_positions
        )

    def test_translated_system_gets_an_entry_frame(self, set_test_backend):
        """A laterally translated system must not route through the
        global-origin aiming branch."""
        optic = _translate(straight(), dx=5.0, dy=3.0)
        frame = optic.surfaces._entry_frame()
        assert frame is not None
        anchor, _axial, d0, _u, _v = frame
        assert_allclose(anchor[0], 5.0)
        assert_allclose(anchor[1], 3.0)
        assert_allclose(d0[2], 1.0)

    def test_neg_z_entry_positions_are_the_unfolded_axis(self, set_test_backend):
        optic = entered_along_neg_z()
        assert_allclose(
            be.ravel(optic.surfaces.positions),
            be.array([-be.inf, 0.0, 4.0, 50.0]),
        )

    def test_neg_z_entry_first_order_matches_straight(self, set_test_backend):
        optic, reference = entered_along_neg_z(), straight()
        assert_allclose(optic.paraxial.f2(), reference.paraxial.f2())
        assert_allclose(optic.paraxial.EPL(), reference.paraxial.EPL())
        assert_allclose(optic.paraxial.XPL(), reference.paraxial.XPL())
        y_neg, u_neg = optic.paraxial.marginal_ray()
        y_ref, u_ref = reference.paraxial.marginal_ray()
        assert_allclose(y_neg, y_ref)
        assert_allclose(u_neg, u_ref)

    def test_neg_z_entry_is_not_legacy_aiming_compatible(self, set_test_backend):
        path = entered_along_neg_z().surfaces.build_paraxial_path()
        assert not path.legacy_aiming_compatible
        assert path.is_folded_or_off_axis
        assert path.all_legs_parallel_global_z

    def test_virtual_object_on_z_axis_stays_legacy(self, set_test_backend):
        """A finite object authored past the first surface (virtual object)
        must not be misread as a -z-entered system."""
        optic = Optic(name="virtual-object")
        optic.surfaces.add(index=0, radius=be.inf, thickness=-10.0)  # z = +10
        optic.surfaces.add(
            index=1, radius=25.84, thickness=4.0, material="N-BK7", is_stop=True
        )
        optic.surfaces.add(index=2, radius=be.inf, thickness=46.0)
        optic.surfaces.add(index=3)
        _finish(optic)
        path = optic.surfaces.build_paraxial_path()
        assert path.positions_are_global_z
        assert_array_equal(
            optic.surfaces.positions, optic.surfaces.global_z_positions
        )

    def test_tolerance_gate_is_continuous(self, set_test_backend):
        """Sweeping a mirror tilt across the fold predicate must not jump."""
        reference = be.to_numpy(be.ravel(retro().surfaces.positions))[1:]
        for tilt in (0.0, 1e-12, 3e-11, 1e-9, 1e-6):
            optic = retro()
            optic.surfaces.surfaces[3].geometry.cs.rx = be.array(tilt)
            positions = be.to_numpy(be.ravel(optic.surfaces.positions))[1:]
            deviation = abs(positions - reference).max()
            assert deviation <= max(1e-6, 100.0 * tilt), (
                f"tilt {tilt}: deviation {deviation}"
            )

    def test_interior_infinity_on_folded_arm_raises(self, set_test_backend):
        optic = Optic(name="folded-afocal")
        optic.surfaces.add(index=0, x=-be.inf, y=0.0, z=0.0, radius=be.inf)
        optic.surfaces.add(
            index=1, x=0.0, y=0.0, z=0.0, ry=math.pi / 2, radius=25.84,
            material="N-BK7", is_stop=True,
        )
        optic.surfaces.add(
            index=2, x=4.0, y=0.0, z=0.0, ry=math.pi / 2, radius=be.inf
        )
        optic.surfaces.add(index=3, x=be.inf, y=0.0, z=0.0, ry=math.pi / 2)
        _finish(optic)
        with pytest.raises(UnsupportedParaxialGeometryError, match="NONOBJECT"):
            optic.paraxial.f2()


# ---------------------------------------------------------------------------
# 16.2 Powered surfaces on odd-parity legs, both authorings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [folded_powered_mirror, folded_powered_lens])
class TestPoweredSurfaceAuthoring:
    @staticmethod
    def _reference(builder):
        return trombone_mirror() if builder is folded_powered_mirror else (
            trombone_lens()
        )

    def test_first_order_matches_reference_in_both_authorings(
        self, set_test_backend, builder
    ):
        reference = self._reference(builder)
        f2_ref = reference.paraxial.f2()
        epl_ref = reference.paraxial.EPL()
        xpl_ref = reference.paraxial.XPL()
        y_ref, u_ref = reference.paraxial.marginal_ray()
        yc_ref, uc_ref = reference.paraxial.chief_ray()
        for authoring in ("along", "against"):
            optic = builder(authoring)
            assert_allclose(optic.paraxial.f2(), f2_ref)
            assert_allclose(optic.paraxial.EPL(), epl_ref)
            assert_allclose(optic.paraxial.XPL(), xpl_ref)
            y, u = optic.paraxial.marginal_ray()
            assert_allclose(y, y_ref)
            assert_allclose(u, u_ref)
            yc, uc = optic.paraxial.chief_ray()
            assert_allclose(yc, yc_ref)
            assert_allclose(uc, uc_ref)

    def test_f1_matches_in_both_authorings(self, set_test_backend, builder):
        f1_ref = self._reference(builder).paraxial.f1()
        for authoring in ("along", "against"):
            assert_allclose(builder(authoring).paraxial.f1(), f1_ref)

    def test_real_rays_equivalent_between_authorings(
        self, set_test_backend, builder
    ):
        results = []
        for authoring in ("along", "against"):
            rays = RealRays(
                be.array([0.0]),
                be.array([2.0]),
                be.array([-10.0]),
                be.array([0.0]),
                be.array([0.0]),
                be.array([1.0]),
                be.array([1.0]),
                be.array([0.55]),
            )
            builder(authoring).surfaces.trace(rays)
            results.append(
                (be.copy(rays.x), be.copy(rays.y), be.copy(rays.z))
            )
        for a, b in zip(results[0], results[1], strict=True):
            assert_allclose(a, b, atol=1e-9)

    def test_effective_radii_differ_by_the_orientation_sign(
        self, set_test_backend, builder
    ):
        signs = {}
        radii = {}
        powered_index = 3
        for authoring in ("along", "against"):
            optic = builder(authoring)
            path = optic.surfaces.build_paraxial_path()
            signs[authoring] = float(path.orientation_sign[powered_index])
            radii[authoring] = float(
                be.to_numpy(optic.surfaces.radii[powered_index])
            )
        # Opposite authorings carry opposite authored radii and opposite
        # orientation signs, so the paraxial-effective radius is identical.
        assert signs["along"] == -signs["against"]
        assert_allclose(radii["along"], -radii["against"])
        assert_allclose(
            signs["along"] * radii["along"], signs["against"] * radii["against"]
        )


# ---------------------------------------------------------------------------
# 16.3 Unsupported-power rejection
# ---------------------------------------------------------------------------


class TestUnsupportedGeometryRejection:
    def test_oblique_powered_mirror_raises(self, set_test_backend):
        optic = oblique_powered_mirror()
        with pytest.raises(
            UnsupportedParaxialGeometryError, match="OBLIQUE_POWERED_MIRROR"
        ) as excinfo:
            optic.paraxial.f2()
        # The error names the surface and carries the measured axis-beam dot.
        assert "surface 2" in str(excinfo.value)
        assert "measured" in str(excinfo.value)

    def test_tilted_powered_refractive_surface_raises(self, set_test_backend):
        with pytest.raises(
            UnsupportedParaxialGeometryError, match="TILTED_REFRACTIVE_SURFACE"
        ):
            folded_tilted_lens().paraxial.f2()

    def test_tilted_plane_steering_interface_raises(self, set_test_backend):
        with pytest.raises(
            UnsupportedParaxialGeometryError, match="TILTED_REFRACTIVE_SURFACE"
        ):
            folded_steering_plate().paraxial.f2()

    def test_real_ray_tracing_remains_available(self, set_test_backend):
        """Rejected scalar-paraxial geometries still trace real rays."""
        optic = oblique_powered_mirror()
        rays = RealRays(
            be.array([0.0]),
            be.array([1.0]),
            be.array([-10.0]),
            be.array([0.0]),
            be.array([0.0]),
            be.array([1.0]),
            be.array([1.0]),
            be.array([0.55]),
        )
        optic.surfaces.trace(rays)
        assert bool(be.all(be.isfinite(rays.y)))

    def test_seed_scope_downgrades_rejection_to_warning(self, set_test_backend):
        """Inside the aimers' seed scope, out-of-domain scalar paraxial use
        downgrades to an explicit warning: the aimed result is real-ray
        verified, so the approximation never surfaces as a first-order
        result."""
        from optiland.paraxial_path import paraxial_seed_scope

        optic = oblique_powered_mirror()
        with paraxial_seed_scope(), pytest.warns(ParaxialDomainWarning):
            optic.paraxial.f2()
        # And end-to-end: real tracing through the aiming pipeline works.
        rays = optic.trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=3, distribution="line_y"
        )
        assert bool(be.all(be.isfinite(rays.y)))


# ---------------------------------------------------------------------------
# 16.4 Pupils and paraxial aiming through folds
# ---------------------------------------------------------------------------


class TestFoldedPupilAiming:
    def test_stop_after_fold_matches_retro_reference(self, set_test_backend):
        optic, reference = stop_after_fold(), stop_after_mirror_straight()
        assert_allclose(optic.paraxial.EPL(), reference.paraxial.EPL())
        assert_allclose(optic.paraxial.EPD(), reference.paraxial.EPD())
        t_fold = optic.trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        t_ref = reference.trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        # The fold turns z into y: in-plane image coordinate is z - 24.
        assert_allclose(be.ravel(t_fold.z) - 24.0, be.ravel(t_ref.y), atol=1e-9)

    def test_stop_between_folds_matches_straight_reference(
        self, set_test_backend
    ):
        optic, reference = stop_between_folds(), stop_mid_straight()
        assert_allclose(optic.paraxial.EPL(), reference.paraxial.EPL())
        assert_allclose(optic.paraxial.EPD(), reference.paraxial.EPD())
        t_fold = optic.trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        t_ref = reference.trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        assert_allclose(be.ravel(t_fold.y) - 13.0, be.ravel(t_ref.y), atol=1e-9)

    def test_full_2d_pupil_through_fold(self, set_test_backend):
        """A full hexapolar pupil (not only line_y) matches the straight
        reference: the fold maps (x, y, z) -> (x, z-24 flips), so radial
        distances from the image center must be preserved."""
        t_fold = folded().trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=17, distribution="hexapolar"
        )
        t_ref = straight().trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=17, distribution="hexapolar"
        )
        r_fold = be.sqrt(t_fold.x**2 + (t_fold.z - 24.0) ** 2)
        r_ref = be.sqrt(t_ref.x**2 + t_ref.y**2)
        assert_allclose(r_fold, r_ref, atol=1e-9)

    def test_finite_conjugate_entered_along_x(self, set_test_backend):
        optic, reference = entered_along_x_finite(), retro_finite()
        assert_allclose(optic.paraxial.f2(), reference.paraxial.f2())
        assert_allclose(optic.paraxial.EPD(), reference.paraxial.EPD())
        assert_allclose(
            optic.paraxial.magnification(), reference.paraxial.magnification()
        )
        t_x = optic.trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        t_ref = reference.trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        assert bool(be.all(be.isfinite(t_x.y)))
        # y is the sagittal direction of the x-z fold: untouched.
        assert_allclose(be.ravel(t_x.y), be.ravel(t_ref.y), atol=1e-9)

    def test_infinite_conjugate_entered_along_x(self, set_test_backend):
        t_x = entered_along_x().trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        t_ref = retro().trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        assert_allclose(be.ravel(t_x.y), be.ravel(t_ref.y), atol=1e-9)

    def test_entry_along_neg_z_traces_like_straight(self, set_test_backend):
        t_neg = entered_along_neg_z().trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        t_ref = straight().trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        assert bool(be.all(be.isfinite(t_neg.y)))
        # The pi rotation about x maps y -> -y; magnitudes are preserved.
        assert_allclose(
            be.abs(be.ravel(t_neg.y)), be.abs(be.ravel(t_ref.y)), atol=1e-9
        )

    def test_translated_system_aims_about_the_translated_line(
        self, set_test_backend
    ):
        optic = _translate(straight(), dx=5.0, dy=3.0)
        reference = straight()
        t_shift = optic.trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        t_ref = reference.trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=5, distribution="line_y"
        )
        assert bool(be.all(be.isfinite(t_shift.y)))
        assert_allclose(be.ravel(t_shift.x) - 5.0, be.ravel(t_ref.x), atol=1e-9)
        assert_allclose(be.ravel(t_shift.y) - 3.0, be.ravel(t_ref.y), atol=1e-9)

    def test_vignetting_through_fold_matches_straight(self, set_test_backend):
        optic, reference = folded(), straight()
        for target in (optic, reference):
            target.fields.fields[0].vx = 0.3
            target.fields.fields[0].vy = 0.5
        t_fold = optic.trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=7, distribution="hexapolar"
        )
        t_ref = reference.trace(
            Hx=0, Hy=0, wavelength=0.55, num_rays=7, distribution="hexapolar"
        )
        r_fold = be.sqrt(t_fold.x**2 + (t_fold.z - 24.0) ** 2)
        r_ref = be.sqrt(t_ref.x**2 + t_ref.y**2)
        assert_allclose(r_fold, r_ref, atol=1e-9)

    def test_two_dimensional_field_through_fold(self, set_test_backend):
        """Nonzero Hx and Hy (inside the unambiguous domain) through the
        +x-entered system: image radii match the straight reference."""
        optic, reference = entered_along_x(), retro()
        for target in (optic, reference):
            target.fields.add(x=1.5, y=1.5)
        t_x = optic.trace(
            Hx=1, Hy=1, wavelength=0.55, num_rays=5, distribution="hexapolar"
        )
        t_ref = reference.trace(
            Hx=1, Hy=1, wavelength=0.55, num_rays=5, distribution="hexapolar"
        )
        # Image plane of the +x system is normal to -z at (24, 0, -26); its
        # transverse coordinates are (x - 24, y). The retro image plane is
        # normal to -z at z = -2 with transverse (x, y).
        r_x = be.sqrt((t_x.x - 24.0) ** 2 + t_x.y**2)
        r_ref = be.sqrt(t_ref.x**2 + t_ref.y**2)
        assert_allclose(
            be.sort(be.ravel(r_x)), be.sort(be.ravel(r_ref)), atol=1e-9
        )


# ---------------------------------------------------------------------------
# 16.5 Iterative and robust aiming off the z axis
# ---------------------------------------------------------------------------


class TestOffAxisIterativeAiming:
    def test_real_reference_strategy_no_nan_warning(self, set_test_backend):
        """The frame-aware RealReferenceStrategy must complete without the
        historical NaN warning/fallback for off-z entry."""
        from optiland.rays.ray_aiming.initialization import RealReferenceStrategy

        optic = entered_along_x()
        strategy = RealReferenceStrategy(optic)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            radius = strategy.calculate_stop_radius()
        assert radius > 0

    def test_iterative_aiming_converges_off_axis(self, set_test_backend):
        optic = entered_along_x()
        aimer = IterativeRayAimer(optic, tol=1e-10)
        x, y, z, L, M, N = aimer.aim_rays(
            (be.array([0.0]), be.array([0.0])),
            be.array([0.55]),
            (be.array([0.7]), be.array([0.6])),
        )
        report = aimer.last_report
        assert report is not None
        assert report.converged
        assert report.final_residual <= max(10 * aimer.tol, 1e-9)
        assert report.final_residual <= report.seed_residual + 1e-15
        assert report.iterations >= 0
        assert bool(be.all(be.isfinite(x)))

    def test_jacobian_full_rank_off_axis(self, set_test_backend):
        """Both solver parameters change independent transverse coordinates:
        the finite-difference Jacobian must have full rank for +x entry."""
        optic = entered_along_x()
        aimer = IterativeRayAimer(optic)
        param = LaunchParameterization.for_optic(optic, True)
        seed = aimer._paraxial_aimer.aim_rays(
            (be.array([0.0]), be.array([0.0])),
            be.array([0.55]),
            (be.array([0.0]), be.array([0.0])),
        )
        bound = param.bind(*seed)
        xi = be.zeros(1)
        eta = be.zeros(1)
        launch = bound.launch(xi, eta)
        rays = aimer._trace_subset(
            *launch, be.array([0.55]), optic.surfaces.stop_index, True
        )
        lx, ly = aimer._get_local_stop_coords(rays, optic.surfaces.stop_index)
        J11, J12, J21, J22 = aimer._finite_difference_jacobian(
            bound, xi, eta, be.array([0.55]), optic.surfaces.stop_index,
            True, lx, ly,
        )
        j11 = float(be.to_numpy(J11).reshape(-1)[0])
        j12 = float(be.to_numpy(J12).reshape(-1)[0])
        j21 = float(be.to_numpy(J21).reshape(-1)[0])
        j22 = float(be.to_numpy(J22).reshape(-1)[0])
        det = j11 * j22 - j12 * j21
        norm = max(abs(j11), abs(j12), abs(j21), abs(j22))
        assert abs(det) > 1e-6 * norm**2
        # Each column must be nonzero: both parameters move the stop point.
        assert (j11**2 + j21**2) > 1e-12
        assert (j12**2 + j22**2) > 1e-12

    def test_finite_conjugate_directions_stay_normalized(self, set_test_backend):
        optic = entered_along_x_finite()
        aimer = IterativeRayAimer(optic, tol=1e-10)
        _, _, _, L, M, N = aimer.aim_rays(
            (be.array([0.0]), be.array([0.0])),
            be.array([0.55]),
            (be.array([0.8]), be.array([0.5])),
        )
        norm = be.sqrt(L**2 + M**2 + N**2)
        assert_allclose(norm, be.ones_like(norm), atol=1e-12)

    def test_robust_and_iterative_agree_off_axis(self, set_test_backend):
        results = []
        for aimer_cls in (IterativeRayAimer, RobustRayAimer):
            optic = entered_along_x()
            aimer = aimer_cls(optic, tol=1e-10)
            out = aimer.aim_rays(
                (be.array([0.0]), be.array([0.0])),
                be.array([0.55]),
                (be.array([0.5]), be.array([0.5])),
            )
            results.append(out)
        for a, b in zip(results[0], results[1], strict=True):
            assert_allclose(a, b, atol=1e-8)

    def test_no_false_convergence_report(self, set_test_backend):
        """A short iteration budget must report non-convergence, not a
        finite ray dressed up as success."""
        optic = stop_after_fold()
        aimer = IterativeRayAimer(optic, max_iter=0, tol=0.0)
        with pytest.raises(ValueError, match="failed to converge"):
            aimer.aim_rays(
                (be.array([0.0]), be.array([0.0])),
                be.array([0.55]),
                (be.array([0.9]), be.array([0.0])),
            )
        assert aimer.last_report is not None
        assert not aimer.last_report.converged

    def test_warm_start_survives_rigid_translation(self, set_test_backend):
        """After a rigid lateral translation the cached pupil map is stale;
        recalibration must produce valid rays about the translated line."""
        optic = straight()
        aimer = RobustRayAimer(optic)
        aimer.aim_rays(
            (be.array([0.0]), be.array([0.0])),
            be.array([0.55]),
            (be.array([0.0, 0.5]), be.array([0.0, 0.5])),
        )
        _translate(optic, dx=5.0, dy=3.0)
        x, y, z, L, M, N = aimer.aim_rays(
            (be.array([0.0]), be.array([0.0])),
            be.array([0.55]),
            (be.array([0.0, 0.5]), be.array([0.0, 0.5])),
        )
        assert bool(be.all(be.isfinite(x)))
        # The chief ray must launch on the translated entry line.
        assert_allclose(
            float(be.to_numpy(x).reshape(-1)[0]), 5.0, atol=1e-6
        )
        assert_allclose(
            float(be.to_numpy(y).reshape(-1)[0]), 3.0, atol=1e-6
        )


# ---------------------------------------------------------------------------
# 16.6 Real-space analyses
# ---------------------------------------------------------------------------


class TestRealSpaceReferences:
    def test_entrance_pupil_point_on_the_entry_line(self, set_test_backend):
        optic = entered_along_x()
        epl = float(optic.paraxial.EPL())
        point = optic.paraxial.entrance_pupil_point_gcs()
        values = [float(be.to_numpy(be.array(c)).reshape(-1)[0]) for c in point]
        assert_allclose(values, [epl, 0.0, 0.0], atol=1e-12)

    def test_exit_pupil_point_straight_retro_folded(self, set_test_backend):
        for builder, expected in (
            (straight, lambda xpl: (0.0, 0.0, 50.0 + xpl)),
            (retro, lambda xpl: (0.0, 0.0, -2.0 + xpl)),
            (folded, lambda xpl: (0.0, 26.0 - xpl, 24.0)),
        ):
            optic = builder()
            xpl = float(optic.paraxial.XPL())
            point = optic.paraxial.exit_pupil_point_gcs()
            values = [
                float(be.to_numpy(be.array(c)).reshape(-1)[0]) for c in point
            ]
            assert_allclose(values, list(expected(xpl)), atol=1e-9)

    def test_wavefront_reference_radius_invariant_under_folding(
        self, set_test_backend
    ):
        from optiland.wavefront import Wavefront

        wf_fold = Wavefront(
            folded(), fields=[(0, 0)], wavelengths=[0.55], num_rays=8
        )
        wf_ref = Wavefront(
            straight(), fields=[(0, 0)], wavelengths=[0.55], num_rays=8
        )
        data_fold = wf_fold.get_data((0, 0), 0.55)
        data_ref = wf_ref.get_data((0, 0), 0.55)
        assert_allclose(data_fold.radius, data_ref.radius, atol=1e-8)
        assert_allclose(data_fold.opd, data_ref.opd, atol=1e-8)

    def test_huygens_normalization_uses_the_image_vertex(self, set_test_backend):
        """A system rigidly translated in x/y must keep its Strehl-normalized
        PSF peak: the ideal normalization point follows the image vertex."""
        from optiland.psf import HuygensPSF

        def build(translate):
            optic = straight_finite()
            if translate:
                _translate(optic, dx=4.0, dy=-2.5)
            return optic

        psf_ref = HuygensPSF(
            build(False), field=(0, 0), wavelength=0.55, num_rays=32,
            image_size=8,
        )
        psf_shift = HuygensPSF(
            build(True), field=(0, 0), wavelength=0.55, num_rays=32,
            image_size=8,
        )
        # The ideal normalization is the direct target of the fix: before
        # it, the shifted system normalized against (0, 0, global_z), a
        # point millimetres off the pupil sphere's axis, and the factor was
        # wrong by orders of magnitude.
        norm_ref = float(be.to_numpy(psf_ref._get_normalization()))
        norm_shift = float(be.to_numpy(psf_shift._get_normalization()))
        assert_allclose(norm_shift, norm_ref, rtol=1e-3)
        # Peak comparison stays loose: a pupil-edge ray can flip in/out of
        # the valid set under a rigid translation at machine precision.
        peak_ref = float(be.to_numpy(be.max(psf_ref.psf)))
        peak_shift = float(be.to_numpy(be.max(psf_shift.psf)))
        assert_allclose(peak_shift, peak_ref, rtol=5e-2)


# ---------------------------------------------------------------------------
# 16.7 Gated operations
# ---------------------------------------------------------------------------


class TestGatedOperations:
    def test_folded_thickness_solve_raises_before_mutation(
        self, set_test_backend
    ):
        optic = folded()
        z_before = [
            float(be.to_numpy(s.geometry.cs.z)) for s in optic.surfaces
        ]
        solve = MarginalRayHeightThicknessSolve(optic, 2, 1.0)
        with pytest.raises(UnsupportedParaxialGeometryError):
            solve.apply()
        z_after = [float(be.to_numpy(s.geometry.cs.z)) for s in optic.surfaces]
        assert z_before == z_after

    def test_folded_image_solve_raises_before_mutation(self, set_test_backend):
        optic = folded()
        z_before = [
            float(be.to_numpy(s.geometry.cs.z)) for s in optic.surfaces
        ]
        with pytest.raises(UnsupportedParaxialGeometryError):
            optic.updater.image_solve()
        z_after = [float(be.to_numpy(s.geometry.cs.z)) for s in optic.surfaces]
        assert z_before == z_after

    def test_folded_set_thickness_raises(self, set_test_backend):
        optic = folded()
        with pytest.raises(UnsupportedParaxialGeometryError):
            optic.updater.set_thickness(10.0, 1)

    def test_folded_through_focus_raises_before_mutation(self, set_test_backend):
        """Through-focus analysis steps the image plane along global z only,
        so it is gated on folded paths before any geometry is touched."""
        from optiland.analysis import ThroughFocusSpotDiagram

        optic = folded()
        z_before = float(be.to_numpy(optic.image_surface.geometry.cs.z))
        with pytest.raises(UnsupportedParaxialGeometryError):
            ThroughFocusSpotDiagram(optic, num_steps=1)
        z_after = float(be.to_numpy(optic.image_surface.geometry.cs.z))
        assert z_before == z_after

    def test_translated_straight_through_focus_still_works(
        self, set_test_backend
    ):
        """A laterally translated straight system keeps positions == global
        z, so through-focus analysis remains supported."""
        from optiland.analysis import ThroughFocusSpotDiagram

        optic = _translate(straight_finite(), dx=5.0)
        analysis = ThroughFocusSpotDiagram(optic, num_steps=1)
        assert len(analysis.results) == 1

    def test_neg_z_entry_z_solves_are_gated(self, set_test_backend):
        """-z entry has positions != global z, so z-offset solves must be
        rejected too (the offset would carry the wrong sign)."""
        optic = entered_along_neg_z()
        with pytest.raises(UnsupportedParaxialGeometryError):
            optic.updater.image_solve()

    def test_translated_straight_system_solves_still_work(
        self, set_test_backend
    ):
        """A laterally translated straight system keeps positions == global
        z, so z solves remain supported."""
        optic = _translate(straight_finite(), dx=5.0)
        optic.updater.image_solve()  # must not raise

    @pytest.mark.parametrize(
        "field_type",
        ["object_height", "paraxial_image_height", "real_image_height"],
    )
    def test_folded_z_bound_field_types_raise(self, set_test_backend, field_type):
        optic = entered_along_x_finite()
        optic.fields.set_type(field_type)
        with pytest.raises(UnsupportedParaxialGeometryError) as excinfo:
            optic.trace(
                Hx=0, Hy=0, wavelength=0.55, num_rays=3, distribution="line_y"
            )
        assert "Field" in str(excinfo.value) or "field" in str(excinfo.value)

    @pytest.mark.parametrize("Hy", [0.0, 1.0])
    def test_object_height_supported_on_z_entered_folded_path(
        self, set_test_backend, Hy
    ):
        """Object height lives on the +z entry leg, so a fold downstream
        must not reject it (issue #329's OAP collimator uses exactly this
        combination), and the trace must match the exact trombone reference,
        nonzero heights included."""
        traced = []
        for build in (folded_finite, retro_finite):
            optic = build()
            optic.fields.set_type("object_height")
            optic.fields.add(y=2.0)
            optic.trace(
                Hx=0, Hy=Hy, wavelength=0.55, num_rays=5, distribution="line_y"
            )
            traced.append(optic.surfaces)
        folded_sg, retro_sg = traced
        # Launch points sit at the authored object height on both.
        assert_allclose(folded_sg.y[0], retro_sg.y[0], atol=1e-12)
        # Shared straight leg: identical heights at the lens.
        assert_allclose(folded_sg.y[1], retro_sg.y[1], atol=1e-9)
        # The fold turns z into y: in-plane image coordinate is z - 24.
        assert_allclose(
            be.ravel(folded_sg.z[-1]) - 24.0,
            be.ravel(retro_sg.y[-1]),
            atol=1e-9,
        )

    @pytest.mark.parametrize(
        "field_type", ["paraxial_image_height", "real_image_height"]
    )
    def test_image_height_fields_still_raise_on_z_entered_folds(
        self, set_test_backend, field_type
    ):
        """Image-height coordinates live on the folded arm; a +z entry does
        not make them meaningful there."""
        optic = folded_finite()
        optic.fields.set_type(field_type)
        with pytest.raises(UnsupportedParaxialGeometryError):
            optic.trace(
                Hx=0, Hy=0, wavelength=0.55, num_rays=3, distribution="line_y"
            )

    def test_ambiguous_wide_angle_field_raises(self, set_test_backend):
        optic = straight()
        optic.fields.add(x=70.0, y=70.0)
        Hx, Hy = optic.fields.get_field_coords()[-1]
        with pytest.raises(
            UnsupportedParaxialGeometryError, match="AMBIGUOUS_WIDE_ANGLE_FIELD"
        ) as excinfo:
            optic.trace(
                Hx=Hx, Hy=Hy, wavelength=0.55, num_rays=3, distribution="line_y"
            )
        # The error reports the actual field components.
        assert "70" in str(excinfo.value)

    def test_one_dimensional_wide_angle_field_still_works(self, set_test_backend):
        """The ambiguity gate must not reject the supported 1-D wide-angle
        domain."""
        optic = straight()
        optic.fields.add(x=0.0, y=95.0)
        rays = optic.trace(
            Hx=0, Hy=1, wavelength=0.55, num_rays=3, distribution="line_y"
        )
        assert rays is not None


class TestStraightSystemAdvisories:
    """Tilted/decentered surfaces on straight +z systems: the scalar
    first-order numbers stay the historical ones (tilt ignored), but the
    approximation is now surfaced as a ParaxialDomainWarning instead of
    staying silent."""

    @staticmethod
    def _tilted_lens():
        optic = Optic(name="straight_tilted_lens")
        optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        optic.surfaces.add(
            index=1,
            radius=25.84,
            thickness=4.0,
            material="N-BK7",
            is_stop=True,
            rx=0.05,
        )
        optic.surfaces.add(index=2, radius=be.inf, thickness=46.0)
        optic.surfaces.add(index=3)
        return _finish(optic)

    @staticmethod
    def _decentered_lens():
        optic = Optic(name="straight_decentered_lens")
        optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        optic.surfaces.add(
            index=1,
            radius=25.84,
            thickness=4.0,
            material="N-BK7",
            is_stop=True,
            dy=1.5,
        )
        optic.surfaces.add(index=2, radius=be.inf, thickness=46.0)
        optic.surfaces.add(index=3)
        return _finish(optic)

    def test_tilted_lens_warns_and_keeps_historical_values(
        self, set_test_backend
    ):
        optic = self._tilted_lens()
        with pytest.warns(ParaxialDomainWarning, match="TILTED_REFRACTIVE"):
            f2_tilted = optic.paraxial.f2()
        # The numbers themselves are unchanged: same result as untilted.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ParaxialDomainWarning)
            f2_ref = straight().paraxial.f2()
        assert_allclose(f2_tilted, f2_ref)

    def test_decentered_lens_warns(self, set_test_backend):
        optic = self._decentered_lens()
        path = optic.surfaces.build_paraxial_path()
        # The decenter shows up as an advisory, never as a gating
        # diagnostic: the system stays fully supported.
        assert path.positions_are_global_z
        assert not path.diagnostics
        assert path.advisories
        with pytest.warns(ParaxialDomainWarning, match="NONCOLLINEAR"):
            optic.paraxial.f2()

    def test_plain_straight_system_stays_silent(self, set_test_backend):
        optic = straight()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ParaxialDomainWarning)
            optic.paraxial.f2()
            optic.paraxial.chief_ray()

    def test_advisories_never_gate(self, set_test_backend):
        """Advisory systems keep working end to end (trace, aim), unlike
        diagnostic (folded out-of-domain) systems."""
        optic = self._tilted_lens()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ParaxialDomainWarning)
            rays = optic.trace(
                Hx=0, Hy=0, wavelength=0.55, num_rays=3, distribution="line_y"
            )
        assert rays is not None


def test_domain_exceptions_are_top_level_exports():
    """Users must be able to catch the domain error without knowing the
    internal module layout."""
    import optiland

    assert optiland.UnsupportedParaxialGeometryError is (
        UnsupportedParaxialGeometryError
    )
    assert optiland.ParaxialDomainWarning is ParaxialDomainWarning


def test_entrance_pupil_z_is_deprecated_alias(set_test_backend):
    """The legacy name still returns the identical value but points callers
    to entrance_pupil_axial_position."""
    optic = straight()
    with pytest.warns(DeprecationWarning, match="entrance_pupil_axial_position"):
        legacy = optic.paraxial.entrance_pupil_z()
    assert_allclose(legacy, optic.paraxial.entrance_pupil_axial_position())


class TestViewerRealSpaceLimits:
    def test_folded_view_covers_folded_arm(self, set_test_backend):
        """Default 2-D view limits must cover the whole vertex chain: the
        folded arm of ``folded()`` ends at (0, 26, 24), which the previous
        first-to-last-z / symmetric-about-the-axis sizing clipped."""
        import matplotlib.pyplot as plt

        from optiland.visualization.system import OpticViewer

        optic = folded()
        viewer = OpticViewer(optic)
        fig, ax, _ = viewer.view(projection="YZ")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plt.close(fig)

        assert xlim[0] <= 0.0 and xlim[1] >= 24.0
        assert ylim[1] >= 26.0

    def test_straight_view_limits_unchanged(self, set_test_backend):
        """For a straight +z system the vertex chain has x = y = 0, so the
        vertex-based sizing must reduce to the previous behavior: z from
        first to last surface, transverse limits symmetric about the axis."""
        import matplotlib.pyplot as plt

        from optiland.visualization.system import OpticViewer

        optic = straight()
        viewer = OpticViewer(optic)
        fig, ax, _ = viewer.view(projection="YZ")
        ylim = ax.get_ylim()
        plt.close(fig)

        assert_allclose(ylim[0], -ylim[1])


# ---------------------------------------------------------------------------
# 16.8 Backend / autograd
# ---------------------------------------------------------------------------


class TestBackendBehavior:
    def test_positions_dtype_and_no_detach(self, set_test_backend):
        """Constructing positions from backend scalars must not detach
        tensors from the autograd graph (torch) and must stay float64."""
        optic = folded()
        positions = optic.surfaces.positions
        assert be.size(positions) == 5
        if be.get_backend() == "torch":
            import torch

            assert positions.dtype == torch.float64

    def test_autograd_through_folded_first_order(self, set_test_backend):
        """Gradient of f2 with respect to the lens radius, through the
        folded path, matches a finite-difference estimate."""
        if be.get_backend() != "torch":
            pytest.skip("Autodiff smoke test requires the torch backend.")
        be.grad_mode.enable()

        optic = folded()
        radius = optic.surfaces[1].geometry.radius
        radius.requires_grad_(True)
        f2 = optic.paraxial.f2()
        f2.backward()
        grad = float(be.to_numpy(radius.grad))

        # Finite-difference reference on fresh systems.
        h = 1e-4
        f2_vals = []
        for delta in (+h, -h):
            probe = folded()
            probe.surfaces[1].geometry.radius = be.array(25.84 + delta)
            f2_vals.append(float(be.to_numpy(probe.paraxial.f2())))
        grad_fd = (f2_vals[0] - f2_vals[1]) / (2 * h)
        assert_allclose(grad, grad_fd, rtol=1e-5)
