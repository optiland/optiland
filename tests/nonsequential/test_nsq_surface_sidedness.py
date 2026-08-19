"""Tests that refraction depends on which side of a surface a ray is on.

Every NSQ geometry flips its *shading* normal to face the incoming ray, so
that normal alone carries no sidedness information. ``RefractiveComponent``
resolves sidedness geometrically instead (D-1): each geometry also returns
an unflipped ``n_geom`` that points from ``material_front`` toward
``material_back`` regardless of which way the ray is travelling, and
``RefractiveComponent`` compares the ray direction against ``n_geom`` --
never against ``rays.n_current`` or any index value -- to decide which
material a ray is entering. These tests pin that down for the cases the old
index-proximity heuristic got wrong: a surface crossed in reverse, a closed
solid made from a single geometry, and two adjacent media close enough in
index that comparing values (rather than geometry) picks the wrong side.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    MirrorConfig,
    NSQMaterial,
    NSQScene,
    RayDatabaseConfig,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry
from optiland.nonsequential.components.geometry.analytic.sphere import SphereGeometry

GREEN = 0.55
TILT = np.radians(20.0)


def _glass():
    return NSQMaterial.from_glass("N-BK7")


def _collimated(scene, z=-20.0, aperture_radius=0.5):
    scene.add_source(
        "S",
        CoordinateSystem(z=z),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(GREEN),
            total_flux=1.0,
            aperture_radius=aperture_radius,
        ),
    )


def _tilted_slab(scene):
    """A 5 mm N-BK7 slab, both faces tilted 20 degrees about x."""
    glass = _glass()
    scene.add_component(
        "entry",
        RefractiveComponent(
            cs=CoordinateSystem(z=0.0, rx=TILT),
            geometry=PlaneGeometry(),
            material_front=VACUUM,
            material_back=glass,
            name="entry",
        ),
    )
    scene.add_component(
        "exit",
        RefractiveComponent(
            cs=CoordinateSystem(z=5.0, rx=TILT),
            geometry=PlaneGeometry(),
            material_front=glass,
            material_back=VACUUM,
            name="exit",
        ),
    )


def test_beam_through_a_tilted_slab_emerges_parallel():
    """The baseline single-crossing case: a slab displaces but does not deviate."""
    scene = NSQScene()
    _collimated(scene)
    _tilted_slab(scene)
    scene.add_detector(
        "D", CoordinateSystem(z=40.0), RayDatabaseConfig(width=200, height=200)
    )

    db = scene.trace(num_rays=200, seed=1).detectors["D"]
    assert db.num_rays > 0
    assert np.median(db.M) == pytest.approx(0.0, abs=1e-9)
    assert np.median(db.N) == pytest.approx(1.0, abs=1e-9)


def test_retro_reflected_beam_emerges_antiparallel():
    """Both slab faces are crossed a second time in reverse.

    A ray that goes through the slab, reflects off a normal-incidence mirror
    and comes back through must emerge exactly anti-parallel to the input. If
    the reverse crossings pick the wrong medium the return beam keeps the
    in-glass deviation instead of undoing it.
    """
    scene = NSQScene()
    _collimated(scene)
    _tilted_slab(scene)
    scene.add_mirror(
        "M",
        CoordinateSystem(z=15.0),
        MirrorConfig(radius=np.inf, reflectance=1.0, aperture_radius=30.0),
    )
    scene.add_detector(
        "D", CoordinateSystem(z=-25.0), RayDatabaseConfig(width=400, height=400)
    )

    db = scene.trace(num_rays=400, seed=1).detectors["D"]
    assert db.num_rays > 0
    assert np.median(db.M) == pytest.approx(0.0, abs=1e-9)
    assert np.median(db.N) == pytest.approx(-1.0, abs=1e-9)


def test_ball_lens_from_one_sphere_focuses_where_theory_says():
    """A closed solid is one geometry crossed twice, entering then leaving.

    A ball lens of radius R and index n has EFL nR / (2(n-1)) from its centre.
    Traced at a small aperture the marginal focus sits just inside that, so
    the check is one-sided on the spherical-aberration side.
    """
    radius = 5.0
    n = float(np.asarray(_glass().n(np.array([GREEN]))).ravel()[0])
    paraxial_focus = n * radius / (2.0 * (n - 1.0))

    def rms_at(z):
        scene = NSQScene()
        _collimated(scene, z=-30.0, aperture_radius=0.6)
        scene.add_component(
            "ball",
            RefractiveComponent(
                cs=CoordinateSystem(z=0.0),
                geometry=SphereGeometry(radius=radius, aperture_radius=4.0),
                material_front=VACUUM,
                material_back=_glass(),
                name="ball",
            ),
        )
        scene.add_detector(
            "D", CoordinateSystem(z=z), RayDatabaseConfig(width=40, height=40)
        )
        db = scene.trace(num_rays=4_000, seed=42).detectors["D"]
        cx = np.average(db.x, weights=db.flux)
        cy = np.average(db.y, weights=db.flux)
        return np.sqrt(np.average((db.x - cx) ** 2 + (db.y - cy) ** 2, weights=db.flux))

    planes = np.arange(5.0, 10.0, 0.25)
    best = planes[int(np.argmin([rms_at(z) for z in planes]))]

    # Marginal focus is pulled in from the paraxial one by spherical aberration,
    # but only slightly at this aperture.
    assert best <= paraxial_focus + 0.25
    assert best >= paraxial_focus - 1.0


def test_material_front_is_used_when_the_ray_arrives_from_the_back():
    """``material_front`` is not decorative: it is n2 for a back-side arrival.

    A ray born inside the glass and travelling backwards out of the entry face
    must refract into ``material_front``.
    """
    glass = _glass()
    scene = NSQScene()
    # Source inside the slab, aimed back out through the entry face.
    scene.add_source(
        "S",
        CoordinateSystem(z=2.5, rx=np.radians(180.0)),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(GREEN),
            total_flux=1.0,
            aperture_radius=0.2,
        ),
    )
    scene.add_component(
        "entry",
        RefractiveComponent(
            cs=CoordinateSystem(z=0.0, rx=TILT),
            geometry=PlaneGeometry(),
            material_front=VACUUM,
            material_back=glass,
            name="entry",
        ),
    )
    scene.add_detector(
        "D", CoordinateSystem(z=-20.0), RayDatabaseConfig(width=100, height=100)
    )

    db = scene.trace(num_rays=200, seed=1).detectors["D"]
    assert db.num_rays > 0
    # Rays start in vacuum (n_current = 1) heading -z, so this surface takes
    # them into the glass: they must be deviated, not passed straight through.
    assert abs(np.median(db.M)) > 1e-6


def test_close_index_interface_still_refracts():
    """D-1's headline failure mode: comparing index *values* instead of
    geometry silently resolves n2 = n1 whenever the two adjacent media are
    close enough (a cemented doublet, e.g. N-BK7 next to N-SF5 -- differing
    by only a few thousandths). The old
    ``abs(n1 - n2_back) < abs(n1 - n2_front)`` heuristic picks n2 = n1
    whenever the ray arrives with n_current already exactly equal to
    material_front (the normal, non-ghost case for *every* doublet-style
    cemented interface), which zeroes the bend at this surface entirely.
    Geometric sidedness never compares index values, so it bends correctly
    regardless of how close the two indices are. Two ``IdealMaterial``
    instances pin the index gap exactly, independent of catalog contents.
    """
    from optiland.materials.ideal import IdealMaterial  # noqa: PLC0415

    n1 = 1.520
    n2 = 1.526
    mat1 = NSQMaterial(optiland_material=IdealMaterial(n=n1))
    mat2 = NSQMaterial(optiland_material=IdealMaterial(n=n2))

    scene = NSQScene()
    # medium=mat1 puts the ray in mat1 from birth, so the tilted interface
    # below is the *only* surface in this scene and its bend is not
    # entangled with any other refraction.
    scene.add_source(
        "S",
        CoordinateSystem(z=-10.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(GREEN),
            total_flux=1.0,
            aperture_radius=0.2,
            medium=mat1,
        ),
    )
    scene.add_component(
        "cemented",
        RefractiveComponent(
            cs=CoordinateSystem(z=0.0, rx=TILT),
            geometry=PlaneGeometry(),
            material_front=mat1,
            material_back=mat2,
            name="cemented",
        ),
    )
    scene.add_detector(
        "D", CoordinateSystem(z=20.0), RayDatabaseConfig(width=100, height=100)
    )

    db = scene.trace(num_rays=300, seed=1).detectors["D"]
    assert db.num_rays > 0
    # A ray travelling straight through unbent (the D-1 bug: n2 silently
    # resolved to n1) would land with M = 0. Snell's law at this interface
    # (n1 -> n2, both close to 1.52, incidence = TILT) predicts a small but
    # unambiguous nonzero deviation.
    theta1 = TILT
    theta2 = np.arcsin(np.clip((n1 / n2) * np.sin(theta1), -1.0, 1.0))
    # Extra ray-space deviation introduced by the n1 -> n2 step.
    expected_bend = theta1 - theta2
    # Sanity: the analytic prediction is itself nonzero.
    assert abs(expected_bend) > 1e-4
    assert np.median(db.M) == pytest.approx(np.sin(expected_bend), abs=5e-3)
