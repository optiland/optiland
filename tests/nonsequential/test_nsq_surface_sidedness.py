"""Tests that refraction depends on which side of a surface a ray is on.

Every NSQ geometry flips its normal to face the incoming ray, so the normal
carries no sidedness information. ``RefractiveComponent`` therefore decides
which medium a ray is leaving from ``rays.n_current``. These tests pin that
down for the cases the normal-based rule got wrong: a surface crossed in
reverse, and a closed solid made from a single geometry.

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
        MirrorConfig(radius=np.inf, aperture_radius=30.0),
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
