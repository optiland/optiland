"""Systems and helpers shared by the tests of the compiled trace plan.

``tests/test_trace_plan.py`` compares the plan with the generic trace, and
``tests/test_trace_threads.py`` compares a trace on one thread with a trace
on several threads. Both trace the systems of ``SYSTEMS``, which are in the
domain of the plan, and compare the ray arrays and the records bit for bit.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
import optiland.surfaces._trace_plan as plan
from optiland.geometries import EvenAsphere, StandardGeometry
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.physical_apertures import OffsetRadialAperture, RadialAperture
from optiland.rays import RealRays
from optiland.samples.objectives import CookeTriplet, DoubleGauss
from optiland.samples.simple import AsphericSinglet, CementedAchromat, TelescopeDoublet
from optiland.samples.telescopes import HubbleTelescope
from optiland.surfaces import ImageSurface

from .utils import assert_bits_equal

# The ray arrays and the records that a trace sets, in the order of the plan.
# test_cleared_records_are_the_arrays_of_a_reset_surface in
# tests/test_trace_plan.py checks that the records are complete.
RAY_ATTRS = (*plan._OUTPUTS, "w")
RECORD_ATTRS = plan._CLEARED


def _fan(n, wavelengths=(0.55,), angle=0.05, height=2.0):
    """n rays in a fan that starts at z = 0, with skew rays.

    The heights in y go from -height to height and the angles from -angle to
    angle (radians). L = 0.01 gives skew rays, which leave the y-z plane.
    np.resize repeats the wavelengths from ray to ray.
    """
    ang = np.linspace(-angle, angle, n)
    return RealRays(
        np.zeros(n),
        np.linspace(-height, height, n),
        np.zeros(n),
        np.full(n, 0.01),
        np.sin(ang),
        np.sqrt(1 - 0.01**2 - np.sin(ang) ** 2),
        np.ones(n),
        np.resize(np.asarray(wavelengths, dtype=float), n),
    )


def _image_surface():
    """The Cooke triplet with an ImageSurface as its last surface."""
    optic = CookeTriplet()
    surfaces = optic.surfaces._surfaces
    last = surfaces[-1]
    surfaces[-1] = ImageSurface(surfaces[-2], last.geometry, last.material_post)
    return optic


def _flat_and_conic():
    """The Cooke triplet with a flat StandardGeometry and a conic surface."""
    optic = CookeTriplet()
    surfaces = optic.surfaces.surfaces
    cs = surfaces[3].geometry.cs
    surfaces[3].geometry = StandardGeometry(cs, be.inf)
    surfaces[5].geometry.k = be.array(-0.6)
    return optic


def _decentered():
    """The Cooke triplet with surfaces moved in x and in y."""
    optic = CookeTriplet()
    surfaces = optic.surfaces.surfaces
    surfaces[2].geometry.cs.x = be.array(0.1)
    surfaces[4].geometry.cs.y = be.array(-0.2)
    return optic


def _absorbing_singlet():
    """A singlet of an IdealMaterial with k > 0: one absorption coefficient."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        radius=20.0,
        thickness=5.0,
        material=IdealMaterial(n=1.5, k=1e-5),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=-40.0, thickness=30.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=8.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.fields.add(y=3)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _oap():
    """An off-axis parabolic mirror: decenter, tilt, conic, offset aperture.

    Both roots of the parabola can be hits in front of the ray, and the
    aperture selects the root on the used part of the mirror.
    """
    rfl = 25.4
    optic = Optic(name="oap")
    optic.surfaces.add(index=0, z=0, radius=be.inf)
    optic.surfaces.add(index=1, z=10, is_stop=True)
    optic.surfaces.add(
        index=2,
        z=0,
        y=-rfl / 2,
        radius=-rfl,
        conic=-1,
        aperture=OffsetRadialAperture(r_max=12.7, offset_y=rfl),
        material="mirror",
        rx=np.pi / 2,
    )
    optic.surfaces.add(
        index=3,
        z=rfl,
        y=40,
        radius=be.inf,
        aperture=OffsetRadialAperture(r_max=12.7, offset_y=0),
        rx=np.pi / 2,
    )
    optic.set_aperture(aperture_type="EPD", value=5.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.fields.add(y=1)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _clipped():
    """The Cooke triplet with radial apertures that clip part of the pupil."""
    optic = CookeTriplet()
    surfaces = optic.surfaces.surfaces
    surfaces[1].aperture = RadialAperture(r_max=9.0)
    surfaces[3].aperture = RadialAperture(r_max=3.0, r_min=0.5)
    surfaces[5].aperture = OffsetRadialAperture(r_max=8.0, offset_x=0.4)
    return optic


def _tilted():
    """The telescope doublet with a surface tilted about x, y and z."""
    optic = TelescopeDoublet()
    cs = optic.surfaces.surfaces[2].geometry.cs
    cs.rx = be.array(0.01)
    cs.ry = be.array(-0.02)
    cs.rz = be.array(0.3)
    cs.x = be.array(0.1)
    return optic


def _asphere_triplet():
    """The Cooke triplet with two even aspheres (two and six coefficients)."""
    optic = CookeTriplet()
    surfaces = optic.surfaces.surfaces
    for index, coefficients in (
        (1, [2e-4, -3e-6]),
        (4, [-1e-4, 2e-6, -3e-8, 4e-10, -5e-12, 6e-14]),
    ):
        geometry = surfaces[index].geometry
        surfaces[index].geometry = EvenAsphere(
            geometry.cs, geometry.radius, -0.3, coefficients=coefficients
        )
    return optic


def _flat_asphere():
    """The Cooke triplet with a flat even asphere: R = inf, with coefficients.

    The base conic is the plane z = 0, so the seed is the distance of FLAT.
    """
    optic = CookeTriplet()
    surface = optic.surfaces.surfaces[3]
    surface.geometry = EvenAsphere(
        surface.geometry.cs, be.inf, coefficients=[1e-3, -2e-5]
    )
    return optic


def _clipped_aspheres():
    """The asphere triplet with an aperture on each asphere.

    Each aperture takes part in the conic seed and clips rays.
    """
    optic = _asphere_triplet()
    surfaces = optic.surfaces.surfaces
    surfaces[1].aperture = RadialAperture(r_max=6.0)
    surfaces[4].aperture = OffsetRadialAperture(r_max=4.0, offset_y=0.3)
    return optic


def _oap_asphere():
    """The off-axis parabolic mirror of ``_oap``, with even asphere terms.

    A decentered and tilted asphere mirror with an offset aperture.
    """
    optic = _oap()
    surface = optic.surfaces.surfaces[2]
    geometry = surface.geometry
    surface.geometry = EvenAsphere(
        geometry.cs, geometry.radius, -1.0, coefficients=[2e-6, -1e-9]
    )
    return optic


def _immersed_image():
    """A singlet whose image surface is followed by 3 mm of absorbing glass (N-SF11).

    optic.trace propagates the rays 3 mm past the image surface, and N-SF11
    absorbs, so the final step of the plan has an exponent row.
    """
    optic = Optic(name="immersed")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=25.84, thickness=4.0, material="N-BK7", is_stop=True
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=40.0)
    optic.surfaces.add(index=3, material="N-SF11")
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.fields.add(y=3.0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    optic.surfaces[-1].thickness = 3.0
    return optic


def _thick_image():
    """The Cooke triplet, propagated 2.5 mm past its image surface in air."""
    optic = CookeTriplet()
    optic.surfaces[-1].thickness = 2.5
    return optic


# Systems in the domain of the plan. CookeTriplet has a glass with k > 0
# (one absorption coefficient), so its traces run the intensity pass. The
# image surfaces of immersed_image and thick_image have a thickness that is
# not 0, so optic.trace moves their rays in the final step of the plan.
SYSTEMS = {
    "cooke": CookeTriplet,
    "double_gauss": DoubleGauss,
    "achromat": CementedAchromat,
    "doublet": TelescopeDoublet,
    "image_surface": _image_surface,
    "flat_and_conic": _flat_and_conic,
    "decentered": _decentered,
    "absorbing": _absorbing_singlet,
    "oap": _oap,
    "clipped": _clipped,
    "tilted": _tilted,
    "hubble": HubbleTelescope,
    "aspheric_singlet": AsphericSinglet,
    "asphere_triplet": _asphere_triplet,
    "flat_asphere": _flat_asphere,
    "clipped_aspheres": _clipped_aspheres,
    "oap_asphere": _oap_asphere,
    "immersed_image": _immersed_image,
    "thick_image": _thick_image,
}


def _state(optic, rays):
    """The ray arrays and the records of all surfaces, by name."""
    out = {f"ray.{name}": getattr(rays, name) for name in RAY_ATTRS}
    for k, surface in enumerate(optic.surfaces.surfaces):
        for name in RECORD_ATTRS:
            out[f"s{k}.{name}"] = getattr(surface, name)
    return out


def _assert_same(a, b):
    """Assert that two trace results hold the same arrays, bit for bit.

    Compiled code can give a NaN another sign bit (see the developer page
    on performance internals), so the sign of NaN is not compared.
    """
    assert a.keys() == b.keys()
    for key in a:
        if a[key] is None or b[key] is None:
            assert a[key] is None and b[key] is None, key
        else:
            assert_bits_equal(a[key], b[key], ignore_nan_sign=True)


def _strong_asphere(coefficients=(-1e-2, 2e-5), max_iter=100):
    """A singlet whose first surface (at z = 3) is a strong even asphere.

    The polynomial bends the surface away from its base conic (R = 6,
    k = -0.5), so some steep rays hit the asphere but miss the base conic.
    """
    optic = Optic()
    optic.surfaces.add(index=0, z=0.0)
    optic.surfaces.add(index=1, z=3.0, material="N-BK7")
    optic.surfaces.add(index=2, z=8.0)
    surface = optic.surfaces.surfaces[1]
    surface.geometry = EvenAsphere(
        surface.geometry.cs,
        6.0,
        -0.5,
        max_iter=max_iter,
        coefficients=list(coefficients),
    )
    return optic
