"""Robust ray aiming for a finite-conjugate wide-angle system.

The reference system is Example 1 of US 10,203,487 B2 (Seiko Epson): a
stereographic fisheye projection lens, f = 1.726 mm, F/2.0, half field 80
degrees, traced from a screen 300 mm in front of the first surface. From
about 60 degrees on, its paraxial chief-ray seed misses the system (at 60
degrees it sits on a fold of the stop-height landscape instead), and at 80
degrees the launch directions that reach the stop form a strip about 0.08
degrees wide.

Reference chief rays come from a bisection on the meridional launch angle
that does not use any aimer beyond the paraxial object point.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import AbbeMaterial
from optiland.optic import Optic
from optiland.rays import RealRays
from optiland.rays.ray_aiming.parameterization import LaunchParameterization
from optiland.rays.ray_aiming.paraxial import ParaxialRayAimer
from optiland.rays.ray_aiming.robust import (
    RobustRayAimer,
    _carry_to_seed,
    _launch_from_seed,
    _offset_from_seed,
)

from .test_ray_aiming_numerics import _scan_only

INF = float("inf")

# US 10,203,487 B2, Example 1, Table 1, from the screen (object) plane:
# (radius, thickness to the next surface, nd, vd); nd None = air follows.
_PRESCRIPTION = [
    (INF, 300.000, None, None),
    (18.365, 1.000, 1.81600, 46.62),
    (7.127, 2.687, None, None),
    (12.942, 0.900, 1.81600, 46.62),
    (6.256, 1.951, None, None),
    (20.963, 0.800, 1.80610, 40.93),
    (5.569, 2.103, None, None),
    (-25.048, 0.900, 1.83481, 42.72),
    (7.323, 3.500, 1.82115, 24.06),
    (-7.101, 3.056, None, None),
    (33.957, 1.544, 1.56883, 56.36),
    (-6.554, 0.000, None, None),
    (INF, 0.812, None, None),
    (-4.737, 0.700, 1.84666, 23.78),
    (6.000, 2.147, 1.61800, 63.33),
    (-5.448, 0.100, None, None),
    (12.972, 2.800, 1.59201, 67.02),
    (-4.762, 1.000, None, None),
    (INF, 1.800, 1.51633, 64.14),
    (INF, 3.000, None, None),
    (INF, 0.000, None, None),
]
_STOP_INDEX = 12
# Table 2: conic constant and (A4, A6) of the even aspheres.
_ASPHERES = {
    9: (-1.0434, 1.6161e-04, -1.0469e-05),
    16: (-0.9870, -1.9356e-03, 0.0),
    17: (0.0, 4.4369e-04, 9.9655e-07),
}
_MAX_FIELD = 80.0


def fisheye_projection_lens():
    """US 10,203,487 B2 Example 1: finite conjugate, fields to 80 degrees."""
    optic = Optic(name="US 10,203,487 B2 Example 1")
    for index, (radius, thickness, nd, vd) in enumerate(_PRESCRIPTION):
        kwargs = {
            "index": index,
            "radius": radius,
            "thickness": thickness,
            "material": AbbeMaterial(nd, vd, model="buchdahl") if nd else "air",
            "is_stop": index == _STOP_INDEX,
        }
        if index in _ASPHERES:
            conic, a4, a6 = _ASPHERES[index]
            kwargs.update(
                surface_type="even_asphere", conic=conic, coefficients=[0.0, a4, a6]
            )
        optic.surfaces.add(**kwargs)
    optic.set_aperture(aperture_type="imageFNO", value=2.0)
    optic.fields.set_type("angle")
    for angle in (0.0, 20.0, 40.0, 60.0, _MAX_FIELD):
        optic.fields.add(y=angle)
    optic.wavelengths.add(value=0.588, is_primary=True)
    optic.ray_tracer.set_aiming("robust")
    return optic


def _f(value):
    return float(be.to_numpy(value).reshape(-1)[0])


def _launch_angle(M, N):
    return math.degrees(math.atan2(_f(M), _f(N)))


def _chief_launch_angle(optic, Hy, wavelength):
    """Meridional launch angle (deg) of the real chief ray, by bisection.

    Scans launch directions from the field's object point within one degree
    of the paraxial seed direction, keeps the sign change of the stop height
    nearest the seed, and bisects it.
    """
    seed = ParaxialRayAimer(optic).aim_rays(
        (be.array([0.0]), be.array([Hy])),
        be.array([wavelength]),
        (be.array([0.0]), be.array([0.0])),
    )
    x0, y0, z0, _, M0, N0 = (_f(v) for v in seed)
    stop = optic.surfaces.stop_index

    def stop_height(angles):
        angles = np.atleast_1d(angles)
        n = angles.size
        rays = RealRays(
            be.array(np.full(n, x0)),
            be.array(np.full(n, y0)),
            be.array(np.full(n, z0)),
            be.zeros(n),
            be.array(np.sin(angles)),
            be.array(np.cos(angles)),
            be.ones(n),
            be.array(np.full(n, wavelength)),
        )
        optic.surfaces.trace(rays)
        return be.to_numpy(optic.surfaces.y[stop]).reshape(-1)

    seed_angle = math.atan2(M0, N0)
    angles = seed_angle + np.radians(np.linspace(-1.0, 1.0, 4001))
    height = stop_height(angles)
    alive = np.isfinite(height)
    crossing = alive[:-1] & alive[1:] & (np.sign(height[:-1]) != np.sign(height[1:]))
    brackets = np.nonzero(crossing)[0]
    k = brackets[np.argmin(np.abs(angles[brackets] - seed_angle))]
    lo, hi, sign_lo = angles[k], angles[k + 1], np.sign(height[k])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.sign(stop_height(mid)[0]) == sign_lo:
            lo = mid
        else:
            hi = mid
    return math.degrees(0.5 * (lo + hi))


@pytest.mark.parametrize(
    "angle, wavelength",
    [(60.0, 0.588), (70.0, 0.588), (80.0, 0.588), (80.0, 0.95)],
)
def test_cold_chief_ray_is_found_where_the_paraxial_seed_fails(
    set_test_backend, angle, wavelength
):
    """A fresh aimer finds the real chief ray at fields whose paraxial seed
    does not converge. These fields used to exhaust the marching budget
    and raise."""
    optic = fisheye_projection_lens()
    Hy = angle / _MAX_FIELD
    aimer = RobustRayAimer(optic)
    _, _, _, _, M, N = aimer.aim_rays((0.0, Hy), wavelength, (0.0, 0.0))

    reference = _chief_launch_angle(optic, Hy, wavelength)
    assert _launch_angle(M, N) == pytest.approx(reference, abs=1e-6)


def test_all_fields_in_one_call_warm_start_from_the_previous_field(
    set_test_backend,
):
    """Each field warm-starts from the one before it, and that suffices up
    to 80 degrees: what carries over is the previous chief's rotation away
    from its own paraxial seed, not its absolute direction."""
    optic = fisheye_projection_lens()
    aimer = RobustRayAimer(optic)
    Hy = be.array([0.0, 0.25, 0.5, 0.75, 1.0])
    zeros = be.zeros(5)
    x, y, z, L, M, N = aimer.aim_rays((zeros, Hy), 0.588, (zeros, zeros))

    report = aimer.last_report
    assert report.converged
    strategies = [field.chief_seed_strategy for field in report.field_reports]
    assert strategies == ["direct_paraxial"] + ["warm_map"] * 4

    rays = RealRays(x, y, z, L, M, N, be.ones(5), be.full_like(x, 0.588))
    optic.surfaces.trace(rays)
    stop = optic.surfaces.stop_index
    stop_r = np.hypot(
        be.to_numpy(optic.surfaces.x[stop]), be.to_numpy(optic.surfaces.y[stop])
    )
    assert np.max(stop_r) < 1e-6


def test_full_pupil_is_traced_at_the_edge_of_the_field(set_test_backend):
    optic = fisheye_projection_lens()
    rays = optic.trace(
        Hx=0.0, Hy=1.0, wavelength=0.588, num_rays=6, distribution="hexapolar"
    )
    assert np.all(np.isfinite(be.to_numpy(rays.x)))
    assert np.all(be.to_numpy(rays.i) > 0.0)


def test_direction_scan_finds_the_chief_for_a_finite_conjugate(set_test_backend):
    """With the direct solve failed and marching disabled, the last-resort
    scan (which used to run for infinite conjugates only) finds the
    80-degree chief ray by sweeping launch directions."""
    optic = fisheye_projection_lens()
    aimer = RobustRayAimer(optic)
    with _scan_only() as state:
        _, _, _, _, M, N = aimer.aim_rays((0.0, 1.0), 0.588, (0.0, 0.0))

    assert state["scan_seen"]
    assert aimer.last_report.field_reports[0].chief_seed_strategy == "scan"
    reference = _chief_launch_angle(optic, 1.0, 0.588)
    assert _launch_angle(M, N) == pytest.approx(reference, abs=1e-6)


def _seed(y, angle_deg, z=-300.0):
    angle = math.radians(angle_deg)
    state = (0.0, y, z, 0.0, math.sin(angle), math.cos(angle))
    return tuple(be.array([v]) for v in state)


def test_finite_carry_transports_the_offset_not_the_direction(set_test_backend):
    param = LaunchParameterization(
        is_infinite=False, u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0)
    )
    seed_a = _seed(-250.0, 40.0)
    seed_b = _seed(-1700.0, 80.0)
    offset = (0.0, 1e-3)
    launch_a = tuple(_f(v) for v in _launch_from_seed(param, seed_a, offset))
    assert _offset_from_seed(param, seed_a, launch_a) == pytest.approx(
        offset, abs=1e-12
    )

    carried = _carry_to_seed(param, seed_b, launch_a, offset)
    carried_state = tuple(_f(v) for v in carried)
    # The same rotation away from B's own seed, from B's object point...
    assert _offset_from_seed(param, seed_b, carried_state) == pytest.approx(
        offset, abs=1e-12
    )
    assert carried_state[1] == pytest.approx(-1700.0)
    # ...not A's direction, which runs 40 degrees off B's field.
    assert abs(carried_state[4] - launch_a[4]) > 0.3


def test_infinite_carry_keeps_the_launch_point(set_test_backend):
    param = LaunchParameterization(
        is_infinite=True, u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0)
    )
    seed_a = _seed(-5.0, 40.0, z=-10.0)
    seed_b = _seed(-20.0, 80.0, z=-10.0)
    launch_a = tuple(_f(v) for v in _launch_from_seed(param, seed_a, (0.5, 0.2)))

    carried = tuple(_f(v) for v in _carry_to_seed(param, seed_b, launch_a, None))
    # Launch point from A, direction from B's field.
    assert carried[:3] == pytest.approx(launch_a[:3], abs=1e-12)
    assert carried[3:] == pytest.approx(tuple(_f(v) for v in seed_b[3:]), abs=1e-12)
