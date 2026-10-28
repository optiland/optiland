"""Thin-lens fields follow each completed geometrical bend exactly once."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.coatings import FresnelCoating, SimpleCoating
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays import PolarizedRays, RealRays, create_polarization
from tests.utils import assert_allclose


def bundle(ray_type=PolarizedRays):
    return ray_type(*[be.array([float(v)]) for v in (1, 0, 0, 0, 0, 1, 2, 0.55)])


@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize("coated", [False, True])
def test_thin_lens_transports_field_and_retains_power(
    reflect, coated, set_test_backend
):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=1)
    optic.surfaces.add(
        index=1,
        surface_type="paraxial",
        f=10,
        material="mirror" if reflect else "air",
        coating=SimpleCoating(0.36, 0.25) if coated else None,
    )
    rays, scalar = bundle(), bundle(RealRays)
    optic.surfaces[1].trace(rays)
    optic.surfaces[1].trace(scalar)
    direction = be.stack((rays.L, rays.M, rays.N), axis=1)
    expected_power = 2 * ((0.25 if reflect else 0.36) if coated else 1)
    for name in ("H", "V", "unpolarized", "H"):
        state = create_polarization(name)
        fields = rays.get_exit_fields(state)
        for field in fields:
            assert_allclose(be.sum(field * direction, axis=1), 0, atol=1e-14)
        assert_allclose(
            sum(be.sum(be.abs(e) ** 2, axis=1) for e in fields), expected_power
        )
        rays.update_intensity(state)
        assert_allclose(rays.i, expected_power)
    assert_allclose([rays.L, rays.M, rays.N], [scalar.L, scalar.M, scalar.N])
    assert_allclose(rays.opd, scalar.opd)
    # Analytic bend in the X/Z plane: rotating H once gives (N, 0, -L).
    assert_allclose(
        fields[0],
        be.sqrt(be.array(expected_power))
        * be.stack((rays.N, be.zeros(1), -rays.L), axis=1),
    )


def test_chained_thin_lenses_capture_fresh_incident_directions(set_test_backend):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=1)
    optic.surfaces.add(index=1, surface_type="paraxial", f=10, thickness=1)
    optic.surfaces.add(
        index=2, surface_type="paraxial", f=-20, coating=SimpleCoating(0.25)
    )
    rays = bundle()
    optic.surfaces[1].trace(rays)
    incoming = be.stack((rays.L, rays.M, rays.N), axis=1)
    optic.surfaces[2].trace(rays)
    assert_allclose(be.stack((rays.L0, rays.M0, rays.N0), axis=1), incoming)
    field = rays.get_exit_fields(create_polarization("H"))[0]
    expected = be.sqrt(be.array(0.5)) * be.stack((rays.N, be.zeros(1), -rays.L), axis=1)
    assert_allclose(field, expected, atol=1e-14)


def test_thin_lens_jones_coating_uses_incident_angle(set_test_backend):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=1)
    optic.surfaces.add(
        index=1,
        surface_type="paraxial",
        f=10,
        material=IdealMaterial(1.5),
        coating=FresnelCoating(IdealMaterial(1), IdealMaterial(1.5)),
    )
    rays = bundle()
    optic.surfaces[1].trace(rays)
    # The incoming ray is normal to the surface even though the lens bends it.
    # JonesFresnel returns the electric-field amplitude coefficient, 0.8.
    field = rays.get_exit_fields(create_polarization("H"))[0]
    assert_allclose(be.sum(be.abs(field) ** 2, axis=1), 2 * 0.8**2)
    assert_allclose(
        be.sum(field * be.stack((rays.L, rays.M, rays.N), axis=1), axis=1),
        0,
        atol=1e-14,
    )


def test_thin_lens_does_not_overwrite_scattered_direction(set_test_backend):
    class DeterministicScatter:
        def scatter(self, rays, nx, ny, nz):
            # Rotate a known amount after focusing, retaining a unit direction.
            rays.L, rays.N = rays.N, -rays.L
            return rays

    optic = Optic()
    optic.surfaces.add(index=0, thickness=1)
    optic.surfaces.add(index=1, surface_type="paraxial", f=10)
    optic.surfaces[1].interaction_model.bsdf = DeterministicScatter()
    rays = bundle()
    optic.surfaces[1].trace(rays)
    assert_allclose(rays.L, 10 / be.sqrt(be.array(101.0)))
    assert_allclose(rays.N, 1 / be.sqrt(be.array(101.0)))
    field = rays.get_exit_fields(create_polarization("H"))[0]
    assert_allclose(
        be.sum(field * be.stack((rays.L, rays.M, rays.N), axis=1), axis=1),
        0,
        atol=1e-14,
    )


@pytest.mark.parametrize("focal_length", [float("inf"), -float("inf")])
@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize("ray_type", [RealRays, PolarizedRays])
def test_zero_power_thin_lens_preserves_optical_path(
    focal_length, reflect, ray_type, set_test_backend
):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=1)
    optic.surfaces.add(
        index=1,
        surface_type="paraxial",
        f=focal_length,
        material="mirror" if reflect else "air",
    )
    rays = bundle(ray_type)
    rays.opd = be.array([7.0])
    optic.surfaces[1].interaction_model.interact_real_rays(rays)
    assert_allclose(rays.opd, [7.0])
    assert_allclose(rays.N, [-1.0 if reflect else 1.0])
    if ray_type is PolarizedRays:
        fields = rays.get_exit_fields(create_polarization("H"))[0]
        assert_allclose(be.sum(be.abs(fields) ** 2, axis=1), [2.0])
