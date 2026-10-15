"""Extinct rays remain dark when later intersections are invalid."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

import optiland.backend as be
from optiland.coatings import SimpleCoating
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.rays import PolarizedRays, RealRays, create_polarization
from tests.utils import assert_allclose


def clipped_sphere(k: float) -> Optic:
    """An axial live ray and an outer ray clipped before missing a unit sphere."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=1)
    optic.surfaces.add(
        index=1,
        thickness=1,
        aperture=RadialAperture(r_max=0.4),
        material=IdealMaterial(1, k),
        coating=SimpleCoating(0.3),
    )
    optic.surfaces.add(index=2, radius=1, thickness=1)
    optic.surfaces.add(index=3)
    return optic


def axial_bundle(ray_type=PolarizedRays):
    return ray_type(
        be.array([0, 2]),
        be.zeros(2),
        be.zeros(2),
        be.zeros(2),
        be.zeros(2),
        be.ones(2),
        be.ones(2),
        be.full(2, 0.5),
    )


def assert_clipped_sphere(optic: Optic, k: float) -> None:
    """Check analytic transmitted power without interpreting missed-ray poses."""
    expected = 0.3 * np.exp(-4 * np.pi * k / 0.5 * 1000)
    for ray_type in (RealRays, PolarizedRays):
        rays = axial_bundle(ray_type)
        for surface in optic.surfaces[1:]:
            surface.trace(rays)
        assert_allclose(optic.surfaces[1].intensity, [0.3, 0])
        assert np.isfinite(be.to_numpy(rays.z)[0])
        assert not np.isfinite(be.to_numpy(rays.z)[1])  # The sphere was missed.
        if ray_type is PolarizedRays:
            for name in ("unpolarized", "H", "RCP", "unpolarized"):
                state = create_polarization(name)
                rays.update_intensity(state)
                assert_allclose(rays.i, [expected, 0], rtol=1e-12, atol=1e-14)
                assert rays.i[1].item() == 0
                fields = rays.get_exit_fields(state)
                power = sum(be.sum(be.abs(field) ** 2, axis=1) for field in fields)
                assert_allclose(power, [expected, 0], rtol=1e-12, atol=1e-14)
                assert all(be.all(be.isfinite(field)) for field in fields)
                assert all(be.all(field[1] == 0) for field in fields)
        assert_allclose(rays.i, [expected, 0], rtol=1e-12, atol=1e-14)
        assert rays.i[1].item() == 0
        assert_allclose([rays.L[0], rays.M[0], rays.N[0]], [0, 0, 1])


@pytest.mark.parametrize("k", [0, 1e-6])
def test_extinction_before_geometry_miss_survives_native_reload(
    k, tmp_path, set_test_backend
):
    optic = clipped_sphere(k)
    snapshot = optic.to_dict()
    assert_clipped_sphere(optic, k)
    assert_clipped_sphere(Optic.from_dict(snapshot), k)
    path = tmp_path / "original-native-system.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    script = """
import json, sys
import optiland.backend as be
from optiland.optic import Optic
from optiland.materials.material_file import MaterialFile
from tests.test_polarization_extinction import assert_clipped_sphere
def forbidden(*args, **kwargs):
    raise AssertionError('External material access')
MaterialFile._read_file = forbidden
be.set_backend(sys.argv[2])
be.set_precision('float64')
if be.get_backend() == 'torch':
    be.set_device('cpu')
with open(sys.argv[1], encoding='utf-8') as stream:
    optic = Optic.from_dict(json.load(stream))
assert_clipped_sphere(optic, float(sys.argv[3]))
print('verified')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), be.get_backend(), str(k)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "verified"


@pytest.mark.parametrize("ray_type", [RealRays, PolarizedRays])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
@pytest.mark.parametrize("extinguish", ["clip", "attenuate"])
def test_zero_power_survives_later_invalid_attenuation(
    ray_type, invalid, extinguish, set_test_backend
):
    rays = axial_bundle(ray_type)
    if extinguish == "clip":
        rays.clip(be.array([False, True]))
    else:
        rays.attenuate(be.array([0.5, 0]))
    expected = 0.25 if extinguish == "clip" else 0.125
    rays.attenuate(be.array([0.25, invalid]))
    assert_allclose(rays.i, [expected, 0])
    assert rays.i[1].item() == 0
    if ray_type is PolarizedRays:
        rays.update_intensity(create_polarization("unpolarized"))
        assert_allclose(rays.i, [expected, 0])
    # Invalid attenuation on a ray with nonzero power must remain visible.
    rays.attenuate(be.array([np.nan, 1]))
    assert not be.isfinite(rays.i[0])
    assert rays.i[1].item() == 0


@pytest.mark.parametrize("state", ["unpolarized", "H", "RCP"])
@pytest.mark.parametrize("power", [1, 1e-200])
def test_invalid_live_polarization_is_not_hidden(state, power, set_test_backend):
    rays = axial_bundle()
    rays.attenuate(power)
    rays.p = be.full((2, 3, 3), float("nan"))
    rays.clip(be.array([False, True]))
    state = create_polarization(state)
    rays.update_intensity(state)
    assert not be.isfinite(rays.i[0])
    assert rays.i[1].item() == 0
    for field in rays.get_exit_fields(state):
        assert not be.all(be.isfinite(field[0]))
        assert_allclose(field[1], [0, 0, 0])


def test_zero_for_one_source_state_does_not_extinguish_another(set_test_backend):
    rays = axial_bundle()
    rays.p = be.tile(be.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]), (2, 1, 1))
    rays.update_intensity(create_polarization("V"))
    assert_allclose(rays.i, [0, 0])
    rays.attenuate(0.25)
    rays.update_intensity(create_polarization("H"))
    assert_allclose(rays.i, [0.25, 0.25])
    rays.update_intensity(create_polarization("V"))
    assert_allclose(rays.i, [0, 0])


@pytest.mark.parametrize("state", ["unpolarized", "H", "RCP"])
@pytest.mark.parametrize("output", ["intensity", "fields"])
def test_live_field_gradients_with_an_extinct_invalid_matrix(
    state, output, set_test_backend
):
    if be.get_backend() != "torch":
        pytest.skip("Autograd is a Torch capability")
    import torch

    power = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    amplitude = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    rays = axial_bundle()
    rays.i = power
    rays._i0 = power.clone()
    rays.clip(be.array([False, True]))
    # Differentiate a supported field evaluation: the live matrix has a variable
    # amplitude, while the extinct ray's previously invalid matrix is constant.
    rays.p = torch.stack(
        [
            amplitude * torch.eye(3, dtype=torch.float64),
            torch.full((3, 3), float("nan"), dtype=torch.float64),
        ]
    )
    state = create_polarization(state)
    if output == "intensity":
        rays.update_intensity(state)
        loss = rays.i.sum()
    else:
        loss = sum((field.abs() ** 2).sum() for field in rays.get_exit_fields(state))
    d_power, d_amplitude = torch.autograd.grad(loss, (power, amplitude))
    assert_allclose(loss, 2 * 0.4**2)
    assert_allclose(d_power, [0.4**2, 0])
    assert_allclose(d_amplitude, 4 * 0.4)
