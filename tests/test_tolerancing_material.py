"""Tests for the material tolerance types: index and Abbe number."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial, OffsetMaterial
from optiland.materials.offset import WAVELENGTH_C, WAVELENGTH_D, WAVELENGTH_F
from optiland.samples.objectives import CookeTriplet
from optiland.tolerancing import SensitivityAnalysis, Tolerancing, tolerance_registry
from optiland.tolerancing.base import Phase
from optiland.tolerancing.monte_carlo import MonteCarlo
from optiland.tolerancing.types import AbbeTolerance, IndexTolerance
from tests.utils import assert_allclose


def _f(value) -> float:
    return float(np.ravel(be.to_numpy(value))[0])


def _trace(optic):
    rays = optic.trace(0.0, 0.7, 0.55, num_rays=16)
    return np.stack([be.to_numpy(getattr(rays, n)) for n in ("x", "y", "L", "M")])


@pytest.fixture
def tolerancing(set_test_backend):
    """Tolerancing on a Cooke triplet, built after the backend is set."""
    optic = CookeTriplet()
    tol = Tolerancing(optic)
    tol.add_operand("f2", {"optic": optic})
    return tol


def test_registered():
    assert tolerance_registry.get("index") is IndexTolerance
    assert tolerance_registry.get("abbe") is AbbeTolerance


@pytest.mark.parametrize(
    "tolerance_type, delta",
    [("index", 0.002), ("index", -0.002), ("abbe", 1.0), ("abbe", -1.0)],
)
def test_apply_and_reset(tolerancing, set_test_backend, tolerance_type, delta):
    optic = tolerancing.optic
    materials = [surface.material_post for surface in optic.surfaces]
    nominal_trace = _trace(optic)
    tol = tolerancing.add_tolerance(tolerance_type, limit=abs(delta), surface_number=3)
    assert tol.phase is Phase.MATERIAL
    assert tol.label == f"{'Index' if tolerance_type == 'index' else 'Abbe'} S3"

    tol.apply_value(delta)
    perturbed = optic.surfaces[3].material_post
    assert isinstance(perturbed, OffsetMaterial)
    assert perturbed.material is materials[3]
    for i, surface in enumerate(optic.surfaces):
        if i != 3:
            assert surface.material_post is materials[i]
    assert optic.surfaces[4].material_pre is perturbed
    n_d = _f(materials[3].n(WAVELENGTH_D))
    dispersion = _f(materials[3].n(WAVELENGTH_F) - materials[3].n(WAVELENGTH_C))
    n_d2 = _f(perturbed.n(WAVELENGTH_D))
    v_d2 = (n_d2 - 1) / _f(perturbed.n(WAVELENGTH_F) - perturbed.n(WAVELENGTH_C))
    if tolerance_type == "index":
        assert_allclose(n_d2, n_d + delta, rtol=0, atol=1e-12)
    else:
        assert_allclose(v_d2, (n_d - 1) / dispersion + delta, rtol=1e-10)
    assert not np.allclose(_trace(optic), nominal_trace)

    tol.reset()
    for i, surface in enumerate(optic.surfaces):
        assert surface.material_post is materials[i]
    assert_allclose(_trace(optic), nominal_trace, rtol=0, atol=0)


def test_zero_deviation_keeps_the_nominal_object(tolerancing):
    optic = tolerancing.optic
    nominal = optic.surfaces[1].material_post
    tol = tolerancing.add_tolerance("index", limit=0.001, surface_number=1)
    tol.apply_value(0.0)
    assert optic.surfaces[1].material_post is nominal


@pytest.mark.parametrize("surface_number", [1, 3, 5])
def test_index_increase_shortens_the_focal_length(
    tolerancing, set_test_backend, surface_number
):
    """Raising the index adds power to a positive element and removes it from a
    negative one (the F2 element, surface 3), as dP/dn = P / (n - 1)."""
    optic = tolerancing.optic
    f0 = _f(optic.paraxial.f2())
    tol = tolerancing.add_tolerance("index", limit=0.002, surface_number=surface_number)
    tol.apply_value(0.002)
    f_plus = _f(optic.paraxial.f2())
    tol.apply_value(-0.002)
    f_minus = _f(optic.paraxial.f2())
    tol.reset()
    assert _f(optic.paraxial.f2()) == f0
    sign = 1.0 if surface_number == 3 else -1.0
    assert sign * (f_plus - f0) > 0
    assert sign * (f_minus - f0) < 0


def test_abbe_on_non_dispersive_material_raises(tolerancing):
    optic = tolerancing.optic
    optic.updater.set_material(IdealMaterial(1.5), 1)
    with pytest.raises(ValueError, match="non-dispersive"):
        tolerancing.add_tolerance("abbe", limit=1.0, surface_number=1)
    tol = tolerancing.add_tolerance("index", limit=0.01, surface_number=1)
    tol.apply_value(0.01)
    assert_allclose(optic.surfaces[1].material_post.n(0.55), 1.51)


def _add_tolerances(tolerancing):
    tolerancing.add_tolerance("index", surface_number=1, limit=0.001)
    tolerancing.add_tolerance("abbe", surface_number=3, minus=-0.5, plus=1.0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sensitivity_analysis(tolerancing, set_test_backend):
    optic = tolerancing.optic
    materials = [surface.material_post for surface in optic.surfaces]
    f0 = _f(optic.paraxial.f2())
    _add_tolerances(tolerancing)
    sa = SensitivityAnalysis(tolerancing)
    sa.run()
    df = sa.get_results()
    assert list(df["perturbation_type"]) == ["Index S1"] * 2 + ["Abbe S3"] * 2
    assert_allclose(
        df["perturbation_value"].to_numpy(), np.array([-0.001, 0.001, -0.5, 1.0])
    )
    f2 = df["0: f2"].to_numpy()
    assert f2[0] > f0 > f2[1]
    assert all(s.material_post is m for s, m in zip(optic.surfaces, materials))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_monte_carlo(tolerancing, set_test_backend):
    optic = tolerancing.optic
    materials = [surface.material_post for surface in optic.surfaces]
    _add_tolerances(tolerancing)
    mc = MonteCarlo(tolerancing)
    mc.run(6)
    df = mc.get_results()
    assert len(df) == 6
    for tol in tolerancing.perturbations:
        column = df[tol.label].to_numpy()
        assert column.min() >= tol.minus
        assert column.max() <= tol.plus
    assert np.all(np.isfinite(df["0: f2"].to_numpy()))
    assert all(s.material_post is m for s, m in zip(optic.surfaces, materials))
