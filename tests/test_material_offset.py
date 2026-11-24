"""Tests for OffsetMaterial, a material with offset index and Abbe number."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.materials import (
    AbbeMaterial,
    BaseMaterial,
    IdealMaterial,
    Material,
    MaterialAtConditions,
    OffsetMaterial,
)
from optiland.materials.offset import WAVELENGTH_C, WAVELENGTH_D, WAVELENGTH_F

from .utils import assert_allclose


def _nd_vd(material):
    n_d = material.n(WAVELENGTH_D)
    return n_d, (n_d - 1) / (material.n(WAVELENGTH_F) - material.n(WAVELENGTH_C))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Material("N-BK7"),
        lambda: Material("SF11", "schott"),
        lambda: AbbeMaterial(1.6, 40, model="buchdahl"),
    ],
    ids=["N-BK7", "SF11", "abbe"],
)
@pytest.mark.parametrize(
    "delta_nd, delta_vd", [(0.001, 0.0), (0.0, -0.8), (-2e-3, 1.5)]
)
def test_offsets_are_exact(set_test_backend, factory, delta_nd, delta_vd):
    base = factory()
    n_d, v_d = _nd_vd(base)
    n_d2, v_d2 = _nd_vd(OffsetMaterial(base, delta_nd, delta_vd))
    assert_allclose(n_d2, n_d + delta_nd, rtol=0, atol=1e-12)
    assert_allclose(v_d2, v_d + delta_vd, rtol=1e-10)


def test_zero_offsets_are_the_identity(set_test_backend):
    base = Material("N-BK7")
    wavelengths = be.linspace(0.4, 0.8, 7)
    offset = OffsetMaterial(base)
    assert_allclose(offset.n(wavelengths), base.n(wavelengths), rtol=0, atol=0)
    assert_allclose(offset.n(0.55), base.n(0.55), rtol=0, atol=0)


def test_relative_partial_dispersion_is_kept(set_test_backend):
    base = Material("SF11", "schott")
    offset = OffsetMaterial(base, 0.002, -1.0)
    assert_allclose(offset.PgF(), base.PgF(), rtol=1e-10)
    for wavelength in (0.4, 0.7, 1.0):
        assert_allclose(offset.P_ref(wavelength), base.P_ref(wavelength), rtol=1e-10)


def test_array_wavelengths_keep_their_shape(set_test_backend):
    base = Material("N-BK7")
    offset = OffsetMaterial(base, 0.001, 0.5)
    wavelengths = be.linspace(0.45, 0.65, 5)
    values = offset.n(wavelengths)
    assert be.to_numpy(values).shape == (5,)
    for i, wavelength in enumerate(be.to_numpy(wavelengths)):
        assert_allclose(values[i], offset.n(float(wavelength)))


def test_extinction_coefficient_is_passed_through(set_test_backend):
    base = IdealMaterial(1.6, 0.02)
    assert_allclose(OffsetMaterial(base, 0.01).k(0.55), 0.02)
    glass = Material("N-BK7")
    assert_allclose(OffsetMaterial(glass, 0.01, 1.0).k(0.55), glass.k(0.55))


def test_the_base_material_is_not_modified(set_test_backend):
    base = Material("N-BK7")
    before = base.to_dict()
    n = base.n(0.55)
    OffsetMaterial(base, 0.01, -1.0).n(0.55)
    assert base.to_dict() == before
    assert_allclose(base.n(0.55), n, rtol=0, atol=0)


def test_kwargs_are_forwarded(set_test_backend):
    base = Material("N-BK7")
    offset = OffsetMaterial(base, 0.001)
    hot = MaterialAtConditions(base, 60.0)
    assert_allclose(
        offset.n(0.55, temperature=60.0, pressure=1.0),
        OffsetMaterial(hot, 0.001).n(0.55),
    )


def test_non_dispersive_material_index_offset(set_test_backend):
    offset = OffsetMaterial(IdealMaterial(1.5), 0.01)
    assert_allclose(offset.n(0.45), 1.51)
    assert_allclose(offset.n(be.linspace(0.4, 0.7, 3)), be.full((3,), 1.51))


def test_non_dispersive_material_abbe_offset_raises(set_test_backend):
    with pytest.raises(ValueError, match="non-dispersive"):
        OffsetMaterial(IdealMaterial(1.5), 0.0, 1.0)


def test_gradient_with_respect_to_the_offset(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Torch gradients")
    import torch

    base = Material("SF11", "schott")
    step = 1e-6
    for name in ("delta_nd", "delta_vd"):
        delta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        for _ in range(2):  # a trainable offset is never served from a cache
            OffsetMaterial(base, **{name: delta}).n(0.45).sum().backward()
            grad = delta.grad.clone()
            delta.grad = None
            plus = OffsetMaterial(base, **{name: step}).n(0.45)
            minus = OffsetMaterial(base, **{name: -step}).n(0.45)
            assert grad != 0
            assert_allclose(grad, (plus - minus) / (2 * step), rtol=1e-6)


@pytest.mark.parametrize(
    "factory",
    [lambda: Material("N-BK7"), lambda: IdealMaterial(1.6, 0.01)],
    ids=["catalog", "ideal"],
)
def test_dict_round_trip(set_test_backend, factory):
    offset = OffsetMaterial(factory(), 0.002)
    data = offset.to_dict()
    assert data["type"] == "OffsetMaterial"
    assert data["delta_nd"] == 0.002
    assert data["delta_vd"] == 0.0
    for loaded in (OffsetMaterial.from_dict(data), BaseMaterial.from_dict(data)):
        assert isinstance(loaded, OffsetMaterial)
        assert loaded.to_dict() == data
        assert loaded == offset
        assert_allclose(loaded.n(0.55), offset.n(0.55))


def test_attributes_and_display_name():
    base = Material("N-BK7")
    offset = OffsetMaterial(base, 0.001, -0.5)
    assert offset.material is base
    assert offset.delta_nd == 0.001
    assert offset.delta_vd == -0.5
    assert offset.display_name == "N-BK7 (Δnd +0.001, Δvd -0.5)"
    assert offset.spectral_range() == base.spectral_range()
    assert offset.thermal_expansion == base.thermal_expansion


def test_only_a_material_can_be_offset():
    with pytest.raises(TypeError, match="BaseMaterial"):
        OffsetMaterial(1.5, 0.01)
