"""Tests for MaterialAtConditions, a material at a fixed temperature and pressure."""

from __future__ import annotations

import math

import pytest

import optiland.backend as be
from optiland.materials import (
    AbbeMaterial,
    BaseMaterial,
    IdealMaterial,
    Material,
    MaterialAtConditions,
    MaterialFile,
)
from optiland.samples.objectives import CookeTriplet

from .utils import assert_allclose


def test_index_matches_the_wrapped_material_at_the_conditions(set_test_backend):
    glass = Material("N-BK7")
    wrapped = MaterialAtConditions(Material("N-BK7"), 40.0)
    assert_allclose(
        wrapped.n(0.55), glass.n(0.55, temperature=40.0, pressure=1.0), rtol=0, atol=0
    )
    assert_allclose(
        MaterialAtConditions(glass, 40.0, pressure=0.5).n(0.55),
        glass.n(0.55, temperature=40.0, pressure=0.5),
    )


def test_index_changes_with_temperature(set_test_backend):
    glass = Material("N-BK7")
    cold = MaterialAtConditions(glass, -20.0).n(0.55)
    hot = MaterialAtConditions(glass, 80.0).n(0.55)
    assert not math.isclose(be.to_numpy(cold).item(), be.to_numpy(hot).item())


def test_wrappers_at_different_temperatures_keep_separate_caches(set_test_backend):
    glass = Material("N-BK7")
    cold = MaterialAtConditions(glass, 0.0)
    hot = MaterialAtConditions(glass, 60.0)
    wavelengths = be.linspace(0.45, 0.65, 5)
    expected_cold = glass.n(wavelengths, temperature=0.0, pressure=1.0)
    expected_hot = glass.n(wavelengths, temperature=60.0, pressure=1.0)
    for _ in range(2):  # the second pass reads the caches
        assert_allclose(cold.n(wavelengths), expected_cold)
        assert_allclose(hot.n(wavelengths), expected_hot)
        assert_allclose(cold.n(0.55), glass.n(0.55, temperature=0.0, pressure=1.0))
        assert_allclose(hot.n(0.55), glass.n(0.55, temperature=60.0, pressure=1.0))
    assert not be.allclose(cold.n(wavelengths), hot.n(wavelengths))


def test_a_constant_array_gets_the_scalar_index(set_test_backend):
    wrapped = MaterialAtConditions(Material("N-BK7"), 40.0)
    values = wrapped.n(be.full((4,), 0.55))
    assert be.to_numpy(values).shape == (4,)
    assert_allclose(values, be.full((4,), be.to_numpy(wrapped.n(0.55)).item()))


def test_a_mutation_of_the_wrapped_material_is_seen(set_test_backend):
    ideal = IdealMaterial(1.5)
    ideal.index = be.asarray([1.5])  # a plain array, writable in place on both backends
    wrapped = MaterialAtConditions(ideal, 40.0)
    assert_allclose(wrapped.n(0.55), 1.5)
    ideal.index[...] = 1.7
    assert_allclose(wrapped.n(0.55), 1.7)


@pytest.mark.parametrize("temperature", [-100.0, 20.0, 150.0])
def test_materials_without_a_thermal_model_keep_their_index(
    set_test_backend, temperature
):
    assert_allclose(MaterialAtConditions(IdealMaterial(1.5), temperature).n(0.55), 1.5)
    assert_allclose(
        MaterialAtConditions(IdealMaterial(1.5, 0.02), temperature).k(0.55), 0.02
    )
    abbe = AbbeMaterial(1.5, 60.0)
    assert_allclose(MaterialAtConditions(abbe, temperature).n(0.55), abbe.n(0.55))


def test_extinction_coefficient_is_forwarded(set_test_backend):
    glass = Material("N-BK7")
    wrapped = MaterialAtConditions(glass, 40.0)
    assert_allclose(wrapped.k(0.55), glass.k(0.55, temperature=40.0, pressure=1.0))


def test_tracing_at_the_reference_temperature_matches_the_original(set_test_backend):
    original = CookeTriplet()
    wrapped = CookeTriplet()
    replaced = 0
    for surface in wrapped.surface_group.surfaces:
        material = surface.material_post
        if isinstance(material, MaterialFile) and material._t0 is not None:
            surface.material_post = MaterialAtConditions(material, material._t0)
            replaced += 1
    assert replaced > 0

    expected = original.trace(0.0, 0.7, 0.55, num_rays=32)
    got = wrapped.trace(0.0, 0.7, 0.55, num_rays=32)
    for name in ("x", "y", "z", "L", "M", "N", "opd"):
        assert_allclose(getattr(got, name), getattr(expected, name))


@pytest.mark.parametrize(
    "material",
    [lambda: Material("N-BK7"), lambda: IdealMaterial(1.6, 0.01)],
    ids=["catalog", "ideal"],
)
def test_dict_round_trip(set_test_backend, material):
    wrapped = MaterialAtConditions(material(), 35.0, pressure=0.8)
    data = wrapped.to_dict()
    assert data["type"] == "MaterialAtConditions"
    assert data["temperature"] == 35.0
    assert data["pressure"] == 0.8
    for loaded in (
        MaterialAtConditions.from_dict(data),
        BaseMaterial.from_dict(data),
    ):
        assert isinstance(loaded, MaterialAtConditions)
        assert loaded.to_dict() == data
        assert loaded == wrapped
        assert_allclose(loaded.n(0.55), wrapped.n(0.55))


def test_from_dict_defaults_the_pressure(set_test_backend):
    data = MaterialAtConditions(IdealMaterial(1.5), 30.0).to_dict()
    del data["pressure"]
    assert MaterialAtConditions.from_dict(data).pressure == 1.0


def test_gradients_flow_through_the_wrapped_material(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Torch gradients")
    import torch

    ideal = IdealMaterial(1.5)
    ideal.index = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    wrapped = MaterialAtConditions(ideal, 40.0)
    for _ in range(2):  # a trainable parameter is never served from a cache
        wrapped.n(0.55).sum().backward()
        assert_allclose(ideal.index.grad, [1.0])
        ideal.index.grad = None

    glass = Material("N-BK7")
    wrapped = MaterialAtConditions(glass, 40.0)
    query = torch.tensor([0.5, 0.6], dtype=torch.float64, requires_grad=True)
    wrapped.n(query).sum().backward()
    got = query.grad.clone()
    query.grad = None
    glass.n(query, temperature=40.0, pressure=1.0).sum().backward()
    assert torch.all(got != 0)
    assert_allclose(got, query.grad)


def test_attributes_are_read_only():
    ideal = IdealMaterial(1.5)
    wrapped = MaterialAtConditions(ideal, 40.0, pressure=0.9)
    assert wrapped.material is ideal
    assert wrapped.temperature == 40.0
    assert wrapped.pressure == 0.9
    for name in ("material", "temperature", "pressure"):
        with pytest.raises(AttributeError):
            setattr(wrapped, name, None)


def test_display_name():
    assert (
        MaterialAtConditions(Material("N-BK7"), 40.0).display_name
        == "N-BK7 @ 40.0 °C, 1.00 atm"
    )


def test_thermal_expansion_is_forwarded(set_test_backend):
    glass = Material("N-BK7")
    wrapped = MaterialAtConditions(glass, 40.0)
    assert wrapped.thermal_expansion == glass.thermal_expansion == 7.1e-6

    ideal = IdealMaterial(1.5)
    assert MaterialAtConditions(ideal, 40.0).thermal_expansion is None
    assert ideal.thermal_expansion is None


@pytest.mark.parametrize(
    "temperature, pressure",
    [
        (-273.15, 1.0),
        (-300.0, 1.0),
        (float("nan"), 1.0),
        (20.0, -0.1),
        (20.0, float("nan")),
    ],
)
def test_invalid_conditions_are_refused(temperature, pressure):
    with pytest.raises(ValueError):
        MaterialAtConditions(IdealMaterial(1.5), temperature, pressure)


def test_a_zero_pressure_is_allowed():
    assert MaterialAtConditions(IdealMaterial(1.5), 20.0, pressure=0.0).pressure == 0.0


def test_only_a_material_can_be_wrapped():
    with pytest.raises(TypeError):
        MaterialAtConditions("N-BK7", 20.0)
