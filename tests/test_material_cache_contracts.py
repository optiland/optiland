"""Material cache regressions across mutable data and execution contexts."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial, MaterialFile
from optiland.materials.base import BaseMaterial

from .utils import assert_allclose


class SpectralMaterial(BaseMaterial):
    """An immutable model with an independently known wavelength derivative."""

    def _cache_state(self):
        return ()

    def _calculate_n(self, wavelength, **kwargs):
        query = wavelength if hasattr(wavelength, "shape") else be.asarray(wavelength)
        return 1.2 + 0.5 * query

    def _calculate_k(self, wavelength, **kwargs):
        query = wavelength if hasattr(wavelength, "shape") else be.asarray(wavelength)
        return 0.01 * query


@pytest.mark.parametrize("property_name", ["n", "k"])
def test_cache_preserves_query_shape(set_test_backend, property_name):
    material = SpectralMaterial()
    evaluate = getattr(material, property_name)
    values = be.asarray([0.4, 0.5, 0.6, 0.7])
    flat = evaluate(values)
    matrix = evaluate(values.reshape(2, 2))
    assert tuple(matrix.shape) == (2, 2)
    assert_allclose(matrix.reshape(-1), flat)


@pytest.mark.parametrize("property_name", ["n", "k"])
def test_cache_preserves_query_dtype(set_test_backend, property_name):
    material = SpectralMaterial()
    evaluate = getattr(material, property_name)
    if be.get_backend() == "torch":
        import torch

        wide = torch.tensor([0.5, 0.75], dtype=torch.float64)
        narrow = wide.to(torch.float32)
    else:
        wide = np.array([0.5, 0.75], dtype=np.float64)
        narrow = wide.astype(np.float32)
    evaluate(wide)
    assert evaluate(narrow).dtype == narrow.dtype


@pytest.mark.parametrize("property_name", ["n", "k"])
def test_scalar_cache_follows_backend_precision(set_test_backend, property_name):
    material = SpectralMaterial()
    evaluate = getattr(material, property_name)
    original_precision = be.get_precision()
    try:
        be.set_precision("float64")
        wide = evaluate(0.5)
        be.set_precision("float32")
        narrow = evaluate(0.5)
        assert wide.dtype != narrow.dtype
        assert "32" in str(narrow.dtype)
    finally:
        be.set_precision(f"float{original_precision}")


@pytest.mark.parametrize("property_name", ["n", "k"])
def test_scalar_cache_follows_backend(set_test_backend, property_name):
    if "torch" not in be.list_available_backends():
        pytest.skip("requires both backends")
    import torch

    material = SpectralMaterial()
    evaluate = getattr(material, property_name)
    be.set_backend("numpy")
    assert not isinstance(evaluate(0.5), torch.Tensor)
    be.set_backend("torch")
    be.set_device("cpu")
    assert isinstance(evaluate(0.5), torch.Tensor)
    be.set_backend("numpy")
    assert not isinstance(evaluate(0.5), torch.Tensor)


@pytest.mark.parametrize("uniform", [True, False])
@pytest.mark.parametrize("property_name", ["n", "k"])
def test_large_query_mutation(set_test_backend, uniform, property_name):
    material = SpectralMaterial()
    values = np.full(2048, 0.5) if uniform else np.linspace(0.4, 0.7, 2048)
    query = be.asarray(values)
    evaluate = getattr(material, property_name)
    evaluate(query)
    query[7] = 0.65
    expected = 1.2 + 0.5 * query if property_name == "n" else 0.01 * query
    assert_allclose(evaluate(query), expected)


@pytest.mark.parametrize("property_name", ["n", "k"])
def test_tensor_query_mutated_through_numpy_alias(set_test_backend, property_name):
    if be.get_backend() != "torch":
        pytest.skip("Torch/NumPy shared storage")
    import torch

    values = np.full(2048, 0.5)
    query = torch.from_numpy(values)
    material = SpectralMaterial()
    evaluate = getattr(material, property_name)
    evaluate(query)
    values[7] = 0.65  # A foreign write does not increment tensor._version.
    expected = 1.2 + 0.5 * query if property_name == "n" else 0.01 * query
    assert_allclose(evaluate(query), expected)


@pytest.mark.parametrize("property_name", ["n", "k"])
def test_inference_tensor_mutation(set_test_backend, property_name):
    if be.get_backend() != "torch":
        pytest.skip("Torch inference tensors")
    import torch

    material = SpectralMaterial()
    evaluate = getattr(material, property_name)
    with torch.inference_mode():
        query = torch.full((2048,), 0.5, dtype=torch.float64)
        evaluate(query)
        query[7] = 0.65
        expected = 1.2 + 0.5 * query if property_name == "n" else 0.01 * query
        assert_allclose(evaluate(query), expected)


@pytest.mark.parametrize("property_name, derivative", [("n", 0.5), ("k", 0.01)])
@pytest.mark.parametrize("size", [4, 2048])
def test_cached_constant_then_trainable_query(
    set_test_backend, property_name, derivative, size
):
    if be.get_backend() != "torch":
        pytest.skip("Torch gradients")
    import torch

    material = SpectralMaterial()
    evaluate = getattr(material, property_name)
    query = torch.full((size,), 0.5, dtype=torch.float64)
    evaluate(query)
    query.requires_grad_()
    for _ in range(2):
        result = evaluate(query)
        assert result.requires_grad
        result.sum().backward()
        assert_allclose(query.grad, np.full(size, derivative))
        query.grad = None


@pytest.mark.parametrize("property_name, attribute", [("n", "index"), ("k", "absorp")])
def test_ideal_parameter_mutation(set_test_backend, property_name, attribute):
    material = IdealMaterial(1.5, 0.01)
    setattr(material, attribute, be.asarray([1.5 if property_name == "n" else 0.01]))
    evaluate = getattr(material, property_name)
    evaluate(0.5)
    value = 1.7 if property_name == "n" else 0.02
    getattr(material, attribute)[...] = value
    assert_allclose(evaluate(0.5), value)


@pytest.mark.parametrize("property_name, attribute", [("n", "index"), ("k", "absorp")])
def test_cached_constant_then_trainable_parameter(
    set_test_backend, property_name, attribute
):
    if be.get_backend() != "torch":
        pytest.skip("Torch gradients")
    import torch

    material = IdealMaterial(1.5, 0.01)
    parameter = torch.tensor([1.5], dtype=torch.float64)
    setattr(material, attribute, parameter)
    evaluate = getattr(material, property_name)
    evaluate(0.5)
    parameter.requires_grad_()
    for _ in range(2):
        evaluate(0.5).sum().backward()
        assert_allclose(parameter.grad, [1.0])
        parameter.grad = None


def test_file_coefficient_mutation(set_test_backend, tmp_path):
    path = tmp_path / "constant.yml"
    path.write_text(
        'DATA:\n- type: formula 5\n  wavelength_range: "0.4 0.7"\n'
        '  coefficients: "1.5 0.1 2"\n',
        encoding="utf-8",
    )
    material = MaterialFile(str(path))
    material.coefficients = be.asarray([[1.5], [0.1], [2.0]])
    material.n(0.5)
    material.coefficients[0, 0] = 1.7
    assert_allclose(material.n(0.5), 1.725)


@pytest.mark.parametrize("property_name", ["n", "k"])
def test_empty_sequence_queries_preserve_shape(set_test_backend, property_name):
    evaluate = getattr(SpectralMaterial(), property_name)
    for query, shape in (([], (0,)), ([[], []], (2, 0)), ([], (0,))):
        for _ in range(2):
            result = evaluate(query)
            assert tuple(result.shape) == shape
            assert be.to_numpy(result).size == 0
    expected = 1.45 if property_name == "n" else 0.005
    assert_allclose(evaluate(0.5), expected)


@pytest.mark.parametrize("property_name", ["n", "k"])
def test_cache_hook_can_delegate_to_uncached_default(set_test_backend, property_name):
    class OptionalCacheMaterial(BaseMaterial):
        cache_enabled = True
        offset = 0.0

        def _cache_state(self):
            if self.cache_enabled:
                return (self.offset,)
            return super()._cache_state()

        def _calculate_n(self, wavelength, **kwargs):
            return 1.5 + self.offset

        def _calculate_k(self, wavelength, **kwargs):
            return 0.01 + self.offset

    material = OptionalCacheMaterial()
    evaluate = getattr(material, property_name)
    baseline = 1.5 if property_name == "n" else 0.01
    assert_allclose(evaluate(0.5), baseline)
    material.cache_enabled = False
    for offset in (0.1, 0.2):
        material.offset = offset
        assert_allclose(evaluate(0.5), baseline + offset)
    material.cache_enabled = True
    assert_allclose(evaluate(0.5), baseline + 0.2)


@pytest.mark.parametrize("property_name", ["n", "k"])
def test_mutable_mapping_environment_is_not_cached(set_test_backend, property_name):
    class CalibratedMaterial(SpectralMaterial):
        def _cache_state(self):
            return ()

        def _calculate_n(self, wavelength, **kwargs):
            return super()._calculate_n(wavelength) + kwargs["calibration"]["offset"]

        def _calculate_k(self, wavelength, **kwargs):
            return super()._calculate_k(wavelength) + kwargs["calibration"]["offset"]

    evaluate = getattr(CalibratedMaterial(), property_name)
    calibration = {"offset": 0.0}
    baseline = 1.45 if property_name == "n" else 0.005
    assert_allclose(evaluate(0.5, calibration=calibration), baseline)
    for offset in (0.1, 0.2):
        calibration["offset"] = offset
        assert_allclose(evaluate(0.5, calibration=calibration), baseline + offset)


def test_extinction_only_file_tracks_live_table(set_test_backend, tmp_path):
    path = tmp_path / "extinction.yml"
    path.write_text(
        "DATA:\n- type: tabulated k\n  data: |\n    0.4 0.01\n    0.6 0.03\n",
        encoding="utf-8",
    )
    material = MaterialFile(str(path))
    assert_allclose(material.k(0.5), 0.02)
    material._k[...] = be.asarray([0.03, 0.05])
    assert_allclose(material.k(0.5), 0.04)


def test_custom_file_formula_tracks_closure_state(set_test_backend, tmp_path):
    path = tmp_path / "constant.yml"
    path.write_text(
        'DATA:\n- type: formula 5\n  coefficients: "1.5"\n', encoding="utf-8"
    )
    material = MaterialFile(str(path))
    material.coefficients = be.asarray([[1.5]])
    original_formula = material.formula_map["formula 5"]
    assert_allclose(material.n(0.5), 1.5)
    calibration = {"offset": 0.1}

    def calibrated_formula(wavelength):
        return 1.5 + 0.1 * be.asarray(wavelength) + calibration["offset"]

    material.formula_map["formula 5"] = calibrated_formula
    assert_allclose(material.n(0.5), 1.65)
    calibration["offset"] = 0.2
    assert_allclose(material.n(0.5), 1.75)
    material.formula_map["formula 5"] = original_formula
    assert_allclose(material.n(0.5), 1.5)


def test_untracked_custom_state_is_not_cached(set_test_backend):
    class MutableMaterial(BaseMaterial):
        index = 1.5

        def _calculate_n(self, wavelength, **kwargs):
            return self.index

        def _calculate_k(self, wavelength, **kwargs):
            return 0.0

    material = MutableMaterial()
    assert material.n(0.5) == 1.5
    material.index = 1.7
    assert material.n(0.5) == 1.7


def test_builtin_subclass_must_track_its_additional_state(set_test_backend):
    # not named OffsetMaterial: subclasses register by class name, and this
    # local class would replace optiland.materials.OffsetMaterial there
    class ShiftedIdealMaterial(IdealMaterial):
        offset = 0.0

        def _calculate_n(self, wavelength, **kwargs):
            return super()._calculate_n(wavelength, **kwargs) + self.offset

    material = ShiftedIdealMaterial(1.5)
    material.index = be.asarray([1.5])
    material.absorp = be.asarray([0.0])
    material.n(0.5)
    material.offset = 0.2
    assert_allclose(material.n(0.5), 1.7)


@pytest.mark.parametrize("property_name", ["n", "k"])
def test_array_environment_and_gradients(set_test_backend, property_name):
    class EnvironmentalMaterial(SpectralMaterial):
        def _cache_state(self):
            return ()

        def _calculate_n(self, wavelength, **kwargs):
            return super()._calculate_n(wavelength) + kwargs["temperature"] * 0.001

        def _calculate_k(self, wavelength, **kwargs):
            return super()._calculate_k(wavelength) + kwargs["temperature"] * 0.0001

    material = EnvironmentalMaterial()
    evaluate = getattr(material, property_name)
    temperature = be.asarray([20.0, 25.0])
    initial = evaluate(0.5, temperature=temperature)
    temperature[0] = 30.0
    updated = evaluate(0.5, temperature=temperature)
    slope = 0.001 if property_name == "n" else 0.0001
    assert_allclose(updated - initial, [10 * slope, 0.0])
    waves = be.asarray(np.full(2048, 0.5))
    temperatures = be.asarray(np.linspace(20.0, 30.0, 2048))
    base = 1.45 if property_name == "n" else 0.005
    assert_allclose(
        evaluate(waves, temperature=temperatures), base + slope * temperatures
    )
    if be.get_backend() == "torch":
        temperature.requires_grad_()
        evaluate(0.5, temperature=temperature).sum().backward()
        assert_allclose(temperature.grad, [slope, slope])


@pytest.mark.parametrize("material_kind", ["ideal", "formula", "table"])
def test_live_parameters_across_backend_changes(
    set_test_backend, tmp_path, material_kind
):
    if "torch" not in be.list_available_backends():
        pytest.skip("requires both backends")
    import torch

    initial_backend = be.get_backend()
    if material_kind == "ideal":
        material = IdealMaterial(1.5, 0.01)
        parameter = material.index
        expected = 1.5
    else:
        path = tmp_path / "sample.yml"
        definition = (
            'type: formula 5\n  coefficients: "1.5 0.1 2"\n'
            if material_kind == "formula"
            else "type: tabulated nk\n  data: |\n    0.4 1.5 0.01\n    0.6 1.6 0.02\n"
        )
        path.write_text("DATA:\n- " + definition, encoding="utf-8")
        material = MaterialFile(str(path))
        parameter = material.coefficients if material_kind == "formula" else material._n
        expected = 1.525 if material_kind == "formula" else 1.55
    assert_allclose(material.n(0.5), expected)
    other_backend = "numpy" if initial_backend == "torch" else "torch"
    for backend in (other_backend, initial_backend):
        be.set_backend(backend)
        if backend == "torch":
            be.set_device("cpu")
        be.set_precision("float64")
        result = material.n(0.5)
        assert isinstance(result, torch.Tensor) == (backend == "torch")
        assert_allclose(result, expected)
        current = (
            material.index
            if material_kind == "ideal"
            else material.coefficients
            if material_kind == "formula"
            else material._n
        )
        assert current is parameter
    # File tables use non-trainable asarray data; only existing trainable
    # parameters promise an attached graph across a backend round trip.
    if initial_backend == "torch" and parameter.requires_grad:
        material.n(0.5).sum().backward()
        assert parameter.grad is not None
