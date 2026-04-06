from __future__ import annotations

from importlib import resources
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

import optiland.backend as be
from optiland import materials
from optiland.materials.base import BaseMaterial
from optiland.optic import Optic

from .utils import assert_allclose


class TestBaseMaterial:
    def test_caching(self, set_test_backend):
        class DummyMaterial(BaseMaterial):
            def _calculate_n(self, wavelength, **kwargs):
                pass

            def _calculate_k(self, wavelength, **kwargs):
                pass

        material = DummyMaterial()
        material._calculate_n = MagicMock(return_value=1.5)

        # Test with scalar value
        result1 = material.n(0.5, temperature=25)
        assert result1 == 1.5
        material._calculate_n.assert_called_once_with(0.5, temperature=25)

        result2 = material.n(0.5, temperature=25)
        assert result2 == 1.5
        material._calculate_n.assert_called_once()

        # Test with numpy array
        wavelength_np = np.array([0.5, 0.6])
        material.n(wavelength_np, temperature=25)
        assert material._calculate_n.call_count == 2

        material.n(wavelength_np, temperature=25)
        assert material._calculate_n.call_count == 2

        # Test with torch tensor if backend is torch
        if set_test_backend == "torch":
            # a torch tensor with same values should be a cache hit
            wavelength_torch = be.asarray(np.array([0.5, 0.6]))
            material.n(wavelength_torch, temperature=25)
            assert material._calculate_n.call_count == 2

            # a torch tensor with different values should be a cache miss
            wavelength_torch_2 = be.asarray(np.array([0.7, 0.8]))
            material.n(wavelength_torch_2, temperature=25)
            assert material._calculate_n.call_count == 3

            # and a cache hit
            material.n(wavelength_torch_2, temperature=25)
            assert material._calculate_n.call_count == 3

    def test_detach_if_tensor_plain_value(self, set_test_backend):
        """_detach_if_tensor returns plain values unchanged."""
        assert BaseMaterial._detach_if_tensor(1.5) == 1.5
        assert BaseMaterial._detach_if_tensor(None) is None

    def test_large_array_cache_key_runtime_is_sublinear(self, set_test_backend):
        """Regression guard for O(N) array cache-key construction.

        Large wavelength arrays should use metadata-based keys rather than
        materializing every array element into a Python tuple.
        """

        class DummyMaterial(BaseMaterial):
            def _calculate_n(self, wavelength, **kwargs):
                return wavelength

            def _calculate_k(self, wavelength, **kwargs):
                return wavelength

        material = DummyMaterial()

        small = np.linspace(0.45, 0.65, 1_000, dtype=np.float64)
        large = np.linspace(0.45, 0.65, 200_000, dtype=np.float64)

        # Warm up once to reduce one-time effects.
        material._create_cache_key(small, temperature=25)
        material._create_cache_key(large, temperature=25)

        small_iters = 200
        t0 = time.perf_counter()
        for _ in range(small_iters):
            material._create_cache_key(small, temperature=25)
        small_avg_s = (time.perf_counter() - t0) / small_iters

        large_iters = 5
        t0 = time.perf_counter()
        for _ in range(large_iters):
            material._create_cache_key(large, temperature=25)
        large_avg_s = (time.perf_counter() - t0) / large_iters

        ratio = large_avg_s / max(small_avg_s, 1e-9)
        assert ratio < 40.0, (
            "Large-array cache-key generation appears to scale linearly with array size "
            f"(ratio={ratio:.2f}, small_avg={small_avg_s:.3e}s, large_avg={large_avg_s:.3e}s)"
        )

        # Keys should still be deterministic for repeated calls on same input.
        key1 = material._create_cache_key(large, temperature=25)
        key2 = material._create_cache_key(large, temperature=25)
        assert key1 == key2

    def test_detach_if_tensor_numpy(self, set_test_backend):
        """_detach_if_tensor returns numpy arrays unchanged."""
        arr = np.array([1.5, 1.6])
        result = BaseMaterial._detach_if_tensor(arr)
        np.testing.assert_array_equal(result, arr)

    def test_requires_grad_plain_value(self, set_test_backend):
        """_requires_grad returns False for plain values."""
        assert BaseMaterial._requires_grad(1.5) is False
        assert BaseMaterial._requires_grad(np.array(1.5)) is False


@pytest.mark.skipif(
    "torch" not in be.list_available_backends(),
    reason="PyTorch not installed",
)
class TestBaseMaterialTorchCaching:
    """Tests for material caching behavior specific to torch backend.

    These tests verify:
    - Cached tensors are detached (no stale computation graph)
    - Repeated forward+backward passes don't raise RuntimeError
    - Cache is bypassed when the result requires grad (optimizable index)
    """

    @pytest.fixture(autouse=True)
    def _setup_torch(self):
        be.set_backend("torch")
        be.set_device("cpu")
        be.grad_mode.enable()
        be.set_precision("float64")
        yield
        be.set_backend("numpy")

    def test_detach_if_tensor_torch(self):
        """_detach_if_tensor detaches torch tensors."""
        import torch

        t = torch.tensor(1.5, requires_grad=True) * 2.0  # has grad_fn
        result = BaseMaterial._detach_if_tensor(t)
        assert not result.requires_grad
        assert result.grad_fn is None

    def test_requires_grad_torch(self):
        """_requires_grad correctly detects torch tensors with grad."""
        import torch

        leaf = torch.tensor(1.5, requires_grad=True)
        assert BaseMaterial._requires_grad(leaf) is True

        no_grad = torch.tensor(1.5)
        assert BaseMaterial._requires_grad(no_grad) is False

    def test_cached_value_is_detached_when_no_grad(self):
        """Cached n/k values must be detached when they don't require grad.

        When grad_mode is off, Sellmeier formula results are torch tensors
        without requires_grad. These should be cached and detached to prevent
        stale computation graph references if grad_mode is later re-enabled.
        """
        import torch

        be.grad_mode.disable()
        try:
            mat = materials.Material("N-BK7")
            result = mat.n(0.5876)

            # Should be cached since result doesn't require grad
            assert len(mat._n_cache) > 0, "Result should be cached"

            cached = list(mat._n_cache.values())[0]
            if isinstance(cached, torch.Tensor):
                assert cached.grad_fn is None, (
                    "Cached tensor should be detached (no grad_fn)"
                )
        finally:
            be.grad_mode.enable()

    def test_cache_bypassed_for_real_material_under_grad_mode(self):
        """Under torch + grad_mode, real material results require_grad=True
        because be.array() creates grad-enabled tensors. The cache must be
        bypassed in this case.
        """
        mat = materials.Material("N-BK7")
        mat.n(0.5876)

        assert len(mat._n_cache) == 0, (
            "Cache should be empty — Sellmeier result requires_grad under "
            "grad_mode, so caching must be skipped"
        )

    def test_no_runtime_error_on_repeated_backward(self):
        """Repeated forward+backward passes must not raise RuntimeError.

        This is the regression test for the 'backward through the graph a
        second time' error. We simulate a multi-step optimization by calling
        material.n() repeatedly and computing a loss that flows through it.
        """
        import torch

        mat = materials.Material("N-BK7")

        # Simulate multiple optimizer steps
        param = torch.nn.Parameter(torch.tensor(0.55, dtype=torch.float64))

        for _ in range(3):
            # Forward: compute n at a wavelength that depends on param
            n_val = mat.n(param)
            loss = (n_val - 1.5) ** 2
            loss.backward()

            with torch.no_grad():
                param.data -= 0.01 * param.grad
                param.grad.zero_()

    def test_cache_bypassed_when_result_requires_grad(self):
        """n() must NOT cache when the result requires_grad.

        If the refractive index itself is an optimization variable (e.g.
        IdealMaterial with a nn.Parameter index), caching would break
        gradient flow — grad(loss, n) would always be zero.
        """
        import torch

        # Create an IdealMaterial whose index is a nn.Parameter
        mat = materials.IdealMaterial(n=1.5)
        mat.index = torch.nn.Parameter(torch.tensor([1.5], dtype=torch.float64))

        # First call — result requires grad, should NOT be cached
        result1 = mat.n(0.55)
        assert result1.requires_grad, "Result should require grad"
        assert len(mat._n_cache) == 0, "Should not cache requires_grad result"

        # Second call — should recompute (not return stale cached value)
        result2 = mat.n(0.55)
        assert result2.requires_grad, "Second result should also require grad"
        assert len(mat._n_cache) == 0, "Cache should still be empty"

        # Verify gradient actually flows through
        loss = result2.sum()
        loss.backward()
        assert mat.index.grad is not None, "Gradient should flow to index"
        assert mat.index.grad.abs().sum() > 0, "Gradient should be non-zero"

    def test_cache_bypassed_for_k_when_requires_grad(self):
        """k() has the same bypass logic as n()."""
        import torch

        mat = materials.IdealMaterial(n=1.5, k=0.1)
        mat.absorp = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float64))

        result = mat.k(0.55)
        assert result.requires_grad
        assert len(mat._k_cache) == 0

        loss = result.sum()
        loss.backward()
        assert mat.absorp.grad is not None


def build_model(material: BaseMaterial):
    lens = Optic()

    lens.fields.set_type("angle")
    lens.fields.add(0, 0)
    lens.wavelengths.add(0.550)
    lens.set_aperture("EPD", 2)

    lens.surfaces.add(index=0)
    lens.surfaces.add(index=1, material=material, radius=10, thickness=10, is_stop=True)
    lens.surfaces.add(index=2, radius=-10, thickness=10)
    lens.surfaces.add(index=3, radius=np.inf)

    return lens


class TestIdealMaterial:
    def test_ideal_material_n(self, set_test_backend):
        material = materials.IdealMaterial(n=1.5)
        assert material.n(0.5) == 1.5
        assert material.n(1.0) == 1.5
        assert material.n(2.0) == 1.5
        assert material.k(2.0) == 0.0

    def test_ideal_material_k(self, set_test_backend):
        material = materials.IdealMaterial(n=1.5, k=0.2)
        assert material.k(0.5) == 0.2
        assert material.k(1.0) == 0.2
        assert material.k(2.0) == 0.2

    def test_ideal_to_dict(self, set_test_backend):
        material = materials.IdealMaterial(n=1.5, k=0.2)
        assert material.to_dict() == {
            "index": 1.5,
            "absorp": 0.2,
            "type": materials.IdealMaterial.__name__,
            "propagation_model": {"class": "HomogeneousPropagation"},
        }

    def test_ideal_from_dict(self, set_test_backend):
        material = materials.IdealMaterial.from_dict(
            {"index": 1.5, "absorp": 0.2, "type": materials.IdealMaterial.__name__},
        )
        assert material.n(0.5) == 1.5
        assert material.k(0.5) == 0.2

    def test_ray_trace(self, set_test_backend):
        material = materials.IdealMaterial(n=1.5, k=0.0)
        lens = build_model(material)

        lens.trace(Hx=0, Hy=0, wavelength=0.55)


class TestAbbeMaterial:
    def test_refractive_index(self, set_test_backend):
        abbe_material = materials.AbbeMaterial(n=1.5, abbe=50, model="polynomial")
        wavelength = 0.58756  # in microns
        value = abbe_material.n(wavelength)
        assert_allclose(value, 1.4999167964912952)

    def test_extinction_coefficient(self, set_test_backend):
        abbe_material = materials.AbbeMaterial(n=1.5, abbe=50, model="polynomial")
        wavelength = 0.58756  # in microns
        assert abbe_material.k(wavelength) == 0

    def test_coefficients(self, set_test_backend):
        abbe_material = materials.AbbeMaterial(n=1.5, abbe=50, model="polynomial")
        # Access the private method on the underlying model (AbbePolynomialModel)
        coefficients = abbe_material.model._get_coefficients()
        assert coefficients.shape == (4,)  # Assuming the polynomial is of degree 3

    def test_abbe_to_dict(self, set_test_backend):
        abbe_material = materials.AbbeMaterial(n=1.5, abbe=50, model="polynomial")
        abbe_dict = abbe_material.to_dict()
        assert abbe_dict == {
            "type": "AbbeMaterial",
            "index": 1.5,
            "abbe": 50.0,
            "model": "polynomial",
            "propagation_model": {"class": "HomogeneousPropagation"},
        }

    def test_abbe_from_dict(self, set_test_backend):
        # Legacy dict without model= key triggers FutureWarning; explicitly consume it.
        abbe_dict = {"type": "AbbeMaterial", "index": 1.5, "abbe": 50}
        with pytest.warns(FutureWarning, match="default model for AbbeMaterial"):
            abbe_material = materials.BaseMaterial.from_dict(abbe_dict)
        assert abbe_material.index == 1.5
        assert abbe_material.abbe == 50

    def test_abbe_out_of_bounds_wavelength(self, set_test_backend):
        abbe_material = materials.AbbeMaterial(n=1.5, abbe=50, model="polynomial")
        with pytest.raises(ValueError):
            abbe_material.n(0.3)
        with pytest.raises(ValueError):
            abbe_material.n(0.8)

    def test_ray_trace(self, set_test_backend):
        abbe_material = materials.AbbeMaterial(n=1.5, abbe=50, model="polynomial")
        lens = build_model(abbe_material)

        lens.trace(Hx=0, Hy=0, wavelength=0.55)


class TestMaterialFile:
    def test_formula_1(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/ami/AMTIR-3.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(4), 2.6208713861212907)
        assert_allclose(material.n(6), 2.6144067565243265)
        assert_allclose(material.n(8), 2.6087270552683854)

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_2(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/schott/BAFN6.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.6111748495969627)
        assert_allclose(material.n(0.8), 1.5803913968709888)
        assert_allclose(material.n(1.2), 1.573220342181897)
        assert_allclose(material.k(0.56), 1.3818058823529405e-08)
        assert_allclose(material.k(0.88), 1.18038e-08)
        assert_allclose(material.abbe(), 48.44594399734635)

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_3(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/hikari/BASF6.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.6970537915318815)
        assert_allclose(material.n(0.5), 1.6767571448173404)
        assert_allclose(material.n(0.6), 1.666577226760647)
        assert_allclose(material.k(0.4), 3.3537e-07)
        assert_allclose(material.k(0.5), 2.3945e-08)
        assert_allclose(material.k(0.6), 1.4345e-08)
        assert_allclose(material.abbe(), 42.00944974180074)

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_4(self, set_test_backend):
        rel_file = "data-nk/main/CaGdAlO4/Loiko-o.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.9829612788706874)
        assert_allclose(material.n(0.6), 1.9392994674994937)
        assert_allclose(material.n(1.5), 1.9081487808757178)
        assert_allclose(material.abbe(), 40.87771013627357)

        # This material has no k values, check that it raises a warning
        assert material.k(1.0) == 0.0

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_5(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/main/YbF3/Amotchkina.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.5874342875)
        assert_allclose(material.n(1.0), 1.487170596)
        assert_allclose(material.n(5.0), 1.4844954023999999)
        assert_allclose(material.k(10), 0.004800390585878816)
        assert_allclose(material.k(11), 0.016358499999999998)
        assert_allclose(material.k(12), 0.032864500000000005)
        assert_allclose(material.abbe(), 15.36569851094505)

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_6(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/main/CO2/Bideau-Mehu.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.0004592281255849)
        assert_allclose(material.n(1.0), 1.0004424189669583)
        assert_allclose(material.n(1.5), 1.0004386003514163)
        assert_allclose(material.abbe(), 76.08072467952312)

        # This material has no k values, check that it raises a warning
        assert material.k(1.0) == 0.0

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_7(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/main/Y2O3/Nigara.yml",
            ),
        )
        # No material in the database currently uses formula 7, so we fake it
        material = materials.MaterialFile(filename)
        material._n_formula = "formula 7"
        material.coefficients = [1.0, 0.58, 0.12, 0.87, 0.21, 0.81]

        # We test only the equations. These values are meaningless.
        assert_allclose(material.n(0.4), 12.428885495537186)
        assert_allclose(material.n(1.0), 3.6137209774932684)
        assert_allclose(material.n(1.5), 13.532362213339358)
        assert_allclose(material.abbe(), 1.0836925045533496)

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_8(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/main/AgBr/Schroter.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.5), 2.3094520454859557)
        assert_allclose(material.n(0.55), 2.275584479878346)
        assert_allclose(material.n(0.65), 2.237243954654548)
        assert_allclose(material.abbe(), 14.551572168536392)

        # This material has no k values, check that it raises a warning
        assert material.k(1.0) == 0.0

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_9(self, set_test_backend):
        rel_file = "data-nk/organic/CH4N2O - urea/Rosker-e.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.3), 1.7043928702073146)
        assert_allclose(material.n(0.6), 1.605403788031452)
        assert_allclose(material.n(1.0), 1.5908956870937045)
        assert_allclose(material.abbe(), 34.60221948120884)

        # This material has no k values, check that it raises a warning
        assert material.k(1.0) == 0.0

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_tabulated_n(self, set_test_backend):
        rel_file = "data-nk/main/Y3Al5O12/Bond.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(1.0), 1.8197)
        assert_allclose(material.n(2.0), 1.8035)
        assert_allclose(material.n(3.0), 1.7855)
        assert_allclose(material.abbe(), 52.043469741225195)

        # This material has no k values, check that it raises a warning
        assert material.k(1.0) == 0.0

        # Test case when no tabulated data available
        material._n = None
        with pytest.raises((ValueError, TypeError)):
            material._calculate_n(1.0)

    def test_tabulated_nk(self, set_test_backend):
        rel_file = "data-nk/main/B/Fernandez-Perea.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.005), 0.9947266437313135)
        assert_allclose(material.n(0.02), 0.9358854820031199)
        assert_allclose(material.n(0.15), 1.990336423662574)
        assert_allclose(material.k(0.005), 0.0038685437228138607)
        assert_allclose(material.k(0.02), 0.008158161793528261)
        assert_allclose(material.k(0.15), 1.7791319513647896)

    def test_set_formula_type_twice(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/ami/AMTIR-3.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        with pytest.raises(ValueError):
            material._set_formula_type("formula 2")

    def test_to_dict(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/ami/AMTIR-3.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert material.to_dict() == {
            "filename": filename,
            "type": materials.MaterialFile.__name__,
            "propagation_model": {"class": "HomogeneousPropagation"},
        }

    def test_from_dict(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/ami/AMTIR-3.yml",
            ),
        )
        material_dict = {"filename": filename, "type": materials.MaterialFile.__name__}
        assert materials.MaterialFile.from_dict(material_dict).filename == filename


class TestMaterial:
    def test_standard_material(self, set_test_backend):
        material = materials.Material("N-BK7")
        assert material.n(0.5) == 1.5214144757734767
        assert material.k(0.5) == 9.5781e-09
        assert material.abbe() == 64.1673362374998

    def test_nonexistent_material(self, set_test_backend):
        with pytest.raises(ValueError):
            materials.Material("nonexistent material")

        with pytest.raises(ValueError):
            materials.Material(
                "nonexistent material",
                reference="it really does not exist",
            )

    def test_non_robust_failure(self, set_test_backend):
        # There are many materials matches for BK7. Without robust search,
        # this should fail.
        with pytest.raises(ValueError):
            materials.Material("BK7", robust_search=False)

        # There are also many materials matches for BK7 with schott reference.
        with pytest.raises(ValueError):
            materials.Material("BK7", reference="schott", robust_search=False)

    def test_min_wavelength_filtering(self, set_test_backend):
        material = materials.Material("SF11", min_wavelength=2.0)
        df = material._load_dataframe()
        df_filtered = material._find_material_matches(df)

        # Check that all materials have wavelength ranges including 2.0 µm
        assert np.all(df_filtered["max_wavelength"] >= 2.0)
        assert np.all(df_filtered["min_wavelength"] <= 2.0)

    def test_max_wavelength_filtering(self, set_test_backend):
        material = materials.Material("SF11", max_wavelength=2.0)
        df = material._load_dataframe()
        df_filtered = material._find_material_matches(df)

        # Check that all materials have wavelength ranges including 2.0 µm
        assert np.all(df_filtered["max_wavelength"] >= 2.0)
        assert np.all(df_filtered["min_wavelength"] <= 2.0)

    def test_raise_material_error_method(self, set_test_backend):
        material = materials.Material("SF11")
        with pytest.raises(ValueError):
            material._raise_material_error(no_matches=False, multiple_matches=False)

        # Confirm error raise when wavelength ranges are passed
        material = materials.Material("SF11", min_wavelength=0.5, max_wavelength=0.7)
        with pytest.raises(ValueError):
            material._raise_material_error(no_matches=False, multiple_matches=False)

    def test_to_dict(self, set_test_backend):
        material = materials.Material("SF11")
        mat_dict = material.to_dict()
        assert mat_dict == {
            "type": "Material",
            "filename": material.filename,
            "name": "SF11",
            "reference": None,
            "robust_search": True,
            "min_wavelength": None,
            "max_wavelength": None,
            "propagation_model": {"class": "HomogeneousPropagation"},
        }

    def test_from_dict(self, set_test_backend):
        material_dict = {"name": "SF11", "type": materials.Material.__name__}
        assert materials.Material.from_dict(material_dict).name == "SF11"

    def test_raise_warning(self, set_test_backend):
        materials.Material("LITHOTEC-CAF2")  # prints a warning


def test_glasses_selection(set_test_backend):
    glasses = materials.glasses_selection(0.3, 2.5, catalogs=["schott"])
    expected_glasses = [
        "FK3",
        "FK5HTi",
        "K10",
        "LITHOTEC-CAF2",
        "N-BAK1",
        "N-BAK2",
        "N-BK10",
        "N-BK7",
        "N-BK7HT",
        "N-BK7HTi",
        "N-FK5",
        "N-FK51",
        "N-FK51A",
        "N-FK58",
        "N-LAK33B",
        "N-LAK34",
        "N-LAK7",
        "N-PK51",
        "N-PK52A",
        "N-PSK3",
        "N-SK11",
        "N-SK5",
        "N-ZK7",
        "N-ZK7A",
        "P-LAK35",
        "P-SK60",
    ]
    assert glasses == expected_glasses


def test_get_nd_vd(set_test_backend):
    assert materials.get_nd_vd(glass="N-BK7") == (1.5168, 64.17)


def test_downsample_glass_map(set_test_backend):
    glass_dict = {g: materials.get_nd_vd(g) for g in ["N-BK7", "FK3", "FK5HTi", "K10"]}
    downsampled_glass_dict = materials.downsample_glass_map(
        glass_dict,
        num_glasses_to_keep=3,
    )
    expected_downsample_glass_dict = {
        "K10": (1.50137, 56.41),
        "N-BK7": (1.5168, 64.17),
        "FK5HTi": (1.48748, 70.47),
    }
    assert downsampled_glass_dict == expected_downsample_glass_dict


def test_find_closest_glass(set_test_backend):
    assert (
        materials.find_closest_glass(
            nd_vd=(1.5168, 64.17), catalog=["N-BK7", "F5", "SF5"]
        )
        == "N-BK7"
    )


def test_plot_nk():
    from matplotlib.figure import Figure

    mat = materials.Material("BK7")
    fig, axes = materials.plot_nk(mat, wavelength_range=(0.1, 15))
    assert fig is not None
    assert isinstance(fig, Figure)
    assert isinstance(axes, tuple)
    assert len(axes) == 2


def test___eq__():
    ideal1 = materials.IdealMaterial(1.0, 0.0)
    ideal2 = materials.IdealMaterial(1.0, 0.0)
    ideal3 = materials.IdealMaterial(1.0, 0.1)
    abbe = materials.AbbeMaterial(1.5, 60.0, model="polynomial")
    assert ideal1 == ideal2
    assert ideal1 != ideal3
    assert abbe != ideal1
