"""Explicit table bounds preserve optical data, persistence and derivatives."""

from __future__ import annotations

import json

import pytest

import optiland.backend as be
from optiland.materials import BaseMaterial, DataMaterial, Material, MaterialFile
from tests.utils import assert_allclose


@pytest.fixture
def material_factory(tmp_path, monkeypatch):
    path = tmp_path / "samples.yml"
    path.write_text(
        "DATA:\n- type: tabulated n\n  data: |\n    0.4 1.6\n    0.8 1.4\n"
        "- type: tabulated k\n  data: |\n    0.5 0.01\n    0.7 0.03\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Material, "_retrieve_file", lambda self: (str(path), {}))

    def make(kind, **kwargs):
        if kind == "data":
            return DataMaterial.from_samples(
                [0.4, 0.8],
                [1.6, 1.4],
                extinction={
                    "kind": "tabulated_k",
                    "wavelengths_um": [0.5, 0.7],
                    "values": [0.01, 0.03],
                },
                **kwargs,
            )
        if kind == "file":
            return MaterialFile(str(path), **kwargs)
        return Material("test glass", catalog="test", **kwargs)

    return make


@pytest.mark.parametrize(
    "kind,default", [("data", "raise"), ("file", "clamp"), ("catalog", "clamp")]
)
def test_default_bounds_remain_unchanged(
    kind, default, material_factory, set_test_backend
):
    material = material_factory(kind)
    assert material.bounds == default
    if default == "raise":
        with pytest.raises(ValueError, match="range"):
            material.n(0.9)
    else:
        assert_allclose(material.n(0.9), 1.4)
    with pytest.raises(AttributeError):
        material.bounds = "clamp" if default == "raise" else "raise"


@pytest.mark.parametrize("kind", ["data", "file", "catalog"])
@pytest.mark.parametrize("bounds", ["raise", "clamp"])
def test_bounds_roundtrip_and_independent_tables(
    kind, bounds, material_factory, set_test_backend
):
    original = material_factory(kind, bounds=bounds)
    serialized = json.loads(json.dumps(original.to_dict()))
    assert serialized["bounds"] == bounds
    restored = BaseMaterial.from_dict(serialized)
    for material in (original, restored):
        assert material.bounds == bounds
        assert material.spectral_range("n") == (0.4, 0.8)
        assert material.spectral_range("k") == (0.5, 0.7)
        assert_allclose(material.n(be.asarray([0.4, 0.6, 0.8])), [1.6, 1.5, 1.4])
        assert_allclose(material.k(be.asarray([0.5, 0.6, 0.7])), [0.01, 0.02, 0.03])
        for query in (0.3, 0.9, be.asarray([[0.6, 0.3], [0.9, 0.5]])):
            if bounds == "raise":
                with pytest.raises(ValueError, match="range"):
                    material.n(query)
            else:
                expected = (
                    [[1.5, 1.6], [1.4, 1.55]]
                    if be.is_array_like(query)
                    else (1.6 if query == 0.3 else 1.4)
                )
                assert_allclose(material.n(query), be.asarray(expected))
                assert_allclose(
                    material.n(query), be.asarray(expected)
                )  # Cached query.
        if bounds == "raise":
            with pytest.raises(ValueError, match="range"):
                material.k(0.45)  # n is defined here; k is not.
        else:
            assert_allclose(
                material.k(be.asarray([[0.4, 0.6], [0.8, 0.9]])),
                be.asarray([[0.01, 0.02], [0.03, 0.03]]),
            )


@pytest.mark.parametrize("kind", ["data", "file", "catalog"])
@pytest.mark.parametrize("bounds", ["warn", "extrapolate", None, [], {}])
def test_invalid_bounds_fail_on_construction(kind, bounds, material_factory):
    with pytest.raises(ValueError, match="bounds policy"):
        material_factory(kind, bounds=bounds)


@pytest.mark.parametrize("kind", ["file", "catalog"])
def test_public_file_json_without_policy_keeps_legacy_clamping(
    kind, material_factory, set_test_backend
):
    payload = material_factory(kind, bounds="raise").to_dict()
    del payload["bounds"]
    restored = BaseMaterial.from_dict(payload)
    assert restored.bounds == "clamp"
    assert_allclose(restored.n(0.9), 1.4)
    assert_allclose(restored.k(0.4), 0.01)
    assert restored.to_dict()["bounds"] == "clamp"


@pytest.mark.parametrize("bounds", ["raise", "clamp"])
def test_single_sample_file_has_explicit_bounds(bounds, tmp_path, set_test_backend):
    path = tmp_path / "single.yml"
    path.write_text("DATA:\n- type: tabulated nk\n  data: |\n    0.5 1.55 0.02\n")
    material = MaterialFile(str(path), bounds=bounds)
    for query, expected in ((material.n, 1.55), (material.k, 0.02)):
        assert_allclose(query(0.5), expected)
        if bounds == "raise":
            with pytest.raises(ValueError, match="range"):
                query(be.asarray([0.4, 0.5, 0.6]))
        else:
            assert_allclose(query(be.asarray([0.4, 0.5, 0.6])), [expected] * 3)


@pytest.mark.parametrize("bounds", ["raise", "clamp"])
def test_formula_limits_and_missing_extinction_are_not_table_policies(
    bounds, tmp_path, set_test_backend
):
    material = DataMaterial.from_coefficients(
        "formula 5", [1.5, 0.1, 1], wavelength_range=(0.4, 0.8), bounds=bounds
    )
    assert_allclose(material.n(0.6), 1.56)
    with pytest.raises(ValueError, match="range"):
        material.n(0.9)
    assert_allclose(material.k(0.9), 0)
    path = tmp_path / "formula.yml"
    path.write_text(
        "DATA:\n- type: formula 5\n  wavelength_range: 0.4 0.8\n  coefficients: 1.5 0.1 1\n"
    )
    file_material = MaterialFile(str(path), bounds=bounds)
    assert_allclose(file_material.n(0.9), 1.59)  # Existing analytic behavior.
    assert_allclose(file_material.k(0.9), 0)


def test_analytic_material_can_clamp_only_its_measured_extinction(set_test_backend):
    material = DataMaterial.from_coefficients(
        "formula 5",
        [1.5, 0.1, 1],
        bounds="clamp",
        extinction={
            "kind": "tabulated_k",
            "wavelengths_um": [0.5, 0.7],
            "values": [0.01, 0.03],
        },
    )
    for candidate in (material, BaseMaterial.from_dict(material.to_dict())):
        assert_allclose(candidate.n(be.asarray([0.4, 0.9])), [1.54, 1.59])
        assert_allclose(candidate.k(be.asarray([0.4, 0.9])), [0.01, 0.03])


@pytest.mark.parametrize("query", [0, -0.1, float("nan"), float("inf")])
def test_data_clamping_does_not_accept_invalid_wavelengths(query, set_test_backend):
    material = DataMaterial.from_samples([0.4, 0.8], [1.6, 1.4], bounds="clamp")
    for evaluate in (material.n, material.k):
        with pytest.raises(ValueError, match="finite and positive"):
            evaluate(query)


@pytest.mark.parametrize("kind", ["file", "catalog"])
@pytest.mark.parametrize("query", [float("nan"), float("inf"), -0.1])
def test_strict_files_reject_queries_without_table_support(
    kind, query, material_factory, set_test_backend
):
    material = material_factory(kind, bounds="raise")
    for evaluate in (material.n, material.k):
        with pytest.raises(ValueError, match="range"):
            evaluate(be.asarray([0.6, query]))


@pytest.mark.parametrize("kind", ["data", "file", "catalog"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_clamping_has_zero_wavelength_gradient_outside_samples(
    kind, precision, material_factory, set_test_backend
):
    if be.get_backend() != "torch":
        pytest.skip("Torch gradient contract")
    import torch

    be.set_precision(precision)
    try:
        material = material_factory(kind, bounds="clamp")
        material.n(0.3)
        material.k(0.9)
        for _ in range(2):
            waves = torch.tensor(
                [[0.3, 0.6, 0.9]], dtype=getattr(torch, precision), requires_grad=True
            )
            (material.n(waves) + material.k(waves)).sum().backward()
            torch.testing.assert_close(
                waves.grad, torch.tensor([[0, -0.4, 0]], dtype=waves.dtype)
            )
    finally:
        be.set_precision("float64")


def test_clamped_file_keeps_gradients_to_live_endpoint_samples(
    material_factory, set_test_backend
):
    if be.get_backend() != "torch":
        pytest.skip("Torch gradient contract")
    import torch

    material = material_factory("file", bounds="clamp")
    for _ in range(2):
        samples = torch.tensor([1.6, 1.4], dtype=torch.float64, requires_grad=True)
        material._n = samples
        material.n(be.asarray([0.3, 0.9])).sum().backward()
        torch.testing.assert_close(samples.grad, torch.ones_like(samples))
