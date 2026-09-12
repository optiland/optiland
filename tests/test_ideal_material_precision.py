"""Ideal material parameters retain their precision across execution changes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial

from .utils import assert_allclose

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def restore_context(set_test_backend: None) -> Iterator[None]:
    """Restore each backend's precision and the Torch device after these tests."""
    initial = be.get_backend()
    precisions = {}
    device = None
    for backend in be.list_available_backends():
        be.set_backend(backend)
        precisions[backend] = f"float{be.get_precision()}"
        if backend == "torch":
            device = be.get_device()
    be.set_backend(initial)
    try:
        yield
    finally:
        for backend, precision in precisions.items():
            be.set_backend(backend)
            be.set_precision(precision)
            if backend == "torch":
                be.set_device(device)
        be.set_backend(initial)


@pytest.mark.parametrize("property_name,attribute", [("n", "index"), ("k", "absorp")])
@pytest.mark.parametrize("parameter_precision", ["float32", "float64"])
@pytest.mark.parametrize(
    "query_kind", ["scalar", "singleton", "matrix", "uniform", "list"]
)
def test_parameter_precision_survives_default_change(
    restore_context: None,
    property_name: str,
    attribute: str,
    parameter_precision: str,
    query_kind: str,
) -> None:
    """Both n/k keep stored precision on fresh evaluation and on cache hits."""
    be.set_precision(parameter_precision)
    material = IdealMaterial(1.50000001, 0.100000001)
    # Use constants here so repeat lookups exercise the property caches on Torch.
    material.index = be.asarray([1.50000001])
    material.absorp = be.asarray([0.100000001])
    parameter = getattr(material, attribute)
    expected = be.to_numpy(parameter).copy()[0]
    evaluate = getattr(material, property_name)
    evaluate(0.55)

    other_precision = "float32" if parameter_precision == "float64" else "float64"
    be.set_precision(other_precision)
    if query_kind == "scalar":
        query, shape = 0.55, ()
    elif query_kind == "singleton":
        query, shape = be.asarray([0.55]), ()
    elif query_kind == "matrix":
        query, shape = be.asarray([[0.4, 0.5], [0.6, 0.7]]), (2, 2)
    elif query_kind == "uniform":
        query, shape = be.asarray(np.full((32, 64), 0.55)), (32, 64)
    else:
        query, shape = [[0.4, 0.5], [0.6, 0.7]], (2, 2)

    for _ in range(2):
        result = evaluate(query)
        assert tuple(result.shape) == shape
        assert result.dtype == parameter.dtype
        assert_allclose(result, np.full(shape, expected), rtol=0, atol=0)
        assert getattr(material, attribute) is parameter


@pytest.mark.parametrize("property_name,attribute", [("n", "index"), ("k", "absorp")])
def test_precision_survives_backend_round_trip(
    restore_context: None, property_name: str, attribute: str
) -> None:
    """Backend conversion preserves live storage, precision, and trainable leaves."""
    if "torch" not in be.list_available_backends():
        pytest.skip("requires both backends")
    import torch

    initial = be.get_backend()
    be.set_precision("float64")
    material = IdealMaterial(1.50000001, 0.100000001)
    parameter = getattr(material, attribute)
    expected = be.to_numpy(parameter).copy()[0]
    evaluate = getattr(material, property_name)
    other = "numpy" if initial == "torch" else "torch"
    for backend in (other, initial):
        be.set_backend(backend)
        if backend == "torch":
            be.set_device("cpu")
        be.set_precision("float32")
        for query in (0.55, be.asarray([0.4, 0.6])):
            for _ in range(2):
                result = evaluate(query)
                assert isinstance(result, torch.Tensor) == (backend == "torch")
                assert be.to_numpy(result).dtype == np.dtype("float64")
                assert_allclose(result, expected, rtol=0, atol=0)
                assert getattr(material, attribute) is parameter
                if initial == backend == "torch":
                    (gradient,) = torch.autograd.grad(result.sum(), parameter)
                    assert_allclose(gradient, be.size(result), rtol=0, atol=0)


@pytest.mark.parametrize("property_name,attribute", [("n", "index"), ("k", "absorp")])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_trainable_parameter_precision_and_device(
    restore_context: None, property_name: str, attribute: str, device: str
) -> None:
    """Default changes and device transfers keep fresh graphs to the original leaf."""
    if be.get_backend() != "torch":
        pytest.skip("Torch gradients")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    be.set_precision("float64")
    material = IdealMaterial(1.50000001, 0.100000001)
    parameter = torch.nn.Parameter(getattr(material, attribute).detach().clone())
    setattr(material, attribute, parameter)
    expected = be.to_numpy(parameter).copy()[0]
    evaluate = getattr(material, property_name)
    be.set_precision("float32")
    be.set_device(device)

    # A graph-free first lookup must not poison subsequent gradient evaluations.
    with torch.no_grad():
        evaluate(0.55)
    for query in (0.55, be.asarray([[0.4, 0.5], [0.6, 0.7]])):
        for _ in range(2):
            result = evaluate(query)
            assert result.dtype == torch.float64
            assert result.device.type == device
            assert_allclose(result, expected, rtol=0, atol=0)
            result.sum().backward()
            assert_allclose(parameter.grad, be.size(result), rtol=0, atol=0)
            parameter.grad = None
            assert getattr(material, attribute) is parameter
