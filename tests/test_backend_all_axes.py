"""Axis-wise truth reductions retain backend arrays and the global bool API."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from tests.utils import assert_allclose


@pytest.mark.parametrize("axis", [0, 1, -1, (0, 1), ()])
@pytest.mark.parametrize("host", [False, True])
def test_all_axes_preserve_values_and_boolean_dtype(axis, host, set_test_backend):
    data = np.array([[True, False, True], [True, True, True]])
    values = data.tolist() if host else be.array(data)
    result = be.all(values, axis=axis)
    expected = np.asarray(np.all(data, axis=axis))
    assert_allclose(result, expected)
    assert result.shape == expected.shape
    assert be.to_numpy(result).dtype == np.bool_
    if be.get_backend() == "torch":
        import torch

        assert isinstance(result, torch.Tensor)
        assert result.device == be.zeros(1).device


@pytest.mark.parametrize("shape,axis", [((2, 0), 1), ((0, 3), 0), ((0, 2, 3), (0, 2))])
def test_empty_axis_reduction_is_true(shape, axis, set_test_backend):
    result = be.all(be.zeros(shape), axis=axis)
    assert_allclose(result, np.all(np.zeros(shape), axis=axis))


def test_all_axes_numeric_truth_and_invalid_values(set_test_backend):
    values = be.array([[0, 1, 2], [float("nan"), float("inf"), -1]])
    assert_allclose(be.all(values, axis=1), be.array([False, True]))
    assert be.all(values) is False
    assert be.all(True) is True
    assert be.all([1, 2]) is True
    assert be.all(np.array([False])) is False


def test_all_axes_reject_invalid_axis(set_test_backend):
    with pytest.raises((IndexError, ValueError)):
        be.all(be.ones((2, 3)), axis=2)


def test_torch_all_axes_do_not_extract_python_scalars(monkeypatch, set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Device-resident reductions are a Torch capability")
    import torch

    values = torch.tensor([[True, False], [True, True]])
    with monkeypatch.context() as guard:

        def forbidden(*args, **kwargs):
            raise AssertionError("Axis reduction must not synchronize via item()")

        guard.setattr(torch.Tensor, "item", forbidden)
        result = be.all(values, axis=1)
    assert_allclose(result, be.array([False, True]))
