from __future__ import annotations

import numpy as np
import pytest

import optiland.backend.utils as utils


def test_to_numpy_scalar():
    res = utils.to_numpy(5)
    assert isinstance(res, np.ndarray)
    assert res.item() == 5


def test_to_numpy_array():
    arr = np.array([1, 2])
    res = utils.to_numpy(arr)
    assert res is arr  # Should return same object if already numpy array


def test_to_numpy_list():
    lst = [1, 2.0]
    res = utils.to_numpy(lst)
    assert isinstance(res, np.ndarray)
    assert np.allclose(res, np.array([1.0, 2.0]))


def test_to_numpy_torch():
    import torch

    t = torch.tensor([1.0, 2.0])
    res = utils.to_numpy(t)
    assert isinstance(res, np.ndarray)
    assert np.allclose(res, np.array([1.0, 2.0]))


def test_to_numpy_list_of_tensors():
    import torch

    lst = [torch.tensor(1.0), torch.tensor(2.0)]
    res = utils.to_numpy(lst)
    assert isinstance(res, np.ndarray)
    assert np.allclose(res, np.array([1.0, 2.0]))


def test_to_numpy_unsupported():
    class Dummy:
        pass

    with pytest.raises(TypeError):
        utils.to_numpy(Dummy())


def test_is_torch_tensor():
    import torch

    t = torch.tensor([1.0])
    assert utils.is_torch_tensor(t)
    assert not utils.is_torch_tensor(np.array([1.0]))
    assert not utils.is_torch_tensor(1.0)
