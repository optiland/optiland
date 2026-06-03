from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import optiland.backend as be
from optiland.optimization.variable.nurbs import (
    NurbsPointsVariable,
    NurbsWeightsVariable,
)
from optiland.optimization.variable.torch import TorchVariable


def test_nurbs_points_variable(set_test_backend):
    # Setup mocks
    mock_optic = MagicMock()
    mock_surf = MagicMock()
    # P is typically (3, nu, nv) or similar.
    # index (i, j, k) -> (0, 1, 2)
    shape = (3, 3, 3)
    if be.get_backend() == "torch":
        import torch

        mock_surf.geometry.P = torch.zeros(shape, dtype=torch.float64)
    else:
        mock_surf.geometry.P = np.zeros(shape, dtype=float)

    mock_optic.surfaces = [mock_surf]

    var = NurbsPointsVariable(mock_optic, 0, (0, 1, 1))

    # Check initial value
    val = var.get_value()
    if hasattr(val, "item"):
        val = val.item()
    assert val == 0.0

    # Update value
    var.update_value(5.0)

    # Check updated value
    # For torch, update_value replaces the tensor or modifies it.
    new_val = mock_surf.geometry.P[0, 1, 1]
    if hasattr(new_val, "item"):
        new_val = new_val.item()
    assert new_val == 5.0

    # Check str
    assert "Control Point" in str(var)


def test_nurbs_weights_variable(set_test_backend):
    mock_optic = MagicMock()
    mock_surf = MagicMock()
    shape = (3, 3)
    if be.get_backend() == "torch":
        import torch

        mock_surf.geometry.W = torch.zeros(shape, dtype=torch.float64)
    else:
        mock_surf.geometry.W = np.zeros(shape, dtype=float)

    mock_optic.surfaces = [mock_surf]

    var = NurbsWeightsVariable(mock_optic, 0, (1, 1))

    # Update
    var.update_value(2.0)
    new_val = mock_surf.geometry.W[1, 1]
    if hasattr(new_val, "item"):
        new_val = new_val.item()
    assert new_val == 2.0

    assert "Weight" in str(var)


def test_torch_variable_init(set_test_backend):
    if be.get_backend() != "torch":
        # Should raise ValueError
        with pytest.raises(ValueError, match="TorchVariable can only be used"):
            TorchVariable(None, 0, 1.0)
    else:
        # Should work
        mock_optic = MagicMock()
        import torch

        var = TorchVariable(mock_optic, 0, 1.0)
        assert isinstance(var.value, torch.nn.Parameter)
        assert var.value.item() == 1.0

        # Test update
        var.update_value(2.0)
        assert var.value.item() == 2.0

        # Test update with Parameter
        new_param = torch.nn.Parameter(torch.tensor(3.0))
        var.update_value(new_param)
        assert var.value is new_param
