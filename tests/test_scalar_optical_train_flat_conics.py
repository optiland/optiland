"""Infinite-radius native conics are constant flat-base metadata, not variables."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.geometries.standard import StandardGeometry
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.physical_optics import ScalarField, ScalarOpticalTrain

from .utils import assert_array_equal


@pytest.mark.parametrize("radius", [np.inf, -np.inf])
@pytest.mark.parametrize("conic", [-0.7, 0.35])
@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_explicit_flat_conic_metadata_has_no_infinite_parameter_gradient(
    set_test_backend, radius, conic, precision
):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=np.inf)
    optic.surfaces.add(index=1, z=0, surface_type="plane", material=IdealMaterial(1.5))
    # Ordinary surface creation canonicalizes infinite standard radii to Plane.
    # Also cover an explicitly constructed native geometry without changing it.
    geometry = StandardGeometry(optic.surfaces[1].geometry.cs, radius, conic)
    optic.surfaces[1].geometry = geometry
    data = np.full((17, 23), 1 + 2j, dtype=precision)
    if be.get_backend() == "torch":
        import torch

        data = torch.as_tensor(data).requires_grad_()
        geometry.radius.requires_grad_(True)
        geometry.k.requires_grad_(True)
    field = ScalarField(
        data, dx=0.01, dy=0.013, wavelength=0.0005, center=(0.02, -0.03)
    )
    radius_before, conic_before = geometry.radius, geometry.k
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert_array_equal(output.data, field.data)
    assert output.data.dtype == field.data.dtype
    assert output.center == field.center
    assert output.refractive_index == 1.5
    assert geometry.radius is radius_before and geometry.k is conic_before
    if be.get_backend() == "torch":
        output.data.real.sum().backward()
        assert_array_equal(data.grad, torch.ones_like(data))
        assert geometry.radius.grad is None and geometry.k.grad is None
