"""Validation, interpolation and numerical limits of sampled optical materials."""

from __future__ import annotations

import math

import pytest

import optiland.backend as be
from optiland.materials import BaseMaterial, DataMaterial
from tests.utils import assert_allclose


@pytest.mark.parametrize(
    "waves,indices,message",
    [
        ([], [], "at least two paired samples"),
        ([0.5], [1.5], "at least two paired samples"),
        ([0.5, 0.6], [1.5], "paired samples"),
        ([0, 0.6], [1.5, 1.6], "positive"),
        ([0.5, math.inf], [1.5, 1.6], "finite"),
        ([0.5, 0.6], [1.5, math.nan], "finite"),
        ([0.5, 0.5], [1.5, 1.6], "distinct"),
    ],
)
def test_invalid_samples(waves, indices, message):
    with pytest.raises(ValueError, match=message):
        DataMaterial.from_samples(waves, indices)


def test_interpolation_and_serialization(set_test_backend):
    material = DataMaterial.from_samples([0.6, 0.4, 0.8], [1.5, 1.6, 1.4], name="example")
    assert_allclose(
        material.n(be.array([0.4, 0.5, 0.6, 0.7, 0.8])), [1.6, 1.55, 1.5, 1.45, 1.4]
    )
    assert_allclose(BaseMaterial.from_dict(material.to_dict()).n(0.5), 1.55)
    with pytest.raises(ValueError, match="range"):
        material.n(0.9)


def test_nonabsorbing_samples(set_test_backend):
    material = DataMaterial.from_samples([0.4, 0.8], [1.6, 1.5])
    assert_allclose(material.k(0.5), 0)
    assert_allclose(material.k(be.array([0.4, 0.6, 0.8])), [0, 0, 0])


def test_nonfinite_query(set_test_backend):
    material = DataMaterial.from_samples([0.4, 0.8], [1.6, 1.4])
    with pytest.raises(ValueError, match="finite"):
        material.n(math.nan)


@pytest.mark.parametrize("array_query", [False, True])
def test_samples_must_remain_distinct_in_the_active_precision(
    set_test_backend, array_query
):
    is_torch = be.get_backend() == "torch"
    if is_torch:
        be.set_precision("float32")
    try:
        waves = [0.50000000000001, 0.50000000000002]
        material = DataMaterial.from_samples(waves, [1.5, 1.6])
        query = be.array(waves) if array_query else waves[0]
        if is_torch:
            # Both wavelengths round to 0.5 in float32. Interpolating across
            # this zero-width interval used to silently return NaN indices.
            with pytest.raises(ValueError, match="distinct.*backend precision"):
                material.n(query)
        else:
            assert_allclose(material.n(query), [1.5, 1.6] if array_query else [1.5])

        ordinary = DataMaterial.from_samples([0.5, 0.7], [1.6, 1.5])
        assert_allclose(ordinary.n(be.array([0.5, 0.6, 0.7])), [1.6, 1.55, 1.5])
    finally:
        if is_torch:
            be.set_precision("float64")
