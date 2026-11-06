"""The memoized dtype/device strings in material cache keys match ``str``."""

from __future__ import annotations

import gc

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import base as material_base
from optiland.materials import IdealMaterial

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _empty_cache():
    material_base._TYPE_STR_CACHE.clear()
    yield
    material_base._TYPE_STR_CACHE.clear()


OBJECTS = [
    np.dtype("float64"),
    np.dtype(">f8"),
    np.dtype("float32"),
    np.dtype("complex128"),
    np.dtype([("a", "<f8"), ("b", "i1")]),
    np.dtype([("a", "<f8"), ("b", "i1")], align=True),
    torch.float64,
    torch.float32,
    torch.device("cpu"),
    torch.device("meta"),
    None,
    "list",
    "float",
]


@pytest.mark.parametrize("obj", OBJECTS, ids=repr)
def test_type_str_matches_str(obj):
    assert material_base._type_str(obj) == str(obj)
    assert material_base._type_str(obj) == str(obj)  # cached hit


def test_equal_objects_of_other_types_do_not_share_an_entry():
    # np.dtype("float64") compares equal to None and to "float".
    assert np.dtype("float64") == None  # noqa: E711
    for first, second in [(None, np.dtype("float64")), ("float", np.dtype("float64"))]:
        material_base._TYPE_STR_CACHE.clear()
        assert material_base._type_str(first) == str(first)
        assert material_base._type_str(second) == "float64"
        assert material_base._type_str(first) == str(first)


def test_freed_object_cannot_hand_its_string_to_a_new_one():
    for _ in range(50):
        old = torch.device("meta")
        assert material_base._type_str(old) == "meta"
        del old
        gc.collect()
        new = torch.device("cpu")
        assert material_base._type_str(new) == "cpu"


def test_unhashable_objects_fall_back_to_str():
    class Unhashable:
        __hash__ = None

        def __str__(self):
            return "unhashable"

    assert material_base._type_str(Unhashable()) == "unhashable"


def test_cache_is_bounded():
    for i in range(3 * material_base._TYPE_STR_CACHE_MAX):
        material_base._type_str(f"dtype-{i}")
    assert len(material_base._TYPE_STR_CACHE) <= material_base._TYPE_STR_CACHE_MAX


def _reference_keys(value, digest, scalar):
    return (
        (
            "array-content",
            digest,
            tuple(getattr(value, "shape", ())),
            str(getattr(value, "dtype", type(value).__name__)),
            str(getattr(value, "device", None)),
        ),
        (
            "array-uniform",
            scalar,
            tuple(getattr(value, "shape", ())),
            str(getattr(value, "dtype", type(value).__name__)),
            str(getattr(value, "device", None)),
        ),
    )


@pytest.mark.parametrize(
    "value",
    [
        np.linspace(0.4, 0.7, 5),
        np.linspace(0.4, 0.7, 5).astype(np.float32),
        torch.linspace(0.4, 0.7, 5, dtype=torch.float64),
    ],
    ids=["numpy-f8", "numpy-f4", "torch-f8"],
)
def test_cache_keys_are_unchanged(value):
    content, uniform = _reference_keys(value, b"digest", 0.5)
    assert material_base._array_content_key(value, b"digest") == content
    assert material_base._array_uniform_key(value, 0.5) == uniform


def test_refractive_index_cache_still_distinguishes_dtypes(set_test_backend):
    material = IdealMaterial(n=1.5)
    wl64 = be.array(np.linspace(0.4, 0.7, 5))
    key64 = material._create_cache_key(wl64)
    assert key64[0][3] == str(wl64.dtype)
    if be.get_backend() == "numpy":
        wl32 = wl64.astype(np.float32)
        assert material._create_cache_key(wl32)[0][3] == "float32"
        assert material._create_cache_key(wl32) != key64
