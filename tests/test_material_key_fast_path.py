"""Material cache-key fast paths equal the generic keys (speed PR, item 1)."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import Material
from optiland.materials.base import BaseMaterial

from .utils import assert_allclose

STATE_VALUES = [
    0,
    1.5,
    True,
    None,
    "clamp",
    np.float64(2.0),
    np.int64(3),
    (1, 2.0, "a"),
    [1.0, (2.0, None)],
    np.arange(5.0),
    np.arange(2000.0),
    np.zeros((7, 1)),
    (np.ones(3), 0.5),
    object(),
]


def _assert_same_key(value):
    fast = BaseMaterial._state_key(value)
    generic = BaseMaterial._state_key_generic(value)
    assert fast == generic
    if fast is not None:
        assert hash(fast) == hash(generic)


@pytest.mark.parametrize("value", STATE_VALUES, ids=lambda v: type(v).__name__)
def test_state_key_matches_generic(value):
    _assert_same_key(value)


def test_state_key_unsupported_object_disables_reuse():
    assert BaseMaterial._state_key(object()) is None
    assert BaseMaterial._state_key_generic(object()) is None


def test_state_key_matches_generic_torch():
    torch = pytest.importorskip("torch")
    _assert_same_key(torch.ones(3, dtype=torch.float64))
    _assert_same_key((torch.ones(3, dtype=torch.float64), 1.0))
    trainable = torch.ones(3, dtype=torch.float64, requires_grad=True)
    assert BaseMaterial._state_key(trainable) is None
    assert BaseMaterial._state_key_generic(trainable) is None
    assert BaseMaterial._state_key([1.0, trainable]) is None


def test_alias_mutation_still_invalidates():
    be.set_backend("numpy")
    m = Material("N-BK7")
    w = np.full(91, 0.55)
    n0 = np.array(m.n(w), copy=True)
    # coefficients[0] is zero for N-BK7, so scale a nonzero Sellmeier term.
    m.coefficients[1] *= 1.01
    assert not np.allclose(m.n(w), n0)
    m.coefficients[1] /= 1.01
    assert_allclose(m.n(w), n0)


def test_cache_keys_unchanged_by_kwargs_fast_path():
    be.set_backend("numpy")
    m = Material("N-BK7")
    w = np.full(91, 0.55)
    assert m._create_cache_key(w)[1] == ()
    assert m._create_cache_key(w, temperature=25)[1] == m._state_key(
        (("temperature", 25),)
    )


def test_small_array_key_equals_float_tuple():
    be.set_backend("numpy")
    m = Material("N-BK7")
    w = np.array([0.5, 0.6])
    key = m._create_cache_key(w)[0][1]
    assert key == (0.5, 0.6)
    assert hash(key) == hash(tuple(np.ravel(w)))
