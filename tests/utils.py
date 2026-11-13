from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal as np_assert_array_equal

import optiland.backend as be


def assert_allclose(a, b, rtol=1.0e-5, atol=1.0e-7):
    """Assert that two arrays or tensors are element-wise equal within
    tolerance.
    """
    a = be.to_numpy(a)
    b = be.to_numpy(b)
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def assert_array_equal(a, b):
    """Assert that two arrays or tensors are element-wise equal."""
    a = be.to_numpy(a)
    b = be.to_numpy(b)

    np_assert_array_equal(a, b)


def assert_bits_equal(a, b, *, ignore_nan_sign: bool = False) -> None:
    """Assert that two arrays or tensors of real numbers hold the same bits.

    A fast path must give the bits of the generic code, so its tests compare
    bits, not values within a tolerance. A value comparison would accept
    ``-0.0`` for ``0.0``. NaN is not equal to itself, so NaN entries are
    compared by position and by sign bit. Compiled code can give a NaN
    another sign bit than NumPy does; the tests of such code pass
    ``ignore_nan_sign=True``.

    Args:
        a: The first array or tensor.
        b: The second array or tensor.
        ignore_nan_sign: If True, do not compare the sign bits of NaN
            entries.

    Raises:
        AssertionError: If the dtypes, the shapes or the NaN positions
            differ, if an entry that is not NaN has other bytes, or, unless
            ``ignore_nan_sign`` is True, if a NaN has another sign bit.
    """
    a = np.asarray(be.to_numpy(a))
    b = np.asarray(be.to_numpy(b))
    assert a.dtype == b.dtype, f"dtype {a.dtype} != {b.dtype}"
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    assert np.array_equal(nan_a, nan_b), "NaN at other positions"
    assert a[~nan_a].tobytes() == b[~nan_b].tobytes(), "other bits"
    if not ignore_nan_sign:
        same_sign = np.array_equal(np.signbit(a[nan_a]), np.signbit(b[nan_b]))
        assert same_sign, "NaN with another sign bit"
