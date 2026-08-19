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
