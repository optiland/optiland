from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_equal as np_assert_array_equal

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Iterator


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


def emulated_df64() -> bool:
    """True on ``torch`` / ``mps`` / float64, i.e. the df64 emulation (48 bits)."""
    return (
        be.get_backend() == "torch"
        and be.get_device() == "mps"
        and str(be.get_precision()).endswith(("float64", "64"))
        and be.metal_mode() == "df64"
    )


@contextlib.contextmanager
def xfail_if_emulated_df64(reason: str) -> Iterator[None]:
    """Turn an ``AssertionError`` inside the block into a non-strict xfail on df64.

    ``torch`` on ``mps`` with float64 precision emulates float64 with a
    double-single representation (48-bit significand, ``be.metal_mode() ==
    "df64"``). An assertion at the 1e-15..1e-13 absolute level is a
    representation limit there, not a defect; the exact ``sf64`` mode and every
    other backend keep asserting it, and a pass on df64 is still a pass (the
    marker is non-strict). Each use is documented in ``NOTES/05-m2-status.md``.
    """
    try:
        yield
    except AssertionError as exc:
        if emulated_df64():
            pytest.xfail(f"{reason}: {exc}")
        raise
