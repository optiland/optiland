"""Forbes Clenshaw sums of one family in one downward pass.

``qpoly._qbfs_pass`` and ``qpoly._q2d_pass`` give the sum S of one family
and its slope dS/d(u**2) in one pass, and keep only the last two terms of
each recurrence. The sum S has the operations, and so the bits, of the
Clenshaw sums of before. The slope is the derivative of the same
recurrence.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.geometries.forbes import qpoly

from .nr_implicit_test_utils import backend_state


def _bits(value):
    return np.asarray(be.to_numpy(value), dtype=np.float64).view(np.int64)


def _assert_bits(value, expected):
    np.testing.assert_array_equal(_bits(value), _bits(expected))


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("m", [0, 1, 2, 3])
@pytest.mark.parametrize("size", [1, 2, 3, 6])
def test_one_pass_gives_the_clenshaw_sums(size, m, backend):
    """S of one pass has the bits of the Clenshaw sums; dS/du**2 matches."""
    if backend == "torch":
        pytest.importorskip("torch")
    rng = np.random.default_rng(size + 10 * m)
    coefficients = list(rng.uniform(-1.0, 1.0, size))
    with backend_state(backend, "float64"):
        usq = be.array(np.linspace(0.0, 1.3, 27))
        basis = qpoly._family_basis(coefficients, m)
        total, slope = (
            qpoly._qbfs_pass(basis, usq, slope=True)
            if m == 0
            else qpoly._q2d_pass(basis, m, usq, slope=True)
        )
        if m == 0:
            alphas = qpoly.clenshaw_qbfs_der(coefficients, usq, j=1)
            first = 2 * (alphas[0, 0] + alphas[0, 1]) if size > 1 else None
            expected = 2 * alphas[0, 0] if size == 1 else first
            expected_slope = (
                2 * alphas[1, 0] if size == 1 else 2 * (alphas[1, 0] + alphas[1, 1])
            )
        else:
            alphas = qpoly.clenshaw_q2d_der(coefficients, m, usq, j=1)
            expected = qpoly.q2d_sum_from_alphas(alphas[0], m, size)
            expected_slope = qpoly.q2d_sum_from_alphas(alphas[1], m, size)
        _assert_bits(total, expected)
        np.testing.assert_allclose(
            be.to_numpy(slope), be.to_numpy(expected_slope), rtol=1e-13, atol=1e-14
        )
        if m == 0:
            _assert_bits(qpoly._qbfs_pass(basis, usq), total)
        else:
            _assert_bits(qpoly._q2d_pass(basis, m, usq), total)
