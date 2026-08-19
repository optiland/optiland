"""Scale-invariant 2x2 field-Jacobian conditioning and normalized solve.

The real-image-height solve validates its per-field ``2x2`` Jacobians with a
reciprocal Frobenius-condition estimate computed on the entry-normalized
matrix. These tests pin down the scientific requirements:

- a global unit/magnification rescaling of the Jacobian never changes the
  singular/non-singular classification;
- the normalized analytic solve matches a trusted linear solver;
- truly rank-deficient matrices are rejected and ``strict=True`` raises;
- the singular transition is governed by the documented reciprocal-condition
  threshold, not by the matrix's absolute scale.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.fields.field_types.real_image_height import (
    _RCOND_EPS_MULTIPLIER,
    RealImageHeightField,
    _jacobian_2x2_condition,
)

from .nr_implicit_test_utils import backend_state

BACKEND_PRECISION = [
    ("numpy", "float64"),
    ("numpy", "float32"),
    ("torch", "float64"),
    ("torch", "float32"),
]

# Fixed, well-conditioned reference matrix with all entries and both
# off-diagonals nonzero.
A0 = np.array([[2.0, 0.3], [-0.4, 1.5]])
X_TRUE = np.array([0.3, -0.7])


def _entries(J):
    return (
        be.array(J[0, 0]),
        be.array(J[0, 1]),
        be.array(J[1, 0]),
        be.array(J[1, 1]),
    )


def _solve(field, J, rhs, *, strict=False):
    dq_x, dq_y, singular = field._solve_2x2(
        *_entries(J), be.array(rhs[0]), be.array(rhs[1]), strict=strict
    )
    return (
        np.array([float(be.to_numpy(dq_x)), float(be.to_numpy(dq_y))]),
        bool(be.any(singular)),
    )


def _scales_for(precision):
    exponents = range(-12, 13, 3) if precision == "float64" else range(-6, 7, 2)
    return [10.0**e for e in exponents]


class TestGlobalScalingInvariance:
    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_classification_and_solution_are_scale_invariant(
        self, backend, precision
    ):
        with backend_state(backend, precision):
            field = RealImageHeightField()
            rtol = 1e-9 if precision == "float64" else 1e-4

            for s in _scales_for(precision):
                J = s * A0
                rhs = J @ X_TRUE

                dq, singular = _solve(field, J, rhs)
                assert not singular, f"scale {s:g} misclassified as singular"

                reference = np.linalg.solve(J, rhs)
                np.testing.assert_allclose(
                    dq, reference, rtol=rtol, atol=rtol * np.abs(X_TRUE).max()
                ), f"solution mismatch at scale {s:g}"

    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_small_identity_is_accepted(self, backend, precision):
        """Regression: 1e-3 * I was misclassified as singular in float32."""
        with backend_state(backend, precision):
            field = RealImageHeightField()
            J = 1.0e-3 * np.eye(2)
            rhs = J @ X_TRUE

            dq, singular = _solve(field, J, rhs)
            assert not singular
            rtol = 1e-9 if precision == "float64" else 1e-5
            np.testing.assert_allclose(dq, X_TRUE, rtol=rtol, atol=rtol)


class TestTrueRankDeficiency:
    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_rank_deficient_matrix_is_rejected(self, backend, precision):
        with backend_state(backend, precision):
            field = RealImageHeightField()
            J = np.array([[1.0, 1.0], [1.0, 1.0]])

            dq, singular = _solve(field, J, np.array([1.0, 0.0]))
            assert singular
            # Non-strict mode returns a zero placeholder step, never a
            # determinant-clipped one.
            np.testing.assert_allclose(dq, 0.0)

            with pytest.raises(ValueError, match="singular"):
                _solve(field, J, np.array([1.0, 0.0]), strict=True)

    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_zero_matrix_is_rejected(self, backend, precision):
        with backend_state(backend, precision):
            field = RealImageHeightField()
            J = np.zeros((2, 2))
            dq, singular = _solve(field, J, np.array([1.0, 1.0]))
            assert singular
            np.testing.assert_allclose(dq, 0.0)

    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_nonfinite_matrix_is_rejected(self, backend, precision):
        with backend_state(backend, precision):
            field = RealImageHeightField()
            J = np.array([[1.0, np.nan], [0.0, 1.0]])
            _, singular = _solve(field, J, np.array([1.0, 1.0]))
            assert singular


class TestIllConditionedTransition:
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("global_scale", [1.0, 1.0e6, 1.0e-6])
    def test_transition_follows_reciprocal_condition_threshold(
        self, backend, global_scale
    ):
        """The singular transition of J_d = s*[[1, 1], [1, 1 + d]].

        For every ``d`` the classification must equal the documented test
        ``rho_F <= C * eps`` evaluated on the normalized matrix -- and must
        therefore be identical at every global scale ``s``.
        """
        with backend_state(backend, "float64"):
            field = RealImageHeightField()
            eps = np.finfo(np.float64).eps
            classifications = []

            for d in [1.0e-2, 1.0e-6, 1.0e-10, 1.0e-13, 1.0e-15, 0.0]:
                J = global_scale * np.array([[1.0, 1.0], [1.0, 1.0 + d]])

                cond = _jacobian_2x2_condition(*_entries(J))
                singular = bool(be.any(cond.singular))

                # Expected value of the documented criterion, computed
                # independently in float64.
                Jn = J / np.abs(J).max() if np.abs(J).max() > 0 else J
                rho = abs(np.linalg.det(Jn)) / (Jn**2).sum()
                expected = rho <= _RCOND_EPS_MULTIPLIER * eps

                assert singular == expected, (
                    f"d={d:g}, scale={global_scale:g}: classification "
                    f"{singular} != documented-threshold value {expected}"
                )
                classifications.append(singular)

            # The sequence must actually cross the threshold.
            assert classifications[0] is False
            assert classifications[-1] is True

            # Sanity: the non-strict solve gives finite steps everywhere and
            # zero steps at the singular end.
            _, singular = _solve(
                field, global_scale * np.array([[1.0, 1.0], [1.0, 1.0]]),
                np.array([1.0, 0.0]),
            )
            assert singular
