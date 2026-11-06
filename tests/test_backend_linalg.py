"""Tests for be.linalg.qr and be.linalg.cholesky on both backends."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from tests.utils import assert_allclose


def _spd(n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


class TestQR:
    def test_reduced_tall(self, set_test_backend):
        rng = np.random.default_rng(1)
        a_np = rng.standard_normal((7, 4))
        q, r = be.linalg.qr(be.array(a_np))
        q_np, r_np = be.to_numpy(q), be.to_numpy(r)
        assert q_np.shape == (7, 4)
        assert r_np.shape == (4, 4)
        assert_allclose(q_np @ r_np, a_np, atol=1e-12)
        assert_allclose(q_np.T @ q_np, np.eye(4), atol=1e-12)
        assert_allclose(np.tril(r_np, -1), np.zeros((4, 4)), atol=1e-12)

    def test_complete_mode(self, set_test_backend):
        rng = np.random.default_rng(2)
        a_np = rng.standard_normal((6, 3))
        q, r = be.linalg.qr(be.array(a_np), mode="complete")
        q_np, r_np = be.to_numpy(q), be.to_numpy(r)
        assert q_np.shape == (6, 6)
        assert r_np.shape == (6, 3)
        assert_allclose(q_np @ r_np, a_np, atol=1e-12)
        assert_allclose(q_np.T @ q_np, np.eye(6), atol=1e-12)

    def test_batched(self, set_test_backend):
        rng = np.random.default_rng(3)
        a_np = rng.standard_normal((3, 5, 4))
        q, r = be.linalg.qr(be.array(a_np))
        assert_allclose(be.to_numpy(be.matmul(q, r)), a_np, atol=1e-12)


class TestCholesky:
    def test_lower_factor(self, set_test_backend):
        a_np = _spd()
        low = be.to_numpy(be.linalg.cholesky(be.array(a_np)))
        assert_allclose(low @ low.T, a_np, atol=1e-12)
        assert_allclose(np.triu(low, 1), np.zeros_like(a_np), atol=0)

    def test_matches_numpy(self, set_test_backend):
        a_np = _spd(4, seed=5)
        low = be.to_numpy(be.linalg.cholesky(be.array(a_np)))
        assert_allclose(low, np.linalg.cholesky(a_np), atol=1e-12)

    def test_batched(self, set_test_backend):
        a_np = np.stack([_spd(3, seed=s) for s in range(4)])
        low = be.to_numpy(be.linalg.cholesky(be.array(a_np)))
        assert_allclose(low @ np.swapaxes(low, -1, -2), a_np, atol=1e-12)

    def test_not_positive_definite_raises(self, set_test_backend):
        a = be.array([[1.0, 2.0], [2.0, 1.0]])
        # numpy raises LinAlgError; torch's LinAlgError subclasses RuntimeError.
        with pytest.raises((np.linalg.LinAlgError, RuntimeError)):
            be.linalg.cholesky(a)

    def test_solve_via_factor(self, set_test_backend):
        a_np = _spd(5, seed=7)
        b_np = np.arange(5.0)
        low = be.linalg.cholesky(be.array(a_np))
        # A x = b  ->  L y = b, L^T x = y
        y = be.linalg.solve(low, be.array(b_np))
        x = be.linalg.solve(be.transpose(low), y)
        assert_allclose(be.to_numpy(x), np.linalg.solve(a_np, b_np), atol=1e-10)


class TestCudaSmoke:
    @pytest.fixture
    def cuda_backend(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        be.set_backend("torch")
        be.set_device("cuda")
        be.set_precision("float64")
        yield
        be.set_device("cpu")
        be.set_backend("numpy")

    def test_qr_cholesky_on_gpu(self, cuda_backend):
        a_np = _spd(6, seed=11)
        a = be.array(a_np)
        assert a.device.type == "cuda"

        q, r = be.linalg.qr(a)
        assert q.device.type == "cuda"
        assert_allclose(be.to_numpy(q @ r), a_np, atol=1e-10)

        low = be.linalg.cholesky(a)
        assert low.device.type == "cuda"
        assert_allclose(be.to_numpy(low @ low.T), a_np, atol=1e-10)
