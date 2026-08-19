"""
Contract tests for the AbstractBackend interface.

Verifies that:
1. Both backends implement every method (no AttributeError / NotImplementedError).
2. Passthrough math functions produce numerically equivalent results on both backends.
3. Capability-gated methods raise BackendCapabilityError on the NumPy backend.
4. Capability flags have the expected values.

Kramer Harrison, 2025
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.backend.base import BackendCapabilityError
from tests.utils import assert_allclose

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_backend():
    """Restore the active backend after each test."""
    original = be.get_backend()
    yield
    be.set_backend(original)
    # Reset torch precision to float32 if torch is available
    if "torch" in be.list_available_backends():
        be._backends["torch"].set_precision("float32")


def _np_backend():
    return be._backends["numpy"]


def _torch_backend():
    inst = be._backends.get("torch")
    if inst is None:
        pytest.skip("torch backend not available")
    return inst


# ---------------------------------------------------------------------------
# TestBackendContract — parametrized over both backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_name", be.list_available_backends())
class TestBackendContract:
    """Verify that every backend method is callable and produces sane output."""

    # ---- Trig ----

    def test_sin(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([0.0, math.pi / 2, math.pi])
        result = be.sin(x)
        expected = np.array([0.0, 1.0, 0.0])
        assert_allclose(result, expected, atol=1e-6)

    def test_cos(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([0.0, math.pi / 2, math.pi])
        result = be.cos(x)
        expected = np.array([1.0, 0.0, -1.0])
        assert_allclose(result, expected, atol=1e-6)

    def test_tan(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([0.0, math.pi / 4])
        result = be.tan(x)
        expected = np.array([0.0, 1.0])
        assert_allclose(result, expected, atol=1e-6)

    def test_arcsin(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([0.0, 1.0, -1.0])
        result = be.arcsin(x)
        expected = np.arcsin(np.array([0.0, 1.0, -1.0]))
        assert_allclose(result, expected, atol=1e-6)

    def test_arccos(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([1.0, 0.0, -1.0])
        result = be.arccos(x)
        expected = np.arccos(np.array([1.0, 0.0, -1.0]))
        assert_allclose(result, expected, atol=1e-6)

    def test_arctan(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([0.0, 1.0, -1.0])
        result = be.arctan(x)
        expected = np.arctan(np.array([0.0, 1.0, -1.0]))
        assert_allclose(result, expected, atol=1e-6)

    def test_arctan2(self, backend_name: str, set_test_backend: None) -> None:
        y = be.array([1.0, 0.0])
        x = be.array([0.0, 1.0])
        result = be.arctan2(y, x)
        expected = np.arctan2([1.0, 0.0], [0.0, 1.0])
        assert_allclose(result, expected, atol=1e-6)

    def test_sinh_cosh_tanh(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([0.0, 1.0])
        assert_allclose(be.sinh(x), np.sinh([0.0, 1.0]), atol=1e-6)
        assert_allclose(be.cosh(x), np.cosh([0.0, 1.0]), atol=1e-6)
        assert_allclose(be.tanh(x), np.tanh([0.0, 1.0]), atol=1e-6)

    # ---- Math ----

    def test_exp_log(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([1.0, 2.0, math.e])
        assert_allclose(be.exp(be.log(x)), be.to_numpy(x), atol=1e-6)

    def test_log2(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([1.0, 2.0, 4.0])
        assert_allclose(be.log2(x), np.array([0.0, 1.0, 2.0]), atol=1e-6)

    def test_log10(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([1.0, 10.0, 100.0])
        assert_allclose(be.log10(x), np.array([0.0, 1.0, 2.0]), atol=1e-6)

    def test_sqrt(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([0.0, 1.0, 4.0, 9.0])
        assert_allclose(be.sqrt(x), np.array([0.0, 1.0, 2.0, 3.0]), atol=1e-6)

    def test_abs(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([-3.0, 0.0, 2.0])
        assert_allclose(be.abs(x), np.array([3.0, 0.0, 2.0]), atol=1e-6)

    def test_sign(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([-1.0, 0.0, 2.0])
        assert_allclose(be.sign(x), np.array([-1.0, 0.0, 1.0]), atol=1e-6)

    def test_floor_ceil(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([1.2, 2.7, -1.3])
        assert_allclose(be.floor(x), np.array([1.0, 2.0, -2.0]), atol=1e-6)
        assert_allclose(be.ceil(x), np.array([2.0, 3.0, -1.0]), atol=1e-6)

    def test_deg2rad_rad2deg(self, backend_name: str, set_test_backend: None) -> None:
        # Test with Python float (scalar case)
        r = be.deg2rad(180.0)
        assert_allclose(r, math.pi, atol=1e-6)
        d = be.rad2deg(math.pi)
        assert_allclose(d, 180.0, atol=1e-5)

    def test_radians_degrees(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([0.0, 90.0, 180.0])
        r = be.radians(x)
        assert_allclose(r, np.array([0.0, math.pi / 2, math.pi]), atol=1e-6)
        d = be.degrees(be.array([0.0, math.pi / 2, math.pi]))
        assert_allclose(d, np.array([0.0, 90.0, 180.0]), atol=1e-5)

    # ---- Checks ----

    def test_isnan_isinf_isfinite(
        self, backend_name: str, set_test_backend: None
    ) -> None:
        # Scalar inputs
        assert be.isnan(float("nan"))
        assert not be.isnan(1.0)
        assert be.isinf(float("inf"))
        assert not be.isinf(1.0)
        assert be.isfinite(1.0)
        assert not be.isfinite(float("inf"))

    # ---- Array creation ----

    def test_array(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([1.0, 2.0, 3.0])
        assert be.to_numpy(x).tolist() == [1.0, 2.0, 3.0]

    def test_zeros_ones_full(self, backend_name: str, set_test_backend: None) -> None:
        assert_allclose(be.zeros((3,)), np.zeros(3), atol=0)
        assert_allclose(be.ones((3,)), np.ones(3), atol=0)
        assert_allclose(be.full((3,), 5.0), np.full(3, 5.0), atol=0)

    def test_linspace_arange(self, backend_name: str, set_test_backend: None) -> None:
        assert_allclose(be.linspace(0.0, 1.0, 5), np.linspace(0, 1, 5), atol=1e-6)
        assert_allclose(be.arange(0, 5), np.arange(5), atol=0)

    def test_zeros_ones_full_like(
        self, backend_name: str, set_test_backend: None
    ) -> None:
        ref = be.array([1.0, 2.0, 3.0])
        assert_allclose(be.zeros_like(ref), np.zeros(3), atol=0)
        assert_allclose(be.ones_like(ref), np.ones(3), atol=0)
        assert_allclose(be.full_like(ref, 7.0), np.full(3, 7.0), atol=0)

    def test_empty(self, backend_name: str, set_test_backend: None) -> None:
        x = be.empty((4,))
        assert be.to_numpy(x).shape == (4,)

    def test_eye(self, backend_name: str, set_test_backend: None) -> None:
        result = be.eye(3)
        assert_allclose(result, np.eye(3), atol=1e-6)

    def test_asarray(self, backend_name: str, set_test_backend: None) -> None:
        x = be.asarray([1.0, 2.0])
        assert_allclose(x, np.array([1.0, 2.0]), atol=0)

    # ---- Reductions ----

    def test_sum(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([1.0, 2.0, 3.0])
        assert_allclose(be.sum(x), np.array([6.0]), atol=1e-6)
        x2d = be.array([[1.0, 2.0], [3.0, 4.0]])
        assert_allclose(be.sum(x2d, axis=0), np.array([4.0, 6.0]), atol=1e-6)

    def test_mean(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([1.0, 2.0, 3.0])
        assert_allclose(be.mean(x), np.array([2.0]), atol=1e-6)

    def test_max_min(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([3.0, 1.0, 2.0])
        assert float(be.max(x)) == pytest.approx(3.0)
        assert float(be.min(x)) == pytest.approx(1.0)

    def test_clip(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([-2.0, 0.5, 3.0])
        assert_allclose(be.clip(x, 0.0, 1.0), np.array([0.0, 0.5, 1.0]), atol=1e-6)

    def test_concatenate(self, backend_name: str, set_test_backend: None) -> None:
        a = be.array([1.0, 2.0])
        b = be.array([3.0, 4.0])
        assert_allclose(be.concatenate([a, b]), np.array([1.0, 2.0, 3.0, 4.0]), atol=0)

    def test_expand_dims(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([1.0, 2.0])
        result = be.expand_dims(x, 0)
        assert be.to_numpy(result).shape == (1, 2)

    def test_where(self, backend_name: str, set_test_backend: None) -> None:
        cond = be.array([1.0, 0.0, 1.0]) > 0.5
        a = be.array([10.0, 10.0, 10.0])
        b = be.array([0.0, 0.0, 0.0])
        assert_allclose(be.where(cond, a, b), np.array([10.0, 0.0, 10.0]), atol=0)

    # ---- Shape ----

    def test_transpose(self, backend_name: str, set_test_backend: None) -> None:
        x = be.array([[1.0, 2.0], [3.0, 4.0]])
        result = be.transpose(x)
        assert_allclose(result, np.array([[1.0, 3.0], [2.0, 4.0]]), atol=0)

    def test_reshape(self, backend_name: str, set_test_backend: None) -> None:
        x = be.arange(0, 6)
        result = be.reshape(x, (2, 3))
        assert be.to_numpy(result).shape == (2, 3)

    def test_stack(self, backend_name: str, set_test_backend: None) -> None:
        a = be.array([1.0, 2.0])
        b = be.array([3.0, 4.0])
        result = be.stack([a, b])
        assert_allclose(result, np.array([[1.0, 2.0], [3.0, 4.0]]), atol=0)

    # ---- Linear algebra ----

    def test_matmul(self, backend_name: str, set_test_backend: None) -> None:
        a = be.array([[1.0, 0.0], [0.0, 1.0]])
        b = be.array([2.0, 3.0])
        assert_allclose(be.matmul(a, b), np.array([2.0, 3.0]), atol=1e-6)

    def test_lstsq(self, backend_name: str, set_test_backend: None) -> None:
        a = be.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = be.array([1.0, 2.0, 3.0])
        sol = be.lstsq(a, b)
        assert be.to_numpy(sol).shape == (2,)

    # ---- Precision ----

    def test_set_get_precision(self, backend_name: str, set_test_backend: None) -> None:
        be.set_precision("float32")
        assert be.get_precision() == 32
        be.set_precision("float64")
        assert be.get_precision() == 64

    # ---- Interpolation ----

    def test_interp(self, backend_name: str, set_test_backend: None) -> None:
        xp = be.array([0.0, 1.0, 2.0])
        fp = be.array([0.0, 1.0, 4.0])
        x = be.array([0.5, 1.5])
        result = be.interp(x, xp, fp)
        expected = np.interp(np.array([0.5, 1.5]), [0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        assert_allclose(result, expected, atol=1e-5)

    # ---- Capability flags ----

    def test_capability_flags(self, backend_name: str, set_test_backend: None) -> None:
        active = be.get_backend()
        if active == "numpy":
            assert be.supports_gradients is False
        elif active == "torch":
            assert be.supports_gradients is True

    # ---- Polynomial ----

    def test_polyval(self, backend_name: str, set_test_backend: None) -> None:
        # p(x) = x^2 + 1, coeffs = [1, 0, 1] (highest power first)
        x = be.array([0.0, 1.0, 2.0])
        coeffs = be.array([1.0, 0.0, 1.0])
        result = be.polyval(coeffs, x)
        expected = np.array([1.0, 2.0, 5.0])
        assert_allclose(result, expected, atol=1e-5)

    # ---- Random ----

    def test_rand_shape(self, backend_name: str, set_test_backend: None) -> None:
        x = be.rand(10)
        arr = be.to_numpy(x)
        assert arr.shape in ((10,), (1, 10))  # torch returns (1, 10)

    def test_random_uniform_bounds(
        self, backend_name: str, set_test_backend: None
    ) -> None:
        x = be.random_uniform(0.0, 1.0, size=100)
        arr = be.to_numpy(x)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    # ---- Misc ----

    def test_errstate_is_context_manager(
        self, backend_name: str, set_test_backend: None
    ) -> None:
        with be.errstate(invalid="ignore"):
            _ = be.sqrt(be.array([-1.0]))  # should not raise


# ---------------------------------------------------------------------------
# TestNumpyCapabilityErrors — numpy-only, always uses numpy backend
# ---------------------------------------------------------------------------


class TestNumpyCapabilityErrors:
    """Verify that torch-only features raise BackendCapabilityError on NumPy."""

    def setup_method(self) -> None:
        be.set_backend("numpy")

    def test_grad_mode_raises(self) -> None:
        with pytest.raises(BackendCapabilityError, match="grad_mode"):
            _ = be.grad_mode

    def test_set_device_raises(self) -> None:
        with pytest.raises(BackendCapabilityError, match="set_device"):
            be.set_device("cpu")

    def test_get_device_raises(self) -> None:
        with pytest.raises(BackendCapabilityError, match="get_device"):
            be.get_device()

    def test_get_complex_precision_raises(self) -> None:
        with pytest.raises(BackendCapabilityError, match="get_complex_precision"):
            be.get_complex_precision()

    def test_autograd_raises(self) -> None:
        with pytest.raises(BackendCapabilityError, match="autograd"):
            _ = be.autograd


# ---------------------------------------------------------------------------
# TestTorchCapabilityOverrides — verify torch-only features work on TorchBackend
# ---------------------------------------------------------------------------


class TestTorchCapabilityOverrides:
    """Verify torch-only features function correctly on the TorchBackend."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        pytest.importorskip("torch")
        if "torch" not in be.list_available_backends():
            pytest.skip("torch backend not available")
        be.set_backend("torch")

    def test_grad_mode_accessible(self) -> None:
        gm = be.grad_mode
        assert hasattr(gm, "enable")
        assert hasattr(gm, "disable")
        assert hasattr(gm, "requires_grad")

    def test_set_get_device(self) -> None:
        be.set_device("cpu")
        assert be.get_device() == "cpu"

    def test_get_complex_precision(self) -> None:
        import torch

        be.set_precision("float32")
        assert be.get_complex_precision() == torch.complex64
        be.set_precision("float64")
        assert be.get_complex_precision() == torch.complex128
        be.set_precision("float32")  # restore

    def test_supports_gradients(self) -> None:
        assert be.supports_gradients is True

    def test_copy_to(self) -> None:
        import torch

        src = be.array([1.0, 2.0, 3.0])
        dst = be.zeros((3,))
        be.copy_to(src, dst)
        assert torch.allclose(src, dst)
