"""Regression tests for the upstream-suite triage round 1 (M3).

One test per root cause found by ``scripts/metal_suite.py`` under ``torch-mps``
(``NOTES/05-m2-status.md`` section 8). The oracle is torch CPU float64 or the
numpy backend; structural results are exact, arithmetic uses the df64 / sf64
tolerances of the other dispatch test modules.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("torch MPS (Metal GPU) is not available", allow_module_level=True)

import optiland.backend as be  # noqa: E402
from optiland.backend.torch_backend import metal  # noqa: E402
from optiland.backend.torch_backend.metal import tensor as mt  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import (  # noqa: E402
    MetalFloat64,
    is_metal,
)

MODES = ("df64", "sf64")
RTOL = {"df64": 1e-13, "sf64": 1e-15}
MPS = torch.device("mps")


@pytest.fixture
def mps_backend():
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.enable()
    be.metal_reset_stats()
    yield
    be.grad_mode.disable()
    be.set_backend("numpy")


def mk(a, mode: str = "df64", requires_grad: bool = False, host=None):
    return MetalFloat64.from_numpy(
        np.asarray(a, dtype=np.float64), mode, requires_grad=requires_grad, host=host
    )


# ---------------------------------------------------------------------------
# (1) plain leaves that require grad stay differentiable through be.asarray/cast
# ---------------------------------------------------------------------------
def test_as_tensor_keeps_the_graph_of_a_cpu_leaf(mps_backend):
    p = torch.nn.Parameter(torch.tensor([1.5], dtype=torch.float64))
    x = be.asarray(p)
    assert is_metal(x) and x.requires_grad
    (x * 3.0).sum().backward()
    assert p.grad is not None
    np.testing.assert_array_equal(p.grad.numpy(), [3.0])
    # cast() is the same path; a leaf without grad is still detached exactly
    q = torch.tensor([2.0, 4.0], dtype=torch.float64)
    y = be.cast(q)
    assert is_metal(y) and not y.requires_grad
    np.testing.assert_array_equal(be.to_numpy(y), [2.0, 4.0])
    # a plain mps float32 leaf requiring grad goes the same route
    r = torch.tensor([0.5], dtype=torch.float32, device=MPS, requires_grad=True)
    z = be.asarray(r)
    assert is_metal(z) and z.requires_grad
    (z * 2.0).sum().backward()
    np.testing.assert_array_equal(r.grad.cpu().numpy(), [2.0])


def test_asarray_of_a_metal_tensor_under_float32_precision_is_kept(mps_backend):
    x = be.array([1.5])
    be.set_precision("float32")
    try:
        y = be.asarray(x, dtype=None)
        assert y is x
        z = be.asarray(np.array([0.25]), dtype=np.float64)
        assert is_metal(z)
        w = be.asarray([1.0, 2.0])
        assert not is_metal(w) and w.dtype == torch.float32 and w.device.type == "mps"
    finally:
        be.set_precision("float64")


# ---------------------------------------------------------------------------
# (2) device reconciliation of plain operands the host path split apart
# ---------------------------------------------------------------------------
def test_cpu_bool_mask_indexed_by_mps_indices(mps_backend):
    hopeless = be.zeros(8) > 0.0  # host path: CPU bool
    assert hopeless.device.type == "cpu"
    idx = torch.tensor([1, 3, 5], device=MPS)
    sel = hopeless[idx]  # index.Tensor with mps indices: self moves to mps
    assert sel.device.type == "mps" and sel.shape == (3,)
    assert not sel.any()
    # in-place write into the CPU mask with mps indices / values stays on the CPU
    hopeless[idx] = torch.tensor([True, False, True], device=MPS)
    assert hopeless.device.type == "cpu"
    np.testing.assert_array_equal(hopeless.numpy(), [0, 1, 0, 0, 0, 1, 0, 0])
    # a 0-d CPU integer index is a scalar to torch: nothing moves
    assert bool(hopeless[torch.tensor(1)]) is True


def test_mps_mask_with_cpu_complex_operands_moves_to_the_cpu(mps_backend):
    values = np.linspace(-1, 1, 300)
    x = mk(values, host=False)
    mask = x > 0.0  # kernel predicate (300 > HOST_THRESHOLD): mps bool
    assert mask.device.type == "mps"
    p = torch.arange(300, dtype=torch.float64).to(torch.complex128)  # CPU complex
    out = be.where(mask, 0.0 + 0.0j, p)
    assert out.device.type == "cpu" and out.dtype == torch.complex128
    np.testing.assert_array_equal(out.numpy(), np.where(values > 0, 0, np.arange(300)))


def test_complex128_factories_aimed_at_mps_live_on_the_cpu(mps_backend):
    _ = be.array([1.0])  # arm the factory intercept
    z = torch.zeros(4, dtype=torch.complex128, device="mps")
    assert z.device.type == "cpu" and z.dtype == torch.complex128
    f = torch.tensor([1.0, 2.0], device="mps").to(torch.complex128)
    assert f.device.type == "cpu"
    np.testing.assert_array_equal(f.numpy(), [1 + 0j, 2 + 0j])
    stats = be.metal_stats()
    assert any(k.startswith("cpu_complex:") for k in stats), stats


def test_abs_real_imag_of_cpu_complex_come_back_as_metal(mps_backend):
    e = torch.tensor([1 + 2j, -3 + 0.5j], dtype=torch.complex128)
    a = be.abs(e)
    assert is_metal(a)
    np.testing.assert_allclose(be.to_numpy(a), np.abs(e.numpy()), rtol=1e-15)
    r, i = be.real(e), be.imag(e)
    assert is_metal(r) and is_metal(i)
    intensity = be.zeros(2) + be.sum(be.abs(e) ** 2)  # PolarizedRays.update_intensity
    assert is_metal(intensity)
    np.testing.assert_allclose(
        be.to_numpy(intensity), np.abs(e.numpy()) ** 2 @ [1, 1], rtol=1e-15
    )


# ---------------------------------------------------------------------------
# (3) reflection / replication padding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("pad_mode", ["reflect", "replicate"])
def test_pad_modes_match_the_cpu_exactly(mode, pad_mode):
    a = np.random.default_rng(0).normal(size=(1, 1, 6, 7))
    x = mk(a, mode, host=False)
    x.requires_grad_(True)
    leaf = torch.tensor(be.to_numpy(x), requires_grad=True)  # the decoded values
    ref = torch.nn.functional.pad(leaf, (2, 1, 1, 2), mode=pad_mode)
    out = torch.nn.functional.pad(x, (2, 1, 1, 2), mode=pad_mode)
    assert is_metal(out) and out.mode == mode
    np.testing.assert_array_equal(be.to_numpy(out), ref.detach().numpy())
    out.sum().backward()
    ref.sum().backward()
    np.testing.assert_array_equal(be.to_numpy(x.grad), leaf.grad.numpy())


# ---------------------------------------------------------------------------
# (4) backend-level fallbacks keep autograd (grid_sample, polyfit)
# ---------------------------------------------------------------------------
def test_grid_sample_fallback_keeps_gradients(mps_backend):
    grid_vals = be.array(np.arange(16, dtype=float).reshape(1, 1, 4, 4))
    grid_vals.requires_grad_(True)
    grid = be.array(np.zeros((1, 1, 1, 2)))
    grid.requires_grad_(True)
    out = be.grid_sample(grid_vals, grid, mode="bilinear", align_corners=True)
    assert is_metal(out) and out.requires_grad
    out.sum().backward()
    assert grid_vals.grad is not None and grid.grad is not None
    assert float(be.to_numpy(grid_vals.grad).sum()) == pytest.approx(1.0)
    assert be.metal_stats().get("cpu_fallback:grid_sample", 0) == 1


def test_fallback_autograd_is_strict_mode_checked(monkeypatch):
    monkeypatch.setenv("OPTILAND_METAL_STRICT", "1")
    with pytest.raises(mt.MetalFallbackError):
        mt.cpu_fallback_autograd(lambda t: t, (mk([1.0]),), {}, "probe")


# ---------------------------------------------------------------------------
# (5) repr follows torch's print options (stable under sub-repr changes)
# ---------------------------------------------------------------------------
def test_repr_uses_torch_formatting():
    a = mk(0.05)
    b = mk(0.05 + 1e-5)
    assert repr(a) == repr(b)
    assert "0.0500" in repr(a) and "mode='df64'" in repr(a)
    assert "requires_grad=True" in repr(mk([1.0, 2.0], requires_grad=True))
    assert repr(mk(np.zeros((2, 0)))) == "MetalFloat64([], mode='df64')"


# ---------------------------------------------------------------------------
# (6) cross product is exactly antisymmetric (cross(a, a) == 0)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_cross_of_a_vector_with_itself_is_exactly_zero(mode):
    a = np.random.default_rng(1).normal(size=(300, 3))
    x = mk(a, mode, host=False)
    z = torch.linalg.cross(x, x)
    np.testing.assert_array_equal(be.to_numpy(z), np.zeros((300, 3)))
    b = np.random.default_rng(2).normal(size=(300, 3))
    y = mk(b, mode, host=False)
    np.testing.assert_allclose(
        be.to_numpy(torch.linalg.cross(x, y)),
        np.cross(a, b),
        rtol=RTOL[mode],
        atol=1e-14,
    )


# ---------------------------------------------------------------------------
# (7) fused conic intersection with Python-float radius / conic differentiates
# ---------------------------------------------------------------------------
def test_conic_backward_with_plain_parameters(mps_backend):
    from optiland.backend.torch_backend.metal.conic import conic_intersection_metal

    n = 300
    rng = np.random.default_rng(3)
    x = mk(rng.uniform(-1, 1, n), requires_grad=True, host=False)
    y = mk(rng.uniform(-1, 1, n), requires_grad=True, host=False)
    z = mk(np.zeros(n), host=False)
    L = mk(np.zeros(n), host=False)
    M = mk(np.zeros(n), host=False)
    N = mk(np.ones(n), host=False)
    t = conic_intersection_metal(x, y, z, L, M, N, 25.0, 0.0)
    t.sum().backward()
    assert x.grad is not None and np.isfinite(be.to_numpy(x.grad)).all()
    # sag of a sphere: t = R - sqrt(R^2 - r^2); dt/dx = x / sqrt(R^2 - r^2)
    r2 = be.to_numpy(x) ** 2 + be.to_numpy(y) ** 2
    np.testing.assert_allclose(
        be.to_numpy(x.grad), be.to_numpy(x) / np.sqrt(25.0**2 - r2), rtol=1e-10
    )


# ---------------------------------------------------------------------------
# (8) the suite runner pins a non-interactive matplotlib backend
# ---------------------------------------------------------------------------
def test_suite_runner_pins_agg_backend():
    from pathlib import Path

    src = (Path(__file__).parents[2] / "scripts" / "metal_suite.py").read_text()
    assert 'MPLBACKEND="Agg"' in src


def test_metal_mode_switch_is_restored():
    assert metal.get_mode() == "df64"
