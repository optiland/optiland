"""Nucleus behaviours of ``MetalFloat64``: residency, aliasing, scalars, factories."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover
    pytest.skip("Metal GPU required", allow_module_level=True)

from optiland.backend.torch_backend.metal import tensor as T  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import MetalFloat64  # noqa: E402


@pytest.fixture
def host_path(monkeypatch):
    monkeypatch.setattr(T, "HOST_THRESHOLD", 256)
    T.reset_stats()
    yield
    T.reset_stats()


def _host_ops() -> int:
    return sum(v for k, v in T.stats().items() if k.startswith("host:"))


def _gpu_ops() -> int:
    return sum(v for k, v in T.stats().items() if k.startswith("gpu:"))


@pytest.mark.parametrize("mode", ["df64", "sf64"])
def test_small_tensors_are_host_resident_and_exact(host_path, mode):
    a = MetalFloat64.from_numpy(np.array([1.0, 2.0, 3.0]), mode)
    b = MetalFloat64.from_numpy(np.array(0.1), mode)
    assert a.is_host_resident and b.is_host_resident
    assert a.device.type == "mps" and a.dtype == torch.float64
    c = a * b + 1
    assert c.is_host_resident
    np.testing.assert_array_equal(c.to_numpy(), np.array([1.0, 2.0, 3.0]) * 0.1 + 1)
    assert _gpu_ops() == 0 and _host_ops() >= 2
    assert c[1].item() == 1.2 and float(c[2]) == 1.3
    assert f"{b:.2f}" == "0.10"


def test_host_views_alias_and_invalidate_encoding(host_path):
    a = MetalFloat64.from_numpy(np.array([1.0, 2.0, 3.0]))
    hi_before = a._comps[0].cpu().numpy().copy()
    v = a[1:]
    v.mul_(10)
    np.testing.assert_array_equal(a.to_numpy(), [1.0, 20.0, 30.0])
    a[0] = 5.0  # Python scalar assignment goes through __setitem__ wrapping
    w = a[2]
    w.mul_(2)
    np.testing.assert_array_equal(a.to_numpy(), [5.0, 20.0, 60.0])
    np.testing.assert_array_equal(a._comps[0].cpu().numpy(), [5.0, 20.0, 60.0])
    assert not np.array_equal(hi_before, a._comps[0].cpu().numpy())


def test_large_tensors_and_their_views_stay_on_gpu(host_path):
    big = MetalFloat64.from_numpy(np.arange(600.0).reshape(20, 30))
    assert not big.is_host_resident
    v = big[:2, :3]
    assert not v.is_host_resident, "small views of GPU tensors must keep aliasing"
    v.zero_()
    out = big.to_numpy()
    assert out[:2, :3].sum() == 0 and out[2, 0] == 60.0
    small = big.sum() * 2 + big[0, 5]
    assert small.is_host_resident
    expected = np.arange(600.0).reshape(20, 30)
    expected[:2, :3] = 0
    assert abs(float(small) - (expected.sum() * 2 + 5)) < 1e-6


def test_mixed_host_gpu_and_predicates(host_path):
    big = MetalFloat64.from_numpy(np.linspace(0, 1, 1000))
    scalar = MetalFloat64.from_numpy(np.array(0.1))
    prod = big * scalar
    assert not prod.is_host_resident
    np.testing.assert_allclose(
        prod.to_numpy(), np.linspace(0, 1, 1000) * 0.1, rtol=1e-13
    )
    mask = big > 0.5
    assert mask.device.type == "mps"
    big[mask] = 0.0
    np.testing.assert_allclose(
        big.to_numpy(),
        np.where(np.linspace(0, 1, 1000) > 0.5, 0, np.linspace(0, 1, 1000)),
    )
    small = MetalFloat64.from_numpy(np.array([1.0, -2.0, 3.0]))
    m = small < 0
    assert m.device.type == "cpu"  # host-path predicates stay on the CPU (no sync)
    small[m] = 0
    np.testing.assert_array_equal(small.to_numpy(), [1.0, 0.0, 3.0])
    assert torch.where(m, small, 7.0).to_numpy().tolist() == [7.0, 0.0, 7.0]


def test_where_scalar_fix_on_gpu_tensor():
    x = MetalFloat64.from_numpy(np.linspace(-1, 1, 500), host=False)
    out = torch.where(x > 0, x, 1.0)  # torch composite would materialize float64 on mps
    ref = np.where(np.linspace(-1, 1, 500) > 0, np.linspace(-1, 1, 500), 1.0)
    np.testing.assert_allclose(out.to_numpy(), ref, rtol=1e-14)
    out2 = torch.where(x > 0, 2.0, x)
    np.testing.assert_allclose(
        out2.to_numpy(),
        np.where(np.linspace(-1, 1, 500) > 0, 2.0, np.linspace(-1, 1, 500)),
        rtol=1e-14,
    )


def test_cdist_decomposition():
    a = np.random.default_rng(0).uniform(-1, 1, (300, 2))
    b = np.random.default_rng(1).uniform(-1, 1, (7, 2))
    d = torch.cdist(
        MetalFloat64.from_numpy(a, host=False), MetalFloat64.from_numpy(b, host=False)
    )
    ref = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    np.testing.assert_allclose(d.to_numpy(), ref, rtol=1e-13, atol=1e-15)


def test_autograd_through_host_path_and_factory_intercept(host_path):
    x = MetalFloat64.from_numpy(np.array([0.5, 1.5]), requires_grad=True)
    y = (x * x * 3).sum()
    y.backward()
    np.testing.assert_array_equal(x.grad.to_numpy(), [3.0, 9.0])
    assert T.factory_intercept_enabled()
    z = torch.zeros((3,), dtype=torch.float64, device="mps")
    assert isinstance(z, MetalFloat64)
    xg = MetalFloat64.from_numpy(
        np.linspace(-1, 1, 400), host=False, requires_grad=True
    )
    yg = MetalFloat64.from_numpy(
        np.linspace(0.5, 2, 400), host=False, requires_grad=True
    )
    out = (torch.clamp(xg, -0.5, 0.5) * torch.copysign(yg, xg) + yg**xg).sum()
    gx, gy = torch.autograd.grad(out, (xg, yg))
    xc = torch.linspace(-1, 1, 400, dtype=torch.float64, requires_grad=True)
    yc = torch.linspace(0.5, 2, 400, dtype=torch.float64, requires_grad=True)
    outc = (torch.clamp(xc, -0.5, 0.5) * torch.copysign(yc, xc) + yc**xc).sum()
    gxc, gyc = torch.autograd.grad(outc, (xc, yc))
    np.testing.assert_allclose(gx.to_numpy(), gxc.numpy(), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(gy.to_numpy(), gyc.numpy(), rtol=1e-12, atol=1e-13)


def test_host_tensors_retag_to_gpu_mode(host_path):
    h = MetalFloat64.from_numpy(np.array(2.0))  # df64-tagged host scalar
    g = MetalFloat64.from_numpy(np.ones(1000), mode="sf64")
    r = g * h
    assert r.mode == "sf64" and h.mode == "sf64"
    np.testing.assert_array_equal(r.to_numpy(), 2.0)


@pytest.mark.parametrize("mode", ["df64", "sf64"])
def test_zero_dim_roundtrip(mode):
    s = MetalFloat64.from_numpy(np.array(3.5), mode, host=False)
    assert s.to_numpy().shape == () and float(s) == 3.5
    assert s.to_cpu_float64().shape == ()


def test_unimplemented_ops_raise():
    """GPU-resident operands with no handler raise (never run on the components).

    A small GPU tensor would be decoded and evaluated exactly on the host under
    the default dual residency, so the operand is larger than the threshold.
    """
    x = MetalFloat64.from_numpy(np.ones(T.HOST_THRESHOLD + 1), host=False)
    with pytest.raises(NotImplementedError):
        torch.ops.aten.special_bessel_j0.default(x)
    small = MetalFloat64.from_numpy(np.ones(3), host=False)
    if T.HOST_THRESHOLD > 0:
        # host path: exact CPU float64 evaluation of the unhandled op
        got = torch.ops.aten.special_bessel_j0.default(small)
        np.testing.assert_array_equal(
            got.to_numpy(), torch.special.bessel_j0(torch.ones(3, dtype=torch.float64))
        )


@pytest.mark.parametrize("mode", ["df64", "sf64"])
def test_complex_operands_run_on_cpu(mode):
    """Complex is not emulated: ops with a complex operand run on the CPU.

    The result is a CPU complex128 tensor (polarization / FFT policy); a complex
    tensor operand must never be silently truncated to its real part.
    """
    T.reset_stats()
    a = np.array([0.5, 1.0, -2.0])
    x = MetalFloat64.from_numpy(a, mode, host=False)
    c = torch.tensor([1 + 2j, 3j, -1.0 + 0j], dtype=torch.complex128)
    for got, ref in (
        (1j * x, 1j * torch.from_numpy(a)),
        (x * 1j, torch.from_numpy(a) * 1j),
        (torch.exp(1j * x), torch.exp(1j * torch.from_numpy(a))),
        (x * c, torch.from_numpy(a) * c),
        (c + x, c + torch.from_numpy(a)),
    ):
        assert got.device.type == "cpu" and got.dtype == torch.complex128
        torch.testing.assert_close(got, ref, rtol=1e-13, atol=0)
    assert sum(v for k, v in T.stats().items() if k.startswith("cpu_complex:")) >= 5
    with pytest.raises(TypeError):
        T.coerce(c, mode)
