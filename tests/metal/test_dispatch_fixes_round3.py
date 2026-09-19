"""Regression tests for the M2 dispatch-layer fix round 3.

One test per verified failure (see ``NOTES/05-m2-status.md`` section 9), plus
one lock per documented limitation. The oracle is torch CPU float64 on the
*decoded* inputs; structural results and error behaviour are exact, arithmetic
chains use the df64 / sf64 tolerances of the other dispatch test modules,
autograd uses rtol 1e-12.
"""

from __future__ import annotations

import os
from fractions import Fraction

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("torch MPS (Metal GPU) is not available", allow_module_level=True)

from optiland.backend.torch_backend import metal  # noqa: E402
from optiland.backend.torch_backend.metal import encode  # noqa: E402
from optiland.backend.torch_backend.metal import tensor as mt  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import MetalFloat64  # noqa: E402

ops_reduce = mt.ops_reduce  # imported by the nucleus (import order matters)

MODES = ("df64", "sf64")
RTOL = {"df64": 1e-13, "sf64": 1e-15}
GRAD_RTOL = 1e-12
MPS = torch.device("mps")
DEFAULT_THRESHOLD = mt.DEFAULT_HOST_THRESHOLD
OVERLAP_ERROR = "refer to a single memory location"


@pytest.fixture(autouse=True)
def gpu_resident(monkeypatch):
    """Every tensor GPU-resident unless a test opts into dual residency."""
    monkeypatch.setattr(mt, "HOST_THRESHOLD", 0)


@pytest.fixture
def dual_residency(monkeypatch):
    """The default residency threshold (small tensors host-resident)."""
    monkeypatch.setattr(mt, "HOST_THRESHOLD", DEFAULT_THRESHOLD)


@pytest.fixture
def other_global_mode():
    """Set the global representation to the *other* one for the duration of a test."""
    saved = metal.get_mode()

    def flip(mode: str) -> None:
        metal.set_mode("sf64" if mode == "df64" else "df64")

    yield flip
    metal.set_mode(saved)


def rt(a) -> np.ndarray:
    """Round through df64 so the GPU operand and the CPU reference are identical."""
    a = np.asarray(a, dtype=np.float64)
    hi, lo = encode.encode_df64(a)
    return np.asarray(encode.decode_df64(hi, lo), dtype=np.float64).reshape(a.shape)


def mk(a, mode: str = "df64", requires_grad: bool = False, **kw) -> MetalFloat64:
    return MetalFloat64.from_numpy(
        np.asarray(a, dtype=np.float64), mode, requires_grad=requires_grad, **kw
    )


def cpu(a, requires_grad: bool = False) -> torch.Tensor:
    t = torch.from_numpy(np.array(a, dtype=np.float64))
    return t.requires_grad_(requires_grad)


def dec(x) -> np.ndarray:
    return x.to_numpy() if isinstance(x, MetalFloat64) else x.detach().cpu().numpy()


def assert_exact(got, want) -> None:
    assert isinstance(got, torch.Tensor)
    assert tuple(got.shape) == tuple(want.shape), (got.shape, want.shape)
    g, w = dec(got), want.detach().numpy()
    assert np.array_equal(g, w, equal_nan=True), (g, w)
    if w.dtype.kind == "f":
        assert np.array_equal(np.signbit(g), np.signbit(w)), "sign of zero differs"


def assert_close(got, want, mode: str, rtol: float | None = None) -> None:
    assert isinstance(got, MetalFloat64) and tuple(got.shape) == tuple(want.shape)
    np.testing.assert_allclose(
        got.to_numpy(), want.detach().numpy(), rtol=rtol or RTOL[mode], atol=0
    )


def assert_grad_close(got, want) -> None:
    assert isinstance(got, MetalFloat64)
    scale = float(np.abs(want.detach().numpy()).max()) or 1.0
    np.testing.assert_allclose(
        got.to_numpy(), want.detach().numpy(), rtol=GRAD_RTOL, atol=GRAD_RTOL * scale
    )


def factory(*shape, dtype=torch.float64) -> torch.Tensor:
    """A tensor-less factory call (served by ``MetalFactoryMode``)."""
    return torch.empty(*shape, dtype=dtype, device=MPS)


def ulps_apart(a: np.ndarray, b: np.ndarray) -> int:
    return int((a.view(np.int64) != b.view(np.int64)).sum())


# ---------------------------------------------------------------------------
# Representation-agnostic factory tensors absorbing fixed data in place
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_agnostic_factory_absorbs_in_place_writes_of_a_fixed_tensor(
    mode, other_global_mode
):
    other_global_mode(mode)  # factory tensors are made in the *other* mode
    a0 = rt(np.random.default_rng(0).standard_normal((40, 40)))
    a, ac = mk(a0, mode), cpu(a0)
    ones = torch.ones(40, 40, dtype=torch.bool)
    idx = torch.tensor([0])
    zero_row = torch.zeros(1, 40, dtype=torch.int64)

    def dev(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return t.to(s.device)

    cases = [
        ("copy_", lambda b, s: b.copy_(s)),
        ("setitem", lambda b, s: b.__setitem__(0, s[0])),
        (
            "setitem slice",
            lambda b, s: b.__setitem__((slice(1, 3), slice(None)), s[1:3]),
        ),
        ("fill_", lambda b, s: b.fill_(s[0, 0])),
        ("masked_fill_", lambda b, s: b.masked_fill_(dev(ones, s), s[0, 0])),
        ("index_put_", lambda b, s: b.index_put_((dev(idx, s),), s[:1])),
        ("scatter_", lambda b, s: b.scatter_(0, dev(zero_row, s), s[:1])),
    ]
    for name, op in cases:
        buf = factory(40, 40)
        assert buf._agnostic and buf.mode != mode, name
        ref = torch.zeros(40, 40, dtype=torch.float64)
        buf.zero_()
        op(buf, a)
        op(ref, ac)
        assert buf.mode == mode, name
        assert not buf._agnostic, name
        assert_exact(buf, ref)
        assert_close(a + buf, ac + ref, mode)  # still combinable with the fixed source

    # out= forms whose destination is a factory buffer
    values, indices = factory(40, 40), factory(40, 40, dtype=torch.int64)
    torch.sort(a, out=(values, indices))
    rv, ri = torch.sort(ac)
    assert_exact(values, rv)
    assert_exact(indices, ri)
    assert values.mode == mode and not values._agnostic
    out = factory(40, 40, dtype=torch.int64)
    torch.ops.aten.argsort.stable_out(a, stable=True, out=out)
    assert_exact(out, torch.argsort(ac, stable=True))
    out = factory(40, 40)
    torch.add(a, 1.0, out=out)
    assert out.mode == mode and not out._agnostic
    assert_close(out, ac + 1.0, mode)

    # a failed write leaves the tagging as it was
    buf = factory(40, 40)
    with pytest.raises(RuntimeError):
        buf.copy_(a[:3])  # shape mismatch
    assert buf._agnostic
    assert_close(a + buf, ac + buf.to_cpu_float64(), mode)


@pytest.mark.parametrize("mode", MODES)
def test_views_of_a_factory_buffer_keep_aliasing_across_the_retag(
    mode, other_global_mode
):
    other_global_mode(mode)
    a = mk(np.arange(16.0).reshape(4, 4), mode)
    buf = factory(4, 4)
    buf.zero_()
    r0, r2 = buf[0], buf[2]  # views taken before the buffer changes mode
    r0.copy_(a[0])  # re-tags the base through the view
    assert buf.mode == mode and not buf._agnostic
    r2.copy_(a[2])  # a view left over from before the re-tag is re-attached
    want = torch.zeros(4, 4, dtype=torch.float64)
    want[0] = torch.arange(4.0)
    want[2] = torch.arange(8.0, 12.0)
    assert_exact(buf, want)
    assert_exact(buf[2], want[2])


# ---------------------------------------------------------------------------
# CPU index / mask tensors (host-path results) on GPU-resident tensors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_index_ops_accept_cpu_index_and_mask_tensors(mode):
    m0 = np.arange(6.0).reshape(2, 3)
    m, mc = mk(m0, mode), cpu(m0)
    idx = torch.tensor([1, 0])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    assert_exact(torch.index_select(m, 1, idx), torch.index_select(mc, 1, idx))
    assert_exact(
        torch.gather(m, 1, idx.view(2, 1)), torch.gather(mc, 1, idx.view(2, 1))
    )
    assert_exact(m.masked_fill(mask, 0.0), mc.masked_fill(mask, 0.0))
    assert_exact(m.clone().masked_fill_(mask, 0.0), mc.clone().masked_fill_(mask, 0.0))
    assert_exact(m.masked_fill(mask, m[0, 1]), mc.masked_fill(mask, mc[0, 1]))
    assert_exact(
        torch.repeat_interleave(m, idx + 1, dim=0),
        torch.repeat_interleave(mc, idx + 1, dim=0),
    )
    assert_exact(m.index_fill(0, idx, 1.0), mc.index_fill(0, idx, 1.0))
    assert_exact(m.index_fill(0, idx, m[0, 1]), mc.index_fill(0, idx, mc[0, 1]))
    assert_exact(
        m.clone().index_fill_(0, idx, -1.0), mc.clone().index_fill_(0, idx, -1.0)
    )
    assert_exact(torch.masked_select(m, mask), torch.masked_select(mc, mask))


def test_gpu_resident_index_ops_accept_host_path_indices(dual_residency):
    big0 = rt(np.random.default_rng(1).standard_normal((300, 4)))
    small0 = np.random.default_rng(2).standard_normal(10)
    big, bigc = mk(big0), cpu(big0)
    small, smallc = mk(small0), cpu(small0)
    assert not big.is_host_resident and small.is_host_resident
    idx = torch.argsort(small)
    assert idx.device.type == "cpu"  # the host path returns its indices there
    assert_exact(big.index_select(0, idx), bigc.index_select(0, torch.argsort(smallc)))
    g = idx.view(10, 1).expand(10, 4)
    assert_exact(
        big.gather(0, g),
        bigc.gather(0, torch.argsort(smallc).view(10, 1).expand(10, 4)),
    )
    assert_exact(
        big.index_fill(0, idx, 1.0), bigc.index_fill(0, torch.argsort(smallc), 1.0)
    )
    pos = torch.searchsorted(torch.sort(small)[0], small)
    assert_exact(
        big.index_select(0, pos),
        bigc.index_select(0, torch.searchsorted(torch.sort(smallc)[0], smallc)),
    )
    nz = torch.nonzero(small > 0).flatten()
    assert_exact(
        big.index_select(0, nz),
        bigc.index_select(0, torch.nonzero(smallc > 0).flatten()),
    )


# ---------------------------------------------------------------------------
# scatter with an empty index of a different rank
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_scatter_with_empty_index_of_other_rank_is_a_noop(mode):
    x0 = np.arange(20.0).reshape(4, 5)
    idx = torch.empty(0, 0, 0, dtype=torch.int64, device=MPS)
    src = mk(np.zeros((0, 0, 0)), mode)
    want = cpu(x0)
    assert_exact(mk(x0, mode).scatter_(0, idx, src), want)
    assert_exact(mk(x0, mode).scatter_add_(0, idx, src), want)
    assert_exact(mk(x0, mode).scatter_(0, idx, 1.0), want)
    assert_exact(torch.scatter(mk(x0, mode), 0, idx, src), want)
    assert_exact(torch.scatter_add(mk(x0, mode), 0, idx, src), want)


# ---------------------------------------------------------------------------
# sf64 add / sub with alpha and a scalar other: fused like torch
# ---------------------------------------------------------------------------
def test_sf64_add_sub_alpha_with_scalar_other_is_bit_exact():
    a = np.r_[-1.11, np.random.default_rng(0).standard_normal(999)]
    x, xc = mk(a, "sf64"), cpu(a)
    for alpha in (3.7, -0.3):
        for other in (0.3, torch.tensor(0.3, dtype=torch.float64)):
            assert (
                ulps_apart(
                    torch.add(x, other, alpha=alpha).to_numpy(),
                    torch.add(xc, other, alpha=alpha).numpy(),
                )
                == 0
            )
            assert (
                ulps_apart(
                    torch.sub(x, other, alpha=alpha).to_numpy(),
                    torch.sub(xc, other, alpha=alpha).numpy(),
                )
                == 0
            )
            assert (
                ulps_apart(
                    x.clone().add_(other, alpha=alpha).to_numpy(),
                    xc.clone().add_(other, alpha=alpha).numpy(),
                )
                == 0
            )
    # the cancellation case the two-rounding path got wrong (0.0 vs -8.5e-17)
    t = torch.add(xc, 0.3, alpha=3.7)[0].item()
    assert t == float(Fraction(-1.11) + Fraction(3.7) * Fraction(0.3))
    assert torch.add(x, 0.3, alpha=3.7).to_numpy()[0] == t


def test_sf64_alpha_with_host_resident_0d_other_is_bit_exact(dual_residency):
    a = np.random.default_rng(0).uniform(-3, 3, 4096)
    x, xc = mk(a, "sf64"), cpu(a)
    other = mk(np.float64(0.7), "sf64")
    assert other.is_host_resident
    assert (
        ulps_apart(
            torch.add(x, other, alpha=0.3).to_numpy(),
            torch.add(xc, 0.7, alpha=0.3).numpy(),
        )
        == 0
    )
    assert (
        ulps_apart(
            torch.add(x, 0.7, alpha=0.3).to_numpy(),
            torch.add(xc, 0.7, alpha=0.3).numpy(),
        )
        == 0
    )


# ---------------------------------------------------------------------------
# logical_* with CPU bool / integer operands
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_logical_ops_accept_cpu_bool_and_integer_operands(mode):
    v = np.linspace(-1, 1, 1000)
    m = torch.tensor(v > 0)
    x, xc = mk(v, mode), cpu(v)
    for fn in (torch.logical_and, torch.logical_or, torch.logical_xor):
        got = fn(x, m)
        assert got.device.type == "mps" and got.dtype == torch.bool
        assert_exact(got, fn(xc, m))
        assert_exact(fn(x, m.to(torch.int64)), fn(xc, m.to(torch.int64)))
        assert_exact(fn(m, x), fn(m, xc))
    got = x.clone().logical_and_(m)
    assert_exact(got, (xc.clone().logical_and_(m)))
    # a CPU N-d *float* operand keeps torch's device error
    with pytest.raises(RuntimeError, match="same device"):
        torch.logical_and(x, torch.tensor(v))


# ---------------------------------------------------------------------------
# Host-resident results of scalar-scalar ops are fresh, writable tensors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_host_expansion_results_are_writable_and_contiguous(dual_residency, mode):
    n1 = mk([1.5], mode).expand(1000)
    n2 = mk([1.7], mode).expand(1000)
    u = n1 / n2
    assert u.is_host_resident and u.shape == (1000,)
    uc = torch.full((1000,), 1.5 / 1.7, dtype=torch.float64)
    u[0] = 5.0
    uc[0] = 5.0
    assert u.stride() == (1,) and u.is_contiguous()
    assert_exact(u, uc)
    u.mul_(2.0)
    uc.mul_(2.0)
    assert_exact(u, uc)
    for op in (
        lambda n: n * 2.0,
        lambda n: n + n,
        lambda n: n**2,
        lambda n: n.clamp(0, 1),
        lambda n: torch.maximum(n, n),
    ):
        r = op(n1)
        r.sqrt_()  # in-place on a fresh result works
        r.copy_(r)
        torch.mul(r, 1.0, out=r)
        v = r[:10]
        v.fill_(3.0)
        assert r.to_numpy()[0] == 3.0 and r.to_numpy()[10] != 3.0
    # gradients flow through a materialized result
    base = mk([1.5], mode, requires_grad=True)
    w = base.expand(1000) * 2.0
    w[0] = 0.0
    w.sum().backward()
    assert base.grad.to_numpy()[0] == 2.0 * 999


def test_views_of_large_host_resident_tensors_alias_the_host_copy(dual_residency):
    """Writes through a view of a host ``cat`` result reach the tensor."""
    c = torch.cat([mk(np.zeros(200)), mk(np.zeros(200))])
    assert c.is_host_resident and c.numel() > DEFAULT_THRESHOLD
    c[0] = 1.0
    c[:10].fill_(2.0)
    c.view(2, 200)[1, 0] = 3.0
    want = torch.zeros(400, dtype=torch.float64)
    want[0] = 1.0
    want[:10] = 2.0
    want[200] = 3.0
    assert_exact(c, want)


# ---------------------------------------------------------------------------
# floor_divide by a scalar and remainder backward under dual residency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_floor_divide_by_scalar_under_dual_residency(dual_residency, mode):
    a = rt(np.random.default_rng(0).uniform(-5, 5, 1000))
    x, xc = mk(a, mode), cpu(a)
    assert_exact(x // 3, xc // 3)
    assert_exact(torch.floor_divide(x, 0.3), torch.floor_divide(xc, 0.3))
    assert_exact(
        torch.div(x, 0.3, rounding_mode="floor"),
        torch.div(xc, 0.3, rounding_mode="floor"),
    )
    assert_exact(x.clone().floor_divide_(2.0), xc.clone().floor_divide_(2.0))
    d, dc = mk(np.array(0.7), mode, True), cpu(0.7, True)
    assert d.is_host_resident
    torch.remainder(x, d).sum().backward()
    torch.remainder(xc, dc).sum().backward()
    assert_grad_close(d.grad, dc.grad)


# ---------------------------------------------------------------------------
# copysign backward w.r.t. a host-resident scalar, masked_fill with a 0-d CPU bool
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_copysign_backward_and_masked_fill_with_host_path_bool(dual_residency, mode):
    a = rt(np.random.default_rng(0).uniform(-5, 5, 1000))
    x, xc = mk(a, mode), cpu(a)
    f, fc = mk(np.array(1.5), mode, True), cpu(1.5, True)
    torch.copysign(f, x).sum().backward()
    torch.copysign(fc, xc).sum().backward()
    assert_grad_close(f.grad, fc.grad)
    d = mk(np.array(0.7), mode)
    assert (d == 0).device.type == "cpu"  # the host path's 0-d bool
    assert_exact(x.masked_fill(d == 0, 1.0), xc)
    assert_exact(x.masked_fill(d == 0.7, 1.0), torch.ones(1000, dtype=torch.float64))


# ---------------------------------------------------------------------------
# all / any with dim=[] and argmax / argmin with dim on empty tensors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_all_any_with_empty_dim_list_reduce_nothing(mode):
    a = np.ones((20, 30))
    a[3, 4] = 0.0
    x, xc = mk(a, mode), cpu(a)
    assert_exact(torch.all(x, dim=[]), torch.all(xc, dim=[]))
    assert_exact(
        torch.any(x, dim=[], keepdim=True), torch.any(xc, dim=[], keepdim=True)
    )
    e = mk(np.zeros((0, 3)), mode)
    assert torch.all(e, dim=[]).shape == (0, 3)
    assert torch.all(x, dim=None).shape == ()  # None still reduces everything


@pytest.mark.parametrize("mode", MODES)
def test_argmax_argmin_with_dim_on_empty_tensors(mode):
    for shape, dim in (((3, 0), 0), ((0, 3), 1), ((2, 0, 4), 0), ((2, 0, 4), 2)):
        for keepdim in (False, True):
            for fn in (torch.argmax, torch.argmin):
                got = fn(mk(np.zeros(shape), mode), dim=dim, keepdim=keepdim)
                want = fn(
                    torch.zeros(shape, dtype=torch.float64), dim=dim, keepdim=keepdim
                )
                assert_exact(got, want)
    with pytest.raises(
        RuntimeError, match="Expected reduction dim 1 to have non-zero size"
    ):
        torch.argmax(mk(np.zeros((3, 0)), mode), dim=1)
    with pytest.raises(RuntimeError, match="Expected reduction dim to be specified"):
        torch.argmax(mk(np.zeros((3, 0)), mode))


# ---------------------------------------------------------------------------
# denormal float32 words of plain operands are flushed on promotion
# ---------------------------------------------------------------------------
def test_coerce_flushes_denormal_float32_words_like_the_encoder():
    p = torch.tensor(
        [1e-40, -1e-40, 1e-39, 0.0, 1.0, 2.0], dtype=torch.float32, device=MPS
    )
    z = mk(np.zeros(6))
    w = torch.where(torch.ones(6, dtype=torch.bool, device=MPS), p, z)
    want = torch.tensor([0.0, -0.0, 0.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    assert_exact(w, want)  # including the sign of the flushed words
    assert_exact(mk(dec(w)), want)
    assert torch.sort(w)[1].tolist() == torch.sort(want)[1].tolist()
    assert torch.nonzero(w).flatten().tolist() == [4, 5]
    assert (w == 0).tolist() == [True, True, True, True, False, False]
    assert_exact(
        torch.cat([z, p]), torch.cat([torch.zeros(6, dtype=torch.float64), want])
    )


# ---------------------------------------------------------------------------
# take / put_ / index_copy_ reject int32 indices as torch does
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_take_put_index_copy_reject_int32_indices(mode):
    x = mk(np.arange(20.0).reshape(4, 5), mode)
    i32 = torch.tensor([1], dtype=torch.int32, device=MPS)
    with pytest.raises(
        RuntimeError, match="take\\(\\): Expected a long tensor for index, but got Int"
    ):
        torch.take(x, i32)
    with pytest.raises(RuntimeError, match="index_copy_\\(\\): Expected a long tensor"):
        x.clone().index_copy_(0, i32, x[:1])
    with pytest.raises(RuntimeError, match="put_\\(\\): Expected a long tensor"):
        x.clone().put_(i32, x.flatten()[:1])
    with pytest.raises(IndexError, match="index_fill_\\(\\): Expected dtype int64"):
        x.clone().index_fill_(0, i32, 1.0)
    # index_select / gather / scatter keep accepting int32, as torch does
    assert_exact(
        torch.index_select(x, 0, i32),
        torch.index_select(cpu(np.arange(20.0).reshape(4, 5)), 0, i32.cpu()),
    )


# ---------------------------------------------------------------------------
# Documented limitations (locks)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_documented_partially_overlapping_sources_use_snapshot_semantics(mode):
    a = rt(np.random.default_rng(1).standard_normal((6, 5)))
    c = torch.tensor(a)
    with pytest.raises(RuntimeError, match=OVERLAP_ERROR):
        c.index_copy_(0, torch.tensor([5, 0]), c[0:2])  # torch's rule
    x = mk(a, mode)
    idx = torch.tensor([5, 0], device=MPS)
    got = x.clone().index_copy_(0, idx, x[0:2])
    want = c.clone().index_copy_(0, idx.cpu(), c[0:2].clone())  # snapshot of the source
    assert_exact(got, want)
    got = x.clone()
    got[1:].add_(got[:-1])
    want = c.clone()
    want[1:] = want[1:] + want[:-1].clone()
    assert_close(got, want, mode)
    got = torch.index_select(
        x, 0, torch.tensor([1, 0, 2, 3, 4, 5], device=MPS), out=x.clone()
    )
    assert_exact(got, torch.index_select(c, 0, torch.tensor([1, 0, 2, 3, 4, 5])))


@pytest.mark.parametrize("mode", MODES)
def test_documented_index_leniency(mode):
    x0 = np.arange(20.0).reshape(4, 5)
    x, xc = mk(x0, mode), cpu(x0)
    icpu = torch.tensor([1])
    # CPU indices are moved to the device (torch raises a device mismatch)
    assert_exact(
        x.clone().scatter_(0, icpu.view(1, 1), 5.0),
        xc.clone().scatter_(0, icpu.view(1, 1), 5.0),
    )
    assert_exact(
        x.clone().index_add_(0, icpu, x[:1]), xc.clone().index_add_(0, icpu, xc[:1])
    )
    assert_exact(
        x.clone().index_copy_(0, icpu, x[:1]), xc.clone().index_copy_(0, icpu, xc[:1])
    )
    assert_exact(x.clone().put_(icpu, x[0, :1]), xc.clone().put_(icpu, xc[0, :1]))
    assert_exact(
        torch.searchsorted(x[0], x[1], sorter=torch.arange(5)),
        torch.searchsorted(xc[0], xc[1], sorter=torch.arange(5)),
    )
    # a 1-element source of any rank into a 0-d index_add_
    got = x[0, 0].clone().index_add_(0, torch.tensor([0], device=MPS), x[0, :1])
    assert dec(got) == 0.0
    # int32 out= for searchsorted (torch: output dtype mismatch)
    out = torch.zeros(5, dtype=torch.int32, device=MPS)
    torch.searchsorted(x[0], x[1], out=out)
    assert out.tolist() == torch.searchsorted(xc[0], xc[1]).tolist()


def test_documented_sf64_accumulating_writes_are_not_bit_identical_to_cpu():
    x = mk([1.0], "sf64")
    x.index_add_(0, torch.tensor([0, 0], device=MPS), mk([2.0**-53, 2.0**-53], "sf64"))
    r = torch.tensor([1.0], dtype=torch.float64)
    r.index_add_(
        0, torch.tensor([0, 0]), torch.tensor([2.0**-53, 2.0**-53], dtype=torch.float64)
    )
    assert float(r[0]) == 1.0  # sequential: (1 + 2^-53) + 2^-53
    assert float(x.to_numpy()[0]) == 1.0 + 2.0**-52  # tree: 1 + (2^-53 + 2^-53)
    np.testing.assert_allclose(x.to_numpy(), r.numpy(), rtol=1e-15)
    g = np.random.default_rng(99)
    a, s = g.standard_normal(2000), g.standard_normal(2000)
    x = mk(a, "sf64")
    x.index_add_(0, torch.arange(2000, device=MPS), mk(s, "sf64"), alpha=0.1)
    r = torch.tensor(a)
    r.index_add_(0, torch.arange(2000), torch.tensor(s), alpha=0.1)
    assert np.array_equal(x.to_numpy(), a + 0.1 * s)  # two roundings, as documented
    assert not np.array_equal(r.numpy(), a + 0.1 * s)  # torch CPU: one (fma)
    # the difference is the rounding of ``alpha * src`` (half an ulp of that
    # product) plus one ulp of the result
    np.testing.assert_allclose(
        x.to_numpy(),
        r.numpy(),
        rtol=2.0**-52,
        atol=2.0**-53 * float(np.abs(0.1 * s).max()),
    )


@pytest.mark.parametrize("mode", MODES)
def test_documented_lerp_accepts_plain_start_and_end(mode):
    a = rt(np.random.default_rng(0).uniform(-2, 2, 1000))
    x = mk(a, mode)
    f = torch.tensor(a, dtype=torch.float32, device=MPS)
    with pytest.raises(RuntimeError, match="expected dtype double for `end`"):
        torch.lerp(cpu(a), torch.tensor(a, dtype=torch.float32), 0.3)
    want = torch.lerp(
        cpu(a), torch.tensor(a, dtype=torch.float32).to(torch.float64), 0.3
    )
    assert_close(torch.lerp(x, f, 0.3), want, mode)
    assert_close(
        torch.lerp(f, x, 0.3),
        torch.lerp(torch.tensor(a, dtype=torch.float32).to(torch.float64), cpu(a), 0.3),
        mode,
    )
    got = f.clone().lerp_(x, 0.3)
    assert got.dtype == torch.float32


@pytest.mark.parametrize("mode", MODES)
def test_documented_inplace_with_partially_overlapping_other(mode):
    c = torch.arange(6.0, dtype=torch.float64)
    with pytest.raises(RuntimeError, match=OVERLAP_ERROR):
        c[1:].add_(c[:-1])
    x = mk(np.arange(6.0), mode)
    x[1:].add_(x[:-1])
    assert_exact(x, torch.tensor([0.0, 1.0, 3.0, 5.0, 7.0, 9.0], dtype=torch.float64))
    x = mk(np.arange(9.0).reshape(3, 3), mode)
    x.t().add_(x)
    want = torch.arange(9.0, dtype=torch.float64).reshape(3, 3)
    want = (want.t() + want).t()
    assert_exact(x, want)


def test_documented_out_into_the_other_fixed_representation_raises():
    with pytest.raises(TypeError, match="cannot mix representations"):
        torch.add(mk([1.0, 2.0], "df64"), 1.0, out=mk([0.0, 0.0], "sf64"))
    with pytest.raises(TypeError, match="cannot mix representations"):
        torch.add(mk([1.0, 2.0], "sf64"), 1.0, out=mk([0.0, 0.0], "df64"))
    # the same representation, and an agnostic factory out, work
    out = mk([0.0, 0.0], "sf64")
    torch.add(mk([1.0, 2.0], "sf64"), 1.0, out=out)
    assert_exact(out, torch.tensor([2.0, 3.0], dtype=torch.float64))


def test_documented_sf64_addcmul_matches_the_vectorized_path_only():
    rng = np.random.default_rng(5)
    for n in (7, 8, 1003):
        a, b, c = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
        g = torch.addcmul(*(mk(v, "sf64") for v in (a, b, c))).to_numpy()
        t = torch.addcmul(*(torch.from_numpy(v) for v in (a, b, c))).numpy()
        unfused = a + b * c  # numpy: separate roundings (the vectorized path)
        assert np.array_equal(g, unfused)
        head = n - n % 8
        assert np.array_equal(g[:head], t[:head])  # the vectorized part agrees
        np.testing.assert_allclose(g, t, rtol=3e-16)  # the tail is within one ulp
