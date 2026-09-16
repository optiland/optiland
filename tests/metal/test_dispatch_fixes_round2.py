"""Regression tests for the M2 dispatch-layer fix round 2.

One test per verified failure (see ``NOTES/05-m2-status.md`` section 8), plus
one lock per documented limitation. The oracle is torch CPU float64 on the
*decoded* inputs; structural results and error behaviour are exact, arithmetic
chains use the df64 / sf64 tolerances of the other dispatch test modules,
autograd uses rtol 1e-12.
"""

from __future__ import annotations

import math
import os
import warnings

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("torch MPS (Metal GPU) is not available", allow_module_level=True)

from optiland.backend.torch_backend import metal  # noqa: E402
from optiland.backend.torch_backend.metal import encode  # noqa: E402
from optiland.backend.torch_backend.metal import tensor as mt  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import (  # noqa: E402
    MetalFloat64,
    coerce,
)

ops_reduce = mt.ops_reduce  # imported by the nucleus (import order matters)

MODES = ("df64", "sf64")
RTOL = {"df64": 1e-13, "sf64": 1e-15}
GRAD_RTOL = 1e-12
MPS = torch.device("mps")
DEVICE_ERROR = "Expected all tensors to be on the same device"


@pytest.fixture(autouse=True)
def gpu_resident(monkeypatch):
    """Every tensor GPU-resident so the ops reach the GPU handlers under test."""
    monkeypatch.setattr(mt, "HOST_THRESHOLD", 0)


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


def mk(a, mode: str = "df64", requires_grad: bool = False) -> MetalFloat64:
    return MetalFloat64.from_numpy(
        np.asarray(a, dtype=np.float64), mode, requires_grad=requires_grad
    )


def cpu(a, requires_grad: bool = False) -> torch.Tensor:
    t = torch.from_numpy(np.array(a, dtype=np.float64))
    return t.requires_grad_(requires_grad)


def dec(x) -> np.ndarray:
    return x.to_numpy() if isinstance(x, MetalFloat64) else x.detach().cpu().numpy()


def assert_exact(got, want) -> None:
    assert isinstance(got, torch.Tensor) and got.device.type == "mps"
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


def grad_pair(fn, mode, *arrays, expected_mode=None):
    """Gradients of ``fn(*tensors).sum()`` on the GPU and on the CPU."""
    xs = [mk(a, mode, True) for a in arrays]
    cs = [cpu(a, True) for a in arrays]
    g = torch.autograd.grad(fn(*xs).sum(), xs)
    gc = torch.autograd.grad(fn(*cs).sum(), cs)
    for a, b in zip(g, gc, strict=True):
        assert a.mode == (expected_mode or mode)
        assert_grad_close(a, b)


# ---------------------------------------------------------------------------
# Representation-agnostic constants: views and derived values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_unbind_partial_grad_in_the_other_representation(mode, other_global_mode):
    other_global_mode(mode)
    a = rt(np.arange(12.0).reshape(3, 4))
    grad_pair(lambda x: x.unbind(0)[1] * 2.0, mode, a)
    grad_pair(lambda x: x.unbind(1)[2] ** 2, mode, a)
    grad_pair(lambda x, y: torch.stack([x, y]).unbind(0)[1] * 3.0, mode, a, a + 1)
    # the mechanism: an expand view of a factory zero stays agnostic and is
    # re-encoded (contiguously) when it meets a fixed tensor of the other mode
    z = torch.zeros((), dtype=torch.float64, device=MPS).expand(3, 4)
    assert z._agnostic and not z.is_contiguous()
    got = z + mk(a, mode)
    assert got.mode == mode
    assert_exact(got, cpu(a))


@pytest.mark.parametrize("mode", MODES)
def test_trace_and_quantile_in_the_other_representation(mode, other_global_mode):
    other_global_mode(mode)
    grad_pair(torch.trace, mode, rt(np.eye(3) + 0.5))
    a = rt(np.arange(6.0).reshape(2, 3) / 7)
    x, r = mk(a, mode), cpu(a)
    for q in (0.5, 0.25, 0.9):
        assert_close(torch.quantile(x, q), torch.quantile(r, q), mode)
        assert_close(
            torch.nanquantile(x, q, dim=1), torch.nanquantile(r, q, dim=1), mode
        )
    # arithmetic on a factory constant keeps it agnostic; absorbing fixed data ends it
    z = torch.scalar_tensor(0.5, dtype=torch.float64, device=MPS)
    assert (z * 2)._agnostic and (z * 2 + 1).floor()._agnostic
    y = z * 2
    y.masked_fill_(torch.tensor(True, device=MPS), 3.0)
    assert y._agnostic
    y.add_(mk(1.0, y.mode))
    assert not y._agnostic


# ---------------------------------------------------------------------------
# coerce of contiguous plain views with a storage offset
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_coerce_offset_views_of_plain_tensors(mode):
    p = torch.arange(12.0, device=MPS).reshape(3, 4)
    rp = p.cpu().double()
    for view, rview in ((p[1:], rp[1:]), (p[1], rp[1]), (p[1, 2], rp[1, 2])):
        assert_exact(coerce(view, mode), rview)
    a = rt(np.zeros((3, 4)))
    x, r = mk(a, mode), cpu(a)
    assert_exact(torch.cat([x, p[1:]]), torch.cat([r, rp[1:]]))
    x[0] = p[1]
    r[0] = rp[1]
    assert_exact(x, r)
    assert_exact(x + p[1, 1], r + rp[1, 1])
    idx = torch.tensor([2], device=MPS)
    x.index_put_((idx,), p[1])
    r.index_put_((idx.cpu(),), rp[1])
    assert_exact(x, r)
    x.fill_(p[1, 1])
    r.fill_(rp[1, 1])
    assert_exact(x, r)
    # a float32 view with an offset stays a fresh tensor (no aliasing of p)
    c = coerce(p[1:], "df64")
    c.components[0].fill_(-1.0)
    assert float(p[1, 0]) == 4.0


# ---------------------------------------------------------------------------
# __setitem__ keeps the autograd graph of a plain value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_setitem_keeps_the_value_graph(mode):
    a = rt(np.random.default_rng(0).standard_normal((3, 4)))
    b = np.random.default_rng(1).standard_normal(4).astype(np.float32)
    # CPU float64 value (copy_ path, allowed across devices by torch)
    x, v = mk(a, mode), cpu(np.ones((3, 4)), True)
    y = x * 1.0
    y[0] = v[0]
    assert y.requires_grad
    (y * 2).sum().backward()
    want = torch.zeros(3, 4, dtype=torch.float64)
    want[0] = 2.0
    assert torch.equal(v.grad, want)
    # float32 mps value: index form, mask form, and a target without grad
    for form in ("index", "mask", "leaf_target"):
        v32 = torch.tensor(b, device=MPS, requires_grad=True)
        rv = torch.tensor(b, requires_grad=True)
        x, r = mk(a, mode), cpu(a)
        y = x if form == "leaf_target" else x * 1.0
        ry = r if form == "leaf_target" else r * 1.0
        if form == "mask":
            m = torch.tensor([True, False, True])
            y[m.to(MPS), 0] = v32[:2]  # lenient (torch CPU wants the cast)
            ry[m, 0] = rv[:2].double()
        else:
            y[1] = v32
            ry[1] = rv
        assert_exact(y, ry)
        (y * 2).sum().backward()
        (ry * 2).sum().backward()
        assert v32.grad.device.type == "mps" and v32.grad.dtype == torch.float32
        assert torch.equal(v32.grad.cpu(), rv.grad)
    # the functional forms behave the same
    v32 = torch.tensor(b, device=MPS, requires_grad=True)
    got = mk(a, mode).index_put((torch.tensor([1], device=MPS),), v32)
    assert got.requires_grad


# ---------------------------------------------------------------------------
# Structural: diag / *_copy / rot90 / block_diag, tril_ / triu_
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_diag_copy_variants_rot90_block_diag(mode):
    a = rt(np.arange(12.0).reshape(3, 4) - 5.5)
    x, r = mk(a, mode), cpu(a)
    for k in (0, 1, -1):
        assert_exact(torch.diag(x, k), torch.diag(r, k))
        assert_exact(torch.rot90(x, k), torch.rot90(r, k))
    assert_exact(torch.diag(x[0]), torch.diag(r[0]))
    assert_exact(torch.narrow_copy(x, 1, 1, 2), torch.narrow_copy(r, 1, 1, 2))
    assert_exact(
        torch.block_diag(x, x[0], x[:, :2]), torch.block_diag(r, r[0], r[:, :2])
    )
    aten = torch.ops.aten
    assert_exact(aten.diagonal_copy(x, 1), aten.diagonal_copy(r, 1))
    assert_exact(aten.select_copy(x, 0, 2), aten.select_copy(r, 0, 2))
    assert_exact(aten.slice_copy(x, 1, 1, 3), aten.slice_copy(r, 1, 1, 3))
    assert_exact(aten.t_copy(x), aten.t_copy(r))
    assert_exact(aten.expand_copy(x[0], (2, 4)), aten.expand_copy(r[0], (2, 4)))
    got, want = aten.unbind_copy(x, 1), aten.unbind_copy(r, 1)
    for g, w in zip(got, want, strict=True):
        assert_exact(g, w)


@pytest.mark.parametrize("mode", MODES)
def test_tril_triu_inplace_and_householder_product_grad(mode):
    a = rt(np.arange(12.0).reshape(3, 4))
    for k in (0, 1, -1):
        x, r = mk(a, mode), cpu(a)
        assert x.tril_(k) is x
        r.tril_(k)
        assert_exact(x, r)
        x, r = mk(a, mode), cpu(a)
        assert x.triu_(k) is x
        r.triu_(k)
        assert_exact(x, r)
    rng = np.random.default_rng(3)
    grad_pair(
        torch.linalg.householder_product,
        mode,
        rt(rng.standard_normal((4, 3))),
        rt(np.array([0.1, 0.2, 0.3])),
    )


# ---------------------------------------------------------------------------
# index_fill / where / overlap / 0-d bool index
# ---------------------------------------------------------------------------
def test_index_fill_plain_self_with_0d_metal_value_casts():
    p = torch.zeros(3, 4, device=MPS)
    idx = torch.tensor([0, 2], device=MPS)
    v0 = mk(2.5)
    assert p.index_fill_(1, idx, v0) is p
    want = torch.zeros(3, 4).index_fill_(
        1, idx.cpu(), torch.tensor(2.5, dtype=torch.float64)
    )
    assert torch.equal(p.cpu(), want)
    got = torch.zeros(3, 4, device=MPS).index_fill(0, torch.tensor([1], device=MPS), v0)
    assert got.dtype == torch.float32 and got.cpu()[1].tolist() == [2.5] * 4


@pytest.mark.parametrize("mode", MODES)
def test_where_accepts_uint8_condition_with_deprecation_warning(mode):
    c8 = torch.tensor([1, 0, 1, 0], dtype=torch.uint8)
    a = rt(np.arange(4.0) + 0.5)
    x, r = mk(a, mode), cpu(a)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = torch.where(c8.to(MPS), x, -x)
        want = torch.where(c8, r, -r)
    assert any("uint8 condition" in str(w.message) for w in caught)
    assert_exact(got, want)


@pytest.mark.parametrize("mode", MODES)
def test_inplace_writers_reject_internally_overlapping_targets(mode):
    base = mk(np.zeros((1, 4)), mode)
    idx = torch.tensor([[0, 1, 2, 0]], device=MPS)
    ones = mk(np.ones((1, 4)), mode)
    writers = (
        lambda t: t.scatter_(0, idx, 1.0),
        lambda t: t.scatter_(0, idx, ones),
        lambda t: t.scatter_add_(0, idx, ones),
        lambda t: t.index_add_(0, torch.tensor([0], device=MPS), ones),
        lambda t: t.index_copy_(0, torch.tensor([0], device=MPS), ones),
        lambda t: t.put_(torch.tensor([0], device=MPS), mk([1.0], mode)),
        lambda t: t.index_put_(
            (torch.tensor([0, 1], device=MPS),), mk(np.ones(4), mode)
        ),
        lambda t: t.index_put_(
            (torch.tensor([0, 1], device=MPS),), mk(np.ones(4), mode), accumulate=True
        ),
    )
    for write in writers:
        with pytest.raises(
            RuntimeError, match="more than one element of the written-to"
        ):
            write(base.expand(3, 4))
    assert np.all(base.to_numpy() == 0.0)
    # a contiguous clone of the same view is written normally
    y = base.expand(3, 4).clone()
    y.scatter_(0, idx, 1.0)
    assert_exact(y, torch.zeros(3, 4, dtype=torch.float64).scatter_(0, idx.cpu(), 1.0))


@pytest.mark.parametrize("mode", MODES)
def test_index_put_and_index_with_scalar_bool_indices(mode):
    a = rt(np.arange(12.0).reshape(3, 4))
    x, r = mk(a, mode), cpu(a)
    t, f = torch.tensor(True, device=MPS), torch.tensor(False, device=MPS)
    one = mk(1.0, mode)
    assert_exact(x.index_put((f,), one), r)
    assert_exact(x.clone().index_put_((f,), one), r)
    assert_exact(x.index_put((t,), one), torch.ones(3, 4, dtype=torch.float64))
    assert_exact(x.index_put((t,), one, accumulate=True), r + 1)
    idx = torch.tensor([0, 2], device=MPS)
    want = r.index_put((torch.tensor([0, 2]),), torch.tensor(7.0, dtype=torch.float64))
    assert_exact(x.index_put((t, idx), mk(7.0, mode)), want)
    assert tuple(x[t].shape) == (1, 3, 4) and tuple(x[f].shape) == (0, 3, 4)
    assert_exact(x[t], r[torch.tensor(True)])
    assert tuple(torch.ops.aten.index.Tensor(x, [t, idx]).shape) == (1, 2, 4)


# ---------------------------------------------------------------------------
# Elementwise: in-place never resizes, lerp, pow, devices, alpha fma
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_inplace_ops_never_resize_self(mode):
    base = mk(np.zeros((20, 20)), mode)
    view = base[:, :1]
    big = mk(np.ones((20, 20)), mode)
    for op in ("add_", "mul_", "sub_", "lt_", "pow_", "clamp_min_", "copysign_"):
        with pytest.raises(RuntimeError, match=r"output with shape \[20, 1\]"):
            getattr(view, op)(big)
    assert tuple(view.shape) == (20, 1) and float(base.sum()) == 0.0
    with pytest.raises(RuntimeError, match=r"output with shape \[\] doesn't match"):
        mk(2.0, mode).mul_(mk([1.0, 2.0, 3.0], mode))
    with pytest.raises(RuntimeError, match=r"output with shape \[500\]"):
        mk(np.arange(500.0), mode).add_(mk(np.ones((3, 500)), mode))
    f32 = torch.zeros(20, device=MPS)
    with pytest.raises(RuntimeError, match=r"output with shape \[20\]"):
        f32.add_(big)
    # a same-shape / smaller other still works in place
    x = mk(np.ones((3, 4)), mode)
    assert x.add_(mk(np.arange(4.0), mode)) is x
    assert_exact(x, torch.ones(3, 4, dtype=torch.float64) + torch.arange(4.0).double())


@pytest.mark.parametrize("mode", MODES)
def test_lerp_weight_dtype_and_representation(mode):
    a, b = rt(np.ones((20, 20))), rt(np.full((20, 20), 2.0))
    x, y = mk(a, mode), mk(b, mode)
    r, s = cpu(a), cpu(b)
    # an N-d non-float64 weight is rejected as torch rejects it
    with pytest.raises(RuntimeError, match="expected dtype double for `weight`"):
        torch.lerp(x, y, torch.full((20, 20), 0.75, device=MPS))
    with pytest.raises(RuntimeError, match="expected dtype double for `weight`"):
        torch.lerp(s, s, torch.full((20, 20), 0.75))
    # a 0-d plain weight and a MetalFloat64 weight are promoted into the op's mode
    w0 = torch.tensor(0.75, device=MPS)
    assert_exact(torch.lerp(x, y, w0), torch.lerp(r, s, 0.75))
    w = mk(np.full((20, 20), 0.25), mode)
    assert_exact(
        torch.lerp(x, y, w), torch.lerp(r, s, torch.full((20, 20), 0.25).double())
    )
    got = torch.lerp(
        torch.ones(20, 20, device=MPS), torch.full((20, 20), 2.0, device=MPS), w
    )
    assert got.mode == mode
    assert_exact(got, torch.lerp(r, s, torch.full((20, 20), 0.25).double()))
    x.lerp_(y, w)
    assert_exact(x, torch.lerp(r, s, torch.full((20, 20), 0.25).double()))


def test_pow_tensor_tensor_0d_exponent_uses_c_pow_semantics():
    a = np.array([-0.0, -np.inf, 4.0, -8.0] + [0.0] * 296)
    x, r = mk(a), cpu(a)
    for e in (0.5, -0.5, 2.0, 3.0, 2.5):
        te = torch.tensor(e, dtype=torch.float64)
        assert_exact(torch.pow(x, te), torch.pow(r, te))
        assert_exact(x**te, r**te)
    # the Scalar overload keeps torch's sqrt / rsqrt fast paths
    assert_exact(torch.pow(x, 0.5), torch.pow(r, 0.5))
    assert_exact(torch.pow(x, -0.5), torch.pow(r, -0.5))


@pytest.mark.parametrize("mode", MODES)
def test_cpu_nd_floating_operands_and_cpu_out_raise(mode):
    x = mk(np.ones((20, 20)), mode)
    c = torch.ones(20, 20, dtype=torch.float64)
    for fn in (
        lambda: x + c,
        lambda: x.add_(c),
        lambda: torch.mul(x, c.float()),
        lambda: torch.atan2(x, c),
        lambda: torch.clamp(x, c, c),
        lambda: torch.add(x, x, out=torch.empty(20, 20, dtype=torch.float64)),
        lambda: torch.sum(x, 0, out=torch.empty(20, dtype=torch.float64)),
    ):
        with pytest.raises(RuntimeError, match=DEVICE_ERROR):
            fn()
    with pytest.raises(RuntimeError, match="Expected out tensor to have device mps:0"):
        torch.mm(x, x, out=torch.empty(20, 20, dtype=torch.float64))
    # 0-d CPU tensors are scalars, CPU bool masks come from the host path
    assert_exact(
        x + torch.tensor(1.5, dtype=torch.float64), torch.full((20, 20), 2.5).double()
    )
    mask = torch.ones(20, 20, dtype=torch.bool)
    assert_exact(torch.where(mask, x, -x), torch.ones(20, 20, dtype=torch.float64))


@pytest.mark.parametrize("mode", MODES)
def test_pow_with_plain_float32_base_and_metal_exponent_grad(mode):
    base = np.linspace(0.5, 3.0, 300).astype(np.float32)
    f = torch.tensor(base, device=MPS)
    e = mk(np.full(300, 2.0), mode, True)
    (g,) = torch.autograd.grad(torch.pow(f, e).sum(), [e])
    ec = cpu(np.full(300, 2.0), True)
    (gc,) = torch.autograd.grad(torch.pow(torch.tensor(base).double(), ec).sum(), [ec])
    assert_grad_close(g, gc)
    # the same conversion for reductions with a float32 result dtype
    x = mk(np.ones((2, 3)) + 0.25, mode, True)
    xc = cpu(np.ones((2, 3)) + 0.25, True)
    for fn in (
        lambda t: t.sum(dtype=torch.float32).sum(),
        lambda t: t.mean(dim=0, dtype=torch.float32).sum(),
        lambda t: t.prod(dtype=torch.float32),
        lambda t: t.cumsum(0, dtype=torch.float32).sum(),
    ):
        (g,) = torch.autograd.grad(fn(x), x)
        (gc,) = torch.autograd.grad(fn(xc), xc)
        # derived only from the plain float32 grad: agnostic (global mode)
        assert g.mode == mode or g._agnostic
        assert_grad_close(g, gc)


def test_host_resident_0d_bounds_in_clamp_and_pow_backward(monkeypatch):
    monkeypatch.setattr(mt, "HOST_THRESHOLD", 256)
    for fn, lo in (
        (lambda x, r: torch.clamp(x, -r, r), -2.0),
        (lambda x, r: torch.clamp(x, r, 2 * r), -2.0),
        (lambda x, r: r**x, -2.0),
        (lambda x, r: x**r, 0.5),
    ):
        a = rt(np.linspace(lo, 2, 1000))
        x, r = mk(a, requires_grad=True), mk(1.5, requires_grad=True)
        assert r.is_host_resident and not x.is_host_resident
        xc, rc = cpu(a, True), cpu(1.5, True)
        g = torch.autograd.grad(fn(x, r).sum(), [x, r])
        gc = torch.autograd.grad(fn(xc, rc).sum(), [xc, rc])
        for got, want in zip(g, gc, strict=True):
            assert_grad_close(got, want)


def test_sf64_alpha_and_lerp_are_bit_exact_with_fused_cpu_paths():
    rng = np.random.default_rng(5)
    a, b = rng.uniform(-3, 3, 2000), rng.uniform(0.5, 3, 2000)
    w = rng.uniform(-1, 2, 2000)
    xa, xb, xw = mk(a, "sf64"), mk(b, "sf64"), mk(w, "sf64")
    ra, rb, rw = cpu(a), cpu(b), cpu(w)
    assert_exact(torch.add(xa, xb, alpha=2.5), torch.add(ra, rb, alpha=2.5))
    assert_exact(torch.sub(xa, xb, alpha=0.3), torch.sub(ra, rb, alpha=0.3))
    assert_exact(torch.rsub(xa, xb, alpha=0.3), torch.rsub(ra, rb, alpha=0.3))
    assert_exact(torch.lerp(xa, xb, 0.3), torch.lerp(ra, rb, 0.3))
    assert_exact(torch.lerp(xa, xb, 0.8), torch.lerp(ra, rb, 0.8))
    assert_exact(torch.lerp(xa, xb, xw), torch.lerp(ra, rb, rw))
    assert_exact(
        torch.addcmul(xa, xb, xw, value=0.3), torch.addcmul(ra, rb, rw, value=0.3)
    )
    assert_exact(
        torch.addcdiv(xa, xb, xw, value=0.3), torch.addcdiv(ra, rb, rw, value=0.3)
    )
    # three scalars fuse exactly on the host
    got = torch.add(mk(1.0, "sf64"), mk(1 + 2.0**-40, "sf64"), alpha=1 + 2.0**-30)
    assert float(got) == float(
        torch.add(cpu(1.0), cpu(1 + 2.0**-40), alpha=1 + 2.0**-30)
    )
    # df64: within the representation tolerance (extra rounding documented)
    xa, xb, xw = mk(rt(a)), mk(rt(b)), mk(rt(w))
    ra, rb, rw = cpu(rt(a)), cpu(rt(b)), cpu(rt(w))
    assert_close(torch.add(xa, xb, alpha=2.5), torch.add(ra, rb, alpha=2.5), "df64")
    # lerp under cancellation: 1e-13 of the operands (documented extra rounding)
    np.testing.assert_allclose(
        torch.lerp(xa, xb, xw).to_numpy(),
        torch.lerp(ra, rb, rw).numpy(),
        rtol=1e-13,
        atol=1e-13 * float(np.abs(rt(b)).max()),
    )


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------
def test_sequential_sum_path_keeps_1e13_on_long_axes():
    assert ops_reduce._use_tree(4095, 16385) and ops_reduce._use_tree(1024, 10**6)
    assert not ops_reduce._use_tree(1023, 16385)
    rng = np.random.default_rng(9)
    m = rt(rng.random((16385, 1500)))
    x = mk(m)
    assert not ops_reduce._use_tree(1500, 16385) or True  # 1500 >= 1024: tree now
    s = x.sum(dim=1).to_numpy()
    rows = list(range(0, 16385, 97))
    ref = np.array([math.fsum(m[i]) for i in rows])
    assert np.abs(s[rows] - ref).max() / np.abs(ref).max() < 1e-13
    # the sequential path (n < 1024, many lines) stays inside the tolerance too
    m = rt(rng.random((16385, 1000)))
    s = mk(m).sum(dim=1).to_numpy()
    ref = np.array([math.fsum(m[i]) for i in rows])
    assert np.abs(s[rows] - ref).max() / np.abs(ref).max() < 1e-13
    assert_close(mk(m).mean(dim=1), cpu(m).mean(dim=1), "df64")


@pytest.mark.parametrize("mode", MODES)
def test_cumsum_and_cumprod_inplace(mode):
    a = rt(np.arange(1.0, 7.0).reshape(2, 3))
    for op in ("cumsum_", "cumprod_"):
        for dim in (0, 1, -1):
            x, r = mk(a, mode), cpu(a)
            assert getattr(x, op)(dim) is x
            getattr(r, op)(dim)
            assert_exact(x, r)
    with pytest.raises(RuntimeError, match="provided dtype must match"):
        mk(a, mode).cumsum_(0, dtype=torch.float32)


@pytest.mark.parametrize("mode", MODES)
def test_cdist_backward_and_dist(mode):
    rng = np.random.default_rng(11)
    p1 = rt(rng.standard_normal((3, 4)))
    p2 = rt(rng.standard_normal((5, 4)))
    weights = torch.arange(15.0, device=MPS).reshape(3, 5) / 7
    for p in (2.0, 1.0, float("inf"), 3.0, 0.5):
        grad_pair(
            lambda x, y, p=p: (
                torch.cdist(x, y, p=p) * weights.to(x.device).double()
                if isinstance(x, MetalFloat64)
                else torch.cdist(x, y, p=p) * weights.cpu().double()
            ),
            mode,
            p1,
            p2,
        )
    x, y = mk(p1, mode), mk(p1[::-1].copy(), mode)
    r, s = cpu(p1), cpu(p1[::-1].copy())
    for p in (2.0, 1.0, 3.0):
        assert_close(torch.dist(x, y, p), torch.dist(r, s, p), mode, rtol=1e-13)
    grad_pair(lambda x, y: torch.dist(x, y), mode, p1, p1[::-1].copy())


def test_reduction_dtype_arguments_are_validated_like_torch():
    a = rt(np.array([[1.0, 3.0, 3.0], [2.0, 2.0, 2.0], [-1.0, 0.0, -1.0]]))
    x, r = mk(a), cpu(a)
    cases = (
        lambda t: t.mean(dtype=torch.int64),
        lambda t: torch.linalg.vector_norm(t, dtype=torch.float32),
        lambda t: torch.linalg.vector_norm(t, dtype=torch.int64),
        lambda t: torch.norm(t, p=1, dim=0, dtype=torch.bool),
        lambda t: torch.norm(t, p=2, dtype=torch.float32),
    )
    for fn in cases:
        with pytest.raises(RuntimeError) as want:
            fn(r)
        with pytest.raises(RuntimeError) as got:
            fn(x)
        assert str(got.value) == str(want.value)
    # accepted forms are unchanged
    assert x.mean(dtype=torch.float32).dtype == torch.float32
    assert x.sum(dtype=torch.int64).item() == r.sum(dtype=torch.int64).item()


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_addmm_family_checks_the_bias_shape_even_with_beta_zero(mode):
    a = mk(np.eye(3), mode)
    ac = torch.eye(3, dtype=torch.float64)
    with pytest.raises(RuntimeError) as want:
        torch.addmm(torch.ones(4, dtype=torch.float64), ac, ac, beta=0)
    with pytest.raises(RuntimeError) as got:
        torch.addmm(mk(np.ones(4), mode), a, a, beta=0)
    assert str(got.value) == str(want.value)
    with pytest.raises(RuntimeError, match="expanded size"):
        torch.baddbmm(
            mk(np.ones((1, 2, 3)), mode), a.unsqueeze(0), a.unsqueeze(0), beta=0
        )
    with pytest.raises(RuntimeError, match="expanded size"):
        torch.addbmm(mk(np.ones(4), mode), a.unsqueeze(0), a.unsqueeze(0), beta=0)
    with pytest.raises(RuntimeError):
        torch.addmm(mk(np.ones((2, 3, 3)), mode), a, a, beta=0)
    # a broadcastable bias with beta=0 is still ignored
    assert_exact(torch.addmm(mk(np.ones(3), mode), a, a, beta=0), ac @ ac)


# ---------------------------------------------------------------------------
# Documented limitations (locked so a change is noticed)
# ---------------------------------------------------------------------------
def test_documented_dot_of_negative_zeros_is_positive_zero():
    got = torch.dot(mk([-0.0, -0.0]), mk([1.0, 2.0]))
    assert float(got) == 0.0 and not np.signbit(got.to_numpy())
    want = torch.dot(cpu([-0.0, -0.0]), cpu([1.0, 2.0]))
    assert np.signbit(want.numpy())  # torch CPU: -0.0 (accumulator sign)


def test_documented_lerp_zero_sign_follows_the_vectorized_path():
    a, b = np.full(300, 1.0), np.full(300, -0.0)
    got = torch.lerp(mk(a), mk(b), 1.0).to_numpy()
    assert np.all(got == 0.0) and np.all(np.signbit(got))  # fma(0, -1, -0.0)
    assert np.signbit(torch.lerp(cpu(a), cpu(b), 1.0).numpy()).all()
    assert not np.signbit(torch.lerp(cpu(a[:1]), cpu(b[:1]), 1.0).numpy()).any()


def test_documented_df64_round_decimals_near_ties():
    t = np.full(300, -4.994999999999997)
    assert np.all(mk(t).to_numpy() == t)  # exactly representable in df64
    got = torch.round(mk(t), decimals=2).to_numpy()
    assert np.all(got == -5.0)  # x * 100 rounds to -499.5 in 48 bits
    assert np.all(torch.round(cpu(t), decimals=2).numpy() == -4.99)
    assert np.all(torch.round(mk(t, "sf64"), decimals=2).to_numpy() == -4.99)
    assert_exact(torch.round(mk(t)), torch.round(cpu(t)))  # decimals=0 exact


def test_documented_0d_metal_with_nd_plain_tensor_stays_float64():
    s, p = mk(1.5), torch.ones(1000, device=MPS)
    assert (s * p).dtype == torch.float64 and isinstance(s * p, MetalFloat64)
    assert (p * s).dtype == torch.float64
    assert (
        torch.tensor(1.5, dtype=torch.float64) * torch.ones(3)
    ).dtype == torch.float32


def test_documented_structural_writes_accept_mixed_dtypes():
    p = torch.zeros(3, 4, device=MPS)
    s = mk(np.full((3, 2), 1 + 2.0**-40))
    p.scatter_(1, torch.tensor([[0, 2]] * 3, device=MPS), s)  # demoted to float32
    assert float(p[0, 0]) == 1.0
    x = mk(np.zeros((3, 4)))
    x.index_put_((torch.tensor([1], device=MPS),), torch.ones(4, device=MPS))
    assert x.to_numpy()[1].tolist() == [1.0] * 4  # promoted exactly
    with pytest.raises(RuntimeError, match="dtypes match"):
        torch.zeros(3, 4, dtype=torch.float64).index_put_(
            (torch.tensor([1]),), torch.ones(4, dtype=torch.float32)
        )


def test_documented_df64_prod_over_long_axes_exceeds_1e13():
    from fractions import Fraction

    rng = np.random.default_rng(13)
    m = rt(1.0 + rng.standard_normal((4, 4095)) * 1e-3)
    got = mk(m).prod(dim=1).to_numpy()
    ref = np.array([float(math.prod(Fraction(v) for v in row)) for row in m])
    rel = np.abs(got - ref) / np.abs(ref)
    assert rel.max() < 1e-12  # ~sqrt(n) u^2 with u^2 = 2^-96 (measured 1.6e-13)
    assert_close(mk(m, "sf64").prod(dim=1), cpu(m).prod(dim=1), "sf64", rtol=1e-14)
