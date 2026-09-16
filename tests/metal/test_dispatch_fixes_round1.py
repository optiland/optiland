"""Regression tests for the M2 dispatch-layer fix round 1.

One test per verified failure (see ``NOTES/05-m2-status.md`` section 7). The
oracle is torch CPU float64 on the *decoded* inputs; structural results and
error behaviour are exact, arithmetic chains use the df64 / sf64 tolerances of
the other dispatch test modules, autograd uses rtol 1e-12.
"""

from __future__ import annotations

import os

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

MODES = ("df64", "sf64")
RTOL = {"df64": 1e-13, "sf64": 1e-15}
GRAD_RTOL = 1e-12
MPS = torch.device("mps")


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


def grad_pair(fn, mode, *arrays):
    """Gradients of ``fn(*tensors).sum()`` on the GPU and on the CPU."""
    xs = [mk(a, mode, True) for a in arrays]
    cs = [cpu(a, True) for a in arrays]
    g = torch.autograd.grad(fn(*xs).sum(), xs)
    gc = torch.autograd.grad(fn(*cs).sum(), cs)
    for a, b in zip(g, gc, strict=True):
        assert isinstance(a, MetalFloat64)
        scale = float(np.abs(b.numpy()).max()) or 1.0
        np.testing.assert_allclose(
            a.to_numpy(), b.numpy(), rtol=GRAD_RTOL, atol=GRAD_RTOL * scale
        )


# ---------------------------------------------------------------------------
# where / factory constants in the non-global representation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_where_method_scalar_uses_self_mode(mode, other_global_mode):
    other_global_mode(mode)
    a = rt(np.arange(6.0).reshape(2, 3) - 2.5)
    x, r = mk(a, mode), cpu(a)
    want = r.where(r > 0, 0.0)
    for got in (
        x.where(x > 0, 0.0),
        x.where(x > 0, other=0.0),
        torch.where(x > 0, x, 0.0),
    ):
        assert got.mode == mode
        assert_exact(got, want)
    assert_exact(x.where(x > 0, 2.5), r.where(r > 0, 2.5))


@pytest.mark.parametrize("mode", MODES)
def test_backward_of_factory_formulas_in_non_global_mode(mode, other_global_mode):
    """where(scalar), index_put, put, *_scatter, split: zeros from the intercept."""
    other_global_mode(mode)
    a = rt(np.random.default_rng(3).standard_normal((3, 4)))
    mask = torch.tensor(
        [[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.bool, device=MPS
    )
    cmask = mask.cpu()
    i0 = torch.tensor([0, 2], device=MPS)
    ones4 = rt(np.arange(1.0, 5.0))
    ones3 = rt(np.arange(1.0, 4.0))
    cases = {
        "where scalar": (
            lambda x: torch.where(mask, x, 0.5),
            lambda x: torch.where(cmask, x, 0.5),
        ),
        "where 2x": (
            lambda x: torch.where(mask, x, 2.0 * x),
            lambda x: torch.where(cmask, x, 2.0 * x),
        ),
        "index_put": (
            lambda x, v: x.index_put((i0,), v) * 3.0,
            lambda x, v: x.index_put((i0.cpu(),), v) * 3.0,
        ),
        "put": (
            lambda x, v: x.put(torch.tensor([0, 5], device=MPS), v[:2]) ** 2,
            lambda x, v: x.put(torch.tensor([0, 5]), v[:2]) ** 2,
        ),
        "select_scatter": (
            lambda x, v: torch.select_scatter(x, v, 0, 1) ** 2,
            lambda x, v: torch.select_scatter(x, v, 0, 1) ** 2,
        ),
        "slice_scatter": (
            lambda x, v: torch.slice_scatter(x, v.unsqueeze(0), 0, 1, 2) ** 2,
            lambda x, v: torch.slice_scatter(x, v.unsqueeze(0), 0, 1, 2) ** 2,
        ),
        "split partial": (
            lambda x: x.split(2, 1)[0] ** 2,
            lambda x: x.split(2, 1)[0] ** 2,
        ),
    }
    for name, (fg, fc) in cases.items():
        xg, xc = mk(a, mode, True), cpu(a, True)
        if name in ("index_put", "put", "select_scatter", "slice_scatter"):
            vg, vc = mk(ones4, mode, True), cpu(ones4, True)
            g = torch.autograd.grad(fg(xg, vg).sum(), (xg, vg))
            gc = torch.autograd.grad(fc(xc, vc).sum(), (xc, vc))
        else:
            g = torch.autograd.grad(fg(xg).sum(), (xg,))
            gc = torch.autograd.grad(fc(xc).sum(), (xc,))
        for got, want in zip(g, gc, strict=True):
            assert got.mode == mode, name
            assert_exact(got, want)
    xg, xc = mk(a, mode, True), cpu(a, True)
    vg, vc = mk(ones3, mode, True), cpu(ones3, True)
    g = torch.autograd.grad((torch.diagonal_scatter(xg, vg) ** 2).sum(), (xg, vg))
    gc = torch.autograd.grad((torch.diagonal_scatter(xc, vc) ** 2).sum(), (xc, vc))
    for got, want in zip(g, gc, strict=True):
        assert_exact(got, want)


@pytest.mark.parametrize("mode", MODES)
def test_factory_constant_stops_being_agnostic_once_mutated(mode, other_global_mode):
    other_global_mode(mode)
    z = torch.zeros(4, dtype=torch.float64, device=MPS)  # global (other) mode
    assert z.mode != mode and z._agnostic
    z += 1.0  # a scalar mutation keeps it a constant (fix round 2: quantile)
    assert z._agnostic
    z += mk(np.ones(4), z.mode)  # absorbed data of a fixed representation
    assert not z._agnostic
    with pytest.raises(TypeError, match="cannot mix representations"):
        torch.add(z, mk(np.ones(4), mode))  # (`+` maps TypeError to NotImplemented)
    z2 = torch.zeros(4, dtype=torch.float64, device=MPS)
    assert_exact(z2 + mk(np.ones(4), mode), torch.ones(4, dtype=torch.float64))
    assert z2.mode == mode


def test_sgn_backward_zerotensor_is_intercepted():
    x = mk([0.5, -1.0, 2.0], requires_grad=True)
    (g,) = torch.autograd.grad(torch.sgn(x), [x], grad_outputs=mk(np.ones(3)))
    assert_exact(g, torch.zeros(3, dtype=torch.float64))
    x = mk([0.5, -1.0, 2.0], requires_grad=True)
    (g,) = torch.autograd.grad(torch.sign(x), [x], grad_outputs=mk(np.ones(3)))
    assert_exact(g, torch.zeros(3, dtype=torch.float64))


# ---------------------------------------------------------------------------
# masked_scatter backward
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_masked_scatter_backward(mode):
    y0 = rt(np.arange(12.0).reshape(3, 4) - 5)
    s0 = rt(np.arange(12.0) + 0.5)

    def fn(y, s):
        return (y.masked_scatter(y > 0, s) * 2.0).sum()

    yg, sg = mk(y0, mode, True), mk(s0, mode, True)
    yc, sc = cpu(y0, True), cpu(s0, True)
    fn(yg, sg).backward()
    fn(yc, sc).backward()
    assert_exact(yg.grad, yc.grad)
    assert_exact(sg.grad, sc.grad)
    # a broadcast mask and a short source
    m = torch.tensor([True, False, True, False], device=MPS)
    yg, sg = mk(y0, mode, True), mk(s0[:6], mode, True)
    yc, sc = cpu(y0, True), cpu(s0[:6], True)
    (yg.masked_scatter(m, sg) ** 2).sum().backward()
    (yc.masked_scatter(m.cpu(), sc) ** 2).sum().backward()
    assert_exact(yg.grad, yc.grad)
    assert_exact(sg.grad, sc.grad)


# ---------------------------------------------------------------------------
# MetalFloat64 in index / mask / sorter / condition slots
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_metal_tensor_rejected_in_index_positions(mode):
    x = mk(np.arange(6.0), mode)
    idx = mk([5e-324, 0.0, 1e-323], mode)  # sf64 bit patterns 1, 0, 2
    cases = [
        (IndexError, "must be long", lambda: x[idx]),
        (IndexError, "must be long", lambda: x.index_put((idx,), mk([1.0], mode))),
        (RuntimeError, "index_select", lambda: x.index_select(0, idx)),
        (RuntimeError, "gather", lambda: x.gather(0, idx)),
        (RuntimeError, "masked_select", lambda: x.masked_select(x)),
        (RuntimeError, "boolean tensor", lambda: torch.where(x, x, x)),
        (RuntimeError, "boolean masks", lambda: x.masked_fill(x, 1.0)),
        (RuntimeError, "boolean masks", lambda: x.masked_fill(x, mk(1.0, mode))),
        (RuntimeError, "boolean masks", lambda: x.masked_scatter(x, x)),
        (RuntimeError, "scatter", lambda: x.clone().scatter_(0, idx, 1.0)),
        (RuntimeError, "scatter", lambda: x.clone().scatter_(0, idx, x[:3])),
        (RuntimeError, "scatter", lambda: x.scatter_add(0, idx, x[:3])),
        (RuntimeError, "index_add", lambda: x.index_add(0, idx, x[:3])),
        (IndexError, "index_fill", lambda: x.clone().index_fill_(0, idx, -1.0)),
        (RuntimeError, "index_copy", lambda: x.index_copy(0, idx, x[:3])),
        (RuntimeError, "put_", lambda: x.clone().put_(idx, x[:3])),
        (RuntimeError, "sorter", lambda: torch.searchsorted(x, x, sorter=idx)),
        (RuntimeError, "take", lambda: x.take(idx)),
    ]
    for exc, msg, fn in cases:
        with pytest.raises(exc, match=msg):
            fn()

    def setitem():
        z = x.clone()
        z[idx] = -1.0

    with pytest.raises(IndexError, match="must be long"):
        setitem()
    assert_exact(x, cpu(np.arange(6.0)))  # nothing was written


# ---------------------------------------------------------------------------
# index validation of the writing ops
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_scatter_family_validates_index_range_and_shape(mode):
    def z():
        return mk(np.zeros((3, 4)), mode)

    def ix(a):
        return torch.tensor(a, device=MPS)

    cases = [
        (
            RuntimeError,
            "index -1 is out of bounds",
            lambda: z().scatter_(0, ix([[-1, 0, 1, 2]]), 5.0),
        ),
        (
            RuntimeError,
            "index 3 is out of bounds",
            lambda: z().scatter(0, ix([[3, 0, 1, 2]]), 5.0),
        ),
        (
            RuntimeError,
            "index 4 is out of bounds for dimension 1",
            lambda: z().scatter(1, ix([[4], [0], [0]]), 5.0),
        ),
        (
            RuntimeError,
            "same number of dimensions",
            lambda: z().scatter_(0, ix([1, 0]), 5.0),
        ),
        (
            RuntimeError,
            "no larger than self",
            lambda: z().scatter(0, ix([[1, 0, 1, 2, 2]]), 5.0),
        ),
        (
            RuntimeError,
            "no larger than src",
            lambda: z().scatter(0, ix([[1, 0]]), mk(np.ones((1, 1)), mode)),
        ),
        (
            RuntimeError,
            "index 4 is out of bounds",
            lambda: z().scatter_add(1, ix([[4], [4], [0]]), mk(np.ones((3, 1)), mode)),
        ),
        (
            RuntimeError,
            "index -1 is out of bounds",
            lambda: z().scatter_add(1, ix([[-1], [0], [0]]), mk(np.ones((3, 1)), mode)),
        ),
        (
            RuntimeError,
            "same number of dimensions",
            lambda: z().scatter_add_(0, ix([1, 0]), mk(np.ones(2), mode)),
        ),
        (
            IndexError,
            "same slice shapes",
            lambda: z().index_add_(0, ix([1, 0]), mk(np.ones((2, 5)), mode)),
        ),
        (
            IndexError,
            "index out of range",
            lambda: z().index_add_(0, ix([-1, 0]), mk(np.ones((2, 4)), mode)),
        ),
        (
            IndexError,
            "index out of range",
            lambda: z().index_add(0, ix([3]), mk(np.ones((1, 4)), mode)),
        ),
        (
            IndexError,
            "supposed to be a vector",
            lambda: z().index_add(0, ix([[1, 0]]), mk(np.ones((2, 4)), mode)),
        ),
        (
            IndexError,
            "Number of indices",
            lambda: z().index_add(0, ix([1, 0, 2]), mk(np.ones((2, 4)), mode)),
        ),
        (
            IndexError,
            "dimensionality must match",
            lambda: z().index_add(0, ix([1, 0]), mk(np.ones((2, 4, 1)), mode)),
        ),
        (
            IndexError,
            "same slice shapes",
            lambda: z().index_copy_(0, ix([1, 0]), mk(np.ones((2, 5)), mode)),
        ),
        (
            IndexError,
            "index out of range",
            lambda: z().index_copy(0, ix([-1, 0]), mk(np.ones((2, 4)), mode)),
        ),
        (
            IndexError,
            "index out of range",
            lambda: z().index_copy(0, ix([3, 0]), mk(np.ones((2, 4)), mode)),
        ),
        (
            IndexError,
            "Number of indices",
            lambda: z().index_copy(0, ix([1, 0, 2]), mk(np.ones((2, 4)), mode)),
        ),
        (
            IndexError,
            "index 3 is out of bounds",
            lambda: z().index_fill(0, ix([3]), 1.0),
        ),
        (
            IndexError,
            "index -4 is out of bounds",
            lambda: z().index_fill_(0, ix([-4]), 1.0),
        ),
        (RuntimeError, "vector/scalar", lambda: z().index_fill(0, ix([[1]]), 1.0)),
        (
            IndexError,
            "tried to access index -13",
            lambda: mk(np.zeros(12), mode).put_(ix([-13]), mk([1.0], mode)),
        ),
        (
            IndexError,
            "tried to access index 12",
            lambda: mk(np.zeros(12), mode).put(ix([12]), mk([1.0], mode)),
        ),
        (
            IndexError,
            "same number of elements",
            lambda: mk(np.zeros(12), mode).put(ix([1, 2]), mk([1.0], mode)),
        ),
    ]
    for exc, msg, fn in cases:
        with pytest.raises(exc, match=msg):
            fn()
    # negative indices that torch accepts still work: index_fill wraps, put wraps
    r = cpu(np.zeros((3, 4)))
    assert_exact(
        z().index_fill(0, ix([-1]), 1.0), r.index_fill(0, torch.tensor([-1]), 1.0)
    )
    assert_exact(
        mk(np.zeros(12), mode).put(ix([-12]), mk([1.0], mode)),
        cpu(np.zeros(12)).put(torch.tensor([-12]), cpu([1.0])),
    )
    # a 0-d self with a one-element index
    got = mk(0.0, mode).scatter(0, torch.tensor(0, device=MPS), 5.0)
    assert_exact(got, cpu(0.0).scatter(0, torch.tensor(0), 5.0))


@pytest.mark.parametrize("mode", MODES)
def test_index_copy_duplicates_keep_pairs_together(mode):
    g = np.random.default_rng(0)
    src = rt(g.standard_normal((4096, 64)))
    idx = g.integers(0, 256, 4096)
    x = mk(np.zeros((256, 64)), mode)
    r = x.index_copy_(0, torch.from_numpy(idx).to(MPS), mk(src, mode))
    assert r is x
    out = x.to_numpy()
    # every element equals a candidate source value (torch: the winner among
    # duplicates is unspecified, but never a split (hi, lo) pair); with the
    # nucleus's last-write-wins rule the whole row is the last candidate
    bad = sum(int(np.sum(~np.isin(out[r], src[idx == r]))) for r in range(256))
    assert bad == 0
    last = {int(r): int(np.flatnonzero(idx == r)[-1]) for r in np.unique(idx)}
    for r, p in last.items():
        np.testing.assert_array_equal(out[r], src[p])
    # functional variant leaves self untouched
    y = mk(np.zeros((3, 4)), mode)
    got = y.index_copy(
        0, torch.tensor([2, 0], device=MPS), mk(rt(np.ones((2, 4)) * 7), mode)
    )
    assert_exact(
        got,
        cpu(np.zeros((3, 4))).index_copy(
            0, torch.tensor([2, 0]), cpu(np.ones((2, 4)) * 7)
        ),
    )
    assert_exact(y, cpu(np.zeros((3, 4))))


# ---------------------------------------------------------------------------
# out= overloads and in-place resize
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_out_overloads_of_ordering_and_indexing_ops(mode):
    a = rt(np.array([3.0, 1.0, 2.0, -4.0]))
    x, r = mk(a, mode), cpu(a)
    o, i = mk(np.zeros(4), mode), torch.zeros(4, dtype=torch.int64, device=MPS)
    ro, ri = torch.zeros(4, dtype=torch.float64), torch.zeros(4, dtype=torch.int64)
    res = torch.sort(x, out=(o, i))
    torch.sort(r, out=(ro, ri))
    assert res[0] is o and res[1] is i
    assert_exact(o, ro)
    assert_exact(i, ri)
    res = torch.sort(x, stable=True, descending=True, out=(o, i))
    torch.sort(r, stable=True, descending=True, out=(ro, ri))
    assert_exact(o, ro)
    assert_exact(i, ri)
    o2, i2 = mk(np.zeros(2), mode), torch.zeros(2, dtype=torch.int64, device=MPS)
    ro2, ri2 = torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.int64)
    torch.topk(x, 2, out=(o2, i2))
    torch.topk(r, 2, out=(ro2, ri2))
    assert_exact(o2, ro2)
    assert_exact(i2, ri2)
    seq, rseq = x.sort().values, r.sort().values
    torch.searchsorted(seq, x, out=i)
    torch.searchsorted(rseq, r, out=ri)
    assert_exact(i, ri)
    torch.bucketize(x, seq, out=i)
    torch.bucketize(r, rseq, out=ri)
    assert_exact(i, ri)
    sel = torch.tensor([2, 0], device=MPS)
    torch.index_select(x, 0, sel, out=o2)
    torch.index_select(r, 0, sel.cpu(), out=ro2)
    assert_exact(o2, ro2)
    torch.gather(x, 0, sel, out=o2)
    torch.gather(r, 0, sel.cpu(), out=ro2)
    assert_exact(o2, ro2)
    torch.take_along_dim(x, sel, 0, out=o2)
    torch.take_along_dim(r, sel.cpu(), 0, out=ro2)
    assert_exact(o2, ro2)
    torch.masked_select(x, x > 1, out=o2)
    torch.masked_select(r, r > 1, out=ro2)
    assert_exact(o2, ro2)
    torch.take(x, sel, out=o2)
    torch.take(r, sel.cpu(), out=ro2)
    assert_exact(o2, ro2)
    v = rt(np.array([10.0, 20.0]))
    torch.index_add(x, 0, sel, mk(v, mode), out=o)
    torch.index_add(r, 0, sel.cpu(), cpu(v), out=ro)
    assert_exact(o, ro)
    torch.scatter_add(x, 0, sel, mk(v, mode), out=o)
    torch.scatter_add(r, 0, sel.cpu(), cpu(v), out=ro)
    assert_exact(o, ro)
    torch.scatter(x, 0, sel, mk(v, mode), out=o)
    torch.scatter(r, 0, sel.cpu(), cpu(v), out=ro)
    assert_exact(o, ro)
    torch.scatter(x, 0, sel, 7.0, out=o)
    torch.scatter(r, 0, sel.cpu(), 7.0, out=ro)
    assert_exact(o, ro)
    torch.index_copy(x, 0, sel, mk(v, mode), out=o)
    torch.index_copy(r, 0, sel.cpu(), cpu(v), out=ro)
    assert_exact(o, ro)
    # an empty out is resized; a mis-sized non-empty out is resized with a warning
    e, ie = mk(np.zeros(0), mode), torch.zeros(0, dtype=torch.int64, device=MPS)
    torch.sort(x, out=(e, ie))
    assert tuple(e.shape) == (4,) and tuple(ie.shape) == (4,)
    assert_exact(e, ro if False else r.sort().values)
    assert_exact(ie, r.sort().indices)
    with pytest.warns(UserWarning, match="was resized"):
        torch.index_select(x, 0, sel, out=o)
    assert tuple(o.shape) == (2,)
    assert_exact(o, r.index_select(0, sel.cpu()))


@pytest.mark.parametrize("mode", MODES)
def test_resize_in_place_and_out_resizing(mode):
    a = rt(np.arange(1.0, 5.0))
    # elementwise out=: empty, larger and smaller destinations
    out = mk(np.zeros(0), mode)
    res = torch.mul(mk(a, mode), 2.0, out=out)
    assert res is out and tuple(out.shape) == (4,)
    assert_exact(out, cpu(a) * 2.0)
    out = mk(np.zeros((3, 2)), mode)
    with pytest.warns(UserWarning, match="was resized"):
        res = torch.mul(mk(a[:2], mode), 2.0, out=out)
    assert res is out and tuple(out.shape) == (2,)
    assert_exact(out, cpu(a[:2]) * 2.0)
    # an empty view resized over its base's storage, exactly like torch
    base, rbase = mk(np.zeros(6), mode), torch.zeros(6, dtype=torch.float64)
    v, rv = base[:0], rbase[:0]
    torch.mul(mk(a, mode), 2.0, out=v)
    torch.mul(cpu(a), 2.0, out=rv)
    assert tuple(v.shape) == (4,)
    assert_exact(v, rv)
    assert_exact(base, rbase)
    # explicit resize_: grow keeps the flat prefix, shrink keeps a prefix
    # (.numpy() pins a CPU storage, so the reference is cloned before comparing)
    x, r = mk(a, mode), cpu(a).clone()
    assert x.resize_((2, 2)) is x
    r.resize_((2, 2))
    assert_exact(x, r.clone())
    x.resize_((6,))
    r.resize_((6,))
    assert_exact(x[:4], r[:4].clone())
    x.resize_((2,))
    r.resize_((2,))
    assert_exact(x, r.clone())
    assert_exact(x * 2.0, r * 2.0)
    with pytest.raises(RuntimeError, match="require grad"):
        mk(a, mode, True).resize_((2, 2))
    # the plain-tensor path is unchanged: a bool out is resized too
    ob = torch.empty(0, dtype=torch.bool, device=MPS)
    torch.lt(mk(a, mode), 2.5, out=ob)
    assert_exact(ob, cpu(a) < 2.5)


def test_out_into_plain_float32_downcasts_and_ints_raise():
    a = rt(np.array([1.0, 2.0]))
    x = mk(a)
    f = torch.zeros(2, dtype=torch.float32, device=MPS)
    assert f.add_(x) is f
    np.testing.assert_array_equal(f.cpu().numpy(), np.array([1.0, 2.0], np.float32))
    torch.exp(x, out=f)
    np.testing.assert_array_equal(f.cpu().numpy(), np.exp(a).astype(np.float32))
    torch.add(x, x, out=f)
    np.testing.assert_array_equal(f.cpu().numpy(), (2 * a).astype(np.float32))
    h = torch.zeros(2, dtype=torch.float16, device=MPS)
    torch.add(x, x, out=h)
    np.testing.assert_array_equal(h.cpu().numpy(), (2 * a).astype(np.float16))
    for dtype in (torch.int64, torch.int32, torch.bool):
        with pytest.raises(RuntimeError, match="can't be cast to the desired output"):
            torch.add(x, x, out=torch.zeros(2, dtype=dtype, device=MPS))
        with pytest.raises(RuntimeError, match="can't be cast"):
            torch.zeros(2, dtype=dtype, device=MPS).add_(x)


# ---------------------------------------------------------------------------
# searchsorted / bucketize with NaN boundaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_searchsorted_bucketize_nan_boundaries_match_torch_cpu(mode):
    g = np.random.default_rng(11)
    seq = np.array([-1.0, 0.0, 1.0, np.nan])
    vals = np.array([np.nan, 2.0, 0.5, -3.0])
    for right in (False, True):
        got = torch.searchsorted(mk(seq, mode), mk(vals, mode), right=right)
        want = torch.searchsorted(cpu(seq), cpu(vals), right=right)
        assert_exact(got, want)
        got = torch.bucketize(mk(vals, mode), mk(seq, mode), right=right)
        want = torch.bucketize(cpu(vals), cpu(seq), right=right)
        assert_exact(got, want)
    for _ in range(60):
        n, m = int(g.integers(1, 12)), int(g.integers(1, 8))
        s = np.sort(rt(g.standard_normal(n)))
        for _ in range(int(g.integers(0, 3))):
            s[g.integers(0, n)] = np.nan
        v = rt(g.standard_normal(m))
        if g.random() < 0.4:
            v[0] = np.nan
        if g.random() < 0.4:
            v[-1] = np.inf
        for right in (False, True):
            got = torch.searchsorted(mk(s, mode), mk(v, mode), right=right)
            want = torch.searchsorted(cpu(s), cpu(v), right=right)
            assert_exact(got, want)
    # sort / argsort / topk keep NaN last (unchanged)
    x = mk(seq, mode)
    assert_exact(torch.sort(x).indices, torch.sort(cpu(seq)).indices)


# ---------------------------------------------------------------------------
# elementwise semantics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_floor_divide_uses_torch_algorithm(mode):
    g = np.random.default_rng(5)
    b = rt(g.uniform(-3, 3, 2000))
    b[np.abs(b) < 1e-3] = 0.25
    k = g.integers(-50, 50, 2000).astype(np.float64)
    a = rt(k * b)  # quotients on or next to an integer
    if mode == "sf64":
        a, b = k * b, b
    x, y = mk(a, mode), mk(b, mode)
    ca, cb = cpu(a), cpu(b)
    assert_exact(torch.floor_divide(x, y), torch.floor_divide(ca, cb))
    assert_exact(x // y, ca // cb)
    assert_exact(
        torch.div(x, y, rounding_mode="floor"), torch.div(ca, cb, rounding_mode="floor")
    )
    assert_exact(
        x.floor_divide(0.1),
        ca.floor_divide(0.1)
        if mode == "sf64"
        else ca.floor_divide(float(mk(0.1).to_numpy())),
    )
    # specials: inf dividend, opposite-sign infinite divisor, rounding-up quotients
    s = np.array([np.inf, -1.0, 1e6, 1.0, -1.0, 0.0, -0.0, 3.0, 7.5])
    t = np.array([1.0, np.inf, -1e-6, 0.1, 0.1, 2.0, 2.0, -0.0, 0.0])
    s, t = (rt(s), rt(t)) if mode == "df64" else (s, t)
    assert_exact(
        torch.floor_divide(mk(s, mode), mk(t, mode)), torch.floor_divide(cpu(s), cpu(t))
    )
    # in place
    z = mk(a, mode)
    z.floor_divide_(y)
    assert_exact(z, ca.floor_divide(cb))
    z = mk(a, mode)
    z.div_(y, rounding_mode="floor")
    assert_exact(z, ca // cb)
    if mode == "sf64":
        assert_exact(mk([1.0, 3.0], mode) // mk([0.1, 0.1], mode), cpu([9.0, 29.0]))


@pytest.mark.parametrize("mode", MODES)
def test_remainder_zero_sign_follows_torch(mode):
    a = np.array([0.0, -0.0, 3.0, -3.0, 4.0, 0.0, 5.5, -5.5])
    b = np.array([-1.0, 1.0, -3.0, 3.0, -2.0, -1.0, 2.0, 2.0])
    x, y = mk(a, mode), mk(b, mode)
    assert_exact(torch.remainder(x, y), torch.remainder(cpu(a), cpu(b)))
    assert_exact(torch.remainder(x, -2.0), torch.remainder(cpu(a), -2.0))
    assert_exact(torch.remainder(4.0, y), torch.remainder(4.0, cpu(b)))
    z = mk(a, mode)
    z.remainder_(y)
    assert_exact(z, torch.remainder(cpu(a), cpu(b)))
    g = np.random.default_rng(6)
    u, v = rt(g.uniform(-9, 9, 500)), rt(g.uniform(-4, 4, 500))
    assert_close(
        torch.remainder(mk(u, mode), mk(v, mode)), torch.remainder(cpu(u), cpu(v)), mode
    )


@pytest.mark.parametrize("mode", MODES)
def test_sign_of_negative_zero(mode):
    a = np.array([-0.0, 0.0, -2.0, 3.0, np.nan, -np.inf])
    x = mk(a, mode)
    for fn in (torch.sign, torch.sgn):
        assert_exact(fn(x), fn(cpu(a)))
    z = mk(a, mode)
    z.sign_()
    assert_exact(z, torch.sign(cpu(a)))
    v = mk(a, mode)[::2]
    v.sgn_()
    assert_exact(v, torch.sgn(cpu(a)[::2]))


def test_pow_negative_half_is_rsqrt():
    a = np.array([-np.inf, -0.0, 0.0, 4.0, np.inf, -4.0])
    assert_exact(torch.pow(mk(a), -0.5), torch.pow(cpu(a), -0.5))
    assert_exact(mk(a) ** -0.5, cpu(a) ** -0.5)
    assert_exact(torch.pow(mk(a), 0.5), torch.pow(cpu(a), 0.5))
    x = rt(np.array([0.5, 2.0, 9.0]))
    assert_close(torch.pow(mk(x), -0.5), torch.pow(cpu(x), -0.5), "df64")


@pytest.mark.parametrize("mode", MODES)
def test_lerp_two_branch_formula(mode):
    s, e = rt(np.array([1.0, 2.5, -3.0])), rt(np.array([1e-6, 1e-6, 4.0]))
    for w in (1.0, 0.999, 0.5, 0.3, -0.7, 0.0, 1.5):
        assert_close(
            torch.lerp(mk(s, mode), mk(e, mode), w), torch.lerp(cpu(s), cpu(e), w), mode
        )
    assert_exact(torch.lerp(mk(s, mode), mk(e, mode), 1.0), cpu(e))
    wt = rt(np.array([1.0, 0.4, 0.75]))
    assert_close(
        torch.lerp(mk(s, mode), mk(e, mode), mk(wt, mode)),
        torch.lerp(cpu(s), cpu(e), cpu(wt)),
        mode,
    )
    inf_s, inf_e = np.array([np.inf, 0.0]), np.array([0.0, np.inf])
    assert_exact(
        torch.lerp(mk(inf_s, mode), mk(inf_e, mode), 0.5),
        torch.lerp(cpu(inf_s), cpu(inf_e), 0.5),
    )
    z = mk(s, mode)
    z.lerp_(mk(e, mode), 1.0)
    assert_exact(z, cpu(e))
    z = mk(s, mode)
    z.lerp_(mk(e, mode), mk(wt, mode))
    assert_close(z, torch.lerp(cpu(s), cpu(e), cpu(wt)), mode)


def test_sub_int_alpha_zero_and_bool_operands():
    a, b = np.array([-0.0, 1.0]), np.array([1.5, -2.0])
    x, y = mk(a), mk(b)
    assert_exact(torch.sub(x, y, alpha=0), torch.sub(cpu(a), cpu(b), alpha=0))
    assert_exact(torch.sub(x, y, alpha=0.0), torch.sub(cpu(a), cpu(b), alpha=0.0))
    assert_exact(torch.add(x, y, alpha=0), torch.add(cpu(a), cpu(b), alpha=0))
    assert_exact(torch.rsub(y, x, alpha=0), torch.rsub(cpu(b), cpu(a), alpha=0))
    assert_exact(torch.sub(x, y, alpha=2), torch.sub(cpu(a), cpu(b), alpha=2))
    assert_exact(torch.sub(x, y, alpha=-1.5), torch.sub(cpu(a), cpu(b), alpha=-1.5))
    z = mk(a)
    z.sub_(y, alpha=0)
    assert_exact(z, torch.sub(cpu(a), cpu(b), alpha=0))
    with pytest.raises(NotImplementedError, match="bool tensor is not supported"):
        x - True
    with pytest.raises(NotImplementedError, match="bool tensor is not supported"):
        torch.sub(x, torch.tensor(True))
    with pytest.raises(RuntimeError, match="Boolean alpha"):
        torch.add(x, y, alpha=True)
    with pytest.raises(RuntimeError, match="Boolean alpha"):
        torch.sub(x, y, alpha=True)


def test_addcmul_addcdiv_sf64_with_plain_operand_and_value():
    xs = mk([1.0, 2.0], "sf64")
    f = torch.tensor([0.5, 0.25], dtype=torch.float32, device=MPS)
    cx, cf = cpu([1.0, 2.0]), f.cpu()
    assert_exact(torch.addcmul(xs, f, f, value=2), torch.addcmul(cx, cf, cf, value=2))
    assert_exact(
        torch.addcdiv(xs, f, f, value=0.5), torch.addcdiv(cx, cf, cf, value=0.5)
    )
    z = mk([1.0, 2.0], "sf64")
    z.addcmul_(f, f, value=2)
    assert_exact(z, torch.addcmul(cx, cf, cf, value=2))
    xd = mk([1.0, 2.0])
    assert_exact(torch.addcmul(xd, f, f, value=2), torch.addcmul(cx, cf, cf, value=2))


@pytest.mark.parametrize("mode", MODES)
def test_inplace_comparison_and_logical_variants(mode):
    a, b = rt(np.array([1.0, 2.0, 0.0, -1.0])), rt(np.array([1.0, 1.0, 0.0, 3.0]))
    for name in ("eq_", "ne_", "lt_", "le_", "gt_", "ge_"):
        x, r = mk(a, mode), cpu(a)
        assert getattr(x, name)(mk(b, mode)) is x
        getattr(r, name)(cpu(b))
        assert_exact(x, r)
        x, r = mk(a, mode), cpu(a)
        getattr(x, name)(0.0)
        getattr(r, name)(0.0)
        assert_exact(x, r)
    x, r = mk(a, mode), cpu(a)
    x.logical_not_()
    r.logical_not_()
    assert_exact(x, r)
    for name in ("logical_and_", "logical_or_", "logical_xor_"):
        x, r = mk(a, mode), cpu(a)
        getattr(x, name)(mk(b, mode))
        getattr(r, name)(cpu(b))
        assert_exact(x, r)
    # bool result into a float64 out (torch casts)
    o = mk(np.zeros(4), mode)
    torch.eq(mk(a, mode), mk(b, mode), out=o)
    assert_exact(o, torch.eq(cpu(a), cpu(b), out=torch.zeros(4, dtype=torch.float64)))


def test_lgamma_backward_through_digamma_fallback():
    mt.reset_stats()
    a = rt(np.array([0.5, 2.5, 7.0]))
    x, r = mk(a, requires_grad=True), cpu(a, True)
    torch.lgamma(x).sum().backward()
    torch.lgamma(r).sum().backward()
    np.testing.assert_allclose(x.grad.to_numpy(), r.grad.numpy(), rtol=GRAD_RTOL)
    assert mt.stats().get("cpu_fallback:digamma") == 1
    assert_close(torch.digamma(mk(a)), torch.digamma(cpu(a)), "df64")
    assert_close(torch.polygamma(2, mk(a)), torch.polygamma(2, cpu(a)), "df64")


def test_int32_operands_promote_exactly():
    i32 = torch.tensor(
        [123456789, 16777217, -2147483648, 2147483647], dtype=torch.int32, device=MPS
    )
    ci = i32.cpu()
    zeros = mk(np.zeros(4))
    assert_exact(zeros + i32, torch.zeros(4, dtype=torch.float64) + ci)
    vals = mk([123456789.0, 16777217.0, -2147483648.0, 2147483647.0])
    assert_exact(vals == i32, torch.tensor([True, True, True, True]))
    assert_exact(vals - i32, torch.zeros(4, dtype=torch.float64))
    s = mk([123456789.0, 16777217.0, -2147483648.0, 2147483647.0], "sf64")
    assert_exact(s == i32, torch.tensor([True, True, True, True]))


# ---------------------------------------------------------------------------
# documented limitations (locked in so a silent change would be noticed)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_documented_signed_zero_tie_rule_is_ieee(mode):
    """maximum(-0, +0) = +0 and minimum(+0, -0) = -0 (IEEE 754-2019, numpy, torch's
    vectorized CPU path); torch's scalar path (< 8 elements) returns the first
    operand instead, so the CPU oracle is only consistent on longer tensors."""
    n = 16
    a, b = np.full(n, -0.0), np.zeros(n)
    x, y = mk(a, mode), mk(b, mode)
    assert_exact(torch.maximum(x, y), torch.maximum(cpu(a), cpu(b)))
    assert_exact(torch.minimum(y, x), torch.minimum(cpu(b), cpu(a)))
    assert_exact(torch.relu(x), torch.relu(cpu(a)))
    assert_exact(torch.clamp(x, 0.0, 1.0), torch.clamp(cpu(a), 0.0, 1.0))
    assert not np.signbit(torch.maximum(mk([-0.0], mode), mk([0.0], mode)).to_numpy())[
        0
    ]
    assert np.signbit(torch.minimum(mk([0.0], mode), mk([-0.0], mode)).to_numpy())[0]


def test_documented_nan_to_num_saturates_at_representation_max():
    assert torch.nan_to_num(mk([np.inf, -np.inf, np.nan])).to_numpy().tolist() == [
        float(np.finfo(np.float32).max),
        -float(np.finfo(np.float32).max),
        0.0,
    ]
    assert_exact(
        torch.nan_to_num(mk([np.inf, -np.inf, np.nan], "sf64")),
        torch.nan_to_num(cpu([np.inf, -np.inf, np.nan])),
    )


def test_documented_df64_sub_ulp_pairs_order_by_represented_value():
    """A canonical df64 pair may carry bits below 2^-53: decoded-equal values can
    still compare / sort as distinct (documented in ops_structural)."""
    s = mk([1.0, 1.0, 1.0]) + mk([2.0**-60, 0.0, -(2.0**-60)])
    assert s.to_numpy().tolist() == [1.0, 1.0, 1.0]
    assert torch.sort(s, stable=True).indices.cpu().tolist() == [2, 1, 0]
    assert (s == 1.0).cpu().tolist() == [False, True, False]
