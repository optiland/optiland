"""Dispatch contract tests for the structural ``MetalFloat64`` handlers.

Every op is compared against torch CPU float64 on the decoded inputs. Structural
ops (views, indexing, concatenation, fills, where, sort, searchsorted) must be
*exact*: the same values, shapes, dtypes and (for views) the same strides and
storage offsets, and in-place ops must be visible through aliases. The only
tolerance is for accumulating writes (``index_put(accumulate=True)``,
``scatter_add``, ``index_add``) whose additions run through the df64 ``add``
kernel (rtol 1e-13) or the correctly rounded sf64 ``add`` (rtol 1e-15, the
summation order differs from the CPU).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("torch MPS (Metal GPU) is not available", allow_module_level=True)

from optiland.backend.torch_backend.metal import encode  # noqa: E402
from optiland.backend.torch_backend.metal import tensor as mt  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import MetalFloat64, aten  # noqa: E402

MODES = ("df64", "sf64")
RTOL = {"df64": 1e-13, "sf64": 1e-15}
MPS = torch.device("mps")


HOST_THRESHOLD_FOR_RESIDENCY_TESTS = 256


@pytest.fixture(autouse=True)
def gpu_resident(request, monkeypatch):
    """Force GPU residency so every op reaches the structural handlers.

    The nucleus runs ops on tensors of at most ``HOST_THRESHOLD`` elements on the
    CPU; the handlers under test would never see the small fixtures otherwise.
    ``test_residency_*`` tests instead pin the design's default threshold (256)
    explicitly to exercise the mixed host/GPU paths.
    """
    if "residency" in request.node.name:
        monkeypatch.setattr(mt, "HOST_THRESHOLD", HOST_THRESHOLD_FOR_RESIDENCY_TESTS)
    else:
        monkeypatch.setattr(mt, "HOST_THRESHOLD", 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def rand(shape, seed=0) -> np.ndarray:
    """Random values spanning several decades, exactly representable in both modes.

    A random float64 carries 53 significand bits; df64 keeps 48, so the values
    are rounded through the df64 encoding once (idempotent) to make every
    structural comparison against the CPU reference exact. Both ``hi`` and
    ``lo`` are populated (the fraction with ``lo == 0`` is negligible).
    """
    g = rng(seed)
    a = g.standard_normal(shape) * 10.0 ** g.uniform(-3, 3, shape)
    return np.asarray(encode.decode_df64(*encode.encode_df64(a)), dtype=np.float64)


def lo_only(n: int, seed: int = 0) -> np.ndarray:
    """Values 1 + k*2^-40 whose df64 ``hi`` words are all 1.0 (they differ in lo)."""
    g = rng(seed)
    k = g.permutation(np.arange(-n // 2, n - n // 2))
    return 1.0 + k.astype(np.float64) * 2.0**-40


def mk(a, mode: str, requires_grad: bool = False) -> MetalFloat64:
    return MetalFloat64.from_numpy(
        np.asarray(a, dtype=np.float64), mode, requires_grad=requires_grad
    )


def ref(a, requires_grad: bool = False) -> torch.Tensor:
    t = torch.from_numpy(np.array(a, dtype=np.float64))
    return t.requires_grad_(requires_grad)


def rt(v: float, mode: str) -> float:
    """The value a Python scalar actually takes after encoding into ``mode``."""
    return float(mk(v, mode).to_numpy().reshape(()))


def dec(x) -> np.ndarray:
    if isinstance(x, MetalFloat64):
        return x.to_numpy()
    return x.detach().cpu().numpy()


def assert_exact(got, want, *, view: bool = False) -> None:
    """``got`` (on the GPU) equals ``want`` (CPU float64 reference) exactly."""
    if isinstance(want, (list, tuple)):
        assert isinstance(got, (list, tuple)) and len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            assert_exact(g, w, view=view)
        return
    assert isinstance(got, torch.Tensor)
    assert got.device.type == "mps"
    assert got.dtype == want.dtype, (got.dtype, want.dtype)
    assert isinstance(got, MetalFloat64) == (want.dtype == torch.float64)
    assert tuple(got.shape) == tuple(want.shape), (got.shape, want.shape)
    if view:
        assert got.stride() == want.stride(), (got.stride(), want.stride())
        assert got.storage_offset() == want.storage_offset()
        for c in got.components:
            assert c.stride() == want.stride()
            assert c.storage_offset() == want.storage_offset()
    np.testing.assert_array_equal(dec(got), want.detach().numpy())


def assert_close(got, want, mode: str) -> None:
    assert isinstance(got, MetalFloat64)
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_allclose(
        got.to_numpy(), want.detach().numpy(), rtol=RTOL[mode], atol=0
    )


def layouts(a: np.ndarray, mode: str):
    """Pairs (MetalFloat64, CPU reference) of the same values in several layouts."""
    x, r = mk(a, mode), ref(a)
    yield "contiguous", x, r
    yield "transposed", x.transpose(0, -1), r.transpose(0, -1)
    yield "strided-slice", x[1:, ::2], r[1:, ::2]
    yield "offset", x[1:], r[1:]
    yield "expanded", x[:1].expand(a.shape), r[:1].expand(a.shape)


# ---------------------------------------------------------------------------
# Views and shape ops
# ---------------------------------------------------------------------------
VIEW_CASES = {
    "view": lambda t: t.view(-1),
    "view2": lambda t: t.view(6, 4),
    "reshape": lambda t: t.reshape(4, 6),
    "permute": lambda t: t.permute(2, 0, 1),
    "transpose": lambda t: t.transpose(0, 2),
    "unsqueeze": lambda t: t.unsqueeze(1),
    "squeeze": lambda t: t.unsqueeze(0).squeeze(),
    "squeeze_dim": lambda t: t.unsqueeze(2).squeeze(2),
    "squeeze_dims": lambda t: t.unsqueeze(0).unsqueeze(-1).squeeze((0, -1)),
    "expand": lambda t: t[:1].expand(5, 3, 4),
    "expand_as": lambda t: t[:, :1].expand_as(t),
    "broadcast_to": lambda t: torch.broadcast_to(t[0], (2, 3, 4)),
    "narrow": lambda t: t.narrow(2, 1, 2),
    "select": lambda t: t.select(1, 2),
    "select_neg": lambda t: t[-1],
    "slice": lambda t: t[:, 1:3],
    "slice_step": lambda t: t[::2, :, ::3],
    "diagonal": lambda t: t.diagonal(0, 1, 2),
    "diagonal_off": lambda t: t.diagonal(-1, 2, 1),
    "movedim": lambda t: t.movedim(0, 2),
    "movedim_list": lambda t: t.movedim((0, 1), (2, 0)),
    "unfold": lambda t: t.unfold(2, 2, 1),
    "as_strided": lambda t: t.as_strided((2, 3), (4, 1), 1),
    "t": lambda t: t[0].t(),
}
COPY_CASES = {
    "flip": lambda t: t.flip(0, 2),
    "roll": lambda t: t.roll(1, 1),
    "roll_multi": lambda t: t.roll((1, -2), (0, 2)),
    "roll_flat": lambda t: t.roll(5),
    "repeat": lambda t: t.repeat(2, 1, 3),
    "tile": lambda t: t.tile((2, 2)),
    "repeat_interleave_int": lambda t: t.repeat_interleave(2, dim=1),
    "repeat_interleave_flat": lambda t: t.repeat_interleave(2),
    "repeat_interleave_tensor": lambda t: t.repeat_interleave(
        (torch.arange(t.shape[1], device=t.device) + 2) % 3, dim=1
    ),
    "clone": lambda t: t.clone(),
    "contiguous": lambda t: t.transpose(0, 1).contiguous(),
}
LIST_CASES = {
    "split": lambda t: torch.split(t, 2, dim=2),
    "split_sizes": lambda t: torch.split(t, [1, 3], dim=2),
    "split_with_sizes": lambda t: t.split_with_sizes([2, 0, 1], dim=1),
    "unbind": lambda t: t.unbind(1),
    "chunk": lambda t: t.chunk(3, dim=1),
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", sorted(VIEW_CASES))
def test_view_ops(mode, name):
    a = rand((2, 3, 4), seed=1)
    fn = VIEW_CASES[name]
    x, r = mk(a, mode), ref(a)
    got, want = fn(x), fn(r)
    assert_exact(got, want, view=True)
    # views alias the input storage
    assert got.components[0].untyped_storage().data_ptr() == (
        x.components[0].untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", sorted(COPY_CASES))
def test_copy_ops_on_layouts(mode, name):
    fn = COPY_CASES[name]
    for _, x, r in layouts(rand((3, 3, 4), seed=2), mode):
        assert_exact(fn(x), fn(r))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", sorted(LIST_CASES))
def test_list_returning_ops(mode, name):
    fn = LIST_CASES[name]
    a = rand((2, 3, 4), seed=3)
    x, r = mk(a, mode), ref(a)
    got, want = fn(x), fn(r)
    assert_exact(list(got), list(want), view=True)


@pytest.mark.parametrize("mode", MODES)
def test_views_on_strided_inputs(mode):
    for _, x, r in layouts(rand((3, 3, 4), seed=4), mode):
        assert_exact(x.reshape(-1), r.reshape(-1))
        assert_exact(x.transpose(0, 1), r.transpose(0, 1), view=True)
        assert_exact(x.flip(0), r.flip(0))
        assert_exact(x.unsqueeze(0).squeeze(0), r.unsqueeze(0).squeeze(0), view=True)
        assert_exact(x[1:, 0], r[1:, 0], view=True)


@pytest.mark.parametrize("mode", MODES)
def test_zero_dim_and_empty(mode):
    s, rs = mk(3.5, mode), ref(3.5)
    assert_exact(s.unsqueeze(0), rs.unsqueeze(0), view=True)
    assert_exact(s.view(()), rs.view(()), view=True)
    assert_exact(s.reshape(1, 1), rs.reshape(1, 1), view=True)
    assert_exact(s.expand(2, 3), rs.expand(2, 3), view=True)
    assert_exact(s.clone(), rs.clone())
    assert_exact(torch.stack([s, s]), torch.stack([rs, rs]))
    e, re = mk(np.zeros((0, 3)), mode), ref(np.zeros((0, 3)))
    assert_exact(e.view(0), re.view(0), view=True)
    assert_exact(e.t(), re.t(), view=True)
    assert_exact(torch.cat([e, e]), torch.cat([re, re]))
    assert_exact(e[e.components[0] > 0], re[re > 0])
    assert_exact(e.flip(1), re.flip(1))
    v, i = torch.sort(e, dim=1)
    rv, ri = torch.sort(re, dim=1)
    assert_exact(v, rv)
    assert_exact(i, ri)
    assert_exact(torch.nonzero(e), torch.nonzero(re))


# ---------------------------------------------------------------------------
# cat / stack
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_cat_stack(mode):
    a, b = rand((2, 3), seed=5), rand((4, 3), seed=6)
    x, y, rx, ry = mk(a, mode), mk(b, mode), ref(a), ref(b)
    assert_exact(torch.cat([x, y]), torch.cat([rx, ry]))
    assert_exact(torch.cat([x.t(), y.t()], dim=1), torch.cat([rx.t(), ry.t()], dim=1))
    assert_exact(torch.cat([x, x[:0]]), torch.cat([rx, rx[:0]]))
    assert_exact(torch.stack([x, x.flip(0)]), torch.stack([rx, rx.flip(0)]))
    assert_exact(torch.stack([x, x], dim=2), torch.stack([rx, rx], dim=2))
    assert_exact(torch.hstack([x, x]), torch.hstack([rx, rx]))
    assert_exact(torch.vstack([x, y]), torch.vstack([rx, ry]))


@pytest.mark.parametrize("mode", MODES)
def test_cat_stack_mixed_plain_operands(mode):
    a = rand((2, 3), seed=7)
    f32 = np.float32(rand((2, 3), seed=8))
    x, rx = mk(a, mode), ref(a)
    p = torch.from_numpy(f32).to(MPS)  # plain float32 mps
    rp = torch.from_numpy(f32)
    assert_exact(torch.cat([x, p]), torch.cat([rx, rp]))
    assert_exact(torch.cat([p, x]), torch.cat([rp, rx]))
    assert_exact(torch.stack([p, x]), torch.stack([rp, rx]))
    ip = torch.arange(6, device=MPS).view(2, 3)  # int64 mps
    assert_exact(torch.cat([x, ip]), torch.cat([rx, ip.cpu()]))
    c64 = ref(rand((2, 3), seed=9))  # cpu float64
    assert_exact(torch.cat([x, c64]), torch.cat([rx, c64]))


@pytest.mark.parametrize("mode", MODES)
def test_cat_out(mode):
    a = rand((2, 3), seed=10)
    x, rx = mk(a, mode), ref(a)
    out = mk(np.zeros((4, 3)), mode)
    rout = ref(np.zeros((4, 3)))
    res = torch.cat([x, x.flip(1)], out=out)
    torch.cat([rx, rx.flip(1)], out=rout)
    assert res is out
    assert_exact(out, rout)
    # a wrong-shaped out is resized to the result shape (torch semantics)
    small = mk(np.zeros((2, 3)), mode)
    with pytest.warns(UserWarning, match="was resized"):
        res = torch.cat([x, x], out=small)
    assert res is small and tuple(small.shape) == (4, 3)
    assert_exact(small, torch.cat([rx, rx]))
    empty = mk(np.zeros(0), mode)
    assert torch.cat([x, x], out=empty) is empty
    assert_exact(empty, torch.cat([rx, rx]))


# ---------------------------------------------------------------------------
# index (getitem) with mixed index types
# ---------------------------------------------------------------------------
def index_cases(t):
    dev = t.device
    idx = torch.tensor([2, 0, 2], device=dev)
    idx2 = torch.tensor([[1], [0]], device=dev)
    mask1 = torch.tensor([True, False, True], device=dev)
    mask2 = torch.tensor([[True, False, True], [False] * 3, [True] * 3], device=dev)
    mask4 = torch.tensor([True, False, True, False], device=dev)
    return {
        "int": t[idx],
        "int_dim1": t[:, idx],
        "int_last": t[..., idx],
        "list": t[[0, 2]],
        "python_int_and_tensor": t[1, idx],
        "broadcast_pair": t[idx2, idx],
        "slice_and_tensor": t[1:, idx],
        "mask_dim0": t[mask1],
        "mask_2d": t[mask2],
        "mask_dim1": t[:, mask1],
        "mask_last": t[:, :, mask4],
        "none_and_tensor": t[None, idx],
        "cpu_index": t[torch.tensor([1, 1])],
        "cpu_mask": t[torch.tensor([False, True, True])],
        "empty_index": t[torch.zeros(0, dtype=torch.int64, device=dev)],
        "ellipsis_none": t[..., None],
        "negative": t[torch.tensor([-1, -3], device=dev), 1],
    }


@pytest.mark.parametrize("mode", MODES)
def test_index_tensor(mode):
    a = rand((3, 3, 4), seed=11)
    got, want = index_cases(mk(a, mode)), index_cases(ref(a))
    for name in want:
        assert_exact(got[name], want[name]), name
    # through a non-contiguous base
    xt, rt_ = mk(a, mode).transpose(0, 2), ref(a).transpose(0, 2)
    m = torch.tensor([True, False, True, True], device=MPS)
    assert_exact(xt[m], rt_[m.cpu()])
    assert_exact(xt[:, torch.tensor([2, 2], device=MPS)], rt_[:, torch.tensor([2, 2])])


# ---------------------------------------------------------------------------
# index_put (setitem) and accumulation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_setitem_mask_and_int_indices(mode):
    a = rand((3, 4), seed=12)
    v = rand((3, 4), seed=13)
    x, r = mk(a, mode), ref(a)
    mask = torch.from_numpy(a > 0)
    # mask assignment from a MetalFloat64 of matching count
    x[mask.to(MPS)] = mk(v, mode)[mask]
    r[mask] = ref(v)[mask]
    assert_exact(x, r)
    # 0-d MetalFloat64 broadcast, and a 0-d cpu float64 value
    x[mask.to(MPS)] = mk(2.5, mode)
    r[mask] = 2.5
    assert_exact(x, r)
    x[~mask.to(MPS)] = torch.tensor(-1.25, dtype=torch.float64)
    r[~mask] = -1.25
    assert_exact(x, r)
    # plain float32 mps values are promoted exactly
    p = torch.tensor([1.5, 2.5, 3.5, 4.5], dtype=torch.float32, device=MPS)
    x[torch.tensor([0, 2], device=MPS)] = p
    r[torch.tensor([0, 2])] = p.cpu().double()
    assert_exact(x, r)
    # integer indices, mixed with slices and broadcasting
    x[1:, torch.tensor([3, 0], device=MPS)] = mk(v[:2, :2], mode)
    r[1:, torch.tensor([3, 0])] = ref(v[:2, :2])
    assert_exact(x, r)
    # values aliasing self
    x[mask.to(MPS)] = x[0, 0]
    r[mask] = r[0, 0]
    assert_exact(x, r)
    # cpu index tensors
    x[torch.tensor([2]), torch.tensor([1])] = mk(9.0, mode)
    r[torch.tensor([2]), torch.tensor([1])] = 9.0
    assert_exact(x, r)
    # Python scalars and plain cpu tensors as values (MetalFloat64.__setitem__)
    x[mask.to(MPS)] = rt(0.1, mode)
    r[mask] = rt(0.1, mode)
    assert_exact(x, r)
    x[0, 1:3] = 5
    r[0, 1:3] = 5
    assert_exact(x, r)
    x[:, torch.tensor([1, 3], device=MPS)] = torch.tensor([[1.0, 2.0]] * 3)
    r[:, torch.tensor([1, 3])] = torch.tensor([[1.0, 2.0]] * 3, dtype=torch.float64)
    assert_exact(x, r)


@pytest.mark.parametrize("mode", MODES)
def test_setitem_duplicate_indices_last_wins(mode):
    a = rand((5, 2), seed=14)
    v = rand((4, 2), seed=15)
    x, r = mk(a, mode), ref(a)
    idx = torch.tensor([1, 3, 1, 1])
    x[idx.to(MPS)] = mk(v, mode)
    r[idx] = ref(v)  # CPU: sequential writes, last one wins
    assert_exact(x, r)
    # both components must agree on the winner (pair stays canonical)
    assert_exact(x[1], ref(v[3]))


@pytest.mark.parametrize("mode", MODES)
def test_setitem_through_views_visible_in_base(mode):
    a = rand((4, 5), seed=16)
    x, r = mk(a, mode), ref(a)
    mask = torch.tensor([True, False, True, False, True])
    x[1:3][:, mask.to(MPS)] = mk(7.0, mode)
    r[1:3][:, mask] = 7.0
    assert_exact(x, r)
    xt, rt_ = x.t(), r.t()  # non-contiguous self for index_put_
    xt[torch.tensor([4, 0], device=MPS)] = mk(v := rand((2, 4), seed=17), mode)
    rt_[torch.tensor([4, 0])] = ref(v)
    assert_exact(x, r)
    xt[torch.tensor([2, 2, 2], device=MPS), torch.tensor([1, 2, 1], device=MPS)] = mk(
        [1.0, 2.0, 3.0], mode
    )
    rt_[torch.tensor([2, 2, 2]), torch.tensor([1, 2, 1])] = ref([1.0, 2.0, 3.0])
    assert_exact(x, r)


@pytest.mark.parametrize("mode", MODES)
def test_index_put_accumulate(mode):
    a = rand((6,), seed=18)
    v = rand((9,), seed=19)
    x, r = mk(a, mode), ref(a)
    idx = torch.tensor([1, 1, 1, 1, 1, 4, 4, 0, 1])  # multiplicity 6 at index 1
    mt.reset_stats()
    x.index_put_((idx.to(MPS),), mk(v, mode), accumulate=True)
    r.index_put_((idx,), ref(v), accumulate=True)
    assert_close(x, r, mode)
    assert mt.stats().get("gpu:add", 0) > 0
    assert not any(k.startswith("cpu_fallback") for k in mt.stats())
    # unique indices: a single exact add per element
    x, r = mk(a, mode), ref(a)
    y = x.index_put((torch.tensor([5, 2], device=MPS),), mk([1.0, 2.0], mode), True)
    ry = r.index_put((torch.tensor([5, 2]),), ref([1.0, 2.0]), True)
    assert_close(y, ry, mode)
    assert_exact(x, r)  # out-of-place left the input alone
    # boolean mask accumulate with a broadcast scalar
    a2 = rand((3, 4), seed=20)
    x, r = mk(a2, mode), ref(a2)
    m = torch.from_numpy(a2 > 0)
    x.index_put_((m.to(MPS),), mk(0.5, mode), accumulate=True)
    r.index_put_((m,), ref(0.5), accumulate=True)
    assert_close(x, r, mode)
    # 2-D int indices with duplicates on a transposed view
    x, r = mk(a2, mode), ref(a2)
    xt, rt_ = x.t(), r.t()
    i0 = torch.tensor([3, 3, 0, 3])
    i1 = torch.tensor([2, 2, 1, 2])
    xt.index_put_((i0.to(MPS), i1.to(MPS)), mk([1.0, 2.0, 3.0, 4.0], mode), True)
    rt_.index_put_((i0, i1), ref([1.0, 2.0, 3.0, 4.0]), True)
    assert_close(x, r, mode)


# ---------------------------------------------------------------------------
# index_select / gather / scatter / masked ops / take / index_add
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_select_gather_scatter(mode):
    a = rand((3, 4), seed=21)
    s = rand((3, 4), seed=22)
    for _, x, r in layouts(rand((3, 3, 4), seed=23), mode):
        i = torch.tensor([2, 0, 2]) % x.shape[1]
        assert_exact(x.index_select(1, i.to(MPS)), r.index_select(1, i))
        i0 = torch.tensor([2, 0]) % x.shape[0]
        assert_exact(x.index_select(0, i0.to(MPS)), r.index_select(0, i0))
        g = torch.randint(
            0, x.shape[1], x.shape, generator=torch.Generator().manual_seed(3)
        )
        assert_exact(x.gather(1, g.to(MPS)), r.gather(1, g))
        assert_exact(torch.take(x, i.to(MPS)), torch.take(r, i))
    x, r = mk(a, mode), ref(a)
    idx = torch.tensor([[0, 1, 2, 0], [3, 2, 1, 0], [1, 1, 1, 1]])
    assert_exact(x.scatter(1, idx.to(MPS), mk(s, mode)), r.scatter(1, idx, ref(s)))
    assert_exact(x.scatter(1, idx.to(MPS), 0.5), r.scatter(1, idx, 0.5))
    idx0 = idx[:2] % 3
    assert_exact(
        x.scatter(0, idx0.to(MPS), rt(0.1, mode)), r.scatter(0, idx0, rt(0.1, mode))
    )
    # duplicate destinations: torch CPU keeps the last write; the pair stays canonical
    dup = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])
    assert_exact(x.scatter(1, dup.to(MPS), mk(s, mode)), r.scatter(1, dup, ref(s)))
    assert_exact(x.scatter(0, dup.to(MPS), mk(s, mode)), r.scatter(0, dup, ref(s)))
    assert_exact(x.scatter(0, dup.to(MPS), 2.0), r.scatter(0, dup, 2.0))
    assert_exact(
        x.t().scatter(0, dup.to(MPS)[:, :3], mk(s, mode)),
        r.t().scatter(0, dup[:, :3], ref(s)),
    )
    y = x.clone()
    ry = r.clone()
    assert y.scatter_(1, idx.to(MPS), mk(s, mode)) is y
    ry.scatter_(1, idx, ref(s))
    assert_exact(y, ry)
    with pytest.raises(NotImplementedError):
        x.scatter(1, idx.to(MPS), mk(s, mode), reduce="add")
    with pytest.raises(NotImplementedError):
        x.scatter_(1, idx.to(MPS), 1.0, reduce="multiply")


@pytest.mark.parametrize("mode", MODES)
def test_scatter_add_index_add(mode):
    a = rand((3, 5), seed=24)
    s = rand((3, 5), seed=25)
    x, r = mk(a, mode), ref(a)
    idx = torch.tensor([[0, 0, 0, 4, 4], [1, 2, 1, 2, 1], [3, 3, 3, 3, 3]])
    assert_close(
        x.scatter_add(1, idx.to(MPS), mk(s, mode)), r.scatter_add(1, idx, ref(s)), mode
    )
    assert_close(
        x.scatter_add(0, idx[:2, :3].to(MPS) % 3, mk(s, mode)),
        r.scatter_add(0, idx[:2, :3] % 3, ref(s)),
        mode,
    )
    y, ry = x.clone(), r.clone()
    assert y.scatter_add_(1, idx.to(MPS), mk(s, mode)) is y
    ry.scatter_add_(1, idx, ref(s))
    assert_close(y, ry, mode)
    assert_exact(x, r)
    i = torch.tensor([4, 4, 0, 4])
    src = rand((3, 4), seed=26)
    assert_close(
        x.index_add(1, i.to(MPS), mk(src, mode)), r.index_add(1, i, ref(src)), mode
    )
    assert_close(
        x.index_add(1, i.to(MPS), mk(src, mode), alpha=2.5),
        r.index_add(1, i, ref(src), alpha=2.5),
        mode,
    )
    xt, rt_ = x.t(), r.t()
    xt.index_add_(0, i.to(MPS), mk(src.T, mode))
    rt_.index_add_(0, i, ref(src.T))
    assert_close(x, r, mode)
    # adding zeros is exact in both representations
    assert_exact(
        x.index_add(0, torch.tensor([1], device=MPS), mk(np.zeros((1, 5)), mode)),
        x.to_cpu_float64(),
    )


@pytest.mark.parametrize("mode", MODES)
def test_masked_ops_and_fills(mode):
    a = rand((3, 4), seed=27)
    x, r = mk(a, mode), ref(a)
    m = torch.from_numpy(a > 0)
    assert_exact(x.masked_select(m.to(MPS)), r.masked_select(m))
    assert_exact(x.t().masked_select(m.t().to(MPS)), r.t().masked_select(m.t()))
    assert_exact(x.masked_fill(m.to(MPS), -0.5), r.masked_fill(m, -0.5))
    assert_exact(
        x.masked_fill(m.to(MPS), rt(0.1, mode)), r.masked_fill(m, rt(0.1, mode))
    )
    assert_exact(x.masked_fill(m.to(MPS), mk(3.0, mode)), r.masked_fill(m, ref(3.0)))
    assert_exact(x.masked_fill(m.to(MPS), True), r.masked_fill(m, True))
    src = rand((12,), seed=28)
    assert_exact(
        x.masked_scatter(m.to(MPS), mk(src, mode)), r.masked_scatter(m, ref(src))
    )
    y, ry = x.clone(), r.clone()
    assert y.masked_fill_(m.to(MPS), 4.0) is y
    ry.masked_fill_(m, 4.0)
    assert_exact(y, ry)
    y.masked_scatter_(~m.to(MPS), mk(src, mode))
    ry.masked_scatter_(~m, ref(src))
    assert_exact(y, ry)
    # fill_, zero_, fill through views (visible in the base)
    y, ry = x.clone(), r.clone()
    assert y.fill_(rt(1e-3, mode)) is y
    ry.fill_(rt(1e-3, mode))
    assert_exact(y, ry)
    y.fill_(mk(-2.0, mode))
    ry.fill_(ref(-2.0))
    assert_exact(y, ry)
    y, ry = x.clone(), r.clone()
    y[1:3].fill_(7.0)
    ry[1:3].fill_(7.0)
    assert_exact(y, ry)
    y.t()[0].zero_()
    ry.t()[0].zero_()
    assert_exact(y, ry)
    y.view(-1)[2:5].fill_(mk(0.25, mode))
    ry.view(-1)[2:5].fill_(ref(0.25))
    assert_exact(y, ry)
    # index_fill / index_copy
    i = torch.tensor([3, 0])
    assert_exact(x.index_fill(1, i.to(MPS), 5.0), r.index_fill(1, i, 5.0))
    assert_exact(
        x.index_fill(1, i.to(MPS), mk(6.0, mode)), r.index_fill(1, i, ref(6.0))
    )
    c = rand((3, 2), seed=29)
    assert_exact(x.index_copy(1, i.to(MPS), mk(c, mode)), r.index_copy(1, i, ref(c)))


# ---------------------------------------------------------------------------
# where
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_where(mode):
    a, b = rand((3, 4), seed=30), rand((3, 4), seed=31)
    x, y, rx, ry = mk(a, mode), mk(b, mode), ref(a), ref(b)
    cond = torch.from_numpy(a > b)
    assert_exact(torch.where(cond.to(MPS), x, y), torch.where(cond, rx, ry))
    assert_exact(
        torch.where(cond.to(MPS), x.t().t(), y[0]), torch.where(cond, rx, ry[0])
    )
    p = torch.from_numpy(np.float32(b)).to(MPS)  # plain float32 mps operand
    assert_exact(
        torch.where(cond.to(MPS), x, p), torch.where(cond, rx, p.cpu().double())
    )
    assert_exact(
        torch.where(cond.to(MPS), p, x), torch.where(cond, p.cpu().double(), rx)
    )
    # 0-d cpu float64 operand (what a wrapped Python scalar looks like at dispatch)
    z = torch.tensor(0.0, dtype=torch.float64)
    assert_exact(aten.where.self(cond.to(MPS), x, z), aten.where.self(cond, rx, z))
    assert_exact(aten.where.self(cond.to(MPS), z, x), aten.where.self(cond, z, rx))
    # broadcasting condition
    c1 = torch.tensor([True, False, True, False])
    assert_exact(torch.where(c1.to(MPS), x, y), torch.where(c1, rx, ry))
    out = mk(np.zeros((3, 4)), mode)
    assert torch.where(cond.to(MPS), x, y, out=out) is out
    assert_exact(out, torch.where(cond, rx, ry))


# ---------------------------------------------------------------------------
# sort / argsort / topk / searchsorted / bucketize / nonzero
# ---------------------------------------------------------------------------
def sort_inputs():
    g = rng(32)
    vals = rand((40,), seed=33)
    special = np.array([np.nan, -0.0, 0.0, np.inf, -np.inf, 1.0, 1.0, -1.0, np.nan])
    lo = lo_only(64, seed=34)
    dup = np.repeat(lo[:8], 3)
    yield "random", vals
    yield "special", np.concatenate([vals[:6], special, vals[6:12]])
    yield "lo_only", lo
    yield "lo_only_dups", g.permutation(dup)
    yield "matrix", np.concatenate([lo[:24], vals[:24]]).reshape(4, 12)
    yield "ties", np.array([2.0, 1.0, 2.0, 1.0, 2.0, 1.0])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("descending", [False, True])
def test_sort_argsort(mode, descending):
    for name, a in sort_inputs():
        x, r = mk(a, mode), ref(a)
        for dim in range(a.ndim):
            v, i = torch.sort(x, dim=dim, descending=descending, stable=True)
            rv, ri = torch.sort(r, dim=dim, descending=descending, stable=True)
            assert_exact(v, rv), name
            assert_exact(i, ri), name
            (
                assert_exact(
                    torch.argsort(x, dim=dim, descending=descending, stable=True), ri
                ),
                name,
            )
            v2, i2 = torch.sort(x, dim=dim, descending=descending)
            assert_exact(v2, rv), name
            if name == "lo_only":
                # numpy oracle on the decoded values
                order = np.argsort(a, kind="stable")
                if descending:
                    order = np.argsort(-a, kind="stable")
                np.testing.assert_array_equal(i.cpu().numpy(), order)
    # transposed input and 0-d
    a = rand((5, 6), seed=35)
    x, r = mk(a, mode).t(), ref(a).t()
    v, i = torch.sort(x, dim=0, descending=descending)
    rv, ri = torch.sort(r, dim=0, descending=descending)
    assert_exact(v, rv)
    assert_exact(i, ri)
    v, i = torch.sort(mk(2.0, mode))
    rv, ri = torch.sort(ref(2.0))
    assert_exact(v, rv)
    assert_exact(i, ri)


@pytest.mark.parametrize("mode", MODES)
def test_topk(mode):
    a = np.concatenate([lo_only(16, seed=36), rand((16,), seed=37)]).reshape(4, 8)
    x, r = mk(a, mode), ref(a)
    for largest in (True, False):
        v, i = torch.topk(x, 3, dim=1, largest=largest)
        rv, ri = torch.topk(r, 3, dim=1, largest=largest)
        assert_exact(v, rv)
        assert_exact(i, ri)
    v, i = torch.topk(x, 2, dim=0)
    rv, ri = torch.topk(r, 2, dim=0)
    assert_exact(v, rv)
    assert_exact(i, ri)


@pytest.mark.parametrize("mode", MODES)
def test_searchsorted_bucketize(mode):
    seq = np.sort(lo_only(32, seed=38))
    seq[10] = seq[9]  # a duplicate boundary
    vals = np.concatenate(
        [seq[[0, 9, 10, 31]], seq[:5] + 2.0**-41, seq[-3:] - 2.0**-41, [0.5, 2.0]]
    )
    xs, xv, rs, rv = mk(seq, mode), mk(vals, mode), ref(seq), ref(vals)
    for right in (False, True):
        assert_exact(
            torch.searchsorted(xs, xv, right=right),
            torch.searchsorted(rs, rv, right=right),
        )
        assert_exact(
            torch.searchsorted(xs, xv, right=right, out_int32=True),
            torch.searchsorted(rs, rv, right=right, out_int32=True),
        )
        assert_exact(
            torch.bucketize(xv, xs, right=right), torch.bucketize(rv, rs, right=right)
        )
        np.testing.assert_array_equal(
            torch.searchsorted(xs, xv, right=right).cpu().numpy(),
            np.searchsorted(seq, vals, side="right" if right else "left"),
        )
    assert_exact(
        torch.searchsorted(xs, xv, side="right"),
        torch.searchsorted(rs, rv, side="right"),
    )
    # scalar value overload and a plain float32 value tensor
    assert_exact(torch.searchsorted(xs, 1.0), torch.searchsorted(rs, 1.0))
    p = torch.tensor([0.0, 1.0, 5.0], dtype=torch.float32, device=MPS)
    assert_exact(torch.searchsorted(xs, p), torch.searchsorted(rs, p.cpu().double()))
    # unsorted sequence with sorter, and a 2-D sequence
    perm = rng(39).permutation(32)
    us, rus = mk(seq[perm], mode), ref(seq[perm])
    sorter = torch.from_numpy(np.argsort(perm))
    assert_exact(
        torch.searchsorted(us, xv, sorter=sorter.to(MPS)),
        torch.searchsorted(rus, rv, sorter=sorter),
    )
    seq2 = np.stack([seq, seq + 1.0])
    v2 = np.stack([vals[:6], vals[:6] + 1.0])
    assert_exact(
        torch.searchsorted(mk(seq2, mode), mk(v2, mode)),
        torch.searchsorted(ref(seq2), ref(v2)),
    )


@pytest.mark.parametrize("mode", MODES)
def test_nonzero(mode):
    a = np.array([[0.0, -0.0, np.nan, 1e-30], [np.inf, 0.0, -3.0, 2.0**-40]])
    x, r = mk(a, mode), ref(a)
    assert_exact(torch.nonzero(x), torch.nonzero(r))
    assert_exact(torch.nonzero(x.t()), torch.nonzero(r.t()))
    assert_exact(torch.nonzero(mk(0.0, mode)), torch.nonzero(ref(0.0)))
    assert_exact(torch.nonzero(mk(2.0, mode)), torch.nonzero(ref(2.0)))
    out = torch.zeros(0, dtype=torch.int64, device=MPS)
    torch.nonzero(x, out=out)
    assert_exact(out, torch.nonzero(r))
    assert_exact(x.nonzero(as_tuple=True), r.nonzero(as_tuple=True))


# ---------------------------------------------------------------------------
# Autograd through structural ops
# ---------------------------------------------------------------------------
# (function, output shape, accumulates): the backward of an op that reads an
# element more than once (duplicate indices, overlapping windows) adds gradient
# contributions with the df64/sf64 ``add`` kernel, so it is compared with RTOL.
# ``split``/``unbind``/``chunk`` consume every piece here: autograd builds the
# gradient of an unused piece with a plain ``aten.zeros(dtype=float64,
# device=mps)`` served by the nucleus's factory intercept in the *global*
# representation (see test_autograd_partial_split_uses_global_mode).
GRAD_CASES = {
    "view_chain": (lambda t: t.view(12).t().unsqueeze(0).permute(1, 0), (12, 1), False),
    "reshape_transposed": (lambda t: t.t().reshape(2, 6), (2, 6), False),
    "select": (lambda t: t.select(1, 2), (3,), False),
    "slice": (lambda t: t[1:, ::2], (2, 2), False),
    "diagonal": (lambda t: t.diagonal(1), (3,), False),
    "expand_view": (lambda t: t[:, :1].expand(3, 4)[:, 0], (3,), False),
    "flip_roll": (lambda t: t.flip(0).roll(1, 1), (3, 4), False),
    "stack_unbind": (lambda t: torch.stack(t.unbind(1), 0), (4, 3), False),
    "stack_unbind_rev": (lambda t: torch.stack(t.unbind(0)[::-1]), (3, 4), False),
    "cat_split": (
        lambda t: torch.cat(torch.split(t, [1, 3], dim=1)[::-1], 1),
        (3, 4),
        False,
    ),
    "cat_chunk": (lambda t: torch.cat(t.chunk(2, dim=0)[::-1], 0), (3, 4), False),
    "masked_select": (lambda t: t.masked_select(MASK.to(t.device)), (6,), False),
    "index_select": (lambda t: t.index_select(0, IDX.to(t.device)), (4, 4), True),
    "index_select_unique": (
        lambda t: t.index_select(1, UIDX.to(t.device)),
        (3, 3),
        False,
    ),
    "gather": (lambda t: t.gather(1, GIDX.to(t.device)), (3, 2), True),
    "take": (lambda t: torch.take(t, TIDX.to(t.device)), (2, 2), True),
    "sort_values": (lambda t: torch.sort(t, dim=1)[0], (3, 4), False),
    "unfold": (lambda t: t.unfold(1, 2, 2), (3, 2, 2), False),
    "unfold_overlap": (lambda t: t.unfold(1, 3, 1), (3, 2, 3), True),
    "narrow_squeeze": (lambda t: t.narrow(0, 1, 1).squeeze(0), (4,), False),
    "index_bool": (lambda t: t[MASK.to(t.device)], (6,), False),
    "index_int": (lambda t: t[IDX.to(t.device), 1], (4,), True),
}
MASK = torch.tensor(
    [[True, False, True, False], [False, True, True, False], [True, False, False, True]]
)
IDX = torch.tensor([2, 0, 2, 1])
GIDX = torch.tensor([[0, 3], [1, 1], [2, 0]])
TIDX = torch.tensor([[0, 11], [5, 5]])
UIDX = torch.tensor([3, 0, 2])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", sorted(GRAD_CASES))
def test_autograd_grad_through_structural_ops(mode, name):
    fn, out_shape, accumulates = GRAD_CASES[name]
    a = rand((3, 4), seed=40)
    g = rand(out_shape, seed=41)
    x, r = mk(a, mode, requires_grad=True), ref(a, requires_grad=True)
    y, ry = fn(x), fn(r)
    assert isinstance(y, MetalFloat64)
    assert y.requires_grad and y.grad_fn is not None
    assert type(y.grad_fn).__name__ == type(ry.grad_fn).__name__
    assert_exact(y, ry)
    mt.reset_stats()
    (gx,) = torch.autograd.grad(y, x, grad_outputs=mk(g, mode))
    (grx,) = torch.autograd.grad(ry, r, grad_outputs=ref(g))
    assert not any(k.startswith("cpu_fallback") for k in mt.stats())
    if accumulates:
        assert_close(gx, grx, mode)
    else:
        assert_exact(gx, grx)


@pytest.mark.parametrize("mode", MODES)
def test_autograd_partial_split_uses_global_mode(mode):
    """Unused split/unbind pieces get zero grads built in the global representation."""
    from optiland.backend.torch_backend import metal

    previous = metal.get_mode()
    metal.set_mode(mode)
    try:
        a = rand((3, 4), seed=44)
        for fn, gs in (
            (lambda t: torch.split(t, [1, 3], dim=1)[1], (3, 3)),
            (lambda t: t.unbind(0)[1], (4,)),
            (lambda t: t.chunk(2, dim=1)[0], (3, 2)),
        ):
            g = rand(gs, seed=45)
            x, r = mk(a, mode, requires_grad=True), ref(a, requires_grad=True)
            (gx,) = torch.autograd.grad(fn(x), x, grad_outputs=mk(g, mode))
            (grx,) = torch.autograd.grad(fn(r), r, grad_outputs=ref(g))
            assert gx.mode == mode
            assert_exact(gx, grx)
    finally:
        metal.set_mode(previous)


@pytest.mark.parametrize("mode", MODES)
def test_requires_grad_propagation_and_leaf_semantics(mode):
    x = mk(rand((3, 4), seed=42), mode, requires_grad=True)
    for y in (x.view(-1), x[1], x[x.components[0] > 0], torch.cat([x, x]), x.t()):
        assert y.requires_grad and y.grad_fn is not None
    z = x.detach()
    assert not z.requires_grad and z.grad_fn is None
    assert z.view(-1).grad_fn is None
    with pytest.raises(RuntimeError):
        x[1].fill_(0.0)  # in-place on a view of a leaf requiring grad
    assert_exact(x.detach()[1], ref(rand((3, 4), seed=42))[1])


# ---------------------------------------------------------------------------
# Contract: unhandled ops raise, nothing structural falls back to the CPU
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_unhandled_and_unsupported_raise(mode):
    x = mk(rand((2, 3), seed=43), mode)
    with pytest.raises(NotImplementedError):
        x.view(torch.int64)  # view.dtype is deliberately not registered
    with pytest.raises(NotImplementedError):
        idx = torch.zeros((1, 3), dtype=torch.int64, device=MPS)
        x.scatter(0, idx, x, reduce="add")


def test_no_cpu_fallbacks_in_structural_lane():
    assert not any(k.startswith("cpu_fallback:") for k in mt.stats())


# ---------------------------------------------------------------------------
# Dual residency (default HOST_THRESHOLD): host-resident operands mixed with
# GPU-resident ones must reach the same values as the CPU oracle
# ---------------------------------------------------------------------------
def assert_values(got, want, mode=None):
    """Value/shape/dtype equality regardless of where a result lives."""
    assert got.dtype == want.dtype
    assert tuple(got.shape) == tuple(want.shape)
    if mode is None:
        np.testing.assert_array_equal(dec(got), want.detach().numpy())
    else:
        np.testing.assert_allclose(
            dec(got), want.detach().numpy(), rtol=RTOL[mode], atol=0
        )


@pytest.mark.parametrize("mode", MODES)
def test_residency_mixed_operands(mode):
    assert mt.HOST_THRESHOLD == HOST_THRESHOLD_FOR_RESIDENCY_TESTS
    big = rand((20, 30), seed=50)  # 600 elements: GPU-resident
    small = rand((3, 30), seed=51)  # host-resident
    x, r = mk(big, mode), ref(big)
    s, rs = mk(small, mode), ref(small)
    assert not x.is_host_resident and s.is_host_resident
    assert_values(torch.cat([x, s]), torch.cat([r, rs]))
    assert_values(torch.cat([s, x, s]), torch.cat([rs, r, rs]))
    idx = torch.tensor([19, 0, 7])
    x[idx.to(MPS)] = s
    r[idx] = rs
    assert_values(x, r)
    mask = torch.from_numpy(big > 0)
    x[mask.to(MPS)] = 2.5
    r[mask] = 2.5
    assert_values(x, r)
    x[~mask.to(MPS)] = mk(-1.0, mode)  # 0-d host-resident value
    r[~mask] = -1.0
    assert_values(x, r)
    c = torch.from_numpy(big > 1.0)
    assert_values(
        torch.where(c.to(MPS), x, s[:1].expand(20, 30)),
        torch.where(c, r, rs[:1].expand(20, 30)),
    )
    # small GPU-resident views of a large tensor keep aliasing under mutation
    v = x[2]
    assert not v.is_host_resident
    v.fill_(mk(3.0, mode))
    r[2].fill_(3.0)
    assert_values(x, r)
    x[2].zero_()
    r[2].zero_()
    assert_values(x, r)
    vals, ind = torch.sort(x[5, :10])
    rv, ri = torch.sort(r[5, :10])
    assert_values(vals, rv)
    assert_values(ind, ri)
    # accumulate into a large GPU tensor from host-resident values
    x.index_put_((idx.to(MPS),), mk(rand((3, 30), seed=52), mode), accumulate=True)
    r.index_put_((idx,), ref(rand((3, 30), seed=52)), accumulate=True)
    assert_values(x, r, mode)


@pytest.mark.parametrize("mode", MODES)
def test_residency_chained_view_of_large_tensor_aliases(mode):
    """``x[:2, :3]`` is a slice of a 60-element GPU view; zero_ must reach ``x``."""
    assert mt.HOST_THRESHOLD == HOST_THRESHOLD_FOR_RESIDENCY_TESTS
    big = rand((20, 30), seed=57)
    x, r = mk(big, mode), ref(big)
    x[:2, :3].zero_()
    r[:2, :3].zero_()
    assert_values(x, r)


@pytest.mark.parametrize("mode", MODES)
def test_residency_host_target_with_large_operand(mode):
    """In-place ops on a host-resident target must never lose the mutation."""
    assert mt.HOST_THRESHOLD == HOST_THRESHOLD_FOR_RESIDENCY_TESTS
    a = rand((10,), seed=53)
    x, r = mk(a, mode), ref(a)
    assert x.is_host_resident
    idx = torch.from_numpy(rng(54).integers(0, 10, 1000))  # > HOST_THRESHOLD entries
    v = rand((1000,), seed=55)
    x.index_put_((idx.to(MPS),), mk(v, mode), accumulate=True)
    r.index_put_((idx,), ref(v), accumulate=True)
    assert_values(x, r, mode)
    x.index_put_((idx.to(MPS),), mk(v, mode))
    r.index_put_((idx,), ref(v))
    assert_values(x, r)
    big_mask_src = rand((1000,), seed=56)
    x.masked_scatter_(torch.tensor([True] * 10, device=MPS), mk(big_mask_src, mode))
    r.masked_scatter_(torch.tensor([True] * 10), ref(big_mask_src))
    assert_values(x, r)
    x.scatter_add_(0, idx.to(MPS), mk(v, mode))
    r.scatter_add_(0, idx, ref(v))
    assert_values(x, r, mode)
    assert x.is_host_resident
    # a host-resident view of a host tensor as the target
    y, ry = mk(a, mode), ref(a)
    y[2:8].index_add_(0, idx.to(MPS) % 6, mk(v, mode))
    ry[2:8].index_add_(0, idx % 6, ref(v))
    assert_values(y, ry, mode)
