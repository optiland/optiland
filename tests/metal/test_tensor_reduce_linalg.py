"""Dispatch contract tests for the reduction and linear-algebra lanes (MetalFloat64).

Oracle: torch CPU float64 (or ``math.fsum`` / exact rationals) evaluated on the
*decoded* inputs, so that the host encoding error never enters a comparison.
Reductions and products are compared with error bounds derived from the kernel
contracts (``reduce.metal``, ``matmul.metal``), structural results, indices and
predicates must be exact, and every CPU fallback must be counted.

The nucleus' host residency is disabled for these tests (small tensors would
otherwise run on the CPU before the handlers are reached); one test re-enables
it to show the two paths agree.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import itertools  # noqa: E402
import math  # noqa: E402
from fractions import Fraction  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
    pytest.skip("torch MPS backend is not available", allow_module_level=True)

from optiland.backend.torch_backend.metal import (  # noqa: E402
    encode,  # noqa: E402
    ops_linalg,
    ops_reduce,
)
from optiland.backend.torch_backend.metal import tensor as mt  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import (  # noqa: E402
    MetalFallbackError,
    MetalFloat64,
    aten,
)

U2 = 2.0**-48
RTOL = 1e-13
NAN, INF = float("nan"), float("inf")


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _gpu_resident(monkeypatch):
    """Force GPU residency so every op reaches the handlers; clean counters."""
    if hasattr(mt, "HOST_THRESHOLD"):
        monkeypatch.setattr(mt, "HOST_THRESHOLD", -1)
    mt.reset_stats()
    yield
    mt.reset_stats()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260916)


@pytest.fixture(scope="module")
def sf64_kernels():
    """The sf64 reduce/matmul kernel library, or skip when its stack is missing."""
    try:
        return ops_reduce.reduce_kernels("sf64")
    except Exception as exc:  # pragma: no cover - depends on kernel lane progress
        pytest.skip(f"sf64 reduce kernels unavailable: {str(exc)[:200]}")


@pytest.fixture(scope="module")
def sf64_elementwise():
    """The sf64 elementwise library (needed by mean/var/norm), or skip."""
    try:
        return mt.library("sf64")
    except Exception as exc:  # pragma: no cover - depends on kernel lane progress
        pytest.skip(f"sf64 elementwise library unavailable: {str(exc)[:200]}")


def dec(a) -> np.ndarray:
    """The float64 values a df64 encoding of ``a`` actually carries."""
    hi, lo = encode.encode_df64(np.asarray(a, dtype=np.float64))
    return encode.decode_df64(hi, lo)


def metal(a, mode: str = "df64", requires_grad: bool = False) -> MetalFloat64:
    return MetalFloat64.from_numpy(
        np.asarray(a, dtype=np.float64), mode=mode, requires_grad=requires_grad
    )


def cpu(a) -> torch.Tensor:
    return torch.from_numpy(np.array(a, dtype=np.float64))


def val(x) -> np.ndarray:
    """Decode a MetalFloat64 (or copy a plain tensor) to numpy, keeping the shape."""
    if isinstance(x, MetalFloat64):
        return np.asarray(x.to_numpy()).reshape(tuple(x.shape))
    return x.detach().cpu().numpy()


def check(got, ref, rtol: float = RTOL, atol=0.0) -> None:
    """Compare ``got`` (Metal result) with ``ref`` (CPU tensor / array).

    Floating results must satisfy ``|got - ref| <= atol + rtol * |ref|`` elementwise
    (``atol`` may be an array), NaNs must coincide; bool/int results must be exact.
    """
    r = ref.detach().numpy() if torch.is_tensor(ref) else np.asarray(ref)
    if isinstance(got, MetalFloat64):
        assert got.dtype == torch.float64 and got.device.type == "mps"
        g = val(got)
    else:
        assert torch.is_tensor(got), type(got)
        assert got.device.type == "mps", got.device
        if torch.is_tensor(ref):
            assert got.dtype == ref.dtype, (got.dtype, ref.dtype)
        g = got.detach().cpu().numpy()
    assert g.shape == r.shape, (g.shape, r.shape)
    if r.dtype.kind in "biu":
        assert np.array_equal(g, r), (g, r)
        return
    nan_g, nan_r = np.isnan(g), np.isnan(r)
    assert np.array_equal(nan_g, nan_r), (g, r)
    with np.errstate(invalid="ignore"):
        ok = np.abs(g - r) <= np.asarray(atol) + rtol * np.abs(r)
    ok = ok | nan_r | (g == r)
    assert np.all(ok), f"max abs err {np.nanmax(np.abs(g - r))}\n got {g}\n ref {r}"


def fsum_axis(x: np.ndarray, axis: int) -> np.ndarray:
    return np.apply_along_axis(lambda v: math.fsum(v.tolist()), axis, x)


def fsum_dims(x: np.ndarray, dims, keepdim: bool = False) -> np.ndarray:
    """Exact (fsum) sum over several dims."""
    dims = sorted(d % x.ndim for d in dims)
    keep = [i for i in range(x.ndim) if i not in dims]
    moved = np.transpose(x, keep + dims).reshape([x.shape[i] for i in keep] + [-1])
    out = np.apply_along_axis(lambda v: math.fsum(v.tolist()), -1, moved)
    if keepdim:
        out = out.reshape([1 if i in dims else x.shape[i] for i in range(x.ndim)])
    return out


def sum_bound(n: int, abs_sum: np.ndarray) -> np.ndarray:
    """Error budget of a sum of ``n`` terms (sequential worst case or tree depth)."""
    seq = 3.0 * n if n < ops_reduce.TREE_MIN_N else 0.0
    return (8.0 * math.log2(max(n, 2)) + seq) * U2 * np.asarray(abs_sum)


def exact_cumsum_1d(v: np.ndarray) -> np.ndarray:
    scale = 2**120
    ints = [int(math.ldexp(float(t), 120)) for t in v]
    return np.array([p / scale for p in itertools.accumulate(ints)], dtype=np.float64)


def stat(name: str) -> int:
    return mt.stats().get(name, 0)


# ---------------------------------------------------------------------------
# sum / nansum
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 5, 100, 1000, 4097, 2**20])
def test_sum_full_random_mixed_scales(n):
    rng = np.random.default_rng(n)
    x = dec(rng.standard_normal(n) * 10.0 ** rng.integers(-3, 4, size=n))
    got = metal(x).sum()
    ref = math.fsum(x.tolist())
    assert got.shape == ()
    assert abs(float(val(got)) - ref) <= sum_bound(n, np.abs(x).sum())
    assert stat("gpu:sum") == 1
    # positive data: relative accuracy
    xp = dec(rng.random(n) * rng.choice([1e-3, 1.0, 1e5], size=n))
    got = float(val(metal(xp).sum()))
    ref = math.fsum(xp.tolist())
    assert abs(got - ref) <= max(RTOL, 8 * math.log2(max(n, 2)) * U2) * ref


def test_sum_integers_exact():
    rng = np.random.default_rng(3)
    for n, low, high in (
        (10**6, -1000, 1000),
        (2**20, 0, 2**20),
        (5000, -(2**30), 2**30),
    ):
        x = rng.integers(low, high, size=n).astype(np.float64)
        assert float(val(metal(x).sum())) == math.fsum(x.tolist())


@pytest.mark.parametrize("shape", [(7, 33, 5), (3, 4, 5, 6), (2, 300, 3)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_dims_and_keepdim(shape, keepdim):
    rng = np.random.default_rng(len(shape) + int(keepdim))
    x = dec(rng.standard_normal(shape) * 3.0)
    xm = metal(x)
    nd = len(shape)
    combos = [(d,) for d in range(nd)] + [(-1,), (0, nd - 1), tuple(range(nd)), ()]
    if nd >= 3:
        combos += [(0, 2), (1, -1)]
    for dims in combos:
        d = tuple(range(nd)) if dims == () else tuple(k % nd for k in dims)
        got = xm.sum(dim=dims, keepdim=keepdim) if dims != () else xm.sum()
        ref = fsum_dims(x, d, keepdim and dims != ())
        n = math.prod(shape[k] for k in d)
        abs_sum = np.abs(x).sum(axis=d, keepdims=keepdim and dims != ())
        check(got, ref, rtol=0.0, atol=sum_bound(n, abs_sum))
        check(
            got,
            cpu(x).sum(dim=dims, keepdim=keepdim) if dims != () else cpu(x).sum(),
            rtol=0.0,
            atol=sum_bound(n, abs_sum) + RTOL * np.abs(ref),
        )


def test_sum_dim_none_and_list_forms():
    x = dec(np.random.default_rng(1).standard_normal((4, 6)))
    xm = metal(x)
    check(xm.sum(dim=None), cpu(x).sum(), rtol=1e-14, atol=1e-15)
    check(xm.sum(dim=[]), cpu(x).sum(), rtol=1e-14, atol=1e-15)
    check(xm.sum(dim=[1]), cpu(x).sum(dim=[1]), rtol=1e-14, atol=1e-15)
    check(torch.sum(xm, (0, 1), True), cpu(x).sum((0, 1), True), rtol=1e-14, atol=1e-15)
    with pytest.raises(RuntimeError, match="appears multiple times"):
        xm.sum(dim=(0, 0))
    with pytest.raises(IndexError):
        xm.sum(dim=2)


def test_sum_views_and_noncontiguous():
    rng = np.random.default_rng(2)
    x = dec(rng.standard_normal((6, 7, 8)))
    xm, xc = metal(x), cpu(x)
    views = [
        (lambda t: t.transpose(0, 2), "transpose"),
        (lambda t: t[1:5, ::2, 3:], "slice"),
        (lambda t: t.permute(2, 0, 1), "permute"),
        (lambda t: t[:, 2], "select"),
        (lambda t: t[0, 0].expand(5, 8), "expand"),
    ]
    for fn, _name in views:
        for dims in ((0,), (-1,), None):
            g = fn(xm).sum(dim=dims) if dims is not None else fn(xm).sum()
            r = fn(xc).sum(dim=dims) if dims is not None else fn(xc).sum()
            # cancellation makes a result-relative tolerance meaningless: bound by
            # the kernel's error budget on the sum of magnitudes
            view = fn(xc)
            n = view.shape[dims[0]] if dims is not None else view.numel()
            abs_sum = view.abs().sum(dim=dims) if dims is not None else view.abs().sum()
            check(g, r, rtol=1e-13, atol=sum_bound(n, abs_sum.numpy()))


def test_sum_special_values_and_empty():
    cases = {
        "nan_propagates": ([1.0, NAN, 2.0], NAN),
        "inf": ([1.0, INF, 2.0], INF),
        "neg_inf": ([-INF, 1.0], -INF),
        "inf_minus_inf": ([INF, 3.0, -INF], NAN),
        "empty": (np.zeros(0), 0.0),
        "single": ([-2.5], -2.5),
        "cancel": ([1e10, 1.0, -1e10], 1.0),
        "ones": (np.ones(1000), 1000.0),
        "many_nan": (np.r_[np.ones(5000), NAN, np.ones(5000)], NAN),
    }
    for name, (x, expect) in cases.items():
        got = float(val(metal(x).sum()))
        assert (math.isnan(got) and math.isnan(expect)) or got == expect, name
    # empty lines / empty reduced axis / 0-d
    check(metal(np.zeros((0, 3))).sum(dim=1), cpu(np.zeros((0, 3))).sum(dim=1))
    check(metal(np.zeros((3, 0))).sum(dim=1), cpu(np.zeros((3, 0))).sum(dim=1))
    check(
        metal(np.zeros((3, 0))).sum(dim=0, keepdim=True),
        cpu(np.zeros((3, 0))).sum(dim=0, keepdim=True),
    )
    check(metal(2.5).sum(), cpu(2.5).sum())
    check(metal(2.5).sum(dim=0), cpu(2.5).sum(dim=0))
    check(metal(2.5).sum(dim=-1, keepdim=True), cpu(2.5).sum(dim=-1, keepdim=True))


def test_nansum():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 5000, 3))
    x[rng.random(x.shape) < 0.1] = NAN
    x = dec(x)
    ref = fsum_axis(np.nan_to_num(x, nan=0.0), 1)
    abs_ref = np.nansum(np.abs(x), axis=1)
    check(torch.nansum(metal(x), dim=1), ref, rtol=0.0, atol=sum_bound(5000, abs_ref))
    check(
        torch.nansum(metal(x)),
        math.fsum(np.nan_to_num(x, nan=0.0).ravel().tolist()),
        rtol=0.0,
        atol=sum_bound(x.size, np.nansum(np.abs(x))),
    )
    check(
        torch.nansum(metal(x), dim=(0, 2), keepdim=True),
        cpu(x).nansum(dim=(0, 2), keepdim=True),
        rtol=1e-13,
        atol=sum_bound(12, np.nansum(np.abs(x), axis=(0, 2), keepdims=True)),
    )
    assert float(val(torch.nansum(metal(np.full(700, NAN))))) == 0.0
    assert math.isnan(float(val(torch.nansum(metal([INF, NAN, -INF] + [1.0] * 600)))))
    assert float(val(torch.nansum(metal([NAN, INF, 2.0])))) == INF


# ---------------------------------------------------------------------------
# mean / prod / trace / sum_to_size
# ---------------------------------------------------------------------------
def test_mean(rng):
    x = dec(
        rng.standard_normal((5, 300, 4)) * 10.0 ** rng.integers(-2, 3, size=(5, 300, 4))
    )
    xm, xc = metal(x), cpu(x)
    check(
        xm.mean(),
        xc.mean(),
        rtol=0.0,
        atol=sum_bound(x.size, np.abs(x).sum()) / x.size
        + 1e-14 * abs(float(xc.mean())),
    )
    for dims, keep in [((1,), False), ((0, 2), True), ((-1,), True), (None, False)]:
        g = xm.mean(dim=dims, keepdim=keep) if dims is not None else xm.mean()
        r = xc.mean(dim=dims, keepdim=keep) if dims is not None else xc.mean()
        n = math.prod(x.shape[d] for d in dims) if dims is not None else x.size
        abs_sum = (
            np.abs(x).sum(axis=dims, keepdims=keep)
            if dims is not None
            else np.abs(x).sum()
        )
        check(g, r, rtol=1e-14, atol=sum_bound(n, abs_sum) / n)
    assert math.isnan(float(val(metal(np.zeros(0)).mean())))
    check(metal(np.zeros((2, 0))).mean(dim=1), cpu(np.zeros((2, 0))).mean(dim=1))
    check(metal(4.0).mean(), cpu(4.0).mean())


@pytest.mark.parametrize("n", [1, 7, 64, 300])
def test_prod(n):
    rng = np.random.default_rng(100 + n)
    x = dec(rng.uniform(0.5, 2.0, size=(3, n)) * rng.choice([-1.0, 1.0], size=(3, n)))
    got = val(metal(x).prod(dim=1))
    for row in range(3):
        ref = float(math.prod(Fraction(float(t)) for t in x[row]))
        assert abs(got[row] - ref) <= 5 * n * U2 * abs(ref)
    check(
        metal(x).prod(dim=1, keepdim=True),
        cpu(x).prod(dim=1, keepdim=True),
        rtol=5 * n * U2,
    )
    check(metal(x[:, :4]).prod(), cpu(x[:, :4]).prod(), rtol=60 * U2)
    check(metal(x).prod(dim=0), cpu(x).prod(dim=0), rtol=15 * U2)


def test_prod_special_values():
    cases = {
        "empty": (np.zeros(0), 1.0),
        "zero": ([2.0, 0.0, 3.0], 0.0),
        "nan": ([2.0, NAN], NAN),
        "inf": ([2.0, INF], INF),
        "inf_zero": ([INF, 0.0], NAN),
        "neg": ([-1.0, -1.0, -1.0], -1.0),
        "ints": (np.arange(1.0, 15.0), float(math.factorial(14))),
    }
    for name, (x, expect) in cases.items():
        got = float(val(metal(x).prod()))
        assert (math.isnan(got) and math.isnan(expect)) or got == expect, name


def test_trace_and_sum_to_size(rng):
    a = dec(rng.standard_normal((6, 6)))
    check(torch.trace(metal(a)), torch.trace(cpu(a)), rtol=1e-13, atol=1e-15)
    x = dec(rng.standard_normal((2, 3, 4, 5)))
    for size in [(1, 3, 1, 5), (4, 5), (3, 1, 1), (2, 3, 4, 5), (1,)]:
        check(
            metal(x).sum_to_size(*size),
            cpu(x).sum_to_size(*size),
            rtol=1e-13,
            atol=1e-14,
        )
    # the handler itself (the composite normally decomposes above dispatch)
    got = ops_reduce._sum_to_size(
        aten.sum_to_size.default, (), (metal(x), (1, 3, 1, 5)), {}
    )
    check(got, cpu(x).sum_to_size(1, 3, 1, 5), rtol=1e-13, atol=1e-14)


# ---------------------------------------------------------------------------
# amax / amin / max / min / argmax / argmin
# ---------------------------------------------------------------------------
def _tie_data(rng, shape):
    return rng.integers(-3, 4, size=shape).astype(np.float64) * 0.25


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_max_min_amax_amin_values_and_ties(axis):
    rng = np.random.default_rng(20 + axis)
    for x in (dec(rng.standard_normal((5, 3000, 4))), _tie_data(rng, (5, 3000, 4))):
        xm, xc = metal(x), cpu(x)
        check(torch.amax(xm, dim=axis), torch.amax(xc, dim=axis), rtol=0)
        check(
            torch.amin(xm, dim=axis, keepdim=True),
            torch.amin(xc, dim=axis, keepdim=True),
            rtol=0,
        )
        v, i = xm.max(dim=axis)
        vr, ir = xc.max(dim=axis)
        check(v, vr, rtol=0)
        check(i, ir)
        v, i = xm.min(dim=axis, keepdim=True)
        vr, ir = xc.min(dim=axis, keepdim=True)
        check(v, vr, rtol=0)
        check(i, ir)
        check(xm.argmax(dim=axis), xc.argmax(dim=axis))
        check(xm.argmin(dim=axis, keepdim=True), xc.argmin(dim=axis, keepdim=True))
    x = dec(rng.standard_normal((5, 3000, 4)))
    xm, xc = metal(x), cpu(x)
    check(xm.max(), xc.max(), rtol=0)
    check(xm.min(), xc.min(), rtol=0)
    check(torch.amax(xm, dim=(0, 2)), torch.amax(xc, dim=(0, 2)), rtol=0)
    check(torch.amin(xm), torch.amin(xc), rtol=0)
    check(xm.argmax(), xc.argmax())
    check(xm.argmin(), xc.argmin())
    check(xm.argmax(keepdim=True), xc.argmax(keepdim=True))


def test_max_min_nan_inf_and_signed_zero():
    rng = np.random.default_rng(21)
    x = dec(rng.standard_normal((6, 2000)))
    x[0, 1234] = NAN
    x[1, 7] = INF
    x[2, 9] = -INF
    x[3, :] = NAN
    x[4, 5] = NAN
    x[4, 1999] = INF
    xm, xc = metal(x), cpu(x)
    for fn in (torch.amax, torch.amin):
        check(fn(xm, dim=1), fn(xc, dim=1), rtol=0)
    for fn in ("max", "min", "argmax", "argmin"):
        g = getattr(xm, fn)(dim=1)
        r = getattr(xc, fn)(dim=1)
        if isinstance(g, tuple):
            check(g[0], r[0], rtol=0)
            check(g[1], r[1])
        else:
            check(g, r)
    assert math.isnan(float(val(xm.max())))
    assert int(xm.argmax()) == int(xc.argmax())
    # +-0 ties are value-equal, first index wins
    for data in ([-0.0, 0.0, -1.0], [0.0, -0.0, 1.0]):
        for fn in ("argmax", "argmin"):
            assert int(getattr(metal(data), fn)()) == int(getattr(cpu(data), fn)())
    assert float(val(metal([-0.0, 0.0, -1.0]).max())) == 0.0
    # ties: first index (torch semantics)
    check(metal([1.0, 3.0, 3.0, 2.0]).argmax(), cpu([1.0, 3.0, 3.0, 2.0]).argmax())
    check(metal([[3.0, 1.0, 1.0]]).argmin(dim=1), cpu([[3.0, 1.0, 1.0]]).argmin(dim=1))
    check(
        metal(np.zeros((5, 3000, 4))).argmax(dim=1),
        cpu(np.zeros((5, 3000, 4))).argmax(dim=1),
    )


def test_argmax_long_line_multipass_and_nan():
    rng = np.random.default_rng(32)
    x = rng.random(2**20)
    x[777_777] = 2.0
    x[999_999] = 2.0  # later tie must lose
    assert int(metal(x).argmax()) == 777_777
    assert int(metal(-x).argmin()) == 777_777
    v, i = metal(x.reshape(1, -1)).max(dim=1)
    assert int(i) == 777_777 and float(val(v)[0]) == 2.0
    x[500_000] = NAN
    x[900_000] = NAN
    assert int(metal(x).argmax()) == 500_000 == int(cpu(x).argmax())
    assert int(metal(x).argmin()) == 500_000


def test_extremum_errors_and_0d():
    with pytest.raises(RuntimeError, match="numel\\(\\) == 0"):
        metal(np.zeros(0)).max()
    with pytest.raises(RuntimeError, match="non-zero size"):
        torch.amax(metal(np.zeros((3, 0))), dim=1)
    with pytest.raises(RuntimeError, match="non-zero size"):
        metal(np.zeros((3, 0))).max(dim=1)
    with pytest.raises(RuntimeError):
        metal(np.zeros(0)).argmax()
    check(
        torch.amax(metal(np.zeros((0, 3))), dim=1),
        torch.amax(cpu(np.zeros((0, 3))), dim=1),
    )
    check(metal(2.5).max(), cpu(2.5).max(), rtol=0)
    v, i = metal(2.5).max(dim=0)
    check(v, cpu(2.5).max(dim=0)[0], rtol=0)
    check(i, cpu(2.5).max(dim=0)[1])
    check(metal(2.5).argmax(), cpu(2.5).argmax())


# ---------------------------------------------------------------------------
# cumsum / cumprod
# ---------------------------------------------------------------------------
def test_cumsum_long_line_prefix_relative():
    rng = np.random.default_rng(40)
    x = dec(rng.random(100_000) * rng.choice([1e-2, 1.0, 1e3], size=100_000))
    got = val(metal(x).cumsum(dim=0))
    ref = exact_cumsum_1d(x)
    err = np.abs(got - ref) / np.abs(ref) / U2
    assert err.max() <= 8, f"cumsum max prefix rel err {err.max():.2f} u^2"
    assert stat("gpu:cumsum") == 1


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_cumsum_axes_integers_and_views(axis):
    rng = np.random.default_rng(42 + axis)
    x = rng.integers(-1000, 1000, size=(4, 50, 3)).astype(np.float64)
    check(metal(x).cumsum(dim=axis), cpu(x).cumsum(dim=axis), rtol=0)
    x = dec(rng.standard_normal((4, 50, 3)))
    got = val(metal(x).cumsum(dim=axis))
    ref = np.apply_along_axis(exact_cumsum_1d, axis, x)
    running_abs = np.cumsum(np.abs(x), axis=axis)
    assert np.all(np.abs(got - ref) <= 8 * U2 * running_abs)
    # non-contiguous input (transposed view) and the dtype keyword
    check(
        metal(x).transpose(0, 1).cumsum(dim=axis),
        cpu(x).transpose(0, 1).cumsum(dim=axis),
        rtol=1e-13,
        atol=8 * U2 * np.cumsum(np.abs(np.swapaxes(x, 0, 1)), axis=axis),
    )
    assert torch.cumsum(metal(x), dim=axis, dtype=torch.float64).dtype == torch.float64


def test_cumsum_special_values_and_0d():
    got = val(metal([1.0, INF, 2.0, -INF, 5.0]).cumsum(dim=0))
    assert np.array_equal(got, [1.0, INF, INF, NAN, NAN], equal_nan=True)
    got = val(metal([1.0, NAN, 2.0]).cumsum(dim=0))
    assert np.array_equal(got, [1.0, NAN, NAN], equal_nan=True)
    assert val(metal(np.zeros(0)).cumsum(dim=0)).shape == (0,)
    check(metal(np.zeros((3, 0))).cumsum(dim=1), cpu(np.zeros((3, 0))).cumsum(dim=1))
    check(metal(2.5).cumsum(dim=0), cpu(2.5).cumsum(dim=0), rtol=0)
    big = dec(np.array([1e12, 1.0, -1e12, 2.0, 1e-3, -3.0]))
    got = val(metal(big).cumsum(dim=0))
    ref = exact_cumsum_1d(big)
    running_max = np.maximum.accumulate(np.abs(big))
    assert np.all(
        np.abs(got - ref) <= 4 * U2 * np.maximum(np.abs(ref), 2.0**-24 * running_max)
    )


def test_cumprod_is_counted_cpu_fallback(rng):
    x = dec(rng.uniform(0.5, 1.5, (3, 20)))
    check(metal(x).cumprod(dim=1), cpu(x).cumprod(dim=1), rtol=1e-14)
    assert stat("cpu_fallback:cumprod") == 1


# ---------------------------------------------------------------------------
# all / any / count_nonzero
# ---------------------------------------------------------------------------
def test_all_any_count_nonzero(rng):
    x = rng.standard_normal((4, 300, 3))
    x[rng.random(x.shape) < 0.3] = 0.0
    x[0, 5, 1] = NAN
    x[1, :, 2] = -0.0
    x[2] = 1.0
    xm, xc = metal(x), cpu(x)
    for fn in (torch.all, torch.any):
        check(fn(xm), fn(xc))
        for dims, keep in [
            (0, False),
            (1, True),
            (-1, False),
            ((0, 2), True),
            ((1, 2), False),
            (None, False),
        ]:
            g = fn(xm, dim=dims, keepdim=keep) if dims is not None else fn(xm)
            r = fn(xc, dim=dims, keepdim=keep) if dims is not None else fn(xc)
            check(g, r)
    check(torch.count_nonzero(xm), torch.count_nonzero(xc))
    check(torch.count_nonzero(xm, dim=1), torch.count_nonzero(xc, dim=1))
    check(torch.count_nonzero(xm, dim=(0, 2)), torch.count_nonzero(xc, dim=(0, 2)))
    assert bool(metal([0.0, -0.0]).any()) is False
    assert bool(metal([0.0, NAN]).any()) is True
    assert bool(metal(np.zeros(0)).all()) is True
    assert bool(metal(np.zeros(0)).any()) is False
    # The encoder flushes denormal float32 words (GPU FTZ, encode.py docstring):
    # 1e-45 is carried as an exact zero, 1e-30 as a normal hi word with lo = 0.
    assert ops_reduce.nonzero_mask(metal([1e-30, 0.0, -0.0, 3.0])).cpu().tolist() == [
        True,
        False,
        False,
        True,
    ]
    assert not bool(metal([1e-45]).any())
    assert float(metal([1e-45]).to_numpy()[0]) == 0.0


# ---------------------------------------------------------------------------
# var / std
# ---------------------------------------------------------------------------
def test_var_std_against_numpy_ddof(rng):
    x = dec(
        rng.standard_normal((6, 500, 4)) * 10.0 ** rng.integers(-2, 3, size=(6, 500, 4))
        + 3.0
    )
    xm = metal(x)
    for dims, keep in [
        ((1,), False),
        ((0, 2), True),
        ((-1,), True),
        ((0, 1, 2), False),
    ]:
        n = math.prod(x.shape[d] for d in dims)
        for ddof in (0, 1, 2):
            ref = np.var(x, axis=dims, ddof=ddof, keepdims=keep)
            tol = 40 * math.log2(n) * U2 * np.abs(ref) + 1e-30
            check(
                xm.var(dim=dims, correction=ddof, keepdim=keep), ref, rtol=0.0, atol=tol
            )
            check(
                xm.std(dim=dims, correction=ddof, keepdim=keep),
                np.sqrt(ref),
                rtol=0.0,
                atol=40 * math.log2(n) * U2 * np.sqrt(ref) + 1e-30,
            )
    check(xm.var(), np.var(x, ddof=1), rtol=4e-13)
    check(xm.std(), np.std(x, ddof=1), rtol=4e-13)
    check(xm.var(unbiased=False), np.var(x, ddof=0), rtol=4e-13)
    check(xm.std(dim=1, unbiased=False), np.std(x, axis=1, ddof=0), rtol=4e-13)
    check(
        xm.var(dim=1, unbiased=True, keepdim=True),
        np.var(x, axis=1, ddof=1, keepdims=True),
        rtol=4e-13,
    )
    check(
        torch.var(xm, dim=(0, 2), correction=0.5),
        cpu(x).var(dim=(0, 2), correction=0.5),
        rtol=4e-13,
    )
    v, m = torch.var_mean(xm, dim=1, correction=1)
    check(v, np.var(x, axis=1, ddof=1), rtol=4e-13)
    check(m, np.mean(x, axis=1), rtol=4e-13)
    s, m = torch.std_mean(xm, dim=(0, 2))
    check(s, np.std(x, axis=(0, 2), ddof=1), rtol=4e-13)
    check(m, np.mean(x, axis=(0, 2)), rtol=4e-13)
    assert all(k.startswith("gpu:") for k in mt.stats()), mt.stats()


def test_var_degenerate_cases():
    assert math.isnan(float(val(metal([2.0]).var())))
    assert float(val(metal([2.0]).var(correction=0))) == 0.0
    assert float(val(metal([1.0, 2.0]).var(correction=3))) == INF
    assert math.isnan(float(val(metal(np.zeros(0)).var())))
    check(metal(np.zeros((2, 0))).var(dim=1), cpu(np.zeros((2, 0))).var(dim=1))
    assert math.isnan(float(val(metal(2.5).var())))
    check(metal([1.0, NAN, 2.0]).var(), cpu([1.0, NAN, 2.0]).var())
    check(
        metal([1e8, 1e8 + 1, 1e8 + 2]).var(),
        cpu([1e8, 1e8 + 1, 1e8 + 2]).var(),
        rtol=1e-13,
    )


# ---------------------------------------------------------------------------
# norms / logsumexp
# ---------------------------------------------------------------------------
def test_vector_norm_orders(rng):
    x = dec(
        rng.standard_normal((5, 400, 3)) * 10.0 ** rng.integers(-2, 3, size=(5, 400, 3))
    )
    x[1, 7, 0] = 0.0
    xm, xc = metal(x), cpu(x)
    for order in (2, 1, INF, -INF, 0, 3, 0.5, -1, 4.5):
        for dims, keep in [(None, False), (1, False), ((0, 2), True), (-1, True)]:
            g = torch.linalg.vector_norm(xm, ord=order, dim=dims, keepdim=keep)
            r = torch.linalg.vector_norm(xc, ord=order, dim=dims, keepdim=keep)
            check(g, r, rtol=2e-13 if order not in (INF, -INF, 0) else 0.0)
    check(xm.norm(), xc.norm(), rtol=2e-13)
    check(xm.norm(p=1, dim=1), xc.norm(p=1, dim=1), rtol=2e-13)
    check(
        xm.norm(p=INF, dim=(0, 2), keepdim=True),
        xc.norm(p=INF, dim=(0, 2), keepdim=True),
        rtol=0,
    )
    check(
        torch.norm(xm, p="fro", dim=(1, 2)),
        torch.norm(xc, p="fro", dim=(1, 2)),
        rtol=2e-13,
    )
    check(torch.norm(xm, p=3), torch.norm(xc, p=3), rtol=2e-13)
    check(torch.linalg.norm(xm), torch.linalg.norm(xc), rtol=2e-13)
    check(
        torch.linalg.norm(xm[0], ord="fro"),
        torch.linalg.norm(xc[0], ord="fro"),
        rtol=2e-13,
    )
    assert all(k.startswith("gpu:") for k in mt.stats()), mt.stats()
    with pytest.raises(RuntimeError, match="empty tensor"):
        torch.linalg.vector_norm(metal(np.zeros(0)), ord=INF)
    check(
        torch.linalg.vector_norm(metal(np.zeros(0))),
        torch.linalg.vector_norm(cpu(np.zeros(0))),
    )
    check(
        torch.linalg.vector_norm(metal([1.0, NAN])),
        torch.linalg.vector_norm(cpu([1.0, NAN])),
    )
    check(
        torch.linalg.vector_norm(metal([1.0, NAN]), ord=INF),
        torch.linalg.vector_norm(cpu([1.0, NAN]), ord=INF),
    )


def test_logsumexp(rng):
    x = dec(rng.standard_normal((4, 300)) * 50.0)
    x[1, 3] = 1000.0
    x[2, :] = -INF
    x[3, 10] = INF
    xm, xc = metal(x), cpu(x)
    check(torch.logsumexp(xm, dim=1), torch.logsumexp(xc, dim=1), rtol=1e-13)
    check(
        torch.logsumexp(xm, dim=(0, 1), keepdim=True),
        torch.logsumexp(xc, dim=(0, 1), keepdim=True),
        rtol=1e-13,
    )
    check(torch.logsumexp(xm, dim=0), torch.logsumexp(xc, dim=0), rtol=1e-13)
    check(
        torch.logsumexp(metal(np.zeros((3, 0))), dim=1),
        torch.logsumexp(cpu(np.zeros((3, 0))), dim=1),
    )
    check(
        torch.logsumexp(metal([1000.0, 1000.0]), dim=0),
        torch.logsumexp(cpu([1000.0, 1000.0]), dim=0),
        rtol=1e-13,
    )
    assert all(k.startswith("gpu:") for k in mt.stats()), mt.stats()


# ---------------------------------------------------------------------------
# dtype / out= / stats
# ---------------------------------------------------------------------------
def test_dtype_kwarg_out_variants_and_counters(rng):
    x = dec(rng.standard_normal((3, 4)))
    xm, xc = metal(x), cpu(x)
    s32 = xm.sum(dtype=torch.float32)
    assert (
        s32.dtype == torch.float32
        and s32.device.type == "mps"
        and not isinstance(s32, MetalFloat64)
    )
    assert abs(float(s32.cpu()) - float(xc.sum())) <= 1e-6 * abs(float(xc.sum()))
    out = metal(np.zeros(3))
    r = torch.sum(xm, dim=1, out=out)
    assert r is out
    check(out, xc.sum(dim=1), rtol=1e-13, atol=1e-15)
    vo, io = metal(np.zeros(4)), torch.zeros(4, dtype=torch.int64, device="mps")
    torch.max(xm, dim=0, out=(vo, io))
    check(vo, xc.max(dim=0)[0], rtol=0)
    check(io, xc.max(dim=0)[1])
    mo = metal(np.zeros((3, 3)))
    torch.mm(xm, metal(x.T.copy()), out=mo)
    check(mo, xc @ xc.T, rtol=1e-13, atol=1e-15)
    stats = mt.stats()
    assert (
        stats["gpu:sum"] >= 2 and stats["gpu:argmax"] == 1 and stats["gpu:matmul"] == 1
    )


def test_out_variants_with_schema_named_destinations(rng):
    """``out=`` tensors are found by their schema names (``max``, ``max_values``)."""
    x = dec(rng.standard_normal((4, 300)))
    xm, xc = metal(x), cpu(x)

    def mo(shape):
        return metal(np.zeros(shape))

    o = mo(())
    assert torch.max(xm, out=o) is o
    check(o, xc.max(), rtol=0)
    o = mo(())
    torch.min(xm, out=o)
    check(o, xc.min(), rtol=0)
    vo, io = mo((300,)), torch.zeros(300, dtype=torch.int64, device="mps")
    r = torch.min(xm, dim=0, out=(vo, io))
    assert r[0] is vo and r[1] is io
    check(vo, xc.min(dim=0)[0], rtol=0)
    check(io, xc.min(dim=0)[1])
    bo = torch.empty(4, dtype=torch.bool, device="mps")
    assert torch.all(xm > -10, dim=1, out=bo) is bo
    check(bo, torch.all(xc > -10, dim=1))
    bo = torch.empty((), dtype=torch.bool, device="mps")
    torch.any(xm > 3, out=bo)
    check(bo, torch.any(xc > 3))
    io = torch.empty(4, dtype=torch.int64, device="mps")
    torch.argmin(xm, dim=1, out=io)
    check(io, xc.argmin(dim=1))
    for fn in (torch.var, torch.std):
        o = mo((4,))
        fn(xm, dim=1, out=o)
        check(o, fn(xc, dim=1), rtol=4e-13)
        o = mo((300,))
        fn(xm, dim=(0,), correction=2, keepdim=False, out=o)
        check(o, fn(xc, dim=(0,), correction=2), rtol=4e-13)
        o = mo((4, 1))
        fn(xm, dim=1, unbiased=False, keepdim=True, out=o)
        check(o, fn(xc, dim=1, unbiased=False, keepdim=True), rtol=4e-13)
        check(fn(xm, unbiased=False), fn(xc, unbiased=False), rtol=4e-13)
        check(
            fn(xm, dim=None, correction=0), fn(xc, dim=None, correction=0), rtol=4e-13
        )
    for fn in (torch.var_mean, torch.std_mean):
        for kw in ({}, {"dim": 1}, {"dim": 1, "correction": 0}, {"unbiased": False}):
            g = fn(xm, **kw)
            r = fn(xc, **kw)
            check(g[0], r[0], rtol=4e-13)
            check(g[1], r[1], rtol=4e-13)
    o = mo((4,))
    torch.norm(xm, p=3, dim=1, out=o)
    check(o, torch.norm(xc, p=3, dim=1), rtol=2e-13)
    o = mo(())
    torch.norm(xm, p=1, out=o)
    check(o, torch.norm(xc, p=1), rtol=2e-13)
    o = mo((4,))
    torch.mean(xm, dim=1, dtype=torch.float64, out=o)
    check(o, xc.mean(dim=1), rtol=1e-13, atol=1e-15)
    o = mo((4, 300))
    torch.cumsum(xm, dim=1, out=o)
    check(o, xc.cumsum(dim=1), rtol=1e-13, atol=8 * U2 * np.cumsum(np.abs(x), axis=1))
    o = mo((4,))
    torch.linalg.vector_norm(xm, dim=1, out=o)
    check(o, torch.linalg.vector_norm(xc, dim=1), rtol=2e-13)
    o = mo((4,))
    torch.nansum(xm, dim=1, out=o)
    check(o, xc.nansum(dim=1), rtol=1e-13, atol=sum_bound(300, np.abs(x).sum(axis=1)))
    assert all(k.startswith("gpu:") for k in mt.stats()), mt.stats()


def test_matrix_norms_through_composites(rng):
    a = dec(rng.standard_normal((6, 6)))
    am, ac = metal(a), cpu(a)
    for order in (1, INF, -1, -INF, "fro"):
        check(
            torch.linalg.norm(am, ord=order),
            torch.linalg.norm(ac, ord=order),
            rtol=2e-13,
        )
        check(
            torch.linalg.matrix_norm(am, ord=order),
            torch.linalg.matrix_norm(ac, ord=order),
            rtol=2e-13,
        )
    assert all(k.startswith("gpu:") for k in mt.stats()), mt.stats()
    for order in (2, -2, "nuc"):
        before = stat("cpu_fallback:linalg_svd")
        check(
            torch.linalg.norm(am, ord=order),
            torch.linalg.norm(ac, ord=order),
            rtol=1e-13,
        )
        assert stat("cpu_fallback:linalg_svd") == before + 1
    check(torch.linalg.cond(am), torch.linalg.cond(ac), rtol=1e-12)
    check(
        torch.linalg.matrix_power(am, 3), torch.linalg.matrix_power(ac, 3), rtol=1e-12
    )


def test_host_resident_small_tensors_agree(rng, monkeypatch):
    """With the nucleus' host residency on, small tensors take the CPU path."""
    if not hasattr(mt, "HOST_THRESHOLD"):
        pytest.skip("nucleus has no host residency")
    monkeypatch.setattr(mt, "HOST_THRESHOLD", 256)
    x = dec(rng.standard_normal((4, 5)))
    xm, xc = metal(x), cpu(x)
    assert xm.is_host_resident
    check(xm.sum(dim=1), xc.sum(dim=1), rtol=1e-13, atol=1e-15)
    assert np.array_equal(xm.max(dim=0)[1].cpu().numpy(), xc.max(dim=0)[1].numpy())
    check(xm @ metal(x.T.copy()), xc @ xc.T, rtol=1e-13, atol=1e-15)
    assert float(xm.var()) == pytest.approx(float(xc.var()), rel=1e-13)
    # GPU-resident bulk operands still reach the handlers, mixed with host ones
    big = dec(rng.standard_normal((40, 40)))
    check(metal(big).sum(dim=0), cpu(big).sum(dim=0), rtol=1e-13, atol=1e-14)
    small = dec(rng.standard_normal((40, 3)))
    ref = cpu(big) @ cpu(small)
    check(
        metal(big) @ metal(small), ref, rtol=0.0, atol=mm_tol(big, small, ref.numpy())
    )
    assert stat("gpu:matmul") == 1 and stat("gpu:sum") == 1


# ---------------------------------------------------------------------------
# matmul family
# ---------------------------------------------------------------------------
def mm_tol(a: np.ndarray, b: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``1e-12 |ref|`` plus the kernel's accumulation budget ``16 u^2 sum|a||b|``."""
    return 1e-12 * np.abs(ref) + 16 * U2 * (np.abs(a) @ np.abs(b))


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1),
        (8, 37, 5),
        (16, 256, 16),
        (3, 4096, 3),
        (2, 40000, 2),
        (4, 40000, 4),
        (1, 64, 9),
        (7, 64, 1),
    ],
    ids=lambda s: "x".join(map(str, s)),
)
def test_mm_against_cpu(shape):
    m, k, n = shape
    rng = np.random.default_rng(sum(shape))
    a = dec(rng.standard_normal((m, k)) * 10.0 ** rng.integers(-2, 3, size=(m, k)))
    b = dec(rng.standard_normal((k, n)) * 10.0 ** rng.integers(-2, 3, size=(k, n)))
    ref = cpu(a) @ cpu(b)
    tol = mm_tol(a, b, ref.numpy())
    check(metal(a) @ metal(b), ref, rtol=0.0, atol=tol)
    check(torch.mm(metal(a), metal(b)), ref, rtol=0.0, atol=tol)
    check(torch.matmul(metal(a), metal(b)), ref, rtol=0.0, atol=tol)
    assert stat("gpu:matmul") == 3
    # positive data: plain relative accuracy (1e-12 for K <= 256, the kernel's
    # ~2u^2 * K/|ref| for longer K is covered by the scaled bound above)
    a = dec(rng.uniform(0.5, 1.5, size=(m, k)))
    b = dec(rng.uniform(0.5, 1.5, size=(k, n)))
    check(
        metal(a) @ metal(b),
        cpu(a) @ cpu(b),
        rtol=1e-12 if k <= 256 else 1e-12 * max(1.0, k / 256),
    )


def test_mm_exact_special_and_mixed(rng):
    a = rng.integers(-1000, 1000, size=(6, 1000)).astype(np.float64)
    b = rng.integers(-1000, 1000, size=(1000, 4)).astype(np.float64)
    check(metal(a) @ metal(b), cpu(a) @ cpu(b), rtol=0)
    x = dec(rng.standard_normal((9, 9)))
    check(metal(x) @ metal(np.eye(9)), cpu(x), rtol=0)
    check(metal(np.eye(9)) @ metal(x), cpu(x), rtol=0)
    a = np.array([[1.0, 2.0], [NAN, 1.0], [INF, 0.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0, -INF], [0.0, 1.0, 1.0]])
    check(metal(a) @ metal(b), cpu(a) @ cpu(b), rtol=0)
    check(
        metal(np.zeros((2, 0))) @ metal(np.zeros((0, 3))),
        cpu(np.zeros((2, 0))) @ cpu(np.zeros((0, 3))),
    )
    check(
        metal(np.zeros((0, 4))) @ metal(np.ones((4, 3))),
        cpu(np.zeros((0, 4))) @ cpu(np.ones((4, 3))),
    )
    # a non-contiguous operand (transposed view) and a float32 MPS operand
    check(
        metal(x).t() @ metal(x),
        cpu(x).t() @ cpu(x),
        rtol=0.0,
        atol=mm_tol(x.T, x, (x.T @ x)),
    )
    f32 = torch.ones(9, 2, device="mps")
    got = metal(x) @ f32
    assert isinstance(got, MetalFloat64)
    check(
        got,
        cpu(x) @ torch.ones(9, 2, dtype=torch.float64),
        rtol=0.0,
        atol=mm_tol(x, np.ones((9, 2)), x @ np.ones((9, 2))),
    )
    with pytest.raises(RuntimeError, match="cannot be multiplied"):
        metal(np.zeros((2, 3))) @ metal(np.zeros((2, 3)))


def test_bmm_addmm_baddbmm_addbmm_mv(rng):
    a = dec(rng.standard_normal((3, 5, 300)))
    b = dec(rng.standard_normal((3, 300, 4)))
    am, bm, ac, bc = metal(a), metal(b), cpu(a), cpu(b)
    ref = torch.bmm(ac, bc)
    tol = np.stack([mm_tol(a[i], b[i], ref[i].numpy()) for i in range(3)])
    check(torch.bmm(am, bm), ref, rtol=0.0, atol=tol)
    bt = metal(np.ascontiguousarray(b.transpose(0, 2, 1))).transpose(
        1, 2
    )  # non-contiguous view
    check(torch.bmm(am, bt), ref, rtol=0.0, atol=tol)
    bias3 = dec(rng.standard_normal((3, 5, 4)))
    bias2 = dec(rng.standard_normal((5, 4)))
    row = dec(rng.standard_normal(4))
    for beta, alpha in [(1, 1), (0.5, 2.0), (0, 1), (-1.5, 0.25)]:
        r = torch.baddbmm(cpu(bias3), ac, bc, beta=beta, alpha=alpha)
        check(
            torch.baddbmm(metal(bias3), am, bm, beta=beta, alpha=alpha),
            r,
            rtol=1e-13,
            atol=abs(alpha) * tol + 3e-15 * np.abs(r.numpy()),
        )
        r = torch.addbmm(cpu(bias2), ac, bc, beta=beta, alpha=alpha)
        check(
            torch.addbmm(metal(bias2), am, bm, beta=beta, alpha=alpha),
            r,
            rtol=1e-13,
            atol=abs(alpha) * tol.sum(axis=0) * 2 + 3e-15 * np.abs(r.numpy()),
        )
        r = torch.addmm(cpu(bias2), ac[0], bc[0], beta=beta, alpha=alpha)
        check(
            torch.addmm(metal(bias2), am[0], bm[0], beta=beta, alpha=alpha),
            r,
            rtol=1e-13,
            atol=abs(alpha) * tol[0] + 3e-15 * np.abs(r.numpy()),
        )
        r = torch.addmm(cpu(row), ac[0], bc[0], beta=beta, alpha=alpha)
        check(
            torch.addmm(metal(row), am[0], bm[0], beta=beta, alpha=alpha),
            r,
            rtol=1e-13,
            atol=abs(alpha) * tol[0] + 3e-15 * np.abs(r.numpy()),
        )
    # beta == 0 ignores the bias entirely (NaN bias)
    nanb = np.full((5, 4), NAN)
    check(
        torch.addmm(metal(nanb), am[0], bm[0], beta=0),
        torch.addmm(cpu(nanb), ac[0], bc[0], beta=0),
        rtol=0.0,
        atol=tol[0],
    )
    v = dec(rng.standard_normal(300))
    check(
        torch.mv(am[1], metal(v)),
        torch.mv(ac[1], cpu(v)),
        rtol=0.0,
        atol=mm_tol(a[1], v[:, None], (a[1] @ v)[:, None])[:, 0],
    )
    check(
        am[1] @ metal(v),
        ac[1] @ cpu(v),
        rtol=0.0,
        atol=mm_tol(a[1], v[:, None], (a[1] @ v)[:, None])[:, 0],
    )
    w = dec(rng.standard_normal(5))
    check(
        metal(w) @ am[1],
        cpu(w) @ ac[1],
        rtol=0.0,
        atol=mm_tol(w[None, :], a[1], (w @ a[1])[None, :])[0],
    )


@pytest.mark.parametrize("n", [0, 1, 1000, 4097, 2**20])
def test_dot_vdot(n):
    rng = np.random.default_rng(n)
    x = dec(rng.standard_normal(n))
    y = dec(rng.standard_normal(n) * 10.0 ** rng.integers(-2, 3, size=n))
    ref = math.fsum((x * y).tolist())
    scale = float(np.abs(x) @ np.abs(y))
    got = torch.dot(metal(x), metal(y))
    assert got.shape == ()
    assert (
        abs(float(val(got)) - ref)
        <= 8 * (math.log2(max(n, 2)) + 1) * U2 * scale + 1e-300
    )
    got = torch.vdot(metal(x), metal(y))
    assert (
        abs(float(val(got)) - ref)
        <= 8 * (math.log2(max(n, 2)) + 1) * U2 * scale + 1e-300
    )
    got = metal(x) @ metal(y)
    assert (
        abs(float(val(got)) - ref)
        <= 8 * (math.log2(max(n, 2)) + 1) * U2 * scale + 1e-300
    )
    with pytest.raises(RuntimeError):
        torch.dot(metal(np.zeros(3)), metal(np.zeros(4)))


def test_outer_and_batched_matmul_broadcasting(rng):
    x = dec(rng.standard_normal(6))
    y = dec(rng.standard_normal(5))
    check(torch.outer(metal(x), metal(y)), torch.outer(cpu(x), cpu(y)), rtol=8 * U2)
    check(torch.ger(metal(x), metal(y)), torch.ger(cpu(x), cpu(y)), rtol=8 * U2)
    # the handler directly (torch decomposes outer above dispatch)
    check(
        ops_linalg._outer(aten.outer.default, (), (metal(x), metal(y)), {}),
        torch.outer(cpu(x), cpu(y)),
        rtol=8 * U2,
    )
    a = dec(rng.standard_normal((2, 1, 4, 3)))
    b = dec(rng.standard_normal((3, 3, 5)))
    ref = torch.matmul(cpu(a), cpu(b))
    check(torch.matmul(metal(a), metal(b)), ref, rtol=1e-13, atol=1e-14)
    check(metal(a) @ metal(b), ref, rtol=1e-13, atol=1e-14)
    v = dec(rng.standard_normal(3))
    check(
        torch.matmul(metal(a), metal(v)),
        torch.matmul(cpu(a), cpu(v)),
        rtol=1e-13,
        atol=1e-14,
    )
    w = dec(rng.standard_normal(4))
    check(
        torch.matmul(metal(w), metal(a)),
        torch.matmul(cpu(w), cpu(a)),
        rtol=1e-13,
        atol=1e-14,
    )
    # the matmul handler itself on every rank combination
    for lhs, rhs in [
        (a, b),
        (a, v),
        (w, a),
        (v, v),
        (a[0, 0], b[0]),
        (a[0, 0], v),
        (w, a[0, 0]),
    ]:
        got = ops_linalg._matmul(aten.matmul.default, (), (metal(lhs), metal(rhs)), {})
        check(got, torch.matmul(cpu(lhs), cpu(rhs)), rtol=1e-13, atol=1e-14)
    with pytest.raises(RuntimeError):
        ops_linalg._matmul(aten.matmul.default, (), (metal(a), metal(w)), {})


def test_cross(rng):
    a = dec(rng.standard_normal((7, 3)))
    b = dec(rng.standard_normal((7, 3)))
    check(
        torch.linalg.cross(metal(a), metal(b)), np.cross(a, b), rtol=1e-13, atol=1e-14
    )
    check(
        torch.cross(metal(a), metal(b), dim=1), np.cross(a, b), rtol=1e-13, atol=1e-14
    )
    check(
        torch.cross(metal(a), metal(b), dim=-1),
        torch.cross(cpu(a), cpu(b), dim=-1),
        rtol=1e-13,
        atol=1e-14,
    )
    at, bt = np.ascontiguousarray(a.T), np.ascontiguousarray(b.T)
    check(
        torch.linalg.cross(metal(at), metal(bt), dim=0),
        np.cross(at, bt, axis=0),
        rtol=1e-13,
        atol=1e-14,
    )
    check(
        torch.linalg.cross(metal(a).t(), metal(b).t(), dim=0),
        np.cross(at, bt, axis=0),
        rtol=1e-13,
        atol=1e-14,
    )
    unit = np.array([[1.0, 0.0, 0.0]])
    check(
        torch.linalg.cross(metal(a), metal(unit)),
        torch.linalg.cross(cpu(a), cpu(unit)),
        rtol=0,
    )
    c = dec(rng.standard_normal((2, 3, 4)))
    d = dec(rng.standard_normal((2, 3, 4)))
    check(
        torch.linalg.cross(metal(c), metal(d), dim=1),
        np.cross(c, d, axis=1),
        rtol=1e-13,
        atol=1e-14,
    )
    check(
        torch.cross(metal(c), metal(d), dim=1),
        np.cross(c, d, axis=1),
        rtol=1e-13,
        atol=1e-14,
    )
    with pytest.raises(RuntimeError):
        torch.linalg.cross(metal(np.zeros((2, 4))), metal(np.zeros((2, 4))))
    assert all(k.startswith("gpu:") for k in mt.stats()), mt.stats()


# ---------------------------------------------------------------------------
# CPU fallbacks
# ---------------------------------------------------------------------------
def test_linalg_fallbacks_values_and_counters(rng):
    a = dec(rng.standard_normal((5, 5)))
    s = dec(a @ a.T + 5 * np.eye(5))
    rhs = dec(rng.standard_normal((5, 2)))
    am, sm, rm = metal(a), metal(s), metal(rhs)
    ac, sc, rc = cpu(a), cpu(s), cpu(rhs)
    tol = 2e-14
    cases = [
        ("linalg_inv", lambda t: torch.linalg.inv(t), am, ac),
        ("linalg_inv", lambda t: torch.inverse(t), am, ac),
        ("linalg_det", lambda t: torch.linalg.det(t), am, ac),
        ("linalg_det", lambda t: torch.det(t), am, ac),
        ("linalg_slogdet", lambda t: torch.linalg.slogdet(t)[1], am, ac),
        (
            "linalg_solve",
            lambda t: torch.linalg.solve(t, rm if isinstance(t, MetalFloat64) else rc),
            am,
            ac,
        ),
        ("linalg_eigh", lambda t: torch.linalg.eigh(t)[0], sm, sc),
        ("linalg_eigh", lambda t: torch.linalg.eigvalsh(t), sm, sc),
        ("linalg_svd", lambda t: torch.linalg.svd(t)[1], am, ac),
        ("linalg_svd", lambda t: torch.linalg.svdvals(t), am, ac),
        ("linalg_svd", lambda t: torch.linalg.matrix_rank(t).to(torch.int64), am, ac),
        ("linalg_qr", lambda t: torch.linalg.qr(t)[1], am, ac),
        ("linalg_cholesky", lambda t: torch.linalg.cholesky(t), sm, sc),
        (
            "linalg_lstsq",
            lambda t: (
                torch.linalg.lstsq(
                    t, rm if isinstance(t, MetalFloat64) else rc
                ).solution
            ),
            am,
            ac,
        ),
        ("linalg_pinv", lambda t: torch.linalg.pinv(t), am, ac),
        ("linalg_lu", lambda t: torch.linalg.lu(t)[2], am, ac),
        ("linalg_lu_factor", lambda t: torch.linalg.lu_factor(t)[0], am, ac),
        ("linalg_lu_factor", lambda t: torch.linalg.lu_factor(t)[1], am, ac),
        (
            "triangular_solve",
            lambda t: torch.triangular_solve(
                rm if isinstance(t, MetalFloat64) else rc, t
            )[0],
            sm,
            sc,
        ),
        (
            "linalg_solve_triangular",
            lambda t: torch.linalg.solve_triangular(
                t, rm if isinstance(t, MetalFloat64) else rc, upper=True
            ),
            sm,
            sc,
        ),
        ("linalg_matrix_exp", lambda t: torch.linalg.matrix_exp(t * 0.1), am, ac),
    ]
    for label, fn, xm, xc in cases:
        before = stat(f"cpu_fallback:{label}")
        got = fn(xm)
        ref = fn(xc)
        if isinstance(got, MetalFloat64):
            check(got, ref, rtol=tol, atol=tol * float(ref.abs().max()))
        else:
            assert got.device.type == "mps", label
            assert torch.equal(got.cpu(), ref), label
        assert stat(f"cpu_fallback:{label}") == before + 1, (label, mt.stats())
    # results are usable downstream on the GPU
    inv = torch.linalg.inv(am)
    check(inv @ am, torch.eye(5, dtype=torch.float64), rtol=0.0, atol=1e-12)
    # solver-family fallbacks reached through composites
    lu, piv = torch.linalg.lu_factor(am)
    got = torch.linalg.lu_solve(lu, piv, rm)
    check(got, torch.linalg.solve(ac, rc), rtol=1e-12, atol=1e-13)
    assert stat("cpu_fallback:linalg_lu_solve") == 1
    p_, l_, u_ = torch.lu_unpack(lu, piv)
    check(p_ @ l_ @ u_, ac, rtol=1e-12, atol=1e-13)
    assert stat("cpu_fallback:linalg_lu_solve") == 2
    chol = torch.linalg.cholesky(sm)
    check(
        torch.cholesky_solve(rm, chol),
        torch.linalg.solve(sc, rc),
        rtol=1e-12,
        atol=1e-13,
    )
    check(torch.cholesky_inverse(chol), torch.linalg.inv(sc), rtol=1e-12, atol=1e-13)
    assert stat("cpu_fallback:cholesky_solve") == 2
    w = torch.linalg.eigvals(am)
    assert w.dtype == torch.complex128 and w.device.type == "cpu"
    assert torch.allclose(
        torch.sort(w.real)[0],
        torch.sort(torch.linalg.eigvals(ac).real)[0],
        rtol=1e-12,
        atol=1e-12,
    )
    assert stat("cpu_fallback:linalg_eig") == 1


def test_fft_and_eig_return_cpu_complex(rng):
    v = dec(rng.standard_normal(64))
    got = torch.fft.fft(metal(v))
    assert not isinstance(got, MetalFloat64)
    assert got.dtype == torch.complex128 and got.device.type == "cpu"
    assert torch.allclose(got, torch.fft.fft(cpu(v)), rtol=1e-13, atol=1e-13)
    assert stat("cpu_fallback:_fft_r2c") == 1
    got = torch.fft.rfft(metal(v))
    assert got.dtype == torch.complex128 and got.device.type == "cpu"
    assert torch.allclose(got, torch.fft.rfft(cpu(v)), rtol=1e-13, atol=1e-13)
    assert stat("cpu_fallback:_fft_r2c") == 2
    a = dec(rng.standard_normal((4, 4)))
    w, vecs = torch.linalg.eig(metal(a))
    assert w.dtype == torch.complex128 and w.device.type == "cpu"
    wr, _ = torch.linalg.eig(cpu(a))
    assert torch.allclose(
        torch.sort(w.real)[0], torch.sort(wr.real)[0], rtol=1e-12, atol=1e-12
    )
    assert stat("cpu_fallback:linalg_eig") == 1


def test_strict_mode_blocks_fallbacks(rng, monkeypatch):
    monkeypatch.setenv("OPTILAND_METAL_STRICT", "1")
    a = metal(dec(rng.standard_normal((3, 3))))
    with pytest.raises(MetalFallbackError):
        torch.linalg.inv(a)
    with pytest.raises(MetalFallbackError):
        torch.fft.fft(a)
    with pytest.raises(MetalFallbackError):
        a.cumprod(dim=0)
    # GPU paths keep working
    check(a.sum(), cpu(a.to_numpy()).sum(), rtol=1e-13, atol=1e-15)
    check(a @ a, cpu(a.to_numpy()) @ cpu(a.to_numpy()), rtol=1e-13, atol=1e-14)
    assert not any(k.startswith("cpu_fallback:") for k in mt.stats())


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------
def test_autograd_reductions_and_matmul(rng):
    x = dec(rng.standard_normal((6, 300)))
    w = dec(rng.standard_normal((6, 300)))
    a = dec(rng.standard_normal((5, 40)))
    b = dec(rng.standard_normal((40, 7)))

    def grads(make):
        xm, wm = make(x), make(w)
        am, bm = make(a), make(b)
        outs = [
            (xm.sum(), [xm]),
            ((xm * wm).sum(dim=1).sum(), [xm, wm]),
            (xm.mean(), [xm]),
            ((xm * wm).mean(dim=0)[3], [xm, wm]),
            ((am @ bm).sum(), [am, bm]),
            (torch.mv(am, bm[:, 0]).sum(), [am, bm]),
            (xm.var(dim=1).sum(), [xm]),
            (torch.dot(xm[0], wm[0]), [xm, wm]),
        ]
        return [torch.autograd.grad(o, ins) for o, ins in outs]

    def make_metal(v):
        return metal(v, requires_grad=True)

    def make_cpu(v):
        return cpu(v).requires_grad_(True)

    got = grads(make_metal)
    ref = grads(make_cpu)
    for g_list, r_list in zip(got, ref, strict=True):
        for g, r in zip(g_list, r_list, strict=True):
            assert isinstance(g, MetalFloat64)
            check(g, r, rtol=1e-12, atol=1e-12 * float(r.abs().max()))


def test_autograd_through_derived_reductions_and_products(rng):
    x = dec(rng.standard_normal((4, 300)))
    w = dec(rng.standard_normal((4, 300)))
    a = dec(rng.standard_normal((7, 3)))
    b = dec(rng.standard_normal((7, 3)))
    cases = [
        ("amax", lambda x, w, a, b: (torch.amax(x * w, dim=1) * 3).sum()),
        ("max.dim", lambda x, w, a, b: (x * w).max(dim=1)[0].sum()),
        ("max", lambda x, w, a, b: (x * w).max()),
        (
            "min.dim keepdim",
            lambda x, w, a, b: (x * w).min(dim=0, keepdim=True)[0].sum(),
        ),
        ("cumsum", lambda x, w, a, b: (x.cumsum(dim=1) * w).sum()),
        (
            "vector_norm 2",
            lambda x, w, a, b: torch.linalg.vector_norm(x * w, dim=1).sum(),
        ),
        (
            "vector_norm 3",
            lambda x, w, a, b: torch.linalg.vector_norm(x, ord=3, dim=1).sum(),
        ),
        (
            "vector_norm inf",
            lambda x, w, a, b: torch.linalg.vector_norm(x, ord=INF, dim=1).sum(),
        ),
        ("norm", lambda x, w, a, b: x.norm()),
        ("std", lambda x, w, a, b: x.std(dim=1).sum() + x.std()),
        ("var_mean", lambda x, w, a, b: sum(t.sum() for t in torch.var_mean(x, dim=1))),
        ("logsumexp", lambda x, w, a, b: torch.logsumexp(x, dim=1).sum()),
        ("nansum", lambda x, w, a, b: torch.nansum(x * w, dim=1).sum()),
        (
            "cross",
            lambda x, w, a, b: (torch.linalg.cross(a, b) * b[:, [1, 2, 0]]).sum(),
        ),
        ("outer", lambda x, w, a, b: (torch.outer(a[:, 0], b[:, 1]) * a[:, 1:2]).sum()),
        (
            "bmm",
            lambda x, w, a, b: torch.bmm(
                x.reshape(4, 30, 10), w.reshape(4, 10, 30)
            ).sum(),
        ),
        (
            "addmm",
            lambda x, w, a, b: torch.addmm(
                a[:3, :3], a.t(), b, beta=0.5, alpha=2.0
            ).sum(),
        ),
        (
            "matmul batched",
            lambda x, w, a, b: torch.matmul(
                x.reshape(2, 2, 30, 10), w.reshape(2, 2, 10, 30)
            ).sum(),
        ),
        ("vector @ matrix", lambda x, w, a, b: (a[:, 0] @ a).sum()),
        ("matrix @ vector", lambda x, w, a, b: (a @ b[0]).sum()),
        (
            "inv fallback",
            lambda x, w, a, b: torch.linalg.inv(a[:3] @ a[:3].t() + 3.0).sum(),
        ),
        ("det fallback", lambda x, w, a, b: torch.linalg.det(a[:3] + 3.0)),
        (
            "cholesky fallback",
            lambda x, w, a, b: torch.linalg.cholesky(a[:3] @ a[:3].t() + 1.0).sum(),
        ),
        ("qr fallback", lambda x, w, a, b: torch.linalg.qr(a)[1].sum()),
        (
            "solve fallback",
            lambda x, w, a, b: torch.linalg.solve(a[:3] + 3.0, b[:3]).sum(),
        ),
        (
            "prod",
            lambda x, w, a, b: (x[:, :10] * 0.5).prod(dim=1).sum() + x[:2, :5].prod(),
        ),
        ("trace", lambda x, w, a, b: torch.trace(a[:3] * b[:3])),
    ]
    for name, fn in cases:
        ins_m = [metal(v, requires_grad=True) for v in (x, w, a, b)]
        ins_c = [cpu(v).requires_grad_(True) for v in (x, w, a, b)]
        got = torch.autograd.grad(fn(*ins_m), ins_m, allow_unused=True)
        ref = torch.autograd.grad(fn(*ins_c), ins_c, allow_unused=True)
        for g, r in zip(got, ref, strict=True):
            if r is None:
                assert g is None, name
                continue
            assert isinstance(g, MetalFloat64), (name, type(g))
            check(g, r, rtol=1e-12, atol=1e-12 * float(r.abs().max()) + 1e-14)


def test_autograd_backward_accumulates(rng):
    x = dec(rng.standard_normal((4, 500)))
    xm = metal(x, requires_grad=True)
    y = (xm * xm).sum(dim=1).sum() + xm.sum()
    y.backward()
    ref = 2 * x + 1
    check(xm.grad, ref, rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------------------
# sf64 mode (exact binary64 kernels)
# ---------------------------------------------------------------------------
def test_sf64_kernel_reductions(sf64_kernels):
    rng = np.random.default_rng(50)
    x = rng.standard_normal((3, 20_000, 2)) * 10.0 ** rng.integers(
        -3, 4, size=(3, 20_000, 2)
    )
    xm = metal(x, mode="sf64")
    assert xm.mode == "sf64"
    ref = fsum_axis(x, 1)
    depth = 20_000 // 256 + 8 + ops_reduce._choose_groups(20_000, 6) + 8
    check(xm.sum(dim=1), ref, rtol=0.0, atol=depth * 2.0**-53 * np.abs(x).sum(axis=1))
    xi = rng.integers(-1000, 1000, size=(3, 20_000, 2)).astype(np.float64)
    check(metal(xi, mode="sf64").sum(dim=1), xi.sum(axis=1), rtol=0)
    check(metal(xi, mode="sf64").sum(), cpu(xi).sum(), rtol=0)
    y = _tie_data(rng, (5, 3000, 4))
    y[0, 100, 0] = NAN
    y[1, :, 1] = NAN
    ym, yc = metal(y, mode="sf64"), cpu(y)
    check(torch.amax(ym, dim=1), torch.amax(yc, dim=1), rtol=0)
    check(torch.amin(ym, dim=(0, 2)), torch.amin(yc, dim=(0, 2)), rtol=0)
    v, i = ym.max(dim=1)
    vr, ir = yc.max(dim=1)
    check(v, vr, rtol=0)
    check(i, ir)
    check(ym.argmin(dim=1), yc.argmin(dim=1))
    z = rng.standard_normal((4, 5000, 3)) * 10.0 ** rng.integers(
        -3, 4, size=(4, 5000, 3)
    )
    z[0, 10, 0] = NAN
    z[1, 20, 1] = INF
    check(
        metal(z, mode="sf64").cumsum(dim=1), np.cumsum(z, axis=1), rtol=0
    )  # sequential = numpy
    p = rng.uniform(0.5, 2.0, size=(3, 40))
    got = val(metal(p, mode="sf64").prod(dim=1))
    for row in range(3):
        exact = float(math.prod(Fraction(float(t)) for t in p[row]))
        assert abs(got[row] - exact) <= 40 * 2.0**-53 * abs(exact)


def test_sf64_kernel_matmul_bit_exact(sf64_kernels):
    rng = np.random.default_rng(77)
    a = rng.standard_normal((2, 3, 64)) * 10.0 ** rng.integers(-3, 4, size=(2, 3, 64))
    b = rng.standard_normal((2, 64, 3)) * 10.0 ** rng.integers(-3, 4, size=(2, 64, 3))
    a[0, 0, 0] = NAN
    got = val(torch.bmm(metal(a, mode="sf64"), metal(b, mode="sf64")))
    ref = np.zeros((2, 3, 3))
    for bb in range(2):
        for i in range(3):
            for j in range(3):
                acc = 0.0
                for kk in range(64):
                    acc += float(a[bb, i, kk]) * float(b[bb, kk, j])
                ref[bb, i, j] = acc
    assert np.array_equal(got, ref, equal_nan=True)
    x = rng.standard_normal(20_000)
    y = rng.standard_normal(20_000)
    got = float(val(torch.dot(metal(x, mode="sf64"), metal(y, mode="sf64"))))
    exact = math.fsum(
        [
            float(Fraction(float(p)) * Fraction(float(q)))
            for p, q in zip(x, y, strict=True)
        ]
    )
    depth = 20_000 // 256 + 8 + ops_reduce._choose_groups(20_000, 1) + 8
    assert abs(got - exact) <= depth * 2.0**-53 * float(np.abs(x) @ np.abs(y))


def test_sf64_derived_reductions(sf64_kernels, sf64_elementwise):
    rng = np.random.default_rng(51)
    x = rng.standard_normal((4, 3000)) * 10.0 ** rng.integers(-3, 4, size=(4, 3000))
    xm, xc = metal(x, mode="sf64"), cpu(x)
    check(xm.mean(dim=1), xc.mean(dim=1), rtol=1e-14)
    check(xm.var(dim=1), xc.var(dim=1), rtol=1e-13)
    check(xm.std(), xc.std(), rtol=1e-13)
    check(
        torch.linalg.vector_norm(xm, dim=1),
        torch.linalg.vector_norm(xc, dim=1),
        rtol=1e-13,
    )
    check(
        torch.addmm(xm[:, :4], xm[:, :5], xm[:, :5].t(), beta=0.5, alpha=2.0),
        torch.addmm(xc[:, :4], xc[:, :5], xc[:, :5].t(), beta=0.5, alpha=2.0),
        rtol=1e-13,
    )
