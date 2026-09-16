"""Regression tests for the dual-residency host path (NOTES/02-design.md section 4).

Covers the residency rules with the default threshold (256), aliasing through
host views and the shared cache cell, ``.item()`` without a device sync,
scalar-operand kernel launches (0-d host-resident operands travel as kernel
constants, never as host-to-GPU copies), mixed host/GPU ops, autograd through
the host path, ``interp`` on the dispatch path and an end-to-end CookeTriplet
trace against the NumPy oracle with the host path on.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
    pytest.skip("Metal GPU required", allow_module_level=True)

import optiland.backend as be  # noqa: E402
from optiland.backend.torch_backend import metal  # noqa: E402
from optiland.backend.torch_backend.metal import encode, factories  # noqa: E402
from optiland.backend.torch_backend.metal import tensor as T  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import MetalFloat64  # noqa: E402

THRESHOLD = 256
BIG = THRESHOLD + 744  # 1000 elements: always GPU-resident
U48 = 2.0**-48
MODES = ("df64", "sf64")


@pytest.fixture(autouse=True)
def host_path(monkeypatch):
    """Pin the design's default threshold (independent of the environment)."""
    monkeypatch.setattr(T, "HOST_THRESHOLD", THRESHOLD)
    T.reset_stats()
    yield
    T.reset_stats()


@pytest.fixture
def encode_spy(monkeypatch):
    """Count host-to-GPU encodings (each is one ``.to(mps)`` per component)."""
    calls: list[tuple[int, ...]] = []
    orig = T._encode_host

    def spy(host: torch.Tensor, mode: str) -> tuple[torch.Tensor, ...]:
        calls.append(tuple(host.shape))
        return orig(host, mode)

    monkeypatch.setattr(T, "_encode_host", spy)
    return calls


@pytest.fixture
def decode_spy(monkeypatch):
    """Count GPU-to-host decodes (the only place a device sync can hide)."""
    calls: list[int] = []
    orig = MetalFloat64.to_numpy

    def spy(self: MetalFloat64) -> np.ndarray:
        if self._host is None:
            calls.append(self.numel())
        return orig(self)

    monkeypatch.setattr(MetalFloat64, "to_numpy", spy)
    return calls


def _count(prefix: str) -> int:
    return sum(v for k, v in T.stats().items() if k.startswith(prefix))


def _decoded(a: np.ndarray) -> np.ndarray:
    """What a GPU-resident df64 tensor holds for the float64 values ``a``."""
    hi, lo = encode.encode_df64(np.asarray(a, dtype=np.float64))
    return np.asarray(encode.decode_df64(hi, lo), dtype=np.float64)


def _gpu(a: np.ndarray, mode: str = "df64", **kw: Any) -> MetalFloat64:
    return MetalFloat64.from_numpy(
        np.asarray(a, dtype=np.float64), mode, host=False, **kw
    )


def _host(a: Any, mode: str = "df64", **kw: Any) -> MetalFloat64:
    return MetalFloat64.from_numpy(
        np.asarray(a, dtype=np.float64), mode, host=True, **kw
    )


# ---------------------------------------------------------------------------
# residency rules
# ---------------------------------------------------------------------------
def test_default_threshold_is_256_with_env_override():
    assert T.DEFAULT_HOST_THRESHOLD == 256
    env = os.environ.get("OPTILAND_METAL_HOST_THRESHOLD")
    expected = int(env) if env is not None else T.DEFAULT_HOST_THRESHOLD
    # the fixture pinned the live value; the import-time value follows the env
    assert T.get_host_threshold() == THRESHOLD
    assert factories.host_threshold() == THRESHOLD
    assert metal.get_host_threshold() == THRESHOLD
    metal.set_host_threshold(0)
    try:
        assert T.HOST_THRESHOLD == 0 and factories.host_threshold() == 0
        assert not MetalFloat64.from_numpy(np.zeros(3)).is_host_resident
    finally:
        metal.set_host_threshold(THRESHOLD)
    assert int(os.environ.get("OPTILAND_METAL_HOST_THRESHOLD", "256")) == expected


@pytest.mark.parametrize("mode", MODES)
def test_creation_residency_follows_size(mode):
    small = MetalFloat64.from_numpy(np.arange(THRESHOLD, dtype=np.float64), mode)
    big = MetalFloat64.from_numpy(np.arange(THRESHOLD + 1, dtype=np.float64), mode)
    assert small.is_host_resident and not big.is_host_resident
    assert small.device.type == "mps" and small.dtype == torch.float64
    metal.set_mode(mode)
    try:
        assert factories.zeros((16, 16)).is_host_resident
        assert not factories.zeros((16, 17)).is_host_resident
        assert factories.scalar(2.5).is_host_resident
        assert factories.linspace(0, 1, THRESHOLD).is_host_resident
        assert not factories.arange(0, THRESHOLD + 1).is_host_resident
        assert factories.tensor([1.0, 2.0]).is_host_resident
        assert not factories.tensor(np.zeros(BIG)).is_host_resident
    finally:
        metal.set_mode("df64")
    # factories emitted by C++ autograd formulas follow the same rule
    assert torch.zeros((4,), dtype=torch.float64, device="mps").is_host_resident
    assert not torch.zeros((BIG,), dtype=torch.float64, device="mps").is_host_resident


def test_results_follow_the_computation_site():
    a, b = _host([1.0, 2.0, 3.0]), _host(0.5)
    c = a * b + 1.0
    assert c.is_host_resident
    np.testing.assert_array_equal(c.to_numpy(), [1.5, 2.0, 2.5])  # exact float64
    assert _count("gpu:") == 0 and _count("host:") == 2
    x = _gpu(np.linspace(0, 1, BIG))
    y = x * b
    assert not y.is_host_resident
    s = y.sum()
    assert not s.is_host_resident  # a GPU reduction result stays on the GPU
    m = x > 0.5
    assert m.device.type == "mps"
    n = a > 1.5
    assert n.device.type == "cpu"  # host-path predicates stay on the CPU


def test_views_keep_the_residency_of_their_base():
    x = _gpu(np.arange(600.0).reshape(20, 30))
    v = x[:2, :3]
    assert not v.is_host_resident and v.numel() <= THRESHOLD
    v.zero_()  # writes through to the GPU base
    out = x.to_numpy()
    assert out[:2, :3].sum() == 0 and out[2, 0] == 60.0
    e = x[0, 5] * 2 + 1  # small GPU view: decoded once, continues on the host
    assert e.is_host_resident and float(e) == 11.0
    h = _host(np.arange(12.0).reshape(3, 4))
    hv = h[1]
    assert hv.is_host_resident and hv._cell is h._cell


# ---------------------------------------------------------------------------
# aliasing through host views and the cache cell
# ---------------------------------------------------------------------------
def test_host_view_aliasing_and_cache_invalidation(encode_spy):
    a = _host([1.0, 2.0, 3.0, 4.0])
    v = a[1:3]
    assert v._cell is a._cell and T._is_host_view(v)
    base_comps = a._comps
    assert a._comps is base_comps and len(encode_spy) == 1  # base cached in cell
    view_comps = v._comps
    assert a[1:3]._comps is view_comps and len(encode_spy) == 2  # geometry cache
    assert a[0:2]._comps is not view_comps and len(encode_spy) == 3
    v.mul_(10)  # in-place through the view: host truth updated, caches dropped
    np.testing.assert_array_equal(a.to_numpy(), [1.0, 20.0, 30.0, 4.0])
    assert a._cell.comps is None and a._cell.views is None
    np.testing.assert_array_equal(a._comps[0].cpu().numpy(), [1.0, 20.0, 30.0, 4.0])
    np.testing.assert_array_equal(a[1:3]._comps[0].cpu().numpy(), [20.0, 30.0])
    a[0] = 7.0  # Python-scalar assignment (__setitem__ wraps the scalar)
    np.testing.assert_array_equal(a.to_numpy(), [7.0, 20.0, 30.0, 4.0])
    np.testing.assert_array_equal(v.to_numpy(), [20.0, 30.0])
    assert a._cell.comps is None
    w = a[3]
    w.add_(1.0)
    assert float(a[3]) == 5.0
    c = a.clone()
    assert c.is_host_resident and c._cell is not a._cell
    c.mul_(0.0)
    assert float(a[3]) == 5.0


def test_encoding_cache_is_reused_across_uses(encode_spy):
    coeffs = _host(np.array([1e-3, -2e-5, 3e-7]))
    x = _gpu(np.linspace(0.1, 0.9, BIG))
    for _ in range(3):
        y = x * coeffs[0] + x * x * coeffs[1]  # 0-d host views: kernel scalars
        assert not y.is_host_resident
    assert encode_spy == []
    row = _host(np.array([[1.0, 2.0, 3.0]]))
    for _ in range(3):
        z = x.reshape(-1, 1) * row  # (BIG, 3): the (1, 3) host row broadcast
    assert encode_spy == [(1, 3)]  # encoded once, reused from the cell
    np.testing.assert_allclose(
        z.to_numpy(), np.linspace(0.1, 0.9, BIG)[:, None] * [1.0, 2.0, 3.0], rtol=1e-13
    )
    for _ in range(3):
        _ = x.reshape(-1, 1) * row[:, 1:]  # a view: cached by geometry
    assert encode_spy == [(1, 3), (1, 2)]


# ---------------------------------------------------------------------------
# .item() / float() without sync
# ---------------------------------------------------------------------------
def test_item_reads_host_truth_without_sync(encode_spy, decode_spy):
    a, b = _host([1.0, 2.0, 3.0]), _host(0.1)
    c = a * b + 1
    assert c[1].item() == 1.2 and float(c[2]) == 1.3 and bool(c[0] > 1.0)
    assert f"{b:.2f}" == "0.10" and int(c[2] * 10) == 13
    assert _count("gpu:") == 0
    assert _count("host:") >= 4  # mul, add, two selects, gt ...
    assert encode_spy == [] and decode_spy == []


# ---------------------------------------------------------------------------
# scalar-operand launches
# ---------------------------------------------------------------------------
_BINARY_CASES = [
    ("mul", lambda x, s: x * s, lambda x, s: x * s),
    ("rmul", lambda x, s: s * x, lambda x, s: s * x),
    ("add", lambda x, s: x + s, lambda x, s: x + s),
    ("sub", lambda x, s: x - s, lambda x, s: x - s),
    ("rsub", lambda x, s: s - x, lambda x, s: s - x),
    ("div", lambda x, s: x / s, lambda x, s: x / s),
    ("rdiv", lambda x, s: s / x, lambda x, s: s / x),
    ("maximum", torch.maximum, np.maximum),
    ("minimum", torch.minimum, np.minimum),
    ("atan2", torch.atan2, np.arctan2),
    ("pow", lambda x, s: x**s, lambda x, s: x**s),
]


@pytest.mark.parametrize(
    "name,op,ref", _BINARY_CASES, ids=[c[0] for c in _BINARY_CASES]
)
def test_binary_op_with_host_scalar_launches_scalar_kernel(name, op, ref, encode_spy):
    a = np.linspace(0.5, 2.5, BIG)
    x = _gpu(a)
    s = _host(0.7)
    T.reset_stats()
    got = op(x, s)
    assert not got.is_host_resident
    assert _count("gpu:") == 1, T.stats()
    assert encode_spy == [], "the 0-d host operand must not be copied to the GPU"
    # the kernel sees the scalar's exact (hi, lo) pair: 0.7 rounded to 48 bits
    expected = ref(_decoded(a), float(_decoded(np.array(0.7))))
    np.testing.assert_allclose(got.to_numpy(), expected, rtol=8 * U48, atol=0)


@pytest.mark.parametrize(
    "name,op,ref", _BINARY_CASES[:9], ids=[c[0] for c in _BINARY_CASES[:9]]
)
def test_sf64_binary_op_with_host_scalar_is_bit_exact(name, op, ref, encode_spy):
    a = np.linspace(0.5, 2.5, BIG)
    x = _gpu(a, "sf64")
    s = _host(0.7, "sf64")
    got = op(x, s)
    assert got.mode == "sf64" and not got.is_host_resident
    assert encode_spy == []
    np.testing.assert_array_equal(got.to_numpy(), ref(a, 0.7))


def test_ternary_and_where_with_host_scalars(encode_spy):
    a = np.linspace(-1.0, 1.0, BIG)
    x = _gpu(a)
    y = _gpu(a[::-1].copy())
    lo, hi, w, v = _host(-0.25), _host(0.5), _host(0.3), _host(2.0)
    T.reset_stats()
    clamped = torch.clamp(x, lo, hi)
    lerped = torch.lerp(x, y, w)
    addc = torch.addcmul(x, y, v)
    addd = torch.addcdiv(x, y, v)
    assert encode_spy == []
    launches_ternary = _count("gpu:")
    assert launches_ternary >= 4
    ad = _decoded(a)
    np.testing.assert_allclose(
        clamped.to_numpy(), np.clip(ad, -0.25, 0.5), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        lerped.to_numpy(), ad + 0.3 * (ad[::-1] - ad), rtol=8 * U48, atol=1e-15
    )
    np.testing.assert_allclose(
        addc.to_numpy(), ad + ad[::-1] * 2.0, rtol=8 * U48, atol=1e-15
    )
    np.testing.assert_allclose(
        addd.to_numpy(), ad + ad[::-1] / 2.0, rtol=8 * U48, atol=1e-15
    )
    # the decomposition is bit-identical to the fused ternary kernel
    fused = torch.lerp(x, y, _gpu(np.full(BIG, 0.3)))
    np.testing.assert_array_equal(lerped.to_numpy(), fused.to_numpy())
    # where: one or two scalar sides never copy anything to the GPU
    mask = x > 0
    r1 = torch.where(mask, x, v)
    r2 = torch.where(mask, v, x)
    r3 = torch.where(mask, v, lo)
    r4 = torch.where(mask, x, math_nan())
    assert encode_spy == []
    np.testing.assert_array_equal(r1.to_numpy(), np.where(ad > 0, ad, 2.0))
    np.testing.assert_array_equal(r2.to_numpy(), np.where(ad > 0, 2.0, ad))
    np.testing.assert_array_equal(r3.to_numpy(), np.where(ad > 0, 2.0, -0.25))
    np.testing.assert_array_equal(r4.to_numpy(), np.where(ad > 0, ad, np.nan))
    for r in (r1, r2, r3, r4):
        assert not r.is_host_resident and r.shape == x.shape


def math_nan() -> float:
    return float("nan")


def test_scalar_path_for_sf64_where_and_clamp(encode_spy):
    a = np.linspace(-1.0, 1.0, BIG)
    x = _gpu(a, "sf64")
    lo, hi, v = _host(-0.25, "sf64"), _host(0.5, "sf64"), _host(2.0, "sf64")
    got = torch.clamp(x, lo, hi)
    r = torch.where(x > 0, x, v)
    assert encode_spy == []
    np.testing.assert_array_equal(got.to_numpy(), np.clip(a, -0.25, 0.5))
    np.testing.assert_array_equal(r.to_numpy(), np.where(a > 0, a, 2.0))


def test_gpu_scalar_operand_still_launches(encode_spy, decode_spy):
    """A 0-d GPU-resident operand is not a host scalar: it is broadcast, not read."""
    a = np.linspace(0.5, 2.5, BIG)
    x = _gpu(a)
    s = _gpu(np.array(0.7))
    got = x * s
    assert not got.is_host_resident and decode_spy == [] and encode_spy == []
    np.testing.assert_allclose(got.to_numpy(), _decoded(a) * 0.7, rtol=8 * U48)


# ---------------------------------------------------------------------------
# mixed host / GPU ops
# ---------------------------------------------------------------------------
def test_mixed_host_gpu_arithmetic_and_masks(encode_spy):
    a = np.linspace(0, 1, BIG)
    big = _gpu(a)
    small = _host([1.0, -2.0, 3.0])
    scalar = _host(0.1)
    prod = big * scalar
    assert not prod.is_host_resident
    np.testing.assert_allclose(prod.to_numpy(), _decoded(a) * 0.1, rtol=8 * U48)
    mask = big > 0.5
    assert mask.device.type == "mps"
    big[mask] = scalar  # host value into a GPU tensor
    np.testing.assert_allclose(
        big.to_numpy(), np.where(a > 0.5, 0.1, _decoded(a)), rtol=8 * U48
    )
    m = small < 0
    assert m.device.type == "cpu"
    small[m] = 0.0
    np.testing.assert_array_equal(small.to_numpy(), [1.0, 0.0, 3.0])
    assert torch.where(m, small, 7.0).to_numpy().tolist() == [7.0, 0.0, 7.0]
    # a host-resident result larger than the threshold (an all-host ``cat``)
    # is encoded on first use, then reused from its cell
    rep = torch.cat([_host(np.full(BIG // 4, 2.0))] * 4)
    assert rep.is_host_resident and rep.numel() == BIG
    n_before = len(encode_spy)
    out = big + rep
    assert not out.is_host_resident and len(encode_spy) == n_before + 1
    out2 = big - rep
    assert len(encode_spy) == n_before + 1  # cached in the cell
    np.testing.assert_allclose(
        (out - out2).to_numpy(), np.full(BIG, 4.0), rtol=8 * U48, atol=1e-14
    )
    # host tensors are representation-agnostic: they follow the GPU operand
    g = _gpu(np.ones(BIG), "sf64")
    h = _host(2.0)
    r = g * h
    assert r.mode == "sf64" and h.mode == "sf64"
    np.testing.assert_array_equal(r.to_numpy(), 2.0)


def test_mixed_reduction_and_cat():
    a = np.linspace(0, 1, BIG)
    big = _gpu(a)
    small = _host([1.0, 2.0])
    total = big.sum() + small.sum()
    assert abs(float(total) - (a.sum() + 3.0)) < 1e-9
    cat = torch.cat([small, big])
    assert not cat.is_host_resident and cat.shape == (BIG + 2,)
    np.testing.assert_allclose(
        cat.to_numpy(), np.concatenate([[1.0, 2.0], _decoded(a)]), rtol=8 * U48
    )
    both = torch.cat([small, small])
    assert both.is_host_resident
    np.testing.assert_array_equal(both.to_numpy(), [1.0, 2.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# autograd through the host path
# ---------------------------------------------------------------------------
def test_autograd_host_only_chain():
    x = _host([0.5, 1.5], requires_grad=True)
    y = (x * x * 3 + torch.sin(x)).sum()
    y.backward()
    ref = 6 * np.array([0.5, 1.5]) + np.cos([0.5, 1.5])
    np.testing.assert_allclose(x.grad.to_numpy(), ref, rtol=1e-15)
    assert x.grad.is_host_resident
    assert _count("gpu:") == 0


def test_autograd_mixed_host_scalar_and_gpu_array():
    a = np.linspace(0.1, 1.0, BIG)
    w = _gpu(a, requires_grad=True)
    s = _host(0.7, requires_grad=True)
    t = _host(2.0, requires_grad=True)
    loss = ((w * s + t) * (w * s + t)).sum()
    loss.backward()
    ad = _decoded(a)
    # dL/dw = 2 (w s + t) s ; dL/ds = sum 2 (w s + t) w ; dL/dt = sum 2 (w s + t)
    inner = ad * 0.7 + 2.0
    np.testing.assert_allclose(w.grad.to_numpy(), 2 * inner * 0.7, rtol=1e-12)
    np.testing.assert_allclose(float(s.grad), float((2 * inner * ad).sum()), rtol=1e-11)
    np.testing.assert_allclose(float(t.grad), float((2 * inner).sum()), rtol=1e-11)
    assert s.grad.shape == () and t.grad.shape == ()


def test_autograd_through_host_views_and_where():
    p = _host([1.0, 2.0, 3.0], requires_grad=True)
    a = np.linspace(-1, 1, BIG)
    x = _gpu(a)
    y = torch.where(x > 0, x * p[0], p[1] * x * x + p[2]).sum()
    y.backward()
    ad = _decoded(a)
    pos = ad > 0
    ref = [ad[pos].sum(), (ad[~pos] ** 2).sum(), float((~pos).sum())]
    np.testing.assert_allclose(p.grad.to_numpy(), ref, rtol=1e-11)


# ---------------------------------------------------------------------------
# interp on the dispatch path
# ---------------------------------------------------------------------------
@pytest.fixture
def mps_backend():
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.disable()
    be.metal_reset_stats()
    yield
    be.grad_mode.disable()
    be.set_backend("numpy")


def _interp_bound(
    x: np.ndarray, xp: np.ndarray, fp: np.ndarray, eps: float
) -> np.ndarray:
    """Error bound of ``interp`` from rounding the *inputs* to ``eps`` relative.

    ``|slope| * |x| * eps`` from the query, plus a few ``eps`` of the result.
    """
    j = np.clip(np.searchsorted(xp, np.nan_to_num(x), side="right"), 1, len(xp) - 1)
    slope = np.abs((fp[j] - fp[j - 1]) / (xp[j] - xp[j - 1]))
    y = np.nan_to_num(np.interp(x, xp, fp))
    return 8 * eps * (slope * np.abs(np.nan_to_num(x)) + np.abs(y)) + 1e-300


@pytest.mark.parametrize("n_tab,n_x", [(9, 5), (300, 5), (9, BIG), (300, BIG)])
def test_interp_matches_numpy_on_dispatch_path(mps_backend, n_tab, n_x, decode_spy):
    rng = np.random.default_rng(20260916 + n_tab + n_x)
    xp = np.sort(rng.uniform(0.3, 2.0, n_tab))
    fp = 1.5 + 0.1 * np.sin(4 * xp) + 0.01 * rng.standard_normal(n_tab)
    x = np.concatenate(
        [rng.uniform(0.0, 2.5, n_x - 4), [np.nan, xp[0], xp[-1], xp[n_tab // 2]]]
    )
    ref = np.interp(x, xp, fp)
    got = be.interp(x, xp, fp)
    assert be.metal_stats() and not any(
        k.startswith("cpu_") for k in be.metal_stats()
    ), be.metal_stats()
    out = got.to_numpy()
    assert out.shape == ref.shape
    assert np.array_equal(np.isnan(out), np.isnan(ref))
    # the result lives where the last (query-sized) arithmetic ran
    assert got.is_host_resident == (n_x <= THRESHOLD)
    finite = ~np.isnan(ref)
    if n_tab <= THRESHOLD and n_x <= THRESHOLD:
        # everything on the host: exact CPU float64 arithmetic, no sync at all
        np.testing.assert_allclose(out[finite], ref[finite], rtol=4e-16, atol=0)
        assert decode_spy == []
    else:
        bound = _interp_bound(x, xp, fp, U48)
        assert np.all(np.abs(out[finite] - ref[finite]) <= bound[finite])
    # both ends clamp (numpy semantics), knots are reproduced
    assert out[-3] == pytest.approx(fp[0], rel=8 * U48)
    assert out[-2] == pytest.approx(fp[-1], rel=8 * U48)
    assert out[-1] == pytest.approx(fp[n_tab // 2], rel=8 * U48)
    below = be.interp(np.array([-5.0, 0.0]), xp, fp).to_numpy()
    above = be.interp(np.array([2.5, 1e9]), xp, fp).to_numpy()
    assert np.allclose(below, fp[0], rtol=8 * U48) and np.allclose(
        above, fp[-1], rtol=8 * U48
    )


def test_interp_scalar_query_unsorted_table_and_errors(mps_backend):
    xp = np.array([0.5, 0.2, 0.9, 0.7])
    fp = xp + 1.0
    got = be.interp(0.6, xp, fp)
    assert got.is_host_resident and float(got) == pytest.approx(1.6, rel=1e-15)
    order = np.argsort(xp)
    x = np.array([0.1, 0.3, 0.6, 0.8, 1.0])
    np.testing.assert_allclose(
        be.interp(x, xp, fp).to_numpy(), np.interp(x, xp[order], fp[order]), rtol=1e-15
    )
    np.testing.assert_allclose(be.interp(x, xp[:1], fp[:1]).to_numpy(), fp[0])
    with pytest.raises(ValueError):
        be.interp(x, np.array([]), np.array([]))
    with pytest.raises(ValueError):
        be.interp(x, xp, fp[:-1])


def test_interp_keeps_gradients(mps_backend):
    be.grad_mode.enable()
    xp = np.linspace(0.4, 0.8, 9)
    fp = 1.5 + 0.01 * np.sin(xp * 20)
    for n in (3, BIG):
        x = np.linspace(0.4137, 0.7913, n)
        # the slope is discontinuous at the knots: keep the queries away
        assert np.abs(x[:, None] - xp[None, :]).min() > 1e-6
        xt = be.array(x)
        xt.requires_grad_(True)
        fpt = be.array(fp)
        fpt.requires_grad_(True)
        y = be.interp(xt, xp, fpt)
        y.sum().backward()
        j = np.clip(np.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
        slope = (fp[j] - fp[j - 1]) / (xp[j] - xp[j - 1])
        np.testing.assert_allclose(xt.grad.to_numpy(), slope, rtol=1e-11)
        # d(sum y)/d fp: the interpolation weights, which sum to n
        wsum = float(fpt.grad.sum())
        assert wsum == pytest.approx(n, rel=1e-12)
        w = np.zeros_like(fp)
        t = (x - xp[j - 1]) / (xp[j] - xp[j - 1])
        np.add.at(w, j - 1, 1 - t)
        np.add.at(w, j, t)
        np.testing.assert_allclose(fpt.grad.to_numpy(), w, rtol=1e-10, atol=1e-12)
    assert not any(k.startswith("cpu_") for k in be.metal_stats())


# ---------------------------------------------------------------------------
# end to end: CookeTriplet on mps float64 vs numpy with the host path on
# ---------------------------------------------------------------------------
def _trace_cooke() -> dict[str, np.ndarray]:
    from optiland.samples.objectives import CookeTriplet

    lens = CookeTriplet()
    lens.trace(Hx=0.0, Hy=1.0, wavelength=0.55, num_rays=16, distribution="hexapolar")
    sg = lens.surfaces
    return {
        k: np.asarray(be.to_numpy(getattr(sg, k)))
        for k in ("x", "y", "z", "L", "M", "N")
    }


def test_cooke_triplet_trace_matches_numpy_with_host_path(mps_backend, encode_spy):
    be.set_backend("numpy")
    ref = _trace_cooke()
    be.set_backend("torch")
    _trace_cooke()  # warm-up: kernel compilation, material caches
    T.reset_stats()
    encode_spy.clear()
    got = _trace_cooke()
    for k in ("x", "y", "z"):
        assert got[k].shape == ref[k].shape
        assert np.array_equal(np.isnan(got[k]), np.isnan(ref[k])), k
        worst = np.nanmax(np.abs(got[k] - ref[k]))
        assert worst < 5e-12, f"{k}: max |delta| = {worst:.3e} mm"
    for k in ("L", "M", "N"):
        assert np.nanmax(np.abs(got[k] - ref[k])) < 1e-13, k
    st = T.stats()
    assert not any(k.startswith("cpu_") for k in st), st
    launches, host_ops = _count("gpu:"), _count("host:")
    assert 0 < launches < 800, launches  # ~575 bulk ops on the 817-ray arrays
    assert host_ops > 1000, host_ops  # the scalar bookkeeping never launches
    copies = 2 * len(encode_spy)  # one .to(mps) per df64 component
    assert copies < 50, (copies, encode_spy)


def test_where_two_scalars_broadcasts_to_the_full_shape(encode_spy):
    a = np.linspace(-1.0, 1.0, BIG)
    expanded = torch.cat([_host(np.full(BIG // 4, 3.0))] * 4)  # host, > threshold
    cond = torch.tensor([True], device="mps")
    r = torch.where(cond, expanded, 5.0)
    assert r.shape == (BIG,) and not r.is_host_resident
    np.testing.assert_array_equal(r.to_numpy(), np.full(BIG, 3.0))
    assert encode_spy == [(BIG,)]  # a real (contiguous) host array is encoded once
    encode_spy.clear()
    single = _host(np.array(2.0)).reshape(1).expand(BIG)  # stride-0 host view
    assert single.is_host_resident
    r2 = torch.where(_gpu(a) > 0, single, 5.0)
    assert r2.shape == (BIG,) and encode_spy == []
    np.testing.assert_array_equal(r2.to_numpy(), np.where(_decoded(a) > 0, 2.0, 5.0))
    # scalar arithmetic on an expanded per-surface constant never launches
    four = _host(np.array(4.0)).reshape(1).expand(BIG)
    T.reset_stats()
    ratio = single / four
    assert ratio.is_host_resident and ratio.shape == (BIG,)
    assert _count("gpu:") == 0 and _count("host:") == 1  # one CPU float64 div
    mask = single > 1.0
    assert mask.device.type == "mps" and mask.shape == (BIG,) and bool(mask.all())
    assert _count("gpu:") == 0 and encode_spy == []
    out = _gpu(a) * ratio  # the expansion is a kernel scalar
    assert _count("gpu:") == 1 and encode_spy == []
    np.testing.assert_allclose(out.to_numpy(), _decoded(a) * 0.5, rtol=8 * U48)
