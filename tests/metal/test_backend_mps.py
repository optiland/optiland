"""Backend integration tests: ``be.set_device('mps')`` with float64 emulation.

Every ``be.<fn>`` creation path must return a ``MetalFloat64`` (never ask
torch for float64 on mps), values are compared against the NumPy backend on
the same inputs, random functions must reproduce the CPU float64 stream for a
seed, and the CPU fallbacks (polyfit, histogram2d, ...; ``interp`` runs on the
dispatch path) must be counted in ``be.metal_stats()``.

Every dispatch lane (elementwise, structural, reductions, linalg) has landed,
so an unregistered aten op reached from here is a real failure: the
``other_lanes`` decorator is a pass-through kept only so the test bodies read
the same as during the lane build.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
    pytest.skip("torch MPS backend is not available", allow_module_level=True)

import optiland.backend as be  # noqa: E402
from optiland.backend.torch_backend import metal  # noqa: E402
from optiland.backend.torch_backend.metal import encode, factories  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import is_metal  # noqa: E402

U2 = 2.0**-48
BIG = factories.host_threshold() + 300  # always GPU-resident under the nucleus rule


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def other_lanes(fn: Any) -> Any:
    """Pass-through: every dispatch lane has landed, missing handlers must fail."""
    return fn


def np_value(fn: Any) -> Any:
    """Evaluate ``fn()`` on the NumPy backend and return the result."""
    be.set_backend("numpy")
    try:
        out = fn()
    finally:
        be.set_backend("torch")
    return out


def decoded(x: Any) -> np.ndarray:
    """Round trip through the df64 encoding (what a GPU-resident df64 holds)."""
    hi, lo = encode.encode_df64(np.asarray(x, dtype=np.float64))
    return encode.decode_df64(hi, lo)


def expected(got: Any, ref: Any) -> np.ndarray:
    """What ``got`` must hold for the float64 values ``ref``.

    A host-resident tensor stores the exact float64 values; a GPU-resident df64
    tensor stores their (hi, lo) encoding, exact to 2^-48.
    """
    ref = np.asarray(ref, dtype=np.float64)
    if got.is_host_resident or got.mode == "sf64":
        return ref
    return decoded(ref)


def assert_metal(x: Any, shape: Any = None, mode: str = "df64") -> None:
    assert is_metal(x), type(x)
    assert x.dtype == torch.float64
    assert x.device.type == "mps"
    assert x.mode == mode
    if shape is not None:
        assert tuple(x.shape) == tuple(shape)


def assert_values(
    got: Any, ref: Any, exact: bool = False, rtol: float = 4 * U2
) -> None:
    g = be.to_numpy(got) if isinstance(got, torch.Tensor) else np.asarray(got)
    r = np.asarray(ref, dtype=np.float64)
    assert g.shape == r.shape, (g.shape, r.shape)
    if exact:
        np.testing.assert_array_equal(g, r)
    else:
        np.testing.assert_allclose(g, r, rtol=rtol, atol=0)


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


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_configuration_and_passthroughs(mps_backend):
    assert be.get_backend() == "torch"
    assert be.get_device() == "mps"
    assert be.get_precision() == 64
    assert be.get_complex_precision() == torch.complex128  # declared precision
    assert metal.is_available(), metal.unavailable_reason()
    assert metal.is_enabled()
    assert be.metal_mode() == "df64"
    be.set_metal_mode("sf64")
    try:
        assert be.metal_mode() == "sf64" and metal.get_mode() == "sf64"
        assert_metal(be.zeros((2,)), (2,), mode="sf64")
    finally:
        be.set_metal_mode("df64")
    with pytest.raises(ValueError):
        be.set_metal_mode("float128")
    assert isinstance(be.metal_stats(), dict)
    be.metal_reset_stats()
    assert be.metal_stats() == {}


def test_disable_makes_native_float64_fail(mps_backend):
    metal.disable()
    try:
        assert not be._backends["torch"]._emulated()
        with pytest.raises(TypeError):
            be.zeros((3,))
    finally:
        metal.enable()
    assert_metal(be.zeros((3,)), (3,))


# ---------------------------------------------------------------------------
# creation functions vs the NumPy backend
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        lambda: be.array([1.0, 0.1, -2.5]),
        lambda: be.array([[1, 2], [3, 4]]),
        lambda: be.array(3.5),
        lambda: be.array(np.linspace(-1, 1, 7)),
        lambda: be.array(np.arange(BIG, dtype=float)),
        lambda: be.zeros((2, 3)),
        lambda: be.zeros(4),
        lambda: be.ones((3,)),
        lambda: be.ones((BIG,)),
        lambda: be.full((2, 2), 0.1),
        lambda: be.full(3, -7),
        lambda: be.full((BIG,), np.pi),
        lambda: be.linspace(0.0, 1.0, 5),
        lambda: be.linspace(0.1, 0.7, 7),
        lambda: be.linspace(-3.0, 3.0, BIG),
        lambda: be.arange(5),
        lambda: be.arange(1, 4),
        lambda: be.arange(0.0, 1.0, 0.25),
        lambda: be.arange(0, BIG, 1),
        lambda: be.eye(3),
        lambda: be.eye(20),
        lambda: be.zeros_like(np.zeros((2, 5))),
        lambda: be.ones_like(np.zeros((BIG,))),
        lambda: be.full_like(np.zeros((3,)), 2.5),
        lambda: be.asarray([0.5, 1.5]),
        lambda: be.asarray(np.arange(6.0).reshape(2, 3)),
        lambda: be.atleast_1d(2.0),
        lambda: be.atleast_2d([1.0, 2.0]),
        lambda: be.as_array_1d(4.0),
        lambda: be.as_array_1d([1.0, 2.0]),
        lambda: be.ravel(np.arange(6.0).reshape(2, 3)),
    ],
    ids=lambda f: "",
)
def test_creation_matches_numpy(mps_backend, call):
    ref = np.asarray(np_value(call), dtype=np.float64)
    got = call()
    assert_metal(got, ref.shape)
    assert_values(got, expected(got, ref), exact=True)
    # small results are host-resident, large ones live on the GPU (nucleus rule)
    assert got.is_host_resident == (got.numel() <= factories.host_threshold())


def test_linspace_endpoints_and_arange_step(mps_backend):
    x = be.linspace(0.0, 1.0, 11)
    v = be.to_numpy(x)
    assert v[0] == 0.0 and v[-1] == 1.0
    assert_values(x, np.linspace(0.0, 1.0, 11), rtol=U2)
    a = be.arange(0.0, 1.0, 0.1)
    assert a.shape == (10,)
    assert_values(a, np.arange(0.0, 1.0, 0.1), rtol=U2)


def test_array_bool_and_tensor_inputs(mps_backend):
    b = be.array(np.array([True, False, True]))
    assert b.dtype == torch.bool and b.device.type == "mps" and not is_metal(b)
    x = be.array([1.0, 2.0])
    assert be.array(x) is x  # tensors pass through untouched
    # plain float32 mps tensor (native) is promoted exactly by cast
    f32 = torch.tensor([1.5, 2.25], device="mps")
    c = be.cast(f32)
    assert_metal(c, (2,))
    assert_values(c, [1.5, 2.25], exact=True)
    # bool mps tensor -> 1.0 / 0.0
    assert_values(
        be.cast(torch.tensor([True, False], device="mps")), [1.0, 0.0], exact=True
    )
    # cast keeps MetalFloat64 identity (autograd history intact)
    assert be.cast(x) is x
    # int64 index tensors stay int64 on mps
    idx = be.arange_indices(0, 5)
    assert idx.dtype == torch.int64 and idx.device.type == "mps"


@other_lanes
def test_array_stacks_metal_scalars(mps_backend):
    a = be.array(1.5)
    b = be.array(2.5)
    s = be.array([a, b])
    assert_metal(s, (2,))
    assert_values(s, [1.5, 2.5], exact=True)
    # mixed 0-d tensors and Python numbers
    m = be.array([a, 4.0])
    assert_values(m, [1.5, 4.0], exact=True)
    # gradients survive the stack
    be.grad_mode.enable()
    p = be.array(3.0)
    assert p.requires_grad
    q = be.array([p, be.array(1.0)])
    assert q.requires_grad


def test_empty_and_like_shapes(mps_backend):
    e = be.empty((3, 4))
    assert_metal(e, (3, 4))
    assert_metal(be.empty((BIG,)), (BIG,))
    x = be.array([[1.0, 2.0], [3.0, 4.0]])
    assert_metal(be.empty_like(x), (2, 2))
    assert_values(be.zeros_like(x), np.zeros((2, 2)), exact=True)
    assert_values(be.ones_like(x), np.ones((2, 2)), exact=True)
    fl = be.full_like(x, 0.3)
    assert_values(fl, expected(fl, np.full((2, 2), 0.3)), exact=True)
    big = be.ones((BIG,))
    assert_values(be.zeros_like(big), np.zeros(BIG), exact=True)
    assert_values(be.full_like(big, -1.25), np.full(BIG, -1.25), exact=True)


@other_lanes
def test_full_like_with_tensor_fill_keeps_graph(mps_backend):
    be.grad_mode.enable()
    fill = be.array(2.0)
    assert fill.requires_grad
    x = be.zeros((3,))
    f = be.full_like(x, fill)
    assert_metal(f, (3,))
    assert_values(f, [2.0, 2.0, 2.0], exact=True)
    assert f.requires_grad
    f.sum().backward()
    assert fill.grad is not None
    assert_values(fill.grad, 3.0, exact=True)


def test_asarray_dtype_handling(mps_backend):
    x = be.asarray(np.arange(3.0), dtype=np.float64)
    assert_metal(x, (3,))
    assert be.asarray(x) is x
    i = be.asarray(np.array([1, 2, 3]), dtype=np.int64)
    assert i.dtype == torch.int64 and i.device.type == "mps" and not is_metal(i)
    b = be.asarray(np.array([True, False]), dtype=np.bool_)
    assert b.dtype == torch.bool
    # dtype=None infers: float64 data becomes MetalFloat64, ints stay ints
    assert_metal(be.asarray(np.array([0.5, 1.5]), dtype=None), (2,))
    assert be.asarray(np.array([1, 2]), dtype=None).dtype == torch.int64


def test_tensor_and_to_tensor(mps_backend):
    t = be.tensor([1.0, 2.0])
    assert_metal(t, (2,))
    assert not t.requires_grad
    g = be.tensor([1.0, 2.0], requires_grad=True)
    assert g.requires_grad and g.is_leaf
    assert_metal(be.to_tensor(np.ones(4)), (4,))
    assert be.to_tensor(t) is t
    cpu = be.to_tensor(t, device="cpu")
    assert not is_metal(cpu) and cpu.dtype == torch.float64 and cpu.device.type == "cpu"
    assert_values(cpu, [1.0, 2.0], exact=True)
    # complex128 is the declared complex precision; complex data lives on the CPU
    c = be.tensor([1.0 + 1j], dtype=torch.complex128)
    assert c.dtype == torch.complex128 and c.device.type == "cpu"
    # float32 request on mps stays native
    f = be.tensor([1.0], dtype=torch.float32)
    assert f.dtype == torch.float32 and f.device.type == "mps" and not is_metal(f)


def test_copy_cast_and_to_numpy_round_trips(mps_backend):
    x = be.array([0.1, 0.2, 0.3])
    y = be.copy(x)
    assert_metal(y, (3,))
    assert y is not x
    assert_values(y, be.to_numpy(x), exact=True)
    npv = be.to_numpy(x)
    assert isinstance(npv, np.ndarray) and npv.dtype == np.float64
    assert_values(be.array(npv), npv, exact=True)
    big = be.linspace(0, 1, BIG)
    assert_values(be.copy(big), be.to_numpy(big), exact=True)
    # a MetalFloat64 is a torch tensor for the boundary utilities
    assert be.is_torch_tensor(x)
    assert be.size(x) == 3 and be.shape(x) == (3,)
    assert float(x[1]) == pytest.approx(0.2, rel=U2)


def test_copy_to_writes_into_leaf(mps_backend):
    be.grad_mode.enable()
    dst = be.zeros((3,))
    assert dst.requires_grad
    be.copy_to(be.array([1.0, 2.0, 3.0]), dst)
    assert_values(dst, [1.0, 2.0, 3.0], exact=True)
    assert dst.requires_grad and dst.is_leaf
    big_dst = be.zeros((BIG,))
    be.copy_to(be.ones((BIG,)), big_dst)
    assert_values(big_dst, np.ones(BIG), exact=True)


def test_load_npy(mps_backend, tmp_path):
    p = tmp_path / "a.npy"
    a = np.random.default_rng(0).standard_normal((4, 3))
    np.save(p, a)
    x = be.load(str(p))
    assert_metal(x, (4, 3))
    assert_values(x, expected(x, a), exact=True)


def test_grad_mode_creates_requires_grad_tensors(mps_backend):
    be.grad_mode.enable()
    for x in (
        be.array([1.0, 2.0]),
        be.zeros((2,)),
        be.ones((BIG,)),
        be.full((2,), 3.0),
        be.linspace(0, 1, 3),
        be.arange(3),
        be.zeros_like(np.zeros(2)),
        be.ones_like(np.zeros(2)),
        be.full_like(np.zeros(2), 1.0),
        be.rand(3),
    ):
        assert_metal(x)
        assert x.requires_grad and x.is_leaf
    be.grad_mode.disable()
    assert not be.array([1.0]).requires_grad
    assert not be.zeros((2,)).requires_grad


# ---------------------------------------------------------------------------
# dispatch-level creation handlers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [4, BIG])
def test_like_and_new_handlers(mps_backend, n):
    x = be.linspace(0, 1, n)
    z = torch.zeros_like(x)
    assert_metal(z, (n,))
    assert_values(z, np.zeros(n), exact=True)
    o = torch.ones_like(x, requires_grad=True)
    assert o.requires_grad and o.is_leaf
    assert_values(o, np.ones(n), exact=True)
    f = torch.full_like(x, 0.1)
    assert_values(f, expected(f, np.full(n, 0.1)), exact=True)
    assert_metal(torch.empty_like(x), (n,))
    # non-float64 dtypes give plain mps tensors
    f32 = torch.zeros_like(x, dtype=torch.float32)
    assert not is_metal(f32) and f32.dtype == torch.float32 and f32.device.type == "mps"
    b = torch.ones_like(x, dtype=torch.bool)
    assert b.dtype == torch.bool and bool(b.all())
    h = torch.full_like(x, 0.1, dtype=torch.float32)
    assert h.dtype == torch.float32 and float(h[0]) == np.float32(0.1)
    # another device gives a plain tensor there
    cpu = torch.zeros_like(x, device="cpu")
    assert not is_metal(cpu) and cpu.device.type == "cpu" and cpu.dtype == torch.float64
    # new_* honour the requested size and fill
    nz = x.new_zeros((2, 3))
    assert_metal(nz, (2, 3))
    assert_values(nz, np.zeros((2, 3)), exact=True)
    assert_values(x.new_ones((BIG,)), np.ones(BIG), exact=True)
    nf = x.new_full((3,), np.pi)
    assert_values(nf, expected(nf, np.full(3, np.pi)), exact=True)
    assert_metal(x.new_empty((5,)), (5,))
    ne = x.new_empty_strided((2, 3), (3, 1))
    assert_metal(ne, (2, 3)) and ne.stride() == (3, 1)
    assert x.new_ones((2,), dtype=torch.int64).dtype == torch.int64


def test_sf64_creation_is_bit_exact(mps_backend):
    be.set_metal_mode("sf64")
    try:
        vals = np.array([0.1, np.pi, -1e-300, 1e300, 0.0, -0.0])
        x = be.array(vals)
        assert_metal(x, (6,), mode="sf64")
        got = be.to_numpy(x)
        assert np.array_equal(got.view(np.int64), vals.view(np.int64))
        f = be.full((BIG,), 0.1)
        assert_metal(f, (BIG,), mode="sf64")
        assert np.all(be.to_numpy(f) == 0.1)
        assert_values(torch.full_like(f, np.e), np.full(BIG, np.e), exact=True)
        assert_values(be.linspace(0, 1, 7), np.linspace(0, 1, 7), exact=True)
        assert be.zeros_like(f).mode == "sf64"
    finally:
        be.set_metal_mode("df64")


def test_factories_direct(mps_backend):
    s = factories.scalar(2.5)
    assert_metal(s, ())
    assert float(s) == 2.5
    assert_metal(factories.eye(2, 3), (2, 3))
    assert_values(factories.eye(2, 3), np.eye(2, 3), exact=True)
    assert_values(factories.arange(3), [0.0, 1.0, 2.0], exact=True)
    t = factories.tensor(torch.tensor([1.0, 2.0]))  # CPU float32 input
    assert_values(t, [1.0, 2.0], exact=True)
    m = factories.tensor(s)  # a copy, not the same object
    assert m is not s and float(m) == 2.5
    assert factories.as_tensor(s) is s
    with pytest.raises(TypeError):
        factories.as_tensor(object())
    # None is rejected like torch.tensor(None) (NumPy would give NaN)
    with pytest.raises(TypeError):
        be.array(None)
    with pytest.raises(TypeError):
        be.asarray([1.0, None])


# ---------------------------------------------------------------------------
# elementwise / structural entry points (other lanes) against NumPy
# ---------------------------------------------------------------------------
@other_lanes
def test_where_maximum_isclose_allclose(mps_backend):
    a = np.array([1.0, -2.0, 3.5, 0.25])
    b = np.array([0.5, -1.0, 4.0, 0.25])
    x, y = be.array(a), be.array(b)
    w = be.where(x > y, x, y)
    assert_metal(w, (4,))
    assert_values(w, np.where(a > b, a, b), exact=True)
    # scalar branches are materialized as MetalFloat64 (never float64-on-mps)
    ws = be.where(x > 0, x, 0.0)
    assert_metal(ws, (4,))
    assert_values(ws, np.where(a > 0, a, 0.0), exact=True)
    assert_values(be.where(x > 0, -1.0, x), np.where(a > 0, -1.0, a), exact=True)
    assert be.where(True, 1, 2) == 1
    assert_values(be.maximum(x, y), np.maximum(a, b), exact=True)
    assert_values(be.minimum(x, 0.0), np.minimum(a, 0.0), exact=True)
    c = be.isclose(x, y, rtol=0, atol=0.6)
    assert c.dtype == torch.bool
    assert np.array_equal(be.to_numpy(c), np.isclose(a, b, rtol=0, atol=0.6))
    assert be.allclose(x, x)
    assert not be.allclose(x, y)
    assert be.allclose(x, be.to_numpy(x) * (1 + 1e-6), rtol=1e-5)


@other_lanes
def test_reductions_and_mean(mps_backend):
    a = np.linspace(-1, 2, BIG).reshape(-1)
    x = be.array(a)
    s = be.sum(x)
    assert_metal(s, ())
    assert float(s) == pytest.approx(a.sum(), rel=1e-13)
    m = be.mean(x)
    assert float(m) == pytest.approx(a.mean(), rel=1e-13)
    a2 = a[: (a.size // 3) * 3].reshape(3, -1)
    x2 = be.array(a2)
    assert_values(be.sum(x2, axis=1), a2.sum(axis=1), rtol=1e-13)
    assert_values(be.mean(x2, axis=0), a2.mean(axis=0), rtol=1e-13)
    n = be.array([1.0, np.nan, 3.0])
    assert float(be.mean(n)) == pytest.approx(2.0)
    assert be.max(x) == pytest.approx(a.max()) and be.min(x) == pytest.approx(a.min())
    assert be.all(x > -2) and not be.any(x > 3)
    assert float(be.nanmax(n)) == 3.0


@other_lanes
def test_stack_concatenate_meshgrid(mps_backend):
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    s = be.stack([be.array(a), be.array(b)])
    assert_metal(s, (2, 2))
    assert_values(s, expected(s, np.stack([a, b])), exact=True)
    s1 = be.stack([be.array(a), b], axis=1)  # cast promotes the ndarray
    assert_values(s1, np.stack([a, b], axis=1), exact=True)
    c = be.concatenate([be.array(a), be.array(b)])
    assert_values(c, np.concatenate([a, b]), exact=True)
    X, Y = be.meshgrid(be.array(a), be.array([5.0, 6.0, 7.0]))
    nX, nY = np.meshgrid(a, [5.0, 6.0, 7.0])
    assert_metal(X, nX.shape)
    assert_values(X, nX, exact=True)
    assert_values(Y, nY, exact=True)
    big = be.linspace(0, 1, BIG)
    cc = be.concatenate([big, big])
    assert_values(cc, expected(cc, np.tile(np.linspace(0, 1, BIG), 2)), exact=True)


@other_lanes
def test_arithmetic_matches_numpy(mps_backend):
    a = np.linspace(0.5, 3.0, BIG)
    x = be.array(a)
    # No cancellation in the chain: relative error stays a few u^2 per op.
    y = be.sqrt(x * x + 1.0) / 3.0 + be.exp(-x)
    ref = np.sqrt(a * a + 1.0) / 3.0 + np.exp(-a)
    assert_metal(y, (BIG,))
    assert_values(y, ref, rtol=1e-13)
    small = be.array(a[:5])
    assert_values(small**2 + 1, a[:5] ** 2 + 1, rtol=1e-13)


# ---------------------------------------------------------------------------
# random
# ---------------------------------------------------------------------------
def test_random_functions_shapes_and_seeds(mps_backend):
    u = be.random_uniform(0.0, 1.0, size=(3, 4))
    assert_metal(u, (3, 4))
    v = be.to_numpy(u)
    assert v.dtype == np.float64 and np.all((v >= 0) & (v < 1))
    n = be.random_normal(1.0, 2.0, size=(BIG,))
    assert_metal(n, (BIG,))
    r = be.rand(2, 3)
    assert_metal(r, (2, 3))
    assert_metal(be.rand(), (1,))
    s = be.sobol_sampler(dim=3, num_samples=10, scramble=True, seed=7)
    assert_metal(s, (10, 3))
    assert np.all((be.to_numpy(s) >= 0) & (be.to_numpy(s) < 1))
    # seeds reproduce the CPU float64 stream exactly
    g1 = be.default_rng(123)
    g2 = be.default_rng(123)
    u1 = be.random_uniform(-1.0, 1.0, size=8, generator=g1)
    a1 = be.to_numpy(u1)
    a2 = be.to_numpy(be.random_uniform(-1.0, 1.0, size=8, generator=g2))
    assert np.array_equal(a1, a2)
    ref = torch.empty(8, dtype=torch.float64).uniform_(
        -1.0, 1.0, generator=torch.Generator().manual_seed(123)
    )
    assert np.array_equal(a1, expected(u1, ref.numpy()))
    b1 = be.to_numpy(be.random_normal(0.0, 1.0, size=5, generator=be.default_rng(9)))
    b2 = be.to_numpy(be.random_normal(0.0, 1.0, size=5, generator=be.default_rng(9)))
    assert np.array_equal(b1, b2)
    assert np.array_equal(
        be.to_numpy(be.sobol_sampler(2, 6, seed=3)),
        be.to_numpy(be.sobol_sampler(2, 6, seed=3)),
    )
    assert be.metal_stats().get("cpu_fallback:random_uniform") is None  # not a fallback


@other_lanes
def test_erfinv_matches_numpy_backend(mps_backend):
    a = np.linspace(-0.9, 0.9, 19)
    ref = np_value(lambda: be.erfinv(a))
    got = be.erfinv(be.array(a))
    assert_metal(got, (19,))
    assert_values(got, ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# interpolation (dispatch path) / polynomials / histograms (CPU fallbacks, counted)
# ---------------------------------------------------------------------------
def test_interp_matches_numpy_backend(mps_backend):
    xp = np.linspace(0.4, 0.8, 9)
    fp = 1.5 + 0.01 * np.sin(xp * 20)
    x = np.array([0.45, 0.5, 0.62, 0.79])
    ref = np_value(lambda: be.interp(x, xp, fp))
    got = be.interp(be.array(x), be.array(xp), be.array(fp))
    assert_metal(got, (4,))
    assert_values(got, ref, rtol=1e-13)
    # NumPy inputs are accepted too; interp runs on the dispatch path (no fallback)
    got2 = be.interp(x, xp, fp)
    assert_values(got2, ref, rtol=1e-13)
    assert not any(k.startswith("cpu_fallback:") for k in be.metal_stats())


def test_polyfit_matches_numpy_backend(mps_backend):
    x = np.linspace(-1, 1, 25)
    y = 0.5 - 1.25 * x + 2.0 * x**2
    ref = np_value(lambda: be.polyfit(x, y, 2))
    got = be.polyfit(be.array(x), be.array(y), 2)
    assert_metal(got, (3,))
    assert_values(got, ref, rtol=1e-9)
    assert be.metal_stats().get("cpu_fallback:polyfit", 0) == 1


@other_lanes
def test_polyval_on_dispatch_path(mps_backend):
    coeffs = [2.0, -1.25, 0.5]
    x = np.linspace(-1, 1, 9)
    ref = np_value(lambda: be.polyval(coeffs, x))
    got = be.polyval(coeffs, be.array(x))
    assert_metal(got, (9,))
    assert_values(got, ref, rtol=1e-13)
    assert_values(be.polyval(be.array(coeffs), be.array(x)), ref, rtol=1e-13)
    assert "cpu_fallback:polyval" not in be.metal_stats()
    # differentiable w.r.t. the coefficients (material dispersion path)
    be.grad_mode.enable()
    c = be.array(coeffs)
    be.polyval(c, be.array(x)).sum().backward()
    assert_values(c.grad, [np.sum(x**2), np.sum(x), 9.0], rtol=1e-13)


def test_histogram2d_matches_numpy(mps_backend):
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 500)
    y = rng.uniform(-1, 1, 500)
    xe = np.linspace(-1, 1, 6)
    ye = np.linspace(-1, 1, 5)
    ref, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    h, gx, gy = be.histogram2d(
        be.array(x), be.array(y), bins=[be.array(xe), be.array(ye)]
    )
    assert_metal(h, ref.shape)
    assert_values(h, ref, exact=True)
    assert_values(gx, expected(gx, xe), exact=True)
    assert_values(gy, expected(gy, ye), exact=True)
    w = rng.uniform(0, 1, 500)
    refw, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=w)
    hw, _, _ = be.histogram2d(
        be.array(x), be.array(y), bins=[be.array(xe), be.array(ye)], weights=be.array(w)
    )
    assert_values(hw, refw, rtol=1e-13)
    assert be.metal_stats().get("cpu_fallback:histogram2d", 0) == 2


def test_histogram_matches_numpy(mps_backend):
    x = np.linspace(0, 1, 101) ** 2
    counts, edges = be.histogram(be.array(x), bins=7)
    ref_counts, ref_edges = np.histogram(x, bins=7)
    assert_values(counts, ref_counts, exact=True)
    assert_values(edges, ref_edges)
    assert be.metal_stats().get("cpu_fallback:histogram", 0) == 1


@other_lanes
def test_fftconvolve_and_vectorize(mps_backend):
    a = np.array([1.0, 2.0, 3.0, 4.0])
    k = np.array([0.5, 0.25])
    ref = np_value(lambda: be.fftconvolve(a, k, mode="full"))
    got = be.fftconvolve(be.array(a), be.array(k), mode="full")
    assert_metal(got, ref.shape)
    assert_values(got, ref, rtol=1e-12)
    assert be.metal_stats().get("cpu_fallback:fftconvolve", 0) == 1
    f = be.vectorize(lambda v: float(v) * 2.0)(be.array([1.0, 2.0]))
    assert_metal(f, (2,))
    assert_values(f, [2.0, 4.0], exact=True)


def test_pad_constant_encodes_fill_value(mps_backend):
    x = be.ones((2, 2))
    p = be.pad(x, ((1, 1), (1, 1)))
    assert_metal(p, (4, 4))
    assert_values(p, np.pad(np.ones((2, 2)), 1), exact=True)
    q = be.pad(be.zeros((BIG, 2)), ((0, 1), (2, 0)), constant_values=0.1)
    ref = np.pad(np.zeros((BIG, 2)), ((0, 1), (2, 0)), constant_values=0.1)
    assert_values(q, expected(q, ref), exact=True)
    assert "cpu_fallback:constant_pad_nd" not in be.metal_stats()


def test_grid_sample_falls_back_to_cpu(mps_backend):
    data = np.zeros((1, 1, 4, 4))
    data[0, 0, 1, 1] = 1.0
    grid = np.zeros((1, 1, 1, 2))
    ref = np_value(lambda: be.grid_sample(data, grid, align_corners=False))
    got = be.grid_sample(be.array(data), be.array(grid), align_corners=False)
    assert_metal(got, (1, 1, 1, 1))
    assert_values(got, np.asarray(ref), rtol=1e-13)
    assert be.metal_stats().get("cpu_fallback:grid_sample", 0) == 1


@other_lanes
def test_get_bilinear_weights(mps_backend):
    coords = be.array([[0.5, 0.5], [2.0, 2.0]])
    edges = (be.array([0.0, 1.0, 2.0]), be.array([0.0, 1.0, 2.0]))
    idx, w = be.get_bilinear_weights(coords, edges)
    assert idx.shape == (2, 4, 2) and idx.dtype == torch.int64
    assert_metal(w, (2, 4))
    # oracle: the same torch code on CPU float64
    be.set_device("cpu")
    try:
        ref_idx, ref_w = be.get_bilinear_weights(
            be.array([[0.5, 0.5], [2.0, 2.0]]),
            (be.array([0.0, 1.0, 2.0]), be.array([0.0, 1.0, 2.0])),
        )
    finally:
        be.set_device("mps")
    assert np.array_equal(be.to_numpy(idx), ref_idx.numpy())
    assert_values(w, ref_w.numpy(), rtol=1e-12)


# ---------------------------------------------------------------------------
# linalg / complex boundary
# ---------------------------------------------------------------------------
@other_lanes
def test_matmul_with_mixed_operands(mps_backend):
    a = np.arange(6.0).reshape(2, 3)
    b = np.arange(12.0).reshape(3, 4)
    x = be.array(a)
    assert_values(be.matmul(x, be.array(b)), a @ b, rtol=1e-13)
    # plain float32 mps operand is promoted exactly
    f32 = torch.tensor(b, dtype=torch.float32, device="mps")
    assert_values(be.matmul(x, f32), a @ b, rtol=1e-13)
    assert_values(
        be.batched_chain_matmul3(x, be.array(b), be.eye(4)), a @ b, rtol=1e-13
    )


def test_to_complex_lives_on_cpu(mps_backend):
    x = be.array([1.0, -2.0])
    c = be.to_complex(x)
    assert c.dtype == torch.complex128 and c.device.type == "cpu"
    np.testing.assert_array_equal(c.numpy(), np.array([1.0, -2.0], dtype=np.complex128))
    p = be.array(np.eye(3)[None].repeat(2, axis=0))
    E = be.array(np.ones((2, 3)))
    out = be.mult_p_E(p, E)
    assert out.dtype == torch.complex128 and out.shape == (2, 3)
    np.testing.assert_array_equal(out.numpy(), np.ones((2, 3), dtype=np.complex128))


def test_conic_epsilon_uses_emulated_precision(mps_backend):
    from optiland.backend.torch_backend.conic import _epsilon
    from optiland.utils import machine_eps

    x = be.array([1.0])
    assert x.machine_eps == 2.0**-48
    assert _epsilon(x) == 2.0**-48
    assert machine_eps(x) == 2.0**-48
    be.set_metal_mode("sf64")
    try:
        assert _epsilon(be.array([1.0])) == 2.0**-53
    finally:
        be.set_metal_mode("df64")
    assert (
        _epsilon(torch.zeros(1, dtype=torch.float64)) == torch.finfo(torch.float64).eps
    )


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["df64", "sf64"])
def test_cooke_triplet_trace_matches_numpy(mps_backend, mode):
    from optiland.samples.objectives import CookeTriplet

    def trace():
        lens = CookeTriplet()
        rays = lens.trace(Hx=0, Hy=1, wavelength=0.55, num_rays=16)
        return tuple(
            be.to_numpy(getattr(rays, k)) for k in ("x", "y", "z", "L", "M", "N")
        )

    ref = np_value(trace)
    be.grad_mode.enable()  # the torch-mps fixture in tests/conftest.py enables it
    be.set_metal_mode(mode)
    try:
        got = trace()
    finally:
        be.set_metal_mode("df64")
    for r, g, name in zip(ref, got, ("x", "y", "z", "L", "M", "N"), strict=True):
        assert g.shape == r.shape, name
        worst = np.nanmax(np.abs(g - r))
        assert worst < 1e-11, f"{name}: max |delta| = {worst:.3e}"
    # Nothing on the trace path falls back to the CPU (material-table ``interp``
    # runs on the dispatch path too).
    fallbacks = {k for k in be.metal_stats() if k.startswith("cpu_fallback:")}
    assert not fallbacks, fallbacks
