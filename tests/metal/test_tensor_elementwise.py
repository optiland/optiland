"""Dispatch contract tests for the MetalFloat64 elementwise handlers.

Oracle: torch CPU float64 evaluated on the *decoded* inputs. df64 arithmetic
chains must agree to ``rtol = 1e-13`` (a few units of u^2 = 2^-48 per op);
structural results, comparisons and rounding ops must be exact; sf64
arithmetic must be bit-exact. Autograd through the handlers is compared to
CPU float64 autograd at ``rtol = 1e-12``.

Ops whose kernel is absent from the compiled library (hyperbolic / special
functions until ``df64_math_special.h`` lands, every transcendental in sf64)
must still be correct through the counted CPU fallback.

Composite autograd expressions whose derivative formulas need handlers from
the structural / creation lanes (``where``, ``masked_fill``, ``zeros_like``,
``empty_like``) or float64 factory tensors created *inside* autograd
(``at::scalar_tensor`` on mps) are skipped with an explicit reason until those
land; everything in this lane's own scope runs unconditionally.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")


import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")

if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
    pytest.skip("torch MPS (Metal) is not available", allow_module_level=True)

from optiland.backend.torch_backend.metal import encode, ops_elementwise  # noqa: E402
from optiland.backend.torch_backend.metal import tensor as mt  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import (  # noqa: E402
    MetalFloat64,
    aten,
    handler_for,
    library,
    wrap,
)

F = torch.nn.functional
RTOL = 1e-13
ATOL_TRANS = 1e-14  # absolute floor for transcendental ops near their zeros
GRAD_RTOL = 1e-12
SHAPE = (3, 4)

TRANSCENDENTAL_UNARY = (
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "erfc",
    "erfinv",
    "lgamma",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rt(a) -> np.ndarray:
    """Round ``a`` through the df64 representation (48-bit significand).

    Test data goes through this so that the MetalFloat64 operand and the CPU
    float64 reference hold *identical* values; otherwise exact ops (minimum,
    copysign, ...) would be compared against values that differ by ~2^-49.
    """
    a = np.asarray(a, dtype=np.float64)
    hi, lo = encode.encode_df64(a)
    return np.asarray(encode.decode_df64(hi, lo), dtype=np.float64).reshape(a.shape)


def U(rng, lo: float, hi: float, shape=SHAPE) -> np.ndarray:
    """Uniform random data, rounded through df64 (see :func:`rt`)."""
    return rt(rng.uniform(lo, hi, shape))


def mk(a, mode: str = "df64", requires_grad: bool = False) -> MetalFloat64:
    return MetalFloat64.from_numpy(
        np.asarray(a, dtype=np.float64), mode, requires_grad=requires_grad
    )


def sf64_library():
    """The sf64 library, or a skip when its kernel headers do not build yet."""
    try:
        return library("sf64")
    except Exception as e:  # noqa: BLE001 - the kernel lane's build is not ours
        pytest.skip(f"sf64 library does not build: {str(e)[:200]}")


def cpu(a) -> torch.Tensor:
    return torch.tensor(np.asarray(a, dtype=np.float64))


def transposed(x: MetalFloat64) -> MetalFloat64:
    """A non-contiguous view of ``x`` built on the component tensors."""
    return wrap(tuple(c.t() for c in x.components), x.mode)


def strided(x: MetalFloat64) -> MetalFloat64:
    """Every other column of ``x`` as a non-contiguous view."""
    return wrap(tuple(c[:, ::2] for c in x.components), x.mode)


def decode(got) -> np.ndarray:
    if isinstance(got, MetalFloat64):
        return got.to_numpy()
    assert isinstance(got, torch.Tensor)
    return got.detach().cpu().numpy()


def check(
    got, want: torch.Tensor, rtol: float = RTOL, atol: float = 0.0, exact: bool = False
) -> None:
    """Compare a MetalFloat64 / bool result against the CPU reference."""
    assert isinstance(got, torch.Tensor)
    assert got.device.type == "mps"
    if want.dtype == torch.bool:
        assert not isinstance(got, MetalFloat64)
        assert got.dtype == torch.bool
        exact = True
    else:
        assert isinstance(got, MetalFloat64), type(got)
        assert got.dtype == torch.float64
    assert tuple(got.shape) == tuple(want.shape)
    g, w = decode(got), want.detach().numpy()
    if exact:
        assert np.array_equal(g, w, equal_nan=True), (g, w)
        if w.dtype != bool:
            assert np.array_equal(np.signbit(g), np.signbit(w)), "sign of zero differs"
    else:
        np.testing.assert_allclose(g, w, rtol=rtol, atol=atol, equal_nan=True)


def needs(*ops) -> None:
    """Skip when a handler registered by another lane is not present yet."""
    missing = [str(op) for op in ops if handler_for(op) is None]
    if missing:
        pytest.skip(f"needs handlers from another lane: {missing}")


def needs_factory_intercept() -> None:
    """Skip when float64 factories on mps (autograd formulas) are not intercepted."""
    try:
        torch.zeros((), dtype=torch.float64, device="mps")
    except (TypeError, RuntimeError):
        pytest.skip(
            "autograd formula creates float64 factory tensors on mps (nucleus gap)"
        )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260916)


@pytest.fixture(autouse=True)
def _gpu_resident(monkeypatch):
    """Force GPU residency (no host-resident small tensors) and clean counters."""
    if hasattr(mt, "HOST_THRESHOLD"):
        monkeypatch.setattr(mt, "HOST_THRESHOLD", -1)
    mt.reset_stats()
    yield
    mt.reset_stats()


def test_host_resident_operands_still_match(rng, monkeypatch) -> None:
    """Small host-resident tensors (nucleus dual residency) obey the same contract."""
    if not hasattr(mt, "HOST_THRESHOLD"):
        pytest.skip("nucleus has no host residency")
    monkeypatch.setattr(mt, "HOST_THRESHOLD", 256)
    a, b = U(rng, -5, 5, SHAPE), U(rng, 0.5, 5, SHAPE)
    x, y = MetalFloat64.from_numpy(a), MetalFloat64.from_numpy(b)
    # values only: the nucleus decides where host-resident results live
    np.testing.assert_allclose((x * y + 1.0).to_numpy(), a * b + 1.0, rtol=RTOL)
    np.testing.assert_allclose((torch.exp(x) / y).to_numpy(), np.exp(a) / b, rtol=RTOL)
    assert np.array_equal((x < y).cpu().numpy(), a < b)
    x.mul_(2.0)
    np.testing.assert_allclose(x.to_numpy(), a * 2.0, rtol=RTOL)
    big_a = U(rng, -1, 1, (40, 40))
    big = MetalFloat64.from_numpy(big_a)
    check(big + 1.0, cpu(big_a) + 1.0)
    check(big * 2.0, cpu(big_a) * 2.0)


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------
def _u(lo: float, hi: float):
    return lambda g, shape: U(g, lo, hi, shape)


UNARY_CASES = {
    # name: (torch fn, domain, exact)
    "neg": (torch.neg, _u(-5, 5), True),
    "abs": (torch.abs, _u(-5, 5), True),
    "sign": (torch.sign, _u(-5, 5), True),
    "sgn": (torch.sgn, _u(-5, 5), True),
    "floor": (torch.floor, _u(-5, 5), True),
    "ceil": (torch.ceil, _u(-5, 5), True),
    "trunc": (torch.trunc, _u(-5, 5), True),
    "round": (torch.round, _u(-5, 5), True),
    "frac": (torch.frac, _u(-5, 5), False),
    "sqrt": (torch.sqrt, _u(0.01, 100), False),
    "rsqrt": (torch.rsqrt, _u(0.01, 100), False),
    "reciprocal": (torch.reciprocal, _u(0.1, 10), False),
    "square": (torch.square, _u(-5, 5), False),
    "exp": (torch.exp, _u(-20, 20), False),
    "exp2": (torch.exp2, _u(-20, 20), False),
    "expm1": (torch.expm1, _u(-3, 3), False),
    "log": (torch.log, _u(0.01, 100), False),
    "log1p": (torch.log1p, _u(-0.9, 10), False),
    "log2": (torch.log2, _u(0.01, 100), False),
    "log10": (torch.log10, _u(0.01, 100), False),
    "sin": (torch.sin, _u(-3, 3), False),
    "cos": (torch.cos, _u(-3, 3), False),
    "tan": (torch.tan, _u(-1.4, 1.4), False),
    "asin": (torch.asin, _u(-0.99, 0.99), False),
    "acos": (torch.acos, _u(-0.99, 0.99), False),
    "atan": (torch.atan, _u(-10, 10), False),
    "sinh": (torch.sinh, _u(-3, 3), False),
    "cosh": (torch.cosh, _u(-3, 3), False),
    "tanh": (torch.tanh, _u(-3, 3), False),
    "asinh": (torch.asinh, _u(-3, 3), False),
    "acosh": (torch.acosh, _u(1.1, 10), False),
    "atanh": (torch.atanh, _u(-0.9, 0.9), False),
    "erf": (torch.erf, _u(-3, 3), False),
    "erfc": (torch.erfc, _u(-3, 3), False),
    "erfinv": (torch.erfinv, _u(-0.9, 0.9), False),
    "lgamma": (torch.lgamma, _u(2.5, 8), False),
    "deg2rad": (torch.deg2rad, _u(-360, 360), False),
    "rad2deg": (torch.rad2deg, _u(-7, 7), False),
    "sigmoid": (torch.sigmoid, _u(-10, 10), False),
    "softplus": (F.softplus, _u(-30, 30), False),
    "relu": (torch.relu, _u(-3, 3), True),
}


@pytest.mark.parametrize("name", sorted(UNARY_CASES))
def test_unary_matches_cpu(name: str, rng) -> None:
    fn, domain, exact = UNARY_CASES[name]
    a = domain(rng, SHAPE)
    x = mk(a)
    want = fn(cpu(a))
    check(fn(x), want, atol=ATOL_TRANS, exact=exact)
    # non-contiguous view of the same values
    check(fn(transposed(x)), fn(cpu(a).t()), atol=ATOL_TRANS, exact=exact)
    check(fn(strided(x)), fn(cpu(a)[:, ::2]), atol=ATOL_TRANS, exact=exact)
    # 0-d
    check(fn(mk(a[0, 0])), fn(cpu(a[0, 0])), atol=ATOL_TRANS, exact=exact)


def test_unary_uses_gpu_or_counted_fallback(rng) -> None:
    lib = library("df64")
    x = mk(rng.uniform(0.1, 0.9, SHAPE))
    for name in ("sqrt", "exp", "sin", *TRANSCENDENTAL_UNARY):
        mt.reset_stats()
        getattr(torch, name)(x)
        stats = mt.stats()
        if lib.has(name):
            assert stats == {f"gpu:{name}": 1}, (name, stats)
        else:
            assert stats == {f"cpu_fallback:{name}:missing_kernel": 1}, (name, stats)


def test_round_decimals(rng) -> None:
    a = rt(
        np.concatenate(
            [rng.uniform(-100, 100, 20), [0.5, 1.5, 2.5, -0.5, -2.5, 0.125, 1e6 + 0.5]]
        )
    )
    x = mk(a)
    for decimals in (0, 1, 2, -1, -2):
        check(
            torch.round(x, decimals=decimals),
            torch.round(cpu(a), decimals=decimals),
            rtol=RTOL,
        )
    check(torch.round(x), torch.round(cpu(a)), exact=True)  # half to even


def test_unary_special_values() -> None:
    a = rt([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 1e-20, 1e15])
    x = mk(a)
    finite = rt([np.nan, 0.0, 1.5, -1.5, 2.5, -2.5, 1e-20, 1e30, -0.75])
    xf = mk(finite)
    with np.errstate(all="ignore"):
        check(torch.neg(x), torch.neg(cpu(a)), exact=True)
        check(torch.abs(x), torch.abs(cpu(a)), rtol=0.0)  # -0.0: test_abs_negative_zero
        for fn in (torch.floor, torch.ceil, torch.trunc, torch.round):
            check(fn(xf), fn(cpu(finite)), exact=True)  # +-inf/-0.0: test_rounding_inf
        for fn in (
            torch.sqrt,
            torch.exp,
            torch.log,
            torch.reciprocal,
            torch.rsqrt,
            torch.sign,
        ):
            check(fn(x), fn(cpu(a)), rtol=RTOL)
    # x**1 is a fresh tensor (no aliasing with x)
    y = x**1
    y.mul_(2.0)
    check(x, cpu(a), exact=True)


def test_abs_negative_zero() -> None:
    check(torch.abs(mk([-0.0, 0.0])), torch.abs(cpu([-0.0, 0.0])), exact=True)


@pytest.mark.parametrize("fn", [torch.floor, torch.ceil, torch.trunc, torch.round])
def test_rounding_inf_and_negative_zero(fn) -> None:
    a = np.array([np.inf, -np.inf, -0.0, -0.25])
    check(fn(mk(a)), fn(cpu(a)), exact=True)


def test_mul_signed_zero() -> None:
    a = np.array([-1.0, 1.0, -0.0])
    check(0.0 * mk(a), 0.0 * cpu(a), exact=True)


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------
BINARY_CASES = {
    # name: (fn, a-domain, b-domain, exact)
    "add": (torch.add, _u(-5, 5), _u(-5, 5), False),
    "sub": (torch.sub, _u(-5, 5), _u(-5, 5), False),
    "mul": (torch.mul, _u(-5, 5), _u(-5, 5), False),
    "div": (torch.div, _u(-5, 5), _u(0.5, 5), False),
    "true_divide": (torch.true_divide, _u(-5, 5), _u(0.5, 5), False),
    "floor_divide": (torch.floor_divide, _u(-50, 50), _u(0.5, 5), False),
    "pow": (torch.pow, _u(0.5, 4), _u(-3, 3), False),
    "atan2": (torch.atan2, _u(-5, 5), _u(-5, 5), False),
    "hypot": (torch.hypot, _u(-5, 5), _u(-5, 5), False),
    "fmod": (torch.fmod, _u(-50, 50), _u(0.5, 5), False),
    "remainder": (torch.remainder, _u(-50, 50), _u(-5, 5), False),
    "copysign": (torch.copysign, _u(-5, 5), _u(-5, 5), True),
    "minimum": (torch.minimum, _u(-5, 5), _u(-5, 5), True),
    "maximum": (torch.maximum, _u(-5, 5), _u(-5, 5), True),
    "fmin": (torch.fmin, _u(-5, 5), _u(-5, 5), True),
    "fmax": (torch.fmax, _u(-5, 5), _u(-5, 5), True),
    "max.other": (torch.max, _u(-5, 5), _u(-5, 5), True),
    "min.other": (torch.min, _u(-5, 5), _u(-5, 5), True),
    "eq": (torch.eq, _u(-5, 5), _u(-5, 5), True),
    "ne": (torch.ne, _u(-5, 5), _u(-5, 5), True),
    "lt": (torch.lt, _u(-5, 5), _u(-5, 5), True),
    "le": (torch.le, _u(-5, 5), _u(-5, 5), True),
    "gt": (torch.gt, _u(-5, 5), _u(-5, 5), True),
    "ge": (torch.ge, _u(-5, 5), _u(-5, 5), True),
    "logical_and": (torch.logical_and, _u(-5, 5), _u(-5, 5), True),
    "logical_or": (torch.logical_or, _u(-5, 5), _u(-5, 5), True),
    "logical_xor": (torch.logical_xor, _u(-5, 5), _u(-5, 5), True),
}

LAYOUTS = (
    "same",
    "broadcast",
    "scalar_right",
    "scalar_left",
    "zero_d",
    "noncontig",
    "zero_d_tensor",
)


def _operands(name: str, layout: str, rng):
    """Build (metal_a, metal_b, cpu_a, cpu_b) for one layout."""
    _fn, da, db, _exact = BINARY_CASES[name]
    a = da(rng, SHAPE)
    b = db(rng, SHAPE)
    if name in (
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "logical_and",
        "logical_or",
        "logical_xor",
    ):
        b[::2, ::2] = a[::2, ::2]  # some equal elements
        a[1, 1] = 0.0
        b[1, 2] = 0.0
    if layout == "same":
        return mk(a), mk(b), cpu(a), cpu(b)
    if layout == "broadcast":
        a31, b14 = a[:, :1], b[:1, :]
        return mk(a31), mk(b14), cpu(a31), cpu(b14)
    if layout == "scalar_right":
        s = float(b[0, 0])
        return mk(a), s, cpu(a), s
    if layout == "scalar_left":
        s = float(a[0, 0])
        return s, mk(b), s, cpu(b)
    if layout == "zero_d":
        return mk(a[0, 0]), mk(b[0, 0]), cpu(a[0, 0]), cpu(b[0, 0])
    if layout == "noncontig":
        return transposed(mk(a)), transposed(mk(b)), cpu(a).t(), cpu(b).t()
    if layout == "zero_d_tensor":
        return mk(a), mk(b[0, 0]), cpu(a), cpu(b[0, 0])
    raise AssertionError(layout)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("name", sorted(BINARY_CASES))
def test_binary_matches_cpu(name: str, layout: str, rng) -> None:
    fn, _da, _db, exact = BINARY_CASES[name]
    xa, xb, ca, cb = _operands(name, layout, rng)
    try:
        want = fn(ca, cb)
    except TypeError:
        pytest.skip(f"torch.{name} has no overload for layout {layout!r}")
    atol = 0.0
    if name in ("fmod", "remainder"):
        # the kernels' error is relative to the operands, not to the (smaller) remainder
        atol = RTOL * max(float(np.abs(np.asarray(v)).max()) for v in (ca, cb))
    check(fn(xa, xb), want, atol=atol, exact=exact)


def test_add_sub_alpha_and_rsub(rng) -> None:
    a, b = U(rng, -5, 5, SHAPE), U(rng, -5, 5, SHAPE)
    x, y = mk(a), mk(b)
    check(torch.add(x, y, alpha=2.5), torch.add(cpu(a), cpu(b), alpha=2.5))
    check(torch.sub(x, y, alpha=-0.1), torch.sub(cpu(a), cpu(b), alpha=-0.1))
    check(torch.add(x, 0.7, alpha=3), torch.add(cpu(a), 0.7, alpha=3))
    check(torch.sub(x, 2, alpha=0.3), torch.sub(cpu(a), 2, alpha=0.3))
    check(torch.rsub(x, 1.5), torch.rsub(cpu(a), 1.5))
    check(torch.rsub(x, y, alpha=2), torch.rsub(cpu(a), cpu(b), alpha=2))
    check(3.0 - x, 3.0 - cpu(a))
    check(3.0 / x, 3.0 / cpu(a))


def test_div_rounding_modes(rng) -> None:
    a, b = U(rng, -50, 50, SHAPE), U(rng, -5, 5, SHAPE)
    a[0, 0], b[0, 0] = -7.0, 2.0
    x, y = mk(a), mk(b)
    for mode in ("trunc", "floor"):
        check(
            torch.div(x, y, rounding_mode=mode),
            torch.div(cpu(a), cpu(b), rounding_mode=mode),
            exact=True,
        )
        check(
            torch.div(x, 2.0, rounding_mode=mode),
            torch.div(cpu(a), 2.0, rounding_mode=mode),
            exact=True,
        )
    check(torch.div(x, y, rounding_mode=None), torch.div(cpu(a), cpu(b)))
    check(x // y, cpu(a) // cpu(b), exact=True)
    with pytest.raises(ValueError):
        torch.div(x, y, rounding_mode="nearest")


def test_pow_exponent_fast_paths(rng) -> None:
    a = U(rng, -3, 3, SHAPE)
    a[0, 0] = 0.0
    x = mk(a)
    for e in (0, 1, 2, 3, -1, -2, 4, -3):
        with np.errstate(all="ignore"):
            check(x**e, cpu(a) ** e, rtol=RTOL)
    p = U(rng, 0.1, 5, SHAPE)
    xp = mk(p)
    for e in (0.5, 1.7, -0.5, 2.5):
        check(xp**e, cpu(p) ** e, rtol=RTOL)
    # negative base with a non-integer exponent: NaN like torch
    check(x**1.5, cpu(a) ** 1.5, rtol=RTOL)
    # scalar base
    check(2.0**x, 2.0 ** cpu(a), rtol=RTOL)
    check(torch.pow(10.0, x), torch.pow(10.0, cpu(a)), rtol=RTOL)
    # tensor exponent incl. negative bases with integer-valued exponents
    e = np.round(rng.uniform(-3, 3, SHAPE))
    check(x ** mk(e), cpu(a) ** cpu(e), rtol=RTOL)
    check(torch.float_power(xp, 1.7), torch.float_power(cpu(p), 1.7), rtol=RTOL)


def test_clamp_family(rng) -> None:
    a = U(rng, -5, 5, SHAPE)
    a[0, 0] = np.nan
    lo, hi = U(rng, -2, 0, SHAPE), U(rng, 0, 2, SHAPE)
    x, xlo, xhi = mk(a), mk(lo), mk(hi)
    c = cpu(a)
    check(torch.clamp(x, -1.0, 1.0), torch.clamp(c, -1.0, 1.0), exact=True)
    check(torch.clamp(x, min=-1.0), torch.clamp(c, min=-1.0), exact=True)
    check(torch.clamp(x, max=1.0), torch.clamp(c, max=1.0), exact=True)
    check(torch.clamp(x, xlo, xhi), torch.clamp(c, cpu(lo), cpu(hi)), exact=True)
    check(torch.clamp(x, min=xlo), torch.clamp(c, min=cpu(lo)), exact=True)
    check(torch.clamp(x, max=xhi), torch.clamp(c, max=cpu(hi)), exact=True)
    check(torch.clamp_min(x, 0.25), torch.clamp_min(c, 0.25), exact=True)
    check(torch.clamp_max(x, 0.25), torch.clamp_max(c, 0.25), exact=True)
    check(torch.clamp_min(x, xlo), torch.clamp_min(c, cpu(lo)), exact=True)
    check(torch.clamp(x, 1.0, -1.0), torch.clamp(c, 1.0, -1.0), exact=True)  # min > max
    with pytest.raises(RuntimeError):
        torch.clamp(x)


def test_ternary_ops(rng) -> None:
    a, b, w = U(rng, -5, 5, SHAPE), U(rng, -5, 5, SHAPE), U(rng, 0, 1, SHAPE)
    d = U(rng, 0.5, 5, SHAPE)
    x, y, xw, xd = mk(a), mk(b), mk(w), mk(d)
    ca, cb, cw, cd = cpu(a), cpu(b), cpu(w), cpu(d)
    check(torch.lerp(x, y, 0.3), torch.lerp(ca, cb, 0.3))
    check(torch.lerp(x, y, xw), torch.lerp(ca, cb, cw))
    check(torch.lerp(x, mk(b[:1]), xw), torch.lerp(ca, cb[:1], cw))  # broadcast
    check(torch.addcmul(x, y, xw), torch.addcmul(ca, cb, cw))
    check(torch.addcmul(x, y, xw, value=-2.5), torch.addcmul(ca, cb, cw, value=-2.5))
    check(torch.addcdiv(x, y, xd), torch.addcdiv(ca, cb, cd))
    check(torch.addcdiv(x, y, xd, value=0.5), torch.addcdiv(ca, cb, cd, value=0.5))


# ---------------------------------------------------------------------------
# Predicates, special values, logical ops
# ---------------------------------------------------------------------------
SPECIAL = rt([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 1e-10, -1e15, 0.5])


def test_predicates_on_special_values() -> None:
    x = mk(SPECIAL)
    c = cpu(SPECIAL)
    for fn in (
        torch.isnan,
        torch.isinf,
        torch.isfinite,
        torch.isposinf,
        torch.isneginf,
        torch.signbit,
        torch.logical_not,
    ):
        got = fn(x)
        check(got, fn(c))
    check(
        torch.signbit(transposed(mk(SPECIAL.reshape(2, 5)))),
        torch.signbit(c.reshape(2, 5).t()),
    )


def test_comparisons_on_special_values() -> None:
    x = mk(SPECIAL)
    c = cpu(SPECIAL)
    for fn in (torch.eq, torch.ne, torch.lt, torch.le, torch.gt, torch.ge):
        check(fn(x, x), fn(c, c))
        check(fn(x, 0.0), fn(c, 0.0))
        check(fn(x, float("nan")), fn(c, float("nan")))
        check(fn(x, mk(SPECIAL[::-1].copy())), fn(c, cpu(SPECIAL[::-1].copy())))


def test_arithmetic_special_values() -> None:
    x = mk(SPECIAL)
    c = cpu(SPECIAL)
    with np.errstate(all="ignore"):
        for fn in (
            torch.add,
            torch.sub,
            torch.mul,
            torch.div,
            torch.minimum,
            torch.maximum,
            torch.fmin,
            torch.fmax,
            torch.copysign,
        ):
            check(fn(x, x), fn(c, c), rtol=RTOL)
            check(
                fn(x, mk(SPECIAL[::-1].copy())),
                fn(c, cpu(SPECIAL[::-1].copy())),
                rtol=RTOL,
            )
        check(x / 0.0, c / 0.0, exact=True)
        check(0.0 * x, 0.0 * c, rtol=0.0)  # sign of zero: test_mul_signed_zero
        check(
            torch.minimum(mk([np.nan, 1.0]), mk([1.0, np.nan])),
            cpu([np.nan, np.nan]),
            exact=True,
        )
        check(
            torch.fmin(mk([np.nan, 1.0]), mk([1.0, np.nan])),
            cpu([1.0, 1.0]),
            exact=True,
        )
        check(
            torch.copysign(mk([1.0, 2.0, 0.0]), mk([-0.0, np.nan, -1.0])),
            cpu([-1.0, 2.0, -0.0]),
            exact=True,
        )


def test_logical_ops_with_plain_and_scalar_operands(rng) -> None:
    a = U(rng, -1, 1, SHAPE)
    a[0, :2] = 0.0
    x = mk(a)
    c = cpu(a)
    mask = torch.tensor(rng.uniform(-1, 1, SHAPE) > 0, device="mps")
    f32 = torch.tensor(a > 0.3, device="mps").to(torch.float32)
    check(torch.logical_and(x, mask), torch.logical_and(c, mask.cpu()))
    check(torch.logical_or(mask, x), torch.logical_or(mask.cpu(), c))
    check(torch.logical_xor(x, f32), torch.logical_xor(c, f32.cpu()))
    check(torch.logical_and(x, mk(a[:1])), torch.logical_and(c, cpu(a[:1])))


def test_nan_to_num() -> None:
    x = mk(SPECIAL)
    c = cpu(SPECIAL)
    big = float(
        np.finfo(np.float32).max
    )  # df64 keeps float32 range: default posinf is float32 max
    want = torch.nan_to_num(c, nan=0.0, posinf=big, neginf=-big)
    check(torch.nan_to_num(x), want, exact=True)
    check(
        torch.nan_to_num(x, nan=-1.0, posinf=100.0, neginf=-200.0),
        torch.nan_to_num(c, nan=-1.0, posinf=100.0, neginf=-200.0),
        exact=True,
    )
    z = x.clone()
    z.nan_to_num_(nan=7.0)
    check(z, torch.nan_to_num(c, nan=7.0, posinf=big, neginf=-big), exact=True)


# ---------------------------------------------------------------------------
# Scalars and mixed dtypes must never pass through float32
# ---------------------------------------------------------------------------
def test_scalar_constants_are_exact() -> None:
    x = mk([1.0, 1e6, 0.1])
    c = cpu([1.0, 1e6, 0.1])
    check(x + 0.1, c + 0.1)
    check(x * 0.1, c * 0.1)
    check(0.1 - x, 0.1 - c)
    check(x / 3.0, c / 3.0)
    check(torch.eq(x, 0.1), torch.eq(c, 0.1))
    check(torch.lt(x, 1.0000000001), torch.lt(c, 1.0000000001))
    # wrapped 0-d CPU tensors arrive as scalars too
    check(x * torch.tensor(0.1, dtype=torch.float64), c * 0.1)
    check(torch.tensor(0.1, dtype=torch.float64) + x, c + 0.1)
    # Python ints and bools
    check(x + 3, c + 3)
    check(x * True, c * True)
    check(x**2, c**2)


def test_mixed_plain_mps_operands_promote_exactly(rng) -> None:
    a = U(rng, -5, 5, SHAPE)
    x = mk(a)
    c = cpu(a)
    f32 = torch.tensor(rng.uniform(-5, 5, SHAPE), dtype=torch.float32, device="mps")
    i64 = torch.arange(SHAPE[1], device="mps") * 1000003
    i32 = torch.arange(SHAPE[1], device="mps", dtype=torch.int32)
    b = torch.tensor(rng.uniform(-1, 1, SHAPE) > 0, device="mps")
    check(x * f32, c * f32.cpu().to(torch.float64))
    check(f32 - x, f32.cpu().to(torch.float64) - c)
    check(x + i64, c + i64.cpu())
    check(i32 * x, i32.cpu() * c)
    check(x * b, c * b.cpu())
    check(x < f32, c < f32.cpu().to(torch.float64))
    check(
        torch.maximum(x, f32), torch.maximum(c, f32.cpu().to(torch.float64)), exact=True
    )
    check(x * b.to(torch.float32), c * b.cpu().to(torch.float64))


@pytest.mark.parametrize("mode", ["df64", "sf64"])
def test_alpha_with_plain_operand_keeps_mode(mode: str, rng) -> None:
    """``alpha`` scaling of a plain float32 operand happens in the op's own mode."""
    if mode == "sf64":
        sf64_library()
    a = U(rng, -2, 2, SHAPE)
    f32 = torch.tensor(rng.uniform(-2, 2, SHAPE), dtype=torch.float32, device="mps")
    x = mk(a, mode)
    c, cf = cpu(a), f32.cpu().to(torch.float64)
    exact = mode == "sf64"
    got = torch.add(x, f32, alpha=2.5)
    assert got.mode == mode
    check(got, torch.add(c, cf, alpha=2.5), exact=exact)
    check(torch.sub(x, f32, alpha=-3), torch.sub(c, cf, alpha=-3), exact=exact)
    check(torch.rsub(x, f32, alpha=2), torch.rsub(c, cf, alpha=2), exact=exact)
    y = x.clone()
    y.add_(f32, alpha=0.5)
    check(y, torch.add(c, cf, alpha=0.5), exact=exact)


def test_zero_size_tensors() -> None:
    x = mk(np.zeros((0, 3)))
    check(x + 1.0, cpu(np.zeros((0, 3))) + 1.0, exact=True)
    check(torch.exp(x), cpu(np.zeros((0, 3))), exact=True)
    check(x < 1.0, cpu(np.zeros((0, 3))) < 1.0)


# ---------------------------------------------------------------------------
# In-place and out= variants
# ---------------------------------------------------------------------------
def test_inplace_contiguous(rng) -> None:
    a, b = U(rng, -5, 5, SHAPE), U(rng, 0.5, 5, SHAPE)
    x, y = mk(a), mk(b)
    ca, cb = cpu(a), cpu(b)
    r = x.add_(y, alpha=0.5)
    ca.add_(cb, alpha=0.5)
    assert r is x
    check(x, ca)
    for name, arg in (
        ("mul_", 1.5),
        ("div_", y),
        ("sub_", 2.0),
        ("pow_", 2),
        ("clamp_", (-3.0, 3.0)),
        ("sqrt_", ()),
        ("exp_", ()),
        ("sinh_", ()),
        ("abs_", ()),
        ("neg_", ()),
        ("round_", ()),
        ("nan_to_num_", ()),
    ):
        args = arg if isinstance(arg, tuple) else (arg,)
        cargs = tuple(cb if isinstance(v, MetalFloat64) else v for v in args)
        if name in ("sqrt_",):
            x.abs_()
            ca.abs_()
        r = getattr(x, name)(*args)
        getattr(ca, name)(*cargs)
        assert r is x, name
        check(x, ca)


def test_inplace_writes_through_noncontiguous_views(rng) -> None:
    a = U(rng, -5, 5, SHAPE)
    base = mk(a)
    view = transposed(base)
    view.mul_(2.0)
    check(base, cpu(a) * 2.0)
    view2 = strided(base)
    view2.add_(mk(np.ones((3, 2))))
    ca = cpu(a) * 2.0
    ca[:, ::2] += 1.0
    check(base, ca)
    # broadcasting other into self
    base.sub_(mk(a[:1]))
    ca.sub_(cpu(a[:1]))
    check(base, ca)


def test_inplace_fallback_op_writes_back(rng) -> None:
    a = U(rng, -2, 2, SHAPE)
    x = mk(a)
    r = x.tanh_()
    assert r is x
    check(x, torch.tanh(cpu(a)), atol=ATOL_TRANS)


def test_out_variants(rng) -> None:
    a, b = U(rng, -5, 5, SHAPE), U(rng, 0.5, 5, SHAPE)
    x, y = mk(a), mk(b)
    out = mk(np.zeros(SHAPE))
    r = torch.add(x, y, out=out)
    assert r is out
    check(out, cpu(a) + cpu(b))
    r = torch.exp(x, out=out)
    assert r is out
    check(out, torch.exp(cpu(a)))
    r = torch.pow(x, 2, out=out)
    assert r is out
    check(out, cpu(a) ** 2)
    ob = torch.empty(SHAPE, dtype=torch.bool, device="mps")
    r = torch.lt(x, y, out=ob)
    assert r is ob
    check(ob, cpu(a) < cpu(b))
    ob0 = torch.empty(0, dtype=torch.bool, device="mps")
    torch.eq(x, y, out=ob0)  # empty out is resized
    check(ob0, torch.eq(cpu(a), cpu(b)))
    # writing through a non-contiguous out
    base = mk(np.zeros(SHAPE))
    torch.mul(transposed(x), 3.0, out=transposed(base))
    check(base, cpu(a) * 3.0)
    # a plain floating out receives the same-category downcast (torch semantics)
    f32 = torch.empty(SHAPE, dtype=torch.float32, device="mps")
    r = torch.add(x, y, out=f32)
    assert r is f32 and f32.dtype == torch.float32
    np.testing.assert_array_equal(f32.cpu().numpy(), (cpu(a) + cpu(b)).float().numpy())
    with pytest.raises(RuntimeError, match="can't be cast to the desired output"):
        torch.add(x, y, out=torch.empty(SHAPE, dtype=torch.int64, device="mps"))


# ---------------------------------------------------------------------------
# sf64 mode
# ---------------------------------------------------------------------------
def test_sf64_arithmetic_bit_exact(rng) -> None:
    sf64_library()
    a, b = U(rng, -5, 5, SHAPE), U(rng, 0.5, 5, SHAPE)
    x, y = mk(a, "sf64"), mk(b, "sf64")
    ca, cb = cpu(a), cpu(b)
    check(x + y, ca + cb, exact=True)
    check(x - y, ca - cb, exact=True)
    check(x * y, ca * cb, exact=True)
    check(x / y, ca / cb, exact=True)
    check(torch.sqrt(y), torch.sqrt(cb), exact=True)
    check(x + 0.1, ca + 0.1, exact=True)
    check(0.1 * x, 0.1 * ca, exact=True)
    check(x / 3.0, ca / 3.0, exact=True)
    check(x**2, ca**2, exact=True)
    check(x < y, ca < cb)
    check(torch.signbit(x), torch.signbit(ca))
    check(torch.fmod(x, y), torch.fmod(ca, cb), exact=True)
    z = x.clone()
    z.mul_(y)
    check(z, ca * cb, exact=True)
    with pytest.raises(TypeError):
        x + mk(b)  # mixing representations


def test_sf64_transcendental_route(rng) -> None:
    lib = sf64_library()
    a = U(rng, -2, 2, SHAPE)
    x = mk(a, "sf64")
    mt.reset_stats()
    got = torch.exp(x)
    assert got.mode == "sf64"
    check(got, torch.exp(cpu(a)))
    key = "gpu:exp" if lib.has("exp") else "cpu_fallback:exp:missing_kernel"
    assert mt.stats() == {key: 1}


# ---------------------------------------------------------------------------
# Fallback accounting
# ---------------------------------------------------------------------------
def test_missing_kernel_falls_back_and_counts(rng, monkeypatch) -> None:
    lib = library("df64")
    orig_has = lib.has
    monkeypatch.setattr(
        lib, "has", lambda op: False if op in ("exp", "atan2") else orig_has(op)
    )
    a, b = U(rng, -2, 2, SHAPE), U(rng, -2, 2, SHAPE)
    x, y = mk(a), mk(b)
    mt.reset_stats()
    check(torch.exp(x), torch.exp(cpu(a)))
    check(torch.atan2(x, y), torch.atan2(cpu(a), cpu(b)))
    check(torch.atan2(x, mk(b[:1])), torch.atan2(cpu(a), cpu(b[:1])))
    stats = mt.stats()
    assert stats["cpu_fallback:exp:missing_kernel"] == 1
    assert stats["cpu_fallback:atan2:missing_kernel"] == 2
    assert "gpu:exp" not in stats and "gpu:atan2" not in stats


def test_transcendentals_missing_from_library_are_counted(rng) -> None:
    lib = library("df64")
    a = U(rng, 0.1, 0.9, SHAPE)
    x = mk(a)
    for name in TRANSCENDENTAL_UNARY:
        mt.reset_stats()
        fn = getattr(torch, name)
        check(fn(x), fn(cpu(a)), atol=ATOL_TRANS)
        assert mt.stats() == {
            (
                f"gpu:{name}"
                if lib.has(name)
                else f"cpu_fallback:{name}:missing_kernel"
            ): 1
        }
    mt.reset_stats()
    check(torch.hypot(x, x), torch.hypot(cpu(a), cpu(a)))
    assert mt.stats() == {
        ("gpu:hypot" if lib.has("hypot") else "cpu_fallback:hypot:missing_kernel"): 1
    }


def test_strict_mode_raises_on_fallback(rng, monkeypatch) -> None:
    lib = library("df64")
    orig_has = lib.has
    monkeypatch.setattr(lib, "has", lambda op: op != "log" and orig_has(op))
    monkeypatch.setenv("OPTILAND_METAL_STRICT", "1")
    x = mk(rng.uniform(0.5, 2, SHAPE))
    with pytest.raises(mt.MetalFallbackError):
        torch.log(x)
    torch.sqrt(x)  # GPU path unaffected


def test_gpu_launches_are_counted(rng) -> None:
    x, y = mk(rng.uniform(-1, 1, SHAPE)), mk(rng.uniform(-1, 1, SHAPE))
    mt.reset_stats()
    _ = x + y
    _ = x * 2.0
    _ = x < y
    _ = torch.frac(x)
    assert mt.stats() == {
        "gpu:add": 1,
        "gpu:mul": 1,
        "gpu:lt": 1,
        "gpu:trunc": 1,
        "gpu:sub": 1,
    }


def test_unregistered_op_raises_not_implemented(rng) -> None:
    x = mk(rng.uniform(-1, 1, SHAPE))
    with pytest.raises(NotImplementedError):
        torch.special.bessel_j0(x)


def test_registry_covers_documented_packets() -> None:
    for packet_name in ops_elementwise.UNARY_KERNEL_OPS:
        assert handler_for(getattr(aten, packet_name).default) is not None, packet_name


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------
GRAD_EXPRS = {
    "sqrt(x^2+y^2)": (lambda x, y: torch.sqrt(x * x + y * y), (), False),
    "atan2*exp": (
        lambda x, y: torch.atan2(y, x) * torch.exp(-x / 3),
        (aten.masked_fill.Scalar,),
        False,
    ),
    "rational": (lambda x, y: (x**3 - 2 * x * y + 1) / (1 + y * y), (), False),
    "trig-log": (
        lambda x, y: torch.sin(x) * torch.cos(y) + torch.log(y) * torch.tan(x / 2),
        (),
        False,
    ),
    "sigmoid-softplus-lerp": (
        lambda x, y: torch.sigmoid(x) * F.softplus(y) + torch.lerp(x, y, 0.3),
        (),
        False,
    ),
    "exp2-log1p-expm1-rsqrt-pow": (
        lambda x, y: (
            torch.exp2(x) / torch.log1p(y) + torch.expm1(x) - torch.rsqrt(y) + y**1.7
        ),
        (),
        False,
    ),
    "fallback-ops": (
        lambda x, y: torch.hypot(x, y) * torch.erf(x) + torch.sinh(y) * torch.tanh(x),
        (),
        False,
    ),
    "addcmul-deg2rad-abs-square": (
        lambda x, y: (
            torch.addcmul(x, y, x, value=2)
            + torch.deg2rad(x)
            - torch.abs(y)
            + torch.square(x)
        ),
        (),
        False,
    ),
    "asin-atan-recip": (
        lambda x, y: (
            torch.asin(x / 3) * torch.atan(y) + torch.reciprocal(y) * torch.acos(x / 3)
        ),
        (),
        False,
    ),
    "alpha-rsub-div": (lambda x, y: torch.sub(x, y, alpha=2) * (5 - x) / y, (), False),
    "maximum": (lambda x, y: torch.maximum(x, y) * y, (aten.where.self,), False),
    "clamp-scalar": (
        lambda x, y: torch.clamp(x, -0.5, 0.5) * y,
        (aten.where.self,),
        True,
    ),
    "copysign": (lambda x, y: torch.copysign(x, y) * x, (aten.where.self,), True),
    "pow-tensor-exp": (lambda x, y: y**x, (aten.where.self,), True),
    "trunc-frac": (
        lambda x, y: (torch.trunc(x) + torch.frac(x)) * y,
        (aten.zeros_like.default,),
        False,
    ),
    "lerp-tensor-addcdiv": (
        lambda x, y: torch.lerp(x, y, x / 4) + torch.addcdiv(x, x, y, value=0.5),
        (aten.empty_like.default,),
        False,
    ),
    "relu": (lambda x, y: torch.relu(x) * y, (), False),
}


def _grad_pair(fn, a, b):
    xg, yg = mk(a, requires_grad=True), mk(b, requires_grad=True)
    cx, cy = cpu(a).requires_grad_(True), cpu(b).requires_grad_(True)
    f = fn(xg, yg)
    fc = fn(cx, cy)
    assert isinstance(f, MetalFloat64) and f.requires_grad
    check(f, fc.detach(), atol=ATOL_TRANS)
    g = torch.autograd.grad(f, [xg, yg], grad_outputs=mk(np.ones(a.shape)))
    gc = torch.autograd.grad(
        fc, [cx, cy], grad_outputs=torch.ones(a.shape, dtype=torch.float64)
    )
    return g, gc


@pytest.mark.parametrize("name", list(GRAD_EXPRS))
def test_autograd_matches_cpu(name: str, rng) -> None:
    fn, required, factory = GRAD_EXPRS[name]
    needs(*required)
    if factory:
        needs_factory_intercept()
    a, b = U(rng, -2, 2, SHAPE), U(rng, 0.5, 2, SHAPE)
    g, gc = _grad_pair(fn, a, b)
    for got, want in zip(g, gc, strict=True):
        scale = float(np.abs(want.numpy()).max())
        check(got, want, rtol=GRAD_RTOL, atol=GRAD_RTOL * scale)


def test_autograd_through_inplace_on_clone(rng) -> None:
    a, b = U(rng, -2, 2, SHAPE), U(rng, 0.5, 2, SHAPE)

    def fn(x, y):
        z = x.clone()
        z.mul_(3.0)
        z.add_(y, alpha=0.5)
        z.sub_(1.0)
        z.div_(y)
        return torch.sin(z) * z

    g, gc = _grad_pair(fn, a, b)
    for got, want in zip(g, gc, strict=True):
        scale = float(np.abs(want.numpy()).max())
        check(got, want, rtol=GRAD_RTOL, atol=GRAD_RTOL * scale)


def test_autograd_scalar_and_zero_d(rng) -> None:
    a = U(rng, -2, 2, SHAPE)
    s = 1.7

    def fn(x, y):
        return torch.exp(x * y) / (y + 2.0) + x**2 * y

    xg, yg = mk(a, requires_grad=True), mk(s, requires_grad=True)
    cx, cy = cpu(a).requires_grad_(True), cpu(s).requires_grad_(True)
    f, fc = fn(xg, yg), fn(cx, cy)
    check(f, fc.detach())
    (gx,) = torch.autograd.grad(f, [xg], grad_outputs=mk(np.ones(SHAPE)))
    (gcx,) = torch.autograd.grad(
        fc, [cx], grad_outputs=torch.ones(SHAPE, dtype=torch.float64)
    )
    check(gx, gcx, rtol=GRAD_RTOL, atol=GRAD_RTOL * float(np.abs(gcx.numpy()).max()))
