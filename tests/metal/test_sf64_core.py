"""Bit-exactness tests for ``sf64_core.h`` (software IEEE binary64 on Metal).

Every sf64 operation is compared bit-for-bit with an IEEE-754 binary64 reference:
NumPy float64 (hardware, correctly rounded) for the basic operations, exact
rational arithmetic (``fractions.Fraction``) for ``fma``, and explicit Python
models where NumPy's semantics differ from torch's (``sign``, integer casts).
NaN results compare equal to NaN regardless of payload; signed zeros are
compared by bit pattern (i.e. by sign bit).

Inputs are passed to the GPU as the int64 bit patterns of float64 arrays
(``x.view(np.int64)``); outputs come back the same way.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

from fractions import Fraction

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("torch MPS (Metal GPU) is not available", allow_module_level=True)

from optiland.backend.torch_backend.metal import compile as mc  # noqa: E402

# ---------------------------------------------------------------------------
# Kernel generation
# ---------------------------------------------------------------------------
UNARY_OPS = [
    "sqrt", "neg", "abs", "floor", "ceil", "trunc", "rint", "round_away", "sign",
    "recip", "rsqrt", "sqr",
]  # fmt: skip
BINARY_OPS = [
    "add", "sub", "mul", "div", "fmod", "remainder_py", "minimum", "maximum",
    "fmin", "fmax", "copysign",
]  # fmt: skip
OPERATOR_OPS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
COMPARE_OPS = ["eq", "ne", "lt", "le", "gt", "ge"]
COMPARE_SYMBOLS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
CLASSIFY_OPS = [
    "is_nan", "is_inf", "is_finite", "is_zero", "is_negative", "signbit",
    "is_positive", "is_subnormal",
]  # fmt: skip

# Argument macros keep the generated kernels readable: IN_L(a, 0) is an int64
# input bound to buffer 0, OUT_L an int64 output, and so on. X(a) decodes a[tid].
_KERNEL_PRELUDE = """
#define IN_L(n, i) device const long* n [[buffer(i)]]
#define IN_I(n, i) device const int* n [[buffer(i)]]
#define IN_F(n, i) device const float* n [[buffer(i)]]
#define IN_F2(n, i) device const float2* n [[buffer(i)]]
#define OUT_L(n, i) device long* n [[buffer(i)]]
#define OUT_I(n, i) device int* n [[buffer(i)]]
#define OUT_F(n, i) device float* n [[buffer(i)]]
#define OUT_F2(n, i) device float2* n [[buffer(i)]]
#define TID uint tid [[thread_position_in_grid]]
#define X(n) sf::make((ulong)n[tid])
#define BITS(v) (long)(v).bits
"""

# fmt: off
_KERNEL_FIXED = """
kernel void t_compare(IN_L(a, 0), IN_L(b, 1), OUT_I(o, 2), TID) {
    sf64 x = X(a); sf64 y = X(b);
    o[tid] = CMP_FN | CMP_OP;
}
kernel void t_classify(IN_L(a, 0), OUT_I(o, 1), TID) {
    sf64 x = X(a);
    o[tid] = CLS;
}
kernel void t_fma(IN_L(a, 0), IN_L(b, 1), IN_L(c, 2), OUT_L(o, 3), TID) {
    o[tid] = BITS(sf::fma(X(a), X(b), X(c)));
}
kernel void t_neg_op(IN_L(a, 0), OUT_L(o, 1), TID) {
    sf64 x = X(a); o[tid] = BITS(-x);
}
kernel void t_to_float(IN_L(a, 0), OUT_F(o, 1), TID) { o[tid] = sf::to_float(X(a)); }
kernel void t_from_float(IN_F(a, 0), OUT_L(o, 1), TID) {
    o[tid] = BITS(sf::from_float(a[tid]));
}
kernel void t_to_long(IN_L(a, 0), OUT_L(o, 1), TID) { o[tid] = sf::to_long(X(a)); }
kernel void t_to_long_rint(IN_L(a, 0), OUT_L(o, 1), TID) {
    o[tid] = sf::to_long_rint(X(a));
}
kernel void t_to_int(IN_L(a, 0), OUT_I(o, 1), TID) { o[tid] = sf::to_int(X(a)); }
kernel void t_from_long(IN_L(a, 0), OUT_L(o, 1), TID) {
    o[tid] = BITS(sf::from_long(a[tid]));
}
kernel void t_from_int(IN_I(a, 0), OUT_L(o, 1), TID) {
    o[tid] = BITS(sf::from_int(a[tid]));
}
kernel void t_ldexp(IN_L(a, 0), IN_I(n, 1), OUT_L(o, 2), TID) {
    o[tid] = BITS(sf::ldexp(X(a), n[tid]));
}
kernel void t_mul_pwr2(IN_L(a, 0), IN_I(n, 1), OUT_L(o, 2), TID) {
    o[tid] = BITS(sf::mul_pwr2(X(a), metal::ldexp(1.0f, n[tid])));
}
kernel void t_frexp(IN_L(a, 0), OUT_L(m, 1), OUT_I(e, 2), TID) {
    int ee = 12345;
    m[tid] = BITS(sf::frexp(X(a), ee));
    e[tid] = ee;
}
kernel void t_to_float2(IN_L(a, 0), OUT_F2(o, 1), TID) { o[tid] = sf::to_float2(X(a)); }
kernel void t_from_df64(IN_F2(a, 0), OUT_L(o, 1), TID) {
    o[tid] = BITS(sf::from_df64(a[tid].x, a[tid].y));
}
kernel void t_df64_roundtrip(IN_L(a, 0), OUT_L(o, 1), TID) {
    df64 d = sf::to_df64(X(a));
    o[tid] = BITS(sf::from_df64(d));
}
kernel void t_constants(OUT_L(o, 0), TID) {
    if (tid != 0) return;
    o[0] = BITS(sf::zero()); o[1] = BITS(sf::neg_zero()); o[2] = BITS(sf::one());
    o[3] = BITS(sf::one_half()); o[4] = BITS(sf::nan()); o[5] = BITS(sf::inf());
    o[6] = BITS(sf::neg_inf());
    o[7] = BITS(sf::mul_add(sf::one(), sf::one(), sf::one()));
}
"""
# fmt: on


def _test_kernels() -> str:
    parts = [_KERNEL_PRELUDE]
    for name in UNARY_OPS:
        parts.append(
            f"kernel void u_{name}(IN_L(a, 0), OUT_L(o, 1), TID) "
            f"{{ o[tid] = BITS(sf::{name}(X(a))); }}"
        )
    for name in BINARY_OPS:
        parts.append(
            f"kernel void b_{name}(IN_L(a, 0), IN_L(b, 1), OUT_L(o, 2), TID) "
            f"{{ o[tid] = BITS(sf::{name}(X(a), X(b))); }}"
        )
    for name, sym in OPERATOR_OPS.items():
        parts.append(
            f"kernel void op_{name}(IN_L(a, 0), IN_L(b, 1), OUT_L(o, 2), TID) "
            f"{{ sf64 x = X(a); sf64 y = X(b); o[tid] = BITS(x {sym} y); }}"
        )
    cmp_fn = " | ".join(
        f"((int)sf::{n}(x, y) << {i})" for i, n in enumerate(COMPARE_OPS)
    )
    cmp_op = " | ".join(
        f"((int)(x {COMPARE_SYMBOLS[n]} y) << {i + 8})"
        for i, n in enumerate(COMPARE_OPS)
    )
    cls = " | ".join(f"((int)sf::{n}(x) << {i})" for i, n in enumerate(CLASSIFY_OPS))
    fixed = _KERNEL_FIXED.replace("CMP_FN", cmp_fn).replace("CMP_OP", cmp_op)
    parts.append(fixed.replace("CLS", cls))
    return "\n".join(parts)


@pytest.fixture(scope="module")
def lib():
    source = mc.kernel_source("vendor/softfloat64.metal", "df64_core.h", "sf64_core.h")
    return mc.compile_library(source + _test_kernels())


# ---------------------------------------------------------------------------
# GPU launch helpers
# ---------------------------------------------------------------------------
MPS = torch.device("mps")


def _bits(x: np.ndarray) -> torch.Tensor:
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    return torch.from_numpy(x64.view(np.int64)).to(MPS)


def _mps(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(MPS)


def _launch(lib, name: str, inputs: list, n: int, dtype=torch.int64, shape=None):
    out = torch.empty(shape or (n,), dtype=dtype, device=MPS)
    getattr(lib, name)(*inputs, out, threads=[n, 1, 1])
    return out.cpu().numpy()


def run_unary(lib, name: str, x: np.ndarray) -> np.ndarray:
    return _launch(lib, name, [_bits(x)], x.size).view(np.float64)


def run_binary(lib, name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _launch(lib, name, [_bits(x), _bits(y)], x.size).view(np.float64)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
INF = np.inf
NAN = np.nan
MIN_NORMAL = np.finfo(np.float64).tiny
MIN_SUB = np.float64(5e-324)
MAX_SUB = np.nextafter(MIN_NORMAL, 0.0)
MAXF = np.finfo(np.float64).max
# A quiet NaN with payload and a negative NaN.
PAYLOAD_NAN = np.array(
    [0x7FF8000000000001, -0x0007FFFFFFFFFFFF - 1], dtype=np.int64
).view(np.float64)
# fmt: off
SPECIALS = np.array([
    0.0, -0.0, 1.0, -1.0, INF, -INF, NAN, PAYLOAD_NAN[0], PAYLOAD_NAN[1],
    0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5,
    0.49999999999999994, -0.49999999999999994, 0.50000000000000011,
    2.0, -2.0, 3.0, -3.0, 0.1, -0.1, 1e-16, 1e16,
    2.0**52, -(2.0**52), 2.0**52 + 1, 2.0**53, -(2.0**53), 2.0**53 - 1, 2.0**52 - 0.5,
    2.0**51 + 0.5, -(2.0**51 + 0.5), 2.0**62, 2.0**63, -(2.0**63), 2.0**64, 2.0**31,
    2.0**31 - 1, -(2.0**31),
    MIN_NORMAL, -MIN_NORMAL, MIN_SUB, -MIN_SUB, MAX_SUB, -MAX_SUB, MAXF, -MAXF,
    1e300, -1e300, 1e-300, -1e-300, 1e308, 1e-308, np.pi, np.e, 7.0, 1e10, 1e-10,
], dtype=np.float64)
# fmt: on


def rand_bits(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniformly random 64-bit patterns (all classes, incl. NaN/inf/subnormal)."""
    return rng.integers(-(2**63), 2**63, size=n, dtype=np.int64).view(np.float64)


def rand_optics(rng: np.random.Generator, n: int) -> np.ndarray:
    """Signed log-uniform magnitudes in the optics working range [1e-12, 1e6]."""
    mag = 10.0 ** rng.uniform(-12, 6, size=n)
    return mag * rng.choice([-1.0, 1.0], size=n)


def rand_subnormal(rng: np.random.Generator, n: int) -> np.ndarray:
    mant = rng.integers(1, 2**52, size=n, dtype=np.int64)
    sign = rng.integers(0, 2, size=n, dtype=np.int64) << 63
    return (mant | sign).view(np.float64)


def rand_halves(rng: np.random.Generator, n: int) -> np.ndarray:
    """Integers, half-integers and quarter-integers over many magnitudes."""
    base = rng.integers(-(2**53), 2**53, size=n, dtype=np.int64).astype(np.float64)
    scale = 2.0 ** rng.integers(-3, 1, size=n)
    return base * scale


def unary_inputs(rng: np.random.Generator, n_random: int = 200_000) -> np.ndarray:
    return np.concatenate(
        [
            SPECIALS,
            rand_bits(rng, n_random),
            rand_optics(rng, n_random),
            rand_subnormal(rng, n_random // 4),
            rand_halves(rng, n_random // 4),
        ]
    )


def binary_inputs(rng: np.random.Generator, n_random: int = 200_000):
    xs, ys = np.meshgrid(SPECIALS, SPECIALS, indexing="ij")
    q = n_random // 4
    parts_x = [
        xs.ravel(), rand_bits(rng, n_random), rand_optics(rng, n_random),
        rand_subnormal(rng, q), rand_optics(rng, q), rand_halves(rng, q),
    ]  # fmt: skip
    parts_y = [
        ys.ravel(), rand_bits(rng, n_random), rand_optics(rng, n_random),
        rand_subnormal(rng, q), rand_bits(rng, q), rand_halves(rng, q),
    ]  # fmt: skip
    return np.concatenate(parts_x), np.concatenate(parts_y)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def _hex(v: np.ndarray) -> str:
    return f"0x{int(v.view(np.int64)) & (2**64 - 1):016x}"


def assert_bits_equal(got: np.ndarray, ref: np.ndarray, name: str, **inputs) -> None:
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    same = (got.view(np.int64) == ref.view(np.int64)) | (np.isnan(got) & np.isnan(ref))
    if same.all():
        return
    bad = np.flatnonzero(~same)
    rows = []
    for i in bad[:8]:
        args = ", ".join(f"{k}={v[i]!r}" for k, v in inputs.items())
        rows.append(
            f"  [{i}] {args} -> got {got[i]!r} ({_hex(got[i])}),"
            f" expected {ref[i]!r} ({_hex(ref[i])})"
        )
    pytest.fail(
        f"{name}: {bad.size} of {same.size} results differ from the reference:\n"
        + "\n".join(rows)
    )


def assert_int_equal(got: np.ndarray, ref: np.ndarray, name: str, x: np.ndarray):
    bad = np.flatnonzero(got != ref)
    assert bad.size == 0, (
        f"{name}: {bad.size} mismatches, e.g. x={x[bad[:5]]} "
        f"got {got[bad[:5]]} want {ref[bad[:5]]}"
    )


# ---------------------------------------------------------------------------
# Constants and construction
# ---------------------------------------------------------------------------
def test_constants(lib):
    got = _launch(lib, "t_constants", [], 1, shape=(8,))
    exp = np.array([0.0, -0.0, 1.0, 0.5, NAN, INF, -INF, 2.0])
    assert_bits_equal(got.view(np.float64), exp, "constants")
    assert got[4] == 0x7FF8000000000000  # canonical quiet NaN


# ---------------------------------------------------------------------------
# Arithmetic: bit-exact against NumPy float64
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_arithmetic_bit_exact(lib, op):
    rng = np.random.default_rng(1234 + len(op))
    x, y = binary_inputs(rng)
    fn = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}
    with np.errstate(all="ignore"):
        ref = fn[op](x, y)
    got = run_binary(lib, f"b_{op}", x, y)
    got_op = run_binary(lib, f"op_{op}", x, y)
    assert_bits_equal(got_op, got, f"operator {OPERATOR_OPS[op]} vs sf::{op}", x=x, y=y)
    assert_bits_equal(got, ref, f"sf::{op}", x=x, y=y)


def test_div_subnormal_output(lib):
    """The vendored header's comment says fdiv flushes subnormal outputs; it doesn't."""
    rng = np.random.default_rng(99)
    x = rand_optics(rng, 20_000) * 1e-300
    y = rand_optics(rng, 20_000) * 1e12
    with np.errstate(all="ignore"):
        ref = x / y
    keep = (ref != 0) & (np.abs(ref) < MIN_NORMAL)
    assert keep.sum() > 1000
    got = run_binary(lib, "b_div", x[keep], y[keep])
    assert_bits_equal(got, ref[keep], "sf::div (subnormal out)", x=x[keep], y=y[keep])


def test_sqrt_bit_exact(lib):
    rng = np.random.default_rng(7)
    x = unary_inputs(rng)
    with np.errstate(all="ignore"):
        ref = np.sqrt(x)
    got = run_unary(lib, "u_sqrt", x)
    assert_bits_equal(got, ref, "sf::sqrt", x=x)
    # Domain error returns NaN, never 0.
    assert np.isnan(run_unary(lib, "u_sqrt", np.array([-4.0, -MIN_SUB, -INF]))).all()


def test_recip_rsqrt_sqr(lib):
    rng = np.random.default_rng(8)
    x = unary_inputs(rng)
    with np.errstate(all="ignore"):
        refs = {"recip": 1.0 / x, "rsqrt": 1.0 / np.sqrt(x), "sqr": x * x}
    for name, ref in refs.items():
        assert_bits_equal(run_unary(lib, f"u_{name}", x), ref, f"sf::{name}", x=x)


def _fma_reference(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Exact a*b+c rounded once to binary64 (Fraction arithmetic, RNE via int/int)."""
    out = np.empty(a.size, dtype=np.float64)
    with np.errstate(all="ignore"):
        naive = a * b + c
    for i in range(a.size):
        ai, bi, ci = float(a[i]), float(b[i]), float(c[i])
        if np.isnan(ai) or np.isnan(bi) or np.isnan(ci):
            out[i] = np.nan
            continue
        if np.isinf(ai) or np.isinf(bi):
            # Infinite product: inf * 0 is invalid, inf + (-inf) is invalid.
            psign = np.copysign(1.0, ai) * np.copysign(1.0, bi)
            if ai == 0 or bi == 0 or (np.isinf(ci) and np.copysign(1.0, ci) != psign):
                out[i] = np.nan
            else:
                out[i] = np.copysign(np.inf, psign)
            continue
        if np.isinf(ci):
            out[i] = ci  # finite exact product (no intermediate overflow in fma)
            continue
        exact = Fraction(ai) * Fraction(bi) + Fraction(ci)
        if exact == 0:
            out[i] = naive[i]  # sign of an exact zero follows the IEEE sum rule
            continue
        try:
            out[i] = float(exact)  # int/int is correctly rounded (incl. subnormals)
        except OverflowError:
            out[i] = np.copysign(np.inf, 1.0 if exact > 0 else -1.0)
    return out


def test_fma_bit_exact(lib):
    rng = np.random.default_rng(11)
    n = 4000
    sp = rng.choice(SPECIALS, size=(3, 3000))
    a = np.concatenate(
        [sp[0], rand_bits(rng, n), rand_optics(rng, n), rand_optics(rng, n),
         rand_subnormal(rng, n // 2), rand_halves(rng, n // 2)]
    )  # fmt: skip
    b = np.concatenate(
        [sp[1], rand_bits(rng, n), rand_optics(rng, n), rand_optics(rng, n),
         rand_subnormal(rng, n // 2), rand_halves(rng, n // 2)]
    )  # fmt: skip
    # The block c = -(a*b rounded) exercises the single-rounding property: the exact
    # result is the rounding error of a*b, which a two-step mul+add would lose.
    with np.errstate(all="ignore"):
        neg_prod = -a[-4 * n : -3 * n] * b[-4 * n : -3 * n]
    c = np.concatenate(
        [sp[2], rand_bits(rng, n), rand_optics(rng, n), neg_prod,
         rand_subnormal(rng, n // 2), rand_halves(rng, n // 2)]
    )  # fmt: skip
    ref = _fma_reference(a, b, c)
    got = _launch(lib, "t_fma", [_bits(a), _bits(b), _bits(c)], a.size)
    assert_bits_equal(got.view(np.float64), ref, "sf::fma", a=a, b=b, c=c)
    # Explicit single-rounding witness: fma(1+2^-52, 1+2^-52, -(1+2^-51)) == 2^-104.
    a1 = np.array([1.0 + 2.0**-52])
    c1 = np.array([-(1.0 + 2.0**-51)])
    got1 = _launch(lib, "t_fma", [_bits(a1), _bits(a1), _bits(c1)], 1)
    assert got1.view(np.float64)[0] == 2.0**-104


# ---------------------------------------------------------------------------
# Sign, magnitude, classification, comparisons
# ---------------------------------------------------------------------------
def test_neg_abs_copysign_sign(lib):
    rng = np.random.default_rng(21)
    x = unary_inputs(rng, 50_000)
    assert_bits_equal(run_unary(lib, "u_neg", x), -x, "sf::neg", x=x)
    assert_bits_equal(run_unary(lib, "t_neg_op", x), -x, "operator-(sf64)", x=x)
    assert_bits_equal(run_unary(lib, "u_abs", x), np.abs(x), "sf::abs", x=x)
    y = unary_inputs(rng, 50_000)
    got = run_binary(lib, "b_copysign", x, y)
    assert_bits_equal(got, np.copysign(x, y), "sf::copysign", x=x, y=y)
    # sign: NaN -> NaN, +-0 preserved (df64_core semantics), else +-1.
    ref = np.where(np.isnan(x), NAN, np.where(x == 0, x, np.copysign(1.0, x)))
    assert_bits_equal(run_unary(lib, "u_sign", x), ref, "sf::sign", x=x)


def test_classification(lib):
    rng = np.random.default_rng(22)
    x = unary_inputs(rng, 50_000)
    got = _launch(lib, "t_classify", [_bits(x)], x.size, dtype=torch.int32)
    refs = {
        "is_nan": np.isnan(x),
        "is_inf": np.isinf(x),
        "is_finite": np.isfinite(x),
        "is_zero": x == 0,
        "is_negative": x < 0,
        "signbit": np.signbit(x),
        "is_positive": x > 0,
        "is_subnormal": (x != 0) & (np.abs(x) < MIN_NORMAL),
    }
    for i, name in enumerate(CLASSIFY_OPS):
        bit = ((got >> i) & 1).astype(bool)
        assert_int_equal(bit, refs[name], f"sf::{name}", x)


def test_comparisons(lib):
    rng = np.random.default_rng(23)
    x, y = binary_inputs(rng, 50_000)
    # Add exact-equal pairs so eq/le/ge see ties.
    x = np.concatenate([x, x[:5000]])
    y = np.concatenate([y, x[:5000]])
    got = _launch(lib, "t_compare", [_bits(x), _bits(y)], x.size, dtype=torch.int32)
    refs = {"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y, "gt": x > y}
    refs["ge"] = x >= y
    for i, name in enumerate(COMPARE_OPS):
        fn_bit = ((got >> i) & 1).astype(bool)
        op_bit = ((got >> (i + 8)) & 1).astype(bool)
        assert_int_equal(fn_bit, refs[name], f"sf::{name}", x)
        assert np.array_equal(op_bit, fn_bit), f"operator {COMPARE_SYMBOLS[name]}"


def test_minimum_maximum_fmin_fmax(lib):
    rng = np.random.default_rng(24)
    x, y = binary_inputs(rng, 50_000)
    x = np.concatenate([x, x[:5000]])
    y = np.concatenate([y, x[:5000]])
    refs = {
        "minimum": np.minimum(x, y),
        "maximum": np.maximum(x, y),
        "fmin": np.fmin(x, y),
        "fmax": np.fmax(x, y),
    }
    for name, ref in refs.items():
        got = run_binary(lib, f"b_{name}", x, y)
        assert_bits_equal(got, ref, f"sf::{name} (numpy)", x=x, y=y)
    # torch CPU agrees except on +-0 ties, where torch's scalar and vectorized
    # paths disagree with each other (the tie result is not a stable torch spec).
    keep = ~((x == y) & (x == 0))
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    fns = {"minimum": torch.minimum, "maximum": torch.maximum, "fmin": torch.fmin}
    fns["fmax"] = torch.fmax
    for name, fn in fns.items():
        got = run_binary(lib, f"b_{name}", x, y)[keep]
        ref = fn(tx, ty).numpy()[keep]
        assert_bits_equal(got, ref, f"sf::{name} (torch CPU)", x=x[keep], y=y[keep])


# ---------------------------------------------------------------------------
# Rounding to integers (exact)
# ---------------------------------------------------------------------------
def _round_away_ref(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        t = np.trunc(x)
        frac = np.abs(x - t)  # exact in binary64
    return np.where(frac >= 0.5, t + np.copysign(1.0, x), t)


@pytest.mark.parametrize("op", ["floor", "ceil", "trunc", "rint", "round_away"])
def test_rounding_exact(lib, op):
    rng = np.random.default_rng(31)
    x = unary_inputs(rng)
    # Dense coverage of every exponent that can hold a fraction.
    e = rng.integers(-4, 54, size=100_000)
    dense = np.ldexp(rng.uniform(1, 2, size=e.size), e)
    x = np.concatenate([x, dense * rng.choice([-1.0, 1.0], size=e.size)])
    fns = {"floor": np.floor, "ceil": np.ceil, "trunc": np.trunc, "rint": np.rint}
    fns["round_away"] = _round_away_ref
    with np.errstate(invalid="ignore"):
        ref = fns[op](x)
    assert_bits_equal(run_unary(lib, f"u_{op}", x), ref, f"sf::{op}", x=x)


def test_rint_halfway_and_signed_zero(lib):
    # fmt: off
    x = np.array([
        0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49999999999999994, -0.49999999999999994,
        2.0**52 - 0.5, -(2.0**52 - 0.5), 2.0**51 + 0.5, -0.3, 0.3, -0.0, 0.0, -0.7,
    ])
    rint_ref = np.array([
        0.0, 2.0, 2.0, -0.0, -2.0, -2.0, 0.0, -0.0,
        2.0**52, -(2.0**52), 2.0**51, -0.0, 0.0, -0.0, 0.0, -1.0,
    ])
    away_ref = np.array([
        1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 0.0, -0.0,
        2.0**52, -(2.0**52), 2.0**51 + 1, -0.0, 0.0, -0.0, 0.0, -1.0,
    ])
    # fmt: on
    assert_bits_equal(run_unary(lib, "u_rint", x), rint_ref, "sf::rint", x=x)
    assert_bits_equal(
        run_unary(lib, "u_round_away", x), away_ref, "sf::round_away", x=x
    )
    small = np.array([-0.3, 0.3, -0.0, 0.0])
    assert_bits_equal(
        run_unary(lib, "u_floor", small), np.array([-1.0, 0.0, -0.0, 0.0]), "sf::floor"
    )
    assert_bits_equal(
        run_unary(lib, "u_ceil", small), np.array([-0.0, 1.0, -0.0, 0.0]), "sf::ceil"
    )


# ---------------------------------------------------------------------------
# fmod / remainder
# ---------------------------------------------------------------------------
def _fmod_inputs(rng: np.random.Generator):
    x, y = binary_inputs(rng, 30_000)
    # Huge quotients (1e300 fmod 3 needs ~1000 long-division steps), tiny divisors.
    huge = np.array(
        [1e300, -1e300, MAXF, 1e200, 2.0**1000, 1e308, 5.5, -5.5, 1e300, 1e18]
    )
    small = np.array([3.0, 3.0, 3.0, 7.0, 3.0, 1e-300, 2.0, 2.0, MIN_SUB, 0.1])
    xs = [x, huge, rand_halves(rng, 20_000), rand_optics(rng, 20_000)]
    ys = [y, small, rand_halves(rng, 20_000), rand_subnormal(rng, 20_000)]
    xs.append(rand_subnormal(rng, 5000))
    ys.append(rand_subnormal(rng, 5000))
    return np.concatenate(xs), np.concatenate(ys)


def test_fmod_exact(lib):
    rng = np.random.default_rng(41)
    x, y = _fmod_inputs(rng)
    with np.errstate(all="ignore"):
        ref = np.fmod(x, y)
    assert_bits_equal(run_binary(lib, "b_fmod", x, y), ref, "sf::fmod", x=x, y=y)
    # Domain: fmod(x, 0) and fmod(inf, y) are NaN, fmod(x, inf) is x, sign of a kept.
    got = run_binary(
        lib, "b_fmod", np.array([1.0, INF, 2.5, -0.0]), np.array([0.0, 2.0, INF, 3.0])
    )
    assert np.isnan(got[0]) and np.isnan(got[1]) and got[2] == 2.5
    assert got[3] == 0 and np.signbit(got[3])


def test_remainder_py_exact(lib):
    rng = np.random.default_rng(42)
    x, y = _fmod_inputs(rng)
    with np.errstate(all="ignore"):
        ref = np.remainder(x, y)
    got = run_binary(lib, "b_remainder_py", x, y)
    assert_bits_equal(got, ref, "sf::remainder_py", x=x, y=y)
    xa = np.array([-1.0, 1.0, 6.0, -6.0])
    ya = np.array([3.0, -3.0, 3.0, -3.0])
    got = run_binary(lib, "b_remainder_py", xa, ya)
    assert list(got[:2]) == [2.0, -2.0]
    assert got[2] == 0 and not np.signbit(got[2])
    assert got[3] == 0 and np.signbit(got[3])


# ---------------------------------------------------------------------------
# ldexp / frexp / mul_pwr2
# ---------------------------------------------------------------------------
def test_ldexp_exact(lib):
    rng = np.random.default_rng(51)
    x = unary_inputs(rng, 100_000)
    n = rng.integers(-1200, 1200, size=x.size, dtype=np.int32)
    xs = [x] + [SPECIALS] * 4 + [rand_optics(rng, 2000)]
    ns = [n] + [np.full(SPECIALS.size, k, np.int32) for k in (5000, -5000, 2100, -2100)]
    ns.append(rng.integers(-2200, 2200, size=2000, dtype=np.int32))
    x, n = np.concatenate(xs), np.concatenate(ns)
    with np.errstate(all="ignore"):
        ref = np.ldexp(x, n)
    got = _launch(lib, "t_ldexp", [_bits(x), _mps(n)], x.size).view(np.float64)
    assert_bits_equal(got, ref, "sf::ldexp", x=x, n=n)
    # mul_pwr2 with a float32 power of two agrees with ldexp inside float32's range.
    keep = (n >= -126) & (n <= 127)
    m = int(keep.sum())
    got2 = _launch(lib, "t_mul_pwr2", [_bits(x[keep]), _mps(n[keep])], m)
    assert_bits_equal(
        got2.view(np.float64), ref[keep], "mul_pwr2", x=x[keep], n=n[keep]
    )


def test_frexp_exact(lib):
    rng = np.random.default_rng(52)
    x = unary_inputs(rng, 100_000)
    m_ref, e_ref = np.frexp(x)
    m_out = torch.empty(x.size, dtype=torch.int64, device=MPS)
    e_out = torch.full((x.size,), 999, dtype=torch.int32, device=MPS)
    lib.t_frexp(_bits(x), m_out, e_out, threads=[x.size, 1, 1])
    assert_bits_equal(m_out.cpu().numpy().view(np.float64), m_ref, "sf::frexp m", x=x)
    assert_int_equal(e_out.cpu().numpy(), e_ref, "sf::frexp exponent", x)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------
def test_to_float_from_float_roundtrip(lib):
    rng = np.random.default_rng(61)
    x = unary_inputs(rng, 100_000)
    # Values straddling float32's limits and ties at float32 precision, plus a dense
    # sweep across the float32 subnormal range (2^-160 .. 2^-120).
    # fmt: off
    edge = np.array([
        3.4028235e38, 3.4028236e38, 3.4028237e38, -3.4028237e38, 1e39,
        1.1754944e-38, 1e-45, 7e-46, 7.006492321624085e-46, 1e-50, 1e-44, 2.1e-45,
        1 + 2.0**-24, 1 + 3 * 2.0**-24, 1 + 2.0**-24 + 2.0**-50,
    ])
    # fmt: on
    sweep = np.ldexp(rng.uniform(1, 2, 20_000), rng.integers(-160, -120, 20_000))
    x = np.concatenate([x, edge, sweep, -sweep])
    with np.errstate(all="ignore"):
        ref32 = x.astype(np.float32)
    got32 = _launch(lib, "t_to_float", [_bits(x)], x.size, dtype=torch.float32)
    assert_bits_equal(
        got32.astype(np.float64), ref32.astype(np.float64), "to_float", x=x
    )
    same32 = got32.view(np.int32) == ref32.view(np.int32)
    assert (same32 | (np.isnan(got32) & np.isnan(ref32))).all()
    # float32 -> sf64 is exact.
    fin = ref32[np.isfinite(ref32)]
    f = np.concatenate([fin, np.array([np.inf, -np.inf, np.nan, -0.0], np.float32)])
    got64 = _launch(lib, "t_from_float", [_mps(f)], f.size).view(np.float64)
    assert_bits_equal(got64, f.astype(np.float64), "sf::from_float", f=f)


def _to_long_ref(x: np.ndarray, rint: bool) -> np.ndarray:
    out = np.zeros(x.size, dtype=np.int64)
    for i, v in enumerate(x):
        if np.isnan(v):
            out[i] = 0
        elif v >= 2.0**63:
            out[i] = 2**63 - 1
        elif v <= -(2.0**63):
            out[i] = -(2**63)
        else:
            out[i] = round(float(v)) if rint else int(float(v))  # round: half-even
    return out


def test_to_long_and_to_int(lib):
    rng = np.random.default_rng(62)
    # fmt: off
    edge = np.array([
        2.0**63 - 1024, -(2.0**63) - 2048, 1e19, -1e19, 2.0**31, -(2.0**31) - 1,
        2.0**31 - 0.5, 2.0**52 + 1, 2.0**53 - 1, -(2.0**52) - 3,
    ])
    # fmt: on
    x = np.concatenate(
        [SPECIALS, rand_halves(rng, 5000), rand_optics(rng, 5000), rand_bits(rng, 5000)]
    )
    x = np.concatenate([x, edge])
    for name, rint in [("t_to_long", False), ("t_to_long_rint", True)]:
        got = _launch(lib, name, [_bits(x)], x.size)
        assert_int_equal(got, _to_long_ref(x, rint), f"sf::{name[2:]}", x)
    # NumPy agrees on the in-range finite cases (truncation).
    finite = np.isfinite(x) & (np.abs(x) < 2.0**62)
    assert np.array_equal(_to_long_ref(x[finite], False), x[finite].astype(np.int64))
    got32 = _launch(lib, "t_to_int", [_bits(x)], x.size, dtype=torch.int32)
    ref32 = np.clip(_to_long_ref(x, False), -(2**31), 2**31 - 1).astype(np.int32)
    assert_int_equal(got32, ref32, "sf::to_int", x)


def test_from_long_from_int(lib):
    rng = np.random.default_rng(63)
    # fmt: off
    edge = np.array([
        0, 1, -1, 2**53, 2**53 + 1, 2**53 + 3, -(2**53) - 1, 2**63 - 1, -(2**63),
        2**62 + 2**10 + 1, 2**54 + 3, 3 * 2**52 + 1,
    ], dtype=np.int64)
    # fmt: on
    ints = np.concatenate(
        [
            rng.integers(-(2**63), 2**63, size=50_000, dtype=np.int64),
            rng.integers(-(2**53), 2**53, size=50_000, dtype=np.int64),
            edge,
        ]
    )
    got = _launch(lib, "t_from_long", [_mps(ints)], ints.size).view(np.float64)
    assert_bits_equal(got, ints.astype(np.float64), "sf::from_long", i=ints)
    i32 = np.concatenate(
        [
            rng.integers(-(2**31), 2**31, size=50_000, dtype=np.int32),
            np.array([0, 1, -1, 2**31 - 1, -(2**31)], dtype=np.int32),
        ]
    )
    got = _launch(lib, "t_from_int", [_mps(i32)], i32.size).view(np.float64)
    assert_bits_equal(got, i32.astype(np.float64), "sf::from_int", i=i32)


def test_df64_conversions(lib):
    rng = np.random.default_rng(64)
    x = np.concatenate([SPECIALS, rand_optics(rng, 100_000), rand_halves(rng, 20_000)])
    x = np.concatenate([x, rand_bits(rng, 20_000)])
    # the top half-ulp window of FLT_MAX (fix round 2: (inf, -inf) pairs before)
    top = np.array(
        [
            0x47EFFFFFEFFFFFF8,
            0x47EFFFFFEFFFFFFF,
            0xC7EFFFFFEFFFFFF8,
            0x47EFFFFFEFFFFFF0,
        ],
        dtype=np.uint64,
    ).view(np.float64)
    x = np.concatenate([x, top])
    # to_float2: hi = RN32(x), lo = RN32(x - hi), then one float32 Fast2Sum so
    # that a tie-valued lo (+-ulp(hi)/2 next to an odd hi) becomes the even-hi
    # canonical form of the df64 kernels; lo = 0 whenever hi is not finite.
    with np.errstate(all="ignore"):
        hi = x.astype(np.float32)
        lo = (x - hi.astype(np.float64)).astype(np.float32)
        lo = np.where(np.isfinite(hi), lo, np.float32(0))
        s = (hi + lo).astype(np.float32)
        e = (lo - (s - hi)).astype(np.float32)
        keep = np.isfinite(hi) & (np.abs(lo) >= np.float32(2.0**-126))  # normal lo
        # a Fast2Sum that overflows (|x| within 2^-50 of the float32 overflow
        # tie) is the df64 overflow threshold: clean +-inf, lo = 0 (fix round 2)
        overflow = keep & ~np.isfinite(s)
        hi = np.where(keep, s, hi)
        lo = np.where(keep & ~overflow, e, np.where(overflow, np.float32(0), lo))
    got = _launch(lib, "t_to_float2", [_bits(x)], x.size, torch.float32, (x.size, 2))
    got_hi, got_lo = got[:, 0].astype(np.float64), got[:, 1].astype(np.float64)
    assert_bits_equal(got_hi, hi.astype(np.float64), "to_float2 hi", x=x)
    assert_bits_equal(got_lo, lo.astype(np.float64), "to_float2 lo", x=x)
    with np.errstate(over="ignore"):  # np.spacing(FLT_MAX) overflows in float32
        assert np.all(
            np.abs(lo[np.isfinite(hi)]) <= np.spacing(np.abs(hi[np.isfinite(hi)])) / 2
        )
    # from_df64(hi, lo) == RN(hi + lo) in binary64 (exact when the pair came from a
    # double, which is the case here); a zero lo returns hi itself so that the
    # sign of a zero pair survives ((-0, +0) is the df64 -0).
    pairs = np.stack([hi, lo], axis=1).astype(np.float32)
    ref = hi.astype(np.float64) + lo.astype(np.float64)
    ref = np.where(lo == 0, hi.astype(np.float64), ref)
    got64 = _launch(lib, "t_from_df64", [_mps(pairs)], x.size).view(np.float64)
    assert_bits_equal(got64, ref, "sf::from_df64", hi=hi, lo=lo)
    neg_zero = _launch(
        lib, "t_from_df64", [_mps(np.array([[-0.0, 0.0]], np.float32))], 1
    )
    assert neg_zero.view(np.float64)[0] == 0 and np.signbit(
        neg_zero.view(np.float64)[0]
    )
    # Round trip through the df64 struct overloads rounds x to 48 bits: exact iff
    # x fits in 48 bits, and within 2^-48 relative otherwise (float32 range).
    rt = _launch(lib, "t_df64_roundtrip", [_bits(x)], x.size).view(np.float64)
    assert_bits_equal(rt, ref, "sf::from_df64(sf::to_df64(x))", x=x)
    assert rt[-4:].tolist() == [np.inf, np.inf, -np.inf, top[3]]  # never NaN
    in_range = np.isfinite(x) & (np.abs(x) < 3e38) & (np.abs(x) > 1e-30)
    fits48 = in_range & ((x.view(np.int64) & 0x1F) == 0)  # low 5 mantissa bits clear
    assert_bits_equal(rt[fits48], x[fits48], "df64 round trip (48-bit)", x=x[fits48])
    with np.errstate(all="ignore"):
        rel = np.abs(rt - x) / np.abs(x)
    assert np.nanmax(rel[in_range]) <= 2.0**-48


# ---------------------------------------------------------------------------
# Amalgamation / sf64_math bridge
# ---------------------------------------------------------------------------
def test_sf64_math_compiles_to_nothing_without_df64():
    src = mc.kernel_source("vendor/softfloat64.metal", "sf64_core.h", "sf64_math.h")
    src += _KERNEL_PRELUDE + (
        "kernel void k(OUT_L(o, 0), TID) "
        "{ o[tid] = BITS(sf::sqrt(sf::make(0x4010000000000000UL))); }"
    )
    lib2 = mc.compile_library(src)
    assert _launch(lib2, "k", [], 1).view(np.float64)[0] == 2.0


# float32 stand-ins for the df64 math lane (only the bridge plumbing is under test).
_DF_STUB_NAMES = [
    "exp", "exp2", "log", "log2", "log10", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
]  # fmt: skip
_DF_STUB_IDENTITY = ["expm1", "log1p", "cbrt", "erf", "erfc", "erfinv", "lgamma"]
# The bridge's range handling (fix round 3) also uses the exp lane's internals
# (log_ext, prod3, mul3, exp_of_sum and the DF64X_* constants); float32 stand-ins
# for those too.
_DF_EXP_LANE_STUBS = """
static constant float DF64X_1_LN2_F = 1.442695f;
static constant float3 DF64X_LN2_3 =
    float3(0.6931472f, -1.9046542e-09f, -8.783184e-17f);
static constant float3 DF64X_1_LN2_3 =
    float3(1.442695f, 1.925963e-08f, -4.2373395e-16f);
static constant float3 DF64X_1_LN10_3 =
    float3(0.4342945f, -1.010305e-08f, -1.00039104e-16f);
namespace df {
inline df64 log_ext(df64 x, thread float &ext) {
    ext = 0.0f; return make(metal::log(x.hi));
}
inline void prod3(df64 a, float ext, float3 L, thread float &P, thread df64 &tail) {
    P = a.hi * L.x; tail = make(ext * L.x, 0.0f);
}
inline df64 mul3(df64 a, float ext, float3 L) { return make((a.hi + ext) * L.x); }
inline df64 exp_of_sum(float P, df64 tail) { return make(metal::exp(P + tail.hi)); }
}  // namespace df
"""
_DF_STUBS = (
    "\n".join(
        f"inline df64 {n}(df64 x) {{ return make(metal::{n}(x.hi)); }}"
        for n in _DF_STUB_NAMES
    )
    + "\n"
    + "\n".join(f"inline df64 {n}(df64 x) {{ return x; }}" for n in _DF_STUB_IDENTITY)
    + """
inline df64 pow(df64 x, df64 y) { return make(metal::pow(x.hi, y.hi)); }
inline df64 atan2(df64 y, df64 x) { return make(metal::atan2(y.hi, x.hi)); }
inline df64 hypot(df64 x, df64 y) {
    return make(metal::sqrt(x.hi * x.hi + y.hi * y.hi));
}
"""
)


def test_sf64_math_bridge_with_df64_stubs():
    """The bridge compiles against df:: functions and rounds through df64."""
    src = mc.kernel_source(
        "vendor/softfloat64.metal", "df64_core.h", "df64_constants.h"
    )
    src += _DF_EXP_LANE_STUBS
    src += "\nnamespace df {\n" + _DF_STUBS + "\n}  // namespace df\n"
    # The bridge emits each family only when its df64 math header was seen.
    src += "\n".join(
        f"#define OPTILAND_DF64_MATH_{fam}_H" for fam in ("EXP", "TRIG", "SPECIAL")
    )
    src += "\n" + mc.kernel_source("sf64_core.h", "sf64_math.h") + _KERNEL_PRELUDE
    src += """
kernel void k(IN_L(a, 0), OUT_L(o, 1), TID) {
    sf64 x = X(a);
    sf64 r;
    switch (tid % 4) {
        case 0: r = sf::exp(x); break;
        case 1: r = sf::atan2(x, sf::one()); break;
        case 2: r = sf::pow(x, sf::make(0x4000000000000000UL)); break;
        default: r = sf::hypot(x, x); break;
    }
    o[tid] = BITS(r);
}
"""
    lib2 = mc.compile_library(src)
    x = np.array([0.5, 0.25, 3.0, 1.5, -1.0, 2.0, 7.0, 0.1])
    got = _launch(lib2, "k", [_bits(x)], x.size).view(np.float64)
    # fmt: off
    ref = np.array([
        np.exp(0.5), np.arctan2(0.25, 1), 9.0, np.hypot(1.5, 1.5),
        np.exp(-1.0), np.arctan2(2.0, 1), 49.0, np.hypot(0.1, 0.1),
    ])
    # fmt: on
    np.testing.assert_allclose(got, ref, rtol=2e-6)


def test_sf64_math_bridge_with_real_df64_headers():
    """exp/log/trig bridged through the real df64 headers stay within ~2^-48."""
    kernels = mc._KERNEL_DIR
    names = ["df64_constants.h", "df64_math_exp.h", "df64_math_trig.h"]
    if not all((kernels / n).exists() for n in names):
        pytest.skip("df64 math headers not present yet")
    src = mc.kernel_source("vendor/softfloat64.metal", "df64_core.h", *names)
    src += mc.kernel_source("sf64_core.h", "sf64_math.h") + _KERNEL_PRELUDE
    src += """
kernel void k(IN_L(a, 0), OUT_L(o, 1), TID) {
    sf64 x = X(a);
    sf64 r;
    switch (tid % 5) {
        case 0: r = sf::exp(x); break;
        case 1: r = sf::log(x); break;
        case 2: r = sf::sin(x); break;
        case 3: r = sf::atan2(x, sf::one()); break;
        default: r = sf::pow(x, sf::make(0x4000000000000000UL)); break;
    }
    o[tid] = BITS(r);
}
"""
    lib2 = mc.compile_library(src)
    x = np.array([0.5, 2.0, 1.0, 0.25, 3.0, -20.0, 1e-3, 100.0, -7.0, 1e-4])
    got = _launch(lib2, "k", [_bits(x)], x.size).view(np.float64)
    # fmt: off
    ref = np.array([
        np.exp(0.5), np.log(2.0), np.sin(1.0), np.arctan2(0.25, 1.0), 9.0,
        np.exp(-20.0), np.log(1e-3), np.sin(100.0), np.arctan2(-7.0, 1.0), 1e-8,
    ])
    # fmt: on
    rel = np.abs(got - ref) / np.abs(ref)
    assert rel.max() <= 64 * 2.0**-48, rel / 2.0**-48


# ---------------------------------------------------------------------------
# Fix round 3 (NOTES/04 section 10): range handling of the sf64 bridge
# ---------------------------------------------------------------------------
_BRIDGE_UNARY = [
    "exp", "exp2", "expm1", "log", "log1p", "log2", "log10", "cbrt", "sin", "cos",
    "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "erf", "erfc", "erfinv", "lgamma",
]  # fmt: skip
_BRIDGE_BINARY = ["pow", "atan2", "hypot"]
_BRIDGE_HEADERS = (
    "vendor/softfloat64.metal",
    "df64_core.h",
    "df64_constants.h",
    "df64_math_exp.h",
    "df64_math_trig.h",
    "df64_math_special.h",
    "sf64_core.h",
    "sf64_math.h",
)


@pytest.fixture(scope="module")
def bridge():
    kernels = mc._KERNEL_DIR
    if not all((kernels / n).exists() for n in _BRIDGE_HEADERS):
        pytest.skip("full df64 / sf64 stack not present")
    parts = [mc.kernel_source(*_BRIDGE_HEADERS), _KERNEL_PRELUDE]
    for name in _BRIDGE_UNARY:
        parts.append(
            f"kernel void m_{name}(IN_L(a, 0), OUT_L(o, 1), TID) "
            f"{{ o[tid] = BITS(sf::{name}(X(a))); }}"
        )
    for name in _BRIDGE_BINARY:
        parts.append(
            f"kernel void m_{name}(IN_L(a, 0), IN_L(b, 1), OUT_L(o, 2), TID) "
            f"{{ o[tid] = BITS(sf::{name}(X(a), X(b))); }}"
        )
    return mc.compile_library("\n".join(parts))


def _bridge1(lib, name, x):
    x = np.asarray(x, dtype=np.float64)
    return _launch(lib, f"m_{name}", [_bits(x)], x.size).view(np.float64)


def _bridge2(lib, name, x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return _launch(lib, f"m_{name}", [_bits(x), _bits(y)], x.size).view(np.float64)


def _assert_close(name, got, ref, rel=4 * 2.0**-48, **inputs):
    """Bit-exact for specials / zeros, ``rel`` relative otherwise."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    both_nan = np.isnan(got) & np.isnan(ref)
    same_bits = got.view(np.int64) == ref.view(np.int64)
    finite = np.isfinite(ref) & (ref != 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        close = finite & (np.abs(got - ref) <= rel * np.abs(ref))
    ok = both_nan | same_bits | close
    if ok.all():
        return
    bad = np.flatnonzero(~ok)
    rows = [
        f"  [{i}] "
        + ", ".join(f"{k}={v[i]!r}" for k, v in inputs.items())
        + f" -> got {got[i]!r}, expected {ref[i]!r}"
        for i in bad[:8]
    ]
    pytest.fail(f"{name}: {bad.size} mismatches\n" + "\n".join(rows))


def test_sf64_bridge_domain_errors_below_float32_range(bridge):
    """log/log2/log10 of a negative value below 2^-126 returned -inf and
    pow(-tiny, 0.5) returned 0: the conversion to df64 gave -0. NaN now."""
    x = np.array([-1e-40, -1e-45, -1e-50, -1e-300, -5e-324, -1e-39, -3e-39, -1e-30])
    for name in ("log", "log2", "log10"):
        assert np.isnan(_bridge1(bridge, name, x)).all(), name
    assert np.isnan(_bridge2(bridge, "pow", x, np.full(x.size, 0.5))).all()
    assert np.isnan(_bridge2(bridge, "pow", x, np.full(x.size, 2.5))).all()
    # signed zero and -inf keep their C99 results
    z = np.array([-0.0, 0.0, -np.inf, np.inf, np.nan])
    got = _bridge1(bridge, "log", z)
    assert got[:2].tolist() == [-np.inf, -np.inf] and got[3] == np.inf
    assert np.isnan(got[2]) and np.isnan(got[4])


def test_sf64_bridge_subnormal_band_and_huge_arguments(bridge):
    """Arguments in float32's subnormal band (and binary64 subnormals) reached the
    df64 functions as denormal hi words: cbrt(1e-40) = 1e-40 (input returned),
    erf/expm1 likewise with 1..23 bits. The bridge now evaluates the leading
    term in sf64 (exact to binary64) or scales the argument (cbrt, log family)."""
    import scipy.special as sp

    tiny = np.array(
        [1e-40, 1e-39, 3e-39, 1e-44, 1e-45, 1e-38, 1e-300, 5e-324, -1e-40, -1e-300]
    )
    huge = np.array([1e300, -1e300, 1e40, 1e308, 4e38])
    _assert_close("cbrt", _bridge1(bridge, "cbrt", tiny), np.cbrt(tiny), x=tiny)
    _assert_close("cbrt", _bridge1(bridge, "cbrt", huge), np.cbrt(huge), x=huge)
    # x + O(x^3) / x + O(x^2): exact binary64 below 2^-27 resp. 2^-54
    small = np.concatenate([tiny, [1e-9, -3e-9, 2.0**-28, -(2.0**-28)]])
    for name, f in (
        ("sin", np.sin), ("tan", np.tan), ("asin", np.arcsin), ("atan", np.arctan),
        ("sinh", np.sinh), ("tanh", np.tanh), ("asinh", np.arcsinh),
        ("atanh", np.arctanh),
    ):  # fmt: skip
        assert_bits_equal(_bridge1(bridge, name, small), small, name, x=small)
        assert np.array_equal(_bridge1(bridge, name, small), f(small))
    lin = np.concatenate([tiny, [2.0**-55, -(2.0**-60)]])
    for name in ("expm1", "log1p"):
        assert_bits_equal(_bridge1(bridge, name, lin), lin, name, x=lin)
    _assert_close("erf", _bridge1(bridge, "erf", small), sp.erf(small), x=small)
    _assert_close(
        "erfinv", _bridge1(bridge, "erfinv", small), sp.erfinv(small), x=small
    )
    # log family over the whole binary64 range (scaled log)
    pos = np.array([1e-40, 1e-300, 5e-324, 1e300, 1e308, 1e-38, 2.0**-100, 2.0**900])
    for name, f in (("log", np.log), ("log2", np.log2), ("log10", np.log10)):
        _assert_close(name, _bridge1(bridge, name, pos), f(pos), x=pos)
    assert _bridge1(bridge, "log2", np.array([2.0**-1074, 2.0**1000])).tolist() == [
        -1074.0,
        1000.0,
    ]
    _assert_close(
        "log1p", _bridge1(bridge, "log1p", huge[[0, 2, 3]]), np.log1p(huge[[0, 2, 3]])
    )
    # lgamma: -log|x| below 2^-126, Stirling's leading term above 2^127, poles
    x = np.array([1e-40, -1e-40, 1e-45, 1e-300, -1e-300, 1e40, 1e300, 2.5])
    _assert_close("lgamma", _bridge1(bridge, "lgamma", x), sp.gammaln(x), x=x)
    poles = _bridge1(bridge, "lgamma", np.array([-1e40, -1e300]))
    assert poles.tolist() == [np.inf, np.inf]
    # asinh / acosh beyond 2^127
    x = np.array([1e300, -1e300, 1e40, 2.0**200])
    _assert_close("asinh", _bridge1(bridge, "asinh", x), np.arcsinh(x), x=x)
    _assert_close("acosh", _bridge1(bridge, "acosh", np.abs(x)), np.arccosh(np.abs(x)))
    assert np.isnan(_bridge1(bridge, "acosh", np.array([-1e300]))).all()
    # functions whose value at a flushed argument is right anyway stay so
    assert _bridge1(bridge, "cos", tiny).tolist() == [1.0] * tiny.size
    assert _bridge1(bridge, "cosh", tiny).tolist() == [1.0] * tiny.size
    assert _bridge1(bridge, "exp", tiny).tolist() == [1.0] * tiny.size
    assert _bridge1(bridge, "erfc", tiny).tolist() == [1.0] * tiny.size
    # in-range values are unaffected by the range handling (48-bit bridge; log
    # is kept away from 1, where the 48-bit input rounding is amplified)
    rng = np.random.default_rng(4)
    x = 10.0 ** rng.uniform(-20, 20, 500)
    x = x[np.abs(np.log10(x)) > 1]
    for name, f in (("log", np.log), ("cbrt", np.cbrt), ("atan", np.arctan)):
        _assert_close(name, _bridge1(bridge, name, x), f(x), rel=8 * 2.0**-48, x=x)


def test_sf64_bridge_pow_parity_and_tiny_exponents(bridge):
    """The parity / integrality of b was decided on its 48-bit rounding (odd b above
    2^48 became even, half-integers integers), exponents below 2^-126 flushed to
    pow(a, 0) = 1 for every base, and results outside float32's range were 0 / inf
    although binary64 holds them."""
    a = np.array(
        [-1.0, -(1 + 2**-50), -2.0, -1.0, np.nan, -2.0, 0.0, -0.0, np.inf, -np.inf,
         0.0, np.inf, np.nan, 0.0, -1e-40, 1e-300, 1e-300, 1e300, -1e300, -1e-300,
         1e-300, 1.0, -8.0, -8.0, 4.0, -1e-300, 2.0, -1e300, 10.0, 10.0, 2.0, 2.0,
         -10.0, -10.0, 1e-100, 3.0, -3.0, 1e-300, 0.5, -0.0, -0.0, -1.0]
    )  # fmt: skip
    b = np.array(
        [4527678976059651.0, 4527678976059651.0, 4527678976059651.0,
         2**51 + 2**27 + 0.5, 1e-300, 1e-300, 1e-300, 1e-300, 1e-300, 1e-300,
         -1e-300, -1e-300, 1e-40, -1e-40, 0.5, 0.5, -1.0, 0.5, 3.0, 3.0, 1e-300,
         np.nan, 1 / 3, 3.0, 0.5, 2.0, 1e300, 1e300, 40.0, -40.0, 1000.0, -1074.0,
         41.0, -41.0, 1.5, 700.0, 701.0, -1.1, 2000.0, -1e-300, -3.0, np.inf]
    )  # fmt: skip
    with np.errstate(all="ignore"):
        ref = np.power(a, b)
    _assert_close("pow", _bridge2(bridge, "pow", a, b), ref, a=a, b=b)
    rng = np.random.default_rng(3)
    odd = (rng.integers(2**47, 2**52, 5000) * 2 + 1).astype(np.float64)
    assert (_bridge2(bridge, "pow", np.full(odd.size, -1.0), odd) == -1.0).all()
    even = odd + 1.0
    assert (_bridge2(bridge, "pow", np.full(even.size, -1.0), even) == 1.0).all()
    half = (rng.integers(2**47, 2**52, 5000) + 0.5).astype(np.float64)
    assert np.isnan(_bridge2(bridge, "pow", np.full(half.size, -1.0), half)).all()
    # hypot: NaN before the conversion (1e300 would become inf), common scaling
    x = np.array([np.nan, np.nan, 1e300, 1e-300, 1e-300, 3e300, 1e-300, 1.0, 0.0])
    y = np.array([1e300, np.inf, 1e300, 1e-300, 1.0, 4e300, 0.0, 1e300, 1e300])
    _assert_close("hypot", _bridge2(bridge, "hypot", x, y), np.hypot(x, y), x=x, y=y)
    # atan2: common scaling keeps the quadrant and the ratio
    x = np.array([1e-300, 1e300, 3e300, 1e-300, 1e-300, 0.0, np.nan, 1e-300])
    y = np.array([1e-300, 1e300, 4e300, 0.0, -1e-300, 1e300, 1e300, -0.0])
    _assert_close("atan2", _bridge2(bridge, "atan2", x, y), np.arctan2(x, y), y=x, x=y)


def test_sf64_bridge_documents_the_remaining_float32_limits():
    header = (mc._KERNEL_DIR / "sf64_math.h").read_text()
    for phrase in (
        "results\n// below 2^-126 in magnitude are +-0",
        "sin/cos/tan of |x| >= 2^127 are NaN",
        "1e7 < |x| < 2^127",
        "float32-accurate (~1e-7 absolute",
        "codegen.OpSpec.sf_inexact",
    ):
        assert phrase in header, phrase
