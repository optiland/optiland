"""Tests for the Metal elementwise launcher: encode, codegen and MetalLibrary.

Oracle: numpy float64 evaluated on the *decoded* inputs (hi + lo), errors in
units of u^2 = 2^-48. Only the ops provided by ``df64_core.h`` are exercised
here; the transcendental headers get their own ULP suites.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")

from optiland.backend.torch_backend.metal import (  # noqa: E402
    codegen,
    encode,
    library,
)
from optiland.backend.torch_backend.metal import (  # noqa: E402
    compile as metal_compile,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="torch MPS (Metal GPU) not available"
)

U2 = 2.0**-48
PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
CORE_HEADERS = ("df64_core.h", "df64_constants.h")
SIZES = (1, 7, 1 << 20)

# ---------------------------------------------------------------------------
# References (numpy float64 on decoded inputs) and tolerances in u^2
# ---------------------------------------------------------------------------
EXACT = 0.0  # result must be bit-identical after decode


def _round_away(x: np.ndarray) -> np.ndarray:
    return np.copysign(np.floor(np.abs(x) + 0.5), x)


UNARY_REF = {
    "neg": (np.negative, EXACT),
    "abs": (np.abs, EXACT),
    "sign": (np.sign, EXACT),
    "sqrt": (np.sqrt, 4.0),
    "rsqrt": (lambda x: 1.0 / np.sqrt(x), 16.0),
    "recip": (lambda x: 1.0 / x, 10.0),
    "floor": (np.floor, EXACT),
    "ceil": (np.ceil, EXACT),
    "trunc": (np.trunc, EXACT),
    "rint": (np.rint, EXACT),
    "round_away": (_round_away, EXACT),
}
UNARY_PRED_REF = {"isnan": np.isnan, "isinf": np.isinf, "isfinite": np.isfinite}
BINARY_REF = {
    "add": (np.add, 3.0),
    "sub": (np.subtract, 3.0),
    "mul": (np.multiply, 5.0),
    "div": (np.divide, 10.0),
    "fmod": (np.fmod, 16.0),
    "remainder_py": (np.remainder, 16.0),
    "copysign": (np.copysign, EXACT),
    "minimum": (np.minimum, EXACT),
    "maximum": (np.maximum, EXACT),
    "fmin": (np.fmin, EXACT),
    "fmax": (np.fmax, EXACT),
}
BINARY_PRED_REF = {
    "eq": np.equal,
    "ne": np.not_equal,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
}
# (reference, tolerance in u^2, magnitude of the largest intermediate). The error
# of a composite op is relative to its intermediates: a + b/c with b/c ~ -a cancels.
TERNARY_REF = {
    "fma": (
        lambda a, b, c: a * b + c,
        10.0,
        lambda a, b, c: np.maximum(np.abs(a * b), np.abs(c)),
    ),
    "lerp": (
        lambda a, b, c: a + c * (b - a),
        16.0,
        lambda a, b, c: np.maximum(np.abs(a), np.abs(c * (b - a))),
    ),
    "addcmul": (
        lambda a, b, c: a + b * c,
        10.0,
        lambda a, b, c: np.maximum(np.abs(a), np.abs(b * c)),
    ),
    "addcdiv": (
        lambda a, b, c: a + b / c,
        16.0,
        lambda a, b, c: np.maximum(np.abs(a), np.abs(b / c)),
    ),
    "clamp": (lambda a, b, c: np.minimum(np.maximum(a, b), c), EXACT, None),
}
# fmod/remainder: absolute error scales with |a| (q*b is a product of size ~|a|).
SCALE_BY_FIRST = {"fmod", "remainder_py"}


def rand(n: int, seed: int, lo: float = 1e-6, hi: float = 1e4, positive: bool = False):
    rng = np.random.default_rng(seed)
    mag = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    if positive:
        return mag
    return mag * rng.choice([-1.0, 1.0], n)


def gpu(x: np.ndarray):
    return encode.to_mps_df64(np.asarray(x, dtype=np.float64))


def decoded(x: np.ndarray) -> np.ndarray:
    hi, lo = encode.encode_df64(x)
    return encode.decode_df64(hi, lo)


def assert_close(got, ref, tol_u2: float, scale=None, check_signbit: bool = True):
    """NaN == NaN, zeros by signbit (exact ops), otherwise relative error <= tol."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert got.shape == ref.shape
    nan_g, nan_r = np.isnan(got), np.isnan(ref)
    assert np.array_equal(nan_g, nan_r), (
        f"NaN pattern differs: {np.flatnonzero(nan_g != nan_r)[:8]}"
    )
    m = ~nan_r
    if tol_u2 == EXACT:
        bad = np.flatnonzero(m & (got != ref))
        assert bad.size == 0, (
            f"mismatch at {bad[:8]}: got {got[bad[:8]]} ref {ref[bad[:8]]}"
        )
        if check_signbit:
            assert np.array_equal(np.signbit(got[m]), np.signbit(ref[m])), (
                "signbit differs"
            )
        return
    inf = np.isinf(ref)
    assert np.array_equal(got[inf], ref[inf]), "inf mismatch"
    f = m & ~inf
    bound = (
        np.abs(ref[f])
        if scale is None
        else np.maximum(np.abs(ref[f]), np.abs(scale[f]))
    )
    err = np.abs(got[f] - ref[f])
    worst = np.max(err / np.where(bound > 0, bound, 1.0) / U2) if err.size else 0.0
    assert worst <= tol_u2, f"max error {worst:.2f} u^2 > {tol_u2} u^2"


@pytest.fixture(scope="module")
def lib() -> library.MetalLibrary:
    return library.get_library("df64")


# ---------------------------------------------------------------------------
# encode.py
# ---------------------------------------------------------------------------
def test_encode_df64_roundtrip_and_specials():
    x = np.concatenate(
        [
            rand(10000, 3, 1e-30, 1e38),
            [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 3e38, -3e38],
        ]
    )
    hi, lo = encode.encode_df64(x)
    assert hi.dtype == np.float32 and lo.dtype == np.float32 and hi.shape == x.shape
    back = encode.decode_df64(hi, lo)
    finite = np.isfinite(x)
    rel = np.abs(back[finite] - x[finite]) / np.maximum(np.abs(x[finite]), 1e-300)
    # u^2 round trip while a flushed denormal lo word (GPU FTZ, < 2^-126) cannot
    # matter, i.e. |x| >= 2^-78 ~ 3e-24; below that float32 accuracy remains.
    lo_matters = np.abs(x[finite]) >= 2.0**-78
    assert np.max(rel[lo_matters]) <= U2
    assert np.max(rel[~lo_matters]) <= 2.0**-24
    assert np.array_equal(np.isnan(back), np.isnan(x))
    assert np.array_equal(back[np.isinf(x)], x[np.isinf(x)])
    assert np.all(lo[~np.isfinite(hi)] == 0.0), "lo must be 0 where hi is inf/NaN"
    assert np.signbit(back[np.flatnonzero(x == 0)]).tolist() == [False, True]
    # |lo| <= ulp(hi)/2 (normalized pair)
    ok = finite & (hi != 0)
    assert np.all(np.abs(lo[ok]) <= np.spacing(np.abs(hi[ok])) / 2)
    # values beyond float32 range become inf with lo = 0
    hi_big, lo_big = encode.encode_df64(np.array([1e300, -1e300]))
    assert hi_big.tolist() == [np.inf, -np.inf] and lo_big.tolist() == [0.0, 0.0]
    # float32-denormal magnitudes are flushed to a signed zero (the GPU treats
    # them as zero in arithmetic and comparisons but would store them unchanged)
    hi_d, lo_d = encode.encode_df64(np.array([1e-40, -1e-40, 1.4e-45, 2**-126]))
    assert hi_d.tolist() == [0.0, 0.0, 0.0, 2**-126] and lo_d.tolist() == [0.0] * 4
    assert np.signbit(hi_d).tolist() == [False, True, False, False]
    # a denormal lo word is flushed too (value 1e-30 has lo ~ 3e-39)
    hi_l, lo_l = encode.encode_df64(np.array([1e-30]))
    assert hi_l[0] == np.float32(1e-30) and lo_l[0] == 0.0
    # 0-d input stays 0-d (df64_scalar relies on it)
    assert encode.encode_df64(np.float64(1.5))[0].shape == ()


def test_encode_df64_canonical_tie_form():
    """hi = RN32(x), lo = RN32(x - hi) can leave lo = +-ulp(hi)/2 next to an odd hi.

    The encoder renormalizes with a float32 Fast2Sum so that both float64 values
    that round to the same 48-bit value get the identical (even-hi) pair the
    kernels produce; the value of the pair is unchanged.
    """
    x1 = float.fromhex("0x1.000002fffffffp+0")  # hi 1+2^-23 (odd), lo +2^-24
    x2 = float.fromhex("0x1.0000030000001p+0")  # hi 1+2^-22 (even), lo -2^-24
    y = 2.0**23 + 1.5 - 2.0**-29  # hi 8388609 (odd), lo +0.5
    hi, lo = encode.encode_df64(np.array([x1, x2, y, -x1, 1.0 + 2.0**-30]))
    assert (
        hi.tolist()[:2] == [1.0 + 2.0**-22] * 2 and lo.tolist()[:2] == [-(2.0**-24)] * 2
    )
    assert hi[2] == 8388610.0 and lo[2] == -0.5
    assert hi[3] == -(1.0 + 2.0**-22) and lo[3] == 2.0**-24
    assert hi[4] == 1.0 and lo[4] == np.float32(2.0**-30)
    assert np.all(np.abs(lo) <= np.spacing(np.abs(hi)) / 2)
    back = encode.decode_df64(hi, lo)
    assert back[0] == back[1] == float.fromhex("0x1.000003p+0")
    assert back[2] == 8388609.5 and back[3] == -back[0]


def test_encode_sf64_roundtrip_and_scalars():
    x = np.concatenate(
        [rand(1000, 4, 1e-300, 1e300), [0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324]]
    )
    bits = encode.encode_sf64(x)
    assert bits.dtype == np.int64
    back = encode.decode_sf64(bits)
    assert np.array_equal(
        back.view(np.int64), x.view(np.int64)
    )  # bit exact incl. NaN/-0
    assert encode.sf64_scalar(1.0) == 0x3FF0000000000000
    assert encode.sf64_scalar(-0.0) == -(1 << 63)
    assert encode.sf64_scalar(-1.0) == int(np.int64(-4616189618054758400))
    s = encode.df64_scalar(1.0 + 2.0**-30)
    assert isinstance(s, list) and len(s) == 2 and all(isinstance(v, float) for v in s)
    assert s == [1.0, 2.0**-30]
    assert encode.df64_scalar(np.inf) == [np.inf, 0.0]
    assert (
        np.isnan(encode.df64_scalar(np.nan)[0]) and encode.df64_scalar(np.nan)[1] == 0.0
    )


def test_torch_transfer_helpers():
    x = rand(1000, 5)
    hi, lo = encode.to_mps_df64(x)
    assert hi.device.type == "mps" and hi.dtype == torch.float32 and hi.is_contiguous()
    assert np.array_equal(encode.from_mps_df64(hi, lo), decoded(x))
    bits = encode.to_mps_sf64(torch.from_numpy(x))
    assert bits.dtype == torch.int64 and bits.device.type == "mps"
    assert np.array_equal(encode.from_mps_sf64(bits), x)


# ---------------------------------------------------------------------------
# codegen.py
# ---------------------------------------------------------------------------
def test_codegen_table_and_names():
    assert set(codegen.UNARY) | set(codegen.BINARY) | set(codegen.TERNARY) == set(
        codegen.OPS
    )
    assert codegen.kernel_name("add", "df64") == "add_df64"
    assert codegen.kernel_name("add", "df64", "right") == "add_df64_s"
    assert codegen.kernel_name("add", "sf64", "left") == "add_sf64_rs"
    with pytest.raises(ValueError):
        codegen.kernel_name("neg", "df64", "right")
    with pytest.raises(ValueError):
        codegen.kernel_name("add", "f64")
    core = codegen.ops_for_headers("df64", CORE_HEADERS)
    assert "add" in core and "where" in core and "exp" not in core
    with_exp = CORE_HEADERS + ("df64_math_exp.h",)
    # table order is kept regardless of the order of the requested subset
    assert codegen.ops_for_headers("df64", with_exp, ["exp", "add"]) == ["exp", "add"]
    assert codegen.ops_for_headers("df64", CORE_HEADERS, ["exp", "add"]) == ["add"]
    with pytest.raises(KeyError):
        codegen.ops_for_headers("df64", CORE_HEADERS, ["nope"])


def test_codegen_full_source_and_cli(tmp_path):
    text = codegen.build()
    for repr_ in codegen.REPRS:
        for name in codegen.kernel_names(None, repr_):
            assert f"kernel void {name}(" in text
    assert (
        "ew::sf_load" in text
        and "constant float2& b_s" in text
        and "constant long& a_s" in text
    )
    assert text.startswith("// ---- elementwise kernels")
    assert "#pragma METAL fp math_mode(safe)" in text
    only_df = codegen.build(["add"], reprs=("df64",))
    assert "add_sf64" not in only_df and "add_df64_rs" in only_df
    out = tmp_path / "gen.metal"
    assert codegen.main(["--out", str(out)]) == 0
    assert out.read_text() == text
    assert codegen.GENERATED_FILE.is_file(), (
        "run python -m ...codegen to refresh the inspection file"
    )


# ---------------------------------------------------------------------------
# library.py: assembly, self-test, compile
# ---------------------------------------------------------------------------
def test_selftest_passes(lib):
    assert lib.selftest_values is not None
    assert library.check_selftest(lib.selftest_values) == []
    assert library.check_selftest(lib.run_selftest()) == []
    assert lib.headers[:2] == CORE_HEADERS


def test_check_selftest_detects_every_probe(lib):
    good = list(lib.selftest_values)
    assert library.check_selftest(good) == []
    corruptions = {
        0: 0.0,
        1: 0.0,
        2: 0.6666666,
        3: 1.0,
        4: 1e-9,
        5: 0.5,
        6: 0.0,
        7: 0.0,
    }
    for idx, bad in corruptions.items():
        v = list(good)
        v[idx] = bad
        problems = library.check_selftest(v)
        assert problems, f"corrupting out[{idx}] went undetected"
        assert any(f"out[{idx}]" in p or "df::div" in p for p in problems)
    assert library.check_selftest(good[:4])


def test_all_available_kernels_compiled(lib):
    expected_ops = codegen.ops_for_headers(
        "df64", [os.path.basename(h) for h in lib.headers]
    )
    assert list(lib.ops) == expected_ops
    assert set(lib.kernel_names) == set(codegen.kernel_names(lib.ops, "df64"))
    for name in lib.kernel_names:
        assert callable(getattr(lib.lib, name)), name
    for op in lib.ops:
        assert lib.has(op)
    assert "df64_selftest" in lib.source and "kernel void add_df64_s(" in lib.source


def test_get_library_singleton(lib):
    assert library.get_library("df64") is lib
    with pytest.raises(ValueError):
        library.MetalLibrary("f64")
    with pytest.raises(ValueError):
        library.MetalLibrary("df64", chunk_size=0)


# ---------------------------------------------------------------------------
# ops vs numpy on random data
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("op", sorted(UNARY_REF))
def test_unary_ops(lib, op, n):
    ref_fn, tol = UNARY_REF[op]
    positive = op in ("sqrt", "rsqrt")
    x = decoded(rand(n, 11, positive=positive))
    if op in ("floor", "ceil", "trunc", "rint", "round_away"):
        # integer-rounding ops: mix fractional values, exact .5 ties and integers
        x = decoded(
            np.concatenate(
                [rand(n, 12, 1e-3, 1e6), np.arange(n) + 0.5, -(np.arange(n) + 0.5)]
            )[:n]
        )
    got = encode.from_mps_df64(*lib.launch(op, gpu(x)))
    assert_close(got, ref_fn(x), tol, check_signbit=(op != "sign"))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("op", sorted(BINARY_REF))
def test_binary_ops(lib, op, n):
    ref_fn, tol = BINARY_REF[op]
    a = decoded(rand(n, 21))
    b = decoded(rand(n, 22))
    got = encode.from_mps_df64(*lib.launch(op, gpu(a), gpu(b)))
    scale = a if op in SCALE_BY_FIRST else None
    assert_close(got, ref_fn(a, b), tol, scale=scale)


@pytest.mark.parametrize("op", sorted(TERNARY_REF))
def test_ternary_ops(lib, op):
    ref_fn, tol, scale_fn = TERNARY_REF[op]
    n = 1 << 20
    a, b, c = (decoded(rand(n, s)) for s in (31, 32, 33))
    if op == "clamp":
        b, c = np.minimum(b, c), np.maximum(b, c)
    got = encode.from_mps_df64(*lib.launch(op, gpu(a), gpu(b), gpu(c)))
    scale = scale_fn(a, b, c) if scale_fn is not None else None
    assert_close(got, ref_fn(a, b, c), tol, scale=scale)


def test_where(lib):
    n = 100_003
    a, b = decoded(rand(n, 41)), decoded(rand(n, 42))
    cond = np.random.default_rng(43).random(n) < 0.5
    ct = torch.from_numpy(cond).to("mps")
    got = encode.from_mps_df64(*lib.launch("where", ct, gpu(a), gpu(b)))
    assert_close(got, np.where(cond, a, b), EXACT)


@pytest.mark.parametrize("op", sorted(BINARY_PRED_REF) + sorted(UNARY_PRED_REF))
def test_bool_outputs(lib, op):
    n = 50_001
    a = decoded(rand(n, 51))
    b = a.copy()
    rng = np.random.default_rng(52)
    idx = rng.random(n) < 0.6
    b[idx] = decoded(rand(int(idx.sum()), 53))
    # nudge some equal pairs by one df64 ulp so lo alone decides the comparison
    tweak = ~idx & (rng.random(n) < 0.5)
    b[tweak] = b[tweak] * (1 + 2.0**-47)
    a[:3] = [np.nan, np.inf, -np.inf]
    b[:3] = [np.nan, np.inf, np.inf]
    if op in UNARY_PRED_REF:
        got = lib.launch(op, gpu(a))
        ref = UNARY_PRED_REF[op](a)
    else:
        got = lib.launch(op, gpu(a), gpu(b))
        ref = BINARY_PRED_REF[op](a, b)
    assert got.dtype == torch.bool and got.shape == (n,)
    assert np.array_equal(got.cpu().numpy(), ref)


@pytest.mark.parametrize(
    "op", ["add", "sub", "mul", "div", "copysign", "minimum", "fmod", "lt", "ge"]
)
@pytest.mark.parametrize("side", ["right", "left"])
def test_scalar_variants(lib, op, side):
    n = 100_001
    x = decoded(rand(n, 61))
    s = -2.718281828459045
    s_dec = float(decoded(np.array([s]))[0])
    if op in BINARY_PRED_REF:
        ref_fn, tol = BINARY_PRED_REF[op], EXACT
    else:
        ref_fn, tol = BINARY_REF[op]
    a, b = (x, s_dec) if side == "right" else (s_dec, x)
    res = lib.launch(op, gpu(x), scalar=s, scalar_side=side)
    ref = ref_fn(a, b)
    if op in BINARY_PRED_REF:
        assert np.array_equal(res.cpu().numpy(), ref)
    else:
        scale = (
            np.broadcast_to(np.asarray(a), x.shape) if op in SCALE_BY_FIRST else None
        )
        assert_close(encode.from_mps_df64(*res), ref, tol, scale=scale)
    # default scalar side is right
    if side == "right" and op not in BINARY_PRED_REF:
        again = lib.launch(op, gpu(x), scalar=s)
        assert np.array_equal(encode.from_mps_df64(*again), encode.from_mps_df64(*res))


def test_broadcasting_and_noncontiguous(lib):
    a = decoded(rand(5 * 1, 71).reshape(5, 1))
    b = decoded(rand(1 * 7, 72).reshape(1, 7))
    ah, al = gpu(a)
    bh, bl = gpu(b)
    ah, al, bh, bl = library.broadcast_contiguous(ah, al, bh, bl)
    assert ah.shape == (5, 7) and all(t.is_contiguous() for t in (ah, al, bh, bl))
    oh, ol = lib.launch("mul", (ah, al), (bh, bl))
    assert oh.shape == (5, 7)
    assert_close(encode.from_mps_df64(oh, ol), a * b, 5.0)

    # non-contiguous inputs: transposed and strided views
    m = decoded(rand(40 * 30, 73).reshape(40, 30))
    mh, ml = gpu(m)
    th, tl = mh.t(), ml.t()  # (30, 40), non-contiguous
    sh, sl = mh[::2, ::3], ml[::2, ::3]  # (20, 10), non-contiguous
    assert not th.is_contiguous() and not sh.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        lib.launch("neg", (th, tl))
    th, tl = library.broadcast_contiguous(th, tl)
    got = encode.from_mps_df64(*lib.launch("neg", (th, tl)))
    assert np.array_equal(got, -m.T)
    sh, sl = library.broadcast_contiguous(sh, sl)
    got = encode.from_mps_df64(*lib.launch("sqrt", (sh.abs(), sl * torch.sign(sh))))
    assert_close(got, np.sqrt(np.abs(m[::2, ::3])), 4.0)
    # a contiguous slice with a storage offset is accepted as-is
    off_h, off_l = mh[10:], ml[10:]
    assert off_h.storage_offset() > 0 and off_h.is_contiguous()
    got = encode.from_mps_df64(*lib.launch("abs", (off_h, off_l)))
    assert np.array_equal(got, np.abs(m[10:]))


# decoded so the oracle sees exactly what the kernel sees (2.5, 3e-7, ... are inexact)
SPECIALS = decoded(
    np.array(
        [
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
            1.0,
            -1.0,
            0.5,
            -2.5,
            1e10,
            -3e-7,
            7.0,
            3.0,
        ]
    )
)
# df64_core.h follows IEEE-754 / numpy for the sign of zero results (abs(-0) = +0,
# add(-0, -0) = -0, mul(-0, 1) = -0, copysign(-0, +1) = +0, fmod(-0, 1) = -0,
# minimum(+0, -0) = -0, floor/ceil/trunc/rint/round_away(-0) = -0), for the
# rounding of +-inf and for fmod/remainder_py(x, +-inf) = x; these are asserted
# by the special-value tests below and by test_core_header_ieee_semantics.
# sign(-0) = -0 is intentional torch semantics (numpy.sign(-0.0) is +0).
ZERO_SIGN_UNSPECIFIED = {"sign"}
ROUNDING_OPS = {"floor", "ceil", "trunc", "rint", "round_away"}


@pytest.mark.parametrize("op", sorted(UNARY_REF) + sorted(UNARY_PRED_REF))
def test_unary_special_values(lib, op):
    x = SPECIALS.copy()
    with np.errstate(all="ignore"):
        if op in UNARY_PRED_REF:
            assert np.array_equal(
                lib.launch(op, gpu(x)).cpu().numpy(), UNARY_PRED_REF[op](x)
            )
            return
        ref_fn, tol = UNARY_REF[op]
        ref = ref_fn(x)
    got = encode.from_mps_df64(*lib.launch(op, gpu(x)))
    # sqrt(-x) must be NaN, never 0
    if op in ("sqrt", "rsqrt"):
        assert np.all(np.isnan(got[x < 0]))
    assert_close(got, ref, tol, check_signbit=(op not in ZERO_SIGN_UNSPECIFIED))
    if op == "sign":  # torch semantics: sign(-0.0) == -0.0
        assert np.signbit(got[1]) and not np.signbit(got[0])


@pytest.mark.parametrize("op", sorted(BINARY_REF) + sorted(BINARY_PRED_REF))
def test_binary_special_values(lib, op):
    a, b = np.meshgrid(SPECIALS, SPECIALS, indexing="ij")
    a, b = a.ravel(), b.ravel()
    with np.errstate(all="ignore"):
        if op in BINARY_PRED_REF:
            got = lib.launch(op, gpu(a), gpu(b)).cpu().numpy()
            assert np.array_equal(got, BINARY_PRED_REF[op](a, b))
            return
        ref_fn, tol = BINARY_REF[op]
        ref = ref_fn(a, b)
    got = encode.from_mps_df64(*lib.launch(op, gpu(a), gpu(b)))
    scale = a if op in SCALE_BY_FIRST else None
    check_sign = op not in ZERO_SIGN_UNSPECIFIED
    assert_close(got, ref, tol, scale=scale, check_signbit=check_sign)
    if check_sign and tol != EXACT:
        zero = (ref == 0) & ~np.isnan(ref)
        assert np.array_equal(np.signbit(got[zero]), np.signbit(ref[zero])), (
            "signed zero differs"
        )


def test_ternary_special_values(lib):
    x = SPECIALS
    a, b, c = np.meshgrid(x, x, x, indexing="ij")
    a, b, c = a.ravel(), b.ravel(), c.ravel()
    with np.errstate(all="ignore"):
        for op in ("fma", "addcmul", "clamp"):
            ref_fn, tol, scale_fn = TERNARY_REF[op]
            if op == "clamp":
                lo, hi = np.minimum(b, c), np.maximum(b, c)
                lo, hi = np.where(np.isnan(lo), b, lo), np.where(np.isnan(hi), c, hi)
                ref = ref_fn(a, lo, hi)
                got = encode.from_mps_df64(*lib.launch(op, gpu(a), gpu(lo), gpu(hi)))
            else:
                ref = ref_fn(a, b, c)
                got = encode.from_mps_df64(*lib.launch(op, gpu(a), gpu(b), gpu(c)))
            scale = scale_fn(a, b, c) if scale_fn is not None else None
            assert_close(
                got,
                ref,
                tol,
                scale=scale,
                check_signbit=(op not in ZERO_SIGN_UNSPECIFIED),
            )
    cond = np.array([True, False, True, False, True] * 3)[: x.size]
    got = encode.from_mps_df64(
        *lib.launch("where", torch.from_numpy(cond).to("mps"), gpu(x), gpu(-x))
    )
    assert_close(got, np.where(cond, x, -x), EXACT)


def test_core_header_ieee_semantics(lib):
    """Strict IEEE / numpy semantics of the core header (signed zeros, inf)."""
    z = np.array([-0.0, 0.0, 1.0, -0.0, -0.0, 0.0])
    w = np.array([1.0, -1.0, np.inf, -0.0, 1.0, -0.0])

    def run(op, *args):
        return encode.from_mps_df64(*lib.launch(op, *map(gpu, args)))

    assert not np.signbit(run("abs", z)[0]), "abs(-0.0) must be +0.0"
    assert np.signbit(run("mul", z, w)[0]), "mul(-0.0, 1.0) must be -0.0"
    assert np.signbit(run("mul", z, w)[1]), "mul(0.0, -1.0) must be -0.0"
    assert np.signbit(run("add", z, w)[3]), "add(-0.0, -0.0) must be -0.0"
    assert np.signbit(run("div", z, w)[4]), "div(-0.0, 1.0) must be -0.0"
    assert not np.signbit(run("copysign", z, w)[0]), "copysign(-0.0, 1.0) must be +0.0"
    assert np.signbit(run("fmod", z, w)[4]), "fmod(-0.0, 1.0) must be -0.0"
    inf = np.array([np.inf, -np.inf])
    for op in sorted(ROUNDING_OPS):
        assert np.array_equal(run(op, inf), inf), f"{op}(+-inf) must be +-inf"
        assert np.signbit(run(op, z)[0]), f"{op}(-0.0) must be -0.0"
    assert np.array_equal(run("fmod", np.array([1.0, -2.5]), inf), [1.0, -2.5]), (
        "fmod(x, inf) = x"
    )
    assert np.array_equal(run("remainder_py", np.array([1.0]), inf[:1]), [1.0]), (
        "rem(x, inf) = x"
    )
    # Python: -1.0 % inf = inf, 1.0 % -inf = -inf, 0.0 % -inf = -0.0
    got = run(
        "remainder_py", np.array([-1.0, 1.0, 0.0]), np.array([np.inf, -np.inf, -np.inf])
    )
    assert np.array_equal(got, [np.inf, -np.inf, 0.0]) and np.signbit(got[2])
    # +-0 ties: minimum -> -0, maximum -> +0 (IEEE 754-2019 / numpy), both orders
    for op, want in (
        ("minimum", True),
        ("maximum", False),
        ("fmin", True),
        ("fmax", False),
    ):
        got = run(op, np.array([0.0, -0.0]), np.array([-0.0, 0.0]))
        assert np.all(np.signbit(got) == want), f"{op}(+0, -0) tie"


def test_fmod_remainder_invariants_near_multiples(lib):
    """C99 fmod keeps the sign of a and |r| < |b|; Python remainder the sign of b.

    Inputs one or two df64 ulps away from k*b make the rounded quotient a/b land
    on the wrong side of the integer k; the header's guards must restore the
    invariants (the result is then b-adjacent, as the exact remainder is for an
    input perturbed by one ulp, so only the invariants are asserted here).
    """
    rng = np.random.default_rng(5)
    n = 100_000
    b = decoded(rng.uniform(0.5, 4, n) * 10.0 ** rng.integers(-6, 6, n))
    b *= rng.choice([-1.0, 1.0], n)
    k = np.floor(2.0 ** rng.uniform(0, 22, n))
    prod = decoded(k * b)
    ulp = 2.0 ** (np.floor(np.log2(np.abs(prod))) - 47)
    a = decoded(prod + rng.choice([-1.0, 1.0], n) * ulp * rng.integers(1, 3, n))
    a *= rng.choice([-1.0, 1.0], n)
    r = encode.from_mps_df64(*lib.launch("fmod", gpu(a), gpu(b)))
    nz = r != 0
    assert np.all(np.signbit(r) == np.signbit(a)), "fmod sign must follow a"
    assert np.all(np.abs(r[nz]) < np.abs(b[nz])), "|fmod| < |b|"
    r = encode.from_mps_df64(*lib.launch("remainder_py", gpu(a), gpu(b)))
    nz = r != 0
    assert np.all(np.signbit(r) == np.signbit(b)), "remainder sign must follow b"
    assert np.all(np.abs(r[nz]) < np.abs(b[nz])), "|remainder| < |b|"


# ---------------------------------------------------------------------------
# launcher mechanics
# ---------------------------------------------------------------------------
def test_chunked_dispatch_matches_single_launch(lib):
    small = library.MetalLibrary("df64", ops=lib.ops, chunk_size=1000, selftest=False)
    assert small.lib is lib.lib  # same source hash -> cached compile
    for n in (12_345, (1 << 20) + 3):
        a, b = decoded(rand(n, 81)), decoded(rand(n, 82))
        ref_h, ref_l = lib.launch("add", gpu(a), gpu(b))
        got_h, got_l = small.launch("add", gpu(a), gpu(b))
        assert torch.equal(ref_h, got_h) and torch.equal(ref_l, got_l)
        assert_close(encode.from_mps_df64(got_h, got_l), a + b, 3.0)
        # scalar and predicate variants also split correctly
        assert torch.equal(
            small.launch("lt", gpu(a), gpu(b)), lib.launch("lt", gpu(a), gpu(b))
        )
        s_small = small.launch("mul", gpu(a), scalar=1.5, scalar_side="left")
        s_ref = lib.launch("mul", gpu(a), scalar=1.5, scalar_side="left")
        assert torch.equal(s_small[0], s_ref[0]) and torch.equal(s_small[1], s_ref[1])
    # 2-D input keeps its shape after a chunked launch
    m = decoded(rand(64 * 64, 83).reshape(64, 64))
    oh, ol = small.launch("neg", gpu(m))
    assert oh.shape == (64, 64)
    assert np.array_equal(encode.from_mps_df64(oh, ol), -m)


def test_out_argument_and_validation(lib):
    x = decoded(rand(1000, 91))
    xh, xl = gpu(x)
    oh = torch.empty_like(xh)
    ol = torch.empty_like(xl)
    res = lib.launch("neg", (xh, xl), out=(oh, ol))
    assert res[0] is oh and res[1] is ol
    assert np.array_equal(encode.from_mps_df64(oh, ol), -x)
    ob = torch.empty(1000, dtype=torch.bool, device="mps")
    assert lib.launch("isnan", (xh, xl), out=ob) is ob
    assert not ob.any()

    with pytest.raises(KeyError):
        lib.launch("nope", (xh, xl))
    if not lib.has("exp"):
        with pytest.raises(KeyError, match="df64_math_exp.h"):
            lib.launch("exp", (xh, xl))
    with pytest.raises(TypeError):
        lib.launch("neg", xh)  # df64 operands are (hi, lo) tuples
    with pytest.raises(TypeError):
        lib.launch("neg", (xh.to(torch.int32), xl))  # wrong dtype
    with pytest.raises(ValueError, match="shape"):
        lib.launch("add", (xh, xl), gpu(x[:10]))
    with pytest.raises(ValueError, match="shape"):
        lib.launch("neg", (xh, xl), out=(oh[:10], ol[:10]))
    with pytest.raises(TypeError):
        lib.launch("add", (xh, xl))  # missing operand
    with pytest.raises(TypeError):
        lib.launch("add", (xh, xl), (xh, xl), scalar=1.0)  # too many with scalar
    with pytest.raises(ValueError):
        lib.launch("add", (xh, xl), (xh, xl), scalar_side="right")
    with pytest.raises(ValueError):
        lib.launch("neg", (xh, xl), scalar=1.0)  # unary has no scalar variant
    with pytest.raises(ValueError, match="mps"):
        lib.launch("neg", (xh.cpu(), xl.cpu()))
    with pytest.raises(TypeError):
        lib.launch("where", (xh, xl), (xh, xl), (xh, xl))  # cond must be bool


def test_empty_input(lib):
    e = torch.empty(0, dtype=torch.float32, device="mps")
    oh, ol = lib.launch("add", (e, e), (e, e))
    assert oh.numel() == 0 and ol.numel() == 0
    assert lib.launch("lt", (e, e), (e, e)).numel() == 0


# ---------------------------------------------------------------------------
# fast-math guard in a fresh process
# ---------------------------------------------------------------------------
def test_fast_math_env_fails_loudly():
    code = (
        "from optiland.backend.torch_backend.metal import library\n"
        "library.get_library('df64')\n"
        "print('LIBRARY LOADED')\n"
    )
    env = dict(os.environ, PYTORCH_MPS_FAST_MATH="1")
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "LIBRARY LOADED" not in proc.stdout
    assert "PYTORCH_MPS_FAST_MATH" in proc.stderr
    assert metal_compile.MetalMathModeError.__name__ in proc.stderr

    # the same code with the variable at "0" loads fine
    env["PYTORCH_MPS_FAST_MATH"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "LIBRARY LOADED" in proc.stdout


# ---------------------------------------------------------------------------
# sf64 (runs only once the softfloat headers land)
# ---------------------------------------------------------------------------
def test_sf64_library_if_available():
    kernels = Path(metal_compile._KERNEL_DIR)
    missing = [
        h for h in library.REQUIRED_HEADERS["sf64"] if not (kernels / h).is_file()
    ]
    if missing:
        with pytest.raises(FileNotFoundError):
            library.MetalLibrary("sf64")
        pytest.skip(f"sf64 headers not present yet: {missing}")
    sf = library.get_library("sf64")
    assert sf.repr == "sf64" and "add" in sf.ops
    n = 1 << 18
    a, b = rand(n, 101, 1e-300, 1e300), rand(n, 102, 1e-300, 1e300)
    for op, fn in (
        ("add", np.add),
        ("sub", np.subtract),
        ("mul", np.multiply),
        ("div", np.divide),
    ):
        got = encode.from_mps_sf64(
            sf.launch(op, encode.to_mps_sf64(a), encode.to_mps_sf64(b))
        )
        with np.errstate(all="ignore"):  # overflow/underflow are part of the test
            ref = fn(a, b)
        assert np.array_equal(got.view(np.int64), ref.view(np.int64)), (
            f"sf64 {op} not bit-exact"
        )
    got = encode.from_mps_sf64(sf.launch("mul", encode.to_mps_sf64(a), scalar=3.0))
    assert np.array_equal(got, a * 3.0)


# ---------------------------------------------------------------------------
# full stacks: every op of the table, both representations, reduce + matmul
# ---------------------------------------------------------------------------
FULL_STACK_HEADERS = {
    "df64": (
        "df64_core.h",
        "df64_constants.h",
        "df64_math_exp.h",
        "df64_math_trig.h",
        "df64_math_special.h",
    ),
    "sf64": (
        "vendor/softfloat64.metal",
        "df64_core.h",
        "df64_constants.h",
        "df64_math_exp.h",
        "df64_math_trig.h",
        "df64_math_special.h",
        "sf64_core.h",
        "sf64_math.h",
    ),
}
REDUCE_MATMUL_KERNELS = ("sum", "nansum", "prod", "max", "min", "argmax", "cumsum")


@pytest.mark.parametrize("repr", ["df64", "sf64"])
def test_full_stack_compiles_every_op(repr):
    lib_r = library.get_library(repr)
    assert lib_r.headers == FULL_STACK_HEADERS[repr], lib_r.missing_headers
    assert lib_r.missing_headers == ()
    assert (
        lib_r.kernel_files == library.KERNEL_FILES == ("reduce.metal", "matmul.metal")
    )
    assert list(lib_r.ops) == list(codegen.OPS), "every op of the table is compiled"
    assert len(lib_r.kernel_names) == len(codegen.kernel_names(None, repr))
    for name in lib_r.kernel_names:
        assert callable(getattr(lib_r.lib, name)), name
    for base in REDUCE_MATMUL_KERNELS:
        assert callable(getattr(lib_r.lib, f"{base}_{repr}")), base
    assert callable(getattr(lib_r.lib, f"matmul_{repr}"))
    assert callable(getattr(lib_r.lib, f"dot_{repr}"))
    # every header (and kernel file) begins with the safe-math pragmas: the
    # first lines after the leading comment block and include guard must be them
    for h in [*lib_r.headers, *lib_r.kernel_files]:
        if h.startswith("vendor/"):
            continue
        lines = (Path(metal_compile._KERNEL_DIR) / h).read_text().splitlines()
        code = [
            ln.strip()
            for ln in lines
            if ln.strip()
            and not ln.lstrip().startswith("//")
            and not ln.startswith(("#ifndef", "#define"))
        ]
        assert code[:2] == [
            "#pragma METAL fp math_mode(safe)",
            "#pragma METAL fp contract(off)",
        ], h


def test_codegen_sf64_bridge_needs_df64_header():
    have = ["vendor/softfloat64.metal", "df64_core.h", "sf64_core.h", "sf64_math.h"]
    assert "exp" not in codegen.ops_for_headers("sf64", have)
    assert "add" in codegen.ops_for_headers("sf64", have)
    assert "exp" in codegen.ops_for_headers("sf64", [*have, "df64_math_exp.h"])
    assert "sin" not in codegen.ops_for_headers("sf64", [*have, "df64_math_exp.h"])
    assert codegen.headers_for("erf", "sf64") == ("sf64_math.h", "df64_math_special.h")
    assert codegen.headers_for("erf", "df64") == ("df64_math_special.h",)


TRANSCENDENTAL_SMOKE = {
    # op: (reference, inputs)  -- df64 kernels through launch(); tolerance is the
    # per-family target of the design brief, the ULP suites measure precisely.
    "exp": (np.exp, np.array([-20.0, -1.0, 0.0, 0.5, 3.0, 20.0])),
    "log": (np.log, np.array([1e-3, 0.5, 2.0, 10.0, 1e10])),
    "sin": (np.sin, np.array([-10.0, -1.0, 0.25, 1.0, 100.0])),
    "atan": (np.arctan, np.array([-7.0, -0.5, 0.25, 3.0])),
    "sinh": (np.sinh, np.array([-3.0, 0.5, 2.0])),
    "erf": (
        lambda x: np.array([float(_erf(v)) for v in x]),
        np.array([-1.5, 0.3, 2.0]),
    ),
    "lgamma": (
        lambda x: np.array([float(_lgamma(v)) for v in x]),
        np.array([0.5, 3.5, 12.0, 45.0]),
    ),
}


def _erf(x: float) -> float:
    import math

    return math.erf(x)


def _lgamma(x: float) -> float:
    import math

    return math.lgamma(x)


@pytest.mark.parametrize("op", sorted(TRANSCENDENTAL_SMOKE))
@pytest.mark.parametrize("repr", ["df64", "sf64"])
def test_transcendental_kernels_smoke(repr, op):
    ref_fn, x = TRANSCENDENTAL_SMOKE[op]
    lib_r = library.get_library(repr)
    if repr == "df64":
        x = decoded(x)
        got = encode.from_mps_df64(*lib_r.launch(op, gpu(x)))
    else:
        got = encode.from_mps_sf64(lib_r.launch(op, encode.to_mps_sf64(x)))
    ref = ref_fn(x)
    # sf64 goes through the 48-bit bridge: input + output rounding + df64 error.
    assert_close(got, ref, 100.0)


# ---------------------------------------------------------------------------
# Regression tests for the fix round 1 findings on df64_core.h (adversarial
# verification, NOTES/04-kernel-status.md section 8). Hand-built (hi, lo) pairs
# are passed straight to ``launch`` so that the exact representation reaches
# the kernel; internal helpers get a small test kernel of their own.
# ---------------------------------------------------------------------------
def _f32(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint32).view(np.float32)[0])


def _pairs(*pairs):
    hi = np.array([p[0] for p in pairs], dtype=np.float32)
    lo = np.array([p[1] for p in pairs], dtype=np.float32)
    return torch.from_numpy(hi).to("mps"), torch.from_numpy(lo).to("mps")


def _launch_pairs(lib, op, a, b=None):
    args = [_pairs(*a)] if b is None else [_pairs(*a), _pairs(*b)]
    out = lib.launch(op, *args)
    if isinstance(out, tuple):
        return out[0].cpu().numpy().astype(np.float64), out[1].cpu().numpy().astype(
            np.float64
        )
    return out.cpu().numpy()


FLT_MAX = _f32(0x7F7FFFFF)

_CORE_HELPER_KERNELS = r"""
kernel void t_to_int(device const float* ah [[buffer(0)]],
                     device const float* al [[buffer(1)]],
                     device int* o [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    o[i] = df::to_int(df::make(ah[i], al[i]));
}
kernel void t_to_float2(device const long* b [[buffer(0)]],
                        device float* oh [[buffer(1)]], device float* ol [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    float2 p = sf::to_float2(sf::make((ulong)b[i])); oh[i] = p.x; ol[i] = p.y;
}
kernel void t_from_int(device const int* n [[buffer(0)]],
                       device float* oh [[buffer(1)]], device float* ol [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    df64 r = df::from_int(n[i]); oh[i] = r.hi; ol[i] = r.lo;
}
kernel void t_helpers(device const float* ah [[buffer(0)]],
                      device const float* al [[buffer(1)]],
                      device const int* which [[buffer(2)]],
                      device const int* arg [[buffer(3)]],
                      device float* oh [[buffer(4)]], device float* ol [[buffer(5)]],
                      device int* oi [[buffer(6)]],
                      uint i [[thread_position_in_grid]]) {
    df64 a = df::make(ah[i], al[i]);
    df64 r;
    oi[i] = 0;
    switch (which[i]) {
        case 0: r = df::ldexp(a, arg[i]); break;
        case 1: r = df::mul_pwr2(a, float(arg[i])); break;
        case 2: r = df::renorm(a); break;
        case 3: { int e; r = df::frexp(a, e); oi[i] = e; break; }
        case 4: r = df::div(a, float(arg[i])); break;
        case 6: r = df::sqr(a); break;
        case 7: r = df::mul(a, as_type<float>(arg[i])); break;
        case 8: r = df::canon(a); break;
        case 9: r = df::div(a, as_type<float>(arg[i])); break;
        default: r = df::make(df::eq(a, df::inf()) ? 1.0f : 0.0f,
                              df::is_finite(a) ? 1.0f : 0.0f); break;
    }
    oh[i] = r.hi; ol[i] = r.lo;
}
"""


@pytest.fixture(scope="module")
def core_helpers():
    source = (
        metal_compile.kernel_source(
            "vendor/softfloat64.metal", *CORE_HEADERS, "sf64_core.h"
        )
        + _CORE_HELPER_KERNELS
    )
    return metal_compile.compile_library(source)


def _run_helper(hlib, which, pairs, args=None):
    n = len(pairs)
    hi, lo = _pairs(*pairs)
    w = torch.full((n,), which, dtype=torch.int32, device="mps")
    a = torch.tensor(args or [0] * n, dtype=torch.int32, device="mps")
    oh = torch.empty(n, dtype=torch.float32, device="mps")
    ol = torch.empty(n, dtype=torch.float32, device="mps")
    oi = torch.empty(n, dtype=torch.int32, device="mps")
    hlib.t_helpers(hi, lo, w, a, oh, ol, oi, threads=[n, 1, 1])
    return (
        oh.cpu().numpy().astype(np.float64),
        ol.cpu().numpy().astype(np.float64),
        oi.cpu().numpy(),
    )


def test_add_sub_overflow_in_recombination_gives_inf(lib):
    """FLT_MAX + 0.4 ulp + 0.2 ulp overflowed to NaN in the second Fast2Sum."""
    a = [
        (FLT_MAX, _f32(0x72CCCCCD)),
        (FLT_MAX, _f32(0x72800000)),
        (-FLT_MAX, -_f32(0x72CCCCCD)),
        (FLT_MAX, _f32(0x72800000)),
    ]
    b = [
        (_f32(0x724CCCCD), 0.0),
        (_f32(0x72800000), 0.0),
        (-_f32(0x724CCCCD), 0.0),
        (-_f32(0x72800000), 0.0),
    ]
    hi, lo = _launch_pairs(lib, "add", a, b)
    assert (
        hi.tolist() == [np.inf, np.inf, -np.inf, FLT_MAX] and lo.tolist() == [0.0] * 4
    )
    hi, lo = _launch_pairs(lib, "sub", a, [(-x, -y) for x, y in b])
    assert (
        hi.tolist() == [np.inf, np.inf, -np.inf, FLT_MAX] and lo.tolist() == [0.0] * 4
    )
    # a finite sum that needs the lo word of FLT_MAX stays finite and exact
    hi, lo = _launch_pairs(lib, "add", [(FLT_MAX, 0.0)], [(2.0**100, 0.0)])
    assert hi[0] == FLT_MAX and lo[0] == 2.0**100


def test_div_at_flt_max_is_finite(lib, core_helpers):
    """FLT_MAX / b returned NaN when RN(q1 * b) overflowed inside the residual."""
    rng = np.random.default_rng(17)
    divisors = np.concatenate(
        [[1e4, 100.0, 1e5, _f32(0x48BF079A), 1.0, 1.5], rng.uniform(1, 1e6, 2000)]
    ).astype(np.float32)
    a = [(FLT_MAX, 0.0)] * divisors.size + [(-FLT_MAX, 0.0)] * 3
    b = [(float(d), 0.0) for d in divisors] + [(1e4, 0.0), (100.0, 0.0), (1.5, 0.0)]
    hi, lo = _launch_pairs(lib, "div", a, b)
    ref = np.array([x[0] / y[0] for x, y in zip(a, b, strict=True)])
    assert np.all(np.isfinite(hi))
    assert_close(hi + lo, ref, 10.0)
    # df::div(df64, float) has the same guard
    hi, lo, _ = _run_helper(
        core_helpers,
        4,
        a[:6] + a[-3:],
        [int(d) for d in divisors[:6]] + [10000, 100, 1],
    )
    ref = np.array(
        [
            x[0] / float(int(y[0]))
            for x, y in zip(
                a[:6] + a[-3:], b[:6] + [(10000.0,), (100.0,), (1.0,)], strict=True
            )
        ]
    )
    assert np.all(np.isfinite(hi))
    assert_close(hi + lo, ref, 10.0)


def test_to_int_truncates_toward_zero_and_int32_edges(core_helpers):
    pairs = [
        (3.0, -1e-8),
        (1.0, -1e-8),
        (16777216.0, -0.5),
        (2147483648.0, -1.0),
        (-3.0, 1e-8),
        (3.0, -(2.0**-23)),
        (-3.0, 2.0**-23),
        (2147483648.0, -0.5),
        (-2147483648.0, 0.5),
        (2.5, 0.0),
        (-2.5, 0.0),
        (2147483008.0, -7.3),
        (1000000.0, -1e-9),
        (0.0, 0.0),
        (-0.0, 0.0),
        (123456.0, 0.001),
    ]
    hi, lo = _pairs(*pairs)
    o = torch.empty(len(pairs), dtype=torch.int32, device="mps")
    core_helpers.t_to_int(hi, lo, o, threads=[len(pairs), 1, 1])
    got = o.cpu().numpy().tolist()
    want = [
        int(np.trunc(np.float64(np.float32(h)) + np.float64(np.float32(low))))
        for h, low in pairs
    ]
    assert got == want, list(zip(pairs, got, want, strict=True))


def test_from_int_exact_near_int32_max(core_helpers):
    ns = [
        2147483647,
        2147483646,
        2147483584,
        2147483583,
        -2147483648,
        -2147483647,
        0,
        1,
        -1,
        12345678,
        16777217,
    ]
    n = torch.tensor(ns, dtype=torch.int32, device="mps")
    oh = torch.empty(len(ns), dtype=torch.float32, device="mps")
    ol = torch.empty(len(ns), dtype=torch.float32, device="mps")
    core_helpers.t_from_int(n, oh, ol, threads=[len(ns), 1, 1])
    got = [
        int(h) + int(low)
        for h, low in zip(oh.cpu().tolist(), ol.cpu().tolist(), strict=True)
    ]
    assert got == ns


def test_ceil_trunc_negative_zero(lib):
    """ceil/trunc of a value in (-1, 0) held as (hi -1, lo > 0) must give -0.0."""
    a = [(-1.0, 1e-9), (-1.0, 2.0**-25), (-0.5, 0.0), (1.0, -1e-9), (-0.0, 0.0)]
    for op, last in (("ceil", 1.0), ("trunc", 0.0)):
        hi, lo = _launch_pairs(lib, op, a)
        assert hi.tolist() == [0.0, 0.0, 0.0, last, 0.0] and lo.tolist() == [0.0] * 5, (
            op
        )
        assert np.signbit(hi).tolist() == [True, True, True, False, True], op
    hi, lo = _launch_pairs(lib, "floor", a)
    assert hi.tolist() == [-1.0, -1.0, -1.0, 0.0, 0.0]
    assert np.signbit(hi).tolist() == [True, True, True, False, True]


def test_ldexp_mul_pwr2_overflow_stays_canonical(core_helpers):
    a = [(_f32(0x7F61B1E6), _f32(0x7149F2CA)), (1.0, 2.0**-25), (-3e38, -1e30)]
    hi, lo, _ = _run_helper(core_helpers, 0, a, [1, 130, 2])
    assert hi.tolist() == [np.inf, np.inf, -np.inf] and lo.tolist() == [0.0] * 3
    hi, lo, _ = _run_helper(core_helpers, 1, a, [2, 2, 2])
    assert hi.tolist()[0::2] == [np.inf, -np.inf] and lo.tolist()[0::2] == [0.0, 0.0]
    assert hi[1] == 2.0 and lo[1] == 2.0**-24
    # eq(x, inf) / is_finite agree with the value once the pair is canonical
    hi, lo, _ = _run_helper(core_helpers, 5, [(np.inf, 0.0), (np.inf, 1.0), (1.0, 0.0)])
    assert hi.tolist() == [1.0, 1.0, 0.0] and lo.tolist() == [0.0, 0.0, 1.0]


def test_renorm_of_infinity_has_zero_lo(core_helpers):
    hi, lo, _ = _run_helper(
        core_helpers, 2, [(np.inf, 0.0), (-np.inf, 0.0), (1.0, 2.0**-24), (np.nan, 0.0)]
    )
    assert hi.tolist()[:3] == [np.inf, -np.inf, 1.0] and np.isnan(hi[3])
    assert lo.tolist() == [0.0, 0.0, 2.0**-24, 0.0]


def test_frexp_mantissa_in_half_one(core_helpers):
    a = [
        (1.0, -(2.0**-25)),
        (2.0, -(2.0**-24)),
        (-1.0, 2.0**-25),
        (1.0, 0.0),
        (0.5, 0.0),
        (0.75, 1e-9),
        (0.0, 0.0),
        (1024.0, 2.0**-20),
    ]
    hi, lo, e = _run_helper(core_helpers, 3, a)
    m = hi + lo
    for (h, low), mm, ee in zip(a, m, e, strict=True):
        x = float(np.float32(h)) + float(np.float32(low))
        if x == 0:
            assert mm == 0 and ee == 0
            continue
        assert 0.5 <= abs(mm) < 1.0, (h, low, mm, ee)
        assert mm * 2.0 ** int(ee) == x, (h, low, mm, ee)


def test_comparisons_are_representation_independent(lib):
    """Odd-hi tie form vs even-hi form of the same value compare equal."""
    x1 = (_f32(0x3F800001), _f32(0x33800000))  # 1+2^-23, +2^-24
    x2 = (_f32(0x3F800002), _f32(0xB3800000))  # 1+2^-22, -2^-24
    y = (8388609.0, 0.5)
    y2 = (8388610.0, -0.5)
    a, b = [x1, x2, y, x1, (1.0, 0.0)], [x2, x1, y2, (2.0, 0.0), (1.0, 0.0)]
    want = {
        "eq": [True, True, True, False, True],
        "ne": [False, False, False, True, False],
        "lt": [False, False, False, True, False],
        "le": [True, True, True, True, True],
        "gt": [False, False, False, False, False],
        "ge": [True, True, True, False, True],
    }
    for op, w in want.items():
        assert _launch_pairs(lib, op, a, b).tolist() == w, op
    # x == x + 0 through the kernels (add canonicalizes, eq must not care)
    hi, lo = _launch_pairs(lib, "add", [x1], [(0.0, 0.0)])
    assert (hi[0], lo[0]) == (x2[0], x2[1])
    assert _launch_pairs(lib, "eq", [x1], [(float(hi[0]), float(lo[0]))]).tolist() == [
        True
    ]
    # minimum/maximum pick either representation of equal values (same value)
    hi, lo = _launch_pairs(lib, "minimum", [x1], [x2])
    assert hi[0] + lo[0] == float.fromhex("0x1.000003p+0")


def test_add_sub_exact_cancellation_across_representations(lib):
    """Equal values in different representations must cancel to +0, not -2^-23."""
    a = [
        (1.0, 2.0**-24),
        (_f32(0x3F800001), _f32(0x33800000)),
        (_f32(0x3F7FFFFF), _f32(0x308AEFC0)),
        (-0.0, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
    ]
    b = [
        (-(1 + 2.0**-23), 2.0**-24),
        (_f32(0xBF800002), _f32(0x33800000)),
        (_f32(0xBF800000), _f32(0x337BA882)),
        (-0.0, 0.0),
        (-1.0, 0.0),
        (-0.0, 0.0),
    ]
    hi, lo = _launch_pairs(lib, "add", a, b)
    assert hi.tolist() == [0.0] * 6 and lo.tolist() == [0.0] * 6
    assert np.signbit(hi).tolist() == [False, False, False, True, False, False]
    hi, lo = _launch_pairs(lib, "sub", a, [(-x, -y) for x, y in b])
    assert hi.tolist() == [0.0] * 6 and lo.tolist() == [0.0] * 6
    # IEEE: (-0) - (+0) = -0, (+0) - (+0) = +0
    assert np.signbit(hi).tolist() == [False, False, False, True, False, False]
    # scalar variant (df::add(df64, float) path through the float2 scalar)
    out = lib.launch("add", _pairs(a[1]), scalar=float.fromhex("-0x1.000003p+0"))
    assert out[0].cpu().item() == 0.0 and out[1].cpu().item() == 0.0


def test_rint_tie_form_rounds_half_to_even(lib):
    a = [
        (8388609.0, 0.5),
        (8388610.0, -0.5),
        (8388611.0, 0.5),
        (2.5, 0.0),
        (-0.5, 0.0),
        (0.5, 1e-9),
        (3.0, -0.5),
        (2.0, 0.5),
    ]
    hi, lo = _launch_pairs(lib, "rint", a)
    want = [8388610.0, 8388610.0, 8388612.0, 2.0, -0.0, 1.0, 2.0, 2.0]
    assert (hi + lo).tolist() == want and lo.tolist() == [0.0] * 8
    assert np.signbit(hi[4])
    # through the host encoder (which now emits the even-hi form itself)
    y = 2.0**23 + 1.5 - 2.0**-29
    got = encode.from_mps_df64(*lib.launch("rint", gpu(np.array([y, -y]))))
    assert got.tolist() == [8388610.0, -8388610.0]


def _exact_fmod(a: float, b: float):
    from fractions import Fraction

    fa, fb = Fraction(a), Fraction(b)
    r = abs(fa) - (abs(fa) // abs(fb)) * abs(fb)
    return -r if a < 0 else r


def _exact_remainder(a: float, b: float):
    from fractions import Fraction

    r = _exact_fmod(a, b)
    if r != 0 and ((r < 0) != (b < 0)):
        r += Fraction(b)
    return r


def test_fmod_remainder_huge_quotients_are_exact_and_finite(lib):
    """fmod used to return -inf when a/b overflowed float32 and lost the sign /
    |r| < |b| invariants above |a/b| ~ 2^47; the staged reduction is exact."""
    from fractions import Fraction

    a = decoded(
        np.array([1e30, 3e38, 5e37, 1e6, 1e5, 4.999999999999999e37, -1e30, FLT_MAX])
    )
    b = decoded(np.array([1e-10, 1e-5, 3.0, 1e-9, 1e-10, 3.0, 7.0, -1e-30]))
    for op, exact in (("fmod", _exact_fmod), ("remainder_py", _exact_remainder)):
        r = encode.from_mps_df64(*lib.launch(op, gpu(a), gpu(b)))
        assert np.all(np.isfinite(r)), op
        for x, y, got in zip(a, b, r, strict=True):
            ref = exact(float(x), float(y))
            assert (
                abs(Fraction(float(got)) - ref)
                <= Fraction(abs(float(y))) * Fraction(2) ** -49
            ), (op, x, y, got, float(ref))
    # sweep over |a/b| in [2^10, 2^70]: invariants and df64-rounded exactness
    rng = np.random.default_rng(11)
    n = 6000
    b = decoded(
        np.exp(rng.uniform(np.log(1e-12), np.log(1e6), n)) * rng.choice([-1.0, 1.0], n)
    )
    a = decoded(b * 2.0 ** rng.uniform(10, 70, n) * rng.choice([-1.0, 1.0], n))
    ok = np.isfinite(a) & (np.abs(a) < 3e38) & (np.abs(a) > 1e-30) & (np.abs(b) > 1e-30)
    a, b = a[ok], b[ok]
    f = encode.from_mps_df64(*lib.launch("fmod", gpu(a), gpu(b)))
    r = encode.from_mps_df64(*lib.launch("remainder_py", gpu(a), gpu(b)))
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(r))
    assert np.all((np.signbit(f) == np.signbit(a)) | (f == 0)) and np.all(
        np.abs(f) < np.abs(b)
    )
    assert np.all((np.signbit(r) == np.signbit(b)) | (r == 0)) and np.all(
        np.abs(r) < np.abs(b)
    )
    ef = np.array(
        [float(_exact_fmod(float(x), float(y))) for x, y in zip(a, b, strict=True)]
    )
    er = np.array(
        [float(_exact_remainder(float(x), float(y))) for x, y in zip(a, b, strict=True)]
    )
    assert (
        np.max(np.abs(f - ef) / np.abs(b)) <= 0.5 * U2 * 1.001
    )  # df64 rounding of the exact remainder
    assert np.max(np.abs(r - er) / np.abs(b)) <= 0.5 * U2 * 1.001


def test_denormal_inputs_are_flushed_consistently(lib):
    """Host encoder flushes float32-denormal words, so the kernels see +-0."""
    x = np.array([1e-40, -1e-40, 1e-38, 0.0, 1.0])
    hi, lo = encode.encode_df64(x)
    assert hi.tolist()[:3] == [0.0, -0.0, 0.0] and np.signbit(hi[1])
    g = gpu(x)
    # inf * (flushed denormal) is inf * 0 = NaN, like inf * 0.0 (IEEE)
    got = encode.from_mps_df64(*lib.launch("mul", gpu(np.full(5, np.inf)), g))
    assert (
        np.isnan(got).tolist() == [True, True, True, True, False] and got[4] == np.inf
    )
    got = encode.from_mps_df64(*lib.launch("sqrt", g))
    assert got.tolist()[:4] == [0.0, 0.0, 0.0, 0.0]
    assert np.signbit(got).tolist()[:4] == [False, True, False, False]  # sqrt(-0) = -0
    assert lib.launch("eq", g, gpu(np.zeros(5))).cpu().tolist() == [
        True,
        True,
        True,
        True,
        False,
    ]
    got = encode.from_mps_df64(*lib.launch("neg", g))
    assert np.signbit(got).tolist() == [True, False, True, True, True]
    assert encode.from_mps_df64(
        *lib.launch("add", gpu(np.array([2.0**-126])), gpu(np.array([1e-38])))
    ).tolist() == [2.0**-126]


@pytest.mark.parametrize(
    "op",
    [
        "sin",
        "tan",
        "atan",
        "sinh",
        "tanh",
        "asinh",
        "cbrt",
        "erf",
        "expm1",
        "log1p",
        "erfinv",
        "asin",
        "atanh",
    ],
)
def test_sf64_bridge_keeps_negative_zero(op):
    """sf::from_df64(-0, +0) returned +0, so every odd bridged function lost -0."""
    sf = library.get_library("sf64")
    if not sf.has(op):
        pytest.skip(f"{op} not in the sf64 library")
    got = encode.from_mps_sf64(sf.launch(op, encode.to_mps_sf64(np.array([-0.0, 0.0]))))
    assert got.tolist() == [0.0, 0.0] and np.signbit(got).tolist() == [True, False], op


# ---------------------------------------------------------------------------
# Fix round 2 regressions (adversarial verification, NOTES/04 section 9).
# ---------------------------------------------------------------------------
TWO_P103 = 2.0**103  # ulp(FLT_MAX) / 2: the df64 overflow tie
FLT_MIN32 = 2.0**-126


def _exact(h, low):
    return Fraction(float(np.float32(h))) + Fraction(float(np.float32(low)))


def test_top_half_ulp_window_stays_finite(lib, core_helpers):
    """mul/sqr/div/add returned +-inf when the hi-word product / quotient / sum
    alone overflowed although the exact result is below the overflow tie
    FLT_MAX + 2^103 (RN32 overflows only from the tie on)."""
    p64 = 2.0**64
    # mul: exact products FLT_MAX and FLT_MAX - 2^104 + 2^80 (hi products = 2^128)
    a = [(p64, -(2.0**40)), (p64, -(2.0**40)), (FLT_MAX, 0.0), (-p64, 2.0**40)]
    b = [(p64, 0.0), (p64, -(2.0**40)), (1.0, 2.0**-25), (p64, 0.0)]
    hi, lo = _launch_pairs(lib, "mul", a, b)
    for (x, y), gh, gl in zip(zip(a, b, strict=True), hi, lo, strict=True):
        ref = _exact(*x) * _exact(*y)
        assert np.isfinite(gh), (x, y, gh)
        assert abs(_exact(gh, gl) - ref) <= 5 * U2 * abs(ref), (x, y, gh, gl)
    assert (hi[0], lo[0]) == (FLT_MAX, 0.0) and (hi[3], lo[3]) == (-FLT_MAX, 0.0)
    assert hi[2] == FLT_MAX and lo[2] == TWO_P103 - 2.0**79
    # sqr through the dedicated path
    sh, sl, _ = _run_helper(core_helpers, 6, [(p64, -(2.0**40)), (-p64, 2.0**40)])
    ref = _exact(p64, -(2.0**40)) ** 2
    for gh, gl in zip(sh, sl, strict=True):
        assert np.isfinite(gh) and abs(_exact(gh, gl) - ref) <= 5 * U2 * ref
    # div: exact quotient FLT_MAX (a.hi / b.hi = 2^128) and FLT_MAX + 2^80
    a = [(2.0**100, -(2.0**76)), (2.0**100, 0.0), (-(2.0**100), 2.0**76)]
    b = [(2.0**-28, 0.0), (2.0**-28, 2.0**-52), (2.0**-28, 0.0)]
    hi, lo = _launch_pairs(lib, "div", a, b)
    assert (hi[0], lo[0]) == (FLT_MAX, 0.0) and (hi[2], lo[2]) == (-FLT_MAX, 0.0)
    ref = _exact(*a[1]) / _exact(*b[1])
    assert np.isfinite(hi[1]) and abs(_exact(hi[1], lo[1]) - ref) <= 10 * U2 * ref
    # add: FLT_MAX + (2^103 - 2^79) is below the tie and representable exactly;
    # FLT_MAX + 2^103 (the tie) and anything above overflow.
    a = [(FLT_MAX, 0.0), (FLT_MAX, 0.0), (FLT_MAX, 2.0**102), (-FLT_MAX, 0.0)]
    b = [
        (TWO_P103, -(2.0**79)),
        (TWO_P103, 0.0),
        (2.0**102, 2.0**79),
        (-TWO_P103, 2.0**79),
    ]
    hi, lo = _launch_pairs(lib, "add", a, b)
    assert (hi[0], lo[0]) == (FLT_MAX, TWO_P103 - 2.0**79)
    assert (hi[1], lo[1]) == (np.inf, 0.0) and (hi[2], lo[2]) == (np.inf, 0.0)
    assert (hi[3], lo[3]) == (-FLT_MAX, -(TWO_P103 - 2.0**79))
    # random products / sums with exact value in (FLT_MAX (1 - 2^-20), FLT_MAX + 2^103)
    rng = np.random.default_rng(23)
    n = 20000
    target = FLT_MAX * (1.0 + rng.uniform(-(2.0**-20), 2.0**-25, n))
    ah = np.ldexp(rng.uniform(1.0, 2.0, n), rng.integers(1, 60, n)).astype(np.float32)
    bh = (target / ah.astype(np.float64)).astype(np.float32)
    bl = (target / ah.astype(np.float64) - bh.astype(np.float64)).astype(np.float32)
    hi, lo = _launch_pairs(
        lib,
        "mul",
        list(zip(ah.tolist(), [0.0] * n, strict=True)),
        list(zip(bh.tolist(), bl.tolist(), strict=True)),
    )
    exact = [_exact(x, 0.0) * _exact(y, z) for x, y, z in zip(ah, bh, bl, strict=True)]
    tie = Fraction(FLT_MAX) + Fraction(TWO_P103)
    spurious = [
        i
        for i, e in enumerate(exact)
        if e < tie * (1 - 6 * U2) and not np.isfinite(hi[i])
    ]
    assert not spurious, spurious[:5]
    finite = np.isfinite(hi)
    err = [
        abs(_exact(hi[i], lo[i]) - exact[i]) / exact[i] / U2
        for i in np.flatnonzero(finite)
    ]
    assert max(err) <= 5.0
    ah = np.full(n, FLT_MAX, dtype=np.float32)
    bh = np.ldexp(rng.uniform(0.0, 1.0, n), 103).astype(np.float32)
    bl = -np.ldexp(rng.uniform(0.0, 1.0, n), 79).astype(np.float32)
    hi, lo = _launch_pairs(
        lib,
        "add",
        list(zip(ah.tolist(), [0.0] * n, strict=True)),
        list(zip(bh.tolist(), bl.tolist(), strict=True)),
    )
    exact = [Fraction(FLT_MAX) + _exact(y, z) for y, z in zip(bh, bl, strict=True)]
    # the df64 rounding of the sum decides: values whose nearest 48-bit
    # neighbour is the tie may overflow, everything further below stays finite
    spurious = [
        i
        for i, e in enumerate(exact)
        if e < tie - Fraction(2**79) and not np.isfinite(hi[i])
    ]
    assert not spurious, spurious[:5]
    for i in np.flatnonzero(np.isfinite(hi)):
        assert abs(_exact(hi[i], lo[i]) - exact[i]) <= 3 * U2 * exact[i]


def test_overflow_tie_is_a_clean_infinity(core_helpers):
    """The encoder and sf::to_float2 turned finite float64 values within 2^-50 of
    the float32 overflow tie into (inf, -inf) (NaN on decode); df::canon of the
    hand-built tie pair (FLT_MAX, 2^103) is +inf by the overflow rule."""
    bits = np.array(
        [
            0x47EFFFFFEFFFFFF8,
            0x47EFFFFFEFFFFFFF,
            0xC7EFFFFFEFFFFFF8,
            0x47EFFFFFEFFFFFF0,
        ],
        dtype=np.uint64,
    )
    x = bits.view(np.float64)
    hi, lo = encode.encode_df64(x)
    assert hi.tolist() == [np.inf, np.inf, -np.inf, FLT_MAX]
    assert lo.tolist() == [0.0, 0.0, 0.0, TWO_P103 - 2.0**79]
    assert encode.decode_df64(hi, lo).tolist() == [np.inf, np.inf, -np.inf, x[3]]
    b = torch.from_numpy(bits.view(np.int64)).to("mps")
    oh = torch.empty(4, dtype=torch.float32, device="mps")
    ol = torch.empty(4, dtype=torch.float32, device="mps")
    core_helpers.t_to_float2(b, oh, ol, threads=[4, 1, 1])
    assert oh.cpu().tolist() == hi.tolist() and ol.cpu().tolist() == lo.tolist()
    ch, cl, _ = _run_helper(
        core_helpers, 8, [(FLT_MAX, TWO_P103), (-FLT_MAX, -TWO_P103), (FLT_MAX, lo[3])]
    )
    assert ch.tolist() == [np.inf, -np.inf, FLT_MAX] and cl.tolist() == [
        0.0,
        0.0,
        lo[3],
    ]
    # the tie pair compares equal to inf; the largest finite value does not
    fh, fl, _ = _run_helper(core_helpers, 5, [(FLT_MAX, TWO_P103), (FLT_MAX, lo[3])])
    assert fh.tolist() == [1.0, 0.0] and fl[1] == 1.0


def test_div_of_flt_min_refines_correctly(lib, core_helpers):
    """div(+-FLT_MIN, b) returned 3x (df64/df64) resp. 2x (df64/float) the
    quotient for half of the divisors in (0.5, 1): the residual product b * q1
    flushed to zero, so every refinement added q1 again."""
    rng = np.random.default_rng(31)
    n = 20000
    bh = rng.uniform(0.5, 1.0, n).astype(np.float32)
    bl = (bh.astype(np.float64) * rng.uniform(-1, 1, n) * 2.0**-25).astype(np.float32)
    sign = rng.choice([-1.0, 1.0], n)
    k = rng.integers(0, 4, n)
    ah = (sign * FLT_MIN32 * (1 + k * 2.0**-23)).astype(np.float32)
    hi, lo = _launch_pairs(
        lib,
        "div",
        list(zip(ah.tolist(), [0.0] * n, strict=True)),
        list(zip(bh.tolist(), bl.tolist(), strict=True)),
    )
    ref = ah.astype(np.float64) / (bh.astype(np.float64) + bl.astype(np.float64))
    got = hi.astype(np.float64) + lo.astype(np.float64)
    rel = np.abs(got - ref) / np.abs(ref)
    assert rel.max() <= 2.0**-23, rel.max()  # float32 accuracy (lo word is denormal)
    # the verifier's exact case, both overloads and float / df64
    b = (_f32(0x3F00B6F1), _f32(0x31EBBED8))
    hi, lo = _launch_pairs(lib, "div", [(FLT_MIN32, 0.0), (-FLT_MIN32, 0.0)], [b, b])
    want = FLT_MIN32 / (float(np.float32(b[0])) + float(np.float32(b[1])))
    assert abs(hi[0] + lo[0] - want) <= 2.0**-23 * want
    assert abs(hi[1] + lo[1] + want) <= 2.0**-23 * want
    dh, dl, _ = _run_helper(
        core_helpers, 7, [(FLT_MIN32, 0.0)], [int(np.float32(1.0).view(np.int32))]
    )
    assert dh[0] == FLT_MIN32
    fh, fl = _launch_pairs(lib, "div", [(2.0**-100, 2.0**-125)], [(3.0, 0.0)])
    ref = (2.0**-100 + 2.0**-125) / 3.0
    assert abs(fh[0] + fl[0] - ref) <= 10 * U2 * ref


def test_products_rounding_up_to_flt_min_are_not_flushed(lib, core_helpers):
    """mul returned +0 when a.hi * b.hi was flushed before rounding (IEEE RN would
    give FLT_MIN) although the exact df64 product is >= FLT_MIN."""
    a = [(_f32(0x1FCA59B8), _f32(0x12943890)), (_f32(0x1F5DE2DF), _f32(0x12BBC5D5))]
    b = [(_f32(0x2021EFD6), _f32(0x13B355F7)), (_f32(0x2093ADE6), _f32(0x13850ECB))]
    hi, lo = _launch_pairs(lib, "mul", a, b)
    for (x, y), gh in zip(zip(a, b, strict=True), hi, strict=True):
        ref = _exact(*x) * _exact(*y)
        assert ref >= Fraction(FLT_MIN32)
        assert gh == FLT_MIN32, (x, y, gh)
    hi, lo = _launch_pairs(lib, "mul", [(-a[0][0], -a[0][1])], [b[0]])
    assert hi[0] == -FLT_MIN32
    # random products with exact value in [FLT_MIN, FLT_MIN (1 + 2^-20)]
    rng = np.random.default_rng(47)
    n = 20000
    target = FLT_MIN32 * (1.0 + rng.uniform(0.0, 2.0**-20, n))
    ah = np.ldexp(rng.uniform(1.0, 2.0, n), rng.integers(-70, -50, n)).astype(
        np.float32
    )
    bh = (target / ah.astype(np.float64)).astype(np.float32)
    bl = (target / ah.astype(np.float64) - bh.astype(np.float64)).astype(np.float32)
    bl[np.abs(bl) < FLT_MIN32] = 0.0
    hi, lo = _launch_pairs(
        lib,
        "mul",
        list(zip(ah.tolist(), [0.0] * n, strict=True)),
        list(zip(bh.tolist(), bl.tolist(), strict=True)),
    )
    exact = np.array(
        [
            float(_exact(x, 0.0) * _exact(y, z))
            for x, y, z in zip(ah, bh, bl, strict=True)
        ]
    )
    above = exact >= FLT_MIN32
    assert np.all(hi[above] >= FLT_MIN32), np.flatnonzero(hi[above] < FLT_MIN32)[:5]
    assert np.all(np.abs(hi[above] - exact[above]) <= 2.0**-23 * exact[above])
    # mul(df64, float) shares the guard: a.hi * b = FLT_MIN (1 - 2^-46) is
    # flushed, the df64 product FLT_MIN (1 + 2^-24 - 2^-46) rounds to FLT_MIN.
    # Exact results below FLT_MIN stay +-0 (FTZ semantics, not IEEE RN).
    b = int(np.float32(2.0**-63 * (1 - 2.0**-23)).view(np.int32))
    mh, ml, _ = _run_helper(
        core_helpers,
        7,
        [
            (2.0**-63 * (1 + 2.0**-23), 2.0**-87),
            (2.0**-63, 2.0**-87),
            (-(2.0**-63), 0.0),
        ],
        [b, b, b],
    )
    assert mh.tolist() == [FLT_MIN32, 0.0, -0.0] and ml.tolist() == [0.0, 0.0, 0.0]
    assert np.signbit(mh).tolist() == [False, False, True]
    # sqr: a.hi^2 flushed and a^2 < FLT_MIN (a.lo cannot lift it above) -> 0;
    # a.hi^2 >= FLT_MIN is not flushed (float32 accuracy: the cross term 2 hi lo
    # is a flushed denormal there)
    sh, sl, _ = _run_helper(
        core_helpers, 6, [(2.0**-63 * (1 - 2.0**-24), 2.0**-88), (2.0**-63, 2.0**-87)]
    )
    assert sh[0] == 0.0 and sh[1] in (FLT_MIN32, FLT_MIN32 * (1 + 2.0**-23))
    assert sl.tolist() == [0.0, 0.0]


def test_round_away_large_integers_are_fixed_points(lib):
    """round_away((2^48, 16777215)) returned x + 1: add(|a|, 0.5) rounded a 24-bit
    lo word plus 1/2 in float32."""
    pairs = [
        (2.0**48, 16777215.0),
        (2.0**48, -8388609.0),
        (2.0**48, 8388609.0),
        (-(2.0**48), -8388609.0),
        (2.0**49, 8388609.0),
        (_f32(0x57A2807F), _f32(0x4B774877)),
        (_f32(0xD9D3C0DB), _f32(0x4B73310B)),
        (2.0**24, 1.0),
        (2.0**24, 0.5),
        (2.0**24, -0.5),
        (2.0**25, -1.5),
        (-(2.0**25), 1.5),
        (2.0**23, 0.25),
        (2.5, 0.0),
        (2.5, -1e-9),
        (-2.5, 1e-9),
        (-2.5, 0.0),
        (0.5, 0.0),
        (-0.5, 0.0),
        (0.5, -1e-9),
        (-0.49999997, 0.0),
    ]
    hi, lo = _launch_pairs(lib, "round_away", pairs)
    for (h, l_), gh, gl in zip(pairs, hi, lo, strict=True):
        x = _exact(h, l_)
        want = math.floor(abs(x) + Fraction(1, 2)) * (1 if x > 0 else -1)
        assert _exact(gh, gl) == want, (h, l_, gh, gl, want)
        assert abs(gl) <= abs(np.spacing(np.float32(gh))) / 2 or gh == 0
    assert np.signbit(hi[-1]) and hi[-1] == 0.0
    rng = np.random.default_rng(11)
    n = 50000
    hi_in = np.ldexp(1.0 + rng.integers(0, 2**23, n) / 2**23, rng.integers(48, 53, n))
    lo_in = rng.integers(-(2**24), 2**24, n).astype(np.float32)
    pairs = list(zip(hi_in.astype(np.float32).tolist(), lo_in.tolist(), strict=True))
    hi, lo = _launch_pairs(lib, "round_away", pairs)
    assert all(
        _exact(gh, gl) == _exact(h, l_)
        for (h, l_), gh, gl in zip(pairs, hi, lo, strict=True)
    )


def test_to_int_saturates_beyond_int32(core_helpers):
    """to_int wrapped modulo 2^32 (2^31 -> -2^31, 1e10 -> 1410065408, +inf -> -1)."""
    pairs = [
        (2.0**31, 0.0),
        (2.0**31, 128.0),
        (2.0**31, -1.0),
        (-(2.0**31), -1.0),
        (-(2.0**31), 0.0),
        (1e10, 0.0),
        (-1e10, 0.0),
        (3e9, 0.0),
        (2.0**40, 0.0),
        (FLT_MAX, 0.0),
        (-FLT_MAX, 0.0),
        (np.inf, 0.0),
        (-np.inf, 0.0),
        (np.nan, 0.0),
        (2147483392.0, 127.0),
    ]
    hi, lo = _pairs(*pairs)
    o = torch.empty(len(pairs), dtype=torch.int32, device="mps")
    core_helpers.t_to_int(hi, lo, o, threads=[len(pairs), 1, 1])
    imax, imin = 2**31 - 1, -(2**31)
    want = [
        imax,
        imax,
        imax,
        imin,
        imin,
        imax,
        imin,
        imax,
        imax,
        imax,
        imin,
        imax,
        imin,
        0,
        2147483519,
    ]
    assert o.cpu().numpy().tolist() == want


def test_frexp_keeps_the_sign_of_zero(core_helpers):
    hi, lo, e = _run_helper(core_helpers, 3, [(-0.0, 0.0), (0.0, 0.0)])
    assert hi.tolist() == [0.0, 0.0] and np.signbit(hi).tolist() == [True, False]
    assert lo.tolist() == [0.0, 0.0] and e.tolist() == [0, 0]


def test_fmod_remainder_tiny_divisors_stay_correctly_rounded(lib):
    """Divisors below ~2^-77 lost accuracy (float32 level at 2^-104, 0 or O(|b|)
    below 2^-122) because the exact residual words fell into the flushed range;
    mod_reduce now rescales its state. Bound: 0.5 u^2 of |b| or the FLT_MIN
    storage floor of the result, whichever is larger."""
    cases = [
        ((1.0, 0.0), (_f32(0x01AAC97C), 0.0)),
        ((_f32(0x7F3C14D5), _f32(0x7224B036)), (_f32(0x01AAC97C), 0.0)),
        ((1.0, 0.0), (_f32(0x0DA66666), _f32(0x014CCCCD))),
        ((7.0, 0.0), (2.0**-100, 2.0**-125)),
        ((1e30, 1e13), (1e-30, 1e-47)),
    ]
    rng = np.random.default_rng(5)
    n = 4000
    ah = np.ldexp(rng.uniform(1, 2, n), rng.integers(-100, 128, n)).astype(np.float32)
    al = (ah.astype(np.float64) * rng.uniform(-1, 1, n) * 2.0**-25).astype(np.float32)
    bh = np.ldexp(rng.uniform(1, 2, n), rng.integers(-126, -60, n)).astype(np.float32)
    bl = (bh.astype(np.float64) * rng.uniform(-1, 1, n) * 2.0**-25).astype(np.float32)
    al[np.abs(al) < FLT_MIN32] = 0.0
    bl[np.abs(bl) < FLT_MIN32] = 0.0
    a = [c[0] for c in cases] + list(zip(ah.tolist(), al.tolist(), strict=True))
    b = [c[1] for c in cases] + list(zip(bh.tolist(), bl.tolist(), strict=True))
    for op in ("fmod", "remainder_py"):
        hi, lo = _launch_pairs(lib, op, a, b)
        worst = 0.0
        for (x, y), gh, gl in zip(zip(a, b, strict=True), hi, lo, strict=True):
            A, B = _exact(*x), _exact(*y)
            r = (
                A - B * math.floor(A / B)
                if op == "remainder_py"
                else A - B * int(A / B)
            )
            got = _exact(gh, gl)
            bound = max(Fraction(1, 2) * Fraction(U2) * abs(B), Fraction(FLT_MIN32))
            assert abs(got - r) <= bound, (op, x, y, float(r), float(got))
            if abs(r) >= Fraction(2) ** -78:
                worst = max(worst, float(abs(got - r) / abs(B) / U2))
        assert worst <= 0.5, (op, worst)


# ---------------------------------------------------------------------------
# Fix round 3 (NOTES/04 section 10): core sign / range edges and launcher checks
# ---------------------------------------------------------------------------
def test_mul_underflow_keeps_sign_of_zero(lib, core_helpers):
    """A negative product that still flushed after the 2^48-scaled retry came
    back as +0 (quick_two_sum(-0, +0) is +0); the hardware product of the hi
    words carries the sign of the exact result and is returned instead."""
    a = [(-1e-30, 0.0), (-FLT_MIN32, 0.0), (-1e-27, 0.0), (1e-30, 0.0), (-1e-30, 0.0)]
    b = [(1e-30, 0.0), (FLT_MIN32, 0.0), (1e-27, 0.0), (1e-30, 0.0), (-1e-30, 0.0)]
    hi, lo = _launch_pairs(lib, "mul", a, b)
    assert hi.tolist() == [0.0] * 5 and lo.tolist() == [0.0] * 5
    assert np.signbit(hi).tolist() == [True, True, True, False, False]
    # the df64 * float overload with the negative operand on the df64 side
    bits = [int(np.float32(y[0]).view(np.int32)) for y in b]
    mh, ml, _ = _run_helper(core_helpers, 7, a, bits)
    assert mh.tolist() == [0.0] * 5 and ml.tolist() == [0.0] * 5
    assert np.signbit(mh).tolist() == [True, True, True, False, False]
    # random sweep over products with exact magnitude 2^-252 .. 2^-126
    rng = np.random.default_rng(11)
    n = 100000
    x = (2.0 ** rng.uniform(-126, 0, n)).astype(np.float32) * rng.choice([-1, 1], n)
    y = (2.0 ** rng.uniform(-126, 0, n)).astype(np.float32) * rng.choice([-1, 1], n)
    keep = np.abs(x) >= FLT_MIN32
    x, y = x[keep], y[keep]
    hi, _ = _launch_pairs(
        lib,
        "mul",
        list(zip(x.tolist(), [0.0] * x.size, strict=True)),
        list(zip(y.tolist(), [0.0] * y.size, strict=True)),
    )
    zero = hi == 0.0
    assert zero.sum() > 1000, "sweep must reach the flushed range"
    want = np.signbit(x[zero]) != np.signbit(y[zero])
    assert np.array_equal(np.signbit(hi[zero]), want)


def test_div_small_dividends_keep_full_accuracy(lib, core_helpers):
    """Dividends in [2^-100, 2^-78) formed their residual at their own scale, where
    the correction terms flushed below FLT_MIN: a quotient of ordinary size
    (3.27 from 1.7e-30 / 5.2e-31) inherited FLT_MIN / |a| relative error (2.15e6
    u^2 for 23% of the dividends near 2^-99). The scaled residual now applies
    below 2^-78."""
    a = [(_f32(0x0E09A549), _f32(0x01A00204))]
    b = [(_f32(0x0D286DC0), _f32(0x00D9DD47))]
    hi, lo = _launch_pairs(lib, "div", a, b)
    ref = _exact(*a[0]) / _exact(*b[0])
    assert abs(_exact(hi[0], lo[0]) - ref) <= 10 * U2 * abs(ref)
    a2 = [(_f32(0x0E2F3147), _f32(0x81F0991E))]
    bf = _f32(0xBA1047E2)
    dh, dl, _ = _run_helper(core_helpers, 9, a2, [int(np.float32(bf).view(np.int32))])
    ref = _exact(*a2[0]) / Fraction(float(bf))
    assert abs(_exact(dh[0], dl[0]) - ref) <= 10 * U2 * abs(ref)
    rng = np.random.default_rng(23)
    n = 6000
    for ea in (-99, -95, -90, -85, -79):
        m = rng.integers(0, 2**48, n)
        av = np.ldexp(1.0 + m / 2.0**48, ea)
        bv = 2.0 ** rng.uniform(-125, -100, n) * rng.choice([-1.0, 1.0], n)
        ah = av.astype(np.float32)
        al = (av - ah.astype(np.float64)).astype(np.float32)
        bh = bv.astype(np.float32)
        bl = (bv - bh.astype(np.float64)).astype(np.float32)
        al[np.abs(al) < FLT_MIN32] = 0.0
        bl[np.abs(bl) < FLT_MIN32] = 0.0
        pa = list(zip(ah.tolist(), al.tolist(), strict=True))
        pb = list(zip(bh.tolist(), bl.tolist(), strict=True))
        hi, lo = _launch_pairs(lib, "div", pa, pb)
        ref = (ah.astype(np.float64) + al) / (bh.astype(np.float64) + bl)
        rel = np.abs(hi + lo - ref) / np.abs(ref) / U2
        assert rel.max() <= 10.0, (ea, rel.max())
        sh, sl, _ = _run_helper(
            core_helpers, 9, pa, [int(v) for v in bh.view(np.int32)]
        )
        ref = (ah.astype(np.float64) + al) / bh.astype(np.float64)
        assert (np.abs(sh + sl - ref) / np.abs(ref) / U2).max() <= 10.0, ea


def test_abs_copysign_handle_nan_sign_bit(lib):
    """abs returned -NaN unchanged and copysign flipped instead of set the sign
    bit of a NaN with its sign bit set (numpy/torch fabs clear it)."""
    neg_nan = _f32(0xFFC00000)
    pos_nan = _f32(0x7FC00000)
    hi, _ = _launch_pairs(lib, "abs", [(neg_nan, 0.0), (pos_nan, 0.0), (-0.0, 0.0)])
    assert np.isnan(hi[:2]).all() and not np.signbit(hi).any()
    a = [(neg_nan, 0.0)] * 4 + [(pos_nan, 0.0)] * 2
    b = [(1.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (-0.0, 0.0), (-1.0, 0.0), (1.0, 0.0)]
    hi, _ = _launch_pairs(lib, "copysign", a, b)
    assert np.isnan(hi).all()
    assert np.signbit(hi).tolist() == [False, True, False, True, True, False]


def test_ldexp_mul_pwr2_tie_form_input_scales_to_inf(core_helpers):
    """(FLT_MAX/2 with odd hi, 2^102) is a valid tie-form pair of 2^127 - 2^102;
    scaled by 2 it landed exactly on (FLT_MAX, 2^103), the overflow tie the
    header excludes from the contract, instead of +inf like the even-hi form
    (2^127, -2^102) of the same value."""
    half_max = _f32(0x7EFFFFFF)  # 0x1.fffffep+126
    pairs = [
        (half_max, 2.0**102),
        (-half_max, -(2.0**102)),
        (2.0**127, -(2.0**102)),
        (FLT_MAX, TWO_P103 - 2.0**79),
    ]
    hi, lo, _ = _run_helper(core_helpers, 0, pairs, [1, 1, 1, 0])
    assert hi.tolist() == [np.inf, -np.inf, np.inf, FLT_MAX]
    assert lo.tolist() == [0.0, 0.0, 0.0, TWO_P103 - 2.0**79]
    hi, lo, _ = _run_helper(core_helpers, 1, pairs, [2, 2, 2, 1])
    assert hi.tolist() == [np.inf, -np.inf, np.inf, FLT_MAX]
    assert lo.tolist() == [0.0, 0.0, 0.0, TWO_P103 - 2.0**79]


def test_canon_keeps_the_canonical_negative_zero(core_helpers):
    """canon((-0, +0)) returned (+0, +0): the sign of a zero pair is now kept
    (identity on canonical pairs, including df64 -0)."""
    pairs = [(-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (0.0, 0.0), (1.0, 2.0**-24)]
    for which in (8, 2):  # canon, renorm
        hi, lo, _ = _run_helper(core_helpers, which, pairs)
        assert hi.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
        assert np.signbit(hi).tolist() == [True, False, True, False, False]
        assert lo.tolist() == [0.0, 0.0, 0.0, 0.0, 2.0**-24]
        assert not np.signbit(lo).any()


def test_chunk_size_must_be_a_positive_integer():
    for bad in (0.9, 1.5, True, "8", 2.0):
        with pytest.raises(TypeError):
            library.MetalLibrary("df64", ops=["neg"], chunk_size=bad, selftest=False)
    for bad in (0, -1, library.DEFAULT_CHUNK + 1):
        with pytest.raises(ValueError):
            library.MetalLibrary("df64", ops=["neg"], chunk_size=bad, selftest=False)
    small = library.MetalLibrary(
        "df64", ops=["neg"], chunk_size=np.int64(1000), selftest=False
    )
    assert small.chunk_size == 1000 and type(small.chunk_size) is int
    x = decoded(rand(2500, 17))
    assert np.array_equal(encode.from_mps_df64(*small.launch("neg", gpu(x))), -x)


def test_encoders_reject_non_real_inputs():
    """None became NaN, strings/bytes were parsed, complex lost its imaginary
    part, an int64 sf64 component fed back to the encoder was re-encoded by
    value, and float64 components were silently rounded to float32 on decode."""
    for bad in (None, "1.5", b"1.5", 1 + 2j, np.array([1.5, None], dtype=object)):
        with pytest.raises(TypeError):
            encode.encode_df64(bad)
        with pytest.raises(TypeError):
            encode.encode_sf64(bad)
        with pytest.raises(TypeError):
            encode.to_mps_df64(bad)
        with pytest.raises(TypeError):
            encode.to_mps_sf64(bad)
    for bad in (None, "1.5", b"2", 1j, True, [1.0]):
        with pytest.raises(TypeError):
            encode.df64_scalar(bad)
        with pytest.raises(TypeError):
            encode.sf64_scalar(bad)
    with pytest.raises(TypeError):
        encode.encode_df64(np.array([1 + 2j, 3 - 4j]))
    with pytest.raises(TypeError):
        encode.to_mps_df64(torch.tensor([1 + 2j], dtype=torch.complex128))
    # integer input is exact for df64 (no component is integer typed) ...
    hi, lo = encode.encode_df64([1, 2, 3])
    assert encode.decode_df64(hi, lo).tolist() == [1.0, 2.0, 3.0]
    # ... but ambiguous for sf64, whose components are int64 bit patterns
    with pytest.raises(TypeError, match="int64"):
        encode.encode_sf64(np.array([4607182418800017408], dtype=np.int64))
    t = encode.to_mps_sf64(np.array([1.5]))
    with pytest.raises(TypeError, match="int64"):
        encode.to_mps_sf64(t)
    with pytest.raises(TypeError):
        encode.decode_sf64(np.array([1.5]))
    assert encode.from_mps_sf64(t).tolist() == [1.5]
    # decode_df64: float32 components of equal shape only
    with pytest.raises(TypeError):
        encode.decode_df64(np.array([1.0 + 2.0**-40]), np.array([0.0]))
    with pytest.raises(TypeError):
        encode.from_mps_df64(
            torch.tensor([1.0 + 2.0**-40], dtype=torch.float64), torch.zeros(1)
        )
    with pytest.raises(TypeError):
        encode.from_mps_df64(
            torch.tensor([4607182418800017408], device="mps"),
            torch.zeros(1, device="mps"),
        )
    with pytest.raises(ValueError, match="shape"):
        encode.decode_df64(np.ones((5, 1), np.float32), np.zeros(7, np.float32))
    assert encode.decode_df64(np.float32(1.5), np.float32(2.0**-30)) == 1.5 + 2.0**-30


def test_launch_rejects_neg_views_overlap_shape_and_bad_scalars(lib):
    x = torch.tensor([1.0, 2.0, 3.0], device="mps")
    z = torch.zeros(3, device="mps")
    nv = x._neg_view()
    assert nv.is_neg() and nv.is_contiguous()
    with pytest.raises(ValueError, match="negated"):
        lib.launch("neg", (nv, z))
    got = encode.from_mps_df64(*lib.launch("neg", (nv.resolve_neg(), z)))
    assert got.tolist() == [1.0, 2.0, 3.0]
    # out buffers aliasing each other or partially overlapping an input
    o = torch.empty(6, device="mps")
    a6 = torch.arange(6.0, device="mps")
    z6 = torch.zeros(6, device="mps")
    with pytest.raises(ValueError, match="overlap"):
        lib.launch("neg", (a6, z6), out=(o, o))
    buf = torch.arange(10.0, device="mps")
    zb = torch.zeros(10, device="mps")
    with pytest.raises(ValueError, match="overlap"):
        lib.launch("neg", (buf[:8], zb[:8]), out=(buf[2:], zb[2:]))
    assert buf.cpu().tolist() == list(range(10))  # nothing was written
    # exact in-place aliasing is allowed
    res = lib.launch("neg", (buf, zb), out=(buf, zb))
    assert res[0] is buf and buf.cpu().tolist() == [-float(i) for i in range(10)]
    ob = torch.empty(10, dtype=torch.bool, device="mps")
    assert lib.launch("isnan", (buf, zb), out=ob) is ob
    # same numel, different shape
    a = torch.arange(6.0, device="mps").reshape(2, 3)
    b = torch.arange(6.0, device="mps").reshape(3, 2)
    za = torch.zeros(2, 3, device="mps")
    zb2 = torch.zeros(3, 2, device="mps")
    with pytest.raises(ValueError, match="shape"):
        lib.launch("add", (a, za), (b, zb2))
    with pytest.raises(ValueError, match="shape"):
        lib.launch("neg", (a, za), out=(b, zb2))
    with pytest.raises(ValueError, match="shape"):
        lib.launch("neg", (a, zb2))
    # scalars must be real numbers
    for bad in ("1.5", b"2", True, torch.tensor(1.5), [1.5], None):
        with pytest.raises((TypeError, ValueError)):
            lib.launch("add", (a, za), scalar=bad, scalar_side="right")
    hi, lo = lib.launch("add", (a, za), scalar=np.float64(1.5))
    assert encode.from_mps_df64(hi, lo).tolist() == [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]


def test_op_table_flags_the_sf64_bridge_as_inexact():
    """The op table promised an 'inexact in sf64 mode' flag: 27 ops (24 unary
    transcendentals + pow/atan2/hypot) carry 2^-48, everything else 2^-53."""
    inexact = codegen.SF64_INEXACT
    assert len(inexact) == 27
    assert set(inexact) == {
        n for n, s in codegen.OPS.items() if s.sf_header == codegen.SF_MATH
    }
    assert {"exp", "sin", "lgamma", "pow", "atan2", "hypot"} <= set(inexact)
    assert not {"add", "sqrt", "rsqrt", "fmod", "eq", "fma"} & set(inexact)
    for op in codegen.OPS:
        assert codegen.op_machine_eps(op, "df64") == 2.0**-48
        assert library.op_machine_eps(op, "sf64") == (
            2.0**-48 if op in inexact else 2.0**-53
        )
    df = library.get_library("df64")
    assert not df.is_inexact("exp") and df.machine_eps("exp") == 2.0**-48
    if (metal_compile._KERNEL_DIR / "vendor" / "softfloat64.metal").exists():
        sf = library.get_library("sf64")
        assert sf.is_inexact("exp") and not sf.is_inexact("sqrt")
        assert sf.machine_eps("exp") == 2.0**-48 and sf.machine_eps("sqrt") == 2.0**-53


def test_design_note_matches_the_dispatch_chunk():
    note = (PROJECT_ROOT / "NOTES" / "02-design.md").read_text()
    assert "split above 2^30 elements" in note
    assert "2^31 elements" not in note
    assert library.DEFAULT_CHUNK == 1 << 30
