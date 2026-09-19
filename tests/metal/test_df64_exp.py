"""Accuracy and special-value tests for ``df64_math_exp.h`` (exp/log family).

The header is amalgamated with ``df64_core.h`` and ``df64_constants.h`` exactly as
``compile.kernel_source`` does it, wrapped with one test kernel per function, and
evaluated on the GPU. References come from mpmath at 120 bits evaluated on the
*decoded* inputs (hi + lo summed exactly), never on the original float64 values.
Errors are reported in units of u^2 = 2^-48.

Denormal flush: this GPU flushes float32 denormals in arithmetic, so a low word
below 2^-126 is lost. That costs up to 2^-126 / |x| relative, which exceeds 1 u^2
for |x| < ~3.4e-24. Inputs are therefore encoded with a denormal low word zeroed
(the pair then decodes to exactly what the GPU sees) and u^2-level claims are
only made where |x| and |f(x)| are >= 1e-23; smaller magnitudes are still checked
at float32 level.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mpmath = pytest.importorskip("mpmath")

if not torch.backends.mps.is_available():
    pytest.skip("torch MPS (Metal) is not available", allow_module_level=True)

from mpmath import mp, mpf  # noqa: E402

from optiland.backend.torch_backend.metal import compile as metal_compile  # noqa: E402

mp.prec = 120

U2 = mpf(2) ** -48
FLT_MIN = 2.0**-126
MAG_MIN = 1e-23  # magnitude below which u^2 claims are impossible (denormal lo)
F32_TOL = 2.0**-22  # relative tolerance for the flushed-lo regime

UNARY = ["exp", "exp2", "exp10", "expm1", "log", "log1p", "log2", "log10", "cbrt"]

_UNARY_KERNEL = """
kernel void t_{name}(device const float* xh [[buffer(0)]],
                     device const float* xl [[buffer(1)]],
                     device float* oh [[buffer(2)]],
                     device float* ol [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {{
    df64 r = df::{name}(df::make(xh[i], xl[i]));
    oh[i] = r.hi;
    ol[i] = r.lo;
}}
"""

_BINARY_KERNELS = """
kernel void t_pow(device const float* ah [[buffer(0)]],
                  device const float* al [[buffer(1)]],
                  device const float* bh [[buffer(2)]],
                  device const float* bl [[buffer(3)]],
                  device float* oh [[buffer(4)]],
                  device float* ol [[buffer(5)]],
                  uint i [[thread_position_in_grid]]) {
    df64 r = df::pow(df::make(ah[i], al[i]), df::make(bh[i], bl[i]));
    oh[i] = r.hi;
    ol[i] = r.lo;
}
kernel void t_powi(device const float* ah [[buffer(0)]],
                   device const float* al [[buffer(1)]],
                   device const int* n [[buffer(2)]],
                   device float* oh [[buffer(3)]],
                   device float* ol [[buffer(4)]],
                   uint i [[thread_position_in_grid]]) {
    df64 r = df::pow(df::make(ah[i], al[i]), n[i]);
    oh[i] = r.hi;
    ol[i] = r.lo;
}
"""


def _kernel_source() -> str:
    """Amalgamate core + constants + exp header and append the test kernels."""
    source = metal_compile.kernel_source(
        "df64_core.h", "df64_constants.h", "df64_math_exp.h"
    )
    kernels = "".join(_UNARY_KERNEL.format(name=name) for name in UNARY)
    return source + "\n" + kernels + _BINARY_KERNELS


@pytest.fixture(scope="module")
def lib():
    return metal_compile.compile_library(_kernel_source())


# ---------------------------------------------------------------------------
# Encoding / decoding / launching helpers
# ---------------------------------------------------------------------------
def encode(x) -> tuple[np.ndarray, np.ndarray]:
    """float64 -> (hi, lo) float32 pair; non-finite hi gets lo = 0, denormal lo -> 0."""
    x = np.asarray(x, dtype=np.float64)
    hi = x.astype(np.float32)
    with np.errstate(invalid="ignore"):
        lo = (x - hi.astype(np.float64)).astype(np.float32)
    lo[~np.isfinite(hi)] = 0.0
    lo[np.abs(lo) < FLT_MIN] = 0.0
    return np.ascontiguousarray(hi), np.ascontiguousarray(lo)


def decode(hi, lo) -> list:
    """Exact hi + lo as mpf (NaN/inf pass through as float)."""
    out = []
    for h, l_ in zip(hi.tolist(), lo.tolist(), strict=True):
        if not math.isfinite(h):
            out.append(h)
        else:
            out.append(mpf(h) + mpf(l_))
    return out


def run(lib, name: str, *arrays) -> tuple[np.ndarray, np.ndarray]:
    n = len(arrays[0])
    dev = [torch.from_numpy(np.ascontiguousarray(a)).to("mps") for a in arrays]
    oh = torch.empty(n, dtype=torch.float32, device="mps")
    ol = torch.empty(n, dtype=torch.float32, device="mps")
    getattr(lib, "t_" + name)(
        *dev, oh, ol, threads=[n, 1, 1], group_size=[min(n, 256), 1, 1]
    )
    return oh.cpu().numpy(), ol.cpu().numpy()


def unary(lib, name: str, x):
    hi, lo = encode(x)
    oh, ol = run(lib, name, hi, lo)
    return decode(hi, lo), decode(oh, ol), oh, ol


def rel_errors_u2(outputs, refs) -> np.ndarray:
    """Relative error in u^2 (absolute when the reference is below MAG_MIN)."""
    errs = []
    for o, r in zip(outputs, refs, strict=True):
        if isinstance(r, float) and not math.isfinite(r):
            errs.append(0.0 if (isinstance(o, float) and o == r) else math.inf)
            continue
        if isinstance(o, float) and not math.isfinite(o):
            errs.append(math.inf)
            continue
        if r == 0:
            errs.append(0.0 if o == 0 else math.inf)
            continue
        errs.append(float(abs(o - r) / abs(r) / U2))
    return np.asarray(errs)


def check_accuracy(x, xs, outs, refs, target: float, label: str) -> float:
    """Assert max error <= target where |x|, |f(x)| >= MAG_MIN; float32-check the rest.

    Returns the max error in u^2 over the claimed region.
    """
    errs = rel_errors_u2(outs, refs)
    mag_ok = np.array(
        [abs(v) >= MAG_MIN and abs(r) >= MAG_MIN for v, r in zip(xs, refs, strict=True)]
    )
    worst = int(np.argmax(np.where(mag_ok, errs, -1.0)))
    max_err = float(errs[mag_ok].max()) if mag_ok.any() else 0.0
    assert max_err <= target, (
        f"{label}: max error {max_err:.2f} u^2 > {target} at x={x[worst]!r} "
        f"(got {outs[worst]}, want {refs[worst]})"
    )
    # Flushed-lo regime: still no gross errors.
    if (~mag_ok).any():
        f32 = errs[~mag_ok] * float(U2)
        assert f32.max() <= F32_TOL, (
            f"{label}: float32-level error {f32.max():.3e} in the denormal regime"
        )
    return max_err


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def log_uniform(gen, lo: float, hi: float, n: int, signed: bool = False):
    v = 10.0 ** gen.uniform(np.log10(lo), np.log10(hi), n)
    if signed:
        v *= np.sign(gen.uniform(-1.0, 1.0, n))
    return v


# ---------------------------------------------------------------------------
# Random-input accuracy
# ---------------------------------------------------------------------------
def test_exp_accuracy(lib):
    gen = rng(1)
    x = np.concatenate(
        [gen.uniform(-87.0, 88.0, 4000), log_uniform(gen, 1e-22, 60.0, 4000, True)]
    )
    xs, outs, _, _ = unary(lib, "exp", x)
    refs = [mpmath.exp(v) for v in xs]
    check_accuracy(x, xs, outs, refs, 20.0, "exp")


def test_exp2_accuracy(lib):
    gen = rng(2)
    x = np.concatenate(
        [gen.uniform(-126.0, 128.0, 4000), log_uniform(gen, 1e-22, 60.0, 2000, True)]
    )
    xs, outs, _, _ = unary(lib, "exp2", x)
    refs = [mpf(2) ** v for v in xs]
    check_accuracy(x, xs, outs, refs, 20.0, "exp2")


def test_exp10_accuracy(lib):
    gen = rng(3)
    x = np.concatenate(
        [gen.uniform(-37.9, 38.5, 4000), log_uniform(gen, 1e-22, 30.0, 2000, True)]
    )
    xs, outs, _, _ = unary(lib, "exp10", x)
    refs = [mpf(10) ** v for v in xs]
    check_accuracy(x, xs, outs, refs, 20.0, "exp10")


def test_expm1_accuracy(lib):
    gen = rng(4)
    x = np.concatenate(
        [gen.uniform(-80.0, 88.0, 3000), log_uniform(gen, 1e-22, 60.0, 5000, True)]
    )
    xs, outs, _, _ = unary(lib, "expm1", x)
    refs = [mpmath.expm1(v) for v in xs]
    # Relative accuracy must hold down to tiny x (expm1(x) ~ x): the magnitude
    # gate only removes the flushed regime.
    check_accuracy(x, xs, outs, refs, 20.0, "expm1")


def test_log_accuracy(lib):
    gen = rng(5)
    x = np.concatenate(
        [
            log_uniform(gen, 1e-30, 1e30, 4000),
            1.0 + log_uniform(gen, 1e-16, 0.5, 4000, True),  # near 1: relative
        ]
    )
    xs, outs, _, _ = unary(lib, "log", x)
    refs = [mpmath.log(v) for v in xs]
    check_accuracy(x, xs, outs, refs, 20.0, "log")


def test_log1p_accuracy(lib):
    gen = rng(6)
    x = np.concatenate(
        [
            log_uniform(gen, 1e-30, 1e30, 3000),
            -log_uniform(gen, 1e-30, 0.999, 2000),
            -1.0 + log_uniform(gen, 1e-16, 0.5, 1500),  # near the pole
            log_uniform(gen, 1e-3, 0.3, 1500, True),  # around the series switch
        ]
    )
    xs, outs, _, _ = unary(lib, "log1p", x)
    refs = [mpmath.log1p(v) for v in xs]
    check_accuracy(x, xs, outs, refs, 20.0, "log1p")


def test_log2_accuracy(lib):
    gen = rng(7)
    x = np.concatenate(
        [
            log_uniform(gen, 1e-30, 1e30, 4000),
            1.0 + log_uniform(gen, 1e-16, 0.5, 2000, True),
        ]
    )
    xs, outs, _, _ = unary(lib, "log2", x)
    refs = [mpmath.log(v, 2) for v in xs]
    check_accuracy(x, xs, outs, refs, 20.0, "log2")


def test_log10_accuracy(lib):
    gen = rng(8)
    x = np.concatenate(
        [
            log_uniform(gen, 1e-30, 1e30, 4000),
            1.0 + log_uniform(gen, 1e-16, 0.5, 2000, True),
        ]
    )
    xs, outs, _, _ = unary(lib, "log10", x)
    refs = [mpmath.log10(v) for v in xs]
    check_accuracy(x, xs, outs, refs, 20.0, "log10")


def test_cbrt_accuracy(lib):
    gen = rng(9)
    x = log_uniform(gen, 1e-30, 1e30, 6000, True)
    xs, outs, _, _ = unary(lib, "cbrt", x)
    refs = [mpmath.sign(v) * mpmath.cbrt(abs(v)) for v in xs]
    check_accuracy(x, xs, outs, refs, 20.0, "cbrt")


def _pow_case(lib, a, b):
    ah, al = encode(a)
    bh, bl = encode(b)
    oh, ol = run(lib, "pow", ah, al, bh, bl)
    a_d, b_d = decode(ah, al), decode(bh, bl)
    outs = decode(oh, ol)
    refs = []
    for x, y in zip(a_d, b_d, strict=True):
        if x < 0:
            n = int(y)
            refs.append((-1) ** (n & 1) * mpmath.power(-x, y))
        else:
            refs.append(mpmath.power(x, y))
    return a_d, outs, refs


def test_pow_accuracy(lib):
    gen = rng(10)
    # General exponents with |b log a| <= 20.
    a = log_uniform(gen, 1e-6, 1e6, 4000)
    b = gen.uniform(-1.0, 1.0, 4000) * 20.0 / np.abs(np.log(a))
    a_d, outs, refs = _pow_case(lib, a, b)
    check_accuracy(a, a_d, outs, refs, 50.0, "pow(general)")
    # Integer exponents (npwr for |n| <= 8, exp-log beyond), both signs of a.
    a = log_uniform(gen, 0.5, 2.0, 3000) * np.sign(gen.uniform(-1, 1, 3000))
    n = gen.integers(-64, 65, 3000).astype(np.float64)
    a_d, outs, refs = _pow_case(lib, a, n)
    check_accuracy(a, a_d, outs, refs, 50.0, "pow(integer b)")
    # Large a with fractional b close to +-0 and near-1 bases with huge b.
    a = 1.0 + log_uniform(gen, 1e-12, 1e-3, 2000, True)
    b = gen.uniform(-1.0, 1.0, 2000) * 20.0 / np.abs(np.log(a))
    a_d, outs, refs = _pow_case(lib, a, b)
    check_accuracy(a, a_d, outs, refs, 50.0, "pow(near-1 base)")


def test_pow_int_accuracy(lib):
    gen = rng(11)
    a = log_uniform(gen, 0.5, 2.0, 4000) * np.sign(gen.uniform(-1, 1, 4000))
    n = gen.integers(-64, 65, 4000).astype(np.int32)
    ah, al = encode(a)
    oh, ol = run(lib, "powi", ah, al, n)
    a_d = decode(ah, al)
    outs = decode(oh, ol)
    refs = [mpmath.power(x, int(k)) for x, k in zip(a_d, n, strict=True)]
    check_accuracy(a, a_d, outs, refs, 50.0, "pow(df64, int)")
    # Small |n| goes through npwr: check the exact cases.
    a = np.array([2.0, -2.0, 1.5, 3.0, 0.5, 10.0, -4.0, 2.0])
    n = np.array([8, 3, 3, 4, -4, 2, -1, -8], dtype=np.int32)
    ah, al = encode(a)
    oh, ol = run(lib, "powi", ah, al, n)
    np.testing.assert_array_equal(
        oh, np.array(a ** n.astype(np.float64), dtype=np.float32)
    )
    np.testing.assert_array_equal(ol, np.zeros_like(ol))


# ---------------------------------------------------------------------------
# Special values and exact identities
# ---------------------------------------------------------------------------
INF = math.inf
NAN = math.nan

UNARY_SPECIALS = {
    "exp": [
        (0.0, 1.0),
        (-0.0, 1.0),
        (INF, INF),
        (-INF, 0.0),
        (NAN, NAN),
        (88.73, INF),
        (-87.34, 0.0),
        (-200.0, 0.0),
        (200.0, INF),
    ],
    "exp2": [
        (0.0, 1.0),
        (-0.0, 1.0),
        (INF, INF),
        (-INF, 0.0),
        (NAN, NAN),
        (128.0, INF),
        (-126.5, 0.0),
        (-200.0, 0.0),
        (10.0, 1024.0),
        (-3.0, 0.125),
        (127.0, 2.0**127),
        (-126.0, 2.0**-126),
    ],
    "exp10": [
        (0.0, 1.0),
        (-0.0, 1.0),
        (INF, INF),
        (-INF, 0.0),
        (NAN, NAN),
        (39.0, INF),
        (-38.0, 0.0),
    ],
    "expm1": [
        (0.0, 0.0),
        (-0.0, -0.0),
        (INF, INF),
        (-INF, -1.0),
        (NAN, NAN),
        (88.73, INF),
        (-100.0, -1.0),
    ],
    "log": [
        (0.0, -INF),
        (-0.0, -INF),
        (-1.0, NAN),
        (-INF, NAN),
        (INF, INF),
        (NAN, NAN),
        (1.0, 0.0),
    ],
    "log1p": [
        (0.0, 0.0),
        (-0.0, -0.0),
        (-1.0, -INF),
        (-1.5, NAN),
        (-INF, NAN),
        (INF, INF),
        (NAN, NAN),
    ],
    "log2": [
        (0.0, -INF),
        (-0.0, -INF),
        (-1.0, NAN),
        (INF, INF),
        (NAN, NAN),
        (1.0, 0.0),
        (8.0, 3.0),
        (0.25, -2.0),
        (2.0**100, 100.0),
        (2.0**-100, -100.0),
    ],
    "log10": [
        (0.0, -INF),
        (-0.0, -INF),
        (-1.0, NAN),
        (INF, INF),
        (NAN, NAN),
        (1.0, 0.0),
    ],
    "cbrt": [
        (0.0, 0.0),
        (-0.0, -0.0),
        (INF, INF),
        (-INF, -INF),
        (NAN, NAN),
        (8.0, 2.0),
        (-27.0, -3.0),
        (1.0, 1.0),
        (-1.0, -1.0),
    ],
}


def _same_value(got_hi, got_lo, want: float, exact: bool) -> bool:
    if math.isnan(want):
        return math.isnan(got_hi)
    if math.isinf(want):
        return got_hi == want and got_lo == 0.0
    if want == 0.0:
        same_sign = math.copysign(1, got_hi) == math.copysign(1, want)
        return got_hi == 0.0 and got_lo == 0.0 and same_sign
    got = mpf(float(got_hi)) + mpf(float(got_lo))
    if exact:
        return got == mpf(want)
    return abs(got - mpf(want)) <= 20 * U2 * abs(mpf(want))


@pytest.mark.parametrize("name", UNARY)
def test_unary_special_values(lib, name):
    cases = UNARY_SPECIALS[name]
    x = np.array([c[0] for c in cases])
    hi, lo = encode(x)
    oh, ol = run(lib, name, hi, lo)
    exact_set = {
        0.0,
        1.0,
        INF,
        -INF,
        NAN,
        3.0,
        -2.0,
        100.0,
        -100.0,
        1024.0,
        0.125,
        2.0**127,
        2.0**-126,
    }
    for (xv, want), h, l_ in zip(cases, oh, ol, strict=True):
        # Powers of two and log2 of them are exact; cbrt(8), 10^x etc. within 20 u^2.
        exact = (
            want in exact_set
            and name not in ("exp10",)
            and not (name == "cbrt" and want not in (0.0, INF, -INF, NAN))
        )
        assert _same_value(h, l_, want, exact), (
            f"{name}({xv!r}) = ({h!r}, {l_!r}), want {want!r} (exact={exact})"
        )


POW_SPECIALS = [
    # (a, b, expected) following C99 pow.
    (0.0, 0.0, 1.0),
    (NAN, 0.0, 1.0),
    (1.0, NAN, 1.0),
    (1.0, INF, 1.0),
    (NAN, 1.0, NAN),
    (2.0, NAN, NAN),
    (0.0, -1.0, INF),
    (-0.0, -1.0, -INF),
    (0.0, -2.0, INF),
    (-0.0, -2.0, INF),
    (0.0, -0.5, INF),
    (-0.0, -0.5, INF),
    (0.0, 3.0, 0.0),
    (-0.0, 3.0, -0.0),
    (0.0, 2.0, 0.0),
    (-0.0, 2.0, 0.0),
    (-0.0, 0.5, 0.0),
    (-1.0, INF, 1.0),
    (-1.0, -INF, 1.0),
    (0.5, -INF, INF),
    (0.5, INF, 0.0),
    (2.0, INF, INF),
    (2.0, -INF, 0.0),
    (-2.0, INF, INF),
    (-0.5, -INF, INF),
    (INF, -1.0, 0.0),
    (INF, 2.0, INF),
    (INF, 0.5, INF),
    (-INF, -3.0, -0.0),
    (-INF, -2.0, 0.0),
    (-INF, 3.0, -INF),
    (-INF, 2.0, INF),
    (-INF, 0.5, INF),
    (-2.0, 0.5, NAN),
    (-2.0, 1.5, NAN),
    (-8.0, 1.0 / 3.0, NAN),
    (-2.0, 3.0, -8.0),
    (-2.0, 2.0, 4.0),
    (-2.0, -3.0, -0.125),
    (-2.0, 10.0, 1024.0),
    (-2.0, 11.0, -2048.0),
    (2.0, 0.5, math.sqrt(2.0)),
    (4.0, 0.5, 2.0),
    (4.0, -0.5, 0.5),
    (2.0, -1.0, 0.5),
    (3.0, 2.0, 9.0),
    (2.0, 10.0, 1024.0),
    (10.0, 2.0, 100.0),
    (2.0, 1.0, 2.0),
    (7.5, 1.0, 7.5),
    (1.0, 1e30, 1.0),
    (1e30, 0.0, 1.0),
    (2.0, 60.0, 2.0**60),
    (2.0, -60.0, 2.0**-60),
    (1e20, 2.0, INF),
    (1e-20, 2.0, 0.0),
    (10.0, 38.6, INF),
    (10.0, -38.0, 0.0),
    (1e10, 1e10, INF),
]


def test_pow_special_values(lib):
    a = np.array([c[0] for c in POW_SPECIALS])
    b = np.array([c[1] for c in POW_SPECIALS])
    ah, al = encode(a)
    bh, bl = encode(b)
    oh, ol = run(lib, "pow", ah, al, bh, bl)
    for (av, bv, want), h, l_ in zip(POW_SPECIALS, oh, ol, strict=True):
        exact = bv == bv and abs(bv) <= 8 and want not in (math.sqrt(2.0),)
        assert _same_value(h, l_, want, exact), (
            f"pow({av!r}, {bv!r}) = ({h!r}, {l_!r}), want {want!r}"
        )


def test_pow_int_special_values(lib):
    cases = [
        (0.0, -1, INF),
        (-0.0, -1, -INF),
        (-0.0, -3, -INF),
        (-0.0, -2, INF),
        (0.0, 5, 0.0),
        (-0.0, 5, -0.0),
        (-0.0, 4, 0.0),
        (INF, -2, 0.0),
        (-INF, 3, -INF),
        (-INF, 2, INF),
        (-INF, -3, -0.0),
        (NAN, 0, 1.0),
        (NAN, 2, NAN),
        (0.0, 0, 1.0),
        (INF, 0, 1.0),
        (2.0, 100, 2.0**100),
        (2.0, -100, 2.0**-100),
        (-2.0, 9, -512.0),
        (-2.0, 33, -(2.0**33)),
        (-1.5, 3, -3.375),
        (1e20, 3, INF),
        (1e-20, 3, 0.0),
    ]
    a = np.array([c[0] for c in cases])
    n = np.array([c[1] for c in cases], dtype=np.int32)
    ah, al = encode(a)
    oh, ol = run(lib, "powi", ah, al, n)
    for (av, nv, want), h, l_ in zip(cases, oh, ol, strict=True):
        exact = abs(nv) <= 8 or av in (0.0, -0.0, INF, -INF)
        assert _same_value(h, l_, want, exact), (
            f"pow({av!r}, {nv}) = ({h!r}, {l_!r}), want {want!r}"
        )


def test_exact_identities(lib):
    hi, lo = encode(np.array([0.0, -0.0]))
    oh, ol = run(lib, "exp", hi, lo)
    assert oh.tolist() == [1.0, 1.0] and ol.tolist() == [0.0, 0.0]
    hi, lo = encode(np.array([1.0]))
    oh, ol = run(lib, "log", hi, lo)
    assert oh.tolist() == [0.0] and ol.tolist() == [0.0]
    assert math.copysign(1, oh[0]) == 1.0
    # exp2 is exact on integers, log2 exact on powers of two.
    k = np.arange(-126, 128, dtype=np.float64)
    hi, lo = encode(k)
    oh, ol = run(lib, "exp2", hi, lo)
    np.testing.assert_array_equal(oh, (2.0**k).astype(np.float32))
    np.testing.assert_array_equal(ol, np.zeros_like(ol))
    hi, lo = encode(2.0**k)
    oh, ol = run(lib, "log2", hi, lo)
    np.testing.assert_array_equal(oh, k.astype(np.float32))
    np.testing.assert_array_equal(ol, np.zeros_like(ol))


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------
def test_exp_log_round_trip(lib):
    gen = rng(12)
    x = log_uniform(gen, 1e-23, 1e23, 20000)
    hi, lo = encode(x)
    lh, ll = run(lib, "log", hi, lo)
    eh, el = run(lib, "exp", lh, ll)
    errs = rel_errors_u2(decode(eh, el), decode(hi, lo))
    # log(x) is rounded to 48 bits before exp sees it, so the round trip carries
    # |log x| * (rounding of log) + exp's own error; 40 u^2 is the contract.
    assert errs.max() <= 40.0, f"exp(log(x)) max error {errs.max():.2f} u^2"


def test_exp_monotonic(lib):
    gen = rng(13)
    x = np.sort(gen.uniform(-85.0, 88.0, 200000))
    hi, lo = encode(x)
    oh, ol = run(lib, "exp", hi, lo)
    y = oh.astype(np.float64) + ol.astype(np.float64)
    assert np.all(np.isfinite(y))
    assert np.all(np.diff(y) >= 0.0), "exp is not monotone on a sorted sample"
    # Same pair-wise check on the pairs (hi, then lo) rather than the float64 sum.
    order = np.lexsort((ol, oh))
    assert np.array_equal(order, np.arange(len(x)))


def test_no_nan_on_finite_domain(lib):
    gen = rng(14)
    n = 200000
    checks = {
        "exp": gen.uniform(-87.0, 88.0, n),
        "expm1": gen.uniform(-87.0, 88.0, n),
        "exp2": gen.uniform(-126.0, 127.9, n),
        "exp10": gen.uniform(-37.9, 38.5, n),
        "log": log_uniform(gen, 1e-37, 1e37, n),
        "log1p": log_uniform(gen, 1e-37, 1e37, n),
        "log2": log_uniform(gen, 1e-37, 1e37, n),
        "log10": log_uniform(gen, 1e-37, 1e37, n),
        "cbrt": log_uniform(gen, 1e-37, 1e37, n, True),
    }
    for name, x in checks.items():
        hi, lo = encode(x)
        oh, ol = run(lib, name, hi, lo)
        assert np.all(np.isfinite(oh)), f"{name}: non-finite hi on finite inputs"
        assert np.all(np.isfinite(ol)), f"{name}: non-finite lo on finite inputs"
        assert np.all(np.abs(ol) <= np.abs(np.spacing(oh))), f"{name}: not normalized"


# ---------------------------------------------------------------------------
# Regression tests for the fix round 1 findings (NOTES/04-kernel-status.md 8).
# Raw (hi, lo) pairs are launched directly so that the exact representation
# reaches the kernel.
# ---------------------------------------------------------------------------
FLT_MAX = float(np.finfo(np.float32).max)


def _raw(lib, name: str, pairs):
    hi = np.array([p[0] for p in pairs], dtype=np.float32)
    lo = np.array([p[1] for p in pairs], dtype=np.float32)
    oh, ol = run(lib, name, hi, lo)
    return oh.astype(np.float64), ol.astype(np.float64)


def _raw2(lib, pairs_a, pairs_b):
    ah = np.array([p[0] for p in pairs_a], dtype=np.float32)
    al = np.array([p[1] for p in pairs_a], dtype=np.float32)
    bh = np.array([p[0] for p in pairs_b], dtype=np.float32)
    bl = np.array([p[1] for p in pairs_b], dtype=np.float32)
    oh, ol = run(lib, "pow", ah, al, bh, bl)
    return oh.astype(np.float64), ol.astype(np.float64)


def _u2(got_hi, got_lo, ref) -> float:
    g = mpf(float(got_hi)) + mpf(float(got_lo))
    return float(abs(g - ref) / abs(ref) / U2)


def test_pow_infinite_exponent_uses_whole_pair(lib):
    """|a| < 1 was decided from a.hi alone: pow((1, -2^-30), +inf) gave +inf."""
    tiny = 2.0**-30
    a = [
        (1.0, -tiny),
        (1.0, -tiny),
        (-1.0, tiny),
        (-1.0, tiny),
        (1.0, tiny),
        (1.0, tiny),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, 0.0),
        (2.0, 0.0),
    ]
    b = [
        (np.inf, 0.0),
        (-np.inf, 0.0),
        (np.inf, 0.0),
        (-np.inf, 0.0),
        (np.inf, 0.0),
        (-np.inf, 0.0),
        (np.inf, 0.0),
        (-np.inf, 0.0),
        (-np.inf, 0.0),
        (np.inf, 0.0),
    ]
    hi, lo = _raw2(lib, a, b)
    assert hi.tolist() == [
        0.0,
        np.inf,
        0.0,
        np.inf,
        np.inf,
        0.0,
        1.0,
        1.0,
        np.inf,
        np.inf,
    ]
    assert lo.tolist() == [0.0] * 10


def test_pow_int_near_int32_max(lib):
    """from_int was off by one for n in [2^31 - 64, 2^31): pow(-1, INT_MAX) = +1."""
    ns = np.array(
        [
            2147483647,
            2147483646,
            2147483585,
            2147483584,
            2147483583,
            -2147483648,
            -2147483647,
            9,
        ],
        dtype=np.int32,
    )
    base = [(-1.0, 0.0)] * ns.size
    hi, lo = run(
        lib,
        "powi",
        np.array([p[0] for p in base], np.float32),
        np.array([p[1] for p in base], np.float32),
        ns,
    )
    assert hi.tolist() == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0], list(
        zip(ns.tolist(), hi.tolist(), strict=True)
    )
    # (1 + 2^-40)^(2^31 - 1) through exp(n log a) (the exponent must be n, not n + 1)
    a_hi, a_lo = np.array([1.0], np.float32), np.array([2.0**-40], np.float32)
    for n in (2147483647, 2147483584, 2147483583):
        oh, ol = run(lib, "powi", a_hi, a_lo, np.array([n], np.int32))
        ref = mpmath.power(mpf(1) + mpf(2) ** -40, n)
        assert _u2(oh[0], ol[0], ref) <= 50.0, (n, oh[0] + ol[0], float(ref))


def test_exp2_just_below_128_is_finite(lib):
    """The guard tested x.hi >= 128 only; (128, negative lo) is below 128."""
    los = [
        -9.536743e-07,
        -1e-6,
        -1e-7,
        -(2.0**-16),
        -(2.0**-23),
        -(2.0**-24),
        -7.6e-6,
        -4.3e-8,
    ]
    hi, lo = _raw(lib, "exp2", [(128.0, l_) for l_ in los])
    assert np.all(np.isfinite(hi)), hi
    for l_, h, low in zip(los, hi, lo, strict=True):
        ref = mpmath.power(2, mpf(128) + mpf(float(np.float32(l_))))
        assert _u2(h, low, ref) <= 20.0, (l_, h + low, float(ref))
    # exactly 128 and (128, +lo) overflow; 2^127.99999 (hi below 128) is finite
    hi, lo = _raw(
        lib,
        "exp2",
        [(128.0, 0.0), (128.0, 1e-7), (np.float32(127.99999), 0.0), (128.0, -1e-9)],
    )
    assert hi.tolist()[:2] == [np.inf, np.inf] and lo.tolist()[:2] == [0.0, 0.0]
    assert np.isfinite(hi[2]) and hi[3] == np.inf  # 2^(128 - 1e-9) > FLT_MAX + ulp/2


def test_exp_family_guards_at_flt_range(lib):
    """Guards moved from +-88.72/-87.33 to ln FLT_MAX / ln FLT_MIN: every
    representable result is produced, results below FLT_MIN flush to +0."""
    ln_max = mpmath.log(mpf(FLT_MAX))
    xs = [
        88.72001,
        88.7228,
        float(np.nextafter(np.float32(ln_max), np.float32(0))),
        -87.33001,
        -87.336544,
        88.7229,
        -87.3366,
        88.73,
        -87.35,
    ]
    hi, lo = _raw(lib, "exp", [(np.float32(x), 0.0) for x in xs])
    for x, h, low in zip(xs, hi, lo, strict=True):
        ref = mpmath.exp(mpf(float(np.float32(x))))
        if ref > mpf(FLT_MAX) * (1 + mpf(2) ** -25):
            assert h == np.inf and low == 0.0, (x, h)
        elif ref < mpf(2) ** -126:
            assert h == 0.0 and low == 0.0, (x, h)
        else:
            assert np.isfinite(h) and h >= 2.0**-126, (x, h)
            # results below ~2e-31 lose their lo word (FTZ): float32 accuracy
            tol = 20.0 if ref >= mpf(2) ** -102 else float(2.0**-23 / U2)
            assert _u2(h, low, ref) <= tol, (x, h + low, float(ref))
    # pow(2, n) agrees with exp2(n) at the range edges
    hi, lo = _raw2(
        lib,
        [(2.0, 0.0), (2.0, 0.0), (2.0, 0.0), (10.0, 0.0), (10.0, 0.0)],
        [
            (-126.0, 0.0),
            (np.float32(127.999), 0.0),
            (-127.0, 0.0),
            (-37.929, 0.0),
            (38.531, 0.0),
        ],
    )
    assert hi[0] == 2.0**-126 and lo[0] == 0.0
    assert _u2(hi[1], lo[1], mpmath.power(2, mpf(float(np.float32(127.999))))) <= 20.0
    assert hi[2] == 0.0  # 2^-127 is below FLT_MIN
    assert np.isfinite(hi[3]) and hi[3] >= 2.0**-126 and np.isfinite(hi[4])
    hi, lo = _raw(
        lib,
        "exp10",
        [
            (np.float32(-37.929), 0.0),
            (np.float32(38.531), 0.0),
            (np.float32(38.532), 0.0),
            (np.float32(-37.93), 0.0),
        ],
    )
    assert np.isfinite(hi[0]) and hi[0] >= 2.0**-126 and np.isfinite(hi[1])
    assert hi[2] == np.inf and hi[3] == 0.0
    # no denormal hi word is ever emitted
    x = np.linspace(-87.4, -87.3, 2000)
    hi, lo = _raw(lib, "exp", [(np.float32(v), 0.0) for v in x])
    assert np.all((hi == 0.0) | (hi >= 2.0**-126))


def test_cbrt_and_sqrt_on_noncanonical_pairs(lib):
    """(512, -3e-5) is a valid pair (|lo| <= ulp/2) whose hi is not RN(x); the
    add() exact-zero shortcut used to return -2^-24 for c - f = 0 inside Newton."""
    pairs = [
        (512.0, -3.0e-5),
        (1024.0, -4.70961e-5),
        (4.0, -2e-7),
        (32.0, -1.8e-6),
        (np.float32(511.99997), 1.4e-5),
    ]
    hi, lo = _raw(lib, "cbrt", pairs)
    for (h, l_), gh, gl in zip(pairs, hi, lo, strict=True):
        x = mpf(float(np.float32(h))) + mpf(float(np.float32(l_)))
        assert _u2(gh, gl, mpmath.cbrt(x)) <= 20.0, (
            h,
            l_,
            gh + gl,
            float(mpmath.cbrt(x)),
        )
    hi, lo = _raw2(lib, pairs, [(0.5, 0.0)] * len(pairs))
    for (h, l_), gh, gl in zip(pairs, hi, lo, strict=True):
        x = mpf(float(np.float32(h))) + mpf(float(np.float32(l_)))
        assert _u2(gh, gl, mpmath.sqrt(x)) <= 4.0, (h, l_, gh + gl)


def test_expm1_log1p_tiny_arguments_return_x(lib):
    """expm1 returned exactly 0 for normal |x| < 512 FLT_MIN (r = x/512 flushed) and
    log1p for |x| < 2 FLT_MIN; both now return x below 2^-50 (error <= |x|/2)."""
    pairs = [
        (1.7782794e-36, 0.0),
        (-5.6234132e-36, 0.0),
        (1.7782794e-38, 0.0),
        (2.0**-126, 0.0),
        (1e-20, 1e-37),
        (-3e-30, 0.0),
        (1e-16, 2e-33),
        (2.0**-51, 0.0),
    ]
    for name in ("expm1", "log1p"):
        hi, lo = _raw(lib, name, pairs)
        for (h, l_), gh, gl in zip(pairs, hi, lo, strict=True):
            assert gh == np.float32(h) and gl == np.float32(l_), (name, h, l_, gh, gl)
    # and just above the cutoff the series takes over with the same accuracy
    x = np.concatenate(
        [
            10.0 ** np.linspace(-15.2, -14.8, 200),
            -(10.0 ** np.linspace(-15.2, -14.8, 200)),
        ]
    )
    for name, fn in (("expm1", mpmath.expm1), ("log1p", mpmath.log1p)):
        xs, outs, _, _ = unary(lib, name, x)
        assert rel_errors_u2(outs, [fn(v) for v in xs]).max() <= 20.0, name


# ---------------------------------------------------------------------------
# Fix round 2 (NOTES/04 section 9): npwr on the frexp mantissa
# ---------------------------------------------------------------------------
def _f32bits(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint32).view(np.float32)[0])


def _pow_both(lib, pairs, ns):
    """pow(a, df64(n)) and pow(a, int n) as decoded float64 pairs."""
    hi, lo = _raw2(lib, pairs, [(float(n), 0.0) for n in ns])
    ih, il = run(
        lib,
        "powi",
        np.array([p[0] for p in pairs], dtype=np.float32),
        np.array([p[1] for p in pairs], dtype=np.float32),
        np.array(ns, dtype=np.int32),
    )
    return (hi, lo), (ih.astype(np.float64), il.astype(np.float64))


def test_pow_negative_int_with_underflowing_intermediate(lib):
    """pow(a, -n), n in [2, 8], returned +-inf when a^n fell below FLT_MIN (a^n was
    formed first and inverted) and only float32 accuracy when a^n < ~3.4e-24;
    npwr now works on the frexp mantissa and applies 2^(n e) once at the end."""
    pairs = [
        (_f32bits(0x1FB504F3), _f32bits(0x12CFE77A)),  # a^2 = 2^-127
        (_f32bits(0x378B95C2), _f32bits(0xAA60ABA1)),  # a^8 = 2^-127
        (_f32bits(0x1FEC1E4A), _f32bits(0x137B6D2B)),  # 1e-19: a^-2 = 1e38
        (_f32bits(0x378E9B39), _f32bits(0xAAA7D673)),  # 1.7e-5: a^-8 = 1.43e38
        (_f32bits(0x2A612E13), _f32bits(0x1D849768)),  # 2e-13: a^-3 = 1.25e38
        (_f32bits(0x2002AB03), _f32bits(0x93FF62FB)),  # a^-2 = 8.16e37
        (_f32bits(0x37B71584), _f32bits(0xAB10F80B)),  # a^-8 = 1.94e37
        (_f32bits(0x2B815A8F), _f32bits(0x9F4AD96D)),  # a^-3 = 1.29e36
        (_f32bits(0x38D1B717), _f32bits(0x2C31C433)),  # 1e-4: a^-8 = 1e32
        (_f32bits(0x225D5C66), _f32bits(0x95289350)),  # 3e-18: a^-2 = 1.1e35
        (-_f32bits(0x2A612E13), -_f32bits(0x1D849768)),  # negative base, odd n
    ]
    ns = [-2, -8, -2, -8, -3, -2, -8, -3, -8, -2, -3]
    for hi, lo in _pow_both(lib, pairs, ns):
        for (h, l_), n, gh, gl in zip(pairs, ns, hi, lo, strict=True):
            a = mpf(float(np.float32(h))) + mpf(float(np.float32(l_)))
            ref = mpmath.power(a, n)
            assert math.isfinite(gh), (h, l_, n, gh)
            assert _u2(gh, gl, ref) <= 15.0, (h, l_, n, gh, gl, float(ref))
    # sweep: n in [-8, -2], a^|n| log-uniform in [2^-126, 2^-60] (result up to 2^126)
    rng = np.random.default_rng(9)
    n_pts = 3000
    ns = rng.integers(-8, -1, n_pts)
    ln_target = rng.uniform(-126 * math.log(2), -60 * math.log(2), n_pts)
    a = np.exp(ln_target / np.abs(ns))
    ah, al = encode(a)
    (hi, lo), (ih, il) = _pow_both(
        lib, list(zip(ah.tolist(), al.tolist(), strict=True)), ns
    )
    for h, l_, n, gh, gl, jh, jl in zip(ah, al, ns, hi, lo, ih, il, strict=True):
        ref = mpmath.power(mpf(float(h)) + mpf(float(l_)), int(n))
        assert math.isfinite(gh) and _u2(gh, gl, ref) <= 15.0, (h, l_, n, gh, gl)
        assert (jh, jl) == (gh, gl)
    # results in the denormal range still underflow to +0 and beyond FLT_MAX to inf
    (hi, lo), (ih, il) = _pow_both(
        lib, [(1e-30, 0.0), (1e30, 0.0), (1e30, 0.0)], [3, 2, -2]
    )
    assert hi.tolist() == [0.0, np.inf, 0.0] and ih.tolist() == [0.0, np.inf, 0.0]
    # (+-inf)^n through the int overload (C99)
    ih, il = run(
        lib,
        "powi",
        np.array([np.inf, -np.inf, -np.inf, np.inf, -np.inf], dtype=np.float32),
        np.zeros(5, dtype=np.float32),
        np.array([3, 3, 2, -2, -3], dtype=np.int32),
    )
    assert (
        ih.tolist() == [np.inf, -np.inf, np.inf, 0.0, 0.0] and il.tolist() == [0.0] * 5
    )
    assert np.signbit(ih).tolist() == [False, True, False, False, True]


def test_pow_small_int_top_half_ulp_window(lib):
    """pow(a, 2/3/4) returned +inf when hi*hi rounded to 2^128 although the exact
    power is below the overflow tie FLT_MAX + 2^103 (df::sqr / df::mul now redo
    the product on halved operands)."""
    pairs = [
        (
            _f32bits(0x5F800000),
            _f32bits(0xD3000000),
        ),  # (2^64 - 2^39)^2 = 2^128 - 2^104 + 2^78
        (_f32bits(0x54CB2FF5), _f32bits(0xC6449BA6)),  # ^3 = 3.40282352528e38
        (_f32bits(0x4F800000), _f32bits(0xC2800000)),  # (2^32 - 64)^4
        (-_f32bits(0x5F800000), -_f32bits(0xD3000000)),
    ]
    ns = [2, 3, 4, 2]
    for hi, lo in _pow_both(lib, pairs, ns):
        for (h, l_), n, gh, gl in zip(pairs, ns, hi, lo, strict=True):
            a = mpf(float(np.float32(h))) + mpf(float(np.float32(l_)))
            ref = mpmath.power(a, n)
            assert math.isfinite(gh), (h, l_, n)
            assert _u2(gh, gl, ref) <= 5.0 * n, (h, l_, n, gh, gl, float(ref))
    # the tie itself and anything above overflow
    (hi, lo), (ih, il) = _pow_both(lib, [(2.0**64, 0.0), (2.0**64, -(2.0**38))], [2, 2])
    assert hi.tolist() == [np.inf, np.inf] and ih.tolist() == [np.inf, np.inf]


# ---------------------------------------------------------------------------
# Fix round 3 (NOTES/04 section 10): log / pow of (1, delta) with a tiny lo word
# ---------------------------------------------------------------------------
def test_log_and_pow_of_one_plus_tiny_delta(lib):
    """log((1, delta)) returned exactly 0 for FLT_MIN <= |delta| < 2 FLT_MIN (u / v =
    delta / 2 and frexp's ldexp(delta, -1) flushed) although delta is a normal
    float32, and pow((1, delta), b) = exp(0) = 1 for any b; log2 additionally hit
    its power-of-two shortcut. log_ext now returns delta itself below 2^-50."""
    rng = np.random.default_rng(3)
    n = 2000
    d = (FLT_MIN * rng.uniform(1.0, 2.0, n) * rng.choice([-1.0, 1.0], n)).astype(
        np.float32
    )
    d = d[(np.abs(d) >= FLT_MIN) & (np.abs(d) < 2 * FLT_MIN)]
    pairs = [(1.0, float(v)) for v in d]
    hi, lo = _raw(lib, "log", pairs)
    assert np.array_equal(hi, d.astype(np.float64)) and not lo.any()
    hi, lo = _raw(lib, "log2", pairs)
    ref = d.astype(np.float64) / math.log(2.0)
    assert np.all(np.abs(hi + lo - ref) <= 2.0**-23 * np.abs(ref))  # lo flushes
    hi, lo = _raw(lib, "log10", pairs)
    ref = d.astype(np.float64) / math.log(10.0)
    assert np.all((hi == 0.0) | (np.abs(hi + lo - ref) <= 2.0**-23 * np.abs(ref)))
    # the reproducers: b = 2^126 / 1.5 makes b * delta = +-1 exactly
    a = [(1.0, _f32bits(0x00C00000)), (1.0, _f32bits(0x80C00000))]
    b = [(_f32bits(0x7E2AAAAB), _f32bits(0xF1AAAAAB))] * 2
    hi, lo = _raw2(lib, a, b)
    for got_h, got_l, want in zip(hi, lo, (mpmath.e, 1 / mpmath.e), strict=True):
        assert _u2(got_h, got_l, want) <= 50.0, (got_h, got_l, want)
    # 2^-50 <= |delta| goes through the general path; delta above 2 FLT_MIN was
    # already right; the whole near-1 band stays inside the pow target
    d = 10.0 ** rng.uniform(-37.9, -7.6, 3000) * rng.choice([-1.0, 1.0], 3000)
    d = d.astype(np.float32)
    d = d[np.abs(d) >= FLT_MIN]
    bb = np.clip(
        rng.uniform(-1, 1, d.size) * 20 / np.abs(d.astype(np.float64)), -3e38, 3e38
    )
    bh, bl = encode(bb)
    a_pairs = [(1.0, float(v)) for v in d]
    b_pairs = list(zip(bh.tolist(), bl.tolist(), strict=True))
    hi, lo = _raw2(lib, a_pairs, b_pairs)
    worst = 0.0
    with mp.workprec(200):  # 1 + delta needs more than 120 bits for delta ~ 2^-126
        for (_, dv), (bhv, blv), gh, gl in zip(a_pairs, b_pairs, hi, lo, strict=True):
            x = 1 + mpf(dv)
            y = mpf(bhv) + mpf(blv)
            worst = max(worst, _u2(gh, gl, mpmath.exp(y * mpmath.log(x))))
    assert worst <= 50.0, worst
    xs, outs, _, _ = unary(lib, "log", 1 + d.astype(np.float64))
    assert rel_errors_u2(outs, [mpmath.log(v) for v in xs]).max() <= 20.0
