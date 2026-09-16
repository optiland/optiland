"""Accuracy and special-value tests for ``df64_math_special.h``.

Covers sinh, cosh, tanh, asinh, acosh, atanh, hypot, erf, erfc, erfinv and lgamma.
The header is amalgamated exactly as ``compile.kernel_source`` does it (core,
constants, exp, trig, special), wrapped with one test kernel per function, and
evaluated on the GPU. References come from mpmath at 120 bits on the *decoded*
inputs (hi + lo summed exactly), never on the original float64 values. Errors are
reported in units of u^2 = 2^-48.

The encoding, launch and accuracy helpers are shared with ``test_df64_exp.py``
(same magnitude gate: u^2-level claims only where |x| and |f(x)| >= 1e-23, the
denormal-lo regime is checked at float32 level).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import math  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
mpmath = pytest.importorskip("mpmath")

if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
    pytest.skip("torch MPS (Metal) is not available", allow_module_level=True)

from mpmath import mp, mpf  # noqa: E402

from optiland.backend.torch_backend.metal import compile as metal_compile  # noqa: E402

from .test_df64_exp import (  # noqa: E402
    MAG_MIN,
    U2,
    check_accuracy,
    decode,
    encode,
    log_uniform,
    rel_errors_u2,
    rng,
    run,
)

mp.prec = 120

# Accuracy targets in u^2 (design context).
TOL_HYP = 50.0  # sinh/cosh/tanh/asinh/acosh/atanh
TOL_HYPOT = 10.0
TOL_ERF = 50.0  # erf/erfc
TOL_ERFINV = 100.0  # relative for |x| <= 0.999
TOL_ERFINV_TAIL_ABS = 1e-12  # absolute beyond
TOL_LGAMMA = 100.0  # relative on [0.5, 60]
TOL_COSH_SINH_IDENTITY = 50.0  # |cosh^2 - sinh^2 - 1| / cosh^2
TOL_ERF_ERFC_IDENTITY = 50.0  # |erf + erfc - 1|
TOL_ERFINV_ROUNDTRIP = 200.0
TOL_FACTORIAL = 100.0

UNARY = [
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
]

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

_EXTRA_KERNELS = """
kernel void t_hypot(device const float* ah [[buffer(0)]],
                    device const float* al [[buffer(1)]],
                    device const float* bh [[buffer(2)]],
                    device const float* bl [[buffer(3)]],
                    device float* oh [[buffer(4)]],
                    device float* ol [[buffer(5)]],
                    uint i [[thread_position_in_grid]]) {
    df64 r = df::hypot(df::make(ah[i], al[i]), df::make(bh[i], bl[i]));
    oh[i] = r.hi;
    ol[i] = r.lo;
}
// erfinv(erf(x)) without leaving the GPU (no host re-encoding in between).
kernel void t_erfinv_erf(device const float* xh [[buffer(0)]],
                         device const float* xl [[buffer(1)]],
                         device float* oh [[buffer(2)]],
                         device float* ol [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    df64 r = df::erfinv(df::erf(df::make(xh[i], xl[i])));
    oh[i] = r.hi;
    ol[i] = r.lo;
}
// exp(lgamma(x)): torch's factorial path.
kernel void t_exp_lgamma(device const float* xh [[buffer(0)]],
                         device const float* xl [[buffer(1)]],
                         device float* oh [[buffer(2)]],
                         device float* ol [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    df64 r = df::exp(df::lgamma(df::make(xh[i], xl[i])));
    oh[i] = r.hi;
    ol[i] = r.lo;
}
"""


def _kernel_source() -> str:
    """Amalgamate the full df64 math stack and append the test kernels."""
    source = metal_compile.kernel_source(
        "df64_core.h",
        "df64_constants.h",
        "df64_math_exp.h",
        "df64_math_trig.h",
        "df64_math_special.h",
    )
    kernels = "".join(_UNARY_KERNEL.format(name=name) for name in UNARY)
    return source + "\n" + kernels + _EXTRA_KERNELS


@pytest.fixture(scope="module")
def lib():
    return metal_compile.compile_library(_kernel_source())


def unary(lib, name: str, x):
    hi, lo = encode(x)
    oh, ol = run(lib, name, hi, lo)
    return decode(hi, lo), decode(oh, ol), oh, ol


def min_rel_abs_errors_u2(outputs, refs) -> np.ndarray:
    """min(relative, absolute) error in u^2; used where a function has zeros."""
    rel = rel_errors_u2(outputs, refs)
    out = []
    for o, r, e in zip(outputs, refs, rel, strict=True):
        if isinstance(o, mpf) and isinstance(r, mpf):
            out.append(min(e, float(abs(o - r) / U2)))
        else:
            out.append(e)
    return np.asarray(out)


def check_min_rel_abs(x, outs, refs, target: float, label: str) -> float:
    errs = min_rel_abs_errors_u2(outs, refs)
    worst = int(np.argmax(errs))
    assert errs.max() <= target, (
        f"{label}: max min(rel, abs) error {errs.max():.2f} u^2 > {target} at "
        f"x={x[worst]!r} (got {outs[worst]}, want {refs[worst]})"
    )
    return float(errs.max())


def loggamma_abs(v):
    """log|Gamma(v)| (torch.lgamma semantics) for real v."""
    if v > 0:
        return mpmath.loggamma(v)
    return mpmath.log(abs(mpmath.gamma(v)))


# ---------------------------------------------------------------------------
# Random-input accuracy: hyperbolic family
# ---------------------------------------------------------------------------
def _hyperbolic_inputs(gen, limit: float, n: int):
    return np.concatenate(
        [
            gen.uniform(-limit, limit, n),
            log_uniform(gen, 1e-22, 1.0, n, True),
            log_uniform(gen, 0.02, 0.1, n // 2, True),  # around the series switch
        ]
    )


def test_sinh_accuracy(lib):
    gen = rng(21)
    x = np.concatenate(
        [_hyperbolic_inputs(gen, 88.0, 2500), gen.uniform(88.0, 89.4, 500)]
    )
    xs, outs, _, _ = unary(lib, "sinh", x)
    refs = [mpmath.sinh(v) for v in xs]
    check_accuracy(x, xs, outs, refs, TOL_HYP, "sinh")


def test_cosh_accuracy(lib):
    gen = rng(22)
    x = np.concatenate(
        [_hyperbolic_inputs(gen, 88.0, 2500), gen.uniform(88.0, 89.4, 500)]
    )
    xs, outs, _, _ = unary(lib, "cosh", x)
    refs = [mpmath.cosh(v) for v in xs]
    check_accuracy(x, xs, outs, refs, TOL_HYP, "cosh")


def test_tanh_accuracy(lib):
    gen = rng(23)
    x = _hyperbolic_inputs(gen, 25.0, 2500)
    xs, outs, _, _ = unary(lib, "tanh", x)
    refs = [mpmath.tanh(v) for v in xs]
    check_accuracy(x, xs, outs, refs, TOL_HYP, "tanh")


def test_asinh_accuracy(lib):
    gen = rng(24)
    x = np.concatenate(
        [
            log_uniform(gen, 1e-22, 1e30, 4000, True),
            gen.uniform(-4.0, 4.0, 2000),  # around the |x| = 2 switch
        ]
    )
    xs, outs, _, _ = unary(lib, "asinh", x)
    refs = [mpmath.asinh(v) for v in xs]
    check_accuracy(x, xs, outs, refs, TOL_HYP, "asinh")


def test_acosh_accuracy(lib):
    gen = rng(25)
    x = np.concatenate(
        [
            1.0 + log_uniform(gen, 1e-14, 1e30, 4000),  # relative accuracy near 1
            gen.uniform(1.0, 4.0, 2000),
        ]
    )
    xs, outs, _, _ = unary(lib, "acosh", x)
    refs = [mpmath.acosh(v) for v in xs]
    check_accuracy(x, xs, outs, refs, TOL_HYP, "acosh")


def test_atanh_accuracy(lib):
    gen = rng(26)
    x = np.concatenate(
        [
            log_uniform(gen, 1e-22, 0.999, 3000, True),
            (1.0 - log_uniform(gen, 1e-14, 0.5, 2000))
            * np.sign(gen.uniform(-1, 1, 2000)),
        ]
    )
    xs, outs, _, _ = unary(lib, "atanh", x)
    refs = [mpmath.atanh(v) for v in xs]
    check_accuracy(x, xs, outs, refs, TOL_HYP, "atanh")


def test_hypot_accuracy(lib):
    gen = rng(27)
    n = 6000
    a = log_uniform(gen, 1e-30, 1e30, n, True)
    # Mix of comparable magnitudes and wildly different ones.
    b = np.where(
        gen.uniform(0, 1, n) < 0.5,
        a * gen.uniform(-3.0, 3.0, n),
        log_uniform(gen, 1e-30, 1e30, n, True),
    )
    ah, al = encode(a)
    bh, bl = encode(b)
    oh, ol = run(lib, "hypot", ah, al, bh, bl)
    a_d, b_d = decode(ah, al), decode(bh, bl)
    outs = decode(oh, ol)
    refs = [mpmath.hypot(p, q) for p, q in zip(a_d, b_d, strict=True)]
    check_accuracy(a, a_d, outs, refs, TOL_HYPOT, "hypot")


# ---------------------------------------------------------------------------
# Random-input accuracy: error function family
# ---------------------------------------------------------------------------
def test_erf_accuracy(lib):
    gen = rng(28)
    x = np.concatenate(
        [
            gen.uniform(-6.5, 6.5, 3000),
            log_uniform(gen, 1e-22, 1.0, 2000, True),
            gen.uniform(0.9, 1.1, 500) * np.sign(gen.uniform(-1, 1, 500)),  # switch
            gen.uniform(2.9, 3.1, 500),  # table -> continued fraction
        ]
    )
    xs, outs, _, _ = unary(lib, "erf", x)
    refs = [mpmath.erf(v) for v in xs]
    check_accuracy(x, xs, outs, refs, TOL_ERF, "erf")


def test_erfc_accuracy(lib):
    gen = rng(29)
    x = np.concatenate(
        [
            gen.uniform(-6.5, 9.1, 3000),  # erfc(9.18) ~ FLT_MIN: 0 beyond
            log_uniform(gen, 1e-22, 1.0, 2000, True),
            gen.uniform(0.45, 0.55, 400),  # 1 - erf -> table
            gen.uniform(2.9, 3.1, 400),  # table -> continued fraction
            gen.uniform(0.5, 3.0, 1200),  # table points and their midpoints
        ]
    )
    xs, outs, _, _ = unary(lib, "erfc", x)
    refs = [mpmath.erfc(v) for v in xs]
    # Results below ~2e-31 (x > ~8.4) lose the low word (denormal flush); the
    # magnitude gate in check_accuracy only claims u^2 accuracy above 1e-23.
    check_accuracy(x, xs, outs, refs, TOL_ERF, "erfc")


def test_erfinv_accuracy(lib):
    gen = rng(30)
    x = np.concatenate(
        [
            gen.uniform(-0.999, 0.999, 2500),
            log_uniform(gen, 1e-22, 0.5, 1000, True),
            gen.uniform(0.45, 0.55, 300)
            * np.sign(gen.uniform(-1, 1, 300)),  # residual switch
        ]
    )
    xs, outs, _, _ = unary(lib, "erfinv", x)
    refs = [mpmath.erfinv(v) for v in xs]
    check_accuracy(x, xs, outs, refs, TOL_ERFINV, "erfinv |x| <= 0.999")


def test_erfinv_tail(lib):
    gen = rng(31)
    # 1 - |x| from 1e-3 down to the df64 resolution (2^-49 ~ 1.8e-15).
    q = log_uniform(gen, 3e-15, 1e-3, 1500)
    x = np.concatenate([1.0 - q, -1.0 + q[:500]])
    xs, outs, _, _ = unary(lib, "erfinv", x)
    refs = [mpmath.erfinv(v) for v in xs]
    abs_errs = np.array([float(abs(o - r)) for o, r in zip(outs, refs, strict=True)])
    worst = int(np.argmax(abs_errs))
    assert abs_errs.max() <= TOL_ERFINV_TAIL_ABS, (
        f"erfinv tail: absolute error {abs_errs.max():.3e} at x={x[worst]!r}"
    )
    # The erfc-based residual keeps relative accuracy in the tail as well.
    check_accuracy(x, xs, outs, refs, TOL_ERFINV, "erfinv tail (relative)")


# ---------------------------------------------------------------------------
# Random-input accuracy: lgamma
# ---------------------------------------------------------------------------
def test_lgamma_accuracy_target_range(lib):
    gen = rng(32)
    x = np.concatenate(
        [
            gen.uniform(0.5, 60.0, 3000),
            gen.uniform(0.5, 8.5, 1500),  # series/recurrence region and the switch
            1.0
            + log_uniform(gen, 1e-13, 0.5, 800, True),  # relative accuracy at the zeros
            2.0 + log_uniform(gen, 1e-13, 0.5, 800, True),
            np.arange(0.5, 60.5, 0.5),  # half-integers and integers
        ]
    )
    xs, outs, _, _ = unary(lib, "lgamma", x)
    refs = [mpmath.loggamma(v) for v in xs]
    check_accuracy(x, xs, outs, refs, TOL_LGAMMA, "lgamma [0.5, 60]")


def test_lgamma_accuracy_outside_target_range(lib):
    gen = rng(33)
    # Positive arguments outside [0.5, 60]: still relative.
    x = np.concatenate(
        [
            log_uniform(gen, 1e-20, 0.5, 1500),
            gen.uniform(60.0, 1e6, 1000),
            log_uniform(gen, 1e6, 1e36, 500),
        ]
    )
    xs, outs, _, _ = unary(lib, "lgamma", x)
    refs = [mpmath.loggamma(v) for v in xs]
    check_accuracy(x, xs, outs, refs, TOL_LGAMMA, "lgamma positive")
    # Negative arguments (reflection): log|Gamma| has zeros between the poles, where
    # the reflection formula cancels, so the metric is min(relative, absolute).
    x = np.concatenate(
        [
            gen.uniform(-20.0, 0.0, 2000),
            -log_uniform(gen, 1e-20, 0.5, 500),
            -np.arange(1, 30) - log_uniform(gen, 1e-12, 1e-3, 29),  # next to poles
        ]
    )
    xs, outs, _, _ = unary(lib, "lgamma", x)
    refs = [loggamma_abs(v) for v in xs]
    check_min_rel_abs(x, outs, refs, TOL_LGAMMA, "lgamma negative")


# ---------------------------------------------------------------------------
# Special values
# ---------------------------------------------------------------------------
INF = math.inf
NAN = math.nan
LN2 = float(mpmath.log(2))
LN_SQRT_PI = float(mpmath.log(mpmath.sqrt(mpmath.pi)))

# (input, expected, exact?) ; inexact expectations are checked to 20 u^2.
UNARY_SPECIALS = {
    "sinh": [
        (0.0, 0.0, True),
        (-0.0, -0.0, True),
        (INF, INF, True),
        (-INF, -INF, True),
        (NAN, NAN, True),
        (90.0, INF, True),
        (-90.0, -INF, True),
        (89.5, INF, True),
        (1.0, float(mpmath.sinh(1)), False),
    ],
    "cosh": [
        (0.0, 1.0, True),
        (-0.0, 1.0, True),
        (INF, INF, True),
        (-INF, INF, True),
        (NAN, NAN, True),
        (90.0, INF, True),
        (-90.0, INF, True),
        (1.0, float(mpmath.cosh(1)), False),
    ],
    "tanh": [
        (0.0, 0.0, True),
        (-0.0, -0.0, True),
        (INF, 1.0, True),
        (-INF, -1.0, True),
        (NAN, NAN, True),
        (30.0, 1.0, True),
        (-30.0, -1.0, True),
        (100.0, 1.0, True),
    ],
    "asinh": [
        (0.0, 0.0, True),
        (-0.0, -0.0, True),
        (INF, INF, True),
        (-INF, -INF, True),
        (NAN, NAN, True),
        (1e38, None, None),  # reference on the decoded input (x^2 would overflow)
        (-1e38, None, None),
    ],
    "acosh": [
        (1.0, 0.0, True),
        (0.5, NAN, True),
        (0.0, NAN, True),
        (-1.0, NAN, True),
        (-INF, NAN, True),
        (INF, INF, True),
        (NAN, NAN, True),
        (2.0, float(mpmath.acosh(2)), False),
        (1e38, None, None),
    ],
    "atanh": [
        (0.0, 0.0, True),
        (-0.0, -0.0, True),
        (1.0, INF, True),
        (-1.0, -INF, True),
        (1.5, NAN, True),
        (-2.0, NAN, True),
        (INF, NAN, True),
        (-INF, NAN, True),
        (NAN, NAN, True),
        (0.5, float(mpmath.atanh(0.5)), False),
    ],
    "erf": [
        (0.0, 0.0, True),
        (-0.0, -0.0, True),
        (INF, 1.0, True),
        (-INF, -1.0, True),
        (NAN, NAN, True),
        (10.0, 1.0, True),
        (-10.0, -1.0, True),
        (6.0, 1.0, True),  # erfc(6) < 2^-55, below df64 resolution
        (0.5, float(mpmath.erf(0.5)), False),
    ],
    "erfc": [
        (0.0, 1.0, True),
        (-0.0, 1.0, True),
        (INF, 0.0, True),
        (-INF, 2.0, True),
        (NAN, NAN, True),
        (20.0, 0.0, True),
        (9.5, 0.0, True),  # below FLT_MIN: underflows to +0, never NaN
        (-20.0, 2.0, True),
        (-6.0, 2.0, False),  # 2 - 2.15e-17 is representable: only 20 u^2-close to 2
        (0.5, float(mpmath.erfc(0.5)), False),
    ],
    "erfinv": [
        (0.0, 0.0, True),
        (-0.0, -0.0, True),
        (1.0, INF, True),
        (-1.0, -INF, True),
        (1.5, NAN, True),
        (-1.0000001, NAN, True),
        (INF, NAN, True),
        (-INF, NAN, True),
        (NAN, NAN, True),
        (0.5, float(mpmath.erfinv(0.5)), False),
    ],
    "lgamma": [
        (1.0, 0.0, True),
        (2.0, 0.0, True),
        (0.0, INF, True),
        (-0.0, INF, True),
        (-1.0, INF, True),
        (-2.0, INF, True),
        (-100.0, INF, True),
        (-1e10, INF, True),
        (INF, INF, True),
        (-INF, INF, True),
        (NAN, NAN, True),
        (1e38, INF, True),
        (0.5, LN_SQRT_PI, False),
        (3.0, LN2, False),
        (-0.5, float(mpmath.log(2 * mpmath.sqrt(mpmath.pi))), False),
    ],
}


REF_FN = {
    "sinh": mpmath.sinh,
    "cosh": mpmath.cosh,
    "tanh": mpmath.tanh,
    "asinh": mpmath.asinh,
    "acosh": mpmath.acosh,
    "atanh": mpmath.atanh,
    "erf": mpmath.erf,
    "erfc": mpmath.erfc,
    "erfinv": mpmath.erfinv,
    "lgamma": loggamma_abs,
}


def same_value(got_hi, got_lo, want: float, exact: bool, tol_u2: float = 20.0) -> bool:
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
    return abs(got - mpf(want)) <= tol_u2 * U2 * abs(mpf(want))


@pytest.mark.parametrize("name", UNARY)
def test_unary_special_values(lib, name):
    cases = UNARY_SPECIALS[name]
    x = np.array([c[0] for c in cases])
    hi, lo = encode(x)
    oh, ol = run(lib, name, hi, lo)
    xs = decode(hi, lo)
    for (xv, want, exact), xd, h, l_ in zip(cases, xs, oh, ol, strict=True):
        if exact is None:
            # Reference on the decoded input (the value is not a float32).
            ref = REF_FN[name](xd)
            got = mpf(float(h)) + mpf(float(l_))
            assert abs(got - ref) <= 20 * U2 * abs(ref), (
                f"{name}({xv!r}) = ({h!r}, {l_!r}), want {ref}"
            )
            continue
        assert same_value(h, l_, want, exact), (
            f"{name}({xv!r}) = ({h!r}, {l_!r}), want {want!r} (exact={exact})"
        )


HYPOT_SPECIALS = [
    # (x, y, expected, exact?)
    (3.0, 4.0, 5.0, True),
    (-3.0, 4.0, 5.0, True),
    (3.0, -4.0, 5.0, True),
    (5.0, 12.0, 13.0, True),
    (0.0, 0.0, 0.0, True),
    (-0.0, -0.0, 0.0, True),
    (0.0, -0.0, 0.0, True),
    (-7.5, 0.0, 7.5, True),
    (0.0, -2.25, 2.25, True),
    (INF, 1.0, INF, True),
    (-INF, 1.0, INF, True),
    (1.0, -INF, INF, True),
    (INF, NAN, INF, True),
    (NAN, -INF, INF, True),
    (NAN, 1.0, NAN, True),
    (1.0, NAN, NAN, True),
    (NAN, NAN, NAN, True),
    (1e30, 1e30, None, None),  # reference on the decoded inputs (prescaling)
    (1e-20, 1e-20, None, None),  # squares would underflow without prescaling
    (1e-20, 1e-32, None, None),
    (-2.5e15, 1e-15, None, None),
    (3e38, 3e38, INF, True),  # true result exceeds FLT_MAX
    (1.0, 1e-30, 1.0, True),  # y^2 negligible
    (1.0, 1.0, float(mpmath.sqrt(2)), False),
]


def test_hypot_special_values(lib):
    a = np.array([c[0] for c in HYPOT_SPECIALS])
    b = np.array([c[1] for c in HYPOT_SPECIALS])
    ah, al = encode(a)
    bh, bl = encode(b)
    oh, ol = run(lib, "hypot", ah, al, bh, bl)
    a_d, b_d = decode(ah, al), decode(bh, bl)
    for (av, bv, want, exact), ad, bd, h, l_ in zip(
        HYPOT_SPECIALS, a_d, b_d, oh, ol, strict=True
    ):
        if exact is None:
            ref = mpmath.hypot(ad, bd)
            got = mpf(float(h)) + mpf(float(l_))
            assert abs(got - ref) <= 10 * U2 * ref, (
                f"hypot({av!r}, {bv!r}) = ({h!r}, {l_!r}), want {ref}"
            )
            continue
        assert same_value(h, l_, want, exact, 10.0), (
            f"hypot({av!r}, {bv!r}) = ({h!r}, {l_!r}), want {want!r} (exact={exact})"
        )


def test_hypot_tiny_results_float32_level(lib):
    """Results below ~1e-29 lose the low word (denormal flush) but hi stays right."""
    a = np.array([1e-30, 3e-31, 1e-35, 5e-38])
    ah, al = encode(a)
    bh, bl = encode(a)
    oh, ol = run(lib, "hypot", ah, al, bh, bl)
    for ad, bd, h in zip(decode(ah, al), decode(bh, bl), oh, strict=True):
        ref = mpmath.hypot(ad, bd)
        assert abs(mpf(float(h)) - ref) <= 2.0**-22 * ref, f"hypot({ad}, {bd}) = {h!r}"


def test_lgamma_exact_zeros(lib):
    hi, lo = encode(np.array([1.0, 2.0]))
    oh, ol = run(lib, "lgamma", hi, lo)
    assert oh.tolist() == [0.0, 0.0] and ol.tolist() == [0.0, 0.0]
    assert all(math.copysign(1, h) == 1.0 for h in oh)


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------
def test_cosh_sinh_identity(lib):
    gen = rng(34)
    x = np.concatenate(
        [gen.uniform(-20.0, 20.0, 6000), log_uniform(gen, 1e-10, 1.0, 2000, True)]
    )
    hi, lo = encode(x)
    ch, cl = run(lib, "cosh", hi, lo)
    sh, sl = run(lib, "sinh", hi, lo)
    worst = 0.0
    for c, s in zip(decode(ch, cl), decode(sh, sl), strict=True):
        # Relative to cosh^2: the identity cannot hold absolutely once the
        # 48-bit rounding of cosh (~cosh u^2) exceeds 1 (|x| > ~17).
        worst = max(worst, float(abs(c * c - s * s - 1) / (c * c) / U2))
    assert worst <= TOL_COSH_SINH_IDENTITY, (
        f"cosh^2 - sinh^2 = 1 violated by {worst:.2f} u^2 (relative to cosh^2)"
    )


def test_erf_erfc_identity(lib):
    gen = rng(35)
    x = np.concatenate(
        [gen.uniform(-6.0, 6.0, 8000), log_uniform(gen, 1e-10, 1.0, 2000, True)]
    )
    hi, lo = encode(x)
    eh, el = run(lib, "erf", hi, lo)
    ch, cl = run(lib, "erfc", hi, lo)
    worst = 0.0
    for e, c in zip(decode(eh, el), decode(ch, cl), strict=True):
        worst = max(worst, float(abs(e + c - 1) / U2))
    assert worst <= TOL_ERF_ERFC_IDENTITY, f"erf + erfc = 1 violated by {worst:.2f} u^2"


def test_erfinv_erf_round_trip(lib):
    gen = rng(36)
    x = np.concatenate(
        [gen.uniform(-2.0, 2.0, 5000), log_uniform(gen, 1e-10, 1.0, 1000, True)]
    )
    hi, lo = encode(x)
    oh, ol = run(lib, "erfinv_erf", hi, lo)
    errs = rel_errors_u2(decode(oh, ol), decode(hi, lo))
    worst = int(np.argmax(errs))
    assert errs.max() <= TOL_ERFINV_ROUNDTRIP, (
        f"erfinv(erf(x)) max error {errs.max():.2f} u^2 at x={x[worst]!r}"
    )


def test_factorial_via_exp_lgamma(lib):
    n = np.arange(0, 21, dtype=np.float64)
    hi, lo = encode(n + 1.0)
    oh, ol = run(lib, "exp_lgamma", hi, lo)
    outs = decode(oh, ol)
    for k, got in zip(n, outs, strict=True):
        want = mpmath.factorial(int(k))
        err = float(abs(got - want) / want / U2)
        assert err <= TOL_FACTORIAL, (
            f"exp(lgamma({int(k) + 1})) = {got}, want {want}: {err:.2f} u^2"
        )


def test_odd_even_symmetry(lib):
    """sinh/tanh/asinh/atanh/erf/erfinv are odd and cosh even bit-for-bit."""
    gen = rng(37)
    x = np.concatenate(
        [gen.uniform(0.0, 5.0, 20000), log_uniform(gen, 1e-20, 1.0, 5000)]
    )
    x = x[x < 0.999]  # keeps erfinv/atanh in-domain
    hi, lo = encode(x)
    nhi, nlo = encode(-x)
    for name in ["sinh", "tanh", "asinh", "atanh", "erf", "erfinv"]:
        ph, pl = run(lib, name, hi, lo)
        mh, ml = run(lib, name, nhi, nlo)
        np.testing.assert_array_equal(ph, -mh, err_msg=f"{name} is not odd (hi)")
        np.testing.assert_array_equal(pl, -ml, err_msg=f"{name} is not odd (lo)")
    ph, pl = run(lib, "cosh", hi, lo)
    mh, ml = run(lib, "cosh", nhi, nlo)
    np.testing.assert_array_equal(ph, mh)
    np.testing.assert_array_equal(pl, ml)


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------
def test_no_nan_on_finite_domain(lib):
    gen = rng(38)
    n = 200000
    checks = {
        "sinh": gen.uniform(-89.0, 89.0, n),
        "cosh": gen.uniform(-89.0, 89.0, n),
        "tanh": gen.uniform(-100.0, 100.0, n),
        "asinh": log_uniform(gen, 1e-37, 1e37, n, True),
        "acosh": 1.0 + log_uniform(gen, 1e-37, 1e37, n),
        "atanh": gen.uniform(-0.999999, 0.999999, n),
        "erf": gen.uniform(-30.0, 30.0, n),
        "erfc": gen.uniform(-30.0, 30.0, n),
        "erfinv": gen.uniform(-0.999999, 0.999999, n),
        "lgamma": log_uniform(gen, 1e-37, 1e30, n),
    }
    for name, x in checks.items():
        hi, lo = encode(x)
        oh, ol = run(lib, name, hi, lo)
        assert np.all(np.isfinite(oh)), f"{name}: non-finite hi on finite inputs"
        assert np.all(np.isfinite(ol)), f"{name}: non-finite lo on finite inputs"
        assert np.all(np.abs(ol) <= np.abs(np.spacing(oh))), f"{name}: not normalized"
    # Negative non-integer lgamma arguments are finite too.
    x = gen.uniform(-50.0, 0.0, n)
    x = x[np.abs(x - np.rint(x)) > 1e-6]
    hi, lo = encode(x)
    oh, ol = run(lib, "lgamma", hi, lo)
    assert np.all(np.isfinite(oh)) and np.all(np.isfinite(ol))


def test_monotonic(lib):
    """sinh, tanh, asinh, erf and erfinv are non-decreasing on a sorted sample."""
    gen = rng(39)
    samples = {
        "sinh": np.sort(gen.uniform(-30.0, 30.0, 100000)),
        "tanh": np.sort(gen.uniform(-20.0, 20.0, 100000)),
        "asinh": np.sort(gen.uniform(-100.0, 100.0, 100000)),
        "erf": np.sort(gen.uniform(-6.0, 6.0, 100000)),
        "erfinv": np.sort(gen.uniform(-0.99999, 0.99999, 100000)),
    }
    for name, x in samples.items():
        hi, lo = encode(x)
        oh, ol = run(lib, name, hi, lo)
        order = np.lexsort((ol, oh))
        y = oh.astype(np.float64) + ol.astype(np.float64)
        assert np.all(np.diff(y) >= 0.0), f"{name} is not monotone on a sorted sample"
        assert np.array_equal(order, np.arange(len(x))), (
            f"{name}: pair order not monotone"
        )


def test_magnitude_gate_documented():
    """The u^2 claims above exclude |f(x)| < MAG_MIN (denormal low word)."""
    assert MAG_MIN == 1e-23


# ---------------------------------------------------------------------------
# Fix round 2 (NOTES/04 section 9)
# ---------------------------------------------------------------------------
def test_lgamma_overflows_only_at_the_true_threshold(lib):
    """lgamma returned +inf for z in [4.037e36, 4.085e36] although the result is
    finite: the Stirling term (z - 1/2) log z overflowed before -z was applied.
    The branch now evaluates (z - 1/2)(log z - 1) on halved operands and scales
    back, so overflow happens exactly when lgamma(z) >= FLT_MAX + 2^103."""
    flt_max = float(np.finfo(np.float32).max)
    tie = mpf(flt_max) + mpf(2) ** 103
    z_star = mpmath.findroot(lambda z: mpmath.loggamma(z) - tie, mpf("4.085e36"))
    x = np.concatenate(
        [
            np.linspace(4.0e36, 4.12e36, 1000),
            [float(z_star) * (1 - 1e-7), float(z_star) * (1 + 1e-7), 4.037e36, 4.06e36],
            [8.0, 8.5, 20.5, 1e10, 1e30, 1e36, 1e37],
        ]
    )
    hi, lo = encode(x)
    oh, ol = run(lib, "lgamma", hi, lo)
    for h, l_, gh, gl in zip(hi, lo, oh, ol, strict=True):
        v = mpf(float(h)) + mpf(float(l_))
        ref = mpmath.loggamma(v)
        if ref >= tie:
            assert gh == np.inf and gl == 0.0, (float(v), gh)
            continue
        assert math.isfinite(gh), (float(v), gh)
        got = mpf(float(gh)) + mpf(float(gl))
        assert abs(got - ref) / abs(ref) <= 4.0 * U2, (float(v), gh, gl, float(ref))


# ---------------------------------------------------------------------------
# Fix round 3 (NOTES/04 section 10)
# ---------------------------------------------------------------------------
def _bits32(b: int) -> float:
    return float(np.array([b], dtype=np.uint32).view(np.float32)[0])


def _pairs_run(lib, name, pairs):
    hi = np.array([p[0] for p in pairs], dtype=np.float32)
    lo = np.array([p[1] for p in pairs], dtype=np.float32)
    oh, ol = run(lib, name, hi, lo)
    return decode(hi, lo), decode(oh, ol), oh, ol


def test_lgamma_tiny_negative_arguments_keep_relative_accuracy(lib):
    """lgamma(x) for -1e-26 < x < 0 with a nonzero lo word lost up to 1.5e3 u^2
    (1.5e4 below 1e-29) through the product pi * r in the reflection formula,
    whose cross terms flushed below FLT_MIN; the small-|r| branch now takes
    log|r| directly. The positive mirror image was 0.5 u^2 all along."""
    pairs = [
        (_bits32(0x8F4B24DC), _bits32(0x02F98CC9)),
        (_bits32(0x8F5AA1D1), _bits32(0x82CA0640)),
        (_bits32(0x8DA3E051), _bits32(0x8104E9D7)),
    ]
    xs, outs, _, _ = _pairs_run(lib, "lgamma", pairs)
    refs = [mpmath.log(abs(mpmath.gamma(v))) for v in xs]
    assert rel_errors_u2(outs, refs).max() <= 4.0
    gen = np.random.default_rng(77)
    for lo_, hi_ in ((1e-29, 1e-28), (1e-28, 1e-26), (1e-26, 1e-15), (1e-15, 1e-6)):
        x = -(10.0 ** gen.uniform(np.log10(lo_), np.log10(hi_), 1500))
        xs, outs, _, _ = unary(lib, "lgamma", x)
        refs = [mpmath.log(abs(mpmath.gamma(v))) for v in xs]
        assert rel_errors_u2(outs, refs).max() <= 4.0, (lo_, hi_)
        # and the mirror images on the positive side stay where they were
        xs, outs, _, _ = unary(lib, "lgamma", -x)
        refs = [mpmath.log(abs(mpmath.gamma(v))) for v in xs]
        assert rel_errors_u2(outs, refs).max() <= 4.0, (lo_, hi_)
    # the same branch next to the poles (r = x - rint(x) tiny, x = -n + r)
    x = np.array([-1.0, -2.0, -3.0, -17.0, -40.0]) + np.array(
        [1e-10, -3e-12, 1e-8, 2e-9, -5e-7]
    )
    xs, outs, _, _ = unary(lib, "lgamma", x)
    refs = [mpmath.log(abs(mpmath.gamma(v))) for v in xs]
    assert rel_errors_u2(outs, refs).max() <= 4.0


def test_erfinv_far_tail_is_not_limited_by_the_erfc_floor(lib):
    """erfinv(1 - q) for q in [FLT_MIN, ~5e-29] was float32-accurate (1e6 u^2 on
    the O(8) result): the Newton residual q - erfc(y) was formed on an erfc
    value below 2^-78, which carries the FLT_MIN absolute floor. The residual
    is now formed on 2^k q and 2^k erfc(y) with 2^k q in [1/2, 1)."""
    pairs = [(1.0, _bits32(0x8D836884))]
    with mp.workprec(200):  # 1 - q needs more than 120 bits for q ~ 2^-100
        xs, outs, _, _ = _pairs_run(lib, "erfinv", pairs)
        assert rel_errors_u2(outs, [mpmath.erfinv(v) for v in xs]).max() <= 4.0
        gen = np.random.default_rng(5)
        for lo2, hi2 in ((-100, -94), (-94, -80), (-80, -60), (-60, -40), (-40, -24)):
            q = (2.0 ** gen.uniform(lo2, hi2, 150)).astype(np.float32)
            pairs = [(1.0, -float(v)) for v in q]
            xs, outs, _, _ = _pairs_run(lib, "erfinv", pairs)
            refs = [mpmath.erfinv(v) for v in xs]
            assert rel_errors_u2(outs, refs).max() <= 4.0, (lo2, hi2)
            # odd: the mirror image on the negative side
            pairs = [(-1.0, float(v)) for v in q]
            xs, outs, _, _ = _pairs_run(lib, "erfinv", pairs)
            refs = [mpmath.erfinv(v) for v in xs]
            assert rel_errors_u2(outs, refs).max() <= 4.0, (lo2, hi2)


def test_erfc_underflow_point_matches_the_header(lib):
    """erfc(x) is +0 exactly when erfc(x) < FLT_MIN, i.e. for x > 9.194549; the
    header comments used to say ~9.34."""
    x = np.array([9.19, 9.1945, 9.1946, 9.2, 9.34, 9.5, 12.0])
    _, _, oh, ol = unary(lib, "erfc", x)
    assert oh[0] > 0 and oh[1] > 0
    assert abs(oh[1] - float(mpmath.erfc(mpf("9.1945")))) <= 2.0**-23 * oh[1]
    assert oh[2:].tolist() == [0.0] * 5 and ol[2:].tolist() == [0.0] * 5
    crossing = mpmath.findroot(lambda v: mpmath.erfc(v) - mpf(2) ** -126, 9.1945)
    assert abs(float(crossing) - 9.194549) < 5e-7
    header = (metal_compile._KERNEL_DIR / "df64_math_special.h").read_text()
    assert "9.194549" in header and "9.34" not in header


def test_lgamma_of_a_denormal_hi_word_is_the_documented_pole(lib):
    """A float32-denormal hi word is +-0 for every kernel (df64_core.h denormal
    contract), so lgamma of it is the pole +inf; the header documents it now.
    The host encoder never produces such a pair (it flushes to +-0 itself)."""
    pairs = [(1e-40, 0.0), (-1e-40, 0.0), (1e-45, 0.0), (0.0, 0.0), (-0.0, 0.0)]
    _, _, oh, ol = _pairs_run(lib, "lgamma", pairs)
    assert oh.tolist() == [np.inf] * 5 and ol.tolist() == [0.0] * 5
    header = (metal_compile._KERNEL_DIR / "df64_math_special.h").read_text()
    assert "A denormal hi word" in header and "lgamma of it is the pole" in header
