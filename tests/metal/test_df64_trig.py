"""Accuracy tests for ``df64_math_trig.h`` (sin, cos, tan, asin, acos, atan, atan2).

The Metal kernels are compiled from the amalgamation ``df64_core.h`` +
``df64_constants.h`` + ``df64_math_trig.h`` + a small test kernel. Every
reference value is computed with mpmath at 120 bits on the *decoded* input
(``hi + lo`` summed exactly), never on the original float64, so the host
encoding error cannot pollute the measurement. Errors are reported in units of
``u^2 = 2^-48``; for sin/cos the metric is ``min(relative, absolute)`` so that
the (unavoidable) relative blow-up right at a zero of the function is judged by
its absolute error instead.
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

from optiland.backend.torch_backend.metal import compile as metal_compile  # noqa: E402

mp = mpmath.mp
mp.prec = 120
U2 = mpmath.mpf(2) ** -48

# Accuracy targets (units of u^2 = 2^-48), from the design context.
TOL_SINCOS = 20.0  # min(rel, abs)
TOL_TAN = 50.0  # rel, away from poles
TOL_INVERSE = 30.0  # atan/atan2/asin/acos (acos near 1: absolute)
TOL_IDENTITY = 40.0  # |sin^2 + cos^2 - 1|

TEST_KERNELS = r"""
kernel void trig_fwd(device const float* xh [[buffer(0)]],
                     device const float* xl [[buffer(1)]],
                     device float* sh [[buffer(2)]], device float* sl [[buffer(3)]],
                     device float* ch [[buffer(4)]], device float* cl [[buffer(5)]],
                     device float* th [[buffer(6)]], device float* tl [[buffer(7)]],
                     uint i [[thread_position_in_grid]]) {
    df64 x = df::make(xh[i], xl[i]);
    df64 s, c;
    df::sincos(x, s, c);
    df64 t = df::tan(x);
    sh[i] = s.hi; sl[i] = s.lo;
    ch[i] = c.hi; cl[i] = c.lo;
    th[i] = t.hi; tl[i] = t.lo;
}
kernel void trig_single(device const float* xh [[buffer(0)]],
                        device const float* xl [[buffer(1)]],
                        device float* sh [[buffer(2)]], device float* sl [[buffer(3)]],
                        device float* ch [[buffer(4)]], device float* cl [[buffer(5)]],
                        uint i [[thread_position_in_grid]]) {
    df64 x = df::make(xh[i], xl[i]);
    df64 s = df::sin(x);
    df64 c = df::cos(x);
    sh[i] = s.hi; sl[i] = s.lo;
    ch[i] = c.hi; cl[i] = c.lo;
}
kernel void trig_inv(device const float* xh [[buffer(0)]],
                     device const float* xl [[buffer(1)]],
                     device float* ah [[buffer(2)]], device float* al [[buffer(3)]],
                     device float* bh [[buffer(4)]], device float* bl [[buffer(5)]],
                     device float* dh [[buffer(6)]], device float* dl [[buffer(7)]],
                     uint i [[thread_position_in_grid]]) {
    df64 x = df::make(xh[i], xl[i]);
    df64 a = df::asin(x);
    df64 b = df::acos(x);
    df64 d = df::atan(x);
    ah[i] = a.hi; al[i] = a.lo;
    bh[i] = b.hi; bl[i] = b.lo;
    dh[i] = d.hi; dl[i] = d.lo;
}
kernel void trig_atan2(device const float* yh [[buffer(0)]],
                       device const float* yl [[buffer(1)]],
                       device const float* xh [[buffer(2)]],
                       device const float* xl [[buffer(3)]],
                       device float* oh [[buffer(4)]], device float* ol [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    df64 r = df::atan2(df::make(yh[i], yl[i]), df::make(xh[i], xl[i]));
    oh[i] = r.hi; ol[i] = r.lo;
}
kernel void trig_reduce(device const float* xh [[buffer(0)]],
                        device const float* xl [[buffer(1)]],
                        device float* rh [[buffer(2)]], device float* rl [[buffer(3)]],
                        device int* q [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
    int qq;
    df64 r = df::reduce_pio2(df::make(xh[i], xl[i]), qq);
    rh[i] = r.hi; rl[i] = r.lo; q[i] = qq;
}
"""


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lib():
    source = metal_compile.kernel_source(
        "df64_core.h", "df64_constants.h", "df64_math_trig.h"
    )
    return metal_compile.compile_library(source + TEST_KERNELS)


def encode(values) -> tuple[np.ndarray, np.ndarray]:
    """Split float64 values into normalized (hi, lo) float32 pairs.

    ``lo`` is forced to 0 where ``hi`` is not finite (inf - inf would be NaN).
    """
    x = np.asarray(values, dtype=np.float64)
    hi = x.astype(np.float32)
    with np.errstate(invalid="ignore"):
        lo = (x - hi.astype(np.float64)).astype(np.float32)
    lo[~np.isfinite(hi)] = 0.0
    return hi, lo


def encode_mp(values) -> tuple[np.ndarray, np.ndarray]:
    """Encode mpmath values as (hi, lo) pairs (exact to ~2^-49 relative)."""
    hi = np.array([np.float32(float(v)) for v in values], dtype=np.float32)
    lo = np.array(
        [
            np.float32(float(v - mpmath.mpf(float(h))))
            for v, h in zip(values, hi, strict=True)
        ],
        dtype=np.float32,
    )
    return hi, lo


def decode_mp(hi: np.ndarray, lo: np.ndarray) -> list:
    """Exact decoded values hi + lo as mpf."""
    return [
        mpmath.mpf(float(h)) + mpmath.mpf(float(v)) for h, v in zip(hi, lo, strict=True)
    ]


def _mps(a: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(a)).to("mps")


def _outs(n: int, k: int):
    return [torch.empty(n, dtype=torch.float32, device="mps") for _ in range(k)]


def run_fwd(lib, hi, lo):
    n = hi.size
    outs = _outs(n, 6)
    lib.trig_fwd(_mps(hi), _mps(lo), *outs, threads=[n, 1, 1])
    return [o.cpu().numpy() for o in outs]


def run_single(lib, hi, lo):
    n = hi.size
    outs = _outs(n, 4)
    lib.trig_single(_mps(hi), _mps(lo), *outs, threads=[n, 1, 1])
    return [o.cpu().numpy() for o in outs]


def run_inv(lib, hi, lo):
    n = hi.size
    outs = _outs(n, 6)
    lib.trig_inv(_mps(hi), _mps(lo), *outs, threads=[n, 1, 1])
    return [o.cpu().numpy() for o in outs]


def run_atan2(lib, yh, yl, xh, xl):
    n = yh.size
    outs = _outs(n, 2)
    lib.trig_atan2(_mps(yh), _mps(yl), _mps(xh), _mps(xl), *outs, threads=[n, 1, 1])
    return [o.cpu().numpy() for o in outs]


def errors_u2(rh, rl, refs, metric: str) -> np.ndarray:
    """Per-element error in units of u^2 (``metric`` in {"rel", "abs", "min"})."""
    out = np.empty(len(refs))
    for i, ref in enumerate(refs):
        got = mpmath.mpf(float(rh[i])) + mpmath.mpf(float(rl[i]))
        if mpmath.isnan(ref) or mpmath.isnan(got) or mpmath.isinf(ref):
            out[i] = 0.0 if (mpmath.isnan(ref) and mpmath.isnan(got)) else math.inf
            if mpmath.isinf(ref):
                out[i] = 0.0 if got == ref else math.inf
            continue
        d = abs(got - ref) / U2
        if metric == "abs":
            out[i] = float(d)
        else:
            rel = d / abs(ref) if ref != 0 else mpmath.inf
            if ref == 0 and d == 0:
                rel = mpmath.mpf(0)
            out[i] = float(min(rel, d)) if metric == "min" else float(rel)
    return out


def _report(name: str, err: np.ndarray, tol: float) -> None:
    worst = int(np.argmax(err))
    assert err.max() <= tol, (
        f"{name}: max error {err.max():.3f} u^2 > {tol} (at index {worst}); "
        f"median {np.median(err):.3f}"
    )


def identity_defect_u2(sh, sl, ch, cl) -> np.ndarray:
    """|sin^2 + cos^2 - 1| in units of u^2, evaluated exactly in mpmath."""
    out = np.empty(len(sh))
    for i in range(len(sh)):
        s = mpmath.mpf(float(sh[i])) + mpmath.mpf(float(sl[i]))
        c = mpmath.mpf(float(ch[i])) + mpmath.mpf(float(cl[i]))
        out[i] = float(abs(s * s + c * c - 1) / U2)
    return out


def _sincos_refs(xd):
    s = [mp.sin(v) for v in xd]
    c = [mp.cos(v) for v in xd]
    return s, c


# ---------------------------------------------------------------------------
# sin / cos / tan
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bound", [10.0, 1.0e4])
def test_sin_cos_tan_uniform(lib, bound):
    rng = np.random.default_rng(int(bound))
    n = 8000
    hi, lo = encode(rng.uniform(-bound, bound, n))
    xd = decode_mp(hi, lo)
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi, lo)
    s_ref, c_ref = _sincos_refs(xd)
    _report(f"sin |x|<={bound}", errors_u2(sh, sl, s_ref, "min"), TOL_SINCOS)
    _report(f"cos |x|<={bound}", errors_u2(ch, cl, c_ref, "min"), TOL_SINCOS)
    t_ref = [s / c for s, c in zip(s_ref, c_ref, strict=True)]
    _report(f"tan |x|<={bound}", errors_u2(th, tl, t_ref, "rel"), TOL_TAN)
    _report(
        f"sin^2+cos^2 |x|<={bound}", identity_defect_u2(sh, sl, ch, cl), TOL_IDENTITY
    )


def test_sin_cos_near_multiples_of_pio2(lib):
    """x = k*pi/2 + delta, delta in 1e-9..1e-3, encoded as exact pairs."""
    rng = np.random.default_rng(7)
    n = 8000
    ks = rng.integers(-3000, 3001, n)
    deltas = 10.0 ** rng.uniform(-9, -3, n) * rng.choice([-1.0, 1.0], n)
    xs = [k * mp.pi / 2 + mpmath.mpf(float(d)) for k, d in zip(ks, deltas, strict=True)]
    hi, lo = encode_mp(xs)
    xd = decode_mp(hi, lo)
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi, lo)
    s_ref, c_ref = _sincos_refs(xd)
    _report("sin near k pi/2", errors_u2(sh, sl, s_ref, "min"), TOL_SINCOS)
    _report("cos near k pi/2", errors_u2(ch, cl, c_ref, "min"), TOL_SINCOS)
    # tan is either ~delta or ~1/delta here; both are away from the pole in the
    # sense that the reduced argument is known to full relative accuracy.
    t_ref = [s / c for s, c in zip(s_ref, c_ref, strict=True)]
    _report("tan near k pi/2", errors_u2(th, tl, t_ref, "rel"), TOL_TAN)
    _report("sin^2+cos^2 near k pi/2", identity_defect_u2(sh, sl, ch, cl), TOL_IDENTITY)


def test_sin_cos_large_arguments(lib):
    """Up to the df64 reduction limit (1e7): absolute accuracy is retained."""
    rng = np.random.default_rng(11)
    n = 6000
    hi, lo = encode(rng.uniform(-1.0e7, 1.0e7, n))
    xd = decode_mp(hi, lo)
    sh, sl, ch, cl, _, _ = run_fwd(lib, hi, lo)
    s_ref, c_ref = _sincos_refs(xd)
    _report("sin |x|<=1e7", errors_u2(sh, sl, s_ref, "min"), TOL_SINCOS)
    _report("cos |x|<=1e7", errors_u2(ch, cl, c_ref, "min"), TOL_SINCOS)


def test_sin_cos_beyond_range_fall_back_to_float32(lib):
    """|x.hi| > 1e7: float32 sin/cos of the whole pair (hi and lo), lo == 0, no garbage.

    The lo word carries up to ulp(hi)/2 of phase (0.5 rad just above 1e7, 32 rad
    at 1e9, 2.6e5 rad at 7.7e12); the fallback folds it in with the angle-addition
    formulas, so the reference is sin/cos of the decoded value hi + lo, not of hi.
    """
    xs = np.array([2.0e7, -3.0e7, 1.5e9, -7.7e12, 3.0e38, 2.0e7 + 0.9, 1e9 + 30.0])
    hi, lo = encode(xs)
    # one pair with the largest lo the format allows just above the cutoff
    hi = np.append(hi, np.float32(1.0000001e7)).astype(np.float32)
    lo = np.append(lo, np.float32(0.3)).astype(np.float32)
    assert lo[3] != 0.0 and lo[5] != 0.0 and lo[6] != 0.0  # the lo word matters here
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi, lo)
    assert np.all(sl == 0.0) and np.all(cl == 0.0)
    assert np.all(np.isfinite(sh)) and np.all(np.isfinite(ch))
    assert np.all(np.abs(sh) <= 1.0) and np.all(np.abs(ch) <= 1.0)
    for i, (h, low) in enumerate(zip(hi, lo, strict=True)):
        xd = mpmath.mpf(float(h)) + mpmath.mpf(float(low))
        ref_s = float(mp.sin(xd))
        ref_c = float(mp.cos(xd))
        assert abs(sh[i] - ref_s) < 2e-6, (h, low, sh[i], ref_s)
        assert abs(ch[i] - ref_c) < 2e-6, (h, low, ch[i], ref_c)
        ref_t = float(mp.tan(xd))
        assert abs((th[i] + tl[i]) - ref_t) < 2e-6 * max(1.0, abs(ref_t) ** 2)
    # sin/cos via the single-function entry points agree with sincos exactly.
    sh2, sl2, ch2, cl2 = run_single(lib, hi, lo)
    assert np.array_equal(sh, sh2) and np.array_equal(ch, ch2)


def test_sin_cos_tan_symmetry_and_identity_gpu_batch(lib):
    """Exact odd/even symmetry and sin^2 + cos^2 = 1 on a large GPU batch."""
    rng = np.random.default_rng(3)
    n = 400_000
    x = np.concatenate(
        [
            rng.uniform(-1.0e4, 1.0e4, n // 2),
            rng.uniform(-10.0, 10.0, n // 4),
            10.0 ** rng.uniform(-20, 4, n // 4) * rng.choice([-1.0, 1.0], n // 4),
        ]
    )
    hi, lo = encode(x)
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi, lo)
    nsh, nsl, nch, ncl, nth, ntl = run_fwd(lib, -hi, -lo)
    # Bit-exact odd symmetry for sin and tan, even symmetry for cos.
    assert np.array_equal(nsh, -sh) and np.array_equal(nsl, -sl)
    assert np.array_equal(nth, -th) and np.array_equal(ntl, -tl)
    assert np.array_equal(nch, ch) and np.array_equal(ncl, cl)
    # sin/cos entry points agree bit-for-bit with sincos.
    sh2, sl2, ch2, cl2 = run_single(lib, hi, lo)
    assert np.array_equal(sh, sh2) and np.array_equal(sl, sl2)
    assert np.array_equal(ch, ch2) and np.array_equal(cl, cl2)
    # Pythagorean identity in float64 (decoding error 2^-53 is negligible here).
    s = sh.astype(np.float64) + sl.astype(np.float64)
    c = ch.astype(np.float64) + cl.astype(np.float64)
    defect = np.abs(s * s + c * c - 1.0) / float(U2)
    assert defect.max() <= TOL_IDENTITY, defect.max()
    # tan == sin / cos as computed on the host to df64 accuracy.
    t = th.astype(np.float64) + tl.astype(np.float64)
    rel = np.abs(t - s / c) / np.abs(s / c)
    assert rel.max() <= 10 * float(U2), rel.max()


def test_sin_cos_tan_special_values(lib):
    xs = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-20, -1e-25, 3e-22, 1e-30])
    hi, lo = encode(xs)
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi, lo)
    # +-0 -> +-0 (signbit preserved), cos(+-0) = 1 exactly.
    for i in (0, 1):
        assert sh[i] == 0.0 and np.signbit(sh[i]) == np.signbit(xs[i]) and sl[i] == 0.0
        assert th[i] == 0.0 and np.signbit(th[i]) == np.signbit(xs[i]) and tl[i] == 0.0
        assert ch[i] == 1.0 and cl[i] == 0.0
    # inf / NaN -> NaN
    for i in (2, 3, 4):
        assert np.isnan(sh[i]) and np.isnan(ch[i]) and np.isnan(th[i])
    # tiny arguments (x^2/2 below the float32 range): sin x = tan x = x exactly
    # and cos x = 1 exactly.
    for i in (5, 6, 7):
        assert sh[i] == hi[i] and sl[i] == lo[i]
        assert th[i] == hi[i] and tl[i] == lo[i]
        assert ch[i] == 1.0 and cl[i] == 0.0
    # 1e-30: its lo word is a float32 denormal that the GPU flushes (documented
    # limit); only hi is guaranteed.
    assert sh[8] == hi[8] and th[8] == hi[8] and ch[8] == 1.0
    # Moderately tiny argument: cos keeps the -x^2/2 term in the lo word.
    hi, lo = encode(np.array([1e-15]))
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi, lo)
    assert sh[0] == hi[0] and sl[0] == lo[0] and th[0] == hi[0]
    assert ch[0] == 1.0 and abs(float(cl[0]) + 0.5e-30) <= 1e-38


# ---------------------------------------------------------------------------
# asin / acos / atan / atan2
# ---------------------------------------------------------------------------
def test_asin_acos(lib):
    rng = np.random.default_rng(5)
    n = 8000
    a = rng.uniform(-1.0, 1.0, n)
    m = n // 8
    a[:m] = 1.0 - 10.0 ** rng.uniform(-15, -1, m)
    a[m : 2 * m] = -1.0 + 10.0 ** rng.uniform(-15, -1, m)
    a[2 * m : 3 * m] = 10.0 ** rng.uniform(-12, -1, m) * rng.choice([-1.0, 1.0], m)
    hi, lo = encode(a)
    ad = decode_mp(hi, lo)
    ah, al, bh, bl, _, _ = run_inv(lib, hi, lo)
    _report("asin", errors_u2(ah, al, [mp.asin(v) for v in ad], "rel"), TOL_INVERSE)
    # acos near 1 is judged by absolute error (the result itself is ~sqrt(2(1-a))).
    _report("acos", errors_u2(bh, bl, [mp.acos(v) for v in ad], "min"), TOL_INVERSE)
    # Away from 1 the relative bound holds too.
    far = np.abs(a) < 0.9
    _report(
        "acos (|a|<0.9, rel)",
        errors_u2(
            bh[far],
            bl[far],
            [mp.acos(v) for v, f in zip(ad, far, strict=True) if f],
            "rel",
        ),
        TOL_INVERSE,
    )


def test_atan_log_uniform(lib):
    rng = np.random.default_rng(9)
    n = 8000
    x = 10.0 ** rng.uniform(-10, 10, n) * rng.choice([-1.0, 1.0], n)
    hi, lo = encode(x)
    xd = decode_mp(hi, lo)
    _, _, _, _, dh, dl = run_inv(lib, hi, lo)
    _report("atan", errors_u2(dh, dl, [mp.atan(v) for v in xd], "rel"), TOL_INVERSE)
    # Exact odd symmetry.
    _, _, _, _, ndh, ndl = run_inv(lib, -hi, -lo)
    assert np.array_equal(ndh, -dh) and np.array_equal(ndl, -dl)


def test_atan2_quadrants_and_axes(lib):
    rng = np.random.default_rng(13)
    n = 10000
    mag_y = 10.0 ** rng.uniform(-6, 6, n)
    mag_x = 10.0 ** rng.uniform(-6, 6, n)
    y = mag_y * rng.choice([-1.0, 1.0], n) * rng.uniform(0.1, 1.0, n)
    x = mag_x * rng.choice([-1.0, 1.0], n) * rng.uniform(0.1, 1.0, n)
    # Near the axes and the diagonals (where the seed quadrant / cancellation matter).
    m = n // 10
    y[:m] = x[:m] * 10.0 ** rng.uniform(-12, -3, m) * rng.choice([-1.0, 1.0], m)
    x[m : 2 * m] = y[m : 2 * m] * 10.0 ** rng.uniform(-12, -3, m)
    x[2 * m : 3 * m] = -np.abs(y[2 * m : 3 * m]) * (1 + 1e-9 * rng.standard_normal(m))
    x[3 * m : 4 * m] = y[3 * m : 4 * m] * (1 + 1e-6 * rng.standard_normal(m))
    yh, yl = encode(y)
    xh, xl = encode(x)
    yd = decode_mp(yh, yl)
    xd = decode_mp(xh, xl)
    oh, ol = run_atan2(lib, yh, yl, xh, xl)
    refs = [mp.atan2(a, b) for a, b in zip(yd, xd, strict=True)]
    _report("atan2", errors_u2(oh, ol, refs, "rel"), TOL_INVERSE)
    # Exact odd symmetry in y.
    noh, nol = run_atan2(lib, -yh, -yl, xh, xl)
    assert np.array_equal(noh, -oh) and np.array_equal(nol, -ol)
    # Scaling invariance: atan2(2^20 y, 2^20 x) and atan2(2^-20 y, 2^-20 x)
    # are bit-identical (the common power-of-two prescaling is exact).
    for p in (2.0**20, 2.0**-20):
        ph, pl = run_atan2(lib, yh * p, yl * p, xh * p, xl * p)
        assert np.array_equal(ph, oh) and np.array_equal(pl, ol)


def test_atan2_special_values(lib):
    inf = np.inf
    pairs = [
        (0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0),
        (0.0, 1.0), (-0.0, 1.0), (0.0, -1.0), (-0.0, -1.0),
        (1.0, 0.0), (-1.0, 0.0), (1.0, -0.0), (-1.0, -0.0),
        (inf, inf), (-inf, inf), (inf, -inf), (-inf, -inf),
        (inf, 1.0), (-inf, 1.0), (inf, -1.0), (-inf, -1.0),
        (1.0, inf), (-1.0, inf), (1.0, -inf), (-1.0, -inf),
        (np.nan, 1.0), (1.0, np.nan), (np.nan, np.nan), (inf, np.nan),
        (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
    ]  # fmt: skip
    yh, yl = encode([p[0] for p in pairs])
    xh, xl = encode([p[1] for p in pairs])
    oh, ol = run_atan2(lib, yh, yl, xh, xl)
    for i, (y, x) in enumerate(pairs):
        ref = np.arctan2(y, x)  # numpy follows IEEE/C99, as torch does
        got = float(oh[i]) + float(ol[i])
        if np.isnan(ref):
            assert np.isnan(oh[i]), (y, x, oh[i])
            continue
        assert np.signbit(oh[i]) == np.signbit(ref), (y, x, got, ref)
        if ref == 0.0:
            assert oh[i] == 0.0 and ol[i] == 0.0, (y, x, got)
        else:
            assert abs(got - ref) <= 4 * float(U2) * abs(ref), (y, x, got, ref)


def test_inverse_special_values(lib):
    xs = np.array(
        [0.0, -0.0, 1.0, -1.0, 1.0000001, -1.0000001, np.inf, -np.inf, np.nan]
    )
    hi, lo = encode(xs)
    ah, al, bh, bl, dh, dl = run_inv(lib, hi, lo)
    pi = math.pi
    # asin(+-0) = +-0, acos(+-0) = pi/2, atan(+-0) = +-0.
    for i in (0, 1):
        assert ah[i] == 0.0 and np.signbit(ah[i]) == np.signbit(xs[i]) and al[i] == 0.0
        assert dh[i] == 0.0 and np.signbit(dh[i]) == np.signbit(xs[i]) and dl[i] == 0.0
        assert abs(float(bh[i]) + float(bl[i]) - pi / 2) <= 2 * float(U2)
    # asin(+-1) = +-pi/2, acos(1) = +0 exactly, acos(-1) = pi.
    assert abs(float(ah[2]) + float(al[2]) - pi / 2) <= 2 * float(U2)
    assert abs(float(ah[3]) + float(al[3]) + pi / 2) <= 2 * float(U2)
    assert bh[2] == 0.0 and not np.signbit(bh[2]) and bl[2] == 0.0
    assert abs(float(bh[3]) + float(bl[3]) - pi) <= 2 * float(U2) * pi
    # |a| > 1 -> NaN (never 0); atan(+-inf) = +-pi/2; NaN passthrough.
    for i in (4, 5, 6, 7):
        assert np.isnan(ah[i]) and np.isnan(bh[i])
    assert abs(float(dh[6]) + float(dl[6]) - pi / 2) <= 2 * float(U2)
    assert abs(float(dh[7]) + float(dl[7]) + pi / 2) <= 2 * float(U2)
    assert np.isnan(ah[8]) and np.isnan(bh[8]) and np.isnan(dh[8])
    # asin(1e-7)/atan(1e-7): tiny arguments stay accurate (no cancellation).
    hi, lo = encode(np.array([1e-7, -3e-9]))
    ah, al, bh, bl, dh, dl = run_inv(lib, hi, lo)
    for i, v in enumerate(decode_mp(hi, lo)):
        e_asin = errors_u2(ah[i : i + 1], al[i : i + 1], [mp.asin(v)], "rel")[0]
        e_atan = errors_u2(dh[i : i + 1], dl[i : i + 1], [mp.atan(v)], "rel")[0]
        assert e_asin <= TOL_INVERSE and e_atan <= TOL_INVERSE, (v, e_asin, e_atan)


# ---------------------------------------------------------------------------
# Regression tests for the fix round 1 findings (NOTES/04-kernel-status.md 8).
# ---------------------------------------------------------------------------
def test_tan_sin_cos_relative_accuracy_at_zeros(lib):
    """x = float64(k pi) sits ~1e-16 |x| from a zero of tan; the old reduction
    (df64 product j * PIO2_REST, pi/2 to 2^-88.8) left r with ~2^-87 |x| absolute
    error and tan with up to ~7000 u^2 relative error there. The four-word
    accumulator and the third pi/2 word keep tan <= 50 u^2 (measured <= 1.4) and
    sin/cos <= 20 u^2 *relative* (not just absolute) at every such point.
    """
    ks = [1, 2, 3, 100, 1000, 2000, 3183, 6366, -1, -3183]
    x = np.array(
        [k * math.pi for k in ks]
        + [k * math.pi / 2 for k in ks]
        + [3183 * math.pi + 1e-10, 3141.5926535897961, 314.15926535897933]
    )
    hi, lo = encode(x)
    xd = decode_mp(hi, lo)
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi, lo)
    _report("tan at k pi", errors_u2(th, tl, [mp.tan(v) for v in xd], "rel"), TOL_TAN)
    # sin and cos: relative error at the points where they are tiny
    s_ref, c_ref = _sincos_refs(xd)
    _report("sin rel", errors_u2(sh, sl, s_ref, "rel"), TOL_SINCOS)
    _report("cos rel", errors_u2(ch, cl, c_ref, "rel"), TOL_SINCOS)


def test_reduction_band_near_multiples_of_pio2(lib):
    """k pi/2 (1 + f), |k| <= 6366, |f| in [1e-17.5, 1e-11]: tan and the reduced
    argument keep relative accuracy far below the smallest distance a df64
    argument can have from a multiple of pi/2 (2^-63 |x| for |x| <= 1e4)."""
    gen = np.random.default_rng(3)
    n = 3000
    k = gen.integers(1, 6366, n) * gen.choice([-1, 1], n)
    f = 10.0 ** gen.uniform(-17.5, -11, n) * gen.choice([-1, 1], n)
    hi, lo = encode((k * math.pi / 2) * (1 + f))
    xd = decode_mp(hi, lo)
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi, lo)
    _report("tan band", errors_u2(th, tl, [mp.tan(v) for v in xd], "rel"), TOL_TAN)
    s_ref, c_ref = _sincos_refs(xd)
    _report("sin band", errors_u2(sh, sl, s_ref, "min"), TOL_SINCOS)
    _report("cos band", errors_u2(ch, cl, c_ref, "min"), TOL_SINCOS)
    # the sin or cos that is tiny there must also be right *relatively*
    tiny_rel = np.array(
        [
            float(
                min(
                    errors_u2(sh[i : i + 1], sl[i : i + 1], [s_ref[i]], "rel")[0],
                    errors_u2(ch[i : i + 1], cl[i : i + 1], [c_ref[i]], "rel")[0],
                )
            )
            for i in range(0, n, 10)
        ]
    )
    assert tiny_rel.max() <= TOL_SINCOS, tiny_rel.max()


def test_tiny_arguments_keep_denormal_lo_word(lib):
    """asin/atan/sin of |a| < 2^-26 return a unchanged, so a lo word below
    FLT_MIN (which any arithmetic flushes) survives: previously atan of
    (1e-29, 1.5e-38) returned (1e-29, 0), a 4e5 u^2 relative error."""
    hi = np.array(
        [1.0000282e-29, -1.011021e-27, 1e-29, 2.0**-27, -3.761972404831528e-28, 1e-9],
        dtype=np.float32,
    )
    lo = np.array([1.5e-38, 1.5e-38, 0.0, 2.0**-52, 9.6e-39, 3e-17], dtype=np.float32)
    ah, al, bh, bl, dh, dl = run_inv(lib, hi, lo)
    assert np.array_equal(ah, hi) and np.array_equal(al, lo)  # asin
    assert np.array_equal(dh, hi) and np.array_equal(dl, lo)  # atan
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi, lo)
    assert np.array_equal(sh, hi) and np.array_equal(sl, lo)
    assert np.array_equal(th, hi) and np.array_equal(tl, lo)
    assert np.all(ch == 1.0)
    # cos keeps -x^2/2 in the lo word while it is a normal float32
    assert abs(float(cl[5]) + 0.5e-18) <= 1e-25 and cl[3] == -(2.0**-55)
    assert np.all(cl[[0, 1, 2, 4]] == 0.0)  # x^2/2 below FLT_MIN flushes
    # acos of a tiny argument is pi/2 - a to df64 accuracy
    ref = [mp.acos(v) for v in decode_mp(hi, lo)]
    _report("acos tiny", errors_u2(bh, bl, ref, "rel"), TOL_INVERSE)


# ---------------------------------------------------------------------------
# Fix round 2 (NOTES/04 section 9)
# ---------------------------------------------------------------------------
def _f32bits(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint32).view(np.float32)[0])


def test_closest_df64_arguments_to_k_pio2_keep_relative_accuracy(lib):
    """The df64 pairs nearest to k pi/2 (exhaustive search up to |x| = 1e7: the
    closest is 2^-75 |x| away, not the ~2^-49 |x| the header used to claim)
    had tan / the tiny sin or cos wrong by 63-519 u^2 relative: the third pi/2
    word was applied with one float rounding and the constant stopped at
    2^-114.8. With two_prod and a fourth word the reduction floor is ~2^-121 |x|
    (measured), so tan and sin/cos stay within a few u^2 at every one of them.
    """
    pairs = [
        (_f32bits(0x4AA562AE), _f32bits(0xB32411DE)),  # k = 3450066, zero of tan
        (_f32bits(0x4A2562AE), _f32bits(0xB2A411DE)),  # k = 1725033, pole
        (_f32bits(0x4AE9F667), _f32bits(0x3CF9413A)),  # k = 4880635, pole, 2^-75 |x|
        (_f32bits(0x4AF253AB), _f32bits(0xBADE54E3)),  # k = 5055121
        (_f32bits(0x4AE695DD), _f32bits(0x347E092A)),  # k = 4810186, zero
        (_f32bits(0x4AF3A065), _f32bits(0xBA68BFA3)),  # k = 5082234, zero
        (_f32bits(0x4AFF15E8), _f32bits(0xBD828D01)),  # k = 5321278, zero
    ]
    pairs += [(-h, -low) for h, low in pairs]
    hi = np.array([p[0] for p in pairs], dtype=np.float32)
    lo = np.array([p[1] for p in pairs], dtype=np.float32)
    xd = decode_mp(hi, lo)
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi, lo)
    _report("tan nearest", errors_u2(th, tl, [mp.tan(v) for v in xd], "rel"), 5.0)
    s_ref, c_ref = _sincos_refs(xd)
    _report("sin nearest rel", errors_u2(sh, sl, s_ref, "rel"), 5.0)
    _report("cos nearest rel", errors_u2(ch, cl, c_ref, "rel"), 5.0)
    # the reduced argument itself: absolute error <= 2^-118 |x|
    n = hi.size
    outs = _outs(n, 2)
    q = torch.empty(n, dtype=torch.int32, device="mps")
    lib.trig_reduce(_mps(hi), _mps(lo), *outs, q, threads=[n, 1, 1])
    rh, rl = (o.cpu().numpy() for o in outs)
    for i, x in enumerate(xd):
        k = mpmath.nint(x * 2 / mp.pi)
        r_ref = x - k * mp.pi / 2
        got = mpmath.mpf(float(rh[i])) + mpmath.mpf(float(rl[i]))
        assert abs(got - r_ref) <= abs(x) * mpmath.mpf(2) ** -118, (pairs[i], i)


def test_reduction_keeps_r_within_pi_over_4(lib):
    """j = rint(jd.hi) alone is off by one for |x| > ~1e4 (jd.hi carries 24 bits),
    which left |r| up to 3 pi/8 and made the k-clamp of the table step
    load-bearing; j is now corrected from the whole pair."""
    gen = np.random.default_rng(7)
    x = np.concatenate(
        [
            gen.uniform(1e4, 6.6e6, 20000),
            gen.uniform(6.6e6, 1e7, 20000),
            -gen.uniform(1e4, 1e7, 10000),
        ]
    )
    hi, lo = encode(x)
    n = hi.size
    outs = _outs(n, 2)
    q = torch.empty(n, dtype=torch.int32, device="mps")
    lib.trig_reduce(_mps(hi), _mps(lo), *outs, q, threads=[n, 1, 1])
    rh, rl = (o.cpu().numpy() for o in outs)
    r = rh.astype(np.float64) + rl.astype(np.float64)
    assert np.abs(r).max() <= math.pi / 4 * (1 + 1e-6), np.abs(r).max()
    # r and q are consistent with x: x = (4 m + q) pi/2 + r
    xd = hi.astype(np.float64) + lo.astype(np.float64)
    j = np.rint((xd - r) / (math.pi / 2))
    assert np.array_equal((j.astype(np.int64) & 3).astype(np.int32), q.cpu().numpy())
    # accuracy is unchanged in the band
    sub = slice(0, 400)
    xm = decode_mp(hi[sub], lo[sub])
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi[sub], lo[sub])
    _report("tan 1e4..1e7", errors_u2(th, tl, [mp.tan(v) for v in xm], "rel"), TOL_TAN)


def test_fallback_sin_cos_are_clamped_to_unit_interval(lib):
    """The float32 fallback (|x.hi| > 1e7) could return |sin| or |cos| = 1 + 2^-23,
    which made asin/acos of the result NaN."""
    hi = np.array([_f32bits(0x7CD174FD), _f32bits(0x6627EB65)], dtype=np.float32)
    lo = np.array([_f32bits(0x705CD7E8), _f32bits(0x594BB337)], dtype=np.float32)
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi, lo)
    assert np.all(np.abs(sh) <= 1.0) and np.all(np.abs(ch) <= 1.0)
    assert sl.tolist() == [0.0, 0.0] and cl.tolist() == [0.0, 0.0]
    # asin/acos of the fallback values are finite
    ah, al, bh, bl, dh, dl = run_inv(
        lib, np.concatenate([sh, ch]), np.zeros(4, np.float32)
    )
    assert np.all(np.isfinite(ah)) and np.all(np.isfinite(bh))
    gen = np.random.default_rng(12)
    x = 10.0 ** gen.uniform(7.1, 38.4, 200000) * gen.choice([-1.0, 1.0], 200000)
    hi, lo = encode(x)
    sh, sl, ch, cl, th, tl = run_fwd(lib, hi, lo)
    assert np.all(np.abs(sh) <= 1.0) and np.all(np.abs(ch) <= 1.0)


def test_atan2_tiny_quotient_keeps_x_lo_and_avoids_flush(lib):
    """(a) atan2(1e-26, (1, 2^-40)) returned exactly y: the x.lo correction was a
    flushed denormal inside the Newton residual (256-4096 u^2). (b) For normal
    operands with |y/x| ~ 1e-29..1e-25 the common power-of-two prescaling put
    every residual into the flushed range (up to 3 FLT_MIN absolute). Tiny
    quotients now go through y/x with each operand scaled into [0.5, 1) and one
    exact rescale; what remains is the FLT_MIN storage floor of the result.
    """
    y = [
        (_f32bits(0x14461206), _f32bits(0x0715D962)),  # 1e-26
        (_f32bits(0x129E74D2), _f32bits(0x8610DC3F)),  # 1e-27
        (_f32bits(0x9406A349), _f32bits(0x073A8616)),  # -6.8e-27
        (_f32bits(0x40989DB0), _f32bits(0xB3EBF5AA)),  # 4.769249 - 1.1e-7
        (_f32bits(0x41101FAB), _f32bits(0xB2B4FD18)),  # 9.007731 - 2.1e-8
        (_f32bits(0x40BE7FBC), _f32bits(0x3404452A)),  # 5.9530926 + 1.2e-7
    ]
    x = [
        (1.0, 2.0**-40),
        (1.0, 2.0**-36),
        (_f32bits(0x4218D8A0), _f32bits(0xB146D020)),  # 38.2115478515625 - 2.9e-9
        (_f32bits(0x6E318EEE), _f32bits(0x5F11B170)),  # 1.373791e28 + 1.05e19
        (_f32bits(0x6F8AFD06), _f32bits(0x63720E54)),  # 8.602964e28 + 4.5e21
        (_f32bits(0x6D817615), _f32bits(0x6129FCEA)),  # 5.00829e27 + 2e20
    ]
    yh = np.array([p[0] for p in y], np.float32)
    yl = np.array([p[1] for p in y], np.float32)
    xh = np.array([p[0] for p in x], np.float32)
    xl = np.array([p[1] for p in x], np.float32)
    oh, ol = run_atan2(lib, yh, yl, xh, xl)
    refs = [
        mp.atan2(a, b)
        for a, b in zip(decode_mp(yh, yl), decode_mp(xh, xl), strict=True)
    ]
    _report("atan2 tiny quotient", errors_u2(oh, ol, refs, "rel"), TOL_INVERSE)
    # random: |y| in [1, 10], |x| in [1e25, 1e30]; bound = 30 u^2 relative or
    # 1 FLT_MIN absolute (the lo word of a result below ~1e-25 can be denormal)
    gen = np.random.default_rng(21)
    n = 3000
    yv = gen.uniform(1, 10, n) * gen.choice([-1.0, 1.0], n)
    xv = 10.0 ** gen.uniform(25, 30, n)
    yh, yl = encode(yv)
    xh, xl = encode(xv)
    oh, ol = run_atan2(lib, yh, yl, xh, xl)
    refs = [
        mp.atan2(a, b)
        for a, b in zip(decode_mp(yh, yl), decode_mp(xh, xl), strict=True)
    ]
    rel = errors_u2(oh, ol, refs, "rel")
    absolute = errors_u2(oh, ol, refs, "abs") * float(U2)
    ok = (rel <= TOL_INVERSE) | (absolute <= 2.0**-126)
    assert np.all(ok), (np.flatnonzero(~ok)[:5], rel[~ok][:5])
    assert np.mean(rel > TOL_INVERSE) < 0.01  # storage floor hits (~0.5%)
    # the same y with x.lo = 0 and x ~ 1: plain Newton path, unchanged accuracy
    xh2 = np.ones(n, np.float32)
    oh, ol = run_atan2(lib, yh, yl, xh2, np.zeros(n, np.float32))
    refs = [mp.atan(a) for a in decode_mp(yh, yl)]
    _report("atan2 y/1", errors_u2(oh, ol, refs, "rel"), TOL_INVERSE)


def test_asin_is_odd_bit_for_bit(lib):
    """asin(-a) != -asin(a) for ~8.5% of inputs (the (1 - a)(1 + a) factors swap
    roles under a -> -a and df::mul is not commutative bit-for-bit); asin is now
    evaluated on |a| with the sign applied afterwards."""
    gen = np.random.default_rng(4)
    a = gen.uniform(-1, 1, 200000)
    a = np.concatenate([a, [-0.4952501654624939 - 4.6502064421360956e-09]])
    hi, lo = encode(a)
    ah, al, bh, bl, dh, dl = run_inv(lib, hi, lo)
    nh, nl, _, _, mh, ml = run_inv(lib, -hi, -lo)
    assert np.array_equal(ah, -nh) and np.array_equal(al, -nl)
    assert np.array_equal(dh, -mh) and np.array_equal(dl, -ml)
