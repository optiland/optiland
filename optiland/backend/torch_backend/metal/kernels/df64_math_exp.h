// df64_math_exp.h — exponential and logarithm family for double-single ("df64")
// values on Apple GPUs (Metal Shading Language).
//
// Provides, in namespace df:: for df64 arguments:
//   exp, exp2, exp10, expm1, log, log1p, log2, log10, pow(df64, df64),
//   pow(df64, int), cbrt.
//
// AMALGAMATION CONVENTION: this header does NOT include df64_core.h or
// df64_constants.h. compile.kernel_source() concatenates df64_core.h,
// df64_constants.h and then the math headers, so everything from those two
// files (struct df64, namespace df, DF64_* constants) is already in scope.
//
// Algorithms (QD dd_real scheme adapted to float32 pairs, NOTES/02-design.md 2.4):
//   * exp: x = m ln2 + r with an exact three-part Cody-Waite ln2 (16-bit chunks so
//     that m * chunk is exact for |m| <= 128; the df64 product m * DF64_LN2 would
//     cost up to ~|x| u^2 of relative error at |x| ~ 88), then r/512, a Taylor
//     polynomial with DF64_INV_FACT, nine s = 2s + s^2 doublings, s + 1, ldexp.
//     The core accepts the argument as a float plus a df64 "tail" so that exp2,
//     exp10 and pow can feed it a triple-precision product b * log(a) without
//     first rounding that product to 48 bits (its rounding alone would cost
//     |b log a| u^2 of relative error).
//   * log: x = f 2^e with f in [sqrt(1/2), sqrt(2)), u = f - 1 exact,
//     log f = 2 atanh(u / (2 + u)) with the quotient corrected by its exact
//     residual and the series evaluated in df64; e ln2 added with the same
//     three-part ln2. The result is carried as df64 + one extra float ("ext")
//     inside this header (about 0.1 u^2) and rounded to df64 only at the API
//     boundary, which keeps log accurate near 1 (relative accuracy for
//     log1p of tiny arguments) and lets pow / log2 / log10 reuse the unrounded
//     value. This replaces the float seed + Newton scheme of the design note,
//     whose accuracy was limited by the exp it depends on.
//   * pow: IEEE/C special cases, npwr by binary exponentiation for integer
//     exponents |b| <= 8 (larger |n| would amplify rounding ~1.4 n u^2) on the
//     frexp mantissa with one exact ldexp at the end (so a^|n| never leaves the
//     normal range before the reciprocal / final scaling), otherwise
//     sign * exp(b * log|a|) via the extended log (~2 u^2).
//   * cbrt: exact prescaling by 2^(3k), precise::pow seed, two df64 Newton steps.
// Every float32 seed calls metal::precise:: explicitly. Domain errors return NaN,
// overflow +-inf, underflow 0 (results below FLT_MIN = 1.18e-38 are returned as
// +0 because this GPU flushes float32 denormals in arithmetic). The overflow /
// underflow window [ln FLT_MIN, ln FLT_MAX] applies to the reduced argument, i.e.
// to x for exp/expm1, x ln2 for exp2, x ln10 for exp10 and b log|a| for pow.
//
// Precision of small results (FTZ storage floor, see df64_core.h): a result
// below 2^-102 ~ 2e-31 has a flushed lo word (24-bit precision); between
// 2^-102 and 2^-78 ~ 3.3e-24 the lo word is kept only when it happens to be a
// normal float32, so the relative error of exp/exp2/exp10/expm1/pow/... there
// is up to FLT_MIN/|f(x)| with probability ~2^-102/|f(x)| (measured for exp:
// 46% of the inputs with results near 2^-101 exceed 20 u^2, 1% near 2^-96,
// 0.2% near 2^-93, none from 2^-91 up; worst 8e6 u^2 at 2^-101). The u^2
// accuracy claims below therefore hold for |f(x)| >= 2^-78; nothing better is
// representable (the returned pair is the correctly rounded df64 on the FTZ
// grid).
//
// Part of Optiland-Metal (MIT). Written fresh; the exp/expm1 reduction scheme
// follows QD (Hida, Li & Bailey, BSD).

#ifndef OPTILAND_DF64_MATH_EXP_H
#define OPTILAND_DF64_MATH_EXP_H

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// ---------------------------------------------------------------------------
// Local constants (candidates for tools/gen_df64_constants.py; values generated
// with mpmath at 200 bits, float32 literals are exact).
// ---------------------------------------------------------------------------
// ln2 = A + B + C: A and B carry 16 significant bits each, so m * A and m * B are
// exact float32 products for integer |m| <= 128; C is a df64 remainder.
// Total representation error of A + B + C vs ln2: 1.7e-26 relative.
static constant float DF64X_LN2_A = 0.69314575f;             // 45426 / 2^16
static constant float DF64X_LN2_B = 1.4286197e-06f;          // 49087 / 2^35
static constant df64 DF64X_LN2_C = {-1.290532e-11f, -2.2829880e-19f};
static constant float DF64X_1_LN2_F = 1.442695f;             // float(1 / ln2), for nint only
// Three-float (~72-bit) constants for triple-precision products: value = x + y + z.
static constant float3 DF64X_LN2_3 = float3(0.6931472f, -1.9046542e-09f, -8.783184e-17f);
static constant float3 DF64X_LN10_3 = float3(2.3025851f, -3.1975436e-08f, -1.105254e-15f);
static constant float3 DF64X_1_LN2_3 = float3(1.442695f, 1.925963e-08f, -4.2373395e-16f);
static constant float3 DF64X_1_LN10_3 = float3(0.4342945f, -1.010305e-08f, -1.00039104e-16f);
// 1 / (2k + 3) for k = 0..8: coefficients of atanh(s)/s - 1 = s^2/3 + s^4/5 + ...
static constant df64 DF64X_INV_ODD[9] = {
    {0.33333334f, -9.934108e-09f},   // 1/3
    {0.2f, -2.9802323e-09f},         // 1/5
    {0.14285715f, -6.386212e-09f},   // 1/7
    {0.11111111f, -8.278423e-10f},   // 1/9
    {0.09090909f, -2.7093021e-09f},  // 1/11
    {0.07692308f, -2.865608e-09f},   // 1/13
    {0.06666667f, -3.4769376e-09f},  // 1/15
    {0.05882353f, -2.1913472e-10f},  // 1/17
    {0.05263158f, -3.9213582e-10f},  // 1/19
};
// Range guards on the leading float of the reduced argument, placed just
// OUTSIDE [ln FLT_MIN, ln FLT_MAX] = [-87.336545, 88.722839] so that every
// representable result is computed (exp(88.7228) = 0.99996 FLT_MAX, exp(-87.3365)
// = 2^-126, pow(2, -126) = 2^-126 exactly, pow(2, 127.999) finite); beyond the
// guards the result would over/underflow even after the df64 tail is applied.
// Overflow then happens naturally in ldexp (hi -> +inf, fix() clears lo) and
// results below FLT_MIN are flushed to +0 by exp_of_sum (this GPU flushes
// float32 denormals in arithmetic, so a denormal hi word must never be emitted).
static constant float DF64X_EXP_OVERFLOW = 88.73f;
static constant float DF64X_EXP_UNDERFLOW = -87.35f;
static constant float DF64X_FLT_MIN = 1.17549435e-38f;  // 2^-126
// Below this |x| the exp/log family returns its argument where f(x) = x + O(x^2):
// expm1, log1p (relative error <= |x|/2 < 2^-51 = u^2/8). This also keeps the
// lo word of tiny arguments intact (a lo word below FLT_MIN is flushed by any
// arithmetic on it) and avoids the series' r = x / 512 flushing to zero for
// |x| < 512 FLT_MIN, which returned exactly 0 for normal inputs.
static constant float DF64X_TINY = 8.8817842e-16f;  // 2^-50
// Largest |n| evaluated by binary exponentiation. Repeated squaring amplifies the
// first rounding by n/2, measured max error ~1.4 n u^2 (n = 8: 11.5 u^2, n = 64:
// 94 u^2), whereas exp(n log|a|) through the extended log stays below ~2 u^2.
// Small |n| keeps npwr so that integer powers of exactly representable values
// (2^8, 1.5^3, ...) stay exact.
static constant int DF64X_NPWR_MAX = 8;

namespace df {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Leading float P and df64 tail such that P + tail ~= a * (L.x + L.y + L.z) + ext * L.x
// with |error| ~ 2^-70 |P|. Requires L normalized like a triple word
// (|L.y| <= ulp(L.x)/2, |L.z| <= ulp(L.y)/2) and |ext| <~ 2^-48 |L.x|.
inline void prod3(df64 a, float ext, float3 L, thread float &P, thread df64 &tail) {
    df64 p1 = two_prod(a.hi, L.x);
    df64 p2 = two_prod(a.hi, L.y);
    df64 p3 = two_prod(a.lo, L.x);
    float p4 = fma(a.lo, L.y, fma(a.hi, L.z, ext * L.x));
    P = p1.hi;
    tail = add(make(p1.lo), p2);
    tail = add(tail, p3);
    tail = add(tail, p4);
}

// df64 rounding of a * (L.x + L.y + L.z) + ext * L.x (error ~0.5 u^2).
inline df64 mul3(df64 a, float ext, float3 L) {
    float P;
    df64 tail;
    prod3(a, ext, L, P, tail);
    return fix(add(make(P), tail));
}

// Core of exp: for y = P + tail (P float, |tail| << 1) returns s = exp(y - m ln2) - 1
// and m = nint(P / ln2). Caller applies ldexp(s + 1, m) and range guards.
inline df64 exp_core(float P, df64 tail, thread int &m) {
    float fm = metal::rint(P * DF64X_1_LN2_F);
    m = int(fm);
    // Exact Cody-Waite reduction: P - fm*A is exact (Sterbenz), fm*A and fm*B are
    // exact products, C is tiny; the df64 adds only round at ~u^2 |r|.
    float t0 = P - fm * DF64X_LN2_A;
    df64 r = add(make(t0), tail);
    r = sub(r, fm * DF64X_LN2_B);
    r = sub(r, mul(DF64X_LN2_C, fm));
    // |r| <= ln2/2 + tiny; scale by 2^-9 so that |r| <= 6.8e-4.
    r = mul_pwr2(r, 1.0f / 512.0f);
    // Taylor: s = r + r^2/2! + ... + r^6/6!  (r^7/7! < 2^-70 relative to r).
    df64 p = sqr(r);
    df64 s = add(r, mul_pwr2(p, 0.5f));
    p = mul(p, r);
    s = add(s, mul(p, DF64_INV_FACT[3]));
    p = mul(p, r);
    s = add(s, mul(p, DF64_INV_FACT[4]));
    p = mul(p, r);
    s = add(s, mul(p, DF64_INV_FACT[5]));
    p = mul(p, r);
    s = add(s, mul(p, DF64_INV_FACT[6]));
    // Undo the 2^-9 scaling: (1+s)^2 - 1 = 2s + s^2, nine times.
    for (int i = 0; i < 9; ++i) {
        s = add(mul_pwr2(s, 2.0f), sqr(s));
    }
    return s;
}

// exp(P + tail) with overflow/underflow guards on P; results below FLT_MIN
// (which this GPU would treat as zero anyway) are returned as +0.
inline df64 exp_of_sum(float P, df64 tail) {
    if (P > DF64X_EXP_OVERFLOW) return inf();
    if (P < DF64X_EXP_UNDERFLOW) return zero();
    int m;
    df64 s = exp_core(P, tail, m);
    df64 r = fix(ldexp(add(s, 1.0f), m));
    if (r.hi < DF64X_FLT_MIN) return zero();
    return r;
}

// log(1 + u) = 2 atanh(u / v) where v + rho_v == 2 + u exactly, |u/v| <= 0.172.
// Returns the df64 part; ext receives an extra low-order float (~0.1 u^2 total).
inline df64 log1p_core(df64 u, df64 v, float rho_v, thread float &ext) {
    df64 q = div(u, v);
    // Residual R = u - q * (v + rho_v) via exact products; q + R / v is the
    // quotient to ~2^-69 relative (v is exact up to rho_v by construction).
    df64 p1 = two_prod(q.hi, v.hi);
    df64 p2 = two_prod(q.hi, v.lo);
    df64 p3 = two_prod(q.lo, v.hi);
    float d0 = u.hi - p1.hi;  // exact: p1.hi is within a factor 2 of u.hi (or both 0)
    df64 acc = two_sum(d0, u.lo);
    acc = add(acc, -p1.lo);
    acc = add(acc, -p2.hi);
    acc = add(acc, -p3.hi);
    float tiny = fma(q.lo, v.lo, fma(q.hi, rho_v, p2.lo + p3.lo));
    acc = add(acc, -tiny);
    float rho = acc.hi / v.hi;  // s = q + rho
    // Series: atanh(s) = s (1 + w/3 + w^2/5 + ... ), w = s^2 <= 0.0295, terms to w^9/19
    // (w^10/21 < 2^-55). The polynomial part is <= 1% of the total, so its df64
    // rounding is damped by 100x.
    df64 w = sqr(q);
    df64 poly = DF64X_INV_ODD[8];
    for (int k = 7; k >= 0; --k) {
        poly = add(mul(poly, w), DF64X_INV_ODD[k]);
    }
    df64 T = mul(mul(q, w), poly);
    df64 c = add(T, rho);  // small correction, |c| << |q|
    // Three-word sum q + c (exact up to the last float add).
    df64 s1 = two_sum(q.hi, c.hi);
    df64 t1 = two_sum(q.lo, c.lo);
    df64 s2 = two_sum(s1.lo, t1.hi);
    df64 y = quick_two_sum(s1.hi, s2.hi);
    float y_ext = s2.lo + t1.lo;
    ext = 2.0f * y_ext;
    return mul_pwr2(y, 2.0f);
}

// Extended log of a finite, strictly positive df64: returns the df64 part and an
// extra low-order float in ext (log x ~= result + ext to ~0.1 u^2).
// x = (1, delta) with |delta| < 2^-50 returns delta directly (log(1 + delta) =
// delta - delta^2/2 + ..., relative error < 2^-51): the general path forms
// u / v = delta / 2 and frexp's ldexp(delta, -1), which flush to zero for
// FLT_MIN <= |delta| < 2 FLT_MIN and returned log(x) = 0 (and pow(x, b) = 1 for
// any b) although delta is a normal float32.
inline df64 log_ext(df64 x, thread float &ext) {
    if (x.hi == 1.0f && metal::abs(x.lo) < DF64X_TINY) {
        ext = 0.0f;
        return make(x.lo, 0.0f);
    }
    int e;
    df64 f = frexp(x, e);  // f in [0.5, 1)
    if (f.hi < 0.70710677f) {
        f = mul_pwr2(f, 2.0f);
        e -= 1;
    }
    // u = f - 1 is exact (Sterbenz on hi, lo untouched); v = u + 2 with exact residual.
    df64 u = quick_two_sum(f.hi - 1.0f, f.lo);
    df64 sv = two_sum(u.hi, 2.0f);
    df64 lv = two_sum(u.lo, sv.lo);
    df64 v = quick_two_sum(sv.hi, lv.hi);
    float yext;
    df64 y = log1p_core(u, v, lv.lo, yext);
    if (e == 0) {
        ext = yext;
        return y;
    }
    // Add e ln2 = e A + e B + e C (e A, e B exact products) as a three-word sum.
    float fe = float(e);
    float eA = fe * DF64X_LN2_A;
    float eB = fe * DF64X_LN2_B;
    df64 eC = mul(DF64X_LN2_C, fe);
    df64 h = two_sum(eA, y.hi);           // exact; |h.hi| >= 0.34
    df64 rest = two_sum(h.lo, eB);        // exact
    rest = add(rest, y.lo);
    rest = add(rest, eC);
    rest = add(rest, yext);
    df64 r = quick_two_sum(h.hi, rest.hi);  // |rest.hi| <= 2^-12 << |h.hi|
    df64 l = two_sum(r.lo, rest.lo);
    ext = l.lo;
    return make(r.hi, l.hi);
}

// Binary exponentiation of a finite nonzero a, |n| <= DF64X_NPWR_MAX. The base
// is split as a = f 2^e with |f| in [0.5, 1) (exact), f^|n| >= 2^-8 is formed
// by repeated squaring far from both ends of the float32 range, negative n
// takes the reciprocal of f^|n| (in (1, 2^8]) and the exact scaling 2^(n e) is
// applied once at the end by df::ldexp, which overflows / underflows exactly at
// the representable thresholds. Forming a^|n| directly and inverting it
// returned +-inf for results in (2^126, 2^128] (a^|n| flushed below FLT_MIN)
// and only float32 accuracy for results above ~3e23 (a^|n| below ~3.4e-24
// carries a flushed lo word).
inline df64 npwr(df64 a, int n) {
    if (n == 0) return one();
    int e;
    df64 f = frexp(a, e);
    unsigned k = (n < 0) ? unsigned(-(long)n) : unsigned(n);
    df64 base = f;
    df64 r = one();
    bool first = true;
    while (k != 0u) {
        if (k & 1u) {
            r = first ? base : mul(r, base);
            first = false;
        }
        k >>= 1;
        if (k != 0u) base = sqr(base);
    }
    if (n < 0) r = recip(r);
    return ldexp(r, n * e);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// exp(x): NaN -> NaN, +inf -> +inf, -inf -> 0, x > ln FLT_MAX = 88.722839 -> +inf,
// x < ln FLT_MIN = -87.336545 -> 0 (results below FLT_MIN flush), exp(+-0) = 1 exactly.
inline df64 exp(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return one();
    if (!isfinite(x.hi)) return x.hi > 0.0f ? inf() : zero();
    return exp_of_sum(x.hi, make(x.lo));
}

// exp(x) - 1 with full relative accuracy near 0 (m == 0 branch returns the
// reduced series directly; |x| < 2^-50 returns x itself); -inf -> -1,
// +inf -> +inf, expm1(+-0) = +-0, x > ln FLT_MAX -> +inf.
inline df64 expm1(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return x;
    if (!isfinite(x.hi)) return x.hi > 0.0f ? inf() : make(-1.0f, 0.0f);
    if (metal::abs(x.hi) < DF64X_TINY) return fix(x);
    if (x.hi > DF64X_EXP_OVERFLOW) return inf();
    if (x.hi < DF64X_EXP_UNDERFLOW) return make(-1.0f, 0.0f);
    int m;
    df64 s = exp_core(x.hi, make(x.lo), m);
    if (m == 0) return fix(s);
    df64 r = fix(ldexp(add(s, 1.0f), m));
    if (r.hi < DF64X_FLT_MIN) return make(-1.0f, 0.0f);
    return fix(sub(r, 1.0f));
}

// 2^x: exact for integer x in [-126, 127]; x >= 128 -> +inf, x < -126 -> 0.
// The guards look at the whole pair: (128, negative lo) is below 128 and 2^x is
// then a finite float32 (2^(128 - 1e-6) = 0.9999993 FLT_MAX); values that still
// overflow after the tail is applied do so in ldexp (fix() clears lo).
inline df64 exp2(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return one();
    if (!isfinite(x.hi)) return x.hi > 0.0f ? inf() : zero();
    if (x.hi > 128.0f || (x.hi == 128.0f && x.lo >= 0.0f)) return inf();
    if (x.hi < -126.0f) return zero();
    float fm = metal::rint(x.hi);
    df64 f = add(x, -fm);  // exact: |x - fm| <= 1/2 on x's grid
    float P;
    df64 tail;
    prod3(f, 0.0f, DF64X_LN2_3, P, tail);  // f ln2 to ~72 bits, |P| <= 0.35
    int m;
    df64 s = exp_core(P, tail, m);
    df64 r = fix(ldexp(add(s, 1.0f), int(fm) + m));
    if (r.hi < DF64X_FLT_MIN) return zero();
    return r;
}

// 10^x = exp(x ln10) with x ln10 formed to ~72 bits; x > log10 FLT_MAX = 38.5318
// -> +inf, x < log10 FLT_MIN = -37.9298 -> 0 (the exp guards apply to x ln10).
inline df64 exp10(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return one();
    if (!isfinite(x.hi)) return x.hi > 0.0f ? inf() : zero();
    float P;
    df64 tail;
    prod3(x, 0.0f, DF64X_LN10_3, P, tail);
    return exp_of_sum(P, tail);
}

// log(x): x < 0 -> NaN, +-0 -> -inf, +inf -> +inf, NaN -> NaN, log(1) = 0 exactly.
inline df64 log(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return make(-INFINITY, 0.0f);
    if (is_negative(x)) return nan();
    if (!isfinite(x.hi)) return inf();
    float ext;
    df64 y = log_ext(x, ext);
    return fix(add(y, ext));
}

// log(1 + x): x < -1 -> NaN, x == -1 -> -inf, log1p(+-0) = +-0, relative accuracy
// for small |x| (|x| < 2^-50 returns x; otherwise the direct series on
// x / (2 + x)); otherwise log(1 + x) with the exact rounding residual of 1 + x
// folded back in.
inline df64 log1p(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return x;
    if (metal::abs(x.hi) < DF64X_TINY) return fix(x);
    if (!isfinite(x.hi)) return x.hi > 0.0f ? inf() : nan();
    if (x.hi < -1.0f || (x.hi == -1.0f && x.lo < 0.0f)) return nan();
    if (x.hi == -1.0f && x.lo == 0.0f) return make(-INFINITY, 0.0f);
    float ext;
    if (metal::abs(x.hi) < 0.25f) {
        df64 sv = two_sum(x.hi, 2.0f);
        df64 lv = two_sum(x.lo, sv.lo);
        df64 v = quick_two_sum(sv.hi, lv.hi);
        df64 y = log1p_core(x, v, lv.lo, ext);
        return fix(add(y, ext));
    }
    // w + rho == 1 + x exactly.
    df64 sw = two_sum(x.hi, 1.0f);
    df64 lw = two_sum(x.lo, sw.lo);
    df64 w = quick_two_sum(sw.hi, lw.hi);
    float rho = lw.lo;
    if (w.hi <= 0.0f) return w.hi == 0.0f && rho == 0.0f ? make(-INFINITY, 0.0f) : nan();
    df64 y = log_ext(w, ext);
    y = add(y, ext);
    return fix(add(y, rho / w.hi));
}

// log2(x) = log(x) / ln2 with the unrounded log; exact for powers of two. The
// power-of-two test must not absorb x = (1, delta) whose lo word flushes under
// frexp's ldexp(delta, -1) (|delta| < 2 FLT_MIN): log2 is then delta / ln2, not
// 0. For hi = 2^k with k != 0 a flushed lo word changes log2 by < 2^-124 |k|,
// far below the rounding of the integer result, so those keep the shortcut.
inline df64 log2(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return make(-INFINITY, 0.0f);
    if (is_negative(x)) return nan();
    if (!isfinite(x.hi)) return inf();
    int e;
    df64 f = frexp(x, e);
    if (f.hi == 0.5f && f.lo == 0.0f && !(x.hi == 1.0f && x.lo != 0.0f)) {
        return from_int(e - 1);
    }
    float ext;
    df64 y = log_ext(x, ext);
    return mul3(y, ext, DF64X_1_LN2_3);
}

// log10(x) = log(x) / ln10 with the unrounded log.
inline df64 log10(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return make(-INFINITY, 0.0f);
    if (is_negative(x)) return nan();
    if (!isfinite(x.hi)) return inf();
    float ext;
    df64 y = log_ext(x, ext);
    return mul3(y, ext, DF64X_1_LN10_3);
}

inline df64 pow(df64 a, df64 b);

// a^n for integer n (C pow semantics: pow(x, 0) = 1 for any x, negative n via
// 1/a^|n|); |n| <= DF64X_NPWR_MAX by binary exponentiation, larger |n| through
// the general path (df::from_int is exact for every int32, including the
// values in [2^31 - 64, 2^31) whose float32 rounds to 2^31).
inline df64 pow(df64 a, int n) {
    if (n == 0) return one();
    if (is_nan(a)) return nan();
    bool odd = (n & 1) != 0;
    if (is_zero(a)) {
        // Signed-zero rules; df64 mul/sqr would turn -0 into +0 (quick_two_sum(-0, +0)).
        if (n < 0) return odd ? make(metal::copysign(INFINITY, a.hi), 0.0f) : inf();
        return odd ? a : zero();
    }
    if (!isfinite(a.hi)) {
        // C99: (+-inf)^n = +-inf resp. +inf for n > 0, +-0 resp. +0 for n < 0.
        float s = odd ? a.hi : metal::abs(a.hi);
        return n > 0 ? make(s, 0.0f) : make(metal::copysign(0.0f, s), 0.0f);
    }
    if (n >= -DF64X_NPWR_MAX && n <= DF64X_NPWR_MAX) return fix(npwr(a, n));
    return pow(a, from_int(n));
}

// a^b following C99/IEEE pow special cases; sign * exp(b log|a|) in general.
inline df64 pow(df64 a, df64 b) {
    // pow(x, +-0) = 1 and pow(1, y) = 1 even for NaN operands.
    if (is_zero(b)) return one();
    if (a.hi == 1.0f && a.lo == 0.0f) return one();
    if (is_nan(a) || is_nan(b)) return nan();
    if (b.hi == 1.0f && b.lo == 0.0f) return a;
    // b = +-inf: |a| < 1 is decided on the whole pair ((1, -2^-30) is below 1).
    if (!isfinite(b.hi)) {
        df64 aa = abs(a);
        if (aa.hi == 1.0f && aa.lo == 0.0f) return one();
        bool small = aa.hi < 1.0f || (aa.hi == 1.0f && aa.lo < 0.0f);
        return (small == (b.hi > 0.0f)) ? zero() : inf();
    }
    bool b_int = eq(floor(b), b);
    df64 half_b = mul_pwr2(b, 0.5f);
    bool b_odd = b_int && ne(floor(half_b), half_b);
    // a = +-0
    if (is_zero(a)) {
        if (is_negative(b)) return b_odd ? make(metal::copysign(INFINITY, a.hi), 0.0f) : inf();
        return b_odd ? a : zero();
    }
    // a = +-inf
    if (!isfinite(a.hi)) {
        if (a.hi > 0.0f) return is_negative(b) ? zero() : inf();
        if (is_negative(b)) return b_odd ? make(-0.0f, 0.0f) : zero();
        return b_odd ? make(-INFINITY, 0.0f) : inf();
    }
    bool a_neg = is_negative(a);
    if (a_neg && !b_int) return nan();
    if (b.lo == 0.0f) {
        if (b.hi == 2.0f) return sqr(a);
        if (b.hi == -1.0f) return recip(a);
        if (b.hi == 0.5f) return sqrt(a);
        if (b.hi == -0.5f) return rsqrt(a);
    }
    if (b_int && b.hi >= -float(DF64X_NPWR_MAX) && b.hi <= float(DF64X_NPWR_MAX)) {
        return fix(npwr(a, to_int(b)));
    }
    df64 aa = a_neg ? neg(a) : a;
    float ext;
    df64 y = log_ext(aa, ext);
    float P;
    df64 tail;
    prod3(b, 0.0f, float3(y.hi, y.lo, ext), P, tail);
    df64 r = exp_of_sum(P, tail);
    return (a_neg && b_odd) ? neg(r) : r;
}

// cbrt(x): odd, cbrt(+-0) = +-0, cbrt(+-inf) = +-inf, NaN -> NaN.
inline df64 cbrt(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x) || !isfinite(x.hi)) return fix(x);
    bool negative = is_negative(x);
    df64 a = negative ? neg(x) : x;
    int e;
    df64 f = frexp(a, e);  // f in [0.5, 1), a = f 2^e
    int rem = e % 3;
    if (rem < 0) rem += 3;
    f = ldexp(f, rem);     // f in [0.5, 4), e - rem divisible by 3
    e -= rem;
    df64 y = make(metal::precise::pow(f.hi, 1.0f / 3.0f));
    // Two Newton steps y -= (y^3 - f) / (3 y^2): float seed -> ~2^-46 -> ~2^-90.
    for (int i = 0; i < 2; ++i) {
        df64 y2 = sqr(y);
        df64 num = sub(mul(y2, y), f);
        df64 den = mul(y2, 3.0f);
        y = sub(y, div(num, den));
    }
    y = ldexp(y, e / 3);
    return fix(negative ? neg(y) : y);
}

}  // namespace df

#endif  // OPTILAND_DF64_MATH_EXP_H
