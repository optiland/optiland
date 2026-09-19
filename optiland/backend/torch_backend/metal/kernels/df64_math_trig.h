// df64_math_trig.h — trigonometric functions for double-single ("df64") values.
//
// Provides, in namespace df:: (df64 arguments, df64 results):
//   sincos(x, s, c), sin(x), cos(x), tan(x), asin(x), acos(x), atan(x), atan2(y, x)
//
// Scheme (QD dd_real.cpp, Hida-Li-Bailey), adapted to float32 pairs:
//   1. Quadrant reduction. j = nint(x * 2/pi) computed as a df64 product (a float
//      product alone would be off by up to 0.4 for |x| ~ 1e7) and corrected from
//      the whole pair (jd.hi carries 24 bits only, so rint(jd.hi) can be off by
//      one for |x| > ~1e4), r = x - j*pi/2 with pi/2 as the Cody-Waite chunks
//      PIO2_1/2/3 (12 significant bits each) plus the df64 remainder PIO2_REST
//      and two further remainder words PIO2_REST3/REST4 (pi/2 to 2^-142.4
//      relative). Every product j*chunk and j*REST word is formed with two_prod
//      (exact pairs for any |j| < 2^24; j*REST4 with one float rounding at
//      ~2^-136 |x|), the chunk hi parts are subtracted from a float accumulator
//      with two_sum (exact), and every other term (x.lo, the two_sum residuals,
//      the two_prod residuals, the REST products) is summed in a four-word
//      accumulator (df::acc4_add, one rounding at ~2^-96 of the running
//      magnitude, i.e. <= ~2^-120 |x|) that is rounded to df64 once at the end.
//      The reduced argument therefore has an absolute error floor of ~2^-121 |x|
//      (measured 2^-120.6..2^-121.4 |x| at the df64 pairs closest to k pi/2) and
//      keeps its relative accuracy (<= 0.5 u^2) while |r| >= ~2^-72 |x|. The
//      df64 pairs closest to a nonzero multiple of pi/2 (exhaustive search,
//      exact integer arithmetic) are 2^-63.2 |x| away for |x| <= 1e4 (k = 1425)
//      and 2^-75.0 |x| away for |x| <= 1e7 (k = 4880635), so every in-range
//      argument is covered: measured at the closest pairs (|x| ~ 2.7e6..7.9e6)
//      tan <= 0.35 u^2 and the tiny one of sin/cos <= 0.07 u^2 relative; on 12k
//      points k pi/2 (1 + f), |k| <= 6366, |f| in [1e-17.5, 1e-11] and on
//      float64(k pi) for k <= 3183: tan <= 1.4 u^2, sin/cos <= 1.1 u^2. (Before
//      fix round 2 the third word entered with one float rounding and the
//      constant stopped at 2^-114.8 |x|, so those closest pairs saw 63-519 u^2;
//      before fix round 1 the df64 product j * PIO2_REST limited r to ~2^-87 |x|
//      and tan to ~7000 u^2 at its zeros.) |r| <= pi/4 (+ a few ulp) for every
//      |x| <= 1e7 (the k-clamp of step 2 is a safety net, not load-bearing),
//      quadrant q = j mod 4.
//   2. Table step. k = nint(r * 16/pi) in [-4, 4], t = r - k*pi/16 with a df64 table
//      of k*pi/16 (|t| <= pi/32 ~ 0.098).
//   3. Taylor series of sin t (odd powers through t^13) and cos t (even powers through
//      t^12) with the DF64_INV_FACT table; truncation error < 2^-60 relative.
//   4. Angle addition with DF64_SIN_TABLE / DF64_COS_TABLE (sin, cos of |k|*pi/16),
//      then rotation by the quadrant.
// Every step is sign-symmetric (RNE float arithmetic is), so sin(-x) == -sin(x) and
// cos(-x) == cos(x) bit-for-bit; sin(+-0) = +-0 and cos(0) = 1 exactly.
//
// Range. Arguments with |x.hi| > DF64_TRIG_MAX_ARG (1e7) are NOT reduced in df64:
// sin/cos/tan fall back to float32 (metal::precise::sin/cos of x.hi combined with
// sin/cos of x.lo by the angle-addition formulas, so the phase carried by the lo
// word, up to ulp(x.hi)/2 = 32 rad at |x| = 1e9, is not dropped; result lo = 0,
// absolute error ~1e-7 against sin/cos of the decoded pair). Optiland never needs
// such arguments (ray angles and phases are far smaller); the fallback exists only
// so that no garbage is produced; its results are clamped to [-1, 1] so that
// asin/acos/sqrt(1 - s^2) of them stay defined. Inside the range sin/cos/tan keep
// their relative accuracy (<= 20 u^2 resp. 50 u^2, measured <= 4.3 u^2) for every
// df64 argument, including the zeros of the function next to k pi/2 (see step 1).
// Tiny arguments: |x.hi| < 2^-26 returns sin x = x, tan x = x, cos x = 1 - x^2/2
// and asin a = atan a = a exactly (the omitted x^3 term is below the ulp of the
// lo word); this also keeps a lo word below FLT_MIN intact, which any arithmetic
// would flush (previously sin/asin/atan of |x| < ~3e-24 dropped it).
//
// Inverse functions. atan2(y, x) uses one Newton step from the float32 seed
// z0 = precise::atan2(y.hi, x.hi) on the residual g(z) = ys*cos z - xs*sin z
// (xs, ys = x, y scaled by a common power of two to avoid overflow; for x > 0
// and |z0| < 2^-26 the quotient y/x is returned instead, formed with each
// operand scaled into [0.5, 1) and one exact rescale, because the common
// scaling would put the residual terms below FLT_MIN where they flush):
//     z = z0 + (ys*cos z0 - xs*sin z0) / (ys*sin z0 + xs*cos z0)
// This is Newton's method on r*sin(theta - z) whose iteration error is cubic
// ((theta - z0)^3 / 3 ~ 1e-22 for a float seed), so one step suffices and no
// normalization sqrt/division is needed; the denominator is r*cos(theta - z0) > 0.
// Special cases follow IEEE/C99 (torch.atan2): atan2(+-0, +0) = +-0,
// atan2(+-0, -0) = +-pi, atan2(+-0, x>0) = +-0, atan2(+-0, x<0) = +-pi,
// atan2(y != 0, +-0) = +-pi/2, atan2(+-inf, +inf) = +-pi/4, atan2(+-inf, -inf) =
// +-3pi/4, atan2(+-inf, finite) = +-pi/2, atan2(finite, +inf) = +-0,
// atan2(finite, -inf) = +-pi, NaN in -> NaN out. atan(x) = atan2(x, 1).
// asin(a) = sign(a) atan2(|a|, sqrt((1-a)(1+a))) (odd bit-for-bit),
// acos(a) = atan2(sqrt((1-a)(1+a)), a);
// |a| > 1 -> NaN, asin(+-1) = +-pi/2, acos(1) = +0, acos(-1) = pi exactly.
// Accuracy floor (df64_core.h): results below ~3e-24 can carry up to FLT_MIN of
// absolute error (their lo word is a flushed denormal), which exceeds the
// relative targets for |atan2| < ~1e-25 in a small fraction of cases.
//
// Amalgamation order: df64_core.h, df64_constants.h, [df64_math_exp.h], this file.
// This header has no #include and depends only on df64_core.h and df64_constants.h.
//
// Part of Optiland-Metal (MIT). Structure follows QD (BSD); all code written fresh.

#ifndef OPTILAND_DF64_MATH_TRIG_H
#define OPTILAND_DF64_MATH_TRIG_H

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// k*pi/16 for k = 0..4 as normalized float32 pairs (mpmath, |lo| <= ulp(hi)/2;
// k = 1, 2, 4 are exact power-of-two multiples of DF64_PI_16).
static constant df64 DF64_TRIG_KPI16[5] = {
    {0.0f, 0.0f},
    {0.196349546f, -5.46392354e-09f},   // pi/16
    {0.392699093f, -1.09278471e-08f},   // 2 pi/16
    {0.589048624f, -1.49061008e-09f},   // 3 pi/16
    {0.785398185f, -2.18556941e-08f},   // 4 pi/16 = pi/4
};
// float32 nearest to 16/pi (only used to pick the table index).
static constant float DF64_TRIG_16_PI_F = 5.09295797f;
// Largest |x.hi| reduced in df64; beyond it sin/cos/tan fall back to float32.
static constant float DF64_TRIG_MAX_ARG = 1.0e7f;
// Third and fourth words of the pi/2 remainder: pi/2 - PIO2_1 - PIO2_2 - PIO2_3
// - PIO2_REST (mpmath 300 bits; the residual after REST3 is 4.34e-35 =
// 2^-114.8 of pi/2, after REST4 2.1e-43 = 2^-142.4).
static constant float DF64_TRIG_PIO2_REST3 = -2.9407887e-27f;
static constant float DF64_TRIG_PIO2_REST4 = 4.3359050e-35f;
// Below this |x| sin/tan/asin/atan return x and cos returns 1 - x^2/2 (see above).
static constant float DF64_TRIG_TINY = 1.4901161e-08f;  // 2^-26

namespace df {

// ---------------------------------------------------------------------------
// Taylor series on |t| <= pi/32 (a little more is tolerated: the reduced argument
// may exceed the nominal bound by a few ulp when j or k round the other way).
// ---------------------------------------------------------------------------
// sin t = t - t^3/3! + t^5/5! - ... + t^13/13!  (t^15/15! < 2^-70 |t|).
inline df64 sin_taylor(df64 t) {
    df64 x2 = neg(sqr(t));      // -t^2, so the alternating sign is automatic
    df64 r = t;                 // running power t^(2n+1) with sign
    df64 s = t;
    for (int i = 3; i <= 13; i += 2) {
        r = mul(r, x2);
        s = add(s, mul(r, DF64_INV_FACT[i]));
    }
    return s;
}
// cos t = 1 - t^2/2! + t^4/4! - ... + t^12/12!  (t^14/14! < 2^-70).
inline df64 cos_taylor(df64 t) {
    df64 x2 = neg(sqr(t));
    df64 r = x2;                // running power t^(2n) with sign
    df64 s = add(one(), mul_pwr2(r, 0.5f));
    for (int i = 4; i <= 12; i += 2) {
        r = mul(r, x2);
        s = add(s, mul(r, DF64_INV_FACT[i]));
    }
    return s;
}

// ---------------------------------------------------------------------------
// Argument reduction: r = x - j*pi/2, |r| <= pi/4 (+ few ulp), q = j mod 4 in [0, 3].
// Requires x finite with |x.hi| <= DF64_TRIG_MAX_ARG (so |j| < 2^23).
// ---------------------------------------------------------------------------
inline df64 reduce_pio2(df64 x, thread int &q) {
    df64 jd = mul(x, DF64_2_PI);
    float j = metal::rint(jd.hi);
    // jd.hi alone carries 24 bits (ulp 0.5 for |x| > 6.6e6, so rint of it can be
    // off by one): correct j from the whole pair so that |r| <= pi/4 (+ ulp).
    float fr = (jd.hi - j) + jd.lo;  // jd.hi - j is exact (Sterbenz)
    if (fr > 0.5f) j += 1.0f;
    else if (fr < -0.5f) j -= 1.0f;
    if (j == 0.0f) { q = 0; return x; }
    q = int(j) & 3;  // two's complement: (-1) & 3 == 3, so negative j map correctly

    // Float accumulator for the leading part, four-word accumulator for every
    // residual (exact except for one rounding per term at the fourth word,
    // ~2^-96 of the running magnitude, which is at most ~2^-24 |x|).
    float acc = x.hi;
    float4 tail = acc4_init(x.lo, 0.0f);
    df64 p;
    df64 s;

    p = two_prod(j, DF64_PIO2_1);           // exact pair
    s = two_sum(acc, -p.hi);                // exact pair (Sterbenz: s.lo == 0)
    acc = s.hi;
    tail = acc4_add(tail, s.lo);
    tail = acc4_add(tail, -p.lo);

    p = two_prod(j, DF64_PIO2_2);
    s = two_sum(acc, -p.hi);
    acc = s.hi;
    tail = acc4_add(tail, s.lo);
    tail = acc4_add(tail, -p.lo);

    p = two_prod(j, DF64_PIO2_3);
    s = two_sum(acc, -p.hi);
    acc = s.hi;
    tail = acc4_add(tail, s.lo);
    tail = acc4_add(tail, -p.lo);

    // Fold the leading float in first (it cancels the bulk of the tail near a
    // multiple of pi/2), then the exact products with the remainder words.
    tail = acc4_add(tail, acc);
    p = two_prod(j, DF64_PIO2_REST.hi);
    tail = acc4_add(tail, -p.hi);
    tail = acc4_add(tail, -p.lo);
    p = two_prod(j, DF64_PIO2_REST.lo);
    tail = acc4_add(tail, -p.hi);
    tail = acc4_add(tail, -p.lo);
    p = two_prod(j, DF64_TRIG_PIO2_REST3);
    tail = acc4_add(tail, -p.hi);
    tail = acc4_add(tail, -p.lo);
    tail = acc4_add(tail, -(j * DF64_TRIG_PIO2_REST4));
    return acc4_to_df64(tail);
}

// ---------------------------------------------------------------------------
// sincos
// ---------------------------------------------------------------------------
inline void sincos(df64 x, thread df64 &s, thread df64 &c) {
    if (!isfinite(x.hi)) {               // +-inf and NaN -> NaN (IEEE)
        s = nan();
        c = nan();
        return;
    }
    if (metal::fabs(x.hi) < DF64_TRIG_TINY) {
        // incl. +-0: sin x = x (x^3/6 is below the ulp of x's lo word) and
        // cos x = 1 - x^2/2 with the correction carried in the lo word while it
        // is a normal float32 (it flushes to 0 below |x| ~ 1.5e-19).
        s = fix(x);
        float half_x2 = 0.5f * (x.hi * x.hi);
        c = make(1.0f, half_x2 == 0.0f ? 0.0f : -half_x2);
        return;
    }
    if (metal::fabs(x.hi) > DF64_TRIG_MAX_ARG) {
        // Out of the df64 reduction range: float32 accuracy only (documented).
        // The lo word (|lo| <= ulp(hi)/2, up to tens of radians for |x| ~ 1e9)
        // is folded in with the angle-addition formulas instead of being dropped.
        float sh = metal::precise::sin(x.hi);
        float ch = metal::precise::cos(x.hi);
        float sl = metal::precise::sin(x.lo);
        float cl = metal::precise::cos(x.lo);
        // The float32 angle-addition sums can exceed 1 by an ulp; clamp so that
        // asin/acos/sqrt(1 - s^2) of the result stay defined.
        s = make(metal::clamp(sh * cl + ch * sl, -1.0f, 1.0f), 0.0f);
        c = make(metal::clamp(ch * cl - sh * sl, -1.0f, 1.0f), 0.0f);
        return;
    }

    int q;
    df64 r = reduce_pio2(x, q);

    // Table step: t = r - k*pi/16, |k| <= 4, |t| <= pi/32.
    float kf = metal::rint(r.hi * DF64_TRIG_16_PI_F);
    int k = int(kf);
    int ak = metal::abs(k);
    if (ak > 4) { ak = 4; k = (k > 0) ? 4 : -4; }  // safety only: |r| <= pi/4 + ulp after the j correction
    df64 t = r;
    if (k > 0) t = sub(r, DF64_TRIG_KPI16[ak]);
    else if (k < 0) t = add(r, DF64_TRIG_KPI16[ak]);

    df64 st = sin_taylor(t);
    df64 ct = cos_taylor(t);
    df64 sr;
    df64 cr;
    if (k == 0) {
        sr = st;
        cr = ct;
    } else {
        df64 u = DF64_COS_TABLE[ak];  // cos(|k| pi/16)
        df64 v = DF64_SIN_TABLE[ak];  // sin(|k| pi/16)
        if (k > 0) {
            sr = add(mul(u, st), mul(v, ct));   // sin(t + a) = sin t cos a + cos t sin a
            cr = sub(mul(u, ct), mul(v, st));   // cos(t + a) = cos t cos a - sin t sin a
        } else {
            sr = sub(mul(u, st), mul(v, ct));   // sin(t - a)
            cr = add(mul(u, ct), mul(v, st));   // cos(t - a)
        }
    }

    // Rotate by the quadrant: (sin, cos) of r + q*pi/2.
    switch (q) {
        case 0:  s = sr;      c = cr;      break;
        case 1:  s = cr;      c = neg(sr); break;
        case 2:  s = neg(sr); c = neg(cr); break;
        default: s = neg(cr); c = sr;      break;
    }
}

inline df64 sin(df64 x) {
    df64 s;
    df64 c;
    sincos(x, s, c);
    return s;
}

inline df64 cos(df64 x) {
    df64 s;
    df64 c;
    sincos(x, s, c);
    return c;
}

// tan = sin / cos (huge but finite near odd multiples of pi/2; tan(+-0) = +-0,
// tan x = x for |x| < 2^-26).
inline df64 tan(df64 x) {
    if (metal::fabs(x.hi) < DF64_TRIG_TINY) return fix(x);
    df64 s;
    df64 c;
    sincos(x, s, c);
    return div(s, c);
}

// ---------------------------------------------------------------------------
// atan2 and friends
// ---------------------------------------------------------------------------
inline df64 atan2(df64 y, df64 x) {
    if (is_nan(x) || is_nan(y)) return nan();
    const bool yneg = metal::signbit(y.hi);
    if (y.hi == 0.0f) {
        // atan2(+-0, x>0 or +0) = +-0; atan2(+-0, x<0 or -0) = +-pi.
        if (x.hi > 0.0f || (x.hi == 0.0f && !metal::signbit(x.hi))) return make(y.hi, 0.0f);
        return yneg ? neg(DF64_PI) : DF64_PI;
    }
    if (x.hi == 0.0f) return yneg ? neg(DF64_PI_2) : DF64_PI_2;
    if (isinf(x.hi)) {
        if (x.hi > 0.0f) {
            if (isinf(y.hi)) return yneg ? neg(DF64_PI_4) : DF64_PI_4;
            return make(yneg ? -0.0f : 0.0f, 0.0f);
        }
        if (isinf(y.hi)) return yneg ? neg(DF64_3PI_4) : DF64_3PI_4;
        return yneg ? neg(DF64_PI) : DF64_PI;
    }
    if (isinf(y.hi)) return yneg ? neg(DF64_PI_2) : DF64_PI_2;

    int ex;
    int ey;
    metal::frexp(x.hi, ex);
    metal::frexp(y.hi, ey);
    const float z0 = metal::precise::atan2(y.hi, x.hi);
    if (x.hi > 0.0f && metal::fabs(z0) < DF64_TRIG_TINY) {
        // Tiny quotient: atan2(y, x) = y/x to within (y/x)^2/3 < 2^-53 relative.
        // Each operand is scaled into [0.5, 1) separately and the quotient
        // rescaled once (exact, or flushed below FLT_MIN), so that no
        // intermediate falls into the flushed range: the common scaling below
        // would put ys and every Newton residual at ~|y/x|, where the two_prod
        // error terms and the x.lo cross term are lost (up to 4096 u^2 at
        // 1e-27, ~3 FLT_MIN absolute for |y/x| ~ 1e-29 with x ~ 1e28).
        return ldexp(div(ldexp(y, -ey), ldexp(x, -ex)), ey - ex);
    }

    // Scale both operands by a common power of two (exact) so that products
    // and sums below cannot overflow (they cannot underflow either once
    // |y/x| >= 2^-26: every cross term is then >= 2^-75 of the larger operand).
    const int e = metal::max(ex, ey);
    df64 xs = ldexp(x, -e);
    df64 ys = ldexp(y, -e);

    // Float seed in the correct quadrant, then one cubic-convergent Newton step
    // on g(z) = ys cos z - xs sin z = r sin(theta - z).
    df64 z = make(z0, 0.0f);
    df64 s;
    df64 c;
    sincos(z, s, c);
    df64 num = sub(mul(ys, c), mul(xs, s));
    df64 den = add(mul(ys, s), mul(xs, c));
    return fix(add(z, div(num, den)));
}

// atan(x) = atan2(x, 1); |x| < 2^-26 returns x (atan x - x = -x^3/3 < u^2/2 |x|),
// which also keeps a lo word below FLT_MIN that the atan2 scaling would flush.
inline df64 atan(df64 x) {
    if (metal::fabs(x.hi) < DF64_TRIG_TINY) return fix(x);
    return atan2(x, one());
}

// sqrt((1 - a)(1 + a)) evaluated without cancellation near |a| = 1.
inline df64 sqrt_one_minus_sqr(df64 a) {
    return sqrt(mul(sub(1.0f, a), add(1.0f, a)));
}

// asin is evaluated on |a| with the sign applied afterwards, so it is odd
// bit-for-bit (df::mul is not commutative bit-for-bit, and (1 - a)(1 + a)
// swaps its factors under a -> -a).
inline df64 asin(df64 a) {
    if (is_nan(a)) return nan();
    df64 aa = abs(a);
    if (gt(aa, one())) return nan();
    if (eq(aa, one())) return is_negative(a) ? neg(DF64_PI_2) : DF64_PI_2;
    if (metal::fabs(a.hi) < DF64_TRIG_TINY) return fix(a);  // asin(+-0) = +-0, asin a = a
    df64 r = atan2(aa, sqrt_one_minus_sqr(aa));
    return is_negative(a) ? neg(r) : r;
}

inline df64 acos(df64 a) {
    if (is_nan(a)) return nan();
    df64 aa = abs(a);
    if (gt(aa, one())) return nan();
    if (eq(aa, one())) return is_negative(a) ? DF64_PI : zero();
    return atan2(sqrt_one_minus_sqr(a), a);
}

}  // namespace df

#endif  // OPTILAND_DF64_MATH_TRIG_H
