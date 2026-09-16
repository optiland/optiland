// df64_core.h — double-single ("df64") arithmetic for Apple GPUs (Metal Shading Language).
//
// A df64 value is an unevaluated sum hi + lo of two float32 with |lo| <= ulp(hi)/2
// (normalized). It carries ~48 significant bits (u^2 = 2^-48 = 3.55e-15 relative)
// with float32 exponent range. Algorithms follow Joldes, Muller & Popescu 2017
// ("Tight and Rigorous Error Bounds for Basic Building Blocks of Double-Word
// Arithmetic", ACM TOMS 44(2)) with the Muller & Rideau 2022 corrections, and the
// QD library (Hida, Li & Bailey) for division and square root. Error bounds quoted
// below are for round-to-nearest-even float32 with an unbounded exponent range.
//
// Canonical form. Every operation here returns the RNE-canonical pair
// (hi = RN(x), lo = x - hi). A value whose lo is exactly +-ulp(hi)/2 has a second
// valid representation with an odd hi word ("tie form"), which the host encoder
// and sf::to_float2 may produce from a float64; comparisons (eq/lt/...), the
// rounding functions and to_int therefore canonicalize their operands first
// (canon() below), and add/sub decide exact-zero results from the value, not
// from the hi words. Hand-built pairs must satisfy |lo| <= ulp(hi)/2 and
// |hi + lo| < FLT_MAX + 2^103 (see the overflow rule).
//
// Overflow rule. A df64 value overflows to +-inf exactly when RN32 of its value
// overflows, i.e. |x| >= FLT_MAX + ulp(FLT_MAX)/2 = 2^128 - 2^103 (the RNE tie
// FLT_MAX + 2^103 itself rounds to inf, so the largest finite df64 is
// (FLT_MAX, 2^103 - 2^79)). add/sub/mul/sqr/div, ldexp and mul_pwr2 therefore
// stay finite for results in (FLT_MAX, FLT_MAX + 2^103) although the float32
// product / quotient / sum of the hi words alone would overflow: the operation
// is redone on halved operands and scaled back with scaled_overflow(), which
// rebuilds the (FLT_MAX, lo) pair. add decides the threshold from an exact
// residual; mul/div carry their normal <= 5 u^2 rounding into the decision,
// so a product within ~5 u^2 of the threshold may still round to inf.
// The encoder (encode.py) and sf::to_float2 map float64 values at or above the
// tie to +-inf with lo = 0, never to an (inf, -inf) pair.
//
// Underflow rule. The GPU flushes float32 denormals before rounding, so a
// product / quotient whose exact value rounds to FLT_MIN can be flushed to zero
// by the hi-word operation alone; mul/sqr/div detect a flushed hi word with
// nonzero operands and redo the operation on 2^48 / 2^64-scaled operands (so
// that the error terms stay normal too), and div forms its residual on a
// 2^64-scaled dividend whenever |a.hi| < 2^-78 (below that the residual
// a - b q1 and its error terms can flush to zero: between 2^-100 and 2^-78
// the quotient lost up to 26 bits for ~23% of the dividends near 2^-99
// although the quotient itself was of ordinary size; below 2^-100 every
// refinement added q1 again).
// Results below FLT_MIN are +-0 (sign of the exact result): when the scaled
// retry of mul still flushes, the sign comes from the hardware product of the
// hi words, which keeps it (a 2^48-scaled retry alone returned +0 for
// negative products below ~2^-174 because quick_two_sum(-0, +0) is +0).
//
// Absolute error floor (FTZ, measured): every error term below FLT_MIN =
// 1.18e-38 is lost (two_sum / two_prod residuals, the lo word itself, div's
// correction quotients), so besides the relative bounds every operation
// carries a sporadic absolute error of up to ~FLT_MIN. For |x| < 2^-102 ~ 2e-31
// the lo word is always a flushed denormal (24-bit precision); between 2^-102
// and 2^-78 ~ 3.3e-24 the precision degrades gradually (48 bits only when the
// lo word happens to be >= FLT_MIN, i.e. relative error up to FLT_MIN/|x| with
// probability ~2^-102/|x| per rounding); above 2^-78 the full u^2 bounds hold.
// Functions that scale operands down to [0.5, 1) must not land intermediates
// below ~2^-78 when their inputs are normal (atan2 scales each operand
// separately for tiny quotients for this reason).
//
// Compiler contract (NON-NEGOTIABLE):
//   * safe math mode and precise float functions (PyTorch: PYTORCH_MPS_FAST_MATH=0
//     before the first Metal compile; standalone: -fmetal-math-mode=safe
//     -fmetal-math-fp32-functions=precise -ffp-contract=off);
//   * the pragmas below re-assert this per translation unit so that a fast-math
//     build cannot silently reassociate the error-free transformations;
//   * every fused operation is written explicitly as fma(); nothing else may be
//     contracted (contract(off)).
//
// Denormals (measured, NOTES/01-environment-findings.md): Apple GPUs flush
// float32 denormals to zero in arithmetic AND in comparisons, while loads and
// stores keep them. Consequences for df64:
//   * below |x| ~ 2e-31 the lo word is lost (precision drops to 24 bits) and
//     below FLT_MIN = 1.18e-38 values become zero;
//   * a denormal hi word (1.4e-45 <= |hi| < 1.18e-38) compares EQUAL TO ZERO
//     (is_zero, eq, lt, ...) and behaves as +-0 in arithmetic (inf * x = NaN,
//     0 / x = NaN, sqrt(x) returns x unchanged), but is stored back unchanged by
//     the functions that return their input; the host encoder (encode.py) therefore
//     flushes denormal hi and lo words to +-0 before dispatch, and kernels never
//     produce a denormal hi word themselves (exp/exp2/... flush their results).
// Callers needing that range must use sf64.
//
// Part of Optiland-Metal (MIT). Structure and constants inspired by Thall 2006,
// QD (BSD), and zapccu/Metal64 (MIT); all arithmetic here is written fresh.

#ifndef OPTILAND_DF64_CORE_H
#define OPTILAND_DF64_CORE_H

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

#include <metal_stdlib>
using namespace metal;

struct df64 {
    float hi;
    float lo;
};

// Range constants (exact float32 literals).
static constant float DF64_FLT_MAX_F = 3.4028234663852886e38f;   // (2 - 2^-23) 2^127
static constant float DF64_TWO_P127_F = 0x1p127f;
static constant float DF64_TWO_P104_F = 0x1p104f;  // ulp(FLT_MAX)
static constant float DF64_TWO_P103_F = 0x1p103f;  // ulp(FLT_MAX) / 2: the overflow tie
static constant float DF64_TWO_P102_F = 0x1p102f;
// Dividends below this magnitude form their division residual on a 2^64-scaled
// copy (the scaled dividend is then below 2^-14 and q1 below 2^112 for any
// normal divisor, so nothing overflows). 2^-78 is the magnitude above which the
// df64 error terms of the residual are always normal floats (df64_core.h
// header, absolute error floor); with the previous 2^-100 the residual of a
// dividend in [2^-100, 2^-78) could flush below FLT_MIN and the quotient
// (of any size) inherited FLT_MIN / |a| relative error.
static constant float DF64_DIV_SMALL_F = 0x1p-78f;

namespace df {

// ---------------------------------------------------------------------------
// Construction and normalization
// ---------------------------------------------------------------------------
inline df64 make(float h, float l) { df64 r; r.hi = h; r.lo = l; return r; }
inline df64 make(float h) { return make(h, 0.0f); }
inline df64 zero() { return make(0.0f, 0.0f); }
inline df64 one() { return make(1.0f, 0.0f); }
inline df64 nan() { return make(NAN, 0.0f); }
inline df64 inf() { return make(INFINITY, 0.0f); }

// Keep lo = 0 whenever hi is not finite so that inf/NaN never leave a NaN lo
// behind (inf - inf inside TwoSum) and later comparisons on hi stay meaningful.
inline df64 fix(df64 a) { return make(a.hi, isfinite(a.hi) ? a.lo : 0.0f); }

// Fast2Sum (Dekker): exact when |a| >= |b| (or a == 0). 3 flops.
inline df64 quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return make(s, e);
}

// 2Sum (Knuth/Moller): exact for any a, b. 6 flops.
inline df64 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    float e = (a - (s - bb)) + (b - bb);
    return make(s, e);
}

// Fast2Mult: exact product error via fused multiply-add. Exact unless a*b underflows.
inline df64 two_prod(float a, float b) {
    float p = a * b;
    float e = fma(a, b, -p);
    return make(p, e);
}

inline df64 two_sqr(float a) {
    float p = a * a;
    float e = fma(a, a, -p);
    return make(p, e);
}

// Renormalize an arbitrary pair (|l| <= |h| or h == 0) into RNE-canonical form
// (|lo| <= ulp(hi)/2, even hi on ties); +-inf/NaN get lo = 0. A pair whose value
// reaches the overflow tie (FLT_MAX, +-2^103) renormalizes to +-inf (overflow rule).
// A zero pair keeps the sign of its hi word: (-0, +0) is the canonical df64 -0
// and quick_two_sum(-0, +0) would turn it into +0 under RNE.
inline df64 renorm(float h, float l) {
    if (h == 0.0f && l == 0.0f) return make(h, 0.0f);
    return fix(quick_two_sum(h, l));
}
inline df64 renorm(df64 a) { return renorm(a.hi, a.lo); }
// Canonical form of a pair that already satisfies |lo| <= ulp(hi)/2 (turns the
// odd-hi tie form into the even-hi form; identity on canonical pairs, including
// the canonical -0 = (-0, +0)).
inline df64 canon(df64 a) { return renorm(a); }

// Scaling a finite canonical pair by 2^e when hi * 2^e overflows: the value
// 2^e (hi + lo) is still finite when |hi| 2^(e-1) == 2^127 exactly and lo (of
// the opposite sign) pulls it below the overflow tie FLT_MAX + 2^103; the
// result is then (FLT_MAX, 2^104 - |lo| 2^e) with the sign of hi, otherwise +-inf.
inline df64 scaled_overflow(df64 a, int e) {
    float hh = metal::ldexp(a.hi, e - 1);
    if (metal::abs(hh) == DF64_TWO_P127_F && a.lo != 0.0f &&
        metal::signbit(a.lo) != metal::signbit(a.hi)) {
        float l = metal::ldexp(metal::abs(a.lo), e);  // exact, <= 2^104
        if (l > DF64_TWO_P103_F) {
            return make(metal::copysign(DF64_FLT_MAX_F, a.hi),
                        metal::copysign(DF64_TWO_P104_F - l, a.hi));
        }
    }
    return make(metal::copysign(INFINITY, a.hi), 0.0f);
}
// A scaled pair whose hi word lands exactly on +-FLT_MAX with lo = +-2^103 of
// the same sign is the overflow tie itself (it arises from the tie form
// (FLT_MAX/2 = 0x1.fffffep+126, 2^102) of 2^127 - 2^102, a valid input, scaled
// by 2); the overflow rule makes it +-inf, like the even-hi form of the same
// value (2^127, -2^102) scaled by 2 (whose hi overflows in ldexp itself).
inline df64 scaled_tie_check(float h, float l) {
    if (metal::abs(h) == DF64_FLT_MAX_F && metal::abs(l) == DF64_TWO_P103_F &&
        signbit(l) == signbit(h)) {
        return make(metal::copysign(INFINITY, h), 0.0f);
    }
    return fix(make(h, l));
}
// Exact power-of-two scaling (no rounding unless over/underflow). Overflow
// follows the overflow rule above; underflow flushes to a signed zero (the GPU
// ldexp never emits a denormal, measured); an infinite hi keeps lo = 0.
inline df64 ldexp(df64 a, int e) {
    float h = metal::ldexp(a.hi, e);
    if (!isfinite(h) && isfinite(a.hi)) return scaled_overflow(a, e);
    return scaled_tie_check(h, metal::ldexp(a.lo, e));
}
inline df64 mul_pwr2(df64 a, float p) {
    float h = a.hi * p;
    if (!isfinite(h) && isfinite(a.hi)) {
        int e;
        metal::frexp(p, e);  // p = 2^(e-1)
        return scaled_overflow(a, e - 1);
    }
    return scaled_tie_check(h, a.lo * p);
}

inline df64 from_float(float a) { return make(a, 0.0f); }
// Exact for |a| < 2^48. The residual is formed in 64-bit integer arithmetic:
// float(a) may round up to 2^31 for a in [2^31 - 64, 2^31), whose int()
// conversion saturates on Metal (from_int used to be off by one there).
inline df64 from_long(long a) {
    float h = float(a);
    float l = float(a - long(h));
    return quick_two_sum(h, l);
}
inline df64 from_int(int a) { return from_long(long(a)); }
// Nearest float32 to hi + lo (hi is already that when normalized).
inline float to_float(df64 a) { return a.hi + a.lo; }

// Four-word accumulator (x + y + z + w, kept roughly normalized by a 2Sum
// cascade). Adding a float costs six 2Sum and ONE rounding, at the fourth word:
// its error is <= 2^-24 ulp(third word) ~ 2^-96 of the second word's magnitude
// before the addition, i.e. <= ~2^-97 |A| when the accumulator holds a df64 A
// whose leading word has just been cancelled. Used by fmod and the trig argument
// reduction, where exact products / residuals must be summed far below u^2 while
// large leading terms cancel. Start with acc4_init(hi, lo).
inline float4 acc4_init(float h, float l) { return float4(h, l, 0.0f, 0.0f); }
inline float4 acc4_add(float4 s, float v) {
    df64 a = two_sum(s.x, v);
    df64 b = two_sum(s.y, a.lo);
    df64 c = two_sum(s.z, b.lo);
    float d = s.w + c.lo;               // the only rounding
    df64 t = two_sum(a.hi, b.hi);       // renormalize: leading word first
    df64 u = two_sum(t.lo, c.hi);
    df64 w = two_sum(u.lo, d);
    return float4(t.hi, u.hi, w.hi, w.lo);
}
// Round a four-word accumulator to a canonical df64.
inline df64 acc4_to_df64(float4 s) {
    df64 t = two_sum(s.x, s.y);
    df64 u = two_sum(t.lo, s.z);
    float l = u.hi + (u.lo + s.w);
    return fix(quick_two_sum(t.hi, l));
}

// ---------------------------------------------------------------------------
// Sign, magnitude, classification
// ---------------------------------------------------------------------------
inline bool is_nan(df64 a) { return isnan(a.hi) || isnan(a.lo); }
inline bool is_inf(df64 a) { return isinf(a.hi); }
inline bool is_finite(df64 a) { return isfinite(a.hi) && isfinite(a.lo); }
inline bool is_zero(df64 a) { return a.hi == 0.0f && a.lo == 0.0f; }
// True sign of the value (lo decides only when hi == 0, which never happens in
// canonical form unless the value is zero; kept for robustness with raw pairs).
inline bool is_negative(df64 a) { return a.hi < 0.0f || (a.hi == 0.0f && a.lo < 0.0f); }
inline bool is_positive(df64 a) { return a.hi > 0.0f || (a.hi == 0.0f && a.lo > 0.0f); }
inline bool signbit_df(df64 a) { return signbit(a.hi); }

inline df64 neg(df64 a) { return make(-a.hi, -a.lo); }
// abs(-0) = +0 (IEEE), so copysign(-0, +y) = +0 as well; the sign bit of a NaN
// is cleared too (numpy/torch fabs), so copysign sets it rather than flips it.
inline df64 abs(df64 a) {
    if (is_negative(a)) return neg(a);
    if (is_zero(a)) return zero();
    return make(metal::abs(a.hi), a.lo);
}
inline df64 copysign(df64 a, df64 b) { return signbit(b.hi) ? neg(abs(a)) : abs(a); }
inline df64 copysign(df64 a, float b) { return signbit(b) ? neg(abs(a)) : abs(a); }
// -1, 0, +1 (NaN -> NaN) as df64, matching torch.sign / numpy.sign.
inline df64 sign(df64 a) {
    if (is_nan(a)) return nan();
    if (is_negative(a)) return make(-1.0f, 0.0f);
    if (is_positive(a)) return one();
    return make(a.hi, 0.0f);  // preserves +-0
}

// ---------------------------------------------------------------------------
// Comparison (NaN compares false for everything except ne). Operands are
// canonicalized first so that the two representations of a tie value compare
// equal (x == x + 0 for host-encoded x); +-0 compare equal through hi.
// ---------------------------------------------------------------------------
inline bool eq(df64 a, df64 b) {
    a = canon(a);
    b = canon(b);
    return a.hi == b.hi && a.lo == b.lo;
}
inline bool ne(df64 a, df64 b) { return !(eq(a, b)); }
inline bool lt(df64 a, df64 b) {
    a = canon(a);
    b = canon(b);
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
inline bool gt(df64 a, df64 b) { return lt(b, a); }
inline bool le(df64 a, df64 b) {
    a = canon(a);
    b = canon(b);
    return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo);
}
inline bool ge(df64 a, df64 b) { return le(b, a); }

// torch.minimum/maximum propagate NaN; fmin/fmax ignore NaN. On a +-0 tie the
// minimum is -0 and the maximum is +0 (IEEE 754-2019 minimum/maximum, numpy).
inline bool zero_tie_min(df64 a, df64 b) { return a.hi == 0.0f && b.hi == 0.0f && signbit(b.hi); }
inline bool zero_tie_max(df64 a, df64 b) { return a.hi == 0.0f && b.hi == 0.0f && !signbit(b.hi); }
inline df64 minimum(df64 a, df64 b) {
    if (is_nan(a) || is_nan(b)) return nan();
    return (lt(b, a) || zero_tie_min(a, b)) ? b : a;
}
inline df64 maximum(df64 a, df64 b) {
    if (is_nan(a) || is_nan(b)) return nan();
    return (gt(b, a) || zero_tie_max(a, b)) ? b : a;
}
inline df64 fmin(df64 a, df64 b) {
    if (is_nan(a)) return b;
    if (is_nan(b)) return a;
    return (lt(b, a) || zero_tie_min(a, b)) ? b : a;
}
inline df64 fmax(df64 a, df64 b) {
    if (is_nan(a)) return b;
    if (is_nan(b)) return a;
    return (gt(b, a) || zero_tie_max(a, b)) ? b : a;
}

// ---------------------------------------------------------------------------
// Addition and subtraction
// ---------------------------------------------------------------------------
// Sign of an exact-zero sum (IEEE: -0 + -0 = -0, everything else +0). The
// float32 sum of the hi words decides only when it is itself zero; when equal
// values in different representations cancel (odd-hi tie form against even-hi
// form) the hi words differ by one ulp and the result is still +0.
inline df64 exact_zero_sum(float hsum) { return make(hsum == 0.0f ? hsum : 0.0f, 0.0f); }

// AccurateDWPlusDW (JMP2017 Alg. 6, QD ieee_add): relative error <= 3u^2 + 13u^3.
// Finite operands; an overflow in any recombination step (the 2Sum of the hi
// words, or a Fast2Sum whose lo word pushes the sum over the tie) leaves hi
// infinite, which the callers detect.
inline df64 add_core(df64 a, df64 b) {
    df64 s = two_sum(a.hi, b.hi);
    df64 t = two_sum(a.lo, b.lo);
    float c = s.lo + t.hi;
    df64 v = quick_two_sum(s.hi, c);
    float w = t.lo + v.lo;
    return quick_two_sum(v.hi, w);
}
// Sum of two finite df64 whose recombination overflowed. The sum is redone on
// the halved operands (exact: the lo words of operands this large are normal).
// When the halved result rounds to +-2^127 the doubled value lies within the
// algorithm's own rounding error (3 u^2) of the overflow tie FLT_MAX + 2^103,
// so the threshold is decided from the EXACT residual delta = (a/2 + b/2) - r
// (a four-word accumulator; all terms are exact 2Sum pairs) and the finite
// result (FLT_MAX, 2^104 + 2 r.lo + 2 delta) is rebuilt directly.
inline df64 add_overflow(df64 a, df64 b) {
    df64 ha = mul_pwr2(a, 0.5f);
    df64 hb = mul_pwr2(b, 0.5f);
    df64 r = add_core(ha, hb);
    if (metal::abs(r.hi) != DF64_TWO_P127_F) return mul_pwr2(r, 2.0f);
    float s = metal::copysign(1.0f, r.hi);
    float4 acc = acc4_init(ha.hi, ha.lo);
    acc = acc4_add(acc, hb.hi);
    acc = acc4_add(acc, hb.lo);
    acc = acc4_add(acc, -r.hi);
    acc = acc4_add(acc, -r.lo);
    // |value|/2 < 2^127 - 2^102  <=>  s (r.lo + delta) < -2^102
    float bound = -DF64_TWO_P102_F - s * r.lo;  // exact (multiples of 2^79)
    float d = s * acc.x;
    bool below = d < bound || (d == bound && s * acc.y < 0.0f);
    if (!below) return make(s * INFINITY, 0.0f);
    float lo = (s * DF64_TWO_P104_F + 2.0f * r.lo) + 2.0f * acc.x;
    lo += 2.0f * acc.y;
    if (metal::abs(lo) >= DF64_TWO_P103_F) return make(s * INFINITY, 0.0f);
    return make(s * DF64_FLT_MAX_F, lo);
}
inline df64 add(df64 a, df64 b) {
    if (!isfinite(a.hi) || !isfinite(b.hi)) return make(a.hi + b.hi, 0.0f);
    df64 r = add_core(a, b);
    if (!isfinite(r.hi)) return add_overflow(a, b);
    if (r.hi == 0.0f && r.lo == 0.0f) return exact_zero_sum(a.hi + b.hi);
    return r;
}
// DWPlusFP (JMP2017 Alg. 4): relative error <= 2u^2.
inline df64 add(df64 a, float b) {
    if (!isfinite(a.hi) || !isfinite(b)) return make(a.hi + b, 0.0f);
    df64 s = two_sum(a.hi, b);
    float v = a.lo + s.lo;
    df64 r = quick_two_sum(s.hi, v);
    if (!isfinite(r.hi)) return add_overflow(a, make(b));
    if (r.hi == 0.0f && r.lo == 0.0f) return exact_zero_sum(a.hi + b);
    return r;
}
inline df64 add(float a, df64 b) { return add(b, a); }
inline df64 sub(df64 a, df64 b) { return add(a, neg(b)); }
inline df64 sub(df64 a, float b) { return add(a, -b); }
inline df64 sub(float a, df64 b) { return add(neg(b), a); }

// ---------------------------------------------------------------------------
// Multiplication
// ---------------------------------------------------------------------------
// DWTimesDW3 (JMP2017 Alg. 12, includes lo*lo): relative error < 5u^2.
// Finite operands whose hi product is finite and nonzero; the result may still
// overflow in the final Fast2Sum (hi infinite).
inline df64 mul_core(df64 a, df64 b) {
    df64 p = two_prod(a.hi, b.hi);
    float t = a.lo * b.lo;
    float u = fma(a.hi, b.lo, t);
    float v = fma(a.lo, b.hi, u);
    float w = p.lo + v;
    return quick_two_sum(p.hi, w);
}
// A zero product with a zero operand returns the float32 product so that the
// IEEE sign of the zero is kept (-0 * 5 = -0). When a.hi * b.hi overflows or is
// flushed to zero although both operands are nonzero, the product is redone on
// a 2^-1 resp. 2^48-scaled operand (exact) and scaled back (overflow /
// underflow rules in the header comment). When the scaled retry flushes as
// well (exact product below ~2^-174) the result is the hardware product p,
// which carries the sign of the exact result (mul_core would return +0 for a
// negative product: quick_two_sum(-0, +0) = +0).
inline df64 mul(df64 a, df64 b) {
    if (!isfinite(a.hi) || !isfinite(b.hi)) return make(a.hi * b.hi, 0.0f);
    float p = a.hi * b.hi;
    if (p == 0.0f) {
        if (a.hi == 0.0f || b.hi == 0.0f) return make(p, 0.0f);
        df64 r = mul_pwr2(mul_core(mul_pwr2(a, 0x1p48f), b), 0x1p-48f);
        return r.hi == 0.0f ? make(p, 0.0f) : r;
    }
    if (!isfinite(p)) {
        df64 r = mul_core(mul_pwr2(a, 0.5f), b);
        if (!isfinite(r.hi)) return make(p, 0.0f);
        return mul_pwr2(r, 2.0f);
    }
    return fix(mul_core(a, b));
}
// DWTimesFP3 (JMP2017 Alg. 9): relative error <= 2u^2. Same guards as above.
inline df64 mul_core(df64 a, float b) {
    df64 p = two_prod(a.hi, b);
    float w = fma(a.lo, b, p.lo);
    return quick_two_sum(p.hi, w);
}
inline df64 mul(df64 a, float b) {
    if (!isfinite(a.hi) || !isfinite(b)) return make(a.hi * b, 0.0f);
    float p = a.hi * b;
    if (p == 0.0f) {
        if (a.hi == 0.0f || b == 0.0f) return make(p, 0.0f);
        df64 r = mul_pwr2(mul_core(mul_pwr2(a, 0x1p48f), b), 0x1p-48f);
        return r.hi == 0.0f ? make(p, 0.0f) : r;
    }
    if (!isfinite(p)) {
        df64 r = mul_core(mul_pwr2(a, 0.5f), b);
        if (!isfinite(r.hi)) return make(p, 0.0f);
        return mul_pwr2(r, 2.0f);
    }
    return fix(mul_core(a, b));
}
inline df64 mul(float a, df64 b) { return mul(b, a); }
inline df64 sqr_core(df64 a) {
    df64 p = two_sqr(a.hi);
    float t = a.lo * a.lo;
    float u = fma(2.0f * a.hi, a.lo, t);
    float w = p.lo + u;
    return quick_two_sum(p.hi, w);
}
inline df64 sqr(df64 a) {
    if (!isfinite(a.hi)) return make(a.hi * a.hi, 0.0f);
    float p = a.hi * a.hi;
    if (p == 0.0f) {
        if (a.hi == 0.0f) return make(p, 0.0f);
        return mul_pwr2(sqr_core(mul_pwr2(a, 0x1p24f)), 0x1p-48f);
    }
    if (!isfinite(p)) {
        df64 r = sqr_core(mul_pwr2(a, 0.5f));
        if (!isfinite(r.hi)) return make(p, 0.0f);
        return mul_pwr2(r, 4.0f);
    }
    return fix(sqr_core(a));
}
// mul then add (two roundings; NOT a single-rounding fma).
inline df64 mul_add(df64 a, df64 b, df64 c) { return add(mul(a, b), c); }

// ---------------------------------------------------------------------------
// Division, reciprocal, square root
// ---------------------------------------------------------------------------
// QD accurate division (long-division form; JMP2017 Alg. 17 bound 15u^2, observed ~2u^2).
// The dividend is rescaled (exactly) in three situations and the quotient
// scaled back at the end (mul_pwr2 applies the overflow / underflow rules):
//   * a.hi / b.hi overflows although the quotient may be representable: a/2;
//   * |a.hi| < 2^-78 (DF64_DIV_SMALL_F), or a.hi / b.hi was flushed to zero
//     (FTZ before rounding): a * 2^64, so that neither the residual a - b q1
//     (which could be flushed to zero, making every refinement add q1 again)
//     nor its error terms (the lo word of b * q1, the two_prod residual of the
//     df64/float overload) fall below FLT_MIN; the scaled dividend is below
//     2^-14 and q1 < 2^112 there since b.hi >= FLT_MIN, so nothing overflows.
//     A quotient of ordinary size from a dividend in [2^-100, 2^-78) used to
//     carry FLT_MIN / |a| relative error (up to 2^-26) whenever those terms
//     flushed (23% of the dividends near 2^-99);
//   * a.hi within one ulp of FLT_MAX where b * q1 overflows: a/2 as well.
inline df64 div(df64 a, df64 b) {
    // IEEE float semantics for inf/NaN operands and zero divisors (x/0 = +-inf, 0/0 = NaN).
    if (!isfinite(a.hi) || !isfinite(b.hi) || b.hi == 0.0f) return make(a.hi / b.hi, 0.0f);
    if (a.hi == 0.0f) return make(a.hi / b.hi, 0.0f);  // signed zero
    float q1 = a.hi / b.hi;
    df64 aa = a;
    float scale = 1.0f;
    if (!isfinite(q1)) {
        aa = mul_pwr2(a, 0.5f);
        q1 = aa.hi / b.hi;
        if (!isfinite(q1)) return make(q1, 0.0f);  // true overflow
        scale = 2.0f;
    } else if (q1 == 0.0f || metal::abs(a.hi) < DF64_DIV_SMALL_F) {
        aa = mul_pwr2(a, 0x1p64f);
        q1 = aa.hi / b.hi;
        scale = 0x1p-64f;
    }
    df64 bq = mul(b, q1);
    if (!isfinite(bq.hi)) {
        aa = mul_pwr2(aa, 0.5f);
        q1 *= 0.5f;
        bq = mul(b, q1);
        scale *= 2.0f;
    }
    df64 r = sub(aa, bq);
    float q2 = r.hi / b.hi;
    r = sub(r, mul(b, q2));
    float q3 = r.hi / b.hi;
    df64 q = quick_two_sum(q1, q2);
    return mul_pwr2(add(q, q3), scale);
}
// DWDivFP3 (JMP2017 Alg. 15): relative error <= 3u^2. Same rescaling as above.
inline df64 div(df64 a, float b) {
    if (!isfinite(a.hi) || !isfinite(b) || b == 0.0f) return make(a.hi / b, 0.0f);
    if (a.hi == 0.0f) return make(a.hi / b, 0.0f);  // signed zero
    float th = a.hi / b;
    df64 aa = a;
    float scale = 1.0f;
    if (!isfinite(th)) {
        aa = mul_pwr2(a, 0.5f);
        th = aa.hi / b;
        if (!isfinite(th)) return make(th, 0.0f);  // true overflow
        scale = 2.0f;
    } else if (th == 0.0f || metal::abs(a.hi) < DF64_DIV_SMALL_F) {
        aa = mul_pwr2(a, 0x1p64f);
        th = aa.hi / b;
        scale = 0x1p-64f;
    }
    df64 p = two_prod(th, b);
    if (!isfinite(p.hi)) {  // th * b rounded above FLT_MAX: redo the residual on a/2
        aa = mul_pwr2(aa, 0.5f);
        th *= 0.5f;
        p = two_prod(th, b);
        scale *= 2.0f;
    }
    float dh = aa.hi - p.hi;
    float dt = dh - p.lo;
    float d = dt + aa.lo;
    float tl = d / b;
    return mul_pwr2(quick_two_sum(th, tl), scale);
}
inline df64 div(float a, df64 b) { return div(make(a), b); }
inline df64 recip(df64 a) { return div(one(), a); }

// Karp/QD square root with a precise rsqrt seed and one Newton correction
// (result correct to ~48 bits), then a second correction for safety.
// sqrt(+-0) = +-0, sqrt(negative) = NaN, sqrt(inf) = inf.
inline df64 sqrt(df64 a) {
    if (is_zero(a)) return a;
    if (is_negative(a)) return nan();
    if (!isfinite(a.hi)) return fix(a);
    // Prescale into [0.5, 2) by an even power of two (exact) so that the error
    // terms of the squares below never fall into the flushed denormal range.
    int e;
    float m = metal::frexp(a.hi, e);          // a.hi = m * 2^e, m in [0.5, 1)
    if (e & 1) { m *= 2.0f; e -= 1; }         // make e even
    df64 s = make(m, metal::ldexp(a.lo, -e));
    float x = precise::rsqrt(s.hi);
    float ax = s.hi * x;
    df64 y = add(make(ax), (sub(s, two_sqr(ax))).hi * (x * 0.5f));
    // second correction: y += (s - y^2) / (2y)
    df64 r = sub(s, sqr(y));
    y = add(y, r.hi / (2.0f * y.hi));
    return fix(ldexp(y, e / 2));
}
inline df64 rsqrt(df64 a) { return div(one(), sqrt(a)); }

// ---------------------------------------------------------------------------
// Rounding to integers (results are df64; exact). +-inf, NaN and +-0 are
// returned unchanged (quick_two_sum(inf, 0) would leave lo = NaN and
// quick_two_sum(-0, +0) = +0). Operands are canonicalized first so that the
// tie form (odd hi, lo = +-ulp/2) rounds like its value (rint half-to-even).
// A zero result keeps the sign of the input: ceil(-0.999...) = -0, trunc too.
// ---------------------------------------------------------------------------
inline bool round_passthrough(df64 a) { return !isfinite(a.hi) || a.hi == 0.0f; }
inline df64 signed_zero_like(df64 a) { return make(metal::copysign(0.0f, a.hi), 0.0f); }
inline df64 floor(df64 a) {
    if (round_passthrough(a)) return fix(make(a.hi, 0.0f));
    a = canon(a);
    float h = metal::floor(a.hi);
    df64 r = (h == a.hi) ? quick_two_sum(h, metal::floor(a.lo)) : make(h, 0.0f);
    return r.hi == 0.0f ? signed_zero_like(a) : r;
}
inline df64 ceil(df64 a) {
    if (round_passthrough(a)) return fix(make(a.hi, 0.0f));
    a = canon(a);
    float h = metal::ceil(a.hi);
    df64 r = (h == a.hi) ? quick_two_sum(h, metal::ceil(a.lo)) : make(h, 0.0f);
    return r.hi == 0.0f ? signed_zero_like(a) : r;
}
inline df64 trunc(df64 a) { return is_negative(a) ? ceil(a) : floor(a); }
// Round half to even (torch.round / numpy.round semantics).
inline df64 rint(df64 a) {
    if (round_passthrough(a)) return fix(make(a.hi, 0.0f));
    a = canon(a);
    float h = metal::rint(a.hi);
    if (h == a.hi) return quick_two_sum(h, metal::rint(a.lo));
    // hi is not integral: the fractional part lives entirely in hi, and lo can
    // only tip an exact .5 tie. Detect the tie and use lo's sign.
    if (metal::abs(a.hi - h) == 0.5f) {
        if (a.lo == 0.0f) return make(h, 0.0f);          // true tie: rint(hi) already even
        return make(a.lo > 0.0f ? metal::ceil(a.hi) : metal::floor(a.hi), 0.0f);
    }
    return make(h, 0.0f);
}
// Round half away from zero (C round). Exact: when hi is integral the fraction
// lives in lo (a half can only occur for |hi| >= 2^23, where |lo| <= ulp(hi)/2
// reaches 1/2) and floor(|lo| + 1/2) is formed without rounding (the previous
// add(|a|, 1/2) rounded a 24-bit lo word plus 1/2 in float32 and returned
// x + 1 for odd lo words in [2^23, 2^24)); when hi is not integral the fraction
// is in hi and lo can only tip an exact .5 tie.
inline df64 round_away(df64 a) {
    if (round_passthrough(a)) return fix(make(a.hi, 0.0f));
    a = canon(a);
    bool negative = signbit(a.hi);
    float h = metal::trunc(a.hi);
    df64 r;
    if (h == a.hi) {
        float l = negative ? -a.lo : a.lo;  // |value| = |hi| + l
        float f = metal::floor(l);
        float d = l - f;                    // exact, in [0, 1)
        float n = (d >= 0.5f) ? f + 1.0f : f;
        r = quick_two_sum(a.hi, negative ? -n : n);
    } else {
        float rr = metal::round(a.hi);      // half away from zero
        if (metal::abs(a.hi - h) == 0.5f && a.lo != 0.0f && (signbit(a.lo) != negative)) {
            rr = h;                         // lo pulls |value| below the tie
        }
        r = make(rr, 0.0f);
    }
    return r.hi == 0.0f ? signed_zero_like(a) : r;
}
// Nearest integer as float; helper for argument reductions (|x| < 2^24).
inline float nint(float a) { return metal::rint(a); }

// Truncation toward zero of hi + lo, saturating to INT_MIN / INT_MAX beyond the
// int32 range (also for +-inf); NaN -> 0. Goes through the exact df64 trunc and
// a 64-bit sum so that values just below an integer (hi integral, lo < 0)
// truncate down and hi = 2^31 with a negative lo does not saturate.
inline int to_int(df64 a) {
    if (isnan(a.hi)) return 0;
    if (a.hi >= 4294967296.0f) return INT_MAX;
    if (a.hi <= -4294967296.0f) return INT_MIN;
    df64 t = trunc(a);
    long v = long(t.hi) + long(t.lo);
    if (v > (long)INT_MAX) return INT_MAX;
    if (v < (long)INT_MIN) return INT_MIN;
    return int(v);
}

// fmod with C99 semantics: result has the sign of a (also when zero), |result| < |b|;
// fmod(+-0, y) = +-0, fmod(x, +-inf) = x, fmod(inf, y) = fmod(x, 0) = NaN.
//
// Staged reduction (mod_reduce): with A = |a|, B = |b|, each round scales B by a
// power of two (exact) so that A / B' < 2^21, takes the integer quotient q of
// the hi words, forms q * B' EXACTLY as two 2Prod pairs and subtracts them from
// a four-word accumulator that carries A across the rounds (the only roundings
// are at the fourth word, <= ~2^-97 |A| per round); a +-B' correction inside the
// same accumulator absorbs the +-1 uncertainty of the float quotient. Each round
// removes >= 19 bits of the quotient, so the loop ends after <= 15 rounds for
// any float32 exponent gap (no overflow of a / b is ever formed). The sign and
// magnitude invariants therefore hold for every finite a and b != 0, and the
// value is the df64 rounding of the exact remainder (<= 0.5 u^2 of |b|,
// measured for |a / b| up to 2^70 in tests/metal/test_library.py).
//
// FTZ: whenever the running remainder A.hi drops below 2^-20 the whole state
// (accumulator, A and B) is scaled up by an exact power of two so that the
// accumulator words (~2^-96 of A) and the 2Prod residuals of q * B' never fall
// below FLT_MIN (fmod is scale invariant; B <= A there, so nothing overflows).
// The caller receives the scaled remainder, the scaled B and the total shift
// esc, and undoes it once with ldexp (a remainder below FLT_MIN flushes to
// +-0). Without this, divisors below ~2^-77 lost accuracy (|b| ~ 2^-104:
// float32 level; |b| < 2^-122: results of 0 or O(|b|)).
inline df64 mod_reduce(df64 A, thread df64 &B, thread float4 &s, thread int &esc) {
    esc = 0;
    s = acc4_init(A.hi, A.lo);
    for (int round = 0; round < 24 && !lt(A, B); ++round) {
        int ea;
        int eb;
        metal::frexp(A.hi, ea);
        if (ea < -20) {
            int k = 20 - ea;
            s = float4(metal::ldexp(s.x, k), metal::ldexp(s.y, k),
                       metal::ldexp(s.z, k), metal::ldexp(s.w, k));
            A = ldexp(A, k);
            B = ldexp(B, k);
            esc += k;
            ea = 20;
        }
        metal::frexp(B.hi, eb);
        int sh = ea - eb - 20;
        if (sh < 0) sh = 0;
        df64 Bs = ldexp(B, sh);                     // exact: Bs.hi <= 2^(ea - 20)
        float q = metal::floor(A.hi / Bs.hi);       // within +-1 of trunc(A / Bs)
        if (q < 1.0f) q = 1.0f;                     // A >= Bs here (sh == 0 only)
        df64 p1 = two_prod(q, Bs.hi);
        if (!isfinite(p1.hi)) {                     // A.hi ~ FLT_MAX and q one too large
            q -= 1.0f;
            p1 = two_prod(q, Bs.hi);
        }
        df64 p2 = two_prod(q, Bs.lo);
        s = acc4_add(s, -p1.hi);
        s = acc4_add(s, -p1.lo);
        s = acc4_add(s, -p2.hi);
        s = acc4_add(s, -p2.lo);
        A = acc4_to_df64(s);
        if (is_negative(A)) {                       // q overshot by one
            s = acc4_add(s, Bs.hi);
            s = acc4_add(s, Bs.lo);
            A = acc4_to_df64(s);
        } else if (!lt(A, Bs)) {                    // q undershot by one
            s = acc4_add(s, -Bs.hi);
            s = acc4_add(s, -Bs.lo);
            A = acc4_to_df64(s);
        }
    }
    if (is_negative(A)) {                           // final rounding guards
        s = acc4_add(s, B.hi);
        s = acc4_add(s, B.lo);
        A = acc4_to_df64(s);
    } else if (!lt(A, B)) {
        s = acc4_add(s, -B.hi);
        s = acc4_add(s, -B.lo);
        A = acc4_to_df64(s);
    }
    return A;
}
inline df64 fmod(df64 a, df64 b) {
    if (is_nan(a) || is_nan(b) || is_inf(a) || is_zero(b)) return nan();
    if (is_inf(b) || is_zero(a)) return make(a.hi, a.lo);
    float4 s;
    int esc;
    df64 B = abs(b);
    df64 R = mod_reduce(abs(a), B, s, esc);
    if (is_zero(R) || is_negative(R)) return signed_zero_like(a);
    R = ldexp(R, -esc);
    if (R.hi == 0.0f) return signed_zero_like(a);  // remainder below FLT_MIN
    return signbit(a.hi) ? neg(R) : R;
}
// Python/numpy/torch remainder: result has the sign of b (also when zero);
// remainder(x, +-inf) = x when the signs agree, +-inf otherwise; NaN like fmod.
// Same reduction as fmod; when the signs of a and b differ the result is
// sign(b) (|b| - R) formed inside the accumulator (exact to df64 rounding). As
// in numpy, |b| - R rounds to |b| itself when R < 2^-49 |b|
// (np.remainder(-1e-20, 1.0) == 1.0); otherwise |result| < |b|.
inline df64 remainder_py(df64 a, df64 b) {
    if (is_nan(a) || is_nan(b) || is_inf(a) || is_zero(b)) return nan();
    if (is_zero(a)) return make(metal::copysign(0.0f, b.hi), 0.0f);
    if (is_inf(b)) return is_negative(a) == is_negative(b) ? make(a.hi, a.lo) : b;
    float4 s;
    int esc;
    df64 B = abs(b);
    df64 R = mod_reduce(abs(a), B, s, esc);  // R and B are scaled by 2^esc
    if (is_zero(R) || is_negative(R)) return make(metal::copysign(0.0f, b.hi), 0.0f);
    if (signbit(a.hi) != signbit(b.hi)) {
        s = -s;
        s = acc4_add(s, B.hi);
        s = acc4_add(s, B.lo);
        R = acc4_to_df64(s);
        if (is_zero(R) || is_negative(R)) return make(metal::copysign(0.0f, b.hi), 0.0f);
        if (!lt(R, B)) R = B;
    }
    R = ldexp(R, -esc);
    if (R.hi == 0.0f) return make(metal::copysign(0.0f, b.hi), 0.0f);
    return signbit(b.hi) ? neg(R) : R;
}

// Decomposition x = m 2^e with |m| in [0.5, 1) like numpy.frexp (ldexp is
// defined next to mul_pwr2 above): when hi is a power of two and lo pulls the
// value below it (e.g. 1 - 2^-25 = (1, -2^-25)) the float mantissa 0.5 is
// doubled and the exponent decremented. +-0 return themselves (sign kept:
// metal::frexp(-0) gives +0) with e = 0.
inline df64 frexp(df64 a, thread int &e) {
    if (a.hi == 0.0f) {
        e = 0;
        return make(a.hi, 0.0f);
    }
    float h = metal::frexp(a.hi, e);
    float l = metal::ldexp(a.lo, -e);
    if (metal::abs(h) == 0.5f && l != 0.0f && ((l < 0.0f) == (h > 0.0f))) {
        h *= 2.0f;
        l *= 2.0f;
        e -= 1;
    }
    return make(h, l);
}

}  // namespace df

// ---------------------------------------------------------------------------
// Operators (thin sugar over the namespace functions)
// ---------------------------------------------------------------------------
inline df64 operator-(df64 a) { return df::neg(a); }
inline df64 operator+(df64 a, df64 b) { return df::add(a, b); }
inline df64 operator+(df64 a, float b) { return df::add(a, b); }
inline df64 operator+(float a, df64 b) { return df::add(b, a); }
inline df64 operator-(df64 a, df64 b) { return df::sub(a, b); }
inline df64 operator-(df64 a, float b) { return df::sub(a, b); }
inline df64 operator-(float a, df64 b) { return df::sub(a, b); }
inline df64 operator*(df64 a, df64 b) { return df::mul(a, b); }
inline df64 operator*(df64 a, float b) { return df::mul(a, b); }
inline df64 operator*(float a, df64 b) { return df::mul(b, a); }
inline df64 operator/(df64 a, df64 b) { return df::div(a, b); }
inline df64 operator/(df64 a, float b) { return df::div(a, b); }
inline df64 operator/(float a, df64 b) { return df::div(a, b); }
inline bool operator==(df64 a, df64 b) { return df::eq(a, b); }
inline bool operator!=(df64 a, df64 b) { return df::ne(a, b); }
inline bool operator<(df64 a, df64 b) { return df::lt(a, b); }
inline bool operator>(df64 a, df64 b) { return df::gt(a, b); }
inline bool operator<=(df64 a, df64 b) { return df::le(a, b); }
inline bool operator>=(df64 a, df64 b) { return df::ge(a, b); }

// ---------------------------------------------------------------------------
// Startup self-test kernel: the host must see exactly these values, otherwise the
// library was compiled with fast math, contraction, or on round-toward-zero hardware.
//   out[0] = TwoSum(1, 2^-30).lo  == 2^-30 (9.313225746154785e-10)
//   out[1] = TwoProd(1+2^-23, 1+2^-23).lo == 2^-46 (1.4210854715202004e-14)
//   out[2] = 2/3 as float32       == 0x3f2aaaab (round-to-nearest-even)
//   out[3] = (1+3*2^-24) as float == 0x3f800002 (RNE tie handling)
//   out[4] = (1e-9 + 1.0) - 1.0   == 0 (no reassociation)
//   out[5] = df::div(1, 3) reconstructed error vs 1/3 (host checks < 1e-14)
// ---------------------------------------------------------------------------
kernel void df64_selftest(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    df64 s = df::two_sum(1.0f, 9.313225746154785e-10f);
    out[0] = s.lo;
    float c = 1.0f + 1.1920928955078125e-07f;  // 1 + 2^-23
    df64 p = df::two_prod(c, c);
    out[1] = p.lo;
    out[2] = 2.0f / 3.0f;
    out[3] = 1.0f + 3.0f * 5.960464477539063e-08f;
    float t = 1e-9f;
    out[4] = (t + 1.0f) - 1.0f;
    df64 third = df::div(df::one(), df::make(3.0f));
    out[5] = third.hi;
    out[6] = third.lo;
    out[7] = 1.0f;  // completion sentinel
}

#endif  // OPTILAND_DF64_CORE_H
