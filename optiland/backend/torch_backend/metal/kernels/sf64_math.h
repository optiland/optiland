// sf64_math.h — transcendental functions for sf64 (software binary64) on Metal.
//
// *** INEXACT BRIDGE (48-bit): replaced by a native port in M4. ***
//
// v1 evaluates every transcendental by converting the sf64 operand to df64
// (double-single, 48 significant bits), calling the df64 implementation, and
// converting the result back. The result is therefore rounded to ~48 bits (about
// 2^-48 = 3.6e-15 relative plus the df64 function's own error), NOT correctly
// rounded binary64. The op table flags these ops as inexact in sf64 mode
// (codegen.OpSpec.sf_inexact, library.op_machine_eps). Milestone M4 replaces this
// file with a musl-style libm port written on the sf:: arithmetic operators.
//
// Range. The df64 functions only see float32's exponent range, so the bridge
// handles binary64 arguments outside [2^-78, 2^127) itself (fix round 3; before,
// such arguments reached the df64 code as +-0 / +-inf or as a denormal hi word,
// which returned the input for cbrt, erf, expm1, ... and -inf instead of NaN
// for log of a negative tiny value):
//   * arguments are converted with sf_arg(), which flushes a float32-denormal
//     hi or lo word to a signed zero exactly like the host encoder does (the GPU
//     treats such words as zero in arithmetic anyway);
//   * f(x) = x + O(x^3) (sin, tan, asin, atan, sinh, tanh, asinh, atanh) return
//     x itself for |x| < 2^-27 and f(x) = x + O(x^2) (expm1, log1p) for
//     |x| < 2^-54: exact binary64 results (the dropped term is below 2^-54
//     relative), which also covers the whole binary64 subnormal range;
//     erf(x) = 2x/sqrt(pi) and erfinv(x) = x sqrt(pi)/2 for |x| < 2^-27 (one
//     sf64 rounding);
//   * log, log2, log10 (and log1p, lgamma, asinh, acosh above 2^127) evaluate
//     log(m 2^e) = log m + e ln2 with m in [1/2, 1) from sf::frexp and e ln2
//     formed to ~2^-70; log of a negative value (any magnitude) is NaN;
//     lgamma(|x| < 2^-126) = -log|x|; lgamma(x >= 2^127) = x (log x - 1) in sf64
//     (the neglected 1/2 log x and ln sqrt(2 pi) are below 2^-53 relative);
//   * cbrt scales by 2^(3k) and back; atan2 and hypot scale both arguments by
//     the same power of two when either is out of range (hypot scales the
//     result back with gradual underflow); hypot(NaN, finite) = NaN and
//     hypot(NaN, +-inf) = +inf are decided before the conversion (1e300 would
//     become inf);
//   * pow(a, b): the integrality / parity of b is decided on the binary64 b
//     (sf::to_float2 rounds it to 48 bits, which turned odd exponents above
//     2^48 even and half-integers into integers), a negative base is evaluated
//     on |a| with the sign applied afterwards, and an exponent below 2^-126 in
//     magnitude is replaced by +-FLT_MIN (same C99 answers for every base:
//     exp(+-2^-126 log a) rounds to 1) instead of flushing to 0 (which made
//     pow(NaN, 1e-300) = pow(0, -1e-300) = 1); a base outside [2^-78, 2^127), or
//     a df64 result that left float32's range for finite nonzero operands
//     (pow(10, 40), pow(1e-300, 0.5)), goes through exp(b (log m + e ln2)) with
//     the product formed to ~72 bits and the power of two applied in sf64,
//     which carries ~|b log a| 0.5 u^2 of relative error (the b log a
//     amplification the in-range path is documented with too).
// What remains float32-ranged (documented limits of the v1 bridge): results
// below 2^-126 in magnitude are +-0 (they are formed in df64: exp(-800) = 0,
// atan2(1e-300, 1) = 0, sin(x) for |x| < 2^-1022 aside), the lo word of a
// result below ~2e-31 is lost and results in [2^-102, 2^-78] carry the sporadic
// FLT_MIN absolute floor of df64_core.h; sin/cos/tan of |x| >= 2^127 are NaN
// (the df64 reduction has no binary64 Payne-Hanek) and for 1e7 < |x| < 2^127
// they are float32-accurate (~1e-7 absolute: the df64 fallback of
// df64_math_trig.h, on top of the cond |x| 2^-49 of the 48-bit input rounding).
//
// Amalgamation contract: this file compiles to nothing unless df64_core.h has been
// seen (OPTILAND_DF64_CORE_H); when it has, sf64_core.h must precede this file
// and each family below is emitted only when its df64 math header preceded it
// (OPTILAND_DF64_MATH_EXP_H, OPTILAND_DF64_MATH_TRIG_H, OPTILAND_DF64_MATH_SPECIAL_H):
//   vendor/softfloat64.metal -> df64_core.h -> df64_constants.h -> df64_math_exp.h
//   -> df64_math_trig.h -> df64_math_special.h -> sf64_core.h -> sf64_math.h
// The special family relies on the exp family (log_ext, prod3, exp_of_sum,
// mul3, DF64X_* constants), as df64_math_special.h itself does.
//
// Part of Optiland-Metal (MIT).

#ifndef OPTILAND_SF64_MATH_H
#define OPTILAND_SF64_MATH_H

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

#ifdef OPTILAND_DF64_CORE_H

#ifndef OPTILAND_SF64_CORE_H
#error "sf64_math.h requires sf64_core.h earlier in the amalgamation"
#endif

namespace sf {

// ---------------------------------------------------------------------------
// Range helpers
// ---------------------------------------------------------------------------
// Binary exponents (x = m 2^e, 1/2 <= |m| < 1, as sf::frexp returns them) of the
// arguments the df64 code takes as they are: |x| >= 2^-78 (below that the lo
// word of the 48-bit argument can be a flushed float32 denormal, so a function
// that depends on all bits of x, such as log or cbrt, would see a 24-bit input)
// and |x| < 2^127 (one binade below the float32 overflow tie).
constant int SF64_MATH_EMIN = -77;
constant int SF64_MATH_EMAX = 127;
// Scaled exp: exp(P + tail) is +inf above this P and +0 below the other one
// (binary64: exp(709.78) is the largest finite, exp(-745.13) the last subnormal).
constant float SF64_MATH_EXP_MAX = 710.0f;
constant float SF64_MATH_EXP_MIN = -746.0f;
// Below 2^-27 the odd functions return their argument; below 2^-54 expm1/log1p.
constant int SF64_MATH_ODD_E = -27;
constant int SF64_MATH_LINEAR_E = -54;
constant ulong SF64_2_SQRTPI_BITS = 0x3FF20DD750429B6DUL;  // 2 / sqrt(pi)
constant ulong SF64_SQRTPI_2_BITS = 0x3FEC5BF891B4EF6BUL;  // sqrt(pi) / 2

// True for a finite nonzero x, with e its frexp exponent (0 otherwise).
inline bool sf_regular(sf64 x, thread int &e) {
    e = 0;
    if (!is_finite(x) || is_zero(x)) return false;
    frexp(x, e);
    return true;
}
// Finite nonzero x with |x| < 2^emax.
inline bool sf_below(sf64 x, int emax) {
    int e;
    return sf_regular(x, e) && e <= emax;
}
// Finite nonzero x outside [2^-78, 2^127): e receives the frexp exponent.
inline bool sf_out_of_df64_range(sf64 x, thread int &e) {
    return sf_regular(x, e) && (e < SF64_MATH_EMIN || e > SF64_MATH_EMAX);
}
// Common power-of-two scale 2^k for a scale-invariant binary function (atan2,
// hypot): true when a finite nonzero operand lies outside the df64 range, with
// k chosen so that the larger finite nonzero magnitude lands in [1/2, 1) (+-0
// and +-inf operands are unaffected by the scaling).
inline bool sf_common_scale(sf64 a, sf64 b, thread int &k) {
    int ea;
    int eb;
    bool ra = sf_regular(a, ea);
    bool rb = sf_regular(b, eb);
    k = 0;
    bool oa = ra && (ea < SF64_MATH_EMIN || ea > SF64_MATH_EMAX);
    bool ob = rb && (eb < SF64_MATH_EMIN || eb > SF64_MATH_EMAX);
    if (!oa && !ob) return false;
    int em = (ra && rb) ? metal::max(ea, eb) : (ra ? ea : eb);
    k = -em;
    return true;
}
// df64 argument of the bridge: a float32-denormal hi word (|x| < 2^-126) becomes
// a signed zero and a denormal lo word +0, exactly as the host encoder does
// (df64_core.h denormal contract), so that no df64 function can return a
// denormal word unchanged.
inline df64 sf_arg(sf64 x) {
    float2 p = to_float2(x);
    if ((as_type<uint>(p.x) & 0x7F800000u) == 0u) {
        return df::make(metal::copysign(0.0f, p.x), 0.0f);
    }
    if ((as_type<uint>(p.y) & 0x7F800000u) == 0u) p.y = 0.0f;
    return df::make(p.x, p.y);
}

// One-argument bridge: sf64 -> df64 -> df::<name> -> sf64.
#define OPTILAND_SF64_BRIDGE_1(name) \
    inline sf64 name(sf64 x) { return from_df64(df::name(sf_arg(x))); }
// Two-argument bridge.
#define OPTILAND_SF64_BRIDGE_2(name) \
    inline sf64 name(sf64 x, sf64 y) { \
        return from_df64(df::name(sf_arg(x), sf_arg(y))); \
    }
// Odd function with f(x) = x + O(x^3): x itself below 2^-27 (exact binary64).
#define OPTILAND_SF64_BRIDGE_ODD(name) \
    inline sf64 name(sf64 x) { \
        if (sf_below(x, SF64_MATH_ODD_E)) return x; \
        return from_df64(df::name(sf_arg(x))); \
    }

// Exponential and logarithm family (df64_math_exp.h).
#ifdef OPTILAND_DF64_MATH_EXP_H
// log(m 2^e) as df64 + extra float for m in [1/2, 1) (a df64 from sf_arg) and any
// binary64 exponent e: e ln2 is formed to ~2^-70 with the exp lane's three-float
// ln2 (prod3) and combined with log_ext(m) like df::log_ext does for its own
// exponent (which must stay below 2^7 for its Cody-Waite products to be exact).
inline df64 sf_log_ext_scaled(df64 m, int e, thread float &ext) {
    float yext;
    df64 y = df::log_ext(m, yext);
    float P;
    df64 tail;
    df::prod3(df::make(float(e), 0.0f), 0.0f, DF64X_LN2_3, P, tail);
    df64 h = df::two_sum(P, y.hi);
    df64 rest = df::add(tail, h.lo);
    rest = df::add(rest, y.lo);
    rest = df::add(rest, yext);
    df64 r = df::quick_two_sum(h.hi, rest.hi);
    df64 l = df::two_sum(r.lo, rest.lo);
    ext = l.lo;
    return df::make(r.hi, l.hi);
}
// log|x| for a finite nonzero x of any magnitude, as df64 + ext.
inline df64 sf_log_abs_ext(sf64 x, thread float &ext) {
    int e;
    sf64 m = frexp(abs(x), e);
    return sf_log_ext_scaled(sf_arg(m), e, ext);
}

OPTILAND_SF64_BRIDGE_1(exp)
OPTILAND_SF64_BRIDGE_1(exp2)
inline sf64 expm1(sf64 x) {
    if (sf_below(x, SF64_MATH_LINEAR_E)) return x;
    return from_df64(df::expm1(sf_arg(x)));
}
// log(x): a negative argument of any magnitude is NaN (the df64 conversion of a
// negative value below 2^-126 is -0, whose log would be -inf); arguments
// outside the df64 range go through the scaled log.
inline sf64 log(sf64 x) {
    if (is_negative(x)) return nan();
    int e;
    if (!sf_out_of_df64_range(x, e)) return from_df64(df::log(sf_arg(x)));
    float ext;
    df64 L = sf_log_abs_ext(x, ext);
    return from_df64(df::add(L, ext));
}
inline sf64 log1p(sf64 x) {
    if (sf_below(x, SF64_MATH_LINEAR_E)) return x;
    int e;
    if (sf_regular(x, e) && e > SF64_MATH_EMAX && !signbit(x)) {  // 1 + x rounds to x
        float ext;
        df64 L = sf_log_abs_ext(x, ext);
        return from_df64(df::add(L, ext));
    }
    return from_df64(df::log1p(sf_arg(x)));
}
inline sf64 log2(sf64 x) {
    if (is_negative(x)) return nan();
    int e;
    if (!sf_out_of_df64_range(x, e)) return from_df64(df::log2(sf_arg(x)));
    float ext;
    df64 L = sf_log_abs_ext(x, ext);
    return from_df64(df::mul3(L, ext, DF64X_1_LN2_3));
}
inline sf64 log10(sf64 x) {
    if (is_negative(x)) return nan();
    int e;
    if (!sf_out_of_df64_range(x, e)) return from_df64(df::log10(sf_arg(x)));
    float ext;
    df64 L = sf_log_abs_ext(x, ext);
    return from_df64(df::mul3(L, ext, DF64X_1_LN10_3));
}
// Odd integer test for an integral binary64 (every |b| >= 2^53 is even).
inline bool sf_is_odd_integer(sf64 b) {
    int e = (int)((b.bits & SF64_EXP_MASK) >> 52) - 1023;
    if (e < 0 || e > 52) return false;
    ulong m = (b.bits & SF64_MANT_MASK) | SF64_IMPLICIT;
    return ((m >> (52 - e)) & 1UL) != 0UL;
}
// exp(P + tail) for a leading float P of any size: exp(P + tail - k ln2) 2^k with
// k = nint(P / ln2), the reduced argument formed to ~2^-70 (prod3 for k ln2), the
// df64 exp evaluated in [0.7, 1.42] and the power of two applied in sf64 (exact,
// gradual underflow). Used for pow results outside float32's range.
inline sf64 sf_exp_scaled(float P, df64 tail) {
    if (metal::isnan(P)) return nan();
    if (P > SF64_MATH_EXP_MAX) return inf();
    if (P < SF64_MATH_EXP_MIN) return zero();
    float fk = metal::rint(P * DF64X_1_LN2_F);
    float Pk;
    df64 tailk;
    df::prod3(df::make(fk, 0.0f), 0.0f, DF64X_LN2_3, Pk, tailk);
    df64 s = df::two_sum(P, -Pk);          // exact (Sterbenz), |s.hi| <= ~ln2/2
    df64 t = df::add(tail, s.lo);
    t = df::sub(t, tailk);
    return ldexp(from_df64(df::exp_of_sum(s.hi, t)), int(fk));
}
// |a|^b for a non-negative (or +-0, +inf, NaN) base a, the binary64 exponent b
// and its df64 form db (possibly clamped to +-FLT_MIN, see pow). A base outside
// the df64 range, or a df64 result that over/underflowed although a and b are
// finite and nonzero (a^b is then outside float32's range but may be a binary64,
// e.g. pow(10, 40) or pow(1e-300, 0.5)), is evaluated as exp(b (log m + e ln2))
// with the product formed to ~72 bits (prod3) and the scaled exp above.
inline sf64 sf_pow_mag(sf64 a, sf64 b, df64 db) {
    int e;
    bool out = sf_out_of_df64_range(a, e);
    if (!out) {
        df64 r = df::pow(sf_arg(a), db);
        bool edge = r.hi == 0.0f || metal::isinf(r.hi);
        if (!edge || !sf_regular(a, e) || !is_finite(b) || is_zero(b)) return from_df64(r);
    }
    if (df::is_nan(db)) return nan();
    float ext;
    df64 L = sf_log_abs_ext(a, ext);
    float P;
    df64 tail;
    df::prod3(db, 0.0f, float3(L.hi, L.lo, ext), P, tail);
    return sf_exp_scaled(P, tail);
}
// pow(a, b) with C99 special cases decided on the binary64 operands where the
// 48-bit conversion of b would misclassify them (see the file header).
inline sf64 pow(sf64 a, sf64 b) {
    df64 db = sf_arg(b);
    if (db.hi == 0.0f && !is_zero(b) && !is_nan(b)) {
        // 0 < |b| < 2^-126: +-FLT_MIN gives the same C99 answers for every base
        // (pow(0, -tiny) = inf, pow(NaN, tiny) = NaN, pow(-2, tiny) = NaN, ...)
        // and exp(+-2^-126 log a) rounds to 1 for every finite positive a.
        db = df::make(signbit(b) ? -0x1p-126f : 0x1p-126f, 0.0f);
    }
    if (is_negative(a) && is_finite(b) && !is_zero(b)) {
        bool b_int = eq(floor(b), b);
        if (!b_int && is_finite(a)) return nan();  // (-inf)^(non-integer) is +inf / +0
        bool b_odd = b_int && sf_is_odd_integer(b);
        sf64 r = sf_pow_mag(abs(a), b, db);
        return b_odd ? neg(r) : r;
    }
    return sf_pow_mag(a, b, db);
}
// cbrt(x) = cbrt(m 2^(e - 3k)) 2^k with e - 3k in {0, 1, 2} for arguments
// outside the df64 range (both scalings are exact).
inline sf64 cbrt(sf64 x) {
    int e;
    if (!sf_out_of_df64_range(x, e)) return from_df64(df::cbrt(sf_arg(x)));
    int k = (e >= 0) ? e / 3 : -((-e + 2) / 3);
    sf64 xs = ldexp(x, -3 * k);
    return ldexp(from_df64(df::cbrt(sf_arg(xs))), k);
}
#endif  // OPTILAND_DF64_MATH_EXP_H

// Trigonometric family (df64_math_trig.h).
#ifdef OPTILAND_DF64_MATH_TRIG_H
OPTILAND_SF64_BRIDGE_ODD(sin)
OPTILAND_SF64_BRIDGE_1(cos)
OPTILAND_SF64_BRIDGE_ODD(tan)
OPTILAND_SF64_BRIDGE_ODD(asin)
OPTILAND_SF64_BRIDGE_1(acos)
OPTILAND_SF64_BRIDGE_ODD(atan)
// atan2 is scale invariant: when either operand is outside the df64 range both
// are scaled by the same power of two (the larger magnitude lands in [1/2, 1)).
inline sf64 atan2(sf64 y, sf64 x) {
    int k;
    if (sf_common_scale(y, x, k)) {
        y = ldexp(y, k);
        x = ldexp(x, k);
    }
    return from_df64(df::atan2(sf_arg(y), sf_arg(x)));
}
#endif  // OPTILAND_DF64_MATH_TRIG_H

// Hyperbolic and special functions (df64_math_special.h).
#ifdef OPTILAND_DF64_MATH_SPECIAL_H
OPTILAND_SF64_BRIDGE_ODD(sinh)
OPTILAND_SF64_BRIDGE_1(cosh)
OPTILAND_SF64_BRIDGE_ODD(tanh)
// asinh(x) = sign(x) (log|x| + ln2) and acosh(x) = log x + ln2 beyond 2^127
// (x^2 would overflow float32; 1/(4x^2) is below binary64 resolution).
inline sf64 asinh(sf64 x) {
    if (sf_below(x, SF64_MATH_ODD_E)) return x;
    int e;
    if (sf_regular(x, e) && e > SF64_MATH_EMAX) {
        float ext;
        df64 L = df::add(df::add(sf_log_abs_ext(x, ext), ext), DF64_LN2);
        return copysign(from_df64(L), x);
    }
    return from_df64(df::asinh(sf_arg(x)));
}
inline sf64 acosh(sf64 x) {
    int e;
    if (sf_regular(x, e) && e > SF64_MATH_EMAX && !signbit(x)) {
        float ext;
        df64 L = df::add(df::add(sf_log_abs_ext(x, ext), ext), DF64_LN2);
        return from_df64(L);
    }
    return from_df64(df::acosh(sf_arg(x)));
}
OPTILAND_SF64_BRIDGE_ODD(atanh)
// hypot: NaN rules first (a huge finite operand would become inf and make
// hypot(NaN, 1e300) = +inf), then common power-of-two scaling when either
// operand is outside the df64 range; the result is scaled back in sf64 (exact,
// gradual underflow).
inline sf64 hypot(sf64 x, sf64 y) {
    if (is_nan(x) || is_nan(y)) return (is_inf(x) || is_inf(y)) ? inf() : nan();
    int k;
    if (!sf_common_scale(x, y, k)) return from_df64(df::hypot(sf_arg(x), sf_arg(y)));
    x = ldexp(x, k);
    y = ldexp(y, k);
    return ldexp(from_df64(df::hypot(sf_arg(x), sf_arg(y))), -k);
}
// erf(x) = 2x/sqrt(pi) (1 - x^2/3 + ...) and erfinv(x) = x sqrt(pi)/2 (1 + pi x^2/12
// + ...): one sf64 product below 2^-27 (the dropped term is below 2^-55).
inline sf64 erf(sf64 x) {
    if (sf_below(x, SF64_MATH_ODD_E)) return mul(x, make(SF64_2_SQRTPI_BITS));
    return from_df64(df::erf(sf_arg(x)));
}
OPTILAND_SF64_BRIDGE_1(erfc)
inline sf64 erfinv(sf64 x) {
    if (sf_below(x, SF64_MATH_ODD_E)) return mul(x, make(SF64_SQRTPI_2_BITS));
    return from_df64(df::erfinv(sf_arg(x)));
}
// lgamma: |x| < 2^-126 is -log|x| (the neglected gamma x is below 2^-126
// relative); x >= 2^127 is x (log x - 1) in sf64 (Stirling's leading term, the
// dropped (log x)/2 + ln sqrt(2 pi) - 1/(12 x) are below 2^-53 relative); a
// negative x of that size is an integer, i.e. a pole.
inline sf64 lgamma(sf64 x) {
    int e;
    if (!sf_out_of_df64_range(x, e)) return from_df64(df::lgamma(sf_arg(x)));
    float ext;
    df64 L = sf_log_abs_ext(x, ext);
    if (e < 0) return from_df64(df::neg(df::add(L, ext)));
    if (signbit(x)) return inf();
    return mul(x, from_df64(df::add(df::add(L, ext), -1.0f)));
}
#endif  // OPTILAND_DF64_MATH_SPECIAL_H

#undef OPTILAND_SF64_BRIDGE_1
#undef OPTILAND_SF64_BRIDGE_2
#undef OPTILAND_SF64_BRIDGE_ODD

}  // namespace sf

#endif  // OPTILAND_DF64_CORE_H
#endif  // OPTILAND_SF64_MATH_H
