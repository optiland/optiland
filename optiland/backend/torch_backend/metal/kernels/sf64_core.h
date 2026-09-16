// sf64_core.h — software IEEE-754 binary64 ("sf64") for Apple GPUs (Metal Shading
// Language), the exact-mode representation of Optiland-Metal.
//
// An sf64 value is the 64-bit pattern of an IEEE binary64 number held in a `ulong`.
// Arithmetic (add/sub/mul/div/sqrt/fma) and the float32/int64 conversions are
// provided by the vendored metal-softfloat header (`vendor/softfloat64.metal`,
// MIT + BSD-3-Clause), always in rounding mode 0 (nearest, ties to even) and
// without SOFTFLOAT_FTZ, so every operation is correctly rounded with gradual
// underflow (the upstream header comment claiming that fdiv flushes subnormal
// outputs is stale: tests/metal/test_sf64_core.py measures gradual underflow on
// fdiv). Two vendored conversions are NOT used because they are wrong at the
// edges (measured, see the tests): cvt_f64_to_f32 returns 0 instead of the
// smallest float32 subnormal for results in (2^-150, 2^-149), and cvt_f64_to_i64
// in mode 0 rounds odd integers in [2^52, 2^53) upward; to_float and
// to_long_rint below are implemented on the bit patterns instead. Everything
// else here (sign/classification, rounding to integers, fmod/remainder,
// ldexp/frexp, min/max, df64 <-> sf64) is bit manipulation and is exact.
//
// Amalgamation contract (NO #include of the vendor header):
//   vendor/softfloat64.metal  ->  [df64_core.h (+ df64 math)]  ->  sf64_core.h
// The df64 overloads (from_df64(df64), to_df64) are compiled only when
// OPTILAND_DF64_CORE_H is already defined, i.e. df64_core.h preceded this file.
//
// Semantics follow IEEE-754 / C99 where they apply (fmod, copysign, fmin/fmax) and
// torch/NumPy elsewhere (minimum/maximum propagate NaN, remainder_py takes the sign
// of the divisor, rint rounds half to even, to_long truncates toward zero and
// saturates, NaN -> 0). Every NaN produced by the arithmetic is the canonical
// quiet NaN 0x7FF8000000000000; NaN inputs to bit-manipulating helpers pass
// through unchanged.
//
// Part of Optiland-Metal (MIT). fmod follows the classic exact long-division
// scheme (SoftFloat f64_rem / musl fmod); ldexp follows musl scalbn.

#ifndef OPTILAND_SF64_CORE_H
#define OPTILAND_SF64_CORE_H

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

#include <metal_stdlib>

struct sf64 {
    ulong bits;
};

namespace sf {

// ---------------------------------------------------------------------------
// Bit-layout constants
// ---------------------------------------------------------------------------
constant ulong SF64_SIGN_MASK = 0x8000000000000000UL;
constant ulong SF64_EXP_MASK = 0x7FF0000000000000UL;
constant ulong SF64_MANT_MASK = 0x000FFFFFFFFFFFFFUL;
constant ulong SF64_IMPLICIT = 0x0010000000000000UL;  // 1 << 52
constant ulong SF64_ABS_MASK = 0x7FFFFFFFFFFFFFFFUL;
constant ulong SF64_INF_BITS = 0x7FF0000000000000UL;
constant ulong SF64_QNAN_BITS = 0x7FF8000000000000UL;
constant ulong SF64_ONE_BITS = 0x3FF0000000000000UL;
constant ulong SF64_HALF_BITS = 0x3FE0000000000000UL;
constant ulong SF64_TWO_P1023 = 0x7FE0000000000000UL;  // 2^1023
constant ulong SF64_TWO_M969 = 0x0360000000000000UL;   // 2^-969 = 2^-1022 * 2^53
constant ulong SF64_TWO_P64 = 0x43F0000000000000UL;    // 2^64
constant uint SF64_RNE = 0u;  // round to nearest, ties to even
constant uint SF64_RTZ = 3u;  // round toward zero

// ---------------------------------------------------------------------------
// Construction and conversion
// ---------------------------------------------------------------------------
inline sf64 make(ulong b) { sf64 r; r.bits = b; return r; }
inline ulong to_bits(sf64 a) { return a.bits; }
inline sf64 zero() { return make(0UL); }
inline sf64 neg_zero() { return make(SF64_SIGN_MASK); }
inline sf64 one() { return make(SF64_ONE_BITS); }
inline sf64 one_half() { return make(SF64_HALF_BITS); }  // `half` is an MSL type name
inline sf64 nan() { return make(SF64_QNAN_BITS); }
inline sf64 inf() { return make(SF64_INF_BITS); }
inline sf64 neg_inf() { return make(SF64_INF_BITS | SF64_SIGN_MASK); }

// float32 -> binary64 is exact (no rounding mode).
inline sf64 from_float(float f) {
    return make(__softfloat64_cvt_f32_to_f64(as_type<uint>(f)));
}
inline sf64 from_int(int i) { return make(__softfloat64_cvt_i64_to_f64((long)i, SF64_RNE)); }
inline sf64 from_long(long i) { return make(__softfloat64_cvt_i64_to_f64(i, SF64_RNE)); }
inline sf64 from_ulong(ulong u) { return make(__softfloat64_cvt_u64_to_f64(u, SF64_RNE)); }

// Nearest float32 (ties to even; overflow -> +-inf; gradual underflow; NaN keeps
// its sign and the top payload bits, quieted). Implemented here on the bit
// pattern: the vendored __softfloat64_cvt_f64_to_f32 returns 0 for results in
// (2^-150, 2^-149), which must round up to the smallest float32 subnormal.
inline float to_float(sf64 a) {
    ulong b = a.bits;
    uint sign = (uint)(b >> 63) << 31;
    int e = (int)((b >> 52) & 0x7FFUL);
    ulong m = b & SF64_MANT_MASK;
    if (e == 0x7FF) {
        if (m != 0UL) return as_type<float>(sign | 0x7FC00000u | (uint)(m >> 29));
        return as_type<float>(sign | 0x7F800000u);
    }
    // Binary64 subnormals (< 2^-1022) and zero are far below half the smallest
    // float32 subnormal (2^-150): they round to a signed zero.
    if (e == 0) return as_type<float>(sign);
    ulong sig = m | SF64_IMPLICIT;  // 53-bit significand, value = sig * 2^(e-1075)
    int fe = e - 1023 + 127;        // float32 biased exponent of the leading bit
    if (fe >= 0xFF) return as_type<float>(sign | 0x7F800000u);
    uint shift;
    if (fe >= 1) {
        shift = 29u;  // keep 24 bits
    } else {
        // Subnormal float32: units of 2^-149. shift = 29 + (1 - fe); beyond 63 the
        // value is below 2^-150 and rounds to zero.
        int sh = 30 - fe;
        if (sh > 63) return as_type<float>(sign);
        shift = (uint)sh;
        fe = 0;
    }
    ulong q = sig >> shift;
    ulong rem = sig & ((1UL << shift) - 1UL);
    ulong tie = 1UL << (shift - 1u);
    if (rem > tie || (rem == tie && (q & 1UL) != 0UL)) q += 1UL;
    // For normals q holds the implicit bit at position 23: adding it to the
    // exponent field (fe << 23) carries correctly on rounding overflow, including
    // into +-inf when fe == 0xFE. For subnormals fe == 0 and a carry into bit 23 is
    // exactly the smallest normal.
    uint r = ((uint)fe << 23) + (uint)q - ((fe >= 1) ? 0x00800000u : 0u);
    return as_type<float>(sign | r);
}
// C/NumPy/torch integer cast: truncation toward zero, NaN -> 0, saturating.
inline long to_long(sf64 a) { return __softfloat64_cvt_f64_to_i64(a.bits, SF64_RTZ); }
inline sf64 rint(sf64 a);
// Round to nearest even integer (lrint), NaN -> 0, saturating. Goes through the
// exact rint() below because the vendored cvt_f64_to_i64 in mode 0 rounds odd
// integers in [2^52, 2^53) upward.
inline long to_long_rint(sf64 a) { return to_long(rint(a)); }
inline int to_int(sf64 a) {
    long v = to_long(a);
    if (v > (long)INT_MAX) return INT_MAX;
    if (v < (long)INT_MIN) return INT_MIN;
    return (int)v;
}

// ---------------------------------------------------------------------------
// Classification and sign (bit tests; NaN payload preserved)
// ---------------------------------------------------------------------------
inline bool signbit(sf64 a) { return (a.bits & SF64_SIGN_MASK) != 0UL; }
inline bool is_nan(sf64 a) { return (a.bits & SF64_ABS_MASK) > SF64_INF_BITS; }
inline bool is_inf(sf64 a) { return (a.bits & SF64_ABS_MASK) == SF64_INF_BITS; }
inline bool is_finite(sf64 a) { return (a.bits & SF64_EXP_MASK) != SF64_EXP_MASK; }
inline bool is_zero(sf64 a) { return (a.bits & SF64_ABS_MASK) == 0UL; }
inline bool is_subnormal(sf64 a) {
    return (a.bits & SF64_EXP_MASK) == 0UL && (a.bits & SF64_MANT_MASK) != 0UL;
}
inline bool is_signaling_nan(sf64 a) {
    return is_nan(a) && (a.bits & 0x0008000000000000UL) == 0UL;
}
// Strict sign of the value: false for +-0 and NaN (matches df::is_negative).
inline bool is_negative(sf64 a) { return signbit(a) && !is_zero(a) && !is_nan(a); }
inline bool is_positive(sf64 a) { return !signbit(a) && !is_zero(a) && !is_nan(a); }

inline sf64 neg(sf64 a) { return make(a.bits ^ SF64_SIGN_MASK); }
inline sf64 abs(sf64 a) { return make(a.bits & SF64_ABS_MASK); }
inline sf64 copysign(sf64 a, sf64 b) {
    return make((a.bits & SF64_ABS_MASK) | (b.bits & SF64_SIGN_MASK));
}
inline sf64 copysign(sf64 a, float b) {
    return make((a.bits & SF64_ABS_MASK) | (metal::signbit(b) ? SF64_SIGN_MASK : 0UL));
}
// -1, +-0 (preserved), +1; NaN -> NaN. Matches torch.sign / numpy.sign.
inline sf64 sign(sf64 a) {
    if (is_nan(a)) return nan();
    if (is_zero(a)) return a;
    return copysign(one(), a);
}

// ---------------------------------------------------------------------------
// Arithmetic (metal-softfloat, mode 0 = nearest-even, correctly rounded)
// ---------------------------------------------------------------------------
inline sf64 add(sf64 a, sf64 b) { return make(__softfloat64_fadd(a.bits, b.bits, SF64_RNE)); }
inline sf64 sub(sf64 a, sf64 b) { return make(__softfloat64_fsub(a.bits, b.bits, SF64_RNE)); }
inline sf64 mul(sf64 a, sf64 b) { return make(__softfloat64_fmul(a.bits, b.bits, SF64_RNE)); }
inline sf64 div(sf64 a, sf64 b) { return make(__softfloat64_fdiv(a.bits, b.bits, SF64_RNE)); }
inline sf64 sqrt(sf64 a) { return make(__softfloat64_fsqrt(a.bits, SF64_RNE)); }
// a * b + c with a single rounding.
inline sf64 fma(sf64 a, sf64 b, sf64 c) {
    return make(__softfloat64_fma(a.bits, b.bits, c.bits, SF64_RNE));
}
inline sf64 sqr(sf64 a) { return mul(a, a); }
inline sf64 recip(sf64 a) { return div(one(), a); }
// 1/sqrt(a): two roundings (not correctly rounded as a single operation).
inline sf64 rsqrt(sf64 a) { return div(one(), sqrt(a)); }
inline sf64 mul_add(sf64 a, sf64 b, sf64 c) { return fma(a, b, c); }

// ---------------------------------------------------------------------------
// Comparisons (IEEE: any NaN operand makes eq/lt/le/gt/ge false and ne true)
// ---------------------------------------------------------------------------
inline bool eq(sf64 a, sf64 b) { return __softfloat64_feq(a.bits, b.bits); }
inline bool ne(sf64 a, sf64 b) { return !__softfloat64_feq(a.bits, b.bits); }
inline bool lt(sf64 a, sf64 b) { return __softfloat64_flt(a.bits, b.bits); }
inline bool le(sf64 a, sf64 b) { return __softfloat64_fle(a.bits, b.bits); }
inline bool gt(sf64 a, sf64 b) { return __softfloat64_fgt(a.bits, b.bits); }
inline bool ge(sf64 a, sf64 b) { return __softfloat64_fge(a.bits, b.bits); }

// torch.minimum/maximum and numpy.minimum/maximum: NaN-propagating. fmin/fmax:
// C99 / numpy.fmin/fmax, NaN-ignoring. On a +-0 tie the minimum is -0 and the
// maximum is +0 (IEEE 754-2019 minimum/maximum, NumPy, std::fmin/fmax); torch CPU
// is not self-consistent there (its scalar and vectorized paths differ).
inline sf64 minimum(sf64 a, sf64 b) {
    if (is_nan(a) || is_nan(b)) return nan();
    if (lt(b, a)) return b;
    if (lt(a, b)) return a;
    return signbit(a) ? a : b;
}
inline sf64 maximum(sf64 a, sf64 b) {
    if (is_nan(a) || is_nan(b)) return nan();
    if (gt(b, a)) return b;
    if (gt(a, b)) return a;
    return signbit(a) ? b : a;
}
// A signaling NaN operand yields NaN (IEEE minNum/maxNum, hardware fmin/fmax as
// seen through NumPy and torch); quiet NaNs are ignored.
inline sf64 fmin(sf64 a, sf64 b) {
    if (is_signaling_nan(a) || is_signaling_nan(b)) return nan();
    if (is_nan(a)) return b;
    if (is_nan(b)) return a;
    if (lt(b, a)) return b;
    if (lt(a, b)) return a;
    return signbit(a) ? a : b;
}
inline sf64 fmax(sf64 a, sf64 b) {
    if (is_signaling_nan(a) || is_signaling_nan(b)) return nan();
    if (is_nan(a)) return b;
    if (is_nan(b)) return a;
    if (gt(b, a)) return b;
    if (gt(a, b)) return a;
    return signbit(a) ? b : a;
}

// ---------------------------------------------------------------------------
// Rounding to integral values (exact, integer bit manipulation)
// ---------------------------------------------------------------------------
enum SF64RoundKind { SF64_ROUND_FLOOR = 0, SF64_ROUND_CEIL, SF64_ROUND_TRUNC,
                     SF64_ROUND_RINT, SF64_ROUND_AWAY };

inline sf64 round_impl(sf64 x, int kind) {
    ulong b = x.bits;
    ulong sign = b & SF64_SIGN_MASK;
    int e = (int)((b & SF64_EXP_MASK) >> 52) - 1023;
    // e >= 52: already integral (no fraction bits); e == 1024: inf/NaN pass through.
    if (e >= 52) return x;
    bool negative = sign != 0UL;
    if (e < 0) {
        // |x| < 1: the result is +-0 (sign preserved) or +-1.
        if ((b & SF64_ABS_MASK) == 0UL) return x;
        bool mag_one;
        switch (kind) {
            case SF64_ROUND_FLOOR: mag_one = negative; break;
            case SF64_ROUND_CEIL: mag_one = !negative; break;
            case SF64_ROUND_RINT:  // strictly above one half (a tie rounds to 0, even)
                mag_one = (e == -1) && ((b & SF64_MANT_MASK) != 0UL); break;
            case SF64_ROUND_AWAY: mag_one = (e == -1); break;  // |x| >= 0.5
            default: mag_one = false; break;                   // trunc
        }
        return make(sign | (mag_one ? SF64_ONE_BITS : 0UL));
    }
    // 0 <= e <= 51: the low (52 - e) mantissa bits are the fraction.
    ulong frac_mask = SF64_MANT_MASK >> e;
    ulong frac = b & frac_mask;
    if (frac == 0UL) return x;
    ulong unit = SF64_IMPLICIT >> e;  // one integer step in the bit pattern
    ulong tie = unit >> 1;
    ulong t = b & ~frac_mask;         // truncated toward zero
    bool bump;
    switch (kind) {
        case SF64_ROUND_FLOOR: bump = negative; break;
        case SF64_ROUND_CEIL: bump = !negative; break;
        case SF64_ROUND_RINT:
            bump = (frac > tie) || ((frac == tie) && ((t & unit) != 0UL)); break;
        case SF64_ROUND_AWAY: bump = frac >= tie; break;
        default: bump = false; break;
    }
    // Adding `unit` may carry into the exponent field: that yields exactly the
    // next power of two, which is the correct rounded value.
    return make(bump ? t + unit : t);
}
inline sf64 floor(sf64 a) { return round_impl(a, SF64_ROUND_FLOOR); }
inline sf64 ceil(sf64 a) { return round_impl(a, SF64_ROUND_CEIL); }
inline sf64 trunc(sf64 a) { return round_impl(a, SF64_ROUND_TRUNC); }
// Round half to even (torch.round / numpy.rint).
inline sf64 rint(sf64 a) { return round_impl(a, SF64_ROUND_RINT); }
// Round half away from zero (C round).
inline sf64 round_away(sf64 a) { return round_impl(a, SF64_ROUND_AWAY); }

// ---------------------------------------------------------------------------
// fmod / remainder (exact long division on the 53-bit significands)
// ---------------------------------------------------------------------------
// C99 fmod: result has the sign of a and |result| < |b|; exact for every input
// (the remainder is always representable). fmod(x, +-inf) = x, fmod(+-0, y) = +-0,
// fmod(inf, y) = fmod(x, 0) = NaN. Cost is O(exponent difference) iterations, so
// fmod(1e300, 3) runs ~1000 subtract-and-shift steps.
inline sf64 fmod(sf64 a, sf64 b) {
    ulong ux = a.bits;
    ulong uy = b.bits;
    int ex = (int)((ux >> 52) & 0x7FFUL);
    int ey = (int)((uy >> 52) & 0x7FFUL);
    ulong sx = ux & SF64_SIGN_MASK;
    ulong ax = ux & SF64_ABS_MASK;
    ulong ay = uy & SF64_ABS_MASK;
    if (ay == 0UL || ex == 0x7FF || ay > SF64_INF_BITS) return nan();
    if (ax <= ay) return (ax == ay) ? make(sx) : a;  // covers b = +-inf too

    // Normalize both significands to 1.xxx at bit 52 (subnormals get ex <= 0).
    if (ex == 0) {
        for (ulong i = ux << 12; (i >> 63) == 0UL; i <<= 1) ex--;
        ux <<= (uint)(-ex + 1);
    } else {
        ux = (ux & SF64_MANT_MASK) | SF64_IMPLICIT;
    }
    if (ey == 0) {
        for (ulong i = uy << 12; (i >> 63) == 0UL; i <<= 1) ey--;
        uy <<= (uint)(-ey + 1);
    } else {
        uy = (uy & SF64_MANT_MASK) | SF64_IMPLICIT;
    }

    // Long division: subtract when possible, shift the remainder left.
    ulong i;
    for (; ex > ey; ex--) {
        i = ux - uy;
        if ((i >> 63) == 0UL) {
            if (i == 0UL) return make(sx);
            ux = i;
        }
        ux <<= 1;
    }
    i = ux - uy;
    if ((i >> 63) == 0UL) {
        if (i == 0UL) return make(sx);
        ux = i;
    }
    for (; (ux >> 52) == 0UL; ux <<= 1, ex--) {}

    // Repack (the remainder is exact: no rounding, possibly subnormal).
    if (ex > 0) {
        ux -= SF64_IMPLICIT;
        ux |= (ulong)ex << 52;
    } else {
        ux >>= (uint)(-ex + 1);
    }
    return make(ux | sx);
}

// Python / NumPy remainder: sign of the divisor. Same algorithm as NumPy's
// npy_divmod and CPython's float_rem: r = fmod(a, b); if r != 0 and its sign
// differs from b, r += b (one rounding); a zero result takes the sign of b.
inline sf64 remainder_py(sf64 a, sf64 b) {
    sf64 r = fmod(a, b);
    if (is_nan(r)) return r;
    if (is_zero(r)) return copysign(zero(), b);
    if (signbit(r) != signbit(b)) r = add(r, b);
    return r;
}

// ---------------------------------------------------------------------------
// Power-of-two scaling and decomposition
// ---------------------------------------------------------------------------
// x * 2^n, correctly rounded (gradual underflow, overflow -> inf). Each factor is
// an exact power of two; the staged scaling (musl scalbn) avoids double rounding.
inline sf64 ldexp(sf64 x, int n) {
    sf64 y = x;
    if (n > 1023) {
        y = mul(y, make(SF64_TWO_P1023));
        n -= 1023;
        if (n > 1023) {
            y = mul(y, make(SF64_TWO_P1023));
            n -= 1023;
            if (n > 1023) n = 1023;
        }
    } else if (n < -1022) {
        y = mul(y, make(SF64_TWO_M969));
        n += 969;
        if (n < -1022) {
            y = mul(y, make(SF64_TWO_M969));
            n += 969;
            if (n < -1022) n = -1022;
        }
    }
    return mul(y, make((ulong)(n + 1023) << 52));
}
// Exact scaling by a float32 power of two (mirrors df::mul_pwr2).
inline sf64 mul_pwr2(sf64 a, float p) { return mul(a, from_float(p)); }
inline sf64 mul_pwr2(sf64 a, int n) { return ldexp(a, n); }

// x = m * 2^e with 0.5 <= |m| < 1; +-0, +-inf, NaN return x with e = 0.
inline sf64 frexp(sf64 x, thread int &e) {
    ulong b = x.bits;
    int ee = (int)((b >> 52) & 0x7FFUL);
    if (ee == 0x7FF) { e = 0; return x; }
    int bias = 0;
    if (ee == 0) {
        if ((b & SF64_ABS_MASK) == 0UL) { e = 0; return x; }
        b = mul(x, make(SF64_TWO_P64)).bits;  // exact: now normal
        ee = (int)((b >> 52) & 0x7FFUL);
        bias = 64;
    }
    e = ee - 1022 - bias;
    return make((b & ~SF64_EXP_MASK) | SF64_HALF_BITS);
}

// ---------------------------------------------------------------------------
// df64 <-> sf64
// ---------------------------------------------------------------------------
// hi + lo rounded once to binary64: exact whenever the two float32 words span at
// most 53 bits (every pair that came from a binary64, and every normalized pair
// whose words are at most 29 binades apart), correctly rounded otherwise (e.g.
// (1, 2^-60) gives 1.0). A zero lo returns the exact conversion of hi so that
// the sign of a zero pair survives: (-0, +0) is the canonical df64 -0 and
// add(-0, +0) would give +0 under RNE.
inline sf64 from_df64(float hi, float lo) {
    // Bit test: a denormal lo compares equal to zero on this GPU but must be kept.
    if ((as_type<uint>(lo) & 0x7FFFFFFFu) == 0u) return from_float(hi);
    return add(from_float(hi), from_float(lo));
}
inline sf64 from_df64(float2 p) { return from_df64(p.x, p.y); }
// Split into the RNE-canonical (hi, lo) float32 pair of x rounded to 48
// significant bits (the df64 precision): hi = RN32(x), lo = RN32(x - hi), then
// one float32 Fast2Sum so that a lo of exactly +-ulp(hi)/2 next to an odd hi
// (a valid but non-canonical "tie form") becomes the even-hi form the df64
// kernels produce; the pair's value is unchanged. Exact only when x fits in 48
// bits. Non-finite hi gets lo = 0 (inf - inf would otherwise leave a NaN lo).
// A value within 2^-50 of the float32 overflow tie (hi = +-FLT_MAX, lo rounded
// to +-2^103) is at the df64 overflow threshold (df64_core.h overflow rule):
// the Fast2Sum overflows and the result is a clean +-inf with lo = 0, never
// the (inf, -inf) pair of the error term.
inline float2 to_float2(sf64 x) {
    float hi = to_float(x);
    float lo = 0.0f;
    if (metal::isfinite(hi)) {
        lo = to_float(sub(x, from_float(hi)));
        // Only for a normal lo (exponent field != 0): a zero lo needs nothing,
        // and a denormal lo is flushed by the GPU's arithmetic (its tie rounding
        // in the Fast2Sum is not RNE either), so it is stored as is.
        if ((as_type<uint>(lo) & 0x7F800000u) != 0u) {
            float s = hi + lo;
            if (!metal::isfinite(s)) return float2(s, 0.0f);
            lo = lo - (s - hi);
            hi = s;
        }
    }
    return float2(hi, lo);
}
#ifdef OPTILAND_DF64_CORE_H
inline sf64 from_df64(df64 a) { return from_df64(a.hi, a.lo); }
inline df64 to_df64(sf64 x) {
    float2 p = to_float2(x);
    return df::make(p.x, p.y);
}
#endif

}  // namespace sf

// ---------------------------------------------------------------------------
// Operators (thin sugar over the namespace functions)
// ---------------------------------------------------------------------------
inline sf64 operator-(sf64 a) { return sf::neg(a); }
inline sf64 operator+(sf64 a, sf64 b) { return sf::add(a, b); }
inline sf64 operator-(sf64 a, sf64 b) { return sf::sub(a, b); }
inline sf64 operator*(sf64 a, sf64 b) { return sf::mul(a, b); }
inline sf64 operator/(sf64 a, sf64 b) { return sf::div(a, b); }
inline bool operator==(sf64 a, sf64 b) { return sf::eq(a, b); }
inline bool operator!=(sf64 a, sf64 b) { return sf::ne(a, b); }
inline bool operator<(sf64 a, sf64 b) { return sf::lt(a, b); }
inline bool operator>(sf64 a, sf64 b) { return sf::gt(a, b); }
inline bool operator<=(sf64 a, sf64 b) { return sf::le(a, b); }
inline bool operator>=(sf64 a, sf64 b) { return sf::ge(a, b); }

#endif  // OPTILAND_SF64_CORE_H
