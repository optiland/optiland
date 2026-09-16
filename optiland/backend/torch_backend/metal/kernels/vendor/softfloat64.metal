// dist/softfloat64.metal — redistributable IEEE-754 binary64 softfloat
// for Apple Metal. Generated from metal-softfloat's
// shaders/softfloat.metal; do not hand-edit. Re-run the generator with
// `cargo run -p metal-softfloat --bin gen-msl-header`.
//
// SPDX-License-Identifier: BSD-3-Clause AND MIT
//
// Berkeley SoftFloat lookup tables (APPROX_RECIP_SQRT_K0S/K1S,
// APPROX_RECIP_K0S/K1S) and the recip / recipSqrt approximation
// functions are derived from John R. Hauser's Berkeley SoftFloat-3e:
//
//   Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017
//   The Regents of the University of California.
//   All Rights Reserved.
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions
//   are met:
//
//    1. Redistributions of source code must retain the above copyright
//       notice, this list of conditions, and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions, and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//    3. Neither the name of the University nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
//   THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS",
//   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
//   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//   PARTICULAR PURPOSE, ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
//   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//   OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//   SUCH DAMAGE.
//
// All other contributions (the integer-arithmetic dispatchers, IEEE-754
// special-case heads, gradual-underflow code, FMA, MSL kernel layout,
// and tests) are MIT-licensed:
//
//   Copyright (c) 2026 metal-softfloat contributors.
//   See LICENSE for the full text.
//
// Public API:
//
//   ulong __softfloat64_fadd(ulong a, ulong b, uint mode);
//   ulong __softfloat64_fsub(ulong a, ulong b, uint mode);
//   ulong __softfloat64_fmul(ulong a, ulong b, uint mode);
//   ulong __softfloat64_fdiv(ulong a, ulong b, uint mode);
//   ulong __softfloat64_fsqrt(ulong a, uint mode);
//   ulong __softfloat64_fma(ulong a, ulong b, ulong c, uint mode);
//
//   ulong __softfloat64_cvt_i64_to_f64(long  x, uint mode);
//   ulong __softfloat64_cvt_u64_to_f64(ulong x, uint mode);
//   long  __softfloat64_cvt_f64_to_i64(ulong a, uint mode);
//   ulong __softfloat64_cvt_f32_to_f64(uint  a);                // exact, no mode
//   uint  __softfloat64_cvt_f64_to_f32(ulong a, uint mode);
//
//   bool  __softfloat64_feq(ulong a, ulong b);
//   bool  __softfloat64_flt(ulong a, ulong b);
//   bool  __softfloat64_fle(ulong a, ulong b);
//   bool  __softfloat64_fgt(ulong a, ulong b);
//   bool  __softfloat64_fge(ulong a, ulong b);
//
// `mode` is the IEEE-754 rounding mode:
//   0 = nearest-ties-to-even, 1 = toward -inf, 2 = toward +inf, 3 = toward zero.
//
// f64 / u64 / i64 values cross the API as `ulong` / `long` bit patterns;
// f32 values cross as `uint`. Use `as_type<>(...)` on the caller side to
// bit-cast between native floats and these integer payloads.
//
// Compile-time switches:
//   -DSOFTFLOAT_FTZ  flush subnormal inputs/outputs to zero.
//
// IEEE-754 conformance: see metal-softfloat/docs/ieee754_conformance.md.
// Limitation: __softfloat64_fdiv / __softfloat64_fsqrt currently flush
// subnormal *outputs* to zero. Subnormal inputs are handled correctly.
// Other ops (fadd, fsub, fmul, fma) produce gradual-underflow output.

// Software f64 arithmetic for Metal.
//
// Apple GPUs have no hardware f64: no `double` type, no FP64 ALU. This
// file implements IEEE-754 double-precision add/sub/mul/div/sqrt using
// integer arithmetic on u64 bit patterns, with all four rounding modes.
//
// IEEE-754 conformance:
// - All four rounding modes (nearest-ties-to-even, ±∞, zero).
// - NaN / ±Inf / ±0 propagation per §6.
// - Subnormal inputs and outputs (gradual underflow) per §7.4.
// - Canonical qNaN (0x7FF8_0000_0000_0000) for every NaN result;
//   signaling-NaN distinction is not preserved.
//
// Compile-time switch:
// - Define SOFTFLOAT_FTZ (e.g. `xcrun metal -DSOFTFLOAT_FTZ ...`) to
//   flush subnormal inputs and outputs to zero. Mirrors the `ftz`
//   Cargo feature on the Rust side. Default: gradual underflow on for
//   every op (fadd/fsub/fmul/fdiv/fsqrt/fma).
//
// Each op's algorithm mirrors `metal_softfloat::softfloat_ref`
// in the Rust crate — see that module for algorithm commentary and
// the reference tests that validated bit-exactness against native f64.

#include <metal_stdlib>

// We deliberately do **not** `using namespace metal;` at file scope —
// pulling that into a downstream user's translation unit by way of a
// transitive `#include` is exactly the kind of namespace pollution
// `softfloat64_internal` exists to prevent. Implementation code that
// needs `metal::` symbols re-introduces the using-directive inside its
// own namespace below; the public API at file root references the few
// vector types it needs (`metal::ulong2`) with explicit qualification.
// Scalar types (`uint`, `ulong`, `int`, `long`, `bool`) are MSL built-ins
// available at global scope without any using-directive.
//
// Internals (constants, helpers, the `softfloat_*` algorithm functions)
// live inside `softfloat64_internal` so a downstream `#include` of this
// file in two separate .metal translation units doesn't trip duplicate-
// symbol errors at link time and doesn't pollute the user's global
// names with our `U128`, `Unpacked`, `softfloat_add`, etc. The public
// surface (`__softfloat64_*` and `__softfloat64_unp_*`) sits at file
// root and forwards into the namespace.
namespace softfloat64_internal {
using namespace metal;

// --- Constants (match softfloat_ref) ---------------------------------------

constant uint  EXP_BITS = 11u;
constant uint  MANT_BITS = 52u;
constant ulong EXP_MASK = 0x7FFUL;
constant ulong MANT_MASK = 0xFFFFFFFFFFFFFUL;
constant ulong IMPLICIT_BIT = 1UL << 52;
constant int   EXP_BIAS = 1023;

constant uint RMODE_NEAREST = 0u;
constant uint RMODE_DOWN    = 1u;
constant uint RMODE_UP      = 2u;
constant uint RMODE_ZERO    = 3u;

// Canonical IEEE-754 binary64 patterns we emit verbatim from the
// special-case dispatch heads.
constant ulong CANONICAL_QNAN = 0x7FF8000000000000UL;
constant ulong POS_INF        = 0x7FF0000000000000UL;

// Forward declarations for Berkeley reciprocal helpers (definitions
// later in file). Needed because the _unp_normal variants reference
// them earlier than they're defined.
static uint approx_recip32_1(uint a);
static uint approx_recip_sqrt32_1(uint odd_exp_a, uint a);

// IEEE-754 class tags. Match the order of `IeeeClass` in softfloat_ref.rs
// (Zero=0, Subnormal=1, Normal=2, Inf=3, NaN=4) so debug logs line up.
constant uint CLASS_ZERO      = 0u;
constant uint CLASS_SUBNORMAL = 1u;
constant uint CLASS_NORMAL    = 2u;
constant uint CLASS_INF       = 3u;
constant uint CLASS_NAN       = 4u;

static uint classify(ulong bits) {
    ulong exp_raw = (bits >> MANT_BITS) & EXP_MASK;
    ulong mant_raw = bits & MANT_MASK;
    if (exp_raw == 0UL) return (mant_raw == 0UL) ? CLASS_ZERO : CLASS_SUBNORMAL;
    if (exp_raw == EXP_MASK) return (mant_raw == 0UL) ? CLASS_INF : CLASS_NAN;
    return CLASS_NORMAL;
}

// --- 128-bit unsigned integer (hi:lo) --------------------------------------

struct U128 { ulong hi; ulong lo; };

static __attribute__((always_inline)) U128 u128_make(ulong hi, ulong lo) { return U128{hi, lo}; }
static __attribute__((always_inline)) U128 u128_from_u64(ulong x) { return U128{0UL, x}; }

static __attribute__((always_inline)) bool u128_is_zero(U128 a) { return a.hi == 0UL && a.lo == 0UL; }

static __attribute__((always_inline)) int u128_cmp(U128 a, U128 b) {
    if (a.hi < b.hi) return -1;
    if (a.hi > b.hi) return  1;
    if (a.lo < b.lo) return -1;
    if (a.lo > b.lo) return  1;
    return 0;
}

static __attribute__((always_inline)) U128 u128_add(U128 a, U128 b) {
    ulong lo = a.lo + b.lo;
    ulong carry = (lo < a.lo) ? 1UL : 0UL;
    ulong hi = a.hi + b.hi + carry;
    return U128{hi, lo};
}

static __attribute__((always_inline)) U128 u128_sub(U128 a, U128 b) {
    ulong lo = a.lo - b.lo;
    ulong borrow = (a.lo < b.lo) ? 1UL : 0UL;
    ulong hi = a.hi - b.hi - borrow;
    return U128{hi, lo};
}

// Left shift by `n` (0 <= n < 128), branch-free.
//
// Shift amount varies across warp lanes (mantissa alignment shift in
// fadd, normalization shift in fmul), so any branch on `n` causes
// divergence. Use `select` (compiles to cmov) to produce both halves
// unconditionally.
//
// Shifting a u64 by >= 64 is Undefined Behavior in C++/MSL — the LLVM
// MSL compiler is allowed to assume it never happens and delete code
// that depends on it. We therefore always compute `64u - max(n_lo, 1u)`
// and use `select` to discard the result when n_lo == 0, instead of
// branching to skip the shift.
static __attribute__((always_inline)) U128 u128_shl(U128 a, uint n) {
    uint n_lo = n & 63u;
    bool n_cross = (n & 64u) != 0u;  // true when n >= 64
    bool n_zero  = n_lo == 0u;
    // Cross-lane contribution: a.lo >> (64 - n_lo). When n_lo == 0 we
    // avoid the UB shift-by-64 by computing a safe placeholder.
    ulong cross = select(a.lo >> (64u - max(n_lo, 1u)), 0UL, n_zero);
    ulong within_hi = (a.hi << n_lo) | cross;
    ulong within_lo = a.lo << n_lo;
    ulong out_hi = select(within_hi, within_lo, n_cross);
    ulong out_lo = select(within_lo, 0UL,       n_cross);
    return U128{out_hi, out_lo};
}

// Right shift by `n` (0 <= n < 128), branch-free.
static __attribute__((always_inline)) U128 u128_shr(U128 a, uint n) {
    uint n_lo = n & 63u;
    bool n_cross = (n & 64u) != 0u;
    bool n_zero  = n_lo == 0u;
    ulong cross = select(a.hi << (64u - max(n_lo, 1u)), 0UL, n_zero);
    ulong within_hi = a.hi >> n_lo;
    ulong within_lo = (a.lo >> n_lo) | cross;
    ulong out_hi = select(within_hi, 0UL,       n_cross);
    ulong out_lo = select(within_lo, within_hi, n_cross);
    return U128{out_hi, out_lo};
}

// Bitwise OR with a single low-bit sticky flag (branch-free).
static __attribute__((always_inline)) U128 u128_or_sticky(U128 a, bool sticky) {
    return U128{a.hi, a.lo | (ulong)sticky};
}

// Count leading zeros of a 128-bit value (0..=128). Branch-free via
// `select` so warp lanes with different leading-zero counts don't
// diverge on the hi-word test.
static __attribute__((always_inline)) uint u128_clz(U128 a) {
    uint hi_clz = (uint)clz(a.hi);
    uint lo_clz = 64u + (uint)clz(a.lo);
    return select(hi_clz, lo_clz, a.hi == 0UL);
}

// 64 × 64 → 128 multiply via four 32×32→64 partial products. No overflow.
static __attribute__((always_inline)) U128 u128_mul_u64(ulong a, ulong b) {
    uint a_lo = (uint)(a & 0xFFFFFFFFu);
    uint a_hi = (uint)(a >> 32);
    uint b_lo = (uint)(b & 0xFFFFFFFFu);
    uint b_hi = (uint)(b >> 32);

    ulong lolo = (ulong)a_lo * (ulong)b_lo;
    ulong lohi = (ulong)a_lo * (ulong)b_hi;
    ulong hilo = (ulong)a_hi * (ulong)b_lo;
    ulong hihi = (ulong)a_hi * (ulong)b_hi;

    ulong mid = (lolo >> 32) + (lohi & 0xFFFFFFFFUL) + (hilo & 0xFFFFFFFFUL);
    ulong result_lo = (mid << 32) | (lolo & 0xFFFFFFFFUL);
    ulong result_hi = hihi + (lohi >> 32) + (hilo >> 32) + (mid >> 32);
    return U128{result_hi, result_lo};
}

// --- Rounding --------------------------------------------------------------

// Round a 128-bit intermediate (mantissa at bits [116:64], fraction at
// [63:0]) to a 53-bit mantissa + exponent adjustment (0 or 1).
struct RoundResult { ulong mantissa; int exp_adj; };
static __attribute__((always_inline)) RoundResult round_128(U128 m, ulong sign, uint mode) {
    // Mantissa is bits [127:64] of m_128 (our U128.hi). Top 11 bits
    // are zero after normalization, so m.hi == 53-bit mantissa.
    ulong mantissa = m.hi;
    ulong guard = (m.lo >> 63) & 1UL;
    ulong round_bit = (m.lo >> 62) & 1UL;
    ulong sticky = ((m.lo & ((1UL << 62) - 1UL)) != 0UL) ? 1UL : 0UL;

    // Branch-free per-mode round_up. round_mode may differ across warp
    // lanes (e.g. when each lane carries its own dynamic mode), so an
    // if/else-if ladder would serialize up to 3 branches per op. Compute
    // all modes' predicates and select — cheap since each is 2-3 ops.
    ulong grs_any = guard | round_bit | sticky;
    bool round_near = (guard == 1UL) && ((round_bit | sticky | (mantissa & 1UL)) != 0UL);
    bool round_down = (sign == 1UL) && (grs_any != 0UL);
    bool round_up_mode = (sign == 0UL) && (grs_any != 0UL);
    bool round_up = select(
        select(round_near, round_down, mode == RMODE_DOWN),
        select(round_up_mode, false,    mode == RMODE_ZERO),
        mode >= RMODE_UP);

    if (round_up) {
        ulong bumped = mantissa + 1UL;
        if ((bumped >> 53) != 0UL) {
            return RoundResult{bumped >> 1, 1};
        }
        return RoundResult{bumped, 0};
    }
    return RoundResult{mantissa, 0};
}

static __attribute__((always_inline)) ulong overflow_pack(ulong sign, uint mode) {
    bool is_neg = (sign == 1UL);
    ulong to_inf = (sign << 63) | POS_INF;
    ulong to_max = (sign << 63) | (POS_INF - 1UL);
    if (mode == RMODE_NEAREST) return to_inf;
    if (mode == RMODE_UP)      return is_neg ? to_max : to_inf;
    if (mode == RMODE_DOWN)    return is_neg ? to_inf : to_max;
    /* RMODE_ZERO */           return to_max;
}

static __attribute__((always_inline)) ulong pack(ulong sign, int exp_unbiased, ulong mantissa, uint mode) {
    int biased = exp_unbiased + EXP_BIAS;
    if (biased > 2046) return overflow_pack(sign, mode);
    if (biased < 1) {
        // Legacy pack: GRS info is gone by now, callers that need
        // gradual-underflow rounding should use round_and_pack instead.
        return sign << 63;
    }
    return (sign << 63) | (((ulong)biased & EXP_MASK) << MANT_BITS) | (mantissa & MANT_MASK);
}

// Single-rounding round + pack. Mirrors round_and_pack in softfloat_ref.rs.
// `m` has its 53-bit mantissa at bits [116:64] (i.e. m.hi after the
// caller's normalization); bits [63:0] are guard / round / sticky info.
// `exp_pre_round` is the unbiased exp assuming MSB at bit 116.
static __attribute__((always_inline)) ulong round_and_pack(U128 m, ulong sign, int exp_pre_round, uint mode) {
    int biased_pre = exp_pre_round + EXP_BIAS;

    if (biased_pre > 2046) return overflow_pack(sign, mode);

#ifdef SOFTFLOAT_FTZ
    if (biased_pre < 1) {
        // FTZ: any sub-1.0×2^Emin result collapses to zero, sign preserved.
        return sign << 63;
    }
#endif

    if (biased_pre < 1) {
        // Subnormal output — shift right by `shift = 1 - biased_pre`
        // (with sticky-jam) so the new LSB sits at the subnormal field's
        // LSB, then round once.
        uint shift = (uint)(1 - biased_pre);
        U128 m_shifted;
        bool dropped_any;
        if (shift >= 128u) {
            m_shifted = U128{0UL, 0UL};
            dropped_any = !u128_is_zero(m);
        } else {
            // Capture all bits below position `shift` for sticky.
            U128 below = m;
            // dropped_any = (m & ((1 << shift) - 1)) != 0
            if (shift == 0u) {
                dropped_any = false;
            } else if (shift <= 64u) {
                ulong mask = (shift == 64u) ? ~0UL : ((1UL << shift) - 1UL);
                dropped_any = (below.lo & mask) != 0UL;
            } else {
                uint rem = shift - 64u;
                ulong hi_mask = (rem == 64u) ? ~0UL : ((1UL << rem) - 1UL);
                dropped_any = (below.lo != 0UL) || ((below.hi & hi_mask) != 0UL);
            }
            m_shifted = u128_shr(m, shift);
        }
        m_shifted = u128_or_sticky(m_shifted, dropped_any);

        ulong new_mant = m_shifted.hi;
        ulong guard = (m_shifted.lo >> 63) & 1UL;
        ulong round_bit = (m_shifted.lo >> 62) & 1UL;
        ulong sticky = ((m_shifted.lo & ((1UL << 62) - 1UL)) != 0UL) ? 1UL : 0UL;

        ulong grs_any = guard | round_bit | sticky;
        bool round_near = (guard == 1UL) && ((round_bit | sticky | (new_mant & 1UL)) != 0UL);
        bool round_dn = (sign == 1UL) && (grs_any != 0UL);
        bool round_up_mode = (sign == 0UL) && (grs_any != 0UL);
        bool round_up = select(
            select(round_near, round_dn, mode == RMODE_DOWN),
            select(round_up_mode, false, mode == RMODE_ZERO),
            mode >= RMODE_UP);

        ulong mant_after = new_mant + (round_up ? 1UL : 0UL);
        if ((mant_after >> MANT_BITS) != 0UL) {
            // Round-up promoted us to the smallest normal.
            return (sign << 63) | (1UL << MANT_BITS);
        }
        return (sign << 63) | (mant_after & MANT_MASK);
    }

    // Normal output.
    RoundResult r = round_128(m, sign, mode);
    int biased = biased_pre + r.exp_adj;
    if (biased > 2046) return overflow_pack(sign, mode);
    return (sign << 63) | (((ulong)biased & EXP_MASK) << MANT_BITS) | (r.mantissa & MANT_MASK);
}

struct Unpacked { ulong sign; int exp; ulong mantissa; uint class_; };

static Unpacked unpack(ulong x) {
    ulong sign = x >> 63;
    int exp_raw = (int)((x >> MANT_BITS) & EXP_MASK);
    ulong mant_raw = x & MANT_MASK;
    uint class_ = classify(x);

    int exp;
    ulong mantissa;
    if (class_ == CLASS_NORMAL) {
        exp = exp_raw - EXP_BIAS;
        mantissa = mant_raw | IMPLICIT_BIT;
    } else if (class_ == CLASS_SUBNORMAL) {
        // Normalize: shift left until the implicit bit (52) is set, so
        // the rest of the arithmetic loop sees a "very small normal".
        // mant_raw's set bits are in [0..51], so clz(mant_raw) ∈ [12..63].
        // shift = clz - 11.
        int shift = (int)clz(mant_raw) - 11;
        exp = -1022 - shift;
        mantissa = mant_raw << shift;
    } else {
        // Zero / Inf / NaN: mantissa unused by callers (special-case
        // dispatch handles these).
        exp = 0;
        mantissa = 0UL;
    }
    return Unpacked{sign, exp, mantissa, class_};
}

// --- Ops -------------------------------------------------------------------

// Slow path: handles non-normal inputs (NaN/Inf/Zero/Subnormal) and the
// subnormal-output fall-through from softfloat_add's fast path. Kept as
// a separate non-inlined function so the hot softfloat_add stays small
// and the Metal compiler can choose to inline it into kernel loops.
static __attribute__((noinline)) ulong softfloat_add_slow(ulong a, ulong b, uint mode, Unpacked ua, Unpacked ub) {
    if (ua.class_ == CLASS_NAN || ub.class_ == CLASS_NAN) return CANONICAL_QNAN;
    if (ua.class_ == CLASS_INF && ub.class_ == CLASS_INF) {
        return (ua.sign == ub.sign) ? ((ua.sign << 63) | POS_INF) : CANONICAL_QNAN;
    }
    if (ua.class_ == CLASS_INF) return (ua.sign << 63) | POS_INF;
    if (ub.class_ == CLASS_INF) return (ub.sign << 63) | POS_INF;
    if (ua.class_ == CLASS_ZERO && ub.class_ == CLASS_ZERO) {
        ulong sign;
        if (ua.sign == ub.sign) sign = ua.sign;
        else                    sign = (mode == RMODE_DOWN) ? 1UL : 0UL;
        return sign << 63;
    }
    if (ua.class_ == CLASS_ZERO) return b;
    if (ub.class_ == CLASS_ZERO) return a;

    Unpacked big, sml;
    if (ua.exp > ub.exp || (ua.exp == ub.exp && ua.mantissa >= ub.mantissa)) {
        big = ua; sml = ub;
    } else {
        big = ub; sml = ua;
    }

    uint shift = (uint)(big.exp - sml.exp);
    U128 big_128 = u128_make(big.mantissa, 0UL);
    U128 sml_raw = u128_make(sml.mantissa, 0UL);

    U128 sml_aligned;
    if (shift >= 128u) {
        sml_aligned = u128_make(0UL, (sml.mantissa != 0UL) ? 1UL : 0UL);
    } else if (shift == 0u) {
        sml_aligned = sml_raw;
    } else {
        U128 shifted = u128_shr(sml_raw, shift);
        // Branch-free sticky.
        ulong mask_lo = (shift < 64u) ? ((1UL << shift) - 1UL) : ~0UL;
        bool sticky_lt64 = (sml_raw.lo & mask_lo) != 0UL;
        uint rem = (shift >= 64u) ? (shift - 64u) : 0u;
        ulong hi_mask = (rem == 0u) ? 0UL : ((1UL << rem) - 1UL);
        bool sticky_ge64 = (sml_raw.lo != 0UL) || ((sml_raw.hi & hi_mask) != 0UL);
        bool sticky = (shift < 64u) ? sticky_lt64 : sticky_ge64;
        sml_aligned = u128_or_sticky(shifted, sticky);
    }

    ulong result_sign;
    U128 result_128;
    int exp_adjust;

    if (big.sign == sml.sign) {
        U128 sum = u128_add(big_128, sml_aligned);
        if ((sum.hi >> 53) != 0UL) {
            bool sticky = (sum.lo & 1UL) != 0UL;
            sum = u128_shr(sum, 1);
            sum = u128_or_sticky(sum, sticky);
            exp_adjust = 1;
        } else {
            exp_adjust = 0;
        }
        result_sign = big.sign;
        result_128 = sum;
    } else {
        U128 diff = u128_sub(big_128, sml_aligned);
        if (u128_is_zero(diff)) {
            return (mode == RMODE_DOWN) ? (1UL << 63) : 0UL;
        }
        uint lz = u128_clz(diff);
        int msb_pos = 127 - (int)lz;
        int shift_left = 116 - msb_pos;
        if (shift_left > 0) {
            diff = u128_shl(diff, (uint)shift_left);
        } else if (shift_left < 0) {
            uint n = (uint)(-shift_left);
            ulong mask_lo = (n < 64u) ? ((1UL << n) - 1UL) : ~0UL;
            bool sticky_lt64 = (diff.lo & mask_lo) != 0UL;
            uint rem = (n >= 64u) ? (n - 64u) : 0u;
            ulong hi_mask = (rem == 0u) ? 0UL : ((1UL << rem) - 1UL);
            bool sticky_ge64 = (diff.lo != 0UL) || ((diff.hi & hi_mask) != 0UL);
            bool sticky = (n < 64u) ? sticky_lt64 : sticky_ge64;
            diff = u128_or_sticky(u128_shr(diff, n), sticky);
        }
        exp_adjust = -shift_left;
        result_sign = big.sign;
        result_128 = diff;
    }

    return round_and_pack(result_128, result_sign, big.exp + exp_adjust, mode);
}

ulong softfloat_add(ulong a, ulong b, uint mode) {
    Unpacked ua = unpack(a);
    Unpacked ub = unpack(b);

#ifdef SOFTFLOAT_FTZ
    // Treat subnormal inputs as ±0; subnormal outputs are flushed
    // implicitly because the slow path's round_and_pack is bypassed
    // when both inputs are zero.
    if (ua.class_ == CLASS_SUBNORMAL) {
        if (ub.class_ == CLASS_SUBNORMAL) {
            ulong sign = (ua.sign == ub.sign)
                ? ua.sign
                : ((mode == RMODE_DOWN) ? 1UL : 0UL);
            return sign << 63;
        }
        return b;
    }
    if (ub.class_ == CLASS_SUBNORMAL) return a;
#endif

    // Fast path: both operands normal. Uses 56-bit (u64-mantissa)
    // arithmetic with bottom 3 bits as G/R/S — bit-exact vs the u128
    // path on both-normal inputs. Placed BEFORE the special-case
    // dispatch so the common case (tight FP loops, benchmarks) skips
    // the 5+ NaN/Inf/Zero branches. Falls through to the u128 path
    // when subnormal output is detected.
    if (ua.class_ == CLASS_NORMAL && ub.class_ == CLASS_NORMAL) {
        Unpacked big_n, sml_n;
        if (ua.exp > ub.exp || (ua.exp == ub.exp && ua.mantissa >= ub.mantissa)) {
            big_n = ua; sml_n = ub;
        } else {
            big_n = ub; sml_n = ua;
        }
        uint shift_n = (uint)(big_n.exp - sml_n.exp);
        ulong big_sig = big_n.mantissa << 3;
        ulong sml_sig_raw = sml_n.mantissa << 3;

        ulong sml_sig;
        if (shift_n >= 64u)      sml_sig = (sml_sig_raw != 0UL) ? 1UL : 0UL;
        else if (shift_n == 0u)  sml_sig = sml_sig_raw;
        else {
            ulong dropped = sml_sig_raw << (64u - shift_n);
            sml_sig = (sml_sig_raw >> shift_n) | ((dropped != 0UL) ? 1UL : 0UL);
        }

        ulong rs;
        ulong r_sig;
        int exa;
        if (big_n.sign == sml_n.sign) {
            ulong sum = big_sig + sml_sig;
            if ((sum >> 56) != 0UL) {
                ulong st = sum & 1UL;
                sum = (sum >> 1) | st;
                exa = 1;
            } else exa = 0;
            rs = big_n.sign;
            r_sig = sum;
        } else {
            ulong diff = big_sig - sml_sig;
            if (diff == 0UL) {
                return (mode == RMODE_DOWN) ? (1UL << 63) : 0UL;
            }
            int lz = (int)clz(diff);
            int sl = lz - 8;
            r_sig = (sl > 0) ? (diff << sl) : diff;
            exa = -sl;
            rs = big_n.sign;
        }

        ulong guard = (r_sig >> 2) & 1UL;
        ulong round_bit = (r_sig >> 1) & 1UL;
        ulong sticky_b = r_sig & 1UL;
        ulong mant_n = r_sig >> 3;
        ulong grs_any = guard | round_bit | sticky_b;
        bool round_near = (guard == 1UL) && ((round_bit | sticky_b | (mant_n & 1UL)) != 0UL);
        bool round_dn = (rs == 1UL) && (grs_any != 0UL);
        bool round_up_mode = (rs == 0UL) && (grs_any != 0UL);
        bool round_up = select(
            select(round_near, round_dn, mode == RMODE_DOWN),
            select(round_up_mode, false, mode == RMODE_ZERO),
            mode >= RMODE_UP);
        int round_bump = 0;
        if (round_up) {
            ulong bumped = mant_n + 1UL;
            if ((bumped >> 53) != 0UL) {
                mant_n = bumped >> 1;
                round_bump = 1;
            } else {
                mant_n = bumped;
            }
        }

        int final_exp = big_n.exp + exa + round_bump;
        int biased_n = final_exp + EXP_BIAS;
        // Subnormal-output: tail-call slow path for gradual underflow.
        if (biased_n >= 1) {
            if (biased_n > 2046) return overflow_pack(rs, mode);
            return (rs << 63) | (((ulong)biased_n & EXP_MASK) << MANT_BITS) | (mant_n & MANT_MASK);
        }
    }

    // Reached only for non-normal inputs or fast-path subnormal output.
    return softfloat_add_slow(a, b, mode, ua, ub);
}

__attribute__((always_inline)) ulong softfloat_sub(ulong a, ulong b, uint mode) {
    return softfloat_add(a, b ^ (1UL << 63), mode);
}

// Pre-unpacked fadd: takes already-unpacked Unpacked structs (skipping
// the unpack on entry) and returns Unpacked (skipping the final pack).
// For tight loops where the next op also wants unpacked state, this
// eliminates ~15-20 IR ops per call (the pack/unpack churn that would
// otherwise cancel between adjacent calls).
//
// Only handles the both-normal fast path. Result is normal (caller must
// re-pack at loop exit). Masked / pre-validated normal inputs always fall here.
struct Unpacked6 { ulong sign; int exp; ulong mantissa; };

static __attribute__((always_inline)) Unpacked6 softfloat_add_unp_normal(
    Unpacked ua, Unpacked ub, uint mode)
{
    // Branch-free big/sml sort.
    bool a_bigger = (ua.exp > ub.exp) || ((ua.exp == ub.exp) && (ua.mantissa >= ub.mantissa));
    Unpacked big_n;
    Unpacked sml_n;
    big_n.sign = a_bigger ? ua.sign : ub.sign;
    big_n.exp  = a_bigger ? ua.exp  : ub.exp;
    big_n.mantissa = a_bigger ? ua.mantissa : ub.mantissa;
    sml_n.sign = a_bigger ? ub.sign : ua.sign;
    sml_n.exp  = a_bigger ? ub.exp  : ua.exp;
    sml_n.mantissa = a_bigger ? ub.mantissa : ua.mantissa;
    uint shift_n = (uint)(big_n.exp - sml_n.exp);
    ulong big_sig = big_n.mantissa << 3;
    ulong sml_sig_raw = sml_n.mantissa << 3;

    // Branch-free shift-right-jam.
    bool s_zero = shift_n == 0u;
    bool s_huge = shift_n >= 64u;
    uint shift_safe = max(shift_n, 1u);
    ulong dropped = sml_sig_raw << (64u - shift_safe);
    ulong shifted = (sml_sig_raw >> shift_safe) | ((dropped != 0UL) ? 1UL : 0UL);
    ulong sml_sig_normal = select(shifted, sml_sig_raw, s_zero);
    ulong sml_sig_huge = (sml_sig_raw != 0UL) ? 1UL : 0UL;
    ulong sml_sig = select(sml_sig_normal, sml_sig_huge, s_huge);

    ulong rs;
    ulong r_sig;
    int exa;
    if (big_n.sign == sml_n.sign) {
        ulong sum = big_sig + sml_sig;
        if ((sum >> 56) != 0UL) {
            ulong st = sum & 1UL;
            sum = (sum >> 1) | st;
            exa = 1;
        } else exa = 0;
        rs = big_n.sign;
        r_sig = sum;
    } else {
        ulong diff = big_sig - sml_sig;
        if (diff == 0UL) { Unpacked6 z = {0UL, -1023, 0UL}; return z; }
        int lz = (int)clz(diff);
        int sl = lz - 8;
        r_sig = (sl > 0) ? (diff << sl) : diff;
        exa = -sl;
        rs = big_n.sign;
    }

    ulong guard = (r_sig >> 2) & 1UL;
    ulong round_bit = (r_sig >> 1) & 1UL;
    ulong sticky_b = r_sig & 1UL;
    ulong mant_n = r_sig >> 3;
    ulong grs_any = guard | round_bit | sticky_b;
    bool round_near = (guard == 1UL) && ((round_bit | sticky_b | (mant_n & 1UL)) != 0UL);
    bool round_dn = (rs == 1UL) && (grs_any != 0UL);
    bool round_up_mode = (rs == 0UL) && (grs_any != 0UL);
    bool round_up = select(
        select(round_near, round_dn, mode == RMODE_DOWN),
        select(round_up_mode, false, mode == RMODE_ZERO),
        mode >= RMODE_UP);
    int round_bump = 0;
    if (round_up) {
        ulong bumped = mant_n + 1UL;
        if ((bumped >> 53) != 0UL) {
            mant_n = bumped >> 1;
            round_bump = 1;
        } else {
            mant_n = bumped;
        }
    }
    Unpacked6 r = {rs, big_n.exp + exa + round_bump, mant_n};
    return r;
}

// Pre-unpacked fmul / fdiv / fsqrt / fma / fsub: same idea as
// softfloat_add_unp_normal — caller maintains Unpacked state across
// op chains, _unp variants skip pack/unpack churn and are small enough
// to inline. All assume both operands are normal (caller's responsibility).

static __attribute__((always_inline)) Unpacked6 softfloat_sub_unp_normal(
    Unpacked ua, Unpacked ub, uint mode)
{
    Unpacked ub_neg = {ub.sign ^ 1UL, ub.exp, ub.mantissa, ub.class_};
    return softfloat_add_unp_normal(ua, ub_neg, mode);
}

static __attribute__((always_inline)) Unpacked6 softfloat_mul_unp_normal(
    Unpacked ua, Unpacked ub, uint mode)
{
    ulong sign = ua.sign ^ ub.sign;
    int exp = ua.exp + ub.exp;
    U128 product = u128_mul_u64(ua.mantissa, ub.mantissa);
    int exp_adj;
    if ((product.hi >> (105 - 64)) != 0UL) {
        bool sticky = (product.lo & 1UL) != 0UL;
        product = u128_shr(product, 1);
        product = u128_or_sticky(product, sticky);
        exp_adj = 1;
    } else {
        exp_adj = 0;
    }
    U128 m = u128_shl(product, 12);
    // Inline normal-only round_128 (skip subnormal / overflow).
    ulong mantissa = m.hi;
    ulong guard = (m.lo >> 63) & 1UL;
    ulong round_bit = (m.lo >> 62) & 1UL;
    ulong sticky = ((m.lo & ((1UL << 62) - 1UL)) != 0UL) ? 1UL : 0UL;
    ulong grs_any = guard | round_bit | sticky;
    bool round_near = (guard == 1UL) && ((round_bit | sticky | (mantissa & 1UL)) != 0UL);
    bool round_dn = (sign == 1UL) && (grs_any != 0UL);
    bool round_up_mode = (sign == 0UL) && (grs_any != 0UL);
    bool round_up = select(
        select(round_near, round_dn, mode == RMODE_DOWN),
        select(round_up_mode, false, mode == RMODE_ZERO),
        mode >= RMODE_UP);
    int exp_round = 0;
    if (round_up) {
        ulong bumped = mantissa + 1UL;
        if ((bumped >> 53) != 0UL) {
            mantissa = bumped >> 1;
            exp_round = 1;
        } else {
            mantissa = bumped;
        }
    }
    Unpacked6 r = {sign, exp + exp_adj + exp_round, mantissa & MANT_MASK | IMPLICIT_BIT};
    return r;
}

// fdiv_unp_normal: only the Berkeley-style algorithm, no special cases.
// Both inputs assumed normal.
static __attribute__((always_inline)) Unpacked6 softfloat_div_unp_normal(
    Unpacked ua, Unpacked ub, uint mode)
{
    ulong sign_z = ua.sign ^ ub.sign;
    int exp_a_biased = ua.exp + EXP_BIAS;
    int exp_b_biased = ub.exp + EXP_BIAS;
    int exp_z_biased = exp_a_biased - exp_b_biased + 0x3FE;

    ulong sig_a_init = ua.mantissa;
    ulong sig_b = ub.mantissa;
    ulong sig_a;
    if (sig_a_init < sig_b) {
        exp_z_biased -= 1;
        sig_a = sig_a_init << 11;
    } else {
        sig_a = sig_a_init << 10;
    }
    ulong sig_b_11 = sig_b << 11;
    uint sig_b_hi32 = (uint)(sig_b_11 >> 32);

    uint recip32 = approx_recip32_1(sig_b_hi32) - 2u;
    uint sig32_z = (uint)(((ulong)(uint)(sig_a >> 32) * (ulong)recip32) >> 32);
    uint double_term = sig32_z << 1;
    uint sig_b_tail = (uint)sig_b_11 >> 4;
    ulong rem = ((sig_a - (ulong)double_term * (ulong)sig_b_hi32) << 28)
              - (ulong)double_term * (ulong)sig_b_tail;
    uint q = (uint)(((ulong)(uint)(rem >> 32) * (ulong)recip32) >> 32) + 4u;
    ulong sig_z = ((ulong)sig32_z << 32) + ((ulong)q << 4);

    if ((sig_z & 0x1FFUL) < (ulong)(4u << 4)) {
        uint q_corr = q & ~7u;
        sig_z &= ~(ulong)0x7FUL;
        uint double_term2 = q_corr << 1;
        rem = ((rem - (ulong)double_term2 * (ulong)sig_b_hi32) << 28)
            - (ulong)double_term2 * (ulong)sig_b_tail;
        if ((rem & 0x8000000000000000UL) != 0UL) sig_z -= 1UL << 7;
        else if (rem != 0UL) sig_z |= 1UL;
    }

    ulong round_bits = sig_z & 0x3FFUL;
    ulong inc;
    if (mode == RMODE_NEAREST)                       inc = 0x200UL;
    else if (mode == RMODE_UP   && sign_z == 0UL)    inc = 0x3FFUL;
    else if (mode == RMODE_DOWN && sign_z != 0UL)    inc = 0x3FFUL;
    else                                             inc = 0UL;
    ulong rounded_sig = (sig_z + inc) >> 10;
    if (mode == RMODE_NEAREST && round_bits == 0x200UL) rounded_sig &= ~1UL;

    int final_exp_biased;
    ulong final_mantissa;
    if ((rounded_sig >> 53) != 0UL) {
        final_exp_biased = exp_z_biased + 2;
        final_mantissa = rounded_sig >> 1;
    } else {
        final_exp_biased = exp_z_biased + 1;
        final_mantissa = rounded_sig;
    }
    Unpacked6 r = {sign_z, final_exp_biased - EXP_BIAS, final_mantissa};
    return r;
}

// fma_unp_normal: (a × b) + c with one rounding, normal-only.
static __attribute__((always_inline)) Unpacked6 softfloat_fma_unp_normal(
    Unpacked ua, Unpacked ub, Unpacked uc, uint mode)
{
    ulong prod_sign = ua.sign ^ ub.sign;
    U128 prod_sig = u128_mul_u64(ua.mantissa, ub.mantissa);
    bool prod_msb_high = (prod_sig.hi >> (105 - 64)) != 0UL;
    U128 m_prod = u128_shl(prod_sig, prod_msb_high ? 11u : 12u);
    int exp_prod = ua.exp + ub.exp + (prod_msb_high ? 1 : 0);
    U128 c_staged = U128{uc.mantissa, 0UL};
    int exp_diff = exp_prod - uc.exp;
    U128 m_combined;
    int exp_combined;
    ulong sign_combined;

    if (prod_sign == uc.sign) {
        U128 big; U128 sml; int big_exp; ulong sign_out; uint shift_u;
        if (exp_diff >= 0) {
            big = m_prod; sml = c_staged; big_exp = exp_prod;
            sign_out = prod_sign; shift_u = (uint)exp_diff;
        } else {
            big = c_staged; sml = m_prod; big_exp = uc.exp;
            sign_out = uc.sign; shift_u = (uint)(-exp_diff);
        }
        U128 sml_aligned;
        if (shift_u >= 128u) {
            sml_aligned = U128{0UL, u128_is_zero(sml) ? 0UL : 1UL};
        } else if (shift_u == 0u) {
            sml_aligned = sml;
        } else {
            // Branch-free sticky: compute both <64 and >=64 cases, select.
            ulong mask_lo = (shift_u < 64u) ? ((1UL << shift_u) - 1UL) : ~0UL;
            bool sticky_lt64 = (sml.lo & mask_lo) != 0UL;
            uint rem = (shift_u >= 64u) ? (shift_u - 64u) : 0u;
            ulong hi_mask = (rem == 0u) ? 0UL : ((1UL << rem) - 1UL);
            bool sticky_ge64 = (sml.lo != 0UL) || ((sml.hi & hi_mask) != 0UL);
            bool sticky = (shift_u < 64u) ? sticky_lt64 : sticky_ge64;
            sml_aligned = u128_or_sticky(u128_shr(sml, shift_u), sticky);
        }
        U128 sum = u128_add(big, sml_aligned);
        int exp_adj = 0;
        if ((sum.hi >> (117 - 64)) != 0UL) {
            bool sticky = (sum.lo & 1UL) != 0UL;
            sum = u128_shr(sum, 1u);
            sum = u128_or_sticky(sum, sticky);
            exp_adj = 1;
        }
        m_combined = sum;
        exp_combined = big_exp + exp_adj;
        sign_combined = sign_out;
    } else {
        U128 big; U128 sml; int big_exp; ulong sign_out; uint shift_u;
        if (exp_diff > 0) {
            big = m_prod; sml = c_staged; big_exp = exp_prod;
            sign_out = prod_sign; shift_u = (uint)exp_diff;
        } else if (exp_diff < 0) {
            big = c_staged; sml = m_prod; big_exp = uc.exp;
            sign_out = uc.sign; shift_u = (uint)(-exp_diff);
        } else if (u128_cmp(m_prod, c_staged) >= 0) {
            big = m_prod; sml = c_staged; big_exp = exp_prod;
            sign_out = prod_sign; shift_u = 0u;
        } else {
            big = c_staged; sml = m_prod; big_exp = uc.exp;
            sign_out = uc.sign; shift_u = 0u;
        }
        U128 sml_aligned;
        if (shift_u >= 128u) {
            sml_aligned = U128{0UL, u128_is_zero(sml) ? 0UL : 1UL};
        } else if (shift_u == 0u) {
            sml_aligned = sml;
        } else {
            // Branch-free sticky: compute both <64 and >=64 cases, select.
            ulong mask_lo = (shift_u < 64u) ? ((1UL << shift_u) - 1UL) : ~0UL;
            bool sticky_lt64 = (sml.lo & mask_lo) != 0UL;
            uint rem = (shift_u >= 64u) ? (shift_u - 64u) : 0u;
            ulong hi_mask = (rem == 0u) ? 0UL : ((1UL << rem) - 1UL);
            bool sticky_ge64 = (sml.lo != 0UL) || ((sml.hi & hi_mask) != 0UL);
            bool sticky = (shift_u < 64u) ? sticky_lt64 : sticky_ge64;
            sml_aligned = u128_or_sticky(u128_shr(sml, shift_u), sticky);
        }
        U128 diff = u128_sub(big, sml_aligned);
        if (u128_is_zero(diff)) {
            // Exact cancellation. unp variant returns zero (no DOWN-mode -0).
            Unpacked6 z = {0UL, -1023, 0UL};
            return z;
        }
        uint lz = u128_clz(diff);
        int msb_pos = 127 - (int)lz;
        int shift_left = 116 - msb_pos;
        U128 normalized;
        if (shift_left > 0) {
            normalized = u128_shl(diff, (uint)shift_left);
        } else if (shift_left < 0) {
            uint n = (uint)(-shift_left);
            ulong mask_lo = (n < 64u) ? ((1UL << n) - 1UL) : ~0UL;
            bool sticky_lt64 = (diff.lo & mask_lo) != 0UL;
            uint rem = (n >= 64u) ? (n - 64u) : 0u;
            ulong hi_mask = (rem == 0u) ? 0UL : ((1UL << rem) - 1UL);
            bool sticky_ge64 = (diff.lo != 0UL) || ((diff.hi & hi_mask) != 0UL);
            bool sticky = (n < 64u) ? sticky_lt64 : sticky_ge64;
            normalized = u128_or_sticky(u128_shr(diff, n), sticky);
        } else {
            normalized = diff;
        }
        m_combined = normalized;
        exp_combined = big_exp - shift_left;
        sign_combined = sign_out;
    }

    // round_128 inline (normal-only).
    ulong mantissa = m_combined.hi;
    ulong guard = (m_combined.lo >> 63) & 1UL;
    ulong round_bit = (m_combined.lo >> 62) & 1UL;
    ulong sticky = ((m_combined.lo & ((1UL << 62) - 1UL)) != 0UL) ? 1UL : 0UL;
    ulong grs_any = guard | round_bit | sticky;
    bool round_near = (guard == 1UL) && ((round_bit | sticky | (mantissa & 1UL)) != 0UL);
    bool round_dn = (sign_combined == 1UL) && (grs_any != 0UL);
    bool round_up_mode = (sign_combined == 0UL) && (grs_any != 0UL);
    bool round_up = select(
        select(round_near, round_dn, mode == RMODE_DOWN),
        select(round_up_mode, false, mode == RMODE_ZERO),
        mode >= RMODE_UP);
    int exp_round = 0;
    if (round_up) {
        ulong bumped = mantissa + 1UL;
        if ((bumped >> 53) != 0UL) {
            mantissa = bumped >> 1;
            exp_round = 1;
        } else {
            mantissa = bumped;
        }
    }
    Unpacked6 r = {sign_combined, exp_combined + exp_round, mantissa};
    return r;
}

// fsqrt_unp_normal: Berkeley-style sqrt; assumes positive normal input.
static __attribute__((always_inline)) Unpacked6 softfloat_sqrt_unp_normal(
    Unpacked ua, uint mode)
{
    int exp_a_biased = ua.exp + EXP_BIAS;
    uint odd_exp_a = (uint)(exp_a_biased & 1);
    int exp_z_biased = ((exp_a_biased - 0x3FF) >> 1) + 0x3FE;
    ulong sig_a = ua.mantissa;
    uint sig32_a = (uint)(sig_a >> 21);
    uint recip_sqrt32 = approx_recip_sqrt32_1(odd_exp_a, sig32_a);
    uint sig32_z = (uint)(((ulong)sig32_a * (ulong)recip_sqrt32) >> 32);
    if (odd_exp_a != 0u) { sig_a <<= 8; sig32_z >>= 1; } else { sig_a <<= 9; }
    ulong rem = sig_a - (ulong)sig32_z * (ulong)sig32_z;
    uint q = (uint)((((ulong)(uint)(rem >> 2)) * (ulong)recip_sqrt32) >> 32);
    ulong sig_z = ((ulong)sig32_z << 32) | (1UL << 5);
    sig_z += ((ulong)q << 3);
    if ((sig_z & 0x1FFUL) < 0x22UL) {
        sig_z &= ~(ulong)0x3FUL;
        ulong shifted = sig_z >> 6;
        ulong rem2 = (sig_a << 52) - shifted * shifted;
        if ((rem2 & 0x8000000000000000UL) != 0UL) sig_z -= 1UL;
        else if (rem2 != 0UL) sig_z |= 1UL;
    }
    ulong round_bits = sig_z & 0x3FFUL;
    ulong increment;
    if (mode == RMODE_NEAREST)   increment = 0x200UL;
    else if (mode == RMODE_UP)   increment = 0x3FFUL;
    else                         increment = 0UL;
    ulong rounded_sig = (sig_z + increment) >> 10;
    if (mode == RMODE_NEAREST && round_bits == 0x200UL) rounded_sig &= ~1UL;
    int final_exp_biased;
    ulong final_mantissa;
    if ((rounded_sig >> 53) != 0UL) {
        final_exp_biased = exp_z_biased + 2;
        final_mantissa = rounded_sig >> 1;
    } else {
        final_exp_biased = exp_z_biased + 1;
        final_mantissa = rounded_sig;
    }
    Unpacked6 r = {0UL, final_exp_biased - EXP_BIAS, final_mantissa};
    return r;
}



ulong softfloat_mul(ulong a, ulong b, uint mode) {
    Unpacked ua = unpack(a);
    Unpacked ub = unpack(b);
    ulong sign = ua.sign ^ ub.sign;

#ifdef SOFTFLOAT_FTZ
    // FTZ: any subnormal input collapses the product to ±0 (subnormal × x
    // produces a tiny result that would FTZ on output anyway).
    if (ua.class_ == CLASS_SUBNORMAL || ub.class_ == CLASS_SUBNORMAL) {
        if (ua.class_ == CLASS_NAN || ub.class_ == CLASS_NAN) return CANONICAL_QNAN;
        if (ua.class_ == CLASS_INF || ub.class_ == CLASS_INF) return CANONICAL_QNAN;
        return sign << 63;
    }
#endif

    // Special-case dispatch only when an input isn't normal. Hoists 4
    // branches out of the common (hot) path with no code dup.
    if (ua.class_ != CLASS_NORMAL || ub.class_ != CLASS_NORMAL) {
        if (ua.class_ == CLASS_NAN || ub.class_ == CLASS_NAN) return CANONICAL_QNAN;
        if ((ua.class_ == CLASS_ZERO && ub.class_ == CLASS_INF)
         || (ua.class_ == CLASS_INF  && ub.class_ == CLASS_ZERO)) return CANONICAL_QNAN;
        if (ua.class_ == CLASS_INF || ub.class_ == CLASS_INF) return (sign << 63) | POS_INF;
        if (ua.class_ == CLASS_ZERO || ub.class_ == CLASS_ZERO) return sign << 63;
        // Subnormal-input path falls through to the algorithm below.
    }

    int exp = ua.exp + ub.exp;
    U128 product = u128_mul_u64(ua.mantissa, ub.mantissa);

    int exp_adj;
    if ((product.hi >> (105 - 64)) != 0UL) { // bit 105 set?
        bool sticky = (product.lo & 1UL) != 0UL;
        product = u128_shr(product, 1);
        product = u128_or_sticky(product, sticky);
        exp_adj = 1;
    } else {
        exp_adj = 0;
    }
    // Shift MSB (bit 104) up to bit 116.
    U128 m = u128_shl(product, 12);
    return round_and_pack(m, sign, exp + exp_adj, mode);
}

// --- Berkeley SoftFloat reciprocal (non-sqrt) tables (BSD-3-Clause) ---
// Verbatim from softfloat-3e/source/s_approxRecip_1Ks.c
constant ushort APPROX_RECIP_K0S[16] = {
    0xFFC4, 0xF0BE, 0xE363, 0xD76F, 0xCCAD, 0xC2F0, 0xBA16, 0xB201,
    0xAA97, 0xA3C6, 0x9D7A, 0x97A6, 0x923C, 0x8D32, 0x887E, 0x8417,
};
constant ushort APPROX_RECIP_K1S[16] = {
    0xF0F1, 0xD62C, 0xBFA1, 0xAC77, 0x9C0A, 0x8DDB, 0x8185, 0x76BA,
    0x6D3B, 0x64D4, 0x5D5C, 0x56B1, 0x50B6, 0x4B55, 0x4679, 0x4211,
};

// Ported from Berkeley SoftFloat `softfloat_approxRecip32_1`.
// Returns ~32-bit approximation of 2^64 / a where a ∈ [2^31, 2^32).
static uint approx_recip32_1(uint a) {
    uint index = (a >> 27) & 0xF;
    ushort eps = (ushort)(a >> 11);
    ushort r0 = APPROX_RECIP_K0S[index]
                - (ushort)((APPROX_RECIP_K1S[index] * (uint)eps) >> 20);
    uint sigma0 = ~(uint)((((ulong)r0) * (ulong)a) >> 7);
    uint r = ((uint)r0 << 16) + (uint)(((ulong)r0 * (ulong)sigma0) >> 24);
    uint sqr_sigma0 = (uint)(((ulong)sigma0 * (ulong)sigma0) >> 32);
    r += (uint)(((ulong)r * (ulong)sqr_sigma0) >> 48);
    return r;
}

// --- Berkeley SoftFloat recipSqrt tables (BSD-3-Clause, 32 bytes total) ---
// Verbatim from softfloat-3e/source/s_approxRecipSqrt_1Ks.c
constant ushort APPROX_RECIP_SQRT_K0S[16] = {
    0xB4C9, 0xFFAB, 0xAA7D, 0xF11C, 0xA1C5, 0xE4C7, 0x9A43, 0xDA29,
    0x93B5, 0xD0E5, 0x8DED, 0xC8B7, 0x88C6, 0xC16D, 0x8424, 0xBAE1,
};
constant ushort APPROX_RECIP_SQRT_K1S[16] = {
    0xA5A5, 0xEA42, 0x8C21, 0xC62D, 0x788F, 0xAA7F, 0x6928, 0x94B6,
    0x5CC7, 0x8335, 0x52A6, 0x74E2, 0x4A3E, 0x68FE, 0x432B, 0x5EFD,
};

// Ported from Berkeley SoftFloat `softfloat_approxRecipSqrt32_1`.
// Returns ~32-bit approximation of 2^32 / sqrt(a * 2^oddExp_a), bit 31
// always set. Cross-checked bit-exact against Rust reference.
static uint approx_recip_sqrt32_1(uint odd_exp_a, uint a) {
    uint index = ((a >> 27) & 0xE) + odd_exp_a;
    ushort eps = (ushort)(a >> 12);
    ushort r0 = APPROX_RECIP_SQRT_K0S[index]
                - (ushort)((APPROX_RECIP_SQRT_K1S[index] * (uint)eps) >> 20);
    uint e_sqr_r0 = (uint)r0 * (uint)r0;
    if (odd_exp_a == 0u) e_sqr_r0 <<= 1;
    uint sigma0 = ~(uint)(((ulong)e_sqr_r0 * (ulong)a) >> 23);
    uint r = ((uint)r0 << 16) + (uint)(((ulong)r0 * (ulong)sigma0) >> 25);
    uint sqr_sigma0 = (uint)(((ulong)sigma0 * (ulong)sigma0) >> 32);
    uint tail = (uint)(((ulong)(((r >> 1) + (r >> 3)) - ((uint)r0 << 14))
                        * (ulong)sqr_sigma0) >> 48);
    r += tail;
    return ((r & 0x80000000u) == 0u) ? 0x80000000u : r;
}

// Berkeley-style f64_div. Replaces Newton-on-reciprocal + 14-iter
// correction with multiplication-only recip + rem-based refinement.
// Bit-exact vs the old path (Rust ref validated 40K cases).
ulong softfloat_div(ulong a, ulong b, uint mode) {
    Unpacked ua = unpack(a);
    Unpacked ub = unpack(b);
    ulong sign_z = ua.sign ^ ub.sign;

    // Special-case dispatch only when at least one input isn't normal.
    // Hoists the 7-branch ladder out of the common (hot) path.
    if (ua.class_ != CLASS_NORMAL || ub.class_ != CLASS_NORMAL) {
        if (ua.class_ == CLASS_NAN || ub.class_ == CLASS_NAN) return CANONICAL_QNAN;
        if (ua.class_ == CLASS_INF && ub.class_ == CLASS_INF) return CANONICAL_QNAN;
        if (ua.class_ == CLASS_ZERO && ub.class_ == CLASS_ZERO) return CANONICAL_QNAN;
        if (ua.class_ == CLASS_INF) return (sign_z << 63) | POS_INF;
        if (ub.class_ == CLASS_INF) return sign_z << 63;
        if (ub.class_ == CLASS_ZERO) return (sign_z << 63) | POS_INF; // finite/0 = ±∞
        if (ua.class_ == CLASS_ZERO) return sign_z << 63;
#ifdef SOFTFLOAT_FTZ
        // FTZ: subnormal numerator → ±0; subnormal denominator → ±∞.
        if (ua.class_ == CLASS_SUBNORMAL && ub.class_ == CLASS_SUBNORMAL) {
            return CANONICAL_QNAN;
        }
        if (ua.class_ == CLASS_SUBNORMAL) return sign_z << 63;
        if (ub.class_ == CLASS_SUBNORMAL) return (sign_z << 63) | POS_INF;
#endif
    }

    // Hot-path note: callers that pre-mask their operands so both inputs
    // are non-zero normals (a common pattern in compute-heavy inner
    // loops) hit only the algorithm below — the special-case branches
    // above are dead code under that contract.

    int exp_a_biased = ua.exp + EXP_BIAS;
    int exp_b_biased = ub.exp + EXP_BIAS;
    int exp_z_biased = exp_a_biased - exp_b_biased + 0x3FE;

    ulong sig_a_init = ua.mantissa;
    ulong sig_b = ub.mantissa;
    ulong sig_a;
    if (sig_a_init < sig_b) {
        exp_z_biased -= 1;
        sig_a = sig_a_init << 11;
    } else {
        sig_a = sig_a_init << 10;
    }
    ulong sig_b_11 = sig_b << 11;
    uint sig_b_hi32 = (uint)(sig_b_11 >> 32);

    uint recip32 = approx_recip32_1(sig_b_hi32) - 2u;
    uint sig32_z = (uint)(((ulong)(uint)(sig_a >> 32) * (ulong)recip32) >> 32);
    uint double_term = sig32_z << 1;
    uint sig_b_tail = (uint)sig_b_11 >> 4;
    ulong rem = ((sig_a - (ulong)double_term * (ulong)sig_b_hi32) << 28)
              - (ulong)double_term * (ulong)sig_b_tail;
    uint q = (uint)(((ulong)(uint)(rem >> 32) * (ulong)recip32) >> 32) + 4u;
    ulong sig_z = ((ulong)sig32_z << 32) + ((ulong)q << 4);

    if ((sig_z & 0x1FFUL) < (ulong)(4u << 4)) {
        uint q_corr = q & ~7u;
        sig_z &= ~(ulong)0x7FUL;
        uint double_term2 = q_corr << 1;
        rem = ((rem - (ulong)double_term2 * (ulong)sig_b_hi32) << 28)
            - (ulong)double_term2 * (ulong)sig_b_tail;
        if ((rem & 0x8000000000000000UL) != 0UL) sig_z -= 1UL << 7;
        else if (rem != 0UL) sig_z |= 1UL;
    }

    // Stage Berkeley sig_z (53-bit mantissa with implicit at bit 62, plus
    // 10 round/sticky bits at [9:0]) into the m_128 layout that
    // round_and_pack expects: implicit at bit 116, round/sticky in bits
    // [63:54]. round_and_pack handles single-rounding for both normal
    // and subnormal output (gradual underflow) — bit-exact vs the
    // Berkeley round-pack for normal results, and IEEE-correct for
    // subnormal results that the original round-pack flushed to zero.
    U128 m_128 = U128{sig_z >> 10, sig_z << 54};
    return round_and_pack(m_128, sign_z, exp_z_biased + 1 - EXP_BIAS, mode);
}


// Berkeley-style softfloat sqrt. Multiplication-only refinement off a
// HW f32 reciprocal-sqrt seed; bit-exact vs the Rust reference
// (`softfloat_ref::fsqrt_berkeley`, 40K cross-checked cases).
ulong softfloat_sqrt(ulong a, uint mode) {
    Unpacked ua = unpack(a);

#ifdef SOFTFLOAT_FTZ
    // FTZ: positive subnormal collapses to +0. (Negative subnormal stays
    // a NaN-producing input — handled below by the sign-of-finite branch.)
    if (ua.class_ == CLASS_SUBNORMAL && ua.sign == 0UL) return 0UL;
#endif

    // Special-case dispatch only for non-normal-positive inputs.
    if (ua.class_ != CLASS_NORMAL || ua.sign == 1UL) {
        if (ua.class_ == CLASS_NAN) return CANONICAL_QNAN;
        if (ua.class_ == CLASS_ZERO) return a;            // sqrt(±0) = ±0
        if (ua.class_ == CLASS_INF) {
            return (ua.sign == 1UL) ? CANONICAL_QNAN : POS_INF;
        }
        if (ua.sign == 1UL) return CANONICAL_QNAN;        // sqrt(negative finite) invalid
        // Falls through to algorithm below for subnormal-positive (rare).
    }

    // Hot-path note: callers that pre-mask their operand to a positive
    // normal (a common pattern when sqrt is called on bounded inputs)
    // hit only the algorithm below — the special-case branches above
    // are dead code under that contract.
    int exp_a_biased = ua.exp + EXP_BIAS;
    uint odd_exp_a = (uint)(exp_a_biased & 1);
    // Berkeley expZ biased = ((expA - 0x3FF) >> 1) + 0x3FE (signed >>).
    int exp_z_biased = ((exp_a_biased - 0x3FF) >> 1) + 0x3FE;

    ulong sig_a = ua.mantissa;
    uint sig32_a = (uint)(sig_a >> 21);
    uint recip_sqrt32 = approx_recip_sqrt32_1(odd_exp_a, sig32_a);
    uint sig32_z = (uint)(((ulong)sig32_a * (ulong)recip_sqrt32) >> 32);

    if (odd_exp_a != 0u) { sig_a <<= 8; sig32_z >>= 1; }
    else                 { sig_a <<= 9; }

    ulong rem = sig_a - (ulong)sig32_z * (ulong)sig32_z;
    uint q = (uint)((((ulong)(uint)(rem >> 2)) * (ulong)recip_sqrt32) >> 32);
    ulong sig_z = ((ulong)sig32_z << 32) | (1UL << 5);
    sig_z += ((ulong)q << 3);

    // Berkeley low-bit correction (exact-rounding edge cases).
    if ((sig_z & 0x1FFUL) < 0x22UL) {
        sig_z &= ~(ulong)0x3FUL;
        ulong shifted = sig_z >> 6;
        ulong rem2 = (sig_a << 52) - shifted * shifted;
        if ((rem2 & 0x8000000000000000UL) != 0UL) sig_z -= 1UL;
        else if (rem2 != 0UL) sig_z |= 1UL;
    }

    // Inline Berkeley round-pack — IEEE-754 conformant by mathematical
    // invariant: sqrt of any non-negative f64 (range [2^-1074, ~2^1024))
    // lands in [2^-537, 2^512), comfortably inside the normal range
    // [2^-1022, 2^1024). Subnormal output is unreachable, so the legacy
    // `pack` (which FTZs biased<1) cannot be hit here. Skipping
    // round_and_pack's U128 path saves ~4% throughput on this kernel.
    //
    // For positive sqrt (sign = 0):
    //   Nearest  → incr 0x200, with tie-to-even mask
    //   Up       → incr 0x3FF
    //   Down     → incr 0
    //   Zero     → incr 0
    ulong round_bits = sig_z & 0x3FFUL;
    ulong increment;
    if (mode == RMODE_NEAREST)   increment = 0x200UL;
    else if (mode == RMODE_UP)   increment = 0x3FFUL;
    else                         increment = 0UL;
    ulong rounded_sig = (sig_z + increment) >> 10;
    if (mode == RMODE_NEAREST && round_bits == 0x200UL) rounded_sig &= ~1UL;

    // Berkeley's packToF64UI uses addition so the implicit bit carries
    // into the exp field — our pack() OR-masks the implicit bit, so we
    // add 1 (or 2 on double-carry) to the biased exp.
    int final_exp_biased;
    ulong final_mantissa;
    if ((rounded_sig >> 53) != 0UL) {
        final_exp_biased = exp_z_biased + 2;
        final_mantissa = rounded_sig >> 1;
    } else {
        final_exp_biased = exp_z_biased + 1;
        final_mantissa = rounded_sig;
    }
    return pack(0UL, final_exp_biased - EXP_BIAS, final_mantissa, mode);
}

// IEEE-754 fused multiply-add `(a × b) + c` with one rounding. Fully
// IEEE-754 compliant: handles NaN / ±Inf / ±0 / subnormal inputs and
// produces gradual-underflow output (subnormal output FTZ matches the
// `ftz` feature only). Mirrors softfloat_ref::fma in the Rust crate;
// cross-checked vs native HW FMA (f64::mul_add) over the full u64
// domain in the Rust ref tests, then the MSL port is bit-exact vs the
// Rust algorithm.
ulong softfloat_fma(ulong a, ulong b, ulong c, uint mode) {
    Unpacked ua = unpack(a);
    Unpacked ub = unpack(b);
    Unpacked uc = unpack(c);
    ulong prod_sign = ua.sign ^ ub.sign;

    // --- Special-case dispatch (IEEE-754 §6/§7) -----------------
    if (ua.class_ == CLASS_NAN || ub.class_ == CLASS_NAN || uc.class_ == CLASS_NAN) {
        return CANONICAL_QNAN;
    }
    if ((ua.class_ == CLASS_ZERO && ub.class_ == CLASS_INF) ||
        (ua.class_ == CLASS_INF  && ub.class_ == CLASS_ZERO)) {
        return CANONICAL_QNAN;
    }
    if (ua.class_ == CLASS_INF || ub.class_ == CLASS_INF) {
        if (uc.class_ == CLASS_INF && uc.sign != prod_sign) return CANONICAL_QNAN;
        return (prod_sign << 63) | POS_INF;
    }
    if (uc.class_ == CLASS_INF) return (uc.sign << 63) | POS_INF;
    if (ua.class_ == CLASS_ZERO || ub.class_ == CLASS_ZERO) {
        if (uc.class_ == CLASS_ZERO) {
            ulong sign;
            if (prod_sign == uc.sign) sign = prod_sign;
            else                      sign = (mode == RMODE_DOWN) ? 1UL : 0UL;
            return sign << 63;
        }
        return c;
    }
    if (uc.class_ == CLASS_ZERO) return softfloat_mul(a, b, mode);

    // --- Compute the product in full precision (106-bit) -------
    U128 prod_sig = u128_mul_u64(ua.mantissa, ub.mantissa);
    bool prod_msb_high = (prod_sig.hi >> (105 - 64)) != 0UL;
    // Stage product so MSB is at bit 116.
    U128 m_prod = u128_shl(prod_sig, prod_msb_high ? 11u : 12u);
    int exp_prod = ua.exp + ub.exp + (prod_msb_high ? 1 : 0);

    // Stage c with MSB at bit 116 too.
    U128 c_staged = U128{uc.mantissa, 0UL};

    // --- Align and combine ---
    int exp_diff = exp_prod - uc.exp;
    U128 m_combined;
    int exp_combined;
    ulong sign_combined;

    if (prod_sign == uc.sign) {
        // Same-sign add. Align smaller to bigger.
        U128 big; U128 sml; int big_exp; ulong sign_out; uint shift_u;
        if (exp_diff >= 0) {
            big = m_prod; sml = c_staged; big_exp = exp_prod;
            sign_out = prod_sign; shift_u = (uint)exp_diff;
        } else {
            big = c_staged; sml = m_prod; big_exp = uc.exp;
            sign_out = uc.sign; shift_u = (uint)(-exp_diff);
        }
        U128 sml_aligned;
        if (shift_u >= 128u) {
            sml_aligned = U128{0UL, u128_is_zero(sml) ? 0UL : 1UL};
        } else if (shift_u == 0u) {
            sml_aligned = sml;
        } else {
            // sticky-jam shift
            bool sticky;
            if (shift_u < 64u) {
                ulong mask = (1UL << shift_u) - 1UL;
                sticky = (sml.lo & mask) != 0UL;
            } else {
                uint rem = shift_u - 64u;
                ulong hi_mask = (rem == 0u) ? 0UL : ((1UL << rem) - 1UL);
                sticky = (sml.lo != 0UL) || ((sml.hi & hi_mask) != 0UL);
            }
            sml_aligned = u128_or_sticky(u128_shr(sml, shift_u), sticky);
        }
        U128 sum = u128_add(big, sml_aligned);
        // If bit 117 set, mantissa overflowed.
        int exp_adj = 0;
        if ((sum.hi >> (117 - 64)) != 0UL) {
            bool sticky = (sum.lo & 1UL) != 0UL;
            sum = u128_shr(sum, 1u);
            sum = u128_or_sticky(sum, sticky);
            exp_adj = 1;
        }
        m_combined = sum;
        exp_combined = big_exp + exp_adj;
        sign_combined = sign_out;
    } else {
        // Opposite signs — cancellation possible.
        U128 big; U128 sml; int big_exp; ulong sign_out; uint shift_u;
        if (exp_diff > 0) {
            big = m_prod; sml = c_staged; big_exp = exp_prod;
            sign_out = prod_sign; shift_u = (uint)exp_diff;
        } else if (exp_diff < 0) {
            big = c_staged; sml = m_prod; big_exp = uc.exp;
            sign_out = uc.sign; shift_u = (uint)(-exp_diff);
        } else if (u128_cmp(m_prod, c_staged) >= 0) {
            big = m_prod; sml = c_staged; big_exp = exp_prod;
            sign_out = prod_sign; shift_u = 0u;
        } else {
            big = c_staged; sml = m_prod; big_exp = uc.exp;
            sign_out = uc.sign; shift_u = 0u;
        }
        U128 sml_aligned;
        if (shift_u >= 128u) {
            sml_aligned = U128{0UL, u128_is_zero(sml) ? 0UL : 1UL};
        } else if (shift_u == 0u) {
            sml_aligned = sml;
        } else {
            // Branch-free sticky: compute both <64 and >=64 cases, select.
            ulong mask_lo = (shift_u < 64u) ? ((1UL << shift_u) - 1UL) : ~0UL;
            bool sticky_lt64 = (sml.lo & mask_lo) != 0UL;
            uint rem = (shift_u >= 64u) ? (shift_u - 64u) : 0u;
            ulong hi_mask = (rem == 0u) ? 0UL : ((1UL << rem) - 1UL);
            bool sticky_ge64 = (sml.lo != 0UL) || ((sml.hi & hi_mask) != 0UL);
            bool sticky = (shift_u < 64u) ? sticky_lt64 : sticky_ge64;
            sml_aligned = u128_or_sticky(u128_shr(sml, shift_u), sticky);
        }
        U128 diff = u128_sub(big, sml_aligned);
        if (u128_is_zero(diff)) {
            return (mode == RMODE_DOWN) ? (1UL << 63) : 0UL;
        }
        // Renormalize.
        uint lz = u128_clz(diff);
        int msb_pos = 127 - (int)lz;
        int shift_left = 116 - msb_pos;
        U128 normalized;
        if (shift_left > 0) {
            normalized = u128_shl(diff, (uint)shift_left);
        } else if (shift_left < 0) {
            uint n = (uint)(-shift_left);
            ulong mask_lo = (n < 64u) ? ((1UL << n) - 1UL) : ~0UL;
            bool sticky_lt64 = (diff.lo & mask_lo) != 0UL;
            uint rem = (n >= 64u) ? (n - 64u) : 0u;
            ulong hi_mask = (rem == 0u) ? 0UL : ((1UL << rem) - 1UL);
            bool sticky_ge64 = (diff.lo != 0UL) || ((diff.hi & hi_mask) != 0UL);
            bool sticky = (n < 64u) ? sticky_lt64 : sticky_ge64;
            normalized = u128_or_sticky(u128_shr(diff, n), sticky);
        } else {
            normalized = diff;
        }
        m_combined = normalized;
        exp_combined = big_exp - shift_left;
        sign_combined = sign_out;
    }

    return round_and_pack(m_combined, sign_combined, exp_combined, mode);
}

// --- Conversions -----------------------------------------------------------
//
// Bit-exact ports of `softfloat_ref::cvt_*`. All inputs / outputs are
// integer types (u32 / u64 / i64) so callers can stay in pure-integer
// land — no native f64 anywhere on the GPU side.

// Internal: non-zero magnitude `mag` with explicit sign bit → f64 bits.
// Caller guarantees mag != 0.
static __attribute__((always_inline)) ulong cvt_mag_to_f64(ulong sign, ulong mag, uint mode) {
    int msb_pos = 63 - (int)clz(mag);
    // Place MSB at bit 116 of the 128-bit staging (mantissa bits [116:64],
    // GRS at [63:0]). Then unbiased exp == msb_pos.
    uint shift = (uint)(116 - msb_pos);
    U128 m_128 = u128_shl(u128_from_u64(mag), shift);
    return round_and_pack(m_128, sign, msb_pos, mode);
}

// i64 → f64. Mode-controlled rounding.
static ulong softfloat_cvt_i64_to_f64(long x, uint mode) {
    if (x == 0L) return 0UL;
    ulong sign = (x < 0L) ? 1UL : 0UL;
    // unsigned-abs: for LONG_MIN, (ulong)x == 0x8000000000000000 and
    // 0 - that == 0x8000000000000000 (two's complement wraparound), which
    // is exactly the magnitude 2^63. For other negatives, 0 - (ulong)x is
    // the standard unsigned negation. Mirror of `(x as i128).unsigned_abs()`.
    ulong mag = (x < 0L) ? (0UL - (ulong)x) : (ulong)x;
    return cvt_mag_to_f64(sign, mag, mode);
}

// u64 → f64. Mode-controlled rounding.
static ulong softfloat_cvt_u64_to_f64(ulong x, uint mode) {
    if (x == 0UL) return 0UL;
    return cvt_mag_to_f64(0UL, x, mode);
}

// f64 → i64. NaN → 0; out-of-range overflows saturate to LONG_MIN/LONG_MAX.
// Mode-controlled rounding for the fractional bits.
static long softfloat_cvt_f64_to_i64(ulong a, uint mode) {
    Unpacked ua = unpack(a);
    if (ua.class_ == CLASS_NAN) return 0L;
    if (ua.class_ == CLASS_ZERO) return 0L;
    if (ua.class_ == CLASS_INF) {
        return (ua.sign == 1UL) ? (long)0x8000000000000000UL : (long)0x7FFFFFFFFFFFFFFFL;
    }
    // Subnormal: `unpack` already normalized it into a very-small Normal
    // (ua.exp ≪ 0), so it falls through to the |x| < 1 handler below.

    // Normal / normalized-Subnormal: real value = mantissa × 2^(exp - 52),
    // implicit bit at 52.
    if (ua.exp < 0) {
        // |x| < 1 → trunc to 0; mode rounding may bump magnitude to 1.
        // Nearest: |x| > 0.5 rounds away from zero (tie at 0.5 rounds to
        // the even integer 0). exp == −1 means |x| ∈ [0.5, 1.0); any
        // mantissa bit below the implicit 52 puts |x| strictly above 0.5.
        bool round_up = false;
        if (mode == RMODE_UP)      round_up = (ua.sign == 0UL);
        else if (mode == RMODE_DOWN) round_up = (ua.sign == 1UL);
        else if (mode == RMODE_NEAREST)
            round_up = (ua.exp == -1) && ((ua.mantissa & MANT_MASK) != 0UL);
        if (round_up) return (ua.sign == 1UL) ? -1L : 1L;
        return 0L;
    }
    if (ua.exp > 62) {
        // |x| >= 2^63. Only -2^63 (== LONG_MIN) is representable.
        if (ua.exp == 63 && ua.sign == 1UL && ua.mantissa == IMPLICIT_BIT) {
            return (long)0x8000000000000000UL;
        }
        return (ua.sign == 1UL) ? (long)0x8000000000000000UL : (long)0x7FFFFFFFFFFFFFFFL;
    }

    // 0 <= ua.exp <= 62. Two regimes:
    //   exp ∈ [0, 52]: shift mantissa right by (52 - exp) for the integer
    //     part; the dropped bits feed rounding.
    //   exp ∈ [53, 62]: the value is an exact integer larger than the
    //     mantissa; shift left by (exp - 52), no fractional bits.
    ulong int_part;
    ulong dropped;
    // MSL reserves `half` for the 16-bit float type — use `tie` for the
    // halfway-bit value used in nearest-mode rounding.
    ulong tie;
    if (ua.exp <= 52) {
        uint shift = (uint)(52 - ua.exp);
        int_part = ua.mantissa >> shift;
        dropped = (shift == 0u) ? 0UL : (ua.mantissa & ((1UL << shift) - 1UL));
        tie = (shift == 0u) ? 0UL : (1UL << (shift - 1u));
    } else {
        uint shift = (uint)(ua.exp - 52);
        int_part = ua.mantissa << shift;
        dropped = 0UL;
        tie = 0UL;
    }

    bool round_up;
    if (mode == RMODE_NEAREST) {
        round_up = (dropped > tie) || ((dropped == tie) && ((int_part & 1UL) == 1UL));
    } else if (mode == RMODE_DOWN) {
        round_up = (ua.sign == 1UL) && (dropped != 0UL);
    } else if (mode == RMODE_UP) {
        round_up = (ua.sign == 0UL) && (dropped != 0UL);
    } else { // RMODE_ZERO
        round_up = false;
    }
    ulong mag = int_part + (round_up ? 1UL : 0UL);

    if (ua.sign == 1UL) {
        // Negative — saturate at LONG_MIN; otherwise -mag.
        if (mag >= (1UL << 63)) return (long)0x8000000000000000UL;
        return -(long)mag;
    } else {
        if (mag >= (1UL << 63)) return (long)0x7FFFFFFFFFFFFFFFL;
        return (long)mag;
    }
}

// f32 → f64: exact, every f32 representable as f64. No rounding.
static ulong softfloat_cvt_f32_to_f64(uint a) {
    ulong sign = (ulong)((a >> 31u) & 1u);
    uint exp_raw = (a >> 23u) & 0xFFu;
    ulong mant_raw = (ulong)(a & 0x7FFFFFu);

    if (exp_raw == 0u && mant_raw == 0UL) return sign << 63;
    if (exp_raw == 0xFFu) {
        if (mant_raw == 0UL) return (sign << 63) | POS_INF;
        return CANONICAL_QNAN;
    }
    if (exp_raw == 0u) {
        // Subnormal f32 → normalize and re-encode as normal f64.
        // mant_raw fits in 23 bits, so 64-bit clz ∈ [41, 63] for nonzero.
        uint lz = (uint)clz(mant_raw) - 41u;
        ulong mant_norm = (mant_raw << (lz + 1u)) & 0x7FFFFFUL;
        int exp_unbiased = -127 - (int)lz;
        int biased64 = exp_unbiased + EXP_BIAS;
        return (sign << 63) | (((ulong)biased64) << MANT_BITS) | (mant_norm << (MANT_BITS - 23u));
    }
    int exp_unbiased = (int)exp_raw - 127;
    int biased64 = exp_unbiased + EXP_BIAS;
    return (sign << 63) | (((ulong)biased64) << MANT_BITS) | (mant_raw << (MANT_BITS - 23u));
}

// f64 → f32 with mode-controlled rounding.
static uint softfloat_cvt_f64_to_f32(ulong a, uint mode) {
    Unpacked ua = unpack(a);
    uint sign32 = (uint)ua.sign << 31;
    if (ua.class_ == CLASS_NAN)  return 0x7FC00000u;
    if (ua.class_ == CLASS_ZERO) return sign32;
    if (ua.class_ == CLASS_INF)  return sign32 | 0x7F800000u;

    // Normal or Subnormal (subnormal already normalized by `unpack`).
    int exp_unbiased = ua.exp;
    if (exp_unbiased > 127) {
        uint f32_inf = sign32 | 0x7F800000u;
        uint f32_max = sign32 | 0x7F7FFFFFu;
        if (mode == RMODE_NEAREST) return f32_inf;
        if (mode == RMODE_UP)      return (ua.sign == 1UL) ? f32_max : f32_inf;
        if (mode == RMODE_DOWN)    return (ua.sign == 1UL) ? f32_inf : f32_max;
        return f32_max; // RMODE_ZERO
    }
    if (exp_unbiased < -149) {
        bool any_dropped = ua.mantissa != 0UL;
        bool round_up = false;
        if (mode == RMODE_UP)   round_up = (ua.sign == 0UL) && any_dropped;
        if (mode == RMODE_DOWN) round_up = (ua.sign == 1UL) && any_dropped;
        return sign32 | (round_up ? 1u : 0u);
    }

    // Drop 29 bits (53-bit f64 mantissa → 24-bit f32) plus extras for f32
    // subnormal output (when exp_unbiased < -126).
    uint total_drop;
    int new_exp_biased;
    if (exp_unbiased >= -126) {
        total_drop = 29u;
        new_exp_biased = exp_unbiased + 127;
    } else {
        total_drop = 29u + (uint)(-126 - exp_unbiased);
        new_exp_biased = 0;
    }

    ulong new_mant;
    ulong guard, round_bit, sticky;
    if (total_drop == 0u) {
        new_mant = ua.mantissa; guard = 0UL; round_bit = 0UL; sticky = 0UL;
    } else if (total_drop == 1u) {
        new_mant = ua.mantissa >> 1;
        guard = ua.mantissa & 1UL;
        round_bit = 0UL; sticky = 0UL;
    } else if (total_drop == 2u) {
        new_mant = ua.mantissa >> 2;
        guard = (ua.mantissa >> 1) & 1UL;
        round_bit = ua.mantissa & 1UL;
        sticky = 0UL;
    } else if (total_drop < 64u) {
        new_mant = ua.mantissa >> total_drop;
        guard = (ua.mantissa >> (total_drop - 1u)) & 1UL;
        round_bit = (ua.mantissa >> (total_drop - 2u)) & 1UL;
        ulong mask = (1UL << (total_drop - 2u)) - 1UL;
        sticky = ((ua.mantissa & mask) != 0UL) ? 1UL : 0UL;
    } else {
        new_mant = 0UL; guard = 0UL; round_bit = 0UL;
        sticky = (ua.mantissa != 0UL) ? 1UL : 0UL;
    }

    ulong grs_any = guard | round_bit | sticky;
    bool round_up;
    if (mode == RMODE_NEAREST) {
        round_up = (guard == 1UL) && ((round_bit | sticky | (new_mant & 1UL)) != 0UL);
    } else if (mode == RMODE_DOWN) {
        round_up = (ua.sign == 1UL) && (grs_any != 0UL);
    } else if (mode == RMODE_UP) {
        round_up = (ua.sign == 0UL) && (grs_any != 0UL);
    } else {
        round_up = false;
    }
    ulong mant_after = new_mant + (round_up ? 1UL : 0UL);

    ulong f32_implicit = 1UL << 23;
    if (exp_unbiased >= -126) {
        // Normal-output range.
        if ((mant_after >> 24) != 0UL) {
            // Round-up bumped the mantissa across the implicit-bit boundary.
            int bumped_exp = new_exp_biased + 1;
            if (bumped_exp > 254) {
                uint f32_inf = sign32 | 0x7F800000u;
                uint f32_max = sign32 | 0x7F7FFFFFu;
                if (mode == RMODE_NEAREST) return f32_inf;
                if (mode == RMODE_UP)      return (ua.sign == 1UL) ? f32_max : f32_inf;
                if (mode == RMODE_DOWN)    return (ua.sign == 1UL) ? f32_inf : f32_max;
                return f32_max;
            }
            return sign32 | ((uint)bumped_exp << 23) | ((uint)(mant_after >> 1) & 0x7FFFFFu);
        }
        return sign32 | ((uint)new_exp_biased << 23) | ((uint)mant_after & 0x7FFFFFu);
    } else {
        // Subnormal output. Rounding may have promoted to the smallest normal.
        if (mant_after >= f32_implicit) {
            return sign32 | (1u << 23);
        }
        return sign32 | ((uint)mant_after & 0x7FFFFFu);
    }
}

// --- Comparisons -----------------------------------------------------------
//
// IEEE-754 §5.11 quiet comparisons: any NaN operand makes the result false.
// Bit-exact ports of `softfloat_ref::feq/flt/fle/fgt/fge`.

static __attribute__((always_inline)) bool nan_either(ulong a, ulong b) {
    return classify(a) == CLASS_NAN || classify(b) == CLASS_NAN;
}

static bool softfloat_feq(ulong a, ulong b) {
    if (nan_either(a, b)) return false;
    bool za = classify(a) == CLASS_ZERO;
    bool zb = classify(b) == CLASS_ZERO;
    if (za && zb) return true;  // ±0 compare equal regardless of sign bit
    return a == b;
}

static bool softfloat_flt(ulong a, ulong b) {
    if (nan_either(a, b)) return false;
    ulong sa = a >> 63;
    ulong sb = b >> 63;
    if (sa != sb) {
        // Different signs: a < b iff a is negative AND not both ±0.
        if (classify(a) == CLASS_ZERO && classify(b) == CLASS_ZERO) return false;
        return sa == 1UL;
    }
    // Same sign: bit-pattern compare; for negatives the relation is reversed.
    return (sa == 1UL) ? (a > b) : (a < b);
}

static bool softfloat_fle(ulong a, ulong b) { return softfloat_flt(a, b) || softfloat_feq(a, b); }
static bool softfloat_fgt(ulong a, ulong b) { return softfloat_flt(b, a); }
static bool softfloat_fge(ulong a, ulong b) { return softfloat_fle(b, a); }

} // namespace softfloat64_internal


// --- Public API ----------------------------------------------------------
//
// Two families:
//
//  * __softfloat64_*           — full IEEE-754 path. Inputs may be NaN /
//                                Inf / ±0 / subnormal / normal; outputs
//                                follow IEEE §6/§7. Use when you don't
//                                control the input distribution.
//
//  * __softfloat64_unp_*       — "unpacked" path for tight inner loops.
//                                Caller pre-unpacks once with
//                                __softfloat64_unpack(), runs many ops
//                                on the unpacked state, then re-packs
//                                with __softfloat64_pack(). Skips the
//                                pack/unpack churn on every iter and
//                                lets the Metal compiler inline the op
//                                bodies. CALLER MUST GUARANTEE NORMAL
//                                INPUTS — non-normal operands silently
//                                produce wrong answers (no NaN/Inf/±0
//                                special-case dispatch on this path).
//                                Useful for Kahan/Welford reductions,
//                                dot products, layer-norm denominators,
//                                etc., where you've already validated
//                                the data.

// Public unpacked-state struct. Layout intentionally matches the
// internal Unpacked6: a sign bit, an unbiased exponent, and a 53-bit
// mantissa with the implicit bit at position 52. Treat as opaque.
struct __softfloat64_unp {
    ulong sign;
    int   exp;
    ulong mantissa;
};

ulong __softfloat64_fadd(ulong a, ulong b, uint mode) { return softfloat64_internal::softfloat_add(a, b, mode); }
ulong __softfloat64_fsub(ulong a, ulong b, uint mode) { return softfloat64_internal::softfloat_sub(a, b, mode); }
ulong __softfloat64_fmul(ulong a, ulong b, uint mode) { return softfloat64_internal::softfloat_mul(a, b, mode); }
ulong __softfloat64_fdiv(ulong a, ulong b, uint mode) { return softfloat64_internal::softfloat_div(a, b, mode); }
ulong __softfloat64_fsqrt(ulong a, uint mode) { return softfloat64_internal::softfloat_sqrt(a, mode); }
ulong __softfloat64_fma(ulong a, ulong b, ulong c, uint mode) { return softfloat64_internal::softfloat_fma(a, b, c, mode); }

// Adapter: unpack a u64 bit pattern into the public unpacked state.
// Subnormals are normalized in the same way as the internal `unpack`,
// so they're safe inputs. Zero / Inf / NaN inputs are *not* — the
// unpacked path has no special-case dispatch.
__softfloat64_unp __softfloat64_unpack(ulong bits) {
    softfloat64_internal::Unpacked u = softfloat64_internal::unpack(bits);
    return __softfloat64_unp{u.sign, u.exp, u.mantissa};
}

ulong __softfloat64_pack(__softfloat64_unp u, uint mode) {
    return softfloat64_internal::pack(u.sign, u.exp, u.mantissa, mode);
}

// Adapter: synthesize an internal Unpacked from the public unp state,
// hardcoding CLASS_NORMAL (the precondition of the _unp_normal helpers).
static inline softfloat64_internal::Unpacked __softfloat64_unp_to_internal(__softfloat64_unp u) {
    softfloat64_internal::Unpacked out;
    out.sign = u.sign;
    out.exp = u.exp;
    out.mantissa = u.mantissa;
    out.class_ = softfloat64_internal::CLASS_NORMAL;
    return out;
}

static inline __softfloat64_unp __softfloat64_unp_from_internal(softfloat64_internal::Unpacked6 u) {
    return __softfloat64_unp{u.sign, u.exp, u.mantissa};
}

__softfloat64_unp __softfloat64_unp_fadd(__softfloat64_unp a, __softfloat64_unp b, uint mode) {
    return __softfloat64_unp_from_internal(softfloat64_internal::softfloat_add_unp_normal(
        __softfloat64_unp_to_internal(a), __softfloat64_unp_to_internal(b), mode));
}
__softfloat64_unp __softfloat64_unp_fsub(__softfloat64_unp a, __softfloat64_unp b, uint mode) {
    return __softfloat64_unp_from_internal(softfloat64_internal::softfloat_sub_unp_normal(
        __softfloat64_unp_to_internal(a), __softfloat64_unp_to_internal(b), mode));
}
__softfloat64_unp __softfloat64_unp_fmul(__softfloat64_unp a, __softfloat64_unp b, uint mode) {
    return __softfloat64_unp_from_internal(softfloat64_internal::softfloat_mul_unp_normal(
        __softfloat64_unp_to_internal(a), __softfloat64_unp_to_internal(b), mode));
}
__softfloat64_unp __softfloat64_unp_fdiv(__softfloat64_unp a, __softfloat64_unp b, uint mode) {
    return __softfloat64_unp_from_internal(softfloat64_internal::softfloat_div_unp_normal(
        __softfloat64_unp_to_internal(a), __softfloat64_unp_to_internal(b), mode));
}
__softfloat64_unp __softfloat64_unp_fsqrt(__softfloat64_unp a, uint mode) {
    return __softfloat64_unp_from_internal(softfloat64_internal::softfloat_sqrt_unp_normal(
        __softfloat64_unp_to_internal(a), mode));
}
__softfloat64_unp __softfloat64_unp_fma(__softfloat64_unp a, __softfloat64_unp b, __softfloat64_unp c, uint mode) {
    return __softfloat64_unp_from_internal(softfloat64_internal::softfloat_fma_unp_normal(
        __softfloat64_unp_to_internal(a),
        __softfloat64_unp_to_internal(b),
        __softfloat64_unp_to_internal(c),
        mode));
}

// --- Conversions ---------------------------------------------------------
//
// All public-API conversions take/return integer types — no native float
// or double crosses the boundary. f32/f64 inputs and outputs are u32/u64
// bit patterns (use `as_type<>` on the caller side to bit-cast).

ulong __softfloat64_cvt_i64_to_f64(long x, uint mode)  { return softfloat64_internal::softfloat_cvt_i64_to_f64(x, mode); }
ulong __softfloat64_cvt_u64_to_f64(ulong x, uint mode) { return softfloat64_internal::softfloat_cvt_u64_to_f64(x, mode); }
long  __softfloat64_cvt_f64_to_i64(ulong a, uint mode) { return softfloat64_internal::softfloat_cvt_f64_to_i64(a, mode); }
ulong __softfloat64_cvt_f32_to_f64(uint a)             { return softfloat64_internal::softfloat_cvt_f32_to_f64(a); }
uint  __softfloat64_cvt_f64_to_f32(ulong a, uint mode) { return softfloat64_internal::softfloat_cvt_f64_to_f32(a, mode); }

// --- Comparisons ---------------------------------------------------------
//
// IEEE-754 §5.11 quiet comparisons: any NaN input → false (unordered).

bool __softfloat64_feq(ulong a, ulong b) { return softfloat64_internal::softfloat_feq(a, b); }
bool __softfloat64_flt(ulong a, ulong b) { return softfloat64_internal::softfloat_flt(a, b); }
bool __softfloat64_fle(ulong a, ulong b) { return softfloat64_internal::softfloat_fle(a, b); }
bool __softfloat64_fgt(ulong a, ulong b) { return softfloat64_internal::softfloat_fgt(a, b); }
bool __softfloat64_fge(ulong a, ulong b) { return softfloat64_internal::softfloat_fge(a, b); }

// --- Throughput kernels --------------------------------------------------
//
// Each thread runs `__SOFTFLOAT64_CHAIN_OPS` chained softfloat ops.
// `a` and `b` start near 1.0 (low-mantissa bits perturbed by gid so
// distinct threads run distinct trajectories — uniform seeds let the
// compiler broadcast-fold the kernel) and a cheap mantissa-twiddle
// keeps successive ops in the normal fast path. Total ops dispatched
// = `threads × __SOFTFLOAT64_CHAIN_OPS`.
//
// These exist because the per-element host↔device buffer-transfer cost
// of a `_batch`-style API hides the GPU's actual op throughput. By
// keeping ~1k softfloat ops per thread before any host I/O, the chain
// kernels measure raw aggregate softfloat throughput — which beats a
// 14-thread CPU hardware-f64 baseline by 5-11× on Apple Silicon for
// fadd/fsub/fmul/fdiv/fsqrt, and trades roughly 1:1 with hardware FMA.
//
// The result `out[gid]` is the final accumulator value of thread `gid`.
// Callers typically dispatch with the largest thread count their target
// hardware can saturate (~200k–400k on M-series).
constant uint __SOFTFLOAT64_CHAIN_OPS = 1024u;

// Twiddle low mantissa bits of `b` based on `a` so the next iter's `b`
// is data-dependent on this iter's `a`. Stays in the normal range:
// only the low 8 bits of mantissa are touched, so the value stays near
// its initial 1.0 magnitude and avoids NaN / Inf / subnormal paths.
static __attribute__((always_inline)) ulong __softfloat64_twiddle(ulong b, ulong a) {
    return b ^ (a & 0xFFUL);
}

kernel void __softfloat64_fadd_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    ulong b = 0x3FF0000000000001UL ^ ((seed.y ^ (ulong)gid) & 0xFFUL);
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        a = __softfloat64_fadd(a, b, 0u);
        b = __softfloat64_twiddle(b, a);
    }
    out[gid] = a ^ b;
}

kernel void __softfloat64_fsub_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    ulong b = 0x3FF0000000000001UL ^ ((seed.y ^ (ulong)gid) & 0xFFUL);
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        a = __softfloat64_fsub(a, b, 0u);
        b = __softfloat64_twiddle(b, a);
    }
    out[gid] = a ^ b;
}

kernel void __softfloat64_fmul_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    ulong b = 0x3FF0000000000001UL ^ ((seed.y ^ (ulong)gid) & 0xFFUL);
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        a = __softfloat64_fmul(a, b, 0u);
        b = __softfloat64_twiddle(b, a);
    }
    out[gid] = a ^ b;
}

kernel void __softfloat64_fdiv_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    ulong b = 0x3FF0000000000001UL ^ ((seed.y ^ (ulong)gid) & 0xFFUL);
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        a = __softfloat64_fdiv(a, b, 0u);
        b = __softfloat64_twiddle(b, a);
    }
    out[gid] = a ^ b;
}

kernel void __softfloat64_fsqrt_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    ulong b = 0x3FF0000000000001UL ^ ((seed.y ^ (ulong)gid) & 0xFFUL);
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        a = __softfloat64_fsqrt(a, 0u);
        a = __softfloat64_twiddle(a, b);
    }
    out[gid] = a ^ b;
}

kernel void __softfloat64_fma_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    ulong b = 0x3FF0000000000001UL ^ ((seed.y ^ (ulong)gid) & 0xFFUL);
    ulong c = 0x3FF0000000000000UL;
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        a = __softfloat64_fma(a, b, c, 0u);
        b = __softfloat64_twiddle(b, a);
    }
    out[gid] = a ^ b;
}

// --- Conversion chain kernels ---------------------------------------------
//
// Same throughput-measurement pattern as the arithmetic chains: each
// thread runs `__SOFTFLOAT64_CHAIN_OPS` ops with a per-iter feedback so
// the Metal compiler can't broadcast-fold the loop. The conversion ops
// have mismatched input / output widths, so the kernels XOR-fold the
// result back into a u64 accumulator before writing.

kernel void __softfloat64_cvt_i64_to_f64_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    // Start at a few thousand with per-lane jitter — small enough that
    // 1024 perturbations stay well clear of LONG_MAX, large enough that
    // the cvt actually drops some bits during rounding.
    long x = (long)(((seed.x ^ (ulong)gid) & 0xFFFFUL) + 1000UL);
    ulong acc = 0UL;
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        ulong r = __softfloat64_cvt_i64_to_f64(x, 0u);
        acc ^= r;
        x = x ^ (long)(r & 0xFFUL);
    }
    out[gid] = acc ^ (ulong)x;
}

kernel void __softfloat64_cvt_u64_to_f64_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong x = ((seed.x ^ (ulong)gid) & 0xFFFFUL) + 1000UL;
    ulong acc = 0UL;
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        ulong r = __softfloat64_cvt_u64_to_f64(x, 0u);
        acc ^= r;
        x = x ^ (r & 0xFFUL);
    }
    out[gid] = acc ^ x;
}

kernel void __softfloat64_cvt_f64_to_i64_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    // Anchor `a` near 1.0 so 1024 iterations of low-bit twiddling can't
    // walk the value into the saturation regime — we want the Normal
    // branch of cvt_f64_to_i64 to dominate the cost.
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    long acc = 0L;
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        long r = __softfloat64_cvt_f64_to_i64(a, 0u);
        acc ^= r;
        a = a ^ ((ulong)r & 0xFFUL);
    }
    out[gid] = a ^ (ulong)acc;
}

kernel void __softfloat64_cvt_f32_to_f64_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    // 0x3F80_0000 = 1.0f. Same near-1 anchoring as the f64 chains.
    uint x = 0x3F800000u ^ (uint)((seed.x ^ (ulong)gid) & 0xFFUL);
    ulong acc = 0UL;
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        ulong r = __softfloat64_cvt_f32_to_f64(x);
        acc ^= r;
        x = x ^ (uint)(r & 0xFFUL);
    }
    out[gid] = acc ^ (ulong)x;
}

kernel void __softfloat64_cvt_f64_to_f32_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    uint acc = 0u;
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        uint r = __softfloat64_cvt_f64_to_f32(a, 0u);
        acc ^= r;
        a = a ^ (ulong)(r & 0xFFu);
    }
    out[gid] = a ^ (ulong)acc;
}

// --- Comparison chain kernels --------------------------------------------
//
// Comparisons return one bit, so a naive `acc ^= r` chain quickly
// saturates. Instead, shift the accumulator and OR the new bit in — this
// keeps the whole 64-bit register data-dependent on every result, and
// `b = twiddle(b, a ^ acc)` carries the full accumulator into the next
// iteration's operand so the compiler can't lift the comparison out.

kernel void __softfloat64_feq_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    ulong b = 0x3FF0000000000001UL ^ ((seed.y ^ (ulong)gid) & 0xFFUL);
    ulong acc = 0UL;
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        ulong r = (ulong)__softfloat64_feq(a, b);
        acc = (acc << 1) ^ r;
        b = __softfloat64_twiddle(b, a ^ acc);
    }
    out[gid] = acc ^ a ^ b;
}

kernel void __softfloat64_flt_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    ulong b = 0x3FF0000000000001UL ^ ((seed.y ^ (ulong)gid) & 0xFFUL);
    ulong acc = 0UL;
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        ulong r = (ulong)__softfloat64_flt(a, b);
        acc = (acc << 1) ^ r;
        b = __softfloat64_twiddle(b, a ^ acc);
    }
    out[gid] = acc ^ a ^ b;
}

kernel void __softfloat64_fle_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    ulong b = 0x3FF0000000000001UL ^ ((seed.y ^ (ulong)gid) & 0xFFUL);
    ulong acc = 0UL;
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        ulong r = (ulong)__softfloat64_fle(a, b);
        acc = (acc << 1) ^ r;
        b = __softfloat64_twiddle(b, a ^ acc);
    }
    out[gid] = acc ^ a ^ b;
}

kernel void __softfloat64_fgt_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    ulong b = 0x3FF0000000000001UL ^ ((seed.y ^ (ulong)gid) & 0xFFUL);
    ulong acc = 0UL;
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        ulong r = (ulong)__softfloat64_fgt(a, b);
        acc = (acc << 1) ^ r;
        b = __softfloat64_twiddle(b, a ^ acc);
    }
    out[gid] = acc ^ a ^ b;
}

kernel void __softfloat64_fge_chain(
    constant metal::ulong2& seed [[buffer(0)]],
    device ulong*    out  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ulong a = 0x3FF0000000000000UL ^ ((seed.x ^ (ulong)gid) & 0xFFUL);
    ulong b = 0x3FF0000000000001UL ^ ((seed.y ^ (ulong)gid) & 0xFFUL);
    ulong acc = 0UL;
    #pragma unroll 2
    for (uint i = 0; i < __SOFTFLOAT64_CHAIN_OPS; ++i) {
        ulong r = (ulong)__softfloat64_fge(a, b);
        acc = (acc << 1) ^ r;
        b = __softfloat64_twiddle(b, a ^ acc);
    }
    out[gid] = acc ^ a ^ b;
}
