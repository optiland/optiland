// df64_math_special.h — hyperbolic and special functions for double-single ("df64")
// values on Apple GPUs (Metal Shading Language).
//
// Provides, in namespace df:: for df64 arguments:
//   sinh, cosh, tanh, asinh, acosh, atanh, hypot(x, y), erf, erfc, erfinv, lgamma.
//
// AMALGAMATION CONVENTION: this header has no #include. compile.kernel_source()
// concatenates df64_core.h, df64_constants.h, df64_math_exp.h, [df64_math_trig.h]
// and then this file, so struct df64, namespace df (core arithmetic, exp/expm1/log/
// log1p, exp_of_sum, log_ext, prod3) and the DF64_* constants are in scope. Nothing
// from df64_math_trig.h is used (the reflection formula has its own sin(pi r)).
//
// Algorithms (NOTES/02-design.md 2.4):
//   * sinh: |x| <= 0.05 odd Taylor series (DF64_INV_FACT); |x| < 20 from t = expm1(|x|)
//     as (t + t/(1+t))/2 (all terms positive, no cancellation); |x| < 88 exp(|x|)/2;
//     beyond, exp(|x|/2)^2/2 so that the overflow threshold is the true one (~89.4).
//   * cosh: (e + 1/e)/2 with the same large-argument handling; cosh(0) = 1 exactly.
//   * tanh: |x| <= 0.05 s/sqrt(1 + s^2) with s from the sinh series (QD); |x| < 20
//     -m/(2 + m) with m = expm1(-2|x|) (m in (-1, 0), no cancellation); else +-1.
//   * asinh: log1p(|x| + x^2/(1 + sqrt(1 + x^2))) for |x| < 2, log(2|x| + 1/(|x| +
//     sqrt(x^2 + 1))) up to 1e8, log|x| + ln2 beyond (x^2 would overflow).
//   * acosh: t = x - 1 exact, log1p(t + sqrt(t (t + 2))) for x < 2, log(2x - 1/(x +
//     sqrt(x^2 - 1))) up to 1e8, log x + ln2 beyond; x < 1 -> NaN, acosh(1) = +0.
//   * atanh: log1p(2|x| / (1 - |x|)) / 2 (1 - |x| is exact for |x| >= 1/2).
//   * hypot: scale both magnitudes by 2^-e (e from frexp of the larger one) so the
//     squares stay in [2^-2, 2), sqrt, scale back. hypot(+-inf, NaN) = +inf.
//   * erf, 0 <= x < 1: 2/sqrt(pi) exp(-y) x sum_n (2y)^n / (2n+1)!!, y = x^2. All
//     terms are positive and the same rounded y feeds exp and the series, so its
//     rounding error nearly cancels (d/dy [exp(-y) F(y)] is small). x >= 1: 1 - erfc.
//   * erfc, 0 <= x < 1/2: 1 - erf; 1/2 <= x < 3: Taylor expansion about the nearest
//     table point x0 = 1/2 + i/8 (|h| <= 1/16) with erfc(x0) and 2/sqrt(pi) exp(-x0^2)
//     tabulated (mpmath) and the derivatives of exp(-x^2) by the Hermite recurrence;
//     x >= 3: the Laplace continued fraction exp(-x^2)/sqrt(pi) / (x + (1/2)/(x +
//     1/(x + (3/2)/(x + ...)))) evaluated bottom-up with 36 levels (converges to
//     2^-54 at x = 3 with 31); erfc(-x) = 2 - erfc(x). exp(-x^2) is formed from the
//     exact three-word square so the exponent's rounding does not cost x^2 u^2.
//   * erfinv: float32 seed (Giles 2010 single-precision polynomial, checked to
//     2.4e-7 relative against mpmath; for 1 - |x| < 2^-24 an asymptotic seed from
//     erfc(y) ~ exp(-y^2)/(y sqrt(pi)) (1 - 1/(2y^2))), then Newton steps in df64.
//     For |x| >= 1/2 the residual is written (1 - |x|) - erfc(y) with 1 - |x| exact,
//     which keeps the relative error of y at ~erfc's error / (2 y^2) instead of
//     amplifying erf's error by exp(y^2). Two steps (three in the far tail).
//     When q = 1 - |x| is below 2^-60 the residual is formed on 2^k q and
//     2^k erfc(y) with 2^k q in [1/2, 1) (the scale enters the exponent of
//     exp(-y^2) exactly), because an unscaled erfc(y) ~ q < 2^-78 carries the
//     FLT_MIN absolute floor of df64_core.h, which cost up to 1e6 u^2 on the
//     O(8) result for q in [FLT_MIN, ~5e-29].
//   * lgamma: for 0 < x < 8, x = 2 + j + t with |t| <= 1/2, lgamma(2 + t) =
//     (1 - gamma) t + sum_{k>=2} (-1)^k (zeta(k) - 1) t^k / k (28 terms) and the
//     recurrence lgamma(x) = lgamma(2 + t) + log((x-1)(x-2)...(x-j)) (or minus
//     log(x), log(x) + log1p(x) for j = -1, -2). This keeps full relative accuracy
//     at the zeros x = 1 and x = 2 (lgamma(1) = lgamma(2) = +0 exactly). x >= 8:
//     Stirling with ten Bernoulli terms, (z - 1/2) log z formed as a triple-word
//     product from the extended log (exp lane) and the sum assembled so that only
//     the final rounding is at u^2 (exp(lgamma(21)) needs lgamma to ~1 u^2).
//     x < 0: log(pi / |sin(pi x)|) - lgamma(1 - x); poles at 0, -1, -2, ... -> +inf.
//     For |r| < 2^-20 (r = x - rint(x)) the reflection term is evaluated as
//     -log|r| - log1p(sin(pi r)/(pi r) - 1) instead of log(pi) - log(pi |r| ...):
//     the df64 product pi * r loses its cross terms below FLT_MIN for
//     |r| < ~1e-26 (float32 accuracy, up to 1.5e3 u^2 on the O(70) result at
//     |r| ~ 1e-29) whereas log|r| is accurate for any normal r.
// Every float32 seed calls metal::precise:: explicitly. NaN propagates; domain
// errors return NaN (never 0); overflow +-inf (lgamma exactly at lgamma(z) >=
// FLT_MAX + 2^103, z >= 4.085e36); results below ~1.2e-38 flush to 0, and results
// below ~3e-24 can carry up to FLT_MIN of absolute error (df64_core.h floor).
// A denormal hi word (1.4e-45 <= |x| < 1.18e-38) is +-0 for every function here
// (df64_core.h denormal contract): lgamma of it is the pole +inf, cosh/erfc of
// it are 1, acosh of it is NaN, and the odd functions return the word unchanged.
//
// Part of Optiland-Metal (MIT). Written fresh; the sinh/tanh small-argument branches
// follow QD (Hida, Li & Bailey, BSD); the erfinv seed follows M. Giles, "Approximating
// the erfinv function", GPU Computing Gems Jade Edition (2011).

#ifndef OPTILAND_DF64_MATH_SPECIAL_H
#define OPTILAND_DF64_MATH_SPECIAL_H

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// ---------------------------------------------------------------------------
// Local constants (mpmath at 200 bits; normalized pairs with |lo| <= ulp(hi)/2 and
// relative representation error <= 2^-49). Generator: see the file header of
// tools/gen_df64_constants.py for the pair encoding; formulas are given inline.
// ---------------------------------------------------------------------------
// lgamma(2 + t) = sum_{k>=1} c_k t^k, c_1 = 1 - gamma, c_k = (-1)^k (zeta(k) - 1) / k.
static constant df64 DF64S_LGAMMA_C[28] = {
    {0.42278432846069336f, 6.637773886097875e-09f},  // c_1
    {0.32246702909469604f, 4.329417269133273e-09f},  // c_2
    {-0.0673523023724556f, 1.319257481036118e-09f},  // c_3
    {0.020580807700753212f, 7.270313240326232e-10f},  // c_4
    {-0.007385550998151302f, -3.0522681604416846e-11f},  // c_5
    {0.0028905102517455816f, 7.89959417324404e-11f},  // c_6
    {-0.0011927539017051458f, -9.998115182985323e-12f},  // c_7
    {0.0005096695385873318f, -1.3844289430131607e-11f},  // c_8
    {-0.00022315475507639349f, -3.377185900724222e-12f},  // c_9
    {9.945750935003161e-05f, 3.431776997991909e-12f},  // c_10
    {-4.4926237023901194e-05f, 2.857680455117806e-13f},  // c_11
    {2.050721195701044e-05f, 8.186602437620927e-13f},  // c_12
    {-9.439488167117815e-06f, -1.0815058294451688e-13f},  // c_13
    {4.3748668758780695e-06f, -8.597058476353045e-14f},  // c_14
    {-2.039215814875206e-06f, 6.107384021257489e-14f},  // c_15
    {9.55141217673372e-07f, -4.6326298645728986e-15f},  // c_16
    {-4.4924692588210746e-07f, 6.0056507903143326e-15f},  // c_17
    {2.12071853411544e-07f, -5.355997272077492e-15f},  // c_18
    {-1.0043224563105468e-07f, -2.608626214142145e-15f},  // c_19
    {4.76981014685407e-08f, 2.250991116119487e-16f},  // c_20
    {-2.2711095226668476e-08f, 6.177252995043594e-16f},  // c_21
    {1.0838658859313455e-08f, 3.555835048150692e-16f},  // c_22
    {-5.183474982572989e-09f, -5.939706046057474e-17f},  // c_23
    {2.4836745993184195e-09f, -5.551594226136573e-17f},  // c_24
    {-1.192140164363309e-09f, 2.3777217372330625e-17f},  // c_25
    {5.731367047623337e-10f, 1.9405552945736848e-17f},  // c_26
    {-2.7595228879739864e-10f, 2.849753274024879e-19f},  // c_27
    {1.3304764234778332e-10f, 1.3946615645428616e-18f},  // c_28
};
// Stirling correction B_2k / (2k (2k - 1)), k = 1..10.
static constant df64 DF64S_STIRLING_B[10] = {
    {0.0833333358168602f, -2.4835269396561444e-09f},  // k = 1
    {-0.0027777778450399637f, 6.726218887420643e-11f},  // k = 2
    {0.0007936508045531809f, -1.0902387499733823e-11f},  // k = 3
    {-0.0005952381179668009f, 2.2728706070007654e-11f},  // k = 4
    {0.0008417508215643466f, 2.0186494836815783e-11f},  // k = 5
    {-0.0019175269408151507f, 2.3288232453566593e-11f},  // k = 6
    {0.006410256493836641f, -8.358023301235917e-11f},  // k = 7
    {-0.02955065295100212f, -6.437690935889862e-10f},  // k = 8
    {0.179644376039505f, -3.670674431077714e-09f},  // k = 9
    {-1.3924322128295898f, -4.076311288514489e-09f},  // k = 10
};
// erfc(x0) at x0 = 0.5 + i/8, i = 0..20.
static constant df64 DF64S_ERFC_X0[21] = {
    {0.4795001149177551f, 7.269198132320298e-09f},  // erfc(0.5)
    {0.3767591118812561f, 5.93032600804122e-09f},  // erfc(0.625)
    {0.28884437680244446f, -1.0455959653654645e-08f},  // erfc(0.75)
    {0.215924933552742f, 5.387398527290088e-09f},  // erfc(0.875)
    {0.15729920566082f, 1.3894650985335488e-09f},  // erfc(1.0)
    {0.11161176860332489f, -3.0503266579273713e-10f},  // erfc(1.125)
    {0.07709987461566925f, -2.8721274247800466e-09f},  // erfc(1.25)
    {0.051829926669597626f, 5.48312062420564e-10f},  // erfc(1.375)
    {0.0338948518037796f, 1.720909637015211e-09f},  // erfc(1.5)
    {0.021556267514824867f, -7.548085489972323e-10f},  // erfc(1.625)
    {0.013328328728675842f, 5.214171455714123e-11f},  // erfc(1.75)
    {0.00800994224846363f, 8.141640139847084e-11f},  // erfc(1.875)
    {0.004677734803408384f, 1.7763887583122084e-10f},  // erfc(2.0)
    {0.0026540292892605066f, 7.022183529103998e-11f},  // erfc(2.125)
    {0.0014627166092395782f, -2.2558427348329246e-11f},  // erfc(2.25)
    {0.0007829382084310055f, 9.460114115678042e-12f},  // erfc(2.375)
    {0.00040695202187635005f, -4.431391018022701e-12f},  // erfc(2.5)
    {0.00020537573436740786f, 1.773809614816213e-12f},  // erfc(2.625)
    {0.00010062192450277507f, -2.3831381746469704e-12f},  // erfc(2.75)
    {4.785483906744048e-05f, 6.763329588217737e-13f},  // erfc(2.875)
    {2.2090496713644825e-05f, 2.8494060951879396e-13f},  // erfc(3.0)
};
// (2 / sqrt(pi)) exp(-x0^2) = -erfc'(x0) at the same points.
static constant df64 DF64S_ERFC_D0[21] = {
    {0.8787825703620911f, 8.573353582619347e-09f},  // x0 = 0.5
    {0.7634995579719543f, -2.2211349204326325e-08f},  // x0 = 0.625
    {0.6429310441017151f, 2.509349172896691e-08f},  // x0 = 0.75
    {0.5247450470924377f, -1.8022895398317473e-09f},  // x0 = 0.875
    {0.41510748863220215f, 8.788392236169784e-09f},  // x0 = 1.0
    {0.31827396154403687f, -3.0432676378921997e-09f},  // x0 = 1.125
    {0.23652112483978271f, -2.3924919823059554e-09f},  // x0 = 1.25
    {0.1703597754240036f, -1.7364879534298439e-09f},  // x0 = 1.375
    {0.11893028765916824f, 1.564461116210225e-09f},  // x0 = 1.5
    {0.08047226071357727f, -1.6910660649571696e-09f},  // x0 = 1.625
    {0.05277499556541443f, 3.647359358982527e-10f},  // x0 = 1.75
    {0.033545829355716705f, -9.315006543886284e-10f},  // x0 = 1.875
    {0.02066698484122753f, 5.128645286234246e-10f},  // x0 = 2.0
    {0.012340820394456387f, 2.1987731080308492e-10f},  // x0 = 2.125
    {0.00714231887832284f, 1.4369513912093623e-10f},  // x0 = 2.25
    {0.004006478004157543f, -1.4248732749244652e-10f},  // x0 = 2.375
    {0.002178284339606762f, -1.0925405025119517e-10f},  // x0 = 2.5
    {0.0011478750966489315f, 2.9233743897849607e-11f},  // x0 = 2.625
    {0.0005862772231921554f, 2.3901637613565896e-11f},  // x0 = 2.75
    {0.0002902282867580652f, -3.895567028677771e-12f},  // x0 = 2.875
    {0.00013925305393058807f, -1.9838401754679325e-12f},  // x0 = 3.0
};
// Largest |x| handled by the expm1-based sinh/tanh branches; beyond it e^-2|x| < 2^-57.
static constant float DF64S_HYP_SMALL = 0.05f;
static constant float DF64S_HYP_MID = 20.0f;
static constant float DF64S_HYP_BIG = 88.0f;
// erf/erfc region boundaries and continued-fraction depth.
static constant float DF64S_ERF_SERIES_MAX = 1.0f;
static constant float DF64S_ERFC_TABLE_MIN = 0.5f;
static constant float DF64S_ERFC_TABLE_MAX = 3.0f;
// erfc(x) < FLT_MIN = 1.1755e-38 for x > 9.194549 (erfc(9.1945) = 1.1766e-38 is
// still a normal value, erfc(9.1946) = 1.1744e-38 flushes); the continued
// fraction runs up to this guard and exp_of_sum flushes the result to +0 in
// between.
static constant float DF64S_ERFC_ZERO = 9.5f;
static constant int DF64S_ERFC_CF_DEPTH = 36;
static constant int DF64S_ERFC_TAYLOR_TERMS = 16;  // 12 needed for 2^-54 at |h| = 1/16
// Giles single-precision erfinv polynomial covers w = -log(1 - x^2) < ~17.
static constant float DF64S_ERFINV_W_MAX = 17.0f;
// lgamma: Stirling from this argument on; series/recurrence below.
static constant float DF64S_LGAMMA_STIRLING_MIN = 8.0f;
// Reflection: below this |x - rint(x)| the sin(pi r) factor is taken as
// pi r (1 + g) with log|r| evaluated directly (see the file header).
static constant float DF64S_LGAMMA_SINPI_SMALL = 9.5367431640625e-07f;  // 2^-20
// erfinv: below this 1 - |x| the far-tail Newton residual is formed on 2^k-scaled
// operands (see the file header).
static constant float DF64S_ERFINV_SCALE_BELOW = 8.673617379884035e-19f;  // 2^-60
// (ln sqrt(2 pi) - 1/2) / 2 (mpmath 300 bits).
static constant df64 DF64S_LGAMMA_STIRLING_C = {0.20946927f, -7.0840724e-09f};

namespace df {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// x^2 as a leading float P plus a df64 tail, P + tail = x^2 to ~2^-70 relative.
inline void sp_sqr_ext(df64 x, thread float &P, thread df64 &tail) {
    df64 p = two_sqr(x.hi);
    df64 q = two_prod(2.0f * x.hi, x.lo);
    P = p.hi;
    tail = two_sum(p.lo, q.hi);
    tail = add(tail, fma(x.lo, x.lo, q.lo));
}

// exp(-x^2) with the exponent formed to ~2^-70 (x finite).
inline df64 sp_exp_neg_sqr(df64 x) {
    float P;
    df64 tail;
    sp_sqr_ext(x, P, tail);
    return exp_of_sum(-P, neg(tail));
}
// 2^k exp(-x^2) for an integer |k| <= 128, formed as exp(-x^2 + k ln2) with
// k ln2 = k A + k B + k C (exp-lane Cody-Waite parts: k A and k B are exact
// products, k C a df64) so that the scaled result never passes through the
// flushed range; k = 0 is the plain sp_exp_neg_sqr.
inline df64 sp_exp_neg_sqr_scaled(df64 x, int k) {
    float P;
    df64 tail;
    sp_sqr_ext(x, P, tail);
    if (k == 0) return exp_of_sum(-P, neg(tail));
    float fk = float(k);
    df64 lead = two_sum(-P, fk * DF64X_LN2_A);   // exact
    df64 t = add(neg(tail), lead.lo);
    t = add(t, fk * DF64X_LN2_B);                // exact product
    t = add(t, mul(DF64X_LN2_C, fk));
    return exp_of_sum(lead.hi, t);
}

// sinh by its odd Taylor series through x^11/11! (|x| <= 0.05: x^13/13! < 2^-70 x).
inline df64 sp_sinh_taylor(df64 x) {
    df64 y = sqr(x);
    df64 p = DF64_INV_FACT[11];
    p = add(mul(p, y), DF64_INV_FACT[9]);
    p = add(mul(p, y), DF64_INV_FACT[7]);
    p = add(mul(p, y), DF64_INV_FACT[5]);
    p = add(mul(p, y), DF64_INV_FACT[3]);
    // x + x (y p): the polynomial part is <= 4.2e-4 of x, damping its rounding.
    return add(x, mul(x, mul(y, p)));
}

// erf(x) for 0 <= x < 1: 2/sqrt(pi) exp(-y) x sum_{n>=0} (2y)^n / (2n+1)!!, y = x^2.
inline df64 sp_erf_series(df64 x) {
    df64 y = sqr(x);
    df64 y2 = mul_pwr2(y, 2.0f);
    df64 t = x;
    df64 s = x;
    for (int n = 1; n <= 24; ++n) {
        t = div(mul(t, y2), float(2 * n + 1));
        s = add(s, t);
        if (t.hi <= s.hi * 1.2e-16f) break;  // remaining tail < t (ratio <= 2/5)
    }
    return mul(mul(DF64_2_SQRTPI, exp(neg(y))), s);
}

// erfc(x) for 1/2 <= x < 3 by a Taylor expansion about x0 = 1/2 + i/8:
//   erfc(x0 + h) = erfc(x0) - (2/sqrt(pi)) exp(-x0^2) h sum_{m>=0} d_m / (m + 1),
//   d_0 = 1, d_1 = -2 x0 h, d_{m+1} = -2h (x0 d_m + h d_{m-1}) / (m + 1)
// (d_m = g^(m)(x0) h^m / (m! g(x0)) for g = exp(-x^2), by g' = -2x g).
inline df64 sp_erfc_table(df64 x) {
    float fi = metal::rint((x.hi - 0.5f) * 8.0f);
    int i = int(fi);
    float x0 = 0.5f + fi * 0.125f;      // exact
    df64 h = sub(x, x0);                // exact (Sterbenz), |h| <= 1/16
    df64 d_prev = one();
    df64 d = mul(h, -2.0f * x0);
    df64 T = add(one(), mul_pwr2(d, 0.5f));
    for (int m = 1; m < DF64S_ERFC_TAYLOR_TERMS; ++m) {
        df64 inner = add(mul(d, x0), mul(d_prev, h));
        df64 dn = div(mul_pwr2(mul(inner, h), 2.0f), -float(m + 1));
        d_prev = d;
        d = dn;
        T = add(T, div(dn, float(m + 2)));
    }
    return sub(DF64S_ERFC_X0[i], mul(mul(DF64S_ERFC_D0[i], h), T));
}

// 2^k erfc(x) for x >= 3 by the Laplace continued fraction, bottom-up (k = 0:
// erfc itself; erfinv's far tail uses k > 0 to keep the residual normal).
inline df64 sp_erfc_cf_scaled(df64 x, int k) {
    df64 f = x;
    for (int n = DF64S_ERFC_CF_DEPTH; n >= 1; --n) {
        f = add(x, div(0.5f * float(n), f));
    }
    df64 e = sp_exp_neg_sqr_scaled(x, k);
    return div(mul(e, mul_pwr2(DF64_2_SQRTPI, 0.5f)), f);
}
inline df64 sp_erfc_cf(df64 x) { return sp_erfc_cf_scaled(x, 0); }

inline df64 sp_erfc_pos(df64 ax);

// erf(x) for finite x >= 0.
inline df64 sp_erf_pos(df64 ax) {
    if (ax.hi < DF64S_ERF_SERIES_MAX) return sp_erf_series(ax);
    if (ax.hi < 6.0f) return sub(1.0f, sp_erfc_pos(ax));  // erfc(6) < 2^-55
    return one();
}

// erfc(x) for finite x >= 0.
inline df64 sp_erfc_pos(df64 ax) {
    if (ax.hi < DF64S_ERFC_TABLE_MIN) return sub(1.0f, sp_erf_series(ax));
    if (ax.hi < DF64S_ERFC_TABLE_MAX) return sp_erfc_table(ax);
    if (ax.hi < DF64S_ERFC_ZERO) return sp_erfc_cf(ax);
    return zero();
}

// Giles 2010 single-precision erfinv polynomial in w = -log((1-x)(1+x)); returns p
// with erfinv(x) ~= p * x (max relative error 2.4e-7 for |x| <= 1 - 2^-24).
inline float sp_erfinv_giles(float w) {
    float p;
    if (w < 5.0f) {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = fma(p, w, 3.43273939e-07f);
        p = fma(p, w, -3.5233877e-06f);
        p = fma(p, w, -4.39150654e-06f);
        p = fma(p, w, 0.00021858087f);
        p = fma(p, w, -0.00125372503f);
        p = fma(p, w, -0.00417768164f);
        p = fma(p, w, 0.246640727f);
        p = fma(p, w, 1.50140941f);
    } else {
        w = metal::precise::sqrt(w) - 3.0f;
        p = -0.000200214257f;
        p = fma(p, w, 0.000100950558f);
        p = fma(p, w, 0.00134934322f);
        p = fma(p, w, -0.00367342844f);
        p = fma(p, w, 0.00573950773f);
        p = fma(p, w, -0.0076224613f);
        p = fma(p, w, 0.00943887047f);
        p = fma(p, w, 1.00167406f);
        p = fma(p, w, 2.83297682f);
    }
    return p;
}

// |sin(pi r)| for |r| <= 1/2 by Taylor series (sin for |r| <= 1/4, cos of the
// complement otherwise; argument <= pi/4, terms through t^17 / 17!).
inline df64 sp_abs_sinpi(df64 r) {
    df64 ar = abs(r);
    if (ar.hi <= 0.25f) {
        df64 t = mul(DF64_PI, ar);
        df64 y = sqr(t);
        df64 p = DF64_INV_FACT[17];
        p = sub(DF64_INV_FACT[15], mul(p, y));
        p = sub(DF64_INV_FACT[13], mul(p, y));
        p = sub(DF64_INV_FACT[11], mul(p, y));
        p = sub(DF64_INV_FACT[9], mul(p, y));
        p = sub(DF64_INV_FACT[7], mul(p, y));
        p = sub(DF64_INV_FACT[5], mul(p, y));
        p = sub(DF64_INV_FACT[3], mul(p, y));
        return sub(t, mul(mul(t, y), p));
    }
    df64 t = mul(DF64_PI, sub(0.5f, ar));   // 1/2 - |r| is exact
    df64 y = sqr(t);
    df64 p = DF64_INV_FACT[16];
    p = sub(DF64_INV_FACT[14], mul(p, y));
    p = sub(DF64_INV_FACT[12], mul(p, y));
    p = sub(DF64_INV_FACT[10], mul(p, y));
    p = sub(DF64_INV_FACT[8], mul(p, y));
    p = sub(DF64_INV_FACT[6], mul(p, y));
    p = sub(DF64_INV_FACT[4], mul(p, y));
    p = sub(DF64_INV_FACT[2], mul(p, y));
    return sub(1.0f, mul(y, p));
}

// lgamma(2 + t) for |t| <= 1/2 (a few ulp more is fine): t * sum_k c_k t^(k-1).
inline df64 sp_lgamma_series(df64 t) {
    df64 p = DF64S_LGAMMA_C[27];
    for (int k = 26; k >= 0; --k) {
        p = add(mul(p, t), DF64S_LGAMMA_C[k]);
    }
    return mul(t, p);
}

// lgamma(x) for 0 < x < 8 via the series at 2 + t and the recurrence.
inline df64 sp_lgamma_small(df64 x) {
    float fj = metal::rint(x.hi - 2.0f);   // j in {-2, ..., 6}
    int j = int(fj);
    df64 t = sub(x, 2.0f + fj);            // exact
    df64 S = sp_lgamma_series(t);
    if (j == 0) return fix(S);
    if (j == -1) return fix(sub(S, log(x)));
    if (j == -2) return fix(sub(S, add(log(x), log1p(x))));
    df64 P = sub(x, 1.0f);
    for (int i = 2; i <= j; ++i) P = mul(P, sub(x, float(i)));
    return fix(add(S, log(P)));
}

// lgamma(z) for z >= 8: (z - 1/2) log z - z + ln sqrt(2 pi) + sum B_2k / (2k (2k-1) z^(2k-1)),
// evaluated as (z - 1/2)(log z - 1) + (ln sqrt(2 pi) - 1/2) + corr so that the
// leading product overflows only when the result does ((z - 1/2) log z alone
// exceeds FLT_MAX for z >= 4.037e36 although lgamma(z) is finite up to
// 4.085e36). The product is formed on (z - 1/2)/2 and the sum doubled at the
// end (exact), which lets mul_pwr2 apply the df64 overflow rule at the true
// threshold. log z - 1 is exact in the hi word (log z >= 2 here).
inline df64 sp_lgamma_stirling(df64 z) {
    df64 rz = recip(z);
    df64 w = sqr(rz);
    df64 p = DF64S_STIRLING_B[9];
    for (int k = 8; k >= 0; --k) {
        p = add(mul(p, w), DF64S_STIRLING_B[k]);
    }
    df64 corr = mul(p, rz);
    float ext;
    df64 L = log_ext(z, ext);
    df64 Lm = quick_two_sum(L.hi - 1.0f, L.lo);
    df64 a = mul_pwr2(sub(z, 0.5f), 0.5f);
    float P;
    df64 tail;
    prod3(a, 0.0f, float3(Lm.hi, Lm.lo, ext), P, tail);   // (z - 1/2)(log z - 1)/2 to ~2^-70
    if (!isfinite(P)) return inf();
    df64 rest = add(tail, DF64S_LGAMMA_STIRLING_C);
    rest = add(rest, mul_pwr2(corr, 0.5f));
    return mul_pwr2(add(make(P), rest), 2.0f);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// sinh(x): odd, sinh(+-0) = +-0, sinh(+-inf) = +-inf, overflow -> +-inf (|x| > ~89.4).
inline df64 sinh(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x) || !isfinite(x.hi)) return fix(x);
    df64 ax = abs(x);
    df64 r;
    if (ax.hi <= DF64S_HYP_SMALL) {
        r = sp_sinh_taylor(ax);
    } else if (ax.hi < DF64S_HYP_MID) {
        df64 t = expm1(ax);
        r = mul_pwr2(add(t, div(t, add(t, 1.0f))), 0.5f);
    } else if (ax.hi < DF64S_HYP_BIG) {
        r = mul_pwr2(exp(ax), 0.5f);
    } else {
        // (h/2)^2 * 2 = e^|x| / 2 without overflowing before the true threshold (~89.4).
        df64 h = exp(mul_pwr2(ax, 0.5f));
        r = mul_pwr2(sqr(mul_pwr2(h, 0.5f)), 2.0f);
    }
    return fix(is_negative(x) ? neg(r) : r);
}

// cosh(x): even, cosh(+-0) = 1 exactly, cosh(+-inf) = +inf, overflow -> +inf.
inline df64 cosh(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return one();
    if (!isfinite(x.hi)) return inf();
    df64 ax = abs(x);
    if (ax.hi < DF64S_HYP_MID) {
        df64 e = exp(ax);
        return fix(mul_pwr2(add(e, recip(e)), 0.5f));
    }
    if (ax.hi < DF64S_HYP_BIG) return mul_pwr2(exp(ax), 0.5f);
    df64 h = exp(mul_pwr2(ax, 0.5f));
    return fix(mul_pwr2(sqr(mul_pwr2(h, 0.5f)), 2.0f));
}

// tanh(x): odd, tanh(+-0) = +-0, tanh(+-inf) = +-1.
inline df64 tanh(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return x;
    if (!isfinite(x.hi)) return make(x.hi > 0.0f ? 1.0f : -1.0f, 0.0f);
    df64 ax = abs(x);
    df64 r;
    if (ax.hi <= DF64S_HYP_SMALL) {
        df64 s = sp_sinh_taylor(ax);
        r = div(s, sqrt(add(sqr(s), 1.0f)));
    } else if (ax.hi < DF64S_HYP_MID) {
        df64 m = expm1(mul_pwr2(ax, -2.0f));   // in (-1, 0)
        r = div(neg(m), add(m, 2.0f));
    } else {
        r = one();
    }
    return fix(is_negative(x) ? neg(r) : r);
}

// asinh(x): odd, asinh(+-0) = +-0, asinh(+-inf) = +-inf.
inline df64 asinh(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x) || !isfinite(x.hi)) return fix(x);
    df64 ax = abs(x);
    df64 r;
    if (ax.hi < 2.9802322e-08f) {            // 2^-25: x^3/6 < 2^-51 x
        r = ax;
    } else if (ax.hi < 2.0f) {
        df64 y = sqr(ax);
        df64 s = sqrt(add(y, 1.0f));
        r = log1p(add(ax, div(y, add(s, 1.0f))));
    } else if (ax.hi < 1.0e8f) {
        df64 s = sqrt(add(sqr(ax), 1.0f));
        r = log(add(mul_pwr2(ax, 2.0f), recip(add(ax, s))));
    } else {
        r = add(log(ax), DF64_LN2);
    }
    return fix(is_negative(x) ? neg(r) : r);
}

// acosh(x): x < 1 -> NaN, acosh(1) = +0, acosh(+inf) = +inf.
inline df64 acosh(df64 x) {
    if (is_nan(x)) return nan();
    if (x.hi < 1.0f || (x.hi == 1.0f && x.lo < 0.0f)) return nan();
    if (x.hi == 1.0f && x.lo == 0.0f) return zero();
    if (!isfinite(x.hi)) return inf();
    if (x.hi < 2.0f) {
        df64 t = sub(x, 1.0f);               // exact
        return fix(log1p(add(t, sqrt(mul(t, add(t, 2.0f))))));
    }
    if (x.hi < 1.0e8f) {
        df64 s = sqrt(sub(sqr(x), 1.0f));
        return fix(log(sub(mul_pwr2(x, 2.0f), recip(add(x, s)))));
    }
    return fix(add(log(x), DF64_LN2));
}

// atanh(x): odd, |x| > 1 -> NaN, atanh(+-1) = +-inf, atanh(+-0) = +-0.
inline df64 atanh(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return x;
    df64 ax = abs(x);
    if (ax.hi > 1.0f || (ax.hi == 1.0f && ax.lo > 0.0f)) return nan();
    if (ax.hi == 1.0f && ax.lo == 0.0f) return make(metal::copysign(INFINITY, x.hi), 0.0f);
    df64 t = div(mul_pwr2(ax, 2.0f), sub(1.0f, ax));
    df64 r = mul_pwr2(log1p(t), 0.5f);
    return fix(is_negative(x) ? neg(r) : r);
}

// hypot(x, y) = sqrt(x^2 + y^2) without intermediate over/underflow; hypot(+-inf, NaN)
// = hypot(NaN, +-inf) = +inf, other NaN -> NaN, hypot(+-0, +-0) = +0, hypot(x, +-0) = |x|.
inline df64 hypot(df64 x, df64 y) {
    if (isinf(x.hi) || isinf(y.hi)) return inf();
    if (is_nan(x) || is_nan(y)) return nan();
    df64 ax = abs(x);
    df64 ay = abs(y);
    if (is_zero(ax)) return is_zero(ay) ? zero() : ay;
    if (is_zero(ay)) return ax;
    float m = metal::max(ax.hi, ay.hi);
    int e;
    metal::frexp(m, e);
    df64 xs = ldexp(ax, -e);
    df64 ys = ldexp(ay, -e);
    df64 s = add(sqr(xs), sqr(ys));
    return fix(ldexp(sqrt(s), e));
}

// erf(x): odd, erf(+-0) = +-0, erf(+-inf) = +-1.
inline df64 erf(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return x;
    if (!isfinite(x.hi)) return make(x.hi > 0.0f ? 1.0f : -1.0f, 0.0f);
    df64 r = sp_erf_pos(abs(x));
    return fix(is_negative(x) ? neg(r) : r);
}

// erfc(x) = 1 - erf(x) with relative accuracy: erfc(-x) = 2 - erfc(x), erfc(+inf) = 0,
// erfc(-inf) = 2, erfc(+-0) = 1; x > 9.194549 underflows to +0 (erfc(x) < FLT_MIN;
// the last nonzero result is erfc(9.1945) = 1.1766e-38).
inline df64 erfc(df64 x) {
    if (is_nan(x)) return nan();
    if (!isfinite(x.hi)) return x.hi > 0.0f ? zero() : make(2.0f, 0.0f);
    if (is_zero(x)) return one();
    df64 r = sp_erfc_pos(abs(x));
    return fix(is_negative(x) ? sub(2.0f, r) : r);
}

// erfinv(x): odd, |x| > 1 -> NaN, erfinv(+-1) = +-inf, erfinv(+-0) = +-0.
inline df64 erfinv(df64 x) {
    if (is_nan(x)) return nan();
    if (is_zero(x)) return x;
    df64 ax = abs(x);
    if (ax.hi > 1.0f || (ax.hi == 1.0f && ax.lo > 0.0f)) return nan();
    if (ax.hi == 1.0f && ax.lo == 0.0f) return make(metal::copysign(INFINITY, x.hi), 0.0f);
    df64 q = sub(1.0f, ax);                  // exact for |x| >= 1/2
    float w = -metal::precise::log(to_float(mul(q, add(1.0f, ax))));
    df64 y;
    int steps;
    if (w < DF64S_ERFINV_W_MAX) {
        y = make(sp_erfinv_giles(w) * ax.hi);
        steps = 2;
    } else {
        // Far tail: y^2 ~= L - log y + log(1 - 1/(2y^2)), L = -log(q sqrt(pi)).
        float L = -metal::precise::log(to_float(q) * 1.7724539f);
        float yf = metal::precise::sqrt(L);
        for (int i = 0; i < 3; ++i) {
            float c = metal::precise::log(1.0f - 0.5f / (yf * yf));
            yf = metal::precise::sqrt(L - metal::precise::log(yf) + c);
        }
        y = make(yf);
        steps = 3;
    }
    bool big = ax.hi >= 0.5f;
    // Far tail: q = 1 - |x| < 2^-60 (so y > 6.3 and erfc is on its continued
    // fraction) forms the residual on 2^k q in [1/2, 1) and 2^k erfc(y): an
    // unscaled erfc(y) ~ q below 2^-78 carries the FLT_MIN absolute floor,
    // which Newton turned into up to 1e6 u^2 of the O(8) result.
    int k = 0;
    df64 qs = q;
    if (big && q.hi < DF64S_ERFINV_SCALE_BELOW) {
        int e;
        metal::frexp(q.hi, e);   // q = m 2^e, m in [1/2, 1)
        k = -e;
        qs = ldexp(q, k);        // exact (q is a single word)
    }
    for (int it = 0; it < steps; ++it) {
        // f(y) = erf(y) - |x|, written as (1 - |x|) - erfc(y) for |x| >= 1/2.
        df64 f = big ? sub(qs, (k == 0 ? sp_erfc_pos(y) : sp_erfc_cf_scaled(y, k)))
                     : sub(sp_erf_pos(y), ax);
        df64 fp = mul(DF64_2_SQRTPI, sp_exp_neg_sqr_scaled(y, k));
        y = sub(y, div(f, fp));
    }
    return fix(is_negative(x) ? neg(y) : y);
}

// lgamma(x) = log|Gamma(x)|: poles at 0, -1, -2, ... and +-inf -> +inf; lgamma(1) =
// lgamma(2) = +0 exactly; NaN -> NaN. A denormal hi word is zero for the kernel
// and therefore the pole (+inf), like every other df64 function treats it as 0.
inline df64 lgamma(df64 x) {
    if (is_nan(x)) return nan();
    if (!isfinite(x.hi) || is_zero(x)) return inf();
    if (x.hi >= DF64S_LGAMMA_STIRLING_MIN) return sp_lgamma_stirling(x);
    if (x.hi > 0.0f) return sp_lgamma_small(x);
    // Reflection: lgamma(x) = log(pi / |sin(pi x)|) - lgamma(1 - x).
    df64 r = sub(x, rint(x));                // exact, |r| <= 1/2
    if (is_zero(r)) return inf();
    df64 omx = sub(1.0f, x);                 // > 1
    df64 lg = omx.hi >= DF64S_LGAMMA_STIRLING_MIN ? sp_lgamma_stirling(omx)
                                                  : sp_lgamma_small(omx);
    df64 ls;                                 // log(pi / |sin(pi r)|)
    if (metal::abs(r.hi) < DF64S_LGAMMA_SINPI_SMALL) {
        // |sin(pi r)| = pi |r| (1 + g), g = -t^2/6 (1 - t^2/20 (1 - t^2/42)),
        // t = pi r, so log(pi / |sin(pi r)|) = -log|r| - log1p(g). |g| < 2e-12
        // here, so float32-level accuracy of t^2 (its cross terms may flush for
        // tiny r) is far more than the result needs; log|r| itself is accurate
        // for any normal r, whereas log(pi * r) inherited the FLT_MIN floor of
        // the product (up to 1.5e3 u^2 on the O(70) result at |r| ~ 1e-29).
        df64 y = sqr(mul(DF64_PI, r));
        df64 g = mul(y, -1.0f / 6.0f);
        g = mul(g, sub(1.0f, mul(y, mul(sub(1.0f, mul(y, 1.0f / 42.0f)), 0.05f))));
        ls = neg(add(log(abs(r)), log1p(g)));
    } else {
        ls = sub(DF64_LNPI, log(sp_abs_sinpi(r)));
    }
    return fix(sub(ls, lg));
}

}  // namespace df

#endif  // OPTILAND_DF64_MATH_SPECIAL_H
