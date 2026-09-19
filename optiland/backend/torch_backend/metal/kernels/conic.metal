// conic.metal — fused finite-conic ray intersection candidates (df64 and sf64).
//
// One thread per ray computes exactly the quantities of
// optiland/backend/_conic.py::_conic_candidates — same coefficients, same
// stable root formula, same admissibility masks — in emulated float64, so the
// Python side only has to apply the aperture preference and the final
// selection (_select_distance) with a handful of elementwise ops instead of
// ~40 launches per surface.
//
// Requires df64_core.h (and sf64_core.h for the sf64 variant) earlier in the
// amalgamation. Inputs are contiguous component arrays of equal length; radius
// and conic are scalars; eps is the machine epsilon of the representation
// (2^-48 for df64, 2^-53 for sf64) so the roundoff bands match the CPU path's
// finfo-derived thresholds.
//
// Outputs: t1, t2 (components) and one uchar flag word per ray:
//   bit 0 valid1, bit 1 valid2, bit 2 pick2, bit 3 solvable, bit 4 regular,
//   bit 5 solvable1, bit 6 solvable2.

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

#ifndef OPTILAND_CONIC_METAL
#define OPTILAND_CONIC_METAL

template <typename R>
struct conic_ops;

template <>
struct conic_ops<df64> {
    static inline df64 lit(float v) { return df::make(v, 0.0f); }
    static inline df64 abs(df64 a) { return df::abs(a); }
    static inline df64 sqrt(df64 a) { return df::sqrt(a); }
    static inline df64 copysign(df64 a, df64 b) { return df::copysign(a, b); }
    static inline bool is_finite(df64 a) { return df::is_finite(a); }
    static inline bool is_zero(df64 a) { return df::is_zero(a); }
    static inline df64 inf() { return df::inf(); }
};

#ifdef OPTILAND_SF64_CORE_H
template <>
struct conic_ops<sf64> {
    static inline sf64 lit(float v) { return sf::from_float(v); }
    static inline sf64 abs(sf64 a) { return sf::abs(a); }
    static inline sf64 sqrt(sf64 a) { return sf::sqrt(a); }
    static inline sf64 copysign(sf64 a, sf64 b) { return sf::copysign(a, b); }
    static inline bool is_finite(sf64 a) { return sf::is_finite(a); }
    static inline bool is_zero(sf64 a) { return sf::is_zero(a); }
    static inline sf64 inf() { return sf::inf(); }
};
#endif

// Shared body; R is df64 or sf64. Mirrors _conic_candidates line by line.
template <typename R>
inline uchar conic_candidates_body(R x, R y, R z, R L, R M, R N, R radius, R conic, R eps,
                                   thread R &t1_out, thread R &t2_out) {
    typedef conic_ops<R> O;
    const R zero = O::lit(0.0f), one = O::lit(1.0f), two = O::lit(2.0f), four = O::lit(4.0f);
    R k1 = one + conic;
    R kz = k1 * z;
    R transverse = x * x + y * y;
    R transverse_direction = L * L + M * M;
    R N2 = N * N;
    R a = transverse_direction + k1 * N2;
    R b = two * (L * x + M * y + N * (kz - radius));
    R c = transverse + z * (kz - two * radius);
    R residual_scale = transverse + O::abs(z) * (O::abs(kz) + two * O::abs(radius));
    R roundoff = four * eps;
    bool resolved_c = O::abs(c) > roundoff * residual_scale;
    R d = b * b - four * a * c;
    bool d_ok = d >= zero;
    bool positive_d = d > zero;
    R sqrt_d = positive_d ? O::sqrt(d) : zero;
    R q = O::lit(-0.5f) * (b + O::copysign(sqrt_d, b));
    bool a_ok = (a != zero) && O::is_finite(a);
    bool q_ok = (q != zero) && O::is_finite(q);
    R t1 = q / (a_ok ? a : one);
    R t2 = c / (q_ok ? q : one);
    bool resolved_step = t2 * t2 * (transverse_direction + N2) > roundoff * roundoff * (transverse + z * z);
    bool solvable1 = d_ok && a_ok && O::is_finite(t1);
    bool solvable2 = d_ok && q_ok && O::is_finite(t2);
    R z1 = z + t1 * N;
    R z2 = z + t2 * N;
    bool valid1 = solvable1 && (t1 > zero) && ((one - k1 * z1 / radius) >= zero);
    bool valid2 = solvable2 && (resolved_c || resolved_step) && (t2 > zero) && ((one - k1 * z2 / radius) >= zero);
    R az2 = solvable2 ? O::abs(z2) : O::inf();
    R az1 = solvable1 ? O::abs(z1) : O::inf();
    bool vertex2 = az2 < az1;
    bool pick2 = valid2 || (valid1 ? false : vertex2);
    bool solvable = solvable1 || solvable2;
    bool regular = solvable && (positive_d || (a == zero));
    t1_out = t1;
    t2_out = t2;
    return uchar(valid1) | (uchar(valid2) << 1) | (uchar(pick2) << 2) | (uchar(solvable) << 3) |
           (uchar(regular) << 4) | (uchar(solvable1) << 5) | (uchar(solvable2) << 6);
}

kernel void conic_candidates_df64(
    const device float* xh [[buffer(0)]], const device float* xl [[buffer(1)]],
    const device float* yh [[buffer(2)]], const device float* yl [[buffer(3)]],
    const device float* zh [[buffer(4)]], const device float* zl [[buffer(5)]],
    const device float* Lh [[buffer(6)]], const device float* Ll [[buffer(7)]],
    const device float* Mh [[buffer(8)]], const device float* Ml [[buffer(9)]],
    const device float* Nh [[buffer(10)]], const device float* Nl [[buffer(11)]],
    constant float2& radius [[buffer(12)]], constant float2& conic [[buffer(13)]],
    constant float2& eps [[buffer(14)]],
    device float* t1h [[buffer(15)]], device float* t1l [[buffer(16)]],
    device float* t2h [[buffer(17)]], device float* t2l [[buffer(18)]],
    device uchar* flags [[buffer(19)]],
    uint i [[thread_position_in_grid]]) {
    df64 t1, t2;
    uchar f = conic_candidates_body<df64>(
        df::make(xh[i], xl[i]), df::make(yh[i], yl[i]), df::make(zh[i], zl[i]),
        df::make(Lh[i], Ll[i]), df::make(Mh[i], Ml[i]), df::make(Nh[i], Nl[i]),
        df::make(radius.x, radius.y), df::make(conic.x, conic.y), df::make(eps.x, eps.y), t1, t2);
    t1h[i] = t1.hi; t1l[i] = t1.lo; t2h[i] = t2.hi; t2l[i] = t2.lo; flags[i] = f;
}

#ifdef OPTILAND_SF64_CORE_H
kernel void conic_candidates_sf64(
    const device long* x [[buffer(0)]], const device long* y [[buffer(1)]],
    const device long* z [[buffer(2)]], const device long* L [[buffer(3)]],
    const device long* M [[buffer(4)]], const device long* N [[buffer(5)]],
    constant long& radius [[buffer(6)]], constant long& conic [[buffer(7)]],
    constant long& eps [[buffer(8)]],
    device long* t1 [[buffer(9)]], device long* t2 [[buffer(10)]],
    device uchar* flags [[buffer(11)]],
    uint i [[thread_position_in_grid]]) {
    sf64 r1, r2;
    uchar f = conic_candidates_body<sf64>(
        sf::make(ulong(x[i])), sf::make(ulong(y[i])), sf::make(ulong(z[i])),
        sf::make(ulong(L[i])), sf::make(ulong(M[i])), sf::make(ulong(N[i])),
        sf::make(ulong(radius)), sf::make(ulong(conic)), sf::make(ulong(eps)), r1, r2);
    t1[i] = long(r1.bits); t2[i] = long(r2.bits); flags[i] = f;
}
#endif

#endif  // OPTILAND_CONIC_METAL
