// trace.metal — fused sequential ray trace over a whole surface list (df64 and sf64).
//
// One thread per (design b, ray i).  The thread walks surfaces s = 1 .. S-1 of
// design b, reproducing the Python per-op path's operation ORDER exactly
// ("mirror, never improve", plan 0.2.1 / design 4.2), so the acceptance
// criterion is raw-component equality (hi/lo float32 words in df64, int64 bit
// patterns in sf64) with the per-op path on the same GPU.
//
// Requires, earlier in the amalgamation: the mode's header stack
// (df64_core.h ... and, for sf64, sf64_core.h / sf64_math.h), conic.metal
// (for conic_ops<R> / conic_candidates_body<R>) and trace_layout.h (the
// OT_* slot/flag/status #defines rendered from metal/trace_layout.py).
//
// Buffers bind POSITIONALLY in the order of trace_layout.BUFFER_ORDER:
//   launch, surf_int, surf_real, coef, dims, consts, snap, final, status, iters
// R-typed buffers bind as (hi, lo) in df64 (16 bindings) and as one long
// buffer in sf64 (10 bindings).  Every buffer is contiguous; strides are
// implied by dims, never passed.
//
// Offsets (all in ulong, matmul.metal:59-65 precedent), with
// Lb = (launch_stride == 0) ? 1 : B:
//   launch   (q * Lb + b * launch_stride) * N + i
//   snap     ((q * B + b) * n_rows + row) * N + i
//   final    (q * B + b) * N + i
//   surf_int (b * S + s) * OT_SI_STRIDE + slot
//   surf_real(b * S + s) * P + slot          (P = dims[OT_D_P] = OT_SR_STRIDE)
//   coef     (b * S + s) * C + j             (C = dims[OT_D_C])
//   status   (b * S + s) * N + i             (iters likewise)
//
// ---------------------------------------------------------------------------
// STATUS OF THIS FILE: COMPLETE (kernel side).
// ---------------------------------------------------------------------------
// WP1 part 1: `trace_ops<R>` and the literal discipline of design 4.2,
// `pow_scalar`, `localize` / `globalize` in the Python order (design 4.5, note
// 03), `advance` (straight-line propagation), `absorb` (the five-op absorption
// chain of design 4.10), `accumulate_opd`, `ap_contains` for the four aperture
// codes (design 4.9) and `interact` (design 4.8).
//
// WP1 part 2: `sag_of` / `normal_of` for PLANE, STD_INF, CONIC, EVEN and ODD
// (design 4.7), `plane_distance`, `std_inf_distance`, `select_distance` over
// `conic_candidates_body<R>` with the aperture-in-root-selection preference,
// `newton_distance` (a per-thread mirror of `_solve_distance_primal` with the
// 8*eps / 32*eps floors, the per-thread freeze and `iters`) and
// `surface_distance`, the SI_GEOM switch that also raises MISS (design 4.6).
//
// WP1 part 3: `trace_body` wires all of the above in the exact Python order of
// design 4.5 (object row, then per surface: localize -> distance ->
// propagate/absorption -> OPD -> clip -> interact -> globalize -> record), the
// status / iters planes with the object-row contract of plan 3.2, and the
// final planes with the last surface's local-frame L0/M0/N0.  Both entry
// points call it.
//
// Every surface step has a probe kernel and a differential unit test against
// the live Python object, and the assembled body is compared record for record
// with the per-op path over the WP5 fixture set, on raw components, in both
// modes (tests/metal/test_trace_units.py).
//
// The `#ifdef OPTILAND_TRACE_BREAK_<site>` blocks are the round-0 divergence
// injections of plan section 6: U_INKERNEL (the refraction ratio divided in
// the kernel instead of read from the record), HORNER (Horner accumulation of
// the even-asphere sum), COMPOSED_ROTATION (one composed rotation matrix in
// `localize`) and SQR (`df::sqr` instead of `mul(x, x)` in `radial_sq`).  They
// are absent from every production build; `tp.BREAK_SITES` lists the modes
// each one must break and `test_break_hook_changes_the_trace` certifies it.

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

#ifndef OPTILAND_TRACE_METAL
#define OPTILAND_TRACE_METAL

// ---------------------------------------------------------------------------
// R-typed device spans.  df64 carries two float32 planes, sf64 one int64
// plane; everything below is written once against these so the shared body is
// mode-generic and the two entry points differ only in their bindings.
// ---------------------------------------------------------------------------

template <typename R>
struct rspan;  // read-only span of R

template <typename R>
struct wspan;  // writable span of R

template <>
struct rspan<df64> {
    const device float* hi;
    const device float* lo;
    inline df64 get(ulong o) const { return df::make(hi[o], lo[o]); }
};

template <>
struct wspan<df64> {
    device float* hi;
    device float* lo;
    inline void set(ulong o, df64 v) const { hi[o] = v.hi; lo[o] = v.lo; }
};

#ifdef OPTILAND_SF64_CORE_H
template <>
struct rspan<sf64> {
    const device long* bits;
    inline sf64 get(ulong o) const { return sf::make(ulong(bits[o])); }
};

template <>
struct wspan<sf64> {
    device long* bits;
    inline void set(ulong o, sf64 v) const { bits[o] = long(v.bits); }
};
#endif  // OPTILAND_SF64_CORE_H

// ---------------------------------------------------------------------------
// trace_ops<R>: the Real abstraction (design 4.2).
//
// Extends conic_ops<R> so conic_candidates_body<R> and the surface pipeline
// share one set of primitives.  Four rules, each with a test in
// tests/metal/test_trace_units.py:
//   1. no `float` overload is ever called (df::mul(df64, float) and friends):
//      the per-op scalar-variant kernels load their scalar as a full R pair
//      and call the R x R overload (codegen.py:378-381), so this file does
//      too -- every scalar arrives through lit() or an encoded table slot;
//   2. lit(v) only for exactly float32-representable v; everything else
//      (tol, 1e-14, a material index) arrives encoded in surf_real / consts;
//   3. every expression keeps the Python source's association and the operand
//      order the per-op path ACTUALLY launches -- which is not always the
//      order the Python source is written in, because Python's reflected
//      operators decide the kernel variant.  Measured on this checkout
//      (tests/metal/test_trace_units.py::test_scalar_operand_side_matches_
//      the_per_op_kernels), for a tensor `x`:
//        Python float/int literal `c`: `c * x` and `c + x` run as
//          mul/add(x, c) -- `__rmul__`/`__radd__` SWAP; `c - x` runs as
//          sub(c, x); `c / x` runs as recip(x) then mul(recip, c);
//        backend scalar `s` (a 0-d or stride-0 `be.array`, e.g. a material
//          index or a `be.cos` of a pose angle): `s * x`, `s / x`, `s - x`
//          all keep `s` on the LEFT;
//        `c <= x` runs as ge(x, c).
//      df64 multiplication is not commutative (`mul_core` adds `a.hi*b.lo`
//      before `a.lo*b.hi`), so the side is load-bearing, not cosmetic;
//   4. no fused multiply-add (df::mul_add / sf::fma): the mirrored Python
//      expressions are fma-free and one rounding instead of two breaks
//      raw-component equality with the per-op path.
// ---------------------------------------------------------------------------

template <typename R>
struct trace_ops;

// NOTE (deviation from design 4.2, which writes `trace_ops : conic_ops<R>`):
// MSL rejects inheritance -- "derived classes are not supported in Metal" --
// so `trace_ops<R>` restates conic_ops<R>'s members and forwards to the same
// df:: / sf:: functions.  `conic_candidates_body<R>` keeps using conic_ops<R>;
// both structs are thin forwarders to one implementation, so there is still
// exactly one definition of every primitive.
template <>
struct trace_ops<df64> {
    static inline df64 lit(float v) { return df::make(v, 0.0f); }
    static inline df64 abs(df64 a) { return df::abs(a); }
    static inline df64 sqrt(df64 a) { return df::sqrt(a); }
    static inline df64 copysign(df64 a, df64 b) { return df::copysign(a, b); }
    static inline bool is_finite(df64 a) { return df::is_finite(a); }
    static inline bool is_zero(df64 a) { return df::is_zero(a); }
    static inline df64 inf() { return df::inf(); }
    static inline df64 neg(df64 a) { return df::neg(a); }
    static inline df64 sign(df64 a) { return df::sign(a); }
    static inline df64 recip(df64 a) { return df::recip(a); }
    static inline df64 rsqrt(df64 a) { return df::rsqrt(a); }
    static inline df64 maximum(df64 a, df64 b) { return df::maximum(a, b); }
    static inline df64 exp(df64 a) { return df::exp(a); }
    static inline df64 pow(df64 a, df64 b) { return df::pow(a, b); }
    static inline bool is_nan(df64 a) { return df::is_nan(a); }
    static inline df64 nan() { return df::nan(); }
    static inline df64 zero() { return lit(0.0f); }
    static inline df64 one() { return lit(1.0f); }
    static inline df64 two() { return lit(2.0f); }
#ifdef OPTILAND_TRACE_BREAK_SQR
    // Round-0 injection site only (plan 6): the header's dedicated
    // squaring routine.  In df64 it is a DIFFERENT algorithm from
    // mul(a, a) (sqr_core's `fma(2*a.hi, a.lo, a.lo*a.lo)` against
    // mul_core's two chained fma's), in sf64 it is literally
    // mul(a, a), so only df64 can diverge.  The production build does
    // not define this member at all.
    static inline df64 sqr(df64 a) { return df::sqr(a); }
#endif
};

#ifdef OPTILAND_SF64_CORE_H
template <>
struct trace_ops<sf64> {
    static inline sf64 lit(float v) { return sf::from_float(v); }
    static inline sf64 abs(sf64 a) { return sf::abs(a); }
    static inline sf64 sqrt(sf64 a) { return sf::sqrt(a); }
    static inline sf64 copysign(sf64 a, sf64 b) { return sf::copysign(a, b); }
    static inline bool is_finite(sf64 a) { return sf::is_finite(a); }
    static inline bool is_zero(sf64 a) { return sf::is_zero(a); }
    static inline sf64 inf() { return sf::inf(); }
    static inline sf64 neg(sf64 a) { return sf::neg(a); }
    static inline sf64 sign(sf64 a) { return sf::sign(a); }
    static inline sf64 recip(sf64 a) { return sf::recip(a); }
    static inline sf64 rsqrt(sf64 a) { return sf::rsqrt(a); }
    static inline sf64 maximum(sf64 a, sf64 b) { return sf::maximum(a, b); }
    static inline sf64 exp(sf64 a) { return sf::exp(a); }
    static inline sf64 pow(sf64 a, sf64 b) { return sf::pow(a, b); }
    static inline bool is_nan(sf64 a) { return sf::is_nan(a); }
    static inline sf64 nan() { return sf::nan(); }
    static inline sf64 zero() { return lit(0.0f); }
    static inline sf64 one() { return lit(1.0f); }
    static inline sf64 two() { return lit(2.0f); }
#ifdef OPTILAND_TRACE_BREAK_SQR
    // Round-0 injection site only (plan 6): the header's dedicated
    // squaring routine.  In df64 it is a DIFFERENT algorithm from
    // mul(a, a) (sqr_core's `fma(2*a.hi, a.lo, a.lo*a.lo)` against
    // mul_core's two chained fma's), in sf64 it is literally
    // mul(a, a), so only df64 can diverge.  The production build does
    // not define this member at all.
    static inline sf64 sqr(sf64 a) { return sf::sqr(a); }
#endif
};
#endif  // OPTILAND_SF64_CORE_H

// `x ** e` with the fast paths of metal/ops_elementwise.py:740-764
// (`_pow_tensor_scalar`, which is what `aten.pow.Tensor_Scalar` resolves to on
// this backend).  `e` is SIMD-uniform: it comes from a coefficient loop index,
// never from ray data.  The fallback encodes `e` with lit(), which is exact
// for every exponent the trace can produce (asphere orders are small integers)
// and matches the (hi, lo) pair `encode.df64_scalar(e)` hands the per-op
// scalar-variant kernel.
template <typename R>
inline R pow_scalar(R x, float e) {
    typedef trace_ops<R> O;
    if (e == 0.0f) return O::one();
    if (e == 1.0f) return x;
    if (e == 2.0f) return x * x;
    if (e == 3.0f) return (x * x) * x;
    if (e == 0.5f) return O::sqrt(x);
    if (e == -1.0f) return O::recip(x);
    if (e == -2.0f) return O::recip(x * x);
    if (e == -0.5f) return O::rsqrt(x);
    return O::pow(x, O::lit(e));
}

// ---------------------------------------------------------------------------
// Surface-pipeline steps (design 4.5).  Each is an inline function taking and
// returning values, so live ranges stay bounded and no thread-local array is
// ever indexed with a runtime index (design 4.2, register discipline).
// ---------------------------------------------------------------------------

//: The six components a pose transform touches.
template <typename R>
struct ray6 {
    R x, y, z, L, M, N;
};

//: The three direction cosines an interaction produces.
template <typename R>
struct dir3 {
    R L, M, N;
};

//: One slot of a `surf_real[b][s]` row.
template <typename R>
inline R slot(rspan<R> surf_real, ulong row, int k) {
    return surf_real.get(row + ulong(k));
}

// RealRays.rotate_z (real_rays.py:199-211): the four components are assigned
// simultaneously in Python, so all four are computed before any is committed.
template <typename R>
inline void rot_z(thread ray6<R> &r, R c, R s) {
    R xn = r.x * c - r.y * s;
    R yn = r.x * s + r.y * c;
    R Ln = r.L * c - r.M * s;
    R Mn = r.L * s + r.M * c;
    r.x = xn;
    r.y = yn;
    r.L = Ln;
    r.M = Mn;
}

// RealRays.rotate_y (real_rays.py:184-196).  Python's `-self.x * be.sin(ry)`
// binds as `(-self.x) * be.sin(ry)`: a negation, then the product.
template <typename R>
inline void rot_y(thread ray6<R> &r, R c, R s) {
    typedef trace_ops<R> O;
    R xn = r.x * c + r.z * s;
    R zn = O::neg(r.x) * s + r.z * c;
    R Ln = r.L * c + r.N * s;
    R Nn = O::neg(r.L) * s + r.N * c;
    r.x = xn;
    r.z = zn;
    r.L = Ln;
    r.N = Nn;
}

// RealRays.rotate_x (real_rays.py:169-181).
template <typename R>
inline void rot_x(thread ray6<R> &r, R c, R s) {
    R yn = r.y * c - r.z * s;
    R zn = r.y * s + r.z * c;
    R Mn = r.M * c - r.N * s;
    R Nn = r.M * s + r.N * c;
    r.y = yn;
    r.z = zn;
    r.M = Mn;
    r.N = Nn;
}

// CoordinateSystem.localize (coordinate_system.py:127-143): translate by
// (-x, -y, -z) -- the negation is a host op, so the row carries it in NT* --
// then rotate by -rz, -ry, -rx, each skipped when the angle is falsy (the
// `if self.rz:` truth tests, mirrored by the flag bits).
template <typename R>
inline void localize(thread ray6<R> &r, rspan<R> P, ulong row, int flags) {
    r.x = r.x + slot<R>(P, row, OT_SR_NTX);
    r.y = r.y + slot<R>(P, row, OT_SR_NTY);
    r.z = r.z + slot<R>(P, row, OT_SR_NTZ);
#ifdef OPTILAND_TRACE_BREAK_COMPOSED_ROTATION
    // Round-0 injection (plan 6): one composed rotation matrix
    // Rx(-rx) Ry(-ry) Rz(-rz) applied in a single pass instead of three
    // sequential two-term rotations.  Note 03 measured this to be bit-exact
    // for a pose with at most ONE non-zero angle (the extra terms are exact
    // multiplications by 1 and additions of 0), so only a three-angle pose
    // certifies it.  An absent angle contributes (c, s) = (1, 0) regardless of
    // what the record row carries in its unused slots.
    typedef trace_ops<R> O;
    const R cz = (flags & OT_FL_HAS_RZ) ? slot<R>(P, row, OT_SR_CNRZ) : O::one();
    const R sz = (flags & OT_FL_HAS_RZ) ? slot<R>(P, row, OT_SR_SNRZ) : O::zero();
    const R cy = (flags & OT_FL_HAS_RY) ? slot<R>(P, row, OT_SR_CNRY) : O::one();
    const R sy = (flags & OT_FL_HAS_RY) ? slot<R>(P, row, OT_SR_SNRY) : O::zero();
    const R cx = (flags & OT_FL_HAS_RX) ? slot<R>(P, row, OT_SR_CNRX) : O::one();
    const R sx = (flags & OT_FL_HAS_RX) ? slot<R>(P, row, OT_SR_SNRX) : O::zero();
    const R m00 = cy * cz;
    const R m01 = O::neg(cy) * sz;
    const R m02 = sy;
    const R m10 = cx * sz + (sx * sy) * cz;
    const R m11 = cx * cz - (sx * sy) * sz;
    const R m12 = O::neg(sx) * cy;
    const R m20 = sx * sz - (cx * sy) * cz;
    const R m21 = sx * cz + (cx * sy) * sz;
    const R m22 = cx * cy;
    const R px = (m00 * r.x + m01 * r.y) + m02 * r.z;
    const R py = (m10 * r.x + m11 * r.y) + m12 * r.z;
    const R pz = (m20 * r.x + m21 * r.y) + m22 * r.z;
    const R dl = (m00 * r.L + m01 * r.M) + m02 * r.N;
    const R dm = (m10 * r.L + m11 * r.M) + m12 * r.N;
    const R dn = (m20 * r.L + m21 * r.M) + m22 * r.N;
    r.x = px;
    r.y = py;
    r.z = pz;
    r.L = dl;
    r.M = dm;
    r.N = dn;
    return;
#endif
    if (flags & OT_FL_HAS_RZ) {
        rot_z<R>(r, slot<R>(P, row, OT_SR_CNRZ), slot<R>(P, row, OT_SR_SNRZ));
    }
    if (flags & OT_FL_HAS_RY) {
        rot_y<R>(r, slot<R>(P, row, OT_SR_CNRY), slot<R>(P, row, OT_SR_SNRY));
    }
    if (flags & OT_FL_HAS_RX) {
        rot_x<R>(r, slot<R>(P, row, OT_SR_CNRX), slot<R>(P, row, OT_SR_SNRX));
    }
}

// CoordinateSystem.globalize (coordinate_system.py:145-161): the inverse
// order, +rx, +ry, +rz from their own trig slots (no cos(-a) == cos(a)
// assumption), then translate by (+x, +y, +z).
template <typename R>
inline void globalize(thread ray6<R> &r, rspan<R> P, ulong row, int flags) {
    if (flags & OT_FL_HAS_RX) {
        rot_x<R>(r, slot<R>(P, row, OT_SR_CRX), slot<R>(P, row, OT_SR_SRX));
    }
    if (flags & OT_FL_HAS_RY) {
        rot_y<R>(r, slot<R>(P, row, OT_SR_CRY), slot<R>(P, row, OT_SR_SRY));
    }
    if (flags & OT_FL_HAS_RZ) {
        rot_z<R>(r, slot<R>(P, row, OT_SR_CRZ), slot<R>(P, row, OT_SR_SRZ));
    }
    r.x = r.x + slot<R>(P, row, OT_SR_TX);
    r.y = r.y + slot<R>(P, row, OT_SR_TY);
    r.z = r.z + slot<R>(P, row, OT_SR_TZ);
}

// HomogeneousPropagation.propagate, straight-line part (homogeneous.py:40-43).
template <typename R>
inline ray6<R> advance(ray6<R> r, R t) {
    r.x = r.x + t * r.L;
    r.y = r.y + t * r.M;
    r.z = r.z + t * r.N;
    return r;
}

// HomogeneousPropagation.propagate, absorption (homogeneous.py:46-53).
// Python: `alpha = 4 * be.pi * k / rays.w` -- `(4 * pi) * k` is a host scalar
// product (slot ALPHA) and `/ rays.w` the first GPU op -- then
// `rays.i * be.exp(-alpha * t * 1e3)`, i.e. negate, `* t`, `* 1e3`, exp,
// multiply.  Five GPU ops in that order; 1e3 is float32-exact.
template <typename R>
inline R absorb(R inten, R t, R w, R alpha) {
    typedef trace_ops<R> O;
    R a = alpha / w;
    a = O::neg(a);
    a = a * t;
    a = a * O::lit(1e3f);
    return inten * O::exp(a);
}

// Surface._trace_real (standard_surface.py:302-304): the PRE-interaction index
// and the same signed t as the position update, accumulated on the launch
// value, before the clip and the interaction.
template <typename R>
inline R accumulate_opd(R opd, R t, R n_pre) {
    return opd + t * n_pre;
}

// Aperture `contains` in the LOCAL frame (design 4.9), inclusive on both
// bounds and NaN-false on both operators, so a missed (NaN) ray clips exactly
// as `contains(NaN, NaN)` does in Python.  Squares of the parameters are
// computed on the host (radial.py:70) and arrive in AP0/AP1.
template <typename R>
inline bool ap_contains(int code, R p0, R p1, R p2, R p3, R x, R y) {
    typedef trace_ops<R> O;
    switch (code) {
        case OT_AP_RADIAL: {
            // radial.py:69-71
            R r2 = x * x + y * y;
            return (r2 <= p0) && (r2 >= p1);
        }
        case OT_AP_OFFSET_RADIAL: {
            // offset_radial.py:59-60
            R dx = x - p2;
            R dy = y - p3;
            R r2 = dx * dx + dy * dy;
            return (r2 <= p0) && (r2 >= p1);
        }
        case OT_AP_RECT:
            // rectangular.py:54-59 -- `self.x_min <= x` is reflected to
            // `x.__ge__(x_min)`, so the kernel runs ge(x, p0), not le(p0, x).
            return (x >= p0) && (x <= p1) && (y >= p2) && (y <= p3);
        case OT_AP_ELLIPSE: {
            // elliptical.py:59-61 -- division by a**2 / b**2, not a reciprocal
            R dx = x - p2;
            R dy = y - p3;
            return ((dx * dx) / p0 + (dy * dy) / p1) <= O::one();
        }
        default:
            return true;
    }
}

// The interaction step (design 4.8): RealRays._align_surface_normal
// (real_rays.py:561-597) followed by reflect (:213-231) or refract
// (:189-211).  `sign(0) == 0` at exact grazing zeroes the aligned normal and
// `dot`, so refraction returns the unnormalised `u * L0` instead of NaN --
// reproduced, not fixed.  TIR is a NaN `root`, hence NaN L/M/N, and leaves the
// intensity untouched (there is no TIR branch in `refract`).
template <typename R>
inline dir3<R> interact(R L0, R M0, R N0, R nx, R ny, R nz, bool reflective, R u,
                        R u2, thread uchar &st) {
    typedef trace_ops<R> O;
    R dot = (L0 * nx + M0 * ny) + N0 * nz;
    R sgn = O::sign(dot);
    nx = nx * sgn;
    ny = ny * sgn;
    nz = nz * sgn;
    dot = O::abs(dot);
    dir3<R> out;
    if (reflective) {
        // real_rays.py:229 `self.L - 2 * dot * nx`: the literal 2 reaches the
        // kernel through `__rmul__`, i.e. as mul(dot, 2), not mul(2, dot).
        R two_dot = dot * O::two();
        out.L = L0 - two_dot * nx;
        out.M = M0 - two_dot * ny;
        out.N = N0 - two_dot * nz;
    } else {
        R root = O::sqrt(O::one() - u2 * (O::one() - dot * dot));
        if (O::is_nan(root) && O::is_finite(dot)) {
            st |= OT_ST_TIR;
        }
        out.L = (u * L0 + nx * root) - (u * nx) * dot;
        out.M = (u * M0 + ny * root) - (u * ny) * dot;
        out.N = (u * N0 + nz * root) - (u * nz) * dot;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Sag and normal (design 4.7; note 02).  One switch over SI_GEOM.  Every
// expression keeps the Python source's association AND the operand side the
// per-op path actually launches (design 4.2 rule 3), which for these three
// files means:
//   * `self.radius`, `self.k`, `(1 + self.k)` and `self.radius**2` are BACKEND
//     scalars (0-d `be.array`s built in StandardGeometry.__init__,
//     standard.py:117-118, plus two host-scalar ops), so `self.radius * s` and
//     `(1 + self.k) * r2` keep the SLOT on the left while `/ self.radius**2`
//     puts it on the right;
//   * a Python literal swaps to the right -- `1 + be.sqrt(..)` launches
//     add(sqrt, 1) and `2 * (i + 1) * x` launches mul(x, 2(i+1)) -- except
//     `1 - u`, which stays sub(1, u), and `-1 / mag`, which is recip(mag)
//     followed by mul(recip, -1);
//   * an aspheric coefficient is a plain Python float in the geometry's list
//     (even_asphere.py:75), so `Ci * r2**(i+1)` launches mul(r2**(i+1), Ci):
//     the coefficient is on the RIGHT and comes from the `coef` buffer, never
//     from lit() (it is not float32-exact).
// ---------------------------------------------------------------------------

//: A small exact integer taken from a coefficient-loop index (`2*(i+1)` in the
//  even normal, `i+1` in the odd one).  With pow_scalar's exponent these are
//  the only two runtime lit() conversions in the file; both are SIMD-uniform
//  and both are float32-exact for every order a record row can carry
//  (SI_NCOEFF is bounded by the width of the `coef` table).
template <typename R>
inline R index_const(int n) {
    const float v = float(n);
    return trace_ops<R>::lit(v);
}

//: `x**2 + y**2`; each square is the pow fast path mul(x, x)
//  (ops_elementwise.py:752).
template <typename R>
inline R radial_sq(R x, R y) {
#ifdef OPTILAND_TRACE_BREAK_SQR
    // Round-0 injection (plan 6): `df::sqr` instead of `mul(x, x)`.
    typedef trace_ops<R> O;
    return O::sqr(x) + O::sqr(y);
#else
    return x * x + y * y;
#endif
}

//: `1 - (1 + k) * r2 / R**2` -- slot K1 on the left, slot R2 on the right and
//  the literal 1 on the left (standard.py:163, 195).
template <typename R>
inline R conic_radicand(R r2, rspan<R> P, ulong row) {
    typedef trace_ops<R> O;
    return O::one() - (slot<R>(P, row, OT_SR_K1) * r2) / slot<R>(P, row, OT_SR_R2);
}

//: The base-conic sag `r2 / (R * (1 + sqrt(1 - (1+k) r2 / R^2)))`
//  (standard.py:161-164; even_asphere.py:103; odd_asphere.py:96).
template <typename R>
inline R conic_sag(R r2, rspan<R> P, ulong row) {
    typedef trace_ops<R> O;
    R s = O::sqrt(conic_radicand<R>(r2, P, row));
    return r2 / (slot<R>(P, row, OT_SR_R) * (s + O::one()));
}

//: `geometry.sag(x, y)` in the LOCAL frame for one surface row.
template <typename R>
inline R sag_of(int geom, R x, R y, rspan<R> P, ulong row, rspan<R> C, ulong cbase,
                int ncoef) {
    typedef trace_ops<R> O;
    if (geom == OT_GEOM_PLANE) {
        return O::zero();  // plane.py:66 -- be.zeros_like(y)
    }
    R r2 = radial_sq<R>(x, y);
    if (geom == OT_GEOM_ODD) {
        // odd_asphere.py:94-99: r = sqrt(r2) first, then the base conic, then
        // `z + Ci * r**(i+1)` with the coefficient on the right.
        R r = O::sqrt(r2);
        R z = conic_sag<R>(r2, P, row);
        for (int i = 0; i < ncoef; ++i) {
            z = z + pow_scalar<R>(r, float(i + 1)) * C.get(cbase + ulong(i));
        }
        return z;
    }
    R z = conic_sag<R>(r2, P, row);
    if (geom == OT_GEOM_EVEN) {
#ifdef OPTILAND_TRACE_BREAK_HORNER
        // Round-0 injection (plan 6): Horner instead of the Python sum
        // `z + r2**(i+1) * C[i]`, term by term.  With one coefficient it is
        // the same terms in the other operand order (df64 mul is not
        // commutative); with two or more the ASSOCIATION differs too, so both
        // modes must fail.
        if (ncoef > 0) {
            R acc = C.get(cbase + ulong(ncoef - 1));
            for (int i = ncoef - 2; i >= 0; --i) {
                acc = acc * r2 + C.get(cbase + ulong(i));
            }
            z = z + acc * r2;
        }
#else
        // even_asphere.py:105-106
        for (int i = 0; i < ncoef; ++i) {
            z = z + pow_scalar<R>(r2, float(i + 1)) * C.get(cbase + ulong(i));
        }
#endif
    }
    return z;
}

//: The normalised surface normal at a LOCAL (x, y) ON the surface.
//  `PLANE` gives nz = +1 (plane.py:103-105) while every StandardGeometry-derived
//  normal gives nz < 0 (standard.py:213-215); the per-ray alignment in
//  `interact` makes both work, and the conformance test checks the raw sign.
template <typename R>
inline void normal_of(int geom, R x, R y, rspan<R> P, ulong row, rspan<R> C,
                      ulong cbase, int ncoef, thread R &nx, thread R &ny,
                      thread R &nz) {
    typedef trace_ops<R> O;
    if (geom == OT_GEOM_PLANE) {
        nx = O::zero();
        ny = O::zero();
        nz = O::one();
        return;
    }
    R r2 = radial_sq<R>(x, y);
    R denom = slot<R>(P, row, OT_SR_R) * O::sqrt(conic_radicand<R>(r2, P, row));
    R dfdx = x / denom;
    R dfdy = y / denom;
    if (geom == OT_GEOM_EVEN) {
        // even_asphere.py:128-130: `dfdx + 2 * (i + 1) * x * Ci * r2**i`,
        // left-associated, the integer and the coefficient both on the right.
        for (int i = 0; i < ncoef; ++i) {
            R ci = C.get(cbase + ulong(i));
            R p = pow_scalar<R>(r2, float(i));
            R m = index_const<R>(2 * (i + 1));
            dfdx = dfdx + ((x * m) * ci) * p;
            dfdy = dfdy + ((y * m) * ci) * p;
        }
    } else if (geom == OT_GEOM_ODD) {
        // odd_asphere.py:121-131: `r**(i-1)` is +-inf at r = 0 for i = 0, so
        // each term is scrubbed where it is not finite, exactly as
        // `x_term[~be.isfinite(x_term)] = 0` does -- which is what makes the
        // vertex normal exactly (0, 0, -1).
        R r = O::sqrt(r2);
        for (int i = 0; i < ncoef; ++i) {
            R ci = C.get(cbase + ulong(i));
            R p = pow_scalar<R>(r, float(i - 1));
            R m = index_const<R>(i + 1);
            R xt = ((x * m) * ci) * p;
            R yt = ((y * m) * ci) * p;
            dfdx = dfdx + (O::is_finite(xt) ? xt : O::zero());
            dfdy = dfdy + (O::is_finite(yt) ? yt : O::zero());
        }
    }
    // `be.sqrt(dfdx**2 + dfdy**2 + 1)`: the trailing 1 is a Python int (in
    // StandardGeometry it is `dfdz**2` with dfdz = -1, the same int), so it
    // reaches the kernel as add(sum, 1).
    R mag = O::sqrt((dfdx * dfdx + dfdy * dfdy) + O::one());
    nx = dfdx / mag;
    ny = dfdy / mag;
    nz = O::recip(mag) * O::lit(-1.0f);
}

// ---------------------------------------------------------------------------
// Distance (design 4.6).
// ---------------------------------------------------------------------------

//: `Plane.distance` (plane.py:84-86): a bare `-z / N`, with NO floor.  A ray
//  with N == 0 therefore gets NaN here and t = 0 on a GEOM_STD_INF row; the
//  difference is real (WP5 finding 1) and is why the two codes never share a
//  branch.
template <typename R>
inline R plane_distance(R z, R Nd) {
    typedef trace_ops<R> O;
    return O::neg(z) / Nd;
}

//: The infinite-radius branch of `_conic_intersection_distance`
//  (standard.py:88-90): `N_safe = where(abs(N) > 1e-14, N, 1e-14)` -- the
//  floor is POSITIVE for both signs of N -- then `-z / N_safe`.  1e-14 is not
//  float32-exact, so it arrives in consts[C_NFLOOR].
template <typename R>
inline R std_inf_distance(R z, R Nd, R nfloor, thread uchar &st) {
    typedef trace_ops<R> O;
    R ns = (O::abs(Nd) > nfloor) ? Nd : nfloor;
    if (ns != Nd) {
        st |= OT_ST_NZ_FLOORED;
    }
    return O::neg(z) / ns;
}

//: `_select_distance` (backend/_conic.py:113-126) on top of
//  `conic_candidates_body<R>` (conic.metal, reused unchanged).  The aperture
//  only REORDERS the two roots' preference -- it never turns a hit into a miss
//  -- and `contains` runs on the candidate hit points in the LOCAL frame.
template <typename R>
inline R select_distance(R x, R y, R z, R L, R M, R Nd, rspan<R> P, ulong row,
                         R eps, int flags, int apcode) {
    typedef trace_ops<R> O;
    R t1, t2;
    const uchar fl = conic_candidates_body<R>(x, y, z, L, M, Nd,
                                              slot<R>(P, row, OT_SR_R),
                                              slot<R>(P, row, OT_SR_K), eps, t1, t2);
    const bool valid1 = (fl & 1) != 0;
    const bool valid2 = (fl & 2) != 0;
    bool pick2 = (fl & 4) != 0;
    const bool solvable = (fl & 8) != 0;
    if (flags & OT_FL_AP_IN_ROOT) {
        const R p0 = slot<R>(P, row, OT_SR_AP0);
        const R p1 = slot<R>(P, row, OT_SR_AP1);
        const R p2 = slot<R>(P, row, OT_SR_AP2);
        const R p3 = slot<R>(P, row, OT_SR_AP3);
        const bool pref1 =
            valid1 && ap_contains<R>(apcode, p0, p1, p2, p3, x + t1 * L, y + t1 * M);
        const bool pref2 =
            valid2 && ap_contains<R>(apcode, p0, p1, p2, p3, x + t2 * L, y + t2 * M);
        pick2 = pref2 || (pref1 ? false : pick2);
    }
    return solvable ? (pick2 ? t2 : t1) : O::nan();
}

//: `NewtonRaphsonGeometry._solve_distance_primal` (newton_raphson.py:317-375)
//  for ONE thread.
//
//  Two Python-side batch semantics become per-thread rules here:
//
//  1. `_effective_tolerance` (:124-138) takes `scale = max|t|` over the WHOLE
//     batch; a thread only knows its own |t|.  The two agree unless some ray
//     satisfies `8*eps*max(1,|t|) > tol`, which is exactly the per-thread
//     TOL_CROSSOVER condition, and the batch maximum is attained by SOME
//     thread -- so `any(TOL_CROSSOVER)` holds iff Python's floor beats `tol`.
//     The driver refuses the launch when the bit is set anywhere (design 4.6),
//     so an accepted fused result is never affected.
//  2. The batch `if be.all(converged): break` (:355) together with the
//     `be.where(converged, 0, step)` freeze (:363) is, per thread, "stop as
//     soon as I converge": a frozen thread's t, residual and convergence flag
//     can never change again.  `iters` is therefore this thread's own count,
//     not the batch's single integer (design 4.12; WP5 finding 4).
//
//  A non-finite t (a NaN seed, i.e. a conic miss) also stops the thread: in
//  Python it would keep stepping NaN into NaN for the rest of the batch loop,
//  with the same final t and the same `converged = False`.
template <typename R>
inline R newton_distance(R x, R y, R z, R L, R M, R Nd, rspan<R> P, ulong row,
                         rspan<R> C, ulong cbase, int geom, int ncoef, int max_iter,
                         int flags, int apcode, R eps, R nfloor, thread uchar &st,
                         thread uchar &it) {
    typedef trace_ops<R> O;
    // Seed: the base conic through StandardGeometry.distance, aperture and all
    // (newton_raphson.py:347), so the iteration starts in the physical basin.
    R t = (flags & OT_FL_RADIUS_INF)
              ? std_inf_distance<R>(z, Nd, nfloor, st)
              : select_distance<R>(x, y, z, L, M, Nd, P, row, eps, flags, apcode);

    // tol = max(self.tol, 8 * eps * max(1, |t|)).  Both products scale by a
    // power of two, so the df64/sf64 value equals Python's float64 one.
    const R atol = O::abs(t);
    const R floor_tol = (O::lit(8.0f) * eps) * ((atol > O::one()) ? atol : O::one());
    const R tol_user = slot<R>(P, row, OT_SR_TOL);
    const bool crossed = floor_tol > tol_user;
    if (crossed) {
        st |= OT_ST_TOL_CROSSOVER;
    }
    const R tol = crossed ? floor_tol : tol_user;

    // F(t) = sag(x + tL, y + tM) - (z + tN)   (newton_raphson.py:279-284)
    R f_t = sag_of<R>(geom, x + t * L, y + t * M, P, row, C, cbase, ncoef)
            - (z + t * Nd);
    bool conv = O::abs(f_t) < tol;

    const R tau_nz = O::lit(32.0f) * eps;  // _nz_threshold (:59-70)
    const R tau_df = O::lit(32.0f) * eps;  // _denominator_threshold (:88-100)
    int i = 0;
    for (; i < max_iter; ++i) {
        if (conv || !O::is_finite(t)) {
            break;
        }
        // dF/dt from the NORMALISED normal (:302-313), not from sag.
        R nx, ny, nz;
        normal_of<R>(geom, x + t * L, y + t * M, P, row, C, cbase, ncoef, nx, ny, nz);
        // _sign_preserving_floor(nz, tau_nz) (:75-90): sign-preserving, and
        // NaN takes the negative branch (`NaN >= 0` is False).
        const R nzs =
            (O::abs(nz) > tau_nz) ? nz : ((nz >= O::zero()) ? tau_nz : O::neg(tau_nz));
        if (nzs != nz) {
            st |= OT_ST_NZ_FLOORED;
        }
        const R fx = O::neg(nx) / nzs;
        const R fy = O::neg(ny) / nzs;
        const R fxL = fx * L;
        const R fyM = fy * M;
        const R df = (fxL + fyM) - Nd;
        const R scale = (O::abs(fxL) + O::abs(fyM)) + O::abs(Nd);
        // _regularize_signed (:103-118): `tau * be.maximum(scale, ones)` with
        // the host float on the RIGHT, and both branches multiplied by
        // `be.ones_like(value)` -- mirrored, not elided.
        const R tau = O::maximum(scale, O::one()) * tau_df;
        const bool near = O::abs(df) <= tau;
        if (near) {
            st |= OT_ST_DF_FLOORED;
        }
        const R tau_pos = tau * O::one();
        const R tau_neg = O::neg(tau) * O::one();
        const R safe = near ? ((df >= O::zero()) ? tau_pos : tau_neg) : df;
        t = t - f_t / safe;
        f_t = sag_of<R>(geom, x + t * L, y + t * M, P, row, C, cbase, ncoef)
              - (z + t * Nd);
        conv = O::abs(f_t) < tol;
    }
    it = uchar(i);
    // newton_raphson.py:580-581 keeps the last t with no NaN; the status bit
    // and `iters` are what make a non-converged ray visible.
    if (!conv && O::is_finite(t)) {
        st |= OT_ST_NEWTON_NOT_CONVERGED;
    }
    return t;
}

//: The distance step of design 4.5: the SI_GEOM switch, then the MISS bit.
//  Rows s >= 1 carry only the five surface codes; GEOM_OBJECT never reaches
//  the distance step.
template <typename R>
inline R surface_distance(int geom, R x, R y, R z, R L, R M, R Nd, rspan<R> P,
                          ulong row, rspan<R> C, ulong cbase, int flags, int apcode,
                          int ncoef, int max_iter, R eps, R nfloor,
                          thread uchar &st, thread uchar &it) {
    typedef trace_ops<R> O;
    R t;
    if (geom == OT_GEOM_PLANE) {
        t = plane_distance<R>(z, Nd);
    } else if (geom == OT_GEOM_STD_INF) {
        t = std_inf_distance<R>(z, Nd, nfloor, st);
    } else if (geom == OT_GEOM_CONIC) {
        t = select_distance<R>(x, y, z, L, M, Nd, P, row, eps, flags, apcode);
    } else {
        t = newton_distance<R>(x, y, z, L, M, Nd, P, row, C, cbase, geom, ncoef,
                               max_iter, flags, apcode, eps, nfloor, st, it);
    }
    if (O::is_nan(t)) {
        st |= OT_ST_MISS;
    }
    return t;
}

// ---------------------------------------------------------------------------
// Shared per-thread body.  R is df64 or sf64.
// ---------------------------------------------------------------------------

// The Python order of design 4.5, one surface row per iteration:
//
//   ObjectSurface.trace  ->  reset + record only (no physics)
//   Surface.trace        ->  _TracingCoordinator.trace (standard_surface.py:82-101)
//                            geometry.localize
//                            rays.trace_on_surface -> Surface._trace_real (:299-308)
//                              t = _aperture_aware_distance(surface, rays)
//                              material_pre.propagation_model.propagate(rays, t)
//                              rays.opd = rays.opd + t * material_pre.n(rays.w)
//                              if surface.aperture: aperture.clip(rays)
//                              interaction_model.interact_real_rays(rays)
//                            geometry.globalize
//                            if record: rays.record_on_surface(surface)
//
// so the recorded snapshot is in the GLOBAL frame, carries the POST-interaction
// direction and the POST-clip intensity, and the OPD is accumulated with the
// PRE-interaction index and the signed t BEFORE the clip and the interaction.
//
// `L0/M0/N0` are written by `refract`/`reflect` (real_rays.py:197-199, 218-220)
// in the LOCAL frame of the surface that wrote them last and are never rotated
// back by `globalize` (it only touches x, y, L, M / y, z, M, N), so the final
// planes carry the last surface's local pre-interaction direction.  A trace
// with no surface row leaves them at the launch direction.
template <typename R>
inline void trace_body(rspan<R> launch,
                       const device int* surf_int,
                       rspan<R> surf_real,
                       rspan<R> coef,
                       const device int* dims,
                       rspan<R> consts,
                       wspan<R> snap,
                       wspan<R> fin,
                       device uchar* status,
                       device uchar* iters,
                       uint2 g) {
    typedef trace_ops<R> O;
    const int S = dims[OT_D_S];
    const int N = dims[OT_D_N];
    const int B = dims[OT_D_B];
    const int launch_stride = dims[OT_D_LAUNCH_STRIDE];
    const int n_rows = dims[OT_D_NROWS];
    const int C = dims[OT_D_C];
    const int P = dims[OT_D_P];
    const int write_final = dims[OT_D_WRITE_FINAL];

    const int i = dims[OT_D_RAY_BASE] + int(g.x);
    const int b = dims[OT_D_DESIGN_BASE] + int(g.y);
    if (i >= N || b >= B || S <= 0) {
        return;
    }

    const ulong ui = ulong(i);
    const ulong ub = ulong(b);
    const ulong uN = ulong(N);
    const ulong uS = ulong(S);
    const ulong uB = ulong(B);
    const ulong uR = ulong(n_rows);
    const ulong uP = ulong(P);
    const ulong uC = ulong(C);
    const ulong Lb = (launch_stride == 0) ? 1UL : uB;
    const ulong qrow = ub * ulong(launch_stride) * uN;
    const ulong qbase = qrow + ui;
    const ulong qstride = Lb * uN;
    const ulong rstride = uB * uR * uN;

    // ---- state registers, in the Python order (design 4.5)
    R x = launch.get(ulong(OT_Q_X) * qstride + qbase);
    R y = launch.get(ulong(OT_Q_Y) * qstride + qbase);
    R z = launch.get(ulong(OT_Q_Z) * qstride + qbase);
    R L = launch.get(ulong(OT_Q_L) * qstride + qbase);
    R M = launch.get(ulong(OT_Q_M) * qstride + qbase);
    R Nd = launch.get(ulong(OT_Q_N) * qstride + qbase);
    R ii = launch.get(ulong(OT_Q_I) * qstride + qbase);
    R opd = launch.get(ulong(OT_Q_OPD) * qstride + qbase);
    const R w = launch.get(ulong(OT_Q_W) * qstride + qbase);
    R L0 = L, M0 = M, N0 = Nd;

    // Belt-and-braces uniformity check (design 4.5).  The gate refuses a
    // mixed-wavelength bundle before the launch (`mixed_wavelength`), so this
    // bit can only appear when the kernel is driven around the gate; it is
    // OR-ed into every surface row's status so a driver that skipped the
    // readback still sees it.  Ray 0 of this design's launch row is the
    // reference, so the comparison is SIMD-uniform.
    uchar st0 = 0;
    if (w != launch.get(ulong(OT_Q_W) * qstride + qrow)) {
        st0 |= OT_ST_NONUNIFORM_W;
    }

    const R eps = consts.get(ulong(OT_C_EPS));
    const R nfloor = consts.get(ulong(OT_C_NFLOOR));

    // ---- object row (s = 0): ObjectSurface.trace is reset + record only
    // (object_surface.py:66-83; `_trace_real` returns the rays unchanged), so
    // the row carries the launch state.  Plan 3.2 makes the kernel responsible
    // for writing status = 0 and iters = 0 here, so that a surviving
    // ITERS_UNWRITTEN anywhere means an unvisited thread.
    {
        const int row = surf_int[(ub * uS) * ulong(OT_SI_STRIDE) + OT_SI_SNAPROW];
        if (row >= 0 && row < n_rows) {
            const ulong rbase = (ub * uR + ulong(row)) * uN + ui;
            snap.set(ulong(OT_S_X) * rstride + rbase, x);
            snap.set(ulong(OT_S_Y) * rstride + rbase, y);
            snap.set(ulong(OT_S_Z) * rstride + rbase, z);
            snap.set(ulong(OT_S_L) * rstride + rbase, L);
            snap.set(ulong(OT_S_M) * rstride + rbase, M);
            snap.set(ulong(OT_S_N) * rstride + rbase, Nd);
            snap.set(ulong(OT_S_I) * rstride + rbase, ii);
            snap.set(ulong(OT_S_OPD) * rstride + rbase, opd);
        }
        const ulong so = (ub * uS) * uN + ui;
        status[so] = 0;
        iters[so] = 0;
    }

    // ---- walk the surfaces
    for (int s = 1; s < S; ++s) {
        const ulong si = ub * uS + ulong(s);
        const ulong tbl = si * ulong(OT_SI_STRIDE);
        const int geom = surf_int[tbl + OT_SI_GEOM];
        const int flags = surf_int[tbl + OT_SI_FLAGS];
        const int ncoef = surf_int[tbl + OT_SI_NCOEFF];
        const int max_iter = surf_int[tbl + OT_SI_MAXITER];
        const int apcode = surf_int[tbl + OT_SI_APCODE];
        const int row = surf_int[tbl + OT_SI_SNAPROW];
        const ulong prow = si * uP;
        const ulong cbase = si * uC;
        uchar st = 0;
        uchar it = 0;

        // ---- localize (coordinate_system.py:127-143)
        ray6<R> r = {x, y, z, L, M, Nd};
        localize<R>(r, surf_real, prow, flags);

        // ---- distance (_aperture_aware_distance -> geometry.distance)
        const R t =
            surface_distance<R>(geom, r.x, r.y, r.z, r.L, r.M, r.N, surf_real, prow,
                                coef, cbase, flags, apcode, ncoef, max_iter, eps,
                                nfloor, st, it);

        // ---- propagate (homogeneous.py:40-53)
        r = advance<R>(r, t);
        if (flags & OT_FL_ABSORBING) {
            ii = absorb<R>(ii, t, w, slot<R>(surf_real, prow, OT_SR_ALPHA));
        }

        // ---- OPD (standard_surface.py:304): PRE index, signed t, BEFORE the
        // clip and the interaction
        opd = accumulate_opd<R>(opd, t, slot<R>(surf_real, prow, OT_SR_NPRE));

        // ---- clip (base.py:71-79 + real_rays.py:180-187): LOCAL x, y, and
        // only the intensity moves -- `rays.clip(~inside)` is
        // `be.where(~inside, zeros_like(i), i)`.
        if (flags & OT_FL_HAS_APERTURE) {
            const bool inside = ap_contains<R>(
                apcode, slot<R>(surf_real, prow, OT_SR_AP0),
                slot<R>(surf_real, prow, OT_SR_AP1),
                slot<R>(surf_real, prow, OT_SR_AP2),
                slot<R>(surf_real, prow, OT_SR_AP3), r.x, r.y);
            if (!inside) {
                st |= OT_ST_CLIPPED;
                ii = O::zero();
            }
        }

        // ---- interact (refractive_reflective_model.py:41-49): the normal is
        // taken at the LOCAL intersection point, then refract or reflect.
        R nx, ny, nz;
        normal_of<R>(geom, r.x, r.y, surf_real, prow, coef, cbase, ncoef, nx, ny, nz);
        L0 = r.L;
        M0 = r.M;
        N0 = r.N;
        R u = slot<R>(surf_real, prow, OT_SR_U);
        R u2 = slot<R>(surf_real, prow, OT_SR_U2);
#ifdef OPTILAND_TRACE_BREAK_U_INKERNEL
        // Round-0 injection (plan 6): recompute `u = n1 / n2` and `u**2` in the
        // kernel instead of reading the host float64 values out of the record.
        // df64 division carries ~48 bits, so the df64 comparison must fail
        // while sf64 (correctly rounded) may not.
        u = slot<R>(surf_real, prow, OT_SR_NPRE) / slot<R>(surf_real, prow, OT_SR_NPOST);
        u2 = u * u;
#endif
        const dir3<R> d = interact<R>(L0, M0, N0, nx, ny, nz,
                                      (flags & OT_FL_REFLECTIVE) != 0, u, u2, st);
        r.L = d.L;
        r.M = d.M;
        r.N = d.N;

        // ---- globalize (coordinate_system.py:145-161)
        globalize<R>(r, surf_real, prow, flags);

        // ---- record (standard_surface.py:320-334), GLOBAL frame
        if (row >= 0 && row < n_rows) {
            const ulong rbase = (ub * uR + ulong(row)) * uN + ui;
            snap.set(ulong(OT_S_X) * rstride + rbase, r.x);
            snap.set(ulong(OT_S_Y) * rstride + rbase, r.y);
            snap.set(ulong(OT_S_Z) * rstride + rbase, r.z);
            snap.set(ulong(OT_S_L) * rstride + rbase, r.L);
            snap.set(ulong(OT_S_M) * rstride + rbase, r.M);
            snap.set(ulong(OT_S_N) * rstride + rbase, r.N);
            snap.set(ulong(OT_S_I) * rstride + rbase, ii);
            snap.set(ulong(OT_S_OPD) * rstride + rbase, opd);
        }
        const ulong so = si * uN + ui;
        status[so] = st | st0;
        iters[so] = it;

        x = r.x;
        y = r.y;
        z = r.z;
        L = r.L;
        M = r.M;
        Nd = r.N;
    }

    // ---- final plane set: the post-trace state plus the last surface's
    // pre-interaction (local-frame) direction.
    if (write_final != 0) {
        const ulong fbase = ub * uN + ui;
        const ulong fstride = uB * uN;
        fin.set(ulong(OT_F_X) * fstride + fbase, x);
        fin.set(ulong(OT_F_Y) * fstride + fbase, y);
        fin.set(ulong(OT_F_Z) * fstride + fbase, z);
        fin.set(ulong(OT_F_L) * fstride + fbase, L);
        fin.set(ulong(OT_F_M) * fstride + fbase, M);
        fin.set(ulong(OT_F_N) * fstride + fbase, Nd);
        fin.set(ulong(OT_F_I) * fstride + fbase, ii);
        fin.set(ulong(OT_F_OPD) * fstride + fbase, opd);
        fin.set(ulong(OT_F_L0) * fstride + fbase, L0);
        fin.set(ulong(OT_F_M0) * fstride + fbase, M0);
        fin.set(ulong(OT_F_N0) * fstride + fbase, N0);
    }
}

// ---------------------------------------------------------------------------
// Entry points (design 4.3; binding order = trace_layout.BUFFER_ORDER).
// ---------------------------------------------------------------------------

kernel void trace_surfaces_df64(
    const device float* launch_hi [[buffer(0)]],  const device float* launch_lo [[buffer(1)]],
    const device int*   surf_int  [[buffer(2)]],
    const device float* surf_hi   [[buffer(3)]],  const device float* surf_lo   [[buffer(4)]],
    const device float* coef_hi   [[buffer(5)]],  const device float* coef_lo   [[buffer(6)]],
    const device int*   dims      [[buffer(7)]],
    const device float* consts_hi [[buffer(8)]],  const device float* consts_lo [[buffer(9)]],
    device float* snap_hi [[buffer(10)]], device float* snap_lo [[buffer(11)]],
    device float* fin_hi  [[buffer(12)]], device float* fin_lo  [[buffer(13)]],
    device uchar* status  [[buffer(14)]], device uchar* iters   [[buffer(15)]],
    uint2 g [[thread_position_in_grid]])
{
    rspan<df64> launch = {launch_hi, launch_lo};
    rspan<df64> surf_real = {surf_hi, surf_lo};
    rspan<df64> coef = {coef_hi, coef_lo};
    rspan<df64> consts = {consts_hi, consts_lo};
    wspan<df64> snap = {snap_hi, snap_lo};
    wspan<df64> fin = {fin_hi, fin_lo};
    trace_body<df64>(launch, surf_int, surf_real, coef, dims, consts, snap, fin,
                     status, iters, g);
}

#ifdef OPTILAND_SF64_CORE_H
kernel void trace_surfaces_sf64(
    const device long* launch   [[buffer(0)]],
    const device int*  surf_int [[buffer(1)]],
    const device long* surf     [[buffer(2)]],
    const device long* coef     [[buffer(3)]],
    const device int*  dims     [[buffer(4)]],
    const device long* consts   [[buffer(5)]],
    device long*  snap   [[buffer(6)]],
    device long*  fin    [[buffer(7)]],
    device uchar* status [[buffer(8)]],
    device uchar* iters  [[buffer(9)]],
    uint2 g [[thread_position_in_grid]])
{
    rspan<sf64> launch_s = {launch};
    rspan<sf64> surf_real = {surf};
    rspan<sf64> coef_s = {coef};
    rspan<sf64> consts_s = {consts};
    wspan<sf64> snap_s = {snap};
    wspan<sf64> fin_s = {fin};
    trace_body<sf64>(launch_s, surf_int, surf_real, coef_s, dims, consts_s, snap_s,
                     fin_s, status, iters, g);
}
#endif  // OPTILAND_SF64_CORE_H

// ---------------------------------------------------------------------------
// Probe kernels (design 4.13, plan 4/WP1).  Compiled only into the probe
// build, so the production library carries none of them.  Day-1 set: the
// identity copy, which pins the R span load/store path on its own.
// ---------------------------------------------------------------------------

#ifdef OPTILAND_TRACE_PROBES

kernel void probe_copy_df64(
    const device float* in_hi  [[buffer(0)]], const device float* in_lo  [[buffer(1)]],
    device float* out_hi [[buffer(2)]], device float* out_lo [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    rspan<df64> in = {in_hi, in_lo};
    wspan<df64> out = {out_hi, out_lo};
    out.set(ulong(i), in.get(ulong(i)));
}

#ifdef OPTILAND_SF64_CORE_H
kernel void probe_copy_sf64(
    const device long* in  [[buffer(0)]],
    device long* out [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    rspan<sf64> in_s = {in};
    wspan<sf64> out_s = {out};
    out_s.set(ulong(i), in_s.get(ulong(i)));
}
#endif  // OPTILAND_SF64_CORE_H

// ---------------------------------------------------------------------------
// Probe bodies.  Each drives exactly the device function the trace loop calls,
// over flat plane-major arrays (`plane k of ray i` at `k * n + i`), so a unit
// test can compare its RAW COMPONENTS against the live Python object on
// identical encoded inputs.  Integer parameters travel in one small
// `const device int*` buffer (the `dims` precedent) rather than as `constant`
// scalars, and every real parameter is read out of a full `surf_real` ROW, so
// the probes exercise the slot mapping the trace loop uses.
// ---------------------------------------------------------------------------

//: `ip` slots shared by every probe.
#define OT_PROBE_N     0
#define OT_PROBE_PARAM 1
#define OT_PROBE_GEOM    2
#define OT_PROBE_APCODE  3
#define OT_PROBE_NCOEF   4
#define OT_PROBE_MAXITER 5

template <typename R>
inline void probe_pose_body(rspan<R> in, rspan<R> pose, const device int* ip,
                            wspan<R> out, uint i, bool to_global) {
    const ulong n = ulong(ip[OT_PROBE_N]);
    const int flags = ip[OT_PROBE_PARAM];
    const ulong k = ulong(i);
    if (k >= n) {
        return;
    }
    ray6<R> r;
    r.x = in.get(0UL * n + k);
    r.y = in.get(1UL * n + k);
    r.z = in.get(2UL * n + k);
    r.L = in.get(3UL * n + k);
    r.M = in.get(4UL * n + k);
    r.N = in.get(5UL * n + k);
    if (to_global) {
        globalize<R>(r, pose, 0UL, flags);
    } else {
        localize<R>(r, pose, 0UL, flags);
    }
    out.set(0UL * n + k, r.x);
    out.set(1UL * n + k, r.y);
    out.set(2UL * n + k, r.z);
    out.set(3UL * n + k, r.L);
    out.set(4UL * n + k, r.M);
    out.set(5UL * n + k, r.N);
}

template <typename R>
inline void probe_contains_body(rspan<R> in, rspan<R> par, const device int* ip,
                                device uchar* out, uint i) {
    const ulong n = ulong(ip[OT_PROBE_N]);
    const int code = ip[OT_PROBE_PARAM];
    const ulong k = ulong(i);
    if (k >= n) {
        return;
    }
    const bool inside = ap_contains<R>(code, slot<R>(par, 0UL, OT_SR_AP0),
                                       slot<R>(par, 0UL, OT_SR_AP1),
                                       slot<R>(par, 0UL, OT_SR_AP2),
                                       slot<R>(par, 0UL, OT_SR_AP3),
                                       in.get(0UL * n + k), in.get(1UL * n + k));
    out[k] = uchar(inside);
}

template <typename R>
inline void probe_interact_body(rspan<R> in, rspan<R> par, const device int* ip,
                                wspan<R> out, device uchar* status, uint i) {
    const ulong n = ulong(ip[OT_PROBE_N]);
    const bool reflective = (ip[OT_PROBE_PARAM] & OT_FL_REFLECTIVE) != 0;
    const ulong k = ulong(i);
    if (k >= n) {
        return;
    }
    uchar st = 0;
    const dir3<R> d = interact<R>(in.get(0UL * n + k), in.get(1UL * n + k),
                                  in.get(2UL * n + k), in.get(3UL * n + k),
                                  in.get(4UL * n + k), in.get(5UL * n + k),
                                  reflective, slot<R>(par, 0UL, OT_SR_U),
                                  slot<R>(par, 0UL, OT_SR_U2), st);
    out.set(0UL * n + k, d.L);
    out.set(1UL * n + k, d.M);
    out.set(2UL * n + k, d.N);
    status[k] = st;
}

template <typename R>
inline void probe_pow_scalar_body(rspan<R> in, const device float* e,
                                  const device int* ip, wspan<R> out, uint i) {
    const ulong n = ulong(ip[OT_PROBE_N]);
    const ulong k = ulong(i);
    if (k >= n) {
        return;
    }
    out.set(k, pow_scalar<R>(in.get(k), e[0]));
}

// Plane order in: x, y, z, L, M, N, i, opd, w, t.  Out: x, y, z, i, opd.
// The order of the three steps is Surface._trace_real's: propagate (which
// absorbs), then the OPD accumulation.
template <typename R>
inline void probe_propagate_body(rspan<R> in, rspan<R> par, const device int* ip,
                                 wspan<R> out, uint i) {
    const ulong n = ulong(ip[OT_PROBE_N]);
    const int flags = ip[OT_PROBE_PARAM];
    const ulong k = ulong(i);
    if (k >= n) {
        return;
    }
    ray6<R> r;
    r.x = in.get(0UL * n + k);
    r.y = in.get(1UL * n + k);
    r.z = in.get(2UL * n + k);
    r.L = in.get(3UL * n + k);
    r.M = in.get(4UL * n + k);
    r.N = in.get(5UL * n + k);
    R inten = in.get(6UL * n + k);
    R opd = in.get(7UL * n + k);
    const R w = in.get(8UL * n + k);
    const R t = in.get(9UL * n + k);
    r = advance<R>(r, t);
    if (flags & OT_FL_ABSORBING) {
        inten = absorb<R>(inten, t, w, slot<R>(par, 0UL, OT_SR_ALPHA));
    }
    opd = accumulate_opd<R>(opd, t, slot<R>(par, 0UL, OT_SR_NPRE));
    out.set(0UL * n + k, r.x);
    out.set(1UL * n + k, r.y);
    out.set(2UL * n + k, r.z);
    out.set(3UL * n + k, inten);
    out.set(4UL * n + k, opd);
}


// The distance probes (design 4.13).  They take the surface's integer
// parameters in the extra `ip` slots below, its real parameters as a full
// `surf_real` ROW and its aspheric coefficients in a `coef` buffer read from
// offset 0, so they drive exactly the functions the trace loop calls, through
// exactly the same slot mapping.  `consts` carries eps and the |N| floor.

//: `probe_sag` / `probe_normal`: 2 planes in (x, y), 1 / 3 planes out.
template <typename R>
inline void probe_sag_body(rspan<R> in, rspan<R> par, rspan<R> coef,
                           const device int* ip, wspan<R> out, uint i) {
    const ulong n = ulong(ip[OT_PROBE_N]);
    const ulong k = ulong(i);
    if (k >= n) {
        return;
    }
    out.set(k, sag_of<R>(ip[OT_PROBE_GEOM], in.get(0UL * n + k), in.get(1UL * n + k),
                         par, 0UL, coef, 0UL, ip[OT_PROBE_NCOEF]));
}

template <typename R>
inline void probe_normal_body(rspan<R> in, rspan<R> par, rspan<R> coef,
                              const device int* ip, wspan<R> out, uint i) {
    const ulong n = ulong(ip[OT_PROBE_N]);
    const ulong k = ulong(i);
    if (k >= n) {
        return;
    }
    R nx, ny, nz;
    normal_of<R>(ip[OT_PROBE_GEOM], in.get(0UL * n + k), in.get(1UL * n + k), par, 0UL,
                 coef, 0UL, ip[OT_PROBE_NCOEF], nx, ny, nz);
    out.set(0UL * n + k, nx);
    out.set(1UL * n + k, ny);
    out.set(2UL * n + k, nz);
}

//: `probe_distance` (the whole SI_GEOM switch, MISS included) and
//  `probe_newton` (`newton_distance` on its own, no MISS bit): 6 planes in
//  (x, y, z, L, M, N), one plane out plus the status and iteration bytes.
template <typename R>
inline void probe_distance_body(rspan<R> in, rspan<R> par, rspan<R> coef,
                                rspan<R> consts, const device int* ip, wspan<R> out,
                                device uchar* status, device uchar* iters, uint i,
                                bool newton_only) {
    const ulong n = ulong(ip[OT_PROBE_N]);
    const ulong k = ulong(i);
    if (k >= n) {
        return;
    }
    const R x = in.get(0UL * n + k);
    const R y = in.get(1UL * n + k);
    const R z = in.get(2UL * n + k);
    const R L = in.get(3UL * n + k);
    const R M = in.get(4UL * n + k);
    const R Nd = in.get(5UL * n + k);
    const R eps = consts.get(ulong(OT_C_EPS));
    const R nfloor = consts.get(ulong(OT_C_NFLOOR));
    const int geom = ip[OT_PROBE_GEOM];
    const int flags = ip[OT_PROBE_PARAM];
    const int apcode = ip[OT_PROBE_APCODE];
    const int ncoef = ip[OT_PROBE_NCOEF];
    const int max_iter = ip[OT_PROBE_MAXITER];
    uchar st = 0;
    uchar it = 0;
    R t;
    if (newton_only) {
        t = newton_distance<R>(x, y, z, L, M, Nd, par, 0UL, coef, 0UL, geom, ncoef,
                               max_iter, flags, apcode, eps, nfloor, st, it);
    } else {
        t = surface_distance<R>(geom, x, y, z, L, M, Nd, par, 0UL, coef, 0UL, flags,
                                apcode, ncoef, max_iter, eps, nfloor, st, it);
    }
    out.set(k, t);
    status[k] = st;
    iters[k] = it;
}

// ---- df64 entry points ----------------------------------------------------

kernel void probe_localize_df64(
    const device float* in_hi [[buffer(0)]],  const device float* in_lo [[buffer(1)]],
    const device float* pose_hi [[buffer(2)]], const device float* pose_lo [[buffer(3)]],
    const device int* ip [[buffer(4)]],
    device float* out_hi [[buffer(5)]], device float* out_lo [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    rspan<df64> in = {in_hi, in_lo};
    rspan<df64> pose = {pose_hi, pose_lo};
    wspan<df64> out = {out_hi, out_lo};
    probe_pose_body<df64>(in, pose, ip, out, i, false);
}

kernel void probe_globalize_df64(
    const device float* in_hi [[buffer(0)]],  const device float* in_lo [[buffer(1)]],
    const device float* pose_hi [[buffer(2)]], const device float* pose_lo [[buffer(3)]],
    const device int* ip [[buffer(4)]],
    device float* out_hi [[buffer(5)]], device float* out_lo [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    rspan<df64> in = {in_hi, in_lo};
    rspan<df64> pose = {pose_hi, pose_lo};
    wspan<df64> out = {out_hi, out_lo};
    probe_pose_body<df64>(in, pose, ip, out, i, true);
}

kernel void probe_contains_df64(
    const device float* in_hi [[buffer(0)]], const device float* in_lo [[buffer(1)]],
    const device float* par_hi [[buffer(2)]], const device float* par_lo [[buffer(3)]],
    const device int* ip [[buffer(4)]],
    device uchar* out [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    rspan<df64> in = {in_hi, in_lo};
    rspan<df64> par = {par_hi, par_lo};
    probe_contains_body<df64>(in, par, ip, out, i);
}

kernel void probe_interact_df64(
    const device float* in_hi [[buffer(0)]], const device float* in_lo [[buffer(1)]],
    const device float* par_hi [[buffer(2)]], const device float* par_lo [[buffer(3)]],
    const device int* ip [[buffer(4)]],
    device float* out_hi [[buffer(5)]], device float* out_lo [[buffer(6)]],
    device uchar* status [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    rspan<df64> in = {in_hi, in_lo};
    rspan<df64> par = {par_hi, par_lo};
    wspan<df64> out = {out_hi, out_lo};
    probe_interact_body<df64>(in, par, ip, out, status, i);
}

kernel void probe_pow_scalar_df64(
    const device float* in_hi [[buffer(0)]], const device float* in_lo [[buffer(1)]],
    const device float* e [[buffer(2)]],
    const device int* ip [[buffer(3)]],
    device float* out_hi [[buffer(4)]], device float* out_lo [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    rspan<df64> in = {in_hi, in_lo};
    wspan<df64> out = {out_hi, out_lo};
    probe_pow_scalar_body<df64>(in, e, ip, out, i);
}

kernel void probe_propagate_df64(
    const device float* in_hi [[buffer(0)]], const device float* in_lo [[buffer(1)]],
    const device float* par_hi [[buffer(2)]], const device float* par_lo [[buffer(3)]],
    const device int* ip [[buffer(4)]],
    device float* out_hi [[buffer(5)]], device float* out_lo [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    rspan<df64> in = {in_hi, in_lo};
    rspan<df64> par = {par_hi, par_lo};
    wspan<df64> out = {out_hi, out_lo};
    probe_propagate_body<df64>(in, par, ip, out, i);
}


kernel void probe_sag_df64(
    const device float* in_hi [[buffer(0)]], const device float* in_lo [[buffer(1)]],
    const device float* par_hi [[buffer(2)]], const device float* par_lo [[buffer(3)]],
    const device float* coef_hi [[buffer(4)]], const device float* coef_lo [[buffer(5)]],
    const device int* ip [[buffer(6)]],
    device float* out_hi [[buffer(7)]], device float* out_lo [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
    rspan<df64> in = {in_hi, in_lo};
    rspan<df64> par = {par_hi, par_lo};
    rspan<df64> coef = {coef_hi, coef_lo};
    wspan<df64> out = {out_hi, out_lo};
    probe_sag_body<df64>(in, par, coef, ip, out, i);
}

kernel void probe_normal_df64(
    const device float* in_hi [[buffer(0)]], const device float* in_lo [[buffer(1)]],
    const device float* par_hi [[buffer(2)]], const device float* par_lo [[buffer(3)]],
    const device float* coef_hi [[buffer(4)]], const device float* coef_lo [[buffer(5)]],
    const device int* ip [[buffer(6)]],
    device float* out_hi [[buffer(7)]], device float* out_lo [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
    rspan<df64> in = {in_hi, in_lo};
    rspan<df64> par = {par_hi, par_lo};
    rspan<df64> coef = {coef_hi, coef_lo};
    wspan<df64> out = {out_hi, out_lo};
    probe_normal_body<df64>(in, par, coef, ip, out, i);
}

kernel void probe_distance_df64(
    const device float* in_hi [[buffer(0)]], const device float* in_lo [[buffer(1)]],
    const device float* par_hi [[buffer(2)]], const device float* par_lo [[buffer(3)]],
    const device float* coef_hi [[buffer(4)]], const device float* coef_lo [[buffer(5)]],
    const device float* con_hi [[buffer(6)]], const device float* con_lo [[buffer(7)]],
    const device int* ip [[buffer(8)]],
    device float* out_hi [[buffer(9)]], device float* out_lo [[buffer(10)]],
    device uchar* status [[buffer(11)]], device uchar* iters [[buffer(12)]],
    uint i [[thread_position_in_grid]])
{
    rspan<df64> in = {in_hi, in_lo};
    rspan<df64> par = {par_hi, par_lo};
    rspan<df64> coef = {coef_hi, coef_lo};
    rspan<df64> consts = {con_hi, con_lo};
    wspan<df64> out = {out_hi, out_lo};
    probe_distance_body<df64>(in, par, coef, consts, ip, out, status, iters, i, false);
}

kernel void probe_newton_df64(
    const device float* in_hi [[buffer(0)]], const device float* in_lo [[buffer(1)]],
    const device float* par_hi [[buffer(2)]], const device float* par_lo [[buffer(3)]],
    const device float* coef_hi [[buffer(4)]], const device float* coef_lo [[buffer(5)]],
    const device float* con_hi [[buffer(6)]], const device float* con_lo [[buffer(7)]],
    const device int* ip [[buffer(8)]],
    device float* out_hi [[buffer(9)]], device float* out_lo [[buffer(10)]],
    device uchar* status [[buffer(11)]], device uchar* iters [[buffer(12)]],
    uint i [[thread_position_in_grid]])
{
    rspan<df64> in = {in_hi, in_lo};
    rspan<df64> par = {par_hi, par_lo};
    rspan<df64> coef = {coef_hi, coef_lo};
    rspan<df64> consts = {con_hi, con_lo};
    wspan<df64> out = {out_hi, out_lo};
    probe_distance_body<df64>(in, par, coef, consts, ip, out, status, iters, i, true);
}

// ---- sf64 entry points ----------------------------------------------------

#ifdef OPTILAND_SF64_CORE_H

kernel void probe_localize_sf64(
    const device long* in [[buffer(0)]],
    const device long* pose [[buffer(1)]],
    const device int* ip [[buffer(2)]],
    device long* out [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    rspan<sf64> in_s = {in};
    rspan<sf64> pose_s = {pose};
    wspan<sf64> out_s = {out};
    probe_pose_body<sf64>(in_s, pose_s, ip, out_s, i, false);
}

kernel void probe_globalize_sf64(
    const device long* in [[buffer(0)]],
    const device long* pose [[buffer(1)]],
    const device int* ip [[buffer(2)]],
    device long* out [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    rspan<sf64> in_s = {in};
    rspan<sf64> pose_s = {pose};
    wspan<sf64> out_s = {out};
    probe_pose_body<sf64>(in_s, pose_s, ip, out_s, i, true);
}

kernel void probe_contains_sf64(
    const device long* in [[buffer(0)]],
    const device long* par [[buffer(1)]],
    const device int* ip [[buffer(2)]],
    device uchar* out [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    rspan<sf64> in_s = {in};
    rspan<sf64> par_s = {par};
    probe_contains_body<sf64>(in_s, par_s, ip, out, i);
}

kernel void probe_interact_sf64(
    const device long* in [[buffer(0)]],
    const device long* par [[buffer(1)]],
    const device int* ip [[buffer(2)]],
    device long* out [[buffer(3)]],
    device uchar* status [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    rspan<sf64> in_s = {in};
    rspan<sf64> par_s = {par};
    wspan<sf64> out_s = {out};
    probe_interact_body<sf64>(in_s, par_s, ip, out_s, status, i);
}

kernel void probe_pow_scalar_sf64(
    const device long* in [[buffer(0)]],
    const device float* e [[buffer(1)]],
    const device int* ip [[buffer(2)]],
    device long* out [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    rspan<sf64> in_s = {in};
    wspan<sf64> out_s = {out};
    probe_pow_scalar_body<sf64>(in_s, e, ip, out_s, i);
}

kernel void probe_propagate_sf64(
    const device long* in [[buffer(0)]],
    const device long* par [[buffer(1)]],
    const device int* ip [[buffer(2)]],
    device long* out [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    rspan<sf64> in_s = {in};
    rspan<sf64> par_s = {par};
    wspan<sf64> out_s = {out};
    probe_propagate_body<sf64>(in_s, par_s, ip, out_s, i);
}


kernel void probe_sag_sf64(
    const device long* in [[buffer(0)]],
    const device long* par [[buffer(1)]],
    const device long* coef [[buffer(2)]],
    const device int* ip [[buffer(3)]],
    device long* out [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    rspan<sf64> in_s = {in};
    rspan<sf64> par_s = {par};
    rspan<sf64> coef_s = {coef};
    wspan<sf64> out_s = {out};
    probe_sag_body<sf64>(in_s, par_s, coef_s, ip, out_s, i);
}

kernel void probe_normal_sf64(
    const device long* in [[buffer(0)]],
    const device long* par [[buffer(1)]],
    const device long* coef [[buffer(2)]],
    const device int* ip [[buffer(3)]],
    device long* out [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    rspan<sf64> in_s = {in};
    rspan<sf64> par_s = {par};
    rspan<sf64> coef_s = {coef};
    wspan<sf64> out_s = {out};
    probe_normal_body<sf64>(in_s, par_s, coef_s, ip, out_s, i);
}

kernel void probe_distance_sf64(
    const device long* in [[buffer(0)]],
    const device long* par [[buffer(1)]],
    const device long* coef [[buffer(2)]],
    const device long* consts [[buffer(3)]],
    const device int* ip [[buffer(4)]],
    device long* out [[buffer(5)]],
    device uchar* status [[buffer(6)]], device uchar* iters [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    rspan<sf64> in_s = {in};
    rspan<sf64> par_s = {par};
    rspan<sf64> coef_s = {coef};
    rspan<sf64> con_s = {consts};
    wspan<sf64> out_s = {out};
    probe_distance_body<sf64>(in_s, par_s, coef_s, con_s, ip, out_s, status, iters, i,
                              false);
}

kernel void probe_newton_sf64(
    const device long* in [[buffer(0)]],
    const device long* par [[buffer(1)]],
    const device long* coef [[buffer(2)]],
    const device long* consts [[buffer(3)]],
    const device int* ip [[buffer(4)]],
    device long* out [[buffer(5)]],
    device uchar* status [[buffer(6)]], device uchar* iters [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    rspan<sf64> in_s = {in};
    rspan<sf64> par_s = {par};
    rspan<sf64> coef_s = {coef};
    rspan<sf64> con_s = {consts};
    wspan<sf64> out_s = {out};
    probe_distance_body<sf64>(in_s, par_s, coef_s, con_s, ip, out_s, status, iters, i,
                              true);
}

#endif  // OPTILAND_SF64_CORE_H

#endif  // OPTILAND_TRACE_PROBES

#endif  // OPTILAND_TRACE_METAL
