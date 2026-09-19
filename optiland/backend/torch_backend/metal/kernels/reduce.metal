// reduce.metal — reductions for emulated float64 on Apple GPUs.
//
// Layout contract: every kernel reduces the MIDDLE axis of a contiguous
// [outer, n, inner] block. Any single-axis reduction of a (permuted-)contiguous
// tensor maps onto this layout on the host (outer = prod(shape[:axis]),
// n = shape[axis], inner = prod(shape[axis+1:])); a full reduction uses
// outer = inner = 1. Element (o, k, i) lives at index (o * n + k) * inner + i.
//
// Two kernel families per operation:
//
//   <op>_df64      tree kernel. One threadgroup of OPTILAND_REDUCE_TG (256)
//                  threads per (line, group) where line = (o, i) and
//                  g in [0, groups). Thread `lid` of group `g` accumulates the
//                  strided slice k = g*256 + lid, g*256 + lid + groups*256, ...
//                  into a df64 with the op's combine function, then a
//                  threadgroup tree (in a `threadgroup df64 buf[256]` declared
//                  in the kernel body; compile_shader has no [[threadgroup]]
//                  parameters) combines the 256 partials. Output layout is
//                  [outer, groups, inner] so the host can run a second pass
//                  with n = groups until groups == 1.
//                  Launch: threads=[outer*inner*groups*256, 1, 1],
//                          group_size=[256, 1, 1]  (group_size must be a power
//                          of two <= 256; 256 is the intended value).
//   <op>_seq_df64  one thread per (o, i) line, sequential over n. Output layout
//                  [outer, inner]. Meant for short n or when outer*inner is
//                  already large (the tree kernel would waste 255/256 of its
//                  threads on a length-4 axis).
//                  Launch: threads=[outer*inner, 1, 1].
//
// Second-pass rules (the partials are exact members of the reduction, so the
// second pass is the same op EXCEPT where the first pass filtered inputs):
//   sum      -> sum         nansum  -> sum   (NaN partials such as inf + -inf
//                                             must propagate, not be skipped)
//   prod     -> prod        max/min -> max/min
//   nanmax   -> nanmax      nanmin  -> nanmin (a NaN partial means "all NaN"
//                                             in that slice; fmax skips it)
//   argmax   -> argmax with has_idx = 1 and in_idx = previous out_idx
//   argmin   -> argmin      (same)
//
// Semantics (torch/numpy):
//   sum/prod        TwoSum-accumulated df::add / df::mul; NaN and inf
//                   propagate as in IEEE. The accumulation order is
//                   (strided per thread) + (binary tree) + (second pass), so
//                   results are NOT bit-identical to numpy's pairwise sum;
//                   the error is bounded per df::add (3u^2 relative) times the
//                   depth of the accumulation and is exact for integer data.
//   nansum          NaN inputs are treated as 0.
//   max/min         NaN-propagating (torch.amax/amin). Ties between +0 and -0
//                   are value-equal; which sign survives depends on the
//                   accumulation order (numpy keeps the first in memory).
//   nanmax/nanmin   ignore NaN; an all-NaN line yields NaN (numpy nanmax).
//   argmax/argmin   first index on ties (+0 == -0 is a tie); a NaN "wins" and
//                   the first NaN index is returned (torch/numpy). Indices are
//                   int64 (`long`).
//   cumsum          sequential per line with a THREE-word float accumulator
//                   (x0 + x1 + x2, ~72 bits) so that the rounding to df64 at
//                   each prefix is the only significant error (<~1u^2 per
//                   prefix). A plain df::add chain would drift like sqrt(k)
//                   u^2 for long lines. Third-word precision is lost when the
//                   running sum drops below ~1e-24 (float32 denormal flush);
//                   the result then degrades gracefully to df64 precision.
//
// sf64 twins (compiled only when sf64_core.h precedes this file and defines
// OPTILAND_SF64_CORE_H) use the same layouts on one `long` buffer holding the
// binary64 bit pattern, with correctly rounded sf::add / sf::mul per
// operation. Because the accumulation order differs from numpy/torch
// (pairwise), bit-exact agreement is not expected; the error is at most one
// binary64 rounding per accumulation step.
//
// Must be amalgamated after df64_core.h (and sf64_core.h for the sf64 twins).
// Part of Optiland-Metal (MIT).

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

#ifndef OPTILAND_REDUCE_METAL
#define OPTILAND_REDUCE_METAL

#ifndef OPTILAND_DF64_CORE_H
#error "reduce.metal must be amalgamated after df64_core.h"
#endif

#define OPTILAND_REDUCE_TG 256u

namespace reduce_detail {

// ---------------------------------------------------------------------------
// Buffer adapters (device pointers held in thread-space structs)
// ---------------------------------------------------------------------------
struct df64_src {
    device const float* hi;
    device const float* lo;
    inline df64 at(ulong i) const { return df::make(hi[i], lo[i]); }
};
struct df64_dst {
    device float* hi;
    device float* lo;
    inline void put(ulong i, df64 v) const { hi[i] = v.hi; lo[i] = v.lo; }
};

// ---------------------------------------------------------------------------
// Reduction operations. `map` is applied to raw inputs only (never to partials),
// `combine` is associative-enough for tree use.
// ---------------------------------------------------------------------------
struct sum_df64_op {
    typedef df64 value_type;
    static inline df64 identity() { return df::zero(); }
    static inline df64 map(df64 x) { return x; }
    static inline df64 combine(df64 a, df64 b) { return df::add(a, b); }
};
struct nansum_df64_op {
    typedef df64 value_type;
    static inline df64 identity() { return df::zero(); }
    static inline df64 map(df64 x) { return df::is_nan(x) ? df::zero() : x; }
    static inline df64 combine(df64 a, df64 b) { return df::add(a, b); }
};
struct prod_df64_op {
    typedef df64 value_type;
    static inline df64 identity() { return df::one(); }
    static inline df64 map(df64 x) { return x; }
    static inline df64 combine(df64 a, df64 b) { return df::mul(a, b); }
};
struct max_df64_op {
    typedef df64 value_type;
    static inline df64 identity() { return df::make(-INFINITY, 0.0f); }
    static inline df64 map(df64 x) { return x; }
    static inline df64 combine(df64 a, df64 b) { return df::maximum(a, b); }
};
struct min_df64_op {
    typedef df64 value_type;
    static inline df64 identity() { return df::make(INFINITY, 0.0f); }
    static inline df64 map(df64 x) { return x; }
    static inline df64 combine(df64 a, df64 b) { return df::minimum(a, b); }
};
struct nanmax_df64_op {
    typedef df64 value_type;
    static inline df64 identity() { return df::nan(); }
    static inline df64 map(df64 x) { return x; }
    static inline df64 combine(df64 a, df64 b) { return df::fmax(a, b); }
};
struct nanmin_df64_op {
    typedef df64 value_type;
    static inline df64 identity() { return df::nan(); }
    static inline df64 map(df64 x) { return x; }
    static inline df64 combine(df64 a, df64 b) { return df::fmin(a, b); }
};

// ---------------------------------------------------------------------------
// Generic tree reduction over one (line, group) per threadgroup.
// ---------------------------------------------------------------------------
template <class Op, class Src, class Dst>
inline void tree_reduce(Src src, Dst dst, uint n, uint inner, uint groups,
                        uint lid, uint gid, uint tpg,
                        threadgroup typename Op::value_type* buf) {
    ulong line = (ulong)gid / groups;
    uint g = (uint)((ulong)gid - line * groups);
    ulong o = line / inner;
    ulong i = line - o * inner;
    ulong base = o * (ulong)n * inner + i;
    typename Op::value_type acc = Op::identity();
    for (ulong k = (ulong)g * tpg + lid; k < n; k += (ulong)groups * tpg)
        acc = Op::combine(acc, Op::map(src.at(base + k * inner)));
    buf[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg >> 1; s > 0; s >>= 1) {
        if (lid < s) buf[lid] = Op::combine(buf[lid], buf[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) dst.put((o * groups + g) * inner + i, buf[0]);
}

// One thread per (o, i) line, sequential over n.
template <class Op, class Src, class Dst>
inline void line_reduce(Src src, Dst dst, uint n, uint inner, ulong tid) {
    ulong o = tid / inner;
    ulong i = tid - o * inner;
    ulong base = o * (ulong)n * inner + i;
    typename Op::value_type acc = Op::identity();
    for (ulong k = 0; k < n; ++k) acc = Op::combine(acc, Op::map(src.at(base + k * inner)));
    dst.put(tid, acc);
}

// ---------------------------------------------------------------------------
// Index reductions: (value, index) pairs; idx < 0 marks "empty".
// ---------------------------------------------------------------------------
struct df64_arg {
    df64 v;
    long idx;
};
inline df64_arg make_arg(df64 v, long idx) { df64_arg r; r.v = v; r.idx = idx; return r; }

template <bool IS_MAX>
struct arg_df64_op {
    typedef df64_arg value_type;
    static inline df64_arg identity() { return make_arg(df::nan(), -1L); }
    static inline bool better(df64 cand, df64 cur) {
        return IS_MAX ? df::gt(cand, cur) : df::lt(cand, cur);
    }
    static inline df64_arg combine(df64_arg a, df64_arg b) {
        if (b.idx < 0) return a;
        if (a.idx < 0) return b;
        bool an = df::is_nan(a.v);
        bool bn = df::is_nan(b.v);
        if (an || bn) {
            if (an && bn) return (a.idx <= b.idx) ? a : b;
            return an ? a : b;
        }
        if (better(b.v, a.v)) return b;
        if (better(a.v, b.v)) return a;
        return (a.idx <= b.idx) ? a : b;  // tie (including +0 vs -0): first index
    }
};

struct df64_arg_src {
    device const float* hi;
    device const float* lo;
    device const long* idx;
    uint has_idx;
    inline df64_arg at(ulong i, ulong k) const {
        return make_arg(df::make(hi[i], lo[i]), has_idx ? idx[i] : (long)k);
    }
};
struct df64_arg_dst {
    device float* hi;
    device float* lo;
    device long* idx;
    inline void put(ulong i, df64_arg a) const { hi[i] = a.v.hi; lo[i] = a.v.lo; idx[i] = a.idx; }
};

template <class Op, class Src, class Dst>
inline void tree_arg_reduce(Src src, Dst dst, uint n, uint inner, uint groups,
                            uint lid, uint gid, uint tpg,
                            threadgroup typename Op::value_type* buf) {
    ulong line = (ulong)gid / groups;
    uint g = (uint)((ulong)gid - line * groups);
    ulong o = line / inner;
    ulong i = line - o * inner;
    ulong base = o * (ulong)n * inner + i;
    typename Op::value_type acc = Op::identity();
    for (ulong k = (ulong)g * tpg + lid; k < n; k += (ulong)groups * tpg)
        acc = Op::combine(acc, src.at(base + k * inner, k));
    buf[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg >> 1; s > 0; s >>= 1) {
        if (lid < s) buf[lid] = Op::combine(buf[lid], buf[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) dst.put((o * groups + g) * inner + i, buf[0]);
}

template <class Op, class Src, class Dst>
inline void line_arg_reduce(Src src, Dst dst, uint n, uint inner, ulong tid) {
    ulong o = tid / inner;
    ulong i = tid - o * inner;
    ulong base = o * (ulong)n * inner + i;
    typename Op::value_type acc = Op::identity();
    for (ulong k = 0; k < n; ++k) acc = Op::combine(acc, src.at(base + k * inner, k));
    dst.put(tid, acc);
}

// ---------------------------------------------------------------------------
// Three-word accumulator for cumsum: x0 + x1 + x2 with |x1| <= ulp(x0)/2 and
// |x2| <= ulp(x1)/2 after renormalization (~72 significant bits).
// ---------------------------------------------------------------------------
struct tf64 {
    float x0, x1, x2;
};
inline tf64 tf_make(float a, float b, float c) { tf64 r; r.x0 = a; r.x1 = b; r.x2 = c; return r; }
inline tf64 tf_renorm(float x0, float x1, float x2) {
    df64 t = df::quick_two_sum(x1, x2);
    df64 u = df::quick_two_sum(x0, t.hi);
    df64 w = df::quick_two_sum(u.lo, t.lo);
    return tf_make(u.hi, w.hi, w.lo);
}
inline tf64 tf_add(tf64 a, df64 b) {
    if (!isfinite(a.x0) || !isfinite(b.hi)) return tf_make(a.x0 + b.hi, 0.0f, 0.0f);
    df64 s = df::two_sum(a.x0, b.hi);
    if (!isfinite(s.hi)) return tf_make(s.hi, 0.0f, 0.0f);  // overflow
    df64 t = df::two_sum(a.x1, s.lo);
    float x2 = a.x2 + t.lo;
    df64 s2 = df::two_sum(s.hi, b.lo);
    df64 t2 = df::two_sum(t.hi, s2.lo);
    x2 = x2 + t2.lo;
    return tf_renorm(s2.hi, t2.hi, x2);
}
// Nearest-ish df64 to the three-word value (x0, x1 are already renormalized).
inline df64 tf_to_df64(tf64 a) { return df::fix(df::make(a.x0, a.x1)); }

}  // namespace reduce_detail

// ---------------------------------------------------------------------------
// df64 kernels
// ---------------------------------------------------------------------------
#define OPTILAND_DEFINE_TREE_REDUCE_DF64(NAME, OP)                                          \
kernel void NAME(device const float* in_hi [[buffer(0)]],                                   \
                 device const float* in_lo [[buffer(1)]],                                   \
                 device float* out_hi [[buffer(2)]],                                        \
                 device float* out_lo [[buffer(3)]],                                        \
                 constant uint& outer [[buffer(4)]],                                        \
                 constant uint& n [[buffer(5)]],                                            \
                 constant uint& inner [[buffer(6)]],                                        \
                 constant uint& groups [[buffer(7)]],                                       \
                 uint lid [[thread_position_in_threadgroup]],                               \
                 uint gid [[threadgroup_position_in_grid]],                                 \
                 uint tpg [[threads_per_threadgroup]]) {                                    \
    (void)outer;                                                                            \
    threadgroup df64 buf[OPTILAND_REDUCE_TG];                                               \
    reduce_detail::df64_src src; src.hi = in_hi; src.lo = in_lo;                            \
    reduce_detail::df64_dst dst; dst.hi = out_hi; dst.lo = out_lo;                          \
    reduce_detail::tree_reduce<OP>(src, dst, n, inner, groups, lid, gid, tpg, buf);         \
}

#define OPTILAND_DEFINE_LINE_REDUCE_DF64(NAME, OP)                                          \
kernel void NAME(device const float* in_hi [[buffer(0)]],                                   \
                 device const float* in_lo [[buffer(1)]],                                   \
                 device float* out_hi [[buffer(2)]],                                        \
                 device float* out_lo [[buffer(3)]],                                        \
                 constant uint& outer [[buffer(4)]],                                        \
                 constant uint& n [[buffer(5)]],                                            \
                 constant uint& inner [[buffer(6)]],                                        \
                 uint tid [[thread_position_in_grid]]) {                                    \
    (void)outer;                                                                            \
    reduce_detail::df64_src src; src.hi = in_hi; src.lo = in_lo;                            \
    reduce_detail::df64_dst dst; dst.hi = out_hi; dst.lo = out_lo;                          \
    reduce_detail::line_reduce<OP>(src, dst, n, inner, (ulong)tid);                         \
}

#define OPTILAND_DEFINE_TREE_ARG_DF64(NAME, OP)                                             \
kernel void NAME(device const float* in_hi [[buffer(0)]],                                   \
                 device const float* in_lo [[buffer(1)]],                                   \
                 device const long* in_idx [[buffer(2)]],                                   \
                 device float* out_hi [[buffer(3)]],                                        \
                 device float* out_lo [[buffer(4)]],                                        \
                 device long* out_idx [[buffer(5)]],                                        \
                 constant uint& outer [[buffer(6)]],                                        \
                 constant uint& n [[buffer(7)]],                                            \
                 constant uint& inner [[buffer(8)]],                                        \
                 constant uint& groups [[buffer(9)]],                                       \
                 constant uint& has_idx [[buffer(10)]],                                     \
                 uint lid [[thread_position_in_threadgroup]],                               \
                 uint gid [[threadgroup_position_in_grid]],                                 \
                 uint tpg [[threads_per_threadgroup]]) {                                    \
    (void)outer;                                                                            \
    threadgroup reduce_detail::df64_arg buf[OPTILAND_REDUCE_TG];                            \
    reduce_detail::df64_arg_src src;                                                        \
    src.hi = in_hi; src.lo = in_lo; src.idx = in_idx; src.has_idx = has_idx;                \
    reduce_detail::df64_arg_dst dst; dst.hi = out_hi; dst.lo = out_lo; dst.idx = out_idx;   \
    reduce_detail::tree_arg_reduce<OP>(src, dst, n, inner, groups, lid, gid, tpg, buf);     \
}

#define OPTILAND_DEFINE_LINE_ARG_DF64(NAME, OP)                                             \
kernel void NAME(device const float* in_hi [[buffer(0)]],                                   \
                 device const float* in_lo [[buffer(1)]],                                   \
                 device const long* in_idx [[buffer(2)]],                                   \
                 device float* out_hi [[buffer(3)]],                                        \
                 device float* out_lo [[buffer(4)]],                                        \
                 device long* out_idx [[buffer(5)]],                                        \
                 constant uint& outer [[buffer(6)]],                                        \
                 constant uint& n [[buffer(7)]],                                            \
                 constant uint& inner [[buffer(8)]],                                        \
                 constant uint& has_idx [[buffer(9)]],                                      \
                 uint tid [[thread_position_in_grid]]) {                                    \
    (void)outer;                                                                            \
    reduce_detail::df64_arg_src src;                                                        \
    src.hi = in_hi; src.lo = in_lo; src.idx = in_idx; src.has_idx = has_idx;                \
    reduce_detail::df64_arg_dst dst; dst.hi = out_hi; dst.lo = out_lo; dst.idx = out_idx;   \
    reduce_detail::line_arg_reduce<OP>(src, dst, n, inner, (ulong)tid);                     \
}

OPTILAND_DEFINE_TREE_REDUCE_DF64(sum_df64, reduce_detail::sum_df64_op)
OPTILAND_DEFINE_TREE_REDUCE_DF64(nansum_df64, reduce_detail::nansum_df64_op)
OPTILAND_DEFINE_TREE_REDUCE_DF64(prod_df64, reduce_detail::prod_df64_op)
OPTILAND_DEFINE_TREE_REDUCE_DF64(max_df64, reduce_detail::max_df64_op)
OPTILAND_DEFINE_TREE_REDUCE_DF64(min_df64, reduce_detail::min_df64_op)
OPTILAND_DEFINE_TREE_REDUCE_DF64(nanmax_df64, reduce_detail::nanmax_df64_op)
OPTILAND_DEFINE_TREE_REDUCE_DF64(nanmin_df64, reduce_detail::nanmin_df64_op)

OPTILAND_DEFINE_LINE_REDUCE_DF64(sum_seq_df64, reduce_detail::sum_df64_op)
OPTILAND_DEFINE_LINE_REDUCE_DF64(nansum_seq_df64, reduce_detail::nansum_df64_op)
OPTILAND_DEFINE_LINE_REDUCE_DF64(prod_seq_df64, reduce_detail::prod_df64_op)
OPTILAND_DEFINE_LINE_REDUCE_DF64(max_seq_df64, reduce_detail::max_df64_op)
OPTILAND_DEFINE_LINE_REDUCE_DF64(min_seq_df64, reduce_detail::min_df64_op)
OPTILAND_DEFINE_LINE_REDUCE_DF64(nanmax_seq_df64, reduce_detail::nanmax_df64_op)
OPTILAND_DEFINE_LINE_REDUCE_DF64(nanmin_seq_df64, reduce_detail::nanmin_df64_op)

OPTILAND_DEFINE_TREE_ARG_DF64(argmax_df64, reduce_detail::arg_df64_op<true>)
OPTILAND_DEFINE_TREE_ARG_DF64(argmin_df64, reduce_detail::arg_df64_op<false>)
OPTILAND_DEFINE_LINE_ARG_DF64(argmax_seq_df64, reduce_detail::arg_df64_op<true>)
OPTILAND_DEFINE_LINE_ARG_DF64(argmin_seq_df64, reduce_detail::arg_df64_op<false>)

// cumsum along the middle axis; one thread per (o, i) line.
// Launch: threads=[outer*inner, 1, 1]. Output has the input's shape.
kernel void cumsum_df64(device const float* in_hi [[buffer(0)]],
                        device const float* in_lo [[buffer(1)]],
                        device float* out_hi [[buffer(2)]],
                        device float* out_lo [[buffer(3)]],
                        constant uint& outer [[buffer(4)]],
                        constant uint& n [[buffer(5)]],
                        constant uint& inner [[buffer(6)]],
                        uint tid [[thread_position_in_grid]]) {
    (void)outer;
    ulong o = (ulong)tid / inner;
    ulong i = (ulong)tid - o * inner;
    ulong base = o * (ulong)n * inner + i;
    reduce_detail::tf64 acc = reduce_detail::tf_make(0.0f, 0.0f, 0.0f);
    for (ulong k = 0; k < n; ++k) {
        ulong j = base + k * inner;
        acc = reduce_detail::tf_add(acc, df::make(in_hi[j], in_lo[j]));
        df64 r = reduce_detail::tf_to_df64(acc);
        out_hi[j] = r.hi;
        out_lo[j] = r.lo;
    }
}

// ---------------------------------------------------------------------------
// sf64 twins (binary64 bit patterns in one `long` buffer; correctly rounded
// per operation). Assumed sf64_core.h API: struct sf64 { ulong bits; },
// sf::add, sf::mul, sf::maximum, sf::minimum (NaN-propagating), sf::fmax,
// sf::fmin (NaN-ignoring), sf::is_nan, sf::gt, sf::lt.
// ---------------------------------------------------------------------------
#ifdef OPTILAND_SF64_CORE_H

namespace reduce_detail {

inline sf64 sf_from_bits(ulong b) { sf64 v; v.bits = b; return v; }

struct sf64_src {
    device const long* bits;
    inline sf64 at(ulong i) const { return sf_from_bits((ulong)bits[i]); }
};
struct sf64_dst {
    device long* bits;
    inline void put(ulong i, sf64 v) const { bits[i] = (long)v.bits; }
};

struct sum_sf64_op {
    typedef sf64 value_type;
    static inline sf64 identity() { return sf_from_bits(0ul); }
    static inline sf64 map(sf64 x) { return x; }
    static inline sf64 combine(sf64 a, sf64 b) { return sf::add(a, b); }
};
struct nansum_sf64_op {
    typedef sf64 value_type;
    static inline sf64 identity() { return sf_from_bits(0ul); }
    static inline sf64 map(sf64 x) { return sf::is_nan(x) ? sf_from_bits(0ul) : x; }
    static inline sf64 combine(sf64 a, sf64 b) { return sf::add(a, b); }
};
struct prod_sf64_op {
    typedef sf64 value_type;
    static inline sf64 identity() { return sf_from_bits(0x3FF0000000000000ul); }
    static inline sf64 map(sf64 x) { return x; }
    static inline sf64 combine(sf64 a, sf64 b) { return sf::mul(a, b); }
};
struct max_sf64_op {
    typedef sf64 value_type;
    static inline sf64 identity() { return sf_from_bits(0xFFF0000000000000ul); }  // -inf
    static inline sf64 map(sf64 x) { return x; }
    static inline sf64 combine(sf64 a, sf64 b) { return sf::maximum(a, b); }
};
struct min_sf64_op {
    typedef sf64 value_type;
    static inline sf64 identity() { return sf_from_bits(0x7FF0000000000000ul); }  // +inf
    static inline sf64 map(sf64 x) { return x; }
    static inline sf64 combine(sf64 a, sf64 b) { return sf::minimum(a, b); }
};
struct nanmax_sf64_op {
    typedef sf64 value_type;
    static inline sf64 identity() { return sf_from_bits(0x7FF8000000000000ul); }  // NaN
    static inline sf64 map(sf64 x) { return x; }
    static inline sf64 combine(sf64 a, sf64 b) { return sf::fmax(a, b); }
};
struct nanmin_sf64_op {
    typedef sf64 value_type;
    static inline sf64 identity() { return sf_from_bits(0x7FF8000000000000ul); }  // NaN
    static inline sf64 map(sf64 x) { return x; }
    static inline sf64 combine(sf64 a, sf64 b) { return sf::fmin(a, b); }
};

struct sf64_arg {
    sf64 v;
    long idx;
};
inline sf64_arg make_sf_arg(sf64 v, long idx) { sf64_arg r; r.v = v; r.idx = idx; return r; }

template <bool IS_MAX>
struct arg_sf64_op {
    typedef sf64_arg value_type;
    static inline sf64_arg identity() { return make_sf_arg(sf_from_bits(0x7FF8000000000000ul), -1L); }
    static inline bool better(sf64 cand, sf64 cur) {
        return IS_MAX ? sf::gt(cand, cur) : sf::lt(cand, cur);
    }
    static inline sf64_arg combine(sf64_arg a, sf64_arg b) {
        if (b.idx < 0) return a;
        if (a.idx < 0) return b;
        bool an = sf::is_nan(a.v);
        bool bn = sf::is_nan(b.v);
        if (an || bn) {
            if (an && bn) return (a.idx <= b.idx) ? a : b;
            return an ? a : b;
        }
        if (better(b.v, a.v)) return b;
        if (better(a.v, b.v)) return a;
        return (a.idx <= b.idx) ? a : b;
    }
};

struct sf64_arg_src {
    device const long* bits;
    device const long* idx;
    uint has_idx;
    inline sf64_arg at(ulong i, ulong k) const {
        return make_sf_arg(sf_from_bits((ulong)bits[i]), has_idx ? idx[i] : (long)k);
    }
};
struct sf64_arg_dst {
    device long* bits;
    device long* idx;
    inline void put(ulong i, sf64_arg a) const { bits[i] = (long)a.v.bits; idx[i] = a.idx; }
};

}  // namespace reduce_detail

#define OPTILAND_DEFINE_TREE_REDUCE_SF64(NAME, OP)                                          \
kernel void NAME(device const long* in_bits [[buffer(0)]],                                  \
                 device long* out_bits [[buffer(1)]],                                       \
                 constant uint& outer [[buffer(2)]],                                        \
                 constant uint& n [[buffer(3)]],                                            \
                 constant uint& inner [[buffer(4)]],                                        \
                 constant uint& groups [[buffer(5)]],                                       \
                 uint lid [[thread_position_in_threadgroup]],                               \
                 uint gid [[threadgroup_position_in_grid]],                                 \
                 uint tpg [[threads_per_threadgroup]]) {                                    \
    (void)outer;                                                                            \
    threadgroup sf64 buf[OPTILAND_REDUCE_TG];                                               \
    reduce_detail::sf64_src src; src.bits = in_bits;                                        \
    reduce_detail::sf64_dst dst; dst.bits = out_bits;                                       \
    reduce_detail::tree_reduce<OP>(src, dst, n, inner, groups, lid, gid, tpg, buf);         \
}

#define OPTILAND_DEFINE_LINE_REDUCE_SF64(NAME, OP)                                          \
kernel void NAME(device const long* in_bits [[buffer(0)]],                                  \
                 device long* out_bits [[buffer(1)]],                                       \
                 constant uint& outer [[buffer(2)]],                                        \
                 constant uint& n [[buffer(3)]],                                            \
                 constant uint& inner [[buffer(4)]],                                        \
                 uint tid [[thread_position_in_grid]]) {                                    \
    (void)outer;                                                                            \
    reduce_detail::sf64_src src; src.bits = in_bits;                                        \
    reduce_detail::sf64_dst dst; dst.bits = out_bits;                                       \
    reduce_detail::line_reduce<OP>(src, dst, n, inner, (ulong)tid);                         \
}

#define OPTILAND_DEFINE_TREE_ARG_SF64(NAME, OP)                                             \
kernel void NAME(device const long* in_bits [[buffer(0)]],                                  \
                 device const long* in_idx [[buffer(1)]],                                   \
                 device long* out_bits [[buffer(2)]],                                       \
                 device long* out_idx [[buffer(3)]],                                        \
                 constant uint& outer [[buffer(4)]],                                        \
                 constant uint& n [[buffer(5)]],                                            \
                 constant uint& inner [[buffer(6)]],                                        \
                 constant uint& groups [[buffer(7)]],                                       \
                 constant uint& has_idx [[buffer(8)]],                                      \
                 uint lid [[thread_position_in_threadgroup]],                               \
                 uint gid [[threadgroup_position_in_grid]],                                 \
                 uint tpg [[threads_per_threadgroup]]) {                                    \
    (void)outer;                                                                            \
    threadgroup reduce_detail::sf64_arg buf[OPTILAND_REDUCE_TG];                            \
    reduce_detail::sf64_arg_src src; src.bits = in_bits; src.idx = in_idx;                  \
    src.has_idx = has_idx;                                                                  \
    reduce_detail::sf64_arg_dst dst; dst.bits = out_bits; dst.idx = out_idx;                \
    reduce_detail::tree_arg_reduce<OP>(src, dst, n, inner, groups, lid, gid, tpg, buf);     \
}

#define OPTILAND_DEFINE_LINE_ARG_SF64(NAME, OP)                                             \
kernel void NAME(device const long* in_bits [[buffer(0)]],                                  \
                 device const long* in_idx [[buffer(1)]],                                   \
                 device long* out_bits [[buffer(2)]],                                       \
                 device long* out_idx [[buffer(3)]],                                        \
                 constant uint& outer [[buffer(4)]],                                        \
                 constant uint& n [[buffer(5)]],                                            \
                 constant uint& inner [[buffer(6)]],                                        \
                 constant uint& has_idx [[buffer(7)]],                                      \
                 uint tid [[thread_position_in_grid]]) {                                    \
    (void)outer;                                                                            \
    reduce_detail::sf64_arg_src src; src.bits = in_bits; src.idx = in_idx;                  \
    src.has_idx = has_idx;                                                                  \
    reduce_detail::sf64_arg_dst dst; dst.bits = out_bits; dst.idx = out_idx;                \
    reduce_detail::line_arg_reduce<OP>(src, dst, n, inner, (ulong)tid);                     \
}

OPTILAND_DEFINE_TREE_REDUCE_SF64(sum_sf64, reduce_detail::sum_sf64_op)
OPTILAND_DEFINE_TREE_REDUCE_SF64(nansum_sf64, reduce_detail::nansum_sf64_op)
OPTILAND_DEFINE_TREE_REDUCE_SF64(prod_sf64, reduce_detail::prod_sf64_op)
OPTILAND_DEFINE_TREE_REDUCE_SF64(max_sf64, reduce_detail::max_sf64_op)
OPTILAND_DEFINE_TREE_REDUCE_SF64(min_sf64, reduce_detail::min_sf64_op)
OPTILAND_DEFINE_TREE_REDUCE_SF64(nanmax_sf64, reduce_detail::nanmax_sf64_op)
OPTILAND_DEFINE_TREE_REDUCE_SF64(nanmin_sf64, reduce_detail::nanmin_sf64_op)

OPTILAND_DEFINE_LINE_REDUCE_SF64(sum_seq_sf64, reduce_detail::sum_sf64_op)
OPTILAND_DEFINE_LINE_REDUCE_SF64(nansum_seq_sf64, reduce_detail::nansum_sf64_op)
OPTILAND_DEFINE_LINE_REDUCE_SF64(prod_seq_sf64, reduce_detail::prod_sf64_op)
OPTILAND_DEFINE_LINE_REDUCE_SF64(max_seq_sf64, reduce_detail::max_sf64_op)
OPTILAND_DEFINE_LINE_REDUCE_SF64(min_seq_sf64, reduce_detail::min_sf64_op)
OPTILAND_DEFINE_LINE_REDUCE_SF64(nanmax_seq_sf64, reduce_detail::nanmax_sf64_op)
OPTILAND_DEFINE_LINE_REDUCE_SF64(nanmin_seq_sf64, reduce_detail::nanmin_sf64_op)

OPTILAND_DEFINE_TREE_ARG_SF64(argmax_sf64, reduce_detail::arg_sf64_op<true>)
OPTILAND_DEFINE_TREE_ARG_SF64(argmin_sf64, reduce_detail::arg_sf64_op<false>)
OPTILAND_DEFINE_LINE_ARG_SF64(argmax_seq_sf64, reduce_detail::arg_sf64_op<true>)
OPTILAND_DEFINE_LINE_ARG_SF64(argmin_seq_sf64, reduce_detail::arg_sf64_op<false>)

// Sequential cumsum with plain sf::add (each prefix correctly rounded relative
// to the previous prefix, i.e. identical to a sequential numpy cumsum).
kernel void cumsum_sf64(device const long* in_bits [[buffer(0)]],
                        device long* out_bits [[buffer(1)]],
                        constant uint& outer [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        constant uint& inner [[buffer(4)]],
                        uint tid [[thread_position_in_grid]]) {
    (void)outer;
    ulong o = (ulong)tid / inner;
    ulong i = (ulong)tid - o * inner;
    ulong base = o * (ulong)n * inner + i;
    sf64 acc = reduce_detail::sf_from_bits(0ul);
    for (ulong k = 0; k < n; ++k) {
        ulong j = base + k * inner;
        acc = sf::add(acc, reduce_detail::sf_from_bits((ulong)in_bits[j]));
        out_bits[j] = (long)acc.bits;
    }
}

#endif  // OPTILAND_SF64_CORE_H

#endif  // OPTILAND_REDUCE_METAL
