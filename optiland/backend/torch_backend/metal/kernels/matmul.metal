// matmul.metal — batched matrix multiply and dot product for emulated float64.
//
//   matmul_df64   C[b] = A[b] @ B[b] with A [batch, M, K], B [batch, K, N] and
//                 C [batch, M, N], all contiguous (hi and lo as separate float32
//                 buffers). One thread per output element (b, m, n): the dot
//                 product over k is accumulated as df::add(acc, df::mul(a, b)),
//                 i.e. every product carries its lo*lo term (<5u^2) and every
//                 addition is the 3u^2 AccurateDWPlusDW, so the forward error is
//                 bounded by ~8u^2 * sum_k |a_mk b_kn| independent of the
//                 accumulation order and with no length limit on K.
//                 Broadcasting of the batch dimension, transposes and strides
//                 are the host's job (materialize contiguous operands). M or N
//                 may be 1 (matrix-vector / vector-matrix), batch may be 1.
//                 Launch: threads=[batch*M*N, 1, 1].
//   dot_df64      x . y for two length-n vectors, tree-reduced over `groups`
//                 threadgroups of OPTILAND_MATMUL_TG (256) threads; writes
//                 `groups` df64 partials that the host finishes with
//                 sum_df64 (reduce.metal, outer = inner = 1, n = groups) or
//                 reads directly when groups == 1.
//                 Launch: threads=[groups*256, 1, 1], group_size=[256, 1, 1].
//
// The sf64 twins (compiled when sf64_core.h precedes this file) use one
// `long` buffer of binary64 bit patterns per operand and correctly rounded
// sf::mul / sf::add per operation, sequential over k (bit-identical to a
// sequential double-precision loop; numpy/BLAS use blocked orders, so
// bit-exact agreement with them is not expected).
//
// v1 is the straightforward one-thread-per-output kernel (correctness first);
// a 16x16 threadgroup-tiled variant can be added later without changing the
// host interface.
//
// Must be amalgamated after df64_core.h (and sf64_core.h for the sf64 twins).
// Part of Optiland-Metal (MIT).

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

#ifndef OPTILAND_MATMUL_METAL
#define OPTILAND_MATMUL_METAL

#ifndef OPTILAND_DF64_CORE_H
#error "matmul.metal must be amalgamated after df64_core.h"
#endif

#define OPTILAND_MATMUL_TG 256u

kernel void matmul_df64(device const float* a_hi [[buffer(0)]],
                        device const float* a_lo [[buffer(1)]],
                        device const float* b_hi [[buffer(2)]],
                        device const float* b_lo [[buffer(3)]],
                        device float* c_hi [[buffer(4)]],
                        device float* c_lo [[buffer(5)]],
                        constant uint& batch [[buffer(6)]],
                        constant uint& M [[buffer(7)]],
                        constant uint& K [[buffer(8)]],
                        constant uint& N [[buffer(9)]],
                        uint tid [[thread_position_in_grid]]) {
    (void)batch;
    ulong mn = (ulong)M * N;
    ulong b = (ulong)tid / mn;
    ulong r = (ulong)tid - b * mn;
    ulong m = r / N;
    ulong n = r - m * N;
    ulong a_base = (b * M + m) * (ulong)K;   // A[b, m, :]
    ulong b_base = b * (ulong)K * N + n;     // B[b, :, n]
    df64 acc = df::zero();
    for (ulong k = 0; k < K; ++k) {
        df64 av = df::make(a_hi[a_base + k], a_lo[a_base + k]);
        ulong bj = b_base + k * N;
        df64 bv = df::make(b_hi[bj], b_lo[bj]);
        acc = df::add(acc, df::mul(av, bv));
    }
    c_hi[tid] = acc.hi;
    c_lo[tid] = acc.lo;
}

kernel void dot_df64(device const float* x_hi [[buffer(0)]],
                     device const float* x_lo [[buffer(1)]],
                     device const float* y_hi [[buffer(2)]],
                     device const float* y_lo [[buffer(3)]],
                     device float* out_hi [[buffer(4)]],
                     device float* out_lo [[buffer(5)]],
                     constant uint& n [[buffer(6)]],
                     constant uint& groups [[buffer(7)]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint gid [[threadgroup_position_in_grid]],
                     uint tpg [[threads_per_threadgroup]]) {
    threadgroup df64 buf[OPTILAND_MATMUL_TG];
    df64 acc = df::zero();
    for (ulong k = (ulong)gid * tpg + lid; k < n; k += (ulong)groups * tpg) {
        df64 p = df::mul(df::make(x_hi[k], x_lo[k]), df::make(y_hi[k], y_lo[k]));
        acc = df::add(acc, p);
    }
    buf[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg >> 1; s > 0; s >>= 1) {
        if (lid < s) buf[lid] = df::add(buf[lid], buf[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        out_hi[gid] = buf[0].hi;
        out_lo[gid] = buf[0].lo;
    }
}

// ---------------------------------------------------------------------------
// sf64 twins. Assumed sf64_core.h API: struct sf64 { ulong bits; }, sf::add,
// sf::mul.
// ---------------------------------------------------------------------------
#ifdef OPTILAND_SF64_CORE_H

namespace matmul_detail {
inline sf64 sf_from_bits(ulong b) { sf64 v; v.bits = b; return v; }
}  // namespace matmul_detail

kernel void matmul_sf64(device const long* a_bits [[buffer(0)]],
                        device const long* b_bits [[buffer(1)]],
                        device long* c_bits [[buffer(2)]],
                        constant uint& batch [[buffer(3)]],
                        constant uint& M [[buffer(4)]],
                        constant uint& K [[buffer(5)]],
                        constant uint& N [[buffer(6)]],
                        uint tid [[thread_position_in_grid]]) {
    (void)batch;
    ulong mn = (ulong)M * N;
    ulong b = (ulong)tid / mn;
    ulong r = (ulong)tid - b * mn;
    ulong m = r / N;
    ulong n = r - m * N;
    ulong a_base = (b * M + m) * (ulong)K;
    ulong b_base = b * (ulong)K * N + n;
    sf64 acc = matmul_detail::sf_from_bits(0ul);
    for (ulong k = 0; k < K; ++k) {
        sf64 av = matmul_detail::sf_from_bits((ulong)a_bits[a_base + k]);
        sf64 bv = matmul_detail::sf_from_bits((ulong)b_bits[b_base + k * N]);
        acc = sf::add(acc, sf::mul(av, bv));
    }
    c_bits[tid] = (long)acc.bits;
}

kernel void dot_sf64(device const long* x_bits [[buffer(0)]],
                     device const long* y_bits [[buffer(1)]],
                     device long* out_bits [[buffer(2)]],
                     constant uint& n [[buffer(3)]],
                     constant uint& groups [[buffer(4)]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint gid [[threadgroup_position_in_grid]],
                     uint tpg [[threads_per_threadgroup]]) {
    threadgroup sf64 buf[OPTILAND_MATMUL_TG];
    sf64 acc = matmul_detail::sf_from_bits(0ul);
    for (ulong k = (ulong)gid * tpg + lid; k < n; k += (ulong)groups * tpg) {
        sf64 p = sf::mul(matmul_detail::sf_from_bits((ulong)x_bits[k]),
                         matmul_detail::sf_from_bits((ulong)y_bits[k]));
        acc = sf::add(acc, p);
    }
    buf[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg >> 1; s > 0; s >>= 1) {
        if (lid < s) buf[lid] = sf::add(buf[lid], buf[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) out_bits[gid] = (long)buf[0].bits;
}

#endif  // OPTILAND_SF64_CORE_H

#endif  // OPTILAND_MATMUL_METAL
