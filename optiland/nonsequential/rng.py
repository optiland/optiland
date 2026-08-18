"""Counter-based PCG32 RNG for Non-Sequential Raytracing.

Every stochastic decision in the NSQ engine is a pure function of a key
``(seed, ray_id, bounce, event_slot[, offset])`` -- there is no shared
mutable stream. This is what makes results bit-identical across
``batch_size``, across NumPy compaction vs. Torch's fixed-shape bundles, and
across any two conforming backends: a ray's random numbers depend only on
its own identity, never on which other rays happen to be alive in the same
batch or in what order components were visited.

Algorithm
----------------
This is the standard O'Neill PCG32 (XSH-RR 64/32), the same generator used
by Mitsuba 3 (``pcg32.h``) and satisfied by the per-launch-index
counter-based PRNGs conventional in OptiX kernels:

    state_{k+1} = state_k * MULT + inc      (mod 2**64)
    output_k    = xsh_rr(state_k)           (32-bit)

``inc`` (the odd-valued stream selector) is derived from ``(seed, ray_id,
event_slot)`` via SplitMix64, so every ray gets its own independent stream
per event slot. ``bounce`` (plus an optional ``offset`` for multi-draw
slots such as rejection sampling) selects *which* output in that stream via
PCG32's jump-ahead identity -- a closed-form function of the LCG step count,
computed by the standard doubling algorithm in O(64) fixed iterations. This
is what "counter derived arithmetically rather than by stateful advance"
means: computing output number ``k`` never requires having computed outputs
``0..k-1`` first, and no RNG object needs to persist state between calls.

Honest scope of the guarantee: the *random-number stream* per
``(ray_id, bounce, event_slot)`` is bit-identical everywhere this module is
used. Final float *results* are not guaranteed bit-identical across
NumPy/Torch/CPU/GPU, because floating-point summation order and
transcendental implementations differ -- only the random decisions and the
code path they select are guaranteed identical.

Kramer Harrison, 2026
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from optiland.backend.utils import to_numpy

# PCG32 default multiplier (O'Neill, "PCG: A Family of Simple Fast
# Space-Efficient Statistically Good Algorithms for Random Number
# Generation", 2014).
_PCG_MULT = np.uint64(6364136223846793005)

# SplitMix64 (Steele, Lea, Flood 2014) constants, used only to derive
# well-mixed PCG32 seed/stream values from our integer keys -- not part of
# the PCG32 output path itself.
_SM64_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SM64_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_SM64_MIX2 = np.uint64(0x94D049BB133111EB)

_U64_0 = np.uint64(0)
_U64_1 = np.uint64(1)
_U32_31 = np.uint32(31)

_TWO_POW_32 = 4294967296.0


class EventSlot(IntEnum):
    """Discriminates independent PCG32 streams within one (ray, bounce).

    Every stochastic decision draws from its own slot, so adding, removing,
    or reordering an unrelated decision can never perturb another
    decision's stream (defect D-8: a shared, position-dependent stream).
    """

    SOURCE_U1 = 0
    SOURCE_U2 = 1
    SOURCE_U3 = 2
    SOURCE_U4 = 3
    SOURCE_WAVELENGTH = 4
    FRESNEL_BRANCH = 5
    SCATTER_BRANCH = 6
    BSDF_U1 = 7
    BSDF_U2 = 8
    RR = 9
    BSDF_LOBE_BRANCH = 10
    PATH_SAMPLE = 11


def _splitmix64(z: np.ndarray) -> np.ndarray:
    """Well-mixed 64-bit hash (SplitMix64 finalizer), vectorized.

    Args:
        z: uint64 array of arbitrary raw key material.

    Returns:
        uint64 array of well-mixed values, same shape as ``z``.
    """
    z = z + _SM64_GAMMA
    z = (z ^ (z >> np.uint64(30))) * _SM64_MIX1
    z = (z ^ (z >> np.uint64(27))) * _SM64_MIX2
    return z ^ (z >> np.uint64(31))


def _pcg32_seed(
    initstate: np.ndarray, initseq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """PCG32 ``srandom_r``: derive (state, inc) from (initstate, initseq).

    Args:
        initstate: uint64 array.
        initseq: uint64 array, same shape as ``initstate``.

    Returns:
        ``(state, inc)``, each a uint64 array of the same shape.
    """
    inc = (initseq << _U64_1) | _U64_1
    state = _U64_0 * _PCG_MULT + inc
    state = state + initstate
    state = state * _PCG_MULT + inc
    return state, inc


def _pcg32_advance(
    state: np.ndarray, delta: np.ndarray, mult: np.ndarray, inc: np.ndarray
) -> np.ndarray:
    """Closed-form ``state`` after ``delta`` LCG steps (PCG32 jump-ahead).

    Uses the standard doubling identity for ``state_k = mult^k * state_0 +
    inc * (mult^k - 1) / (mult - 1)`` in O(64) fixed vectorized iterations,
    so the result is a pure function of ``(state, delta)`` -- no sequential
    per-step state advance is needed.

    Args:
        state: uint64 array, the step-0 state.
        delta: uint64 array, number of LCG steps to advance.
        mult: uint64 array, LCG multiplier (broadcastable).
        inc: uint64 array, LCG increment (broadcastable).

    Returns:
        uint64 array: state after ``delta`` steps.
    """
    acc_mult = np.ones_like(mult)
    acc_plus = np.zeros_like(mult)
    cur_mult = mult.copy()
    cur_plus = inc.copy()
    d = delta.copy()
    for _ in range(64):
        bit = (d & _U64_1).astype(bool)
        acc_mult = np.where(bit, acc_mult * cur_mult, acc_mult)
        acc_plus = np.where(bit, acc_plus * cur_mult + cur_plus, acc_plus)
        cur_plus = (cur_mult + _U64_1) * cur_plus
        cur_mult = cur_mult * cur_mult
        d = d >> _U64_1
    return acc_mult * state + acc_plus


def _pcg32_output(state: np.ndarray) -> np.ndarray:
    """PCG32 XSH-RR 64/32 output permutation.

    Args:
        state: uint64 array.

    Returns:
        uint32 array, same shape as ``state``.
    """
    xorshifted = (((state >> np.uint64(18)) ^ state) >> np.uint64(27)).astype(np.uint32)
    rot = (state >> np.uint64(59)).astype(np.uint32)
    neg_rot = (np.uint32(0) - rot) & _U32_31
    return (xorshifted >> rot) | (xorshifted << neg_rot)


def pcg32_uint32(
    seed: int,
    ray_id: np.ndarray,
    bounce: np.ndarray,
    event_slot: int,
    offset: int = 0,
) -> np.ndarray:
    """Draw one PCG32 32-bit output per key, as a pure function of the key.

    Args:
        seed: Trace-level RNG seed.
        ray_id: Per-ray identifiers, shape (N,). Must be non-negative.
        bounce: Per-ray bounce/step index, shape (N,) or a scalar
            broadcastable to (N,). Must be non-negative.
        event_slot: Which independent stream within (ray_id, bounce) to
            draw from -- an :class:`EventSlot` value or plain int.
        offset: Extra step count added to ``bounce`` for multi-draw slots
            (e.g. successive attempts in a rejection sampler) that need a
            fresh, deterministic value without consuming a new event slot.

    Returns:
        uint32 array, shape (N,).
    """
    ray_id_np = np.asarray(to_numpy(ray_id))
    bounce_np = np.asarray(to_numpy(bounce))
    ray_id_u64 = ray_id_np.astype(np.uint64)
    bounce_u64, ray_id_u64 = np.broadcast_arrays(
        bounce_np.astype(np.uint64), ray_id_u64
    )
    bounce_u64 = bounce_u64.copy()
    ray_id_u64 = ray_id_u64.copy()

    seed_u64 = np.uint64(np.uint64(seed) & np.uint64(0xFFFFFFFFFFFFFFFF))
    slot_u64 = np.uint64(int(event_slot))

    # Modular (mod 2**64) wraparound is the intended arithmetic throughout
    # this module -- it is how the LCG and the mixing hashes are defined --
    # so overflow is not an error condition here.
    with np.errstate(over="ignore"):
        initstate = _splitmix64(np.full_like(ray_id_u64, seed_u64))
        # Mix ray_id and event_slot into the stream selector so every ray
        # gets an independent stream per slot; the golden-ratio odd constant
        # avoids low-bit correlation between adjacent ray ids.
        initseq = _splitmix64(
            ray_id_u64 * _SM64_GAMMA ^ (slot_u64 * np.uint64(0xC2B2AE3D27D4EB4F))
        )

        state0, inc = _pcg32_seed(initstate, initseq)
        mult = np.full_like(state0, _PCG_MULT)
        delta = bounce_u64 + np.uint64(offset)
        state_k = _pcg32_advance(state0, delta, mult, inc)
        return _pcg32_output(state_k)


def pcg32_uniform(
    seed: int,
    ray_id: np.ndarray,
    bounce: np.ndarray,
    event_slot: int,
    offset: int = 0,
) -> np.ndarray:
    """Draw one PCG32-derived uniform float per key, in [0, 1).

    Args:
        seed: Trace-level RNG seed.
        ray_id: Per-ray identifiers, shape (N,).
        bounce: Per-ray bounce/step index, shape (N,) or scalar.
        event_slot: :class:`EventSlot` value or plain int.
        offset: See :func:`pcg32_uint32`.

    Returns:
        float64 array in [0, 1), shape (N,).
    """
    bits = pcg32_uint32(seed, ray_id, bounce, event_slot, offset)
    return bits.astype(np.float64) / _TWO_POW_32


class NSQRng:
    """Keyed PCG32 RNG for one trace.

    Unlike ``numpy.random.Generator``, this carries no advancing internal
    state: every draw is a pure function of ``(seed, ray_id, bounce,
    event_slot)``, so the result never depends on ``batch_size``, on
    whether the NumPy backend has compacted dead rays out of the bundle, or
    on the order in which scene components were visited.

    Attributes:
        seed: Trace-level RNG seed (defaults to 0 if none was given, so a
            trace is always reproducible even when the user does not pass
            one explicitly).
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize NSQRng.

        Args:
            seed: RNG seed. ``None`` is normalized to 0 -- there is no
                notion of nondeterministic entropy here, since every draw
                must be reproducible from the key alone.
        """
        self.seed = 0 if seed is None else int(seed)

    def uniform(
        self,
        ray_id: np.ndarray,
        bounce: np.ndarray,
        event_slot: int,
        offset: int = 0,
    ) -> np.ndarray:
        """Draw one uniform float per ray, in [0, 1).

        Args:
            ray_id: Per-ray identifiers, shape (N,).
            bounce: Per-ray bounce/step index, shape (N,) or scalar.
            event_slot: :class:`EventSlot` value or plain int.
            offset: See :func:`pcg32_uint32`.

        Returns:
            float64 array in [0, 1), shape (N,).
        """
        return pcg32_uniform(self.seed, ray_id, bounce, event_slot, offset)
