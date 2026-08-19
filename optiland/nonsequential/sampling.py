"""Rare-path sampling helpers for Non-Sequential Raytracing.

Two independent mechanisms, both keyed off ``SceneIR.sampling``
(:class:`~optiland.nonsequential.ir.scene_ir.SamplingPolicy`):

- **Importance biasing** (:func:`resolve_reflect_prob`): generalises the
  existing Fresnel-branch estimator in
  :class:`~optiland.nonsequential.components.refractive.RefractiveComponent`
  so the reflect/transmit branch probability need not equal the Fresnel
  reflectance itself. Works identically on both backends and inside
  autograd -- it only changes which detached probability the branch is
  drawn from and the compensating weight, not the estimator's structure.
- **Russian roulette** (:func:`russian_roulette`): replaces the old biased
  hard kill below ``min_flux`` with an unbiased stochastic kill +
  boost, on both backends.

Bounded splitting (also part of D2/PR11) is NumPy-forward-engine-only and
lives in :mod:`optiland.nonsequential.ir.interpreter` and
:mod:`optiland.nonsequential.backends.array_backend`, since it requires
growing the live ray bundle -- a NumPy-only operation (the Torch backend
keeps a fixed-shape bundle for the autograd graph and forces
``split_depth=0``).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.nonsequential.rng import EventSlot

if TYPE_CHECKING:
    from optiland.nonsequential.ir.scene_ir import SamplingPolicy
    from optiland.nonsequential.rng import NSQRng

# Clamp range for SamplingPolicy.reflect_prob="auto" -- a starting point,
# not yet tuned against the ghost benchmarks in tests/nonsequential/validation/.
_AUTO_CLAMP_LO = 0.25
_AUTO_CLAMP_HI = 0.75

# Floor on the Russian-roulette survival probability, so a ray with flux
# arbitrarily close to zero cannot draw an arbitrarily large boost factor
# (1 / survive_prob) -- that would trade a biased-but-bounded estimator for
# an unbiased-but-unbounded-variance one. The floor keeps the estimator
# unbiased (still exactly flux / survive_prob in expectation) while bounding
# the worst-case boost to 1 / _RR_SURVIVE_FLOOR.
_RR_SURVIVE_FLOOR = 0.05


def resolve_reflect_prob(policy: SamplingPolicy, r_det) -> object:
    """Resolve the reflect-branch sampling probability for a Fresnel split.

    Args:
        policy: The scene's :class:`SamplingPolicy`.
        r_det: Detached Fresnel (or coating) reflectance, shape (N,).

    Returns:
        Backend array, shape (N,): the probability to draw the reflect
        branch from. ``"fresnel"`` reproduces today's behaviour (p = R)
        exactly; ``"auto"`` clamps R into ``[0.25, 0.75]`` so a highly
        transmissive *or* highly reflective surface still samples its rare
        branch often enough to resolve it; an explicit float uses that
        constant probability for every ray.
    """
    mode = policy.reflect_prob
    if mode == "fresnel":
        return r_det
    if mode == "auto":
        return be.clip(r_det, _AUTO_CLAMP_LO, _AUTO_CLAMP_HI)
    return be.ones_like(r_det) * float(mode)


def russian_roulette(
    flux,
    alive,
    rr_start_flux: float,
    flux_per_ray: float,
    rng: NSQRng,
    ray_id: np.ndarray,
    bounce: np.ndarray,
) -> tuple[object, object, object]:
    """Unbiased stochastic termination of low-flux rays.

    Replaces the old hard kill below ``min_flux`` -- which discards exactly
    the low-flux, multiply-scattered paths that make up stray light, biasing
    every stray-light estimate low -- with roulette: below
    ``rr_start_flux * flux_per_ray``, kill with probability ``1 -
    survive_prob`` and boost survivors' flux by ``1 / survive_prob``, so
    ``E[flux_after] == flux_before`` exactly, for every ray, independent of
    ``survive_prob``.

    Args:
        flux: Per-ray flux, shape (N,), backend array.
        alive: Per-ray alive mask, shape (N,), backend array.
        rr_start_flux: Roulette threshold, as a fraction of ``flux_per_ray``.
        flux_per_ray: Mean initial flux per launched ray (``total_flux_in /
            num_rays``), used to convert ``rr_start_flux`` to an absolute
            flux threshold.
        rng: Keyed PCG32 RNG.
        ray_id: Per-ray identifiers, shape (N,).
        bounce: Per-ray bounce index as of this event, shape (N,).

    Returns:
        ``(flux_after, alive_after, killed_mask)``: updated flux (boosted
        for survivors, unchanged for non-candidates), updated alive mask,
        and a NumPy bool mask of rays newly killed by roulette this call
        (for flux-loss bookkeeping -- always ~0 in expectation, unlike the
        old ``total_flux_lost`` truncation bias).
    """
    from optiland.backend.utils import to_numpy  # noqa: PLC0415

    threshold = rr_start_flux * flux_per_ray
    flux_np = to_numpy(flux)
    alive_np = to_numpy(alive).astype(bool)

    candidate_np = alive_np & (flux_np < threshold) & (flux_np > 0.0)
    if not candidate_np.any():
        killed_mask = np.zeros_like(alive_np)
        return flux, alive, killed_mask

    survive_prob_np = np.clip(flux_np / max(threshold, 1e-300), _RR_SURVIVE_FLOOR, 1.0)
    u = rng.uniform(ray_id, bounce, EventSlot.RR)
    survive_np = u < survive_prob_np
    killed_mask = candidate_np & ~survive_np

    # Boost only the surviving candidates -- a killed ray's flux is left
    # untouched (it is immediately excluded via alive_after=False and, on
    # the NumPy backend, compacted away next bounce; leaving it at its
    # pre-roulette value rather than a discarded boosted one avoids a
    # meaningless number sitting in a dead ray's flux field).
    boost_mask_np = candidate_np & survive_np
    boost_np = np.where(boost_mask_np, 1.0 / survive_prob_np, 1.0)
    boost = be.array(boost_np)
    boost_mask = be.array(boost_mask_np)
    flux_after = be.where(boost_mask, flux * boost, flux)

    killed = be.array(killed_mask)
    alive_after = alive & ~killed

    return flux_after, alive_after, killed_mask
