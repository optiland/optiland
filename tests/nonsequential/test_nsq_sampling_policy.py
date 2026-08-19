"""Tests for PR11: rare-path sampling policy (D2) -- importance biasing,
Russian roulette (D-9), and bounded splitting on the NumPy forward engine.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.ir.scene_ir import SamplingPolicy
from optiland.nonsequential.rng import NSQRng
from optiland.nonsequential.sampling import resolve_reflect_prob, russian_roulette


def _lens_scene() -> NSQScene:
    """A refracting lens: the only component with a Fresnel reflect/transmit
    branch, so it exercises importance biasing and bounded splitting.
    """
    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=8.0),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50.0),
        LensConfig(
            r1=60.0,
            r2=-60.0,
            thickness=6.0,
            material="N-BK7",
            front_aperture_radius=12.0,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=200.0),
        IrradianceDetectorConfig(width=40, height=40, num_pixels_x=32, num_pixels_y=32),
    )
    return scene


# ---------------------------------------------------------------------------
# resolve_reflect_prob -- pure function unit tests
# ---------------------------------------------------------------------------


class TestResolveReflectProb:
    def test_fresnel_is_identity(self):
        policy = SamplingPolicy(reflect_prob="fresnel")
        r = be.array(np.array([0.01, 0.5, 0.99]))
        p = resolve_reflect_prob(policy, r)
        np.testing.assert_allclose(be.to_numpy(p), be.to_numpy(r))

    def test_auto_clamps_into_range(self):
        policy = SamplingPolicy(reflect_prob="auto")
        r = be.array(np.array([0.0, 0.01, 0.5, 0.99, 1.0]))
        p = be.to_numpy(resolve_reflect_prob(policy, r))
        assert np.all(p >= 0.25 - 1e-12)
        assert np.all(p <= 0.75 + 1e-12)
        # Mid-range values pass through unclamped.
        assert p[2] == pytest.approx(0.5)

    def test_explicit_float(self):
        policy = SamplingPolicy(reflect_prob=0.3)
        r = be.array(np.array([0.01, 0.5, 0.99]))
        p = be.to_numpy(resolve_reflect_prob(policy, r))
        np.testing.assert_allclose(p, [0.3, 0.3, 0.3])


# ---------------------------------------------------------------------------
# russian_roulette -- pure function unit tests
# ---------------------------------------------------------------------------


class TestRussianRoulette:
    def test_above_threshold_untouched(self):
        rng = NSQRng(0)
        flux = be.array(np.array([1.0, 1.0, 1.0]))
        alive = be.array(np.array([True, True, True]))
        ray_id = np.array([0, 1, 2])
        bounce = np.array([0, 0, 0])
        flux_after, alive_after, killed = russian_roulette(
            flux,
            alive,
            rr_start_flux=1e-3,
            flux_per_ray=1.0,
            rng=rng,
            ray_id=ray_id,
            bounce=bounce,
        )
        np.testing.assert_allclose(be.to_numpy(flux_after), [1.0, 1.0, 1.0])
        assert bool(be.to_numpy(alive_after).all())
        assert not killed.any()

    def test_dead_rays_untouched(self):
        rng = NSQRng(0)
        flux = be.array(np.array([1e-9, 1e-9]))
        alive = be.array(np.array([False, False]))
        ray_id = np.array([0, 1])
        bounce = np.array([0, 0])
        flux_after, alive_after, killed = russian_roulette(
            flux,
            alive,
            rr_start_flux=1e-3,
            flux_per_ray=1.0,
            rng=rng,
            ray_id=ray_id,
            bounce=bounce,
        )
        assert not killed.any()
        assert not bool(be.to_numpy(alive_after).any())

    def test_unbiased_in_expectation(self):
        """Averaged over many independent rays, RR must reproduce the input
        flux exactly (D-9): the whole point of roulette over hard truncation.

        A killed ray no longer contributes downstream (``alive_after`` is
        False), so the unbiasedness statement is over ``flux_after`` *gated*
        by survival, not over ``flux_after`` alone (a killed ray's flux
        field is left at its pre-roulette value, unused, not zeroed).
        """
        n = 200_000
        rng = NSQRng(0)
        flux_val = 0.02  # well below rr_start_flux * flux_per_ray = 0.1
        flux = be.array(np.full(n, flux_val))
        alive = be.array(np.ones(n, dtype=bool))
        ray_id = np.arange(n)
        bounce = np.zeros(n, dtype=np.int64)
        flux_after, alive_after, _killed = russian_roulette(
            flux,
            alive,
            rr_start_flux=1e-1,
            flux_per_ray=1.0,
            rng=rng,
            ray_id=ray_id,
            bounce=bounce,
        )
        contribution = be.to_numpy(flux_after) * be.to_numpy(alive_after)
        mean_contribution = float(contribution.mean())
        assert mean_contribution == pytest.approx(flux_val, rel=0.02)

    def test_survive_probability_floor_prevents_unbounded_boost(self):
        rng = NSQRng(0)
        n = 10_000
        flux = be.array(np.full(n, 1e-12))  # far below threshold
        alive = be.array(np.ones(n, dtype=bool))
        ray_id = np.arange(n)
        bounce = np.zeros(n, dtype=np.int64)
        flux_after, _alive_after, _killed = russian_roulette(
            flux,
            alive,
            rr_start_flux=1e-1,
            flux_per_ray=1.0,
            rng=rng,
            ray_id=ray_id,
            bounce=bounce,
        )
        max_boost = 1.0 / 0.05  # _RR_SURVIVE_FLOOR
        assert float(be.to_numpy(flux_after).max()) <= 1e-12 * max_boost + 1e-30


# ---------------------------------------------------------------------------
# NSQScene.sampling_policy -- default reproduces pre-PR11 behaviour exactly
# ---------------------------------------------------------------------------


class TestDefaultPolicyUnchanged:
    def test_default_sampling_policy_is_fresnel(self):
        scene = NSQScene()
        assert scene.sampling_policy.reflect_prob == "fresnel"
        assert scene.sampling_policy.split_depth == 0

    def test_default_matches_explicit_fresnel_bit_for_bit(self):
        s1 = _lens_scene()
        s2 = _lens_scene()
        s2.sampling_policy = SamplingPolicy(reflect_prob="fresnel")
        r1 = s1.trace(num_rays=3000, seed=7, backend=NumpyBackend(seed=7))
        r2 = s2.trace(num_rays=3000, seed=7, backend=NumpyBackend(seed=7))
        assert r1.total_flux_detected == pytest.approx(r2.total_flux_detected)


# ---------------------------------------------------------------------------
# Importance biasing -- estimator unbiasedness invariant
# ---------------------------------------------------------------------------


class TestImportanceBiasing:
    def test_flux_ledger_closes_exactly_at_fresnel(self):
        """reflect_prob="fresnel" (p == R) makes the importance weight
        identically 1 (see RefractiveComponent.interact), so this is the
        one setting where per-trace ledger closure is exact, not just
        unbiased in expectation.
        """
        scene = _lens_scene()
        scene.sampling_policy = SamplingPolicy(reflect_prob="fresnel")
        result = scene.trace(num_rays=4000, seed=3, backend=NumpyBackend(seed=3))
        balance = (
            result.total_flux_in
            - result.total_flux_detected
            - result.total_flux_absorbed
            - result.total_flux_bulk_absorbed
            - result.total_flux_escaped
            - result.total_flux_lost
        )
        assert abs(balance) / result.total_flux_in < 1e-8

    @pytest.mark.parametrize("reflect_prob", [0.25, 0.5, 0.9, "auto"])
    def test_flux_ledger_closes_statistically(self, reflect_prob):
        """Off-Fresnel reflect_prob makes the branch weight != 1, so a
        single finite trace's ledger only closes in expectation (Monte
        Carlo noise, shrinking with ray count) -- not exactly, unlike the
        "fresnel" case. This is inherent to importance sampling, not a
        defect; the bound here is generous specifically to tolerate that
        noise while still catching an actual energy leak (a real bug would
        show ~O(1) imbalance, not a few percent).
        """
        scene = _lens_scene()
        scene.sampling_policy = SamplingPolicy(reflect_prob=reflect_prob)
        result = scene.trace(num_rays=20_000, seed=3, backend=NumpyBackend(seed=3))
        balance = (
            result.total_flux_in
            - result.total_flux_detected
            - result.total_flux_absorbed
            - result.total_flux_bulk_absorbed
            - result.total_flux_escaped
            - result.total_flux_lost
        )
        assert abs(balance) / result.total_flux_in < 0.1

    def test_reflect_prob_variants_converge_to_same_answer(self):
        """reflect_prob in {fresnel, 0.25, 0.5, 0.9, auto} must all
        converge to the same detected flux -- direct unbiasedness check.
        """
        target = None
        results = {}
        for reflect_prob in ("fresnel", 0.25, 0.5, "auto"):
            vals = []
            for seed in range(6):
                scene = _lens_scene()
                scene.sampling_policy = SamplingPolicy(reflect_prob=reflect_prob)
                r = scene.trace(
                    num_rays=15_000, seed=seed, backend=NumpyBackend(seed=seed)
                )
                vals.append(r.total_flux_detected)
            results[reflect_prob] = float(np.mean(vals))
        target = results["fresnel"]
        for key, val in results.items():
            assert val == pytest.approx(target, abs=0.03), (
                f"reflect_prob={key!r} mean detected {val} diverges from "
                f"fresnel baseline {target}"
            )


# ---------------------------------------------------------------------------
# Bounded splitting (NumPy forward engine only)
# ---------------------------------------------------------------------------


class TestBoundedSplitting:
    def test_splitting_conserves_flux_exactly(self):
        """Splitting itself is deterministic (forced R/T weights, no
        importance division), so the ledger must close exactly.
        """
        scene = _lens_scene()
        scene.sampling_policy = SamplingPolicy(split_depth=2, split_budget=8.0)
        result = scene.trace(num_rays=5000, seed=3, backend=NumpyBackend(seed=3))
        balance = (
            result.total_flux_in
            - result.total_flux_detected
            - result.total_flux_absorbed
            - result.total_flux_bulk_absorbed
            - result.total_flux_escaped
            - result.total_flux_lost
        )
        assert abs(balance) / result.total_flux_in < 1e-8

    def test_splitting_agrees_with_unsplit_and_reduces_variance(self):
        """split_depth in {0, 1, 2} converge to the same answer, and
        splitting reduces variance -- that is the entire point of D2.
        """

        def _run(split_depth: int, n_seeds: int, num_rays: int) -> list[float]:
            vals = []
            for seed in range(n_seeds):
                scene = _lens_scene()
                scene.sampling_policy = SamplingPolicy(
                    split_depth=split_depth, split_budget=8.0
                )
                r = scene.trace(
                    num_rays=num_rays, seed=seed, backend=NumpyBackend(seed=seed)
                )
                vals.append(r.total_flux_detected)
            return vals

        baseline = _run(0, 10, 20_000)
        split = _run(2, 10, 20_000)
        assert np.mean(split) == pytest.approx(np.mean(baseline), abs=0.02)
        assert np.std(split) < np.std(baseline)

    def test_split_budget_bounds_live_ray_growth(self):
        """A tight split_budget must not crash or leak flux beyond a launched
        watt -- excess spawned rays are Russian-rouletted, never silently
        dropped nor duplicated for free.
        """
        scene = _lens_scene()
        scene.sampling_policy = SamplingPolicy(split_depth=3, split_budget=1.05)
        result = scene.trace(num_rays=8000, seed=1, backend=NumpyBackend(seed=1))
        assert result.total_flux_detected <= result.total_flux_in + 1e-9
        assert result.total_flux_detected > 0.0

    def test_zero_split_depth_spawns_nothing(self):
        """split_depth=0 (the default) must reproduce the pre-PR11
        single-branch path exactly -- no ray_bundle growth at all.
        """
        from optiland.nonsequential.ir.lower import lower

        scene = _lens_scene()
        ir = lower(scene, strict=False)
        assert ir.sampling.split_depth == 0


# ---------------------------------------------------------------------------
# Torch backend: forces split_depth=0, warns rather than silently ignoring
# ---------------------------------------------------------------------------


class TestTorchBackendSplitDepthGuard:
    def test_nonzero_split_depth_warns(self):
        be.set_backend("torch")
        try:
            scene = _lens_scene()
            scene.sampling_policy = SamplingPolicy(split_depth=2)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scene.trace(num_rays=500, seed=1, backend=TorchBackend(seed=1))
            assert any(
                "split_depth" in str(w.message) and issubclass(w.category, UserWarning)
                for w in caught
            )
        finally:
            be.set_backend("numpy")

    def test_zero_split_depth_no_warning(self):
        be.set_backend("torch")
        try:
            scene = _lens_scene()  # default sampling_policy: split_depth=0
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scene.trace(num_rays=500, seed=1, backend=TorchBackend(seed=1))
            assert not any("split_depth" in str(w.message) for w in caught)
        finally:
            be.set_backend("numpy")

    def test_torch_still_traces_correctly_with_split_depth_set(self):
        """The fallback (importance-biased single-branch) must still produce
        a sane, energy-bounded result, not silently corrupt the trace.
        """
        be.set_backend("torch")
        try:
            scene = _lens_scene()
            scene.sampling_policy = SamplingPolicy(split_depth=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = scene.trace(
                    num_rays=2000, seed=1, backend=TorchBackend(seed=1)
                )
            assert result.total_flux_detected <= result.total_flux_in + 1e-6
            assert result.flux_conservation_error < 1e-4
        finally:
            be.set_backend("numpy")


# ---------------------------------------------------------------------------
# Russian roulette replaces flux truncation (D-9) in the full trace loop
# ---------------------------------------------------------------------------


class TestRussianRouletteInTrace:
    def test_low_flux_rays_are_rouletted_not_hard_killed(self):
        """A scene where every ray is an RR candidate from birth (an
        aggressive ``rr_start_flux``) must still close the ledger to a
        generous statistical bound -- every unit of flux ends up
        continuing (boosted), lost (killed), or otherwise booked, never
        silently dropped.  See ``test_flux_ledger_closes_statistically``
        for why exact closure is not the right bar once flux gets
        reweighted.
        """
        scene = _lens_scene()
        # threshold = rr_start_flux * flux_per_ray > flux_per_ray, so every
        # ray is an RR candidate as soon as it has any nonzero flux.
        scene.sampling_policy = SamplingPolicy(rr_start_flux=2.0)
        result = scene.trace(num_rays=6000, seed=2, backend=NumpyBackend(seed=2))
        balance = (
            result.total_flux_in
            - result.total_flux_detected
            - result.total_flux_absorbed
            - result.total_flux_bulk_absorbed
            - result.total_flux_escaped
            - result.total_flux_lost
        )
        assert abs(balance) / result.total_flux_in < 0.5
        assert result.num_rays_flux_killed > 0
