"""Tests for PR12: vectorised columnar path recording (D-7).

Kramer Harrison, 2026
"""

from __future__ import annotations

import time

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
from optiland.nonsequential.path_recording import (
    _EVENT_DTYPE,
    ColumnarPathLog,
    PathRecorder,
    resolve_path_sample_mask,
)
from optiland.nonsequential.ray_bundle import NSQRayBundle


def _lens_scene() -> NSQScene:
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


def _bundle(n: int, ray_id: np.ndarray | None = None) -> NSQRayBundle:
    return NSQRayBundle(
        x=np.zeros(n),
        y=np.zeros(n),
        z=np.zeros(n),
        L=np.zeros(n),
        M=np.zeros(n),
        N=np.ones(n),
        flux=np.ones(n),
        wavelength=np.full(n, 0.55),
        n_current=np.ones(n),
        bounce=np.zeros(n, dtype=np.int32),
        alive=np.ones(n, dtype=bool),
        ray_id=ray_id if ray_id is not None else np.arange(n, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# resolve_path_sample_mask -- pure function unit tests
# ---------------------------------------------------------------------------


class TestResolvePathSampleMask:
    def test_false_selects_nothing(self):
        mask = resolve_path_sample_mask(np.arange(100), 100, False, seed=0)
        assert not mask.any()

    def test_true_selects_everything(self):
        mask = resolve_path_sample_mask(np.arange(100), 100, True, seed=0)
        assert mask.all()

    def test_int_ge_total_selects_everything(self):
        mask = resolve_path_sample_mask(np.arange(50), 50, 100, seed=0)
        assert mask.all()

    def test_int_selects_approximately_that_many(self):
        n_total = 200_000
        target = 1_000
        mask = resolve_path_sample_mask(np.arange(n_total), n_total, target, seed=0)
        # Bernoulli-threshold selection: exact count is not guaranteed, but
        # should be close for a target this size (relative std ~3%).
        assert abs(int(mask.sum()) - target) < target * 0.25

    def test_deterministic_for_fixed_seed(self):
        ray_id = np.arange(10_000)
        m1 = resolve_path_sample_mask(ray_id, 10_000, 500, seed=7)
        m2 = resolve_path_sample_mask(ray_id, 10_000, 500, seed=7)
        np.testing.assert_array_equal(m1, m2)

    def test_a_rays_selection_is_independent_of_batch_or_bundle_size(self):
        """A given ray_id's membership depends only on (seed, ray_id, N) --
        not on which other ray_ids happen to be queried alongside it, since
        the hash is a pure per-ray function.
        """
        full = resolve_path_sample_mask(np.arange(1000), 1000, 200, seed=3)
        # Query a subset of ids in a different grouping/order.
        subset_ids = np.array([5, 999, 42, 501, 7])
        subset_mask = resolve_path_sample_mask(subset_ids, 1000, 200, seed=3)
        np.testing.assert_array_equal(subset_mask, full[subset_ids])

    def test_different_seeds_select_different_subsets(self):
        ray_id = np.arange(10_000)
        m1 = resolve_path_sample_mask(ray_id, 10_000, 500, seed=1)
        m2 = resolve_path_sample_mask(ray_id, 10_000, 500, seed=2)
        assert not np.array_equal(m1, m2)

    def test_zero_total_rays_selects_nothing(self):
        mask = resolve_path_sample_mask(np.arange(0), 0, 10, seed=0)
        assert mask.size == 0


# ---------------------------------------------------------------------------
# ColumnarPathLog -- pure unit tests (no per-ray Python loop, single fancy
# -index writes, capacity growth, vectorised final assembly)
# ---------------------------------------------------------------------------


class TestColumnarPathLog:
    def test_empty_log_returns_none(self):
        log = ColumnarPathLog()
        assert log.to_events() is None

    def test_single_event_roundtrip(self):
        log = ColumnarPathLog(initial_capacity=2)
        rays = _bundle(3)
        rays.x = np.array([1.0, 2.0, 3.0])
        mask = np.array([False, True, True])
        log.log_event(1, mask, rays, None, "L1.front")
        events = log.to_events()
        assert events.dtype == _EVENT_DTYPE
        assert len(events) == 2
        np.testing.assert_array_equal(events["x"], [2.0, 3.0])
        assert set(events["component_name"]) == {"L1.front"}
        assert set(events["event_type"]) == {"hit"}

    def test_capacity_grows_beyond_initial(self):
        """Amortised-growth buffer: logging more rows than initial_capacity
        must not lose or corrupt any row (D-7 -- the whole point of
        preallocation is that growth is transparent to the caller).
        """
        log = ColumnarPathLog(initial_capacity=4)
        rays = _bundle(50)
        rays.x = np.arange(50, dtype=np.float64)
        mask = np.ones(50, dtype=bool)
        log.log_event(0, mask, rays, None, "birth")
        events = log.to_events()
        assert len(events) == 50
        np.testing.assert_array_equal(np.sort(events["x"]), np.arange(50))

    def test_t_offset_advances_position(self):
        log = ColumnarPathLog()
        rays = _bundle(2)
        rays.x = np.array([0.0, 0.0])
        rays.L = np.array([1.0, 1.0])
        mask = np.array([True, True])
        t_offset = np.array([5.0, 10.0])
        log.log_event(1, mask, rays, t_offset, "surf")
        events = log.to_events()
        np.testing.assert_allclose(np.sort(events["x"]), [5.0, 10.0])

    def test_multiple_names_get_distinct_codes(self):
        log = ColumnarPathLog()
        rays = _bundle(2)
        log.log_event(0, np.array([True, False]), rays, None, "source_0")
        log.log_event(2, np.array([False, True]), rays, None, "escaped")
        events = log.to_events()
        names = dict(zip(events["event_type"], events["component_name"], strict=True))
        assert names["birth"] == "source_0"
        assert names["death"] == "escaped"


# ---------------------------------------------------------------------------
# PathRecorder -- facade used by both backends
# ---------------------------------------------------------------------------


class TestPathRecorder:
    def test_disabled_is_a_no_op(self):
        rec = PathRecorder(False, num_rays_total=100, seed=0)
        rays = _bundle(5)
        rec.log_birth(rays, "S")
        rec.log_hits(rays, np.ones(5, dtype=bool), "L1")
        rec.log_deaths(rays, np.ones(5, dtype=bool), "escaped")
        assert rec.finalize() is None

    def test_only_sampled_rays_are_logged(self):
        n_total = 2000
        rec = PathRecorder(50, num_rays_total=n_total, seed=1)
        rays = _bundle(n_total)
        rec.log_birth(rays, "S")
        events = rec.finalize()["events"]
        recorded_ids = set(events["ray_id"].tolist())
        # Every recorded id must be in-sample under the same hash rule.
        full_mask = resolve_path_sample_mask(np.arange(n_total), n_total, 50, seed=1)
        expected_ids = set(np.where(full_mask)[0].tolist())
        assert recorded_ids == expected_ids


# ---------------------------------------------------------------------------
# End-to-end: NSQScene.trace(record_paths=...) contract
# ---------------------------------------------------------------------------


class TestRecordPathsContract:
    def test_false_yields_no_ray_paths(self):
        scene = _lens_scene()
        result = scene.trace(num_rays=500, seed=1, backend=NumpyBackend(seed=1))
        assert result.ray_paths is None

    def test_true_records_every_ray(self):
        scene = _lens_scene()
        result = scene.trace(
            num_rays=500, seed=1, record_paths=True, backend=NumpyBackend(seed=1)
        )
        events = result.ray_paths["events"]
        assert len(np.unique(events["ray_id"])) == 500

    def test_int_records_approximate_subset_deterministically(self):
        scene = _lens_scene()
        r1 = scene.trace(
            num_rays=50_000, seed=1, record_paths=200, backend=NumpyBackend(seed=1)
        )
        r2 = scene.trace(
            num_rays=50_000, seed=1, record_paths=200, backend=NumpyBackend(seed=1)
        )
        ids1 = np.unique(r1.ray_paths["events"]["ray_id"])
        ids2 = np.unique(r2.ray_paths["events"]["ray_id"])
        np.testing.assert_array_equal(ids1, ids2)
        assert abs(len(ids1) - 200) < 200 * 0.5

    def test_int_subset_independent_of_batch_size(self):
        scene = _lens_scene()
        r1 = scene.trace(
            num_rays=50_000,
            seed=1,
            record_paths=200,
            batch_size=16_384,
            backend=NumpyBackend(seed=1),
        )
        r2 = scene.trace(
            num_rays=50_000,
            seed=1,
            record_paths=200,
            batch_size=777,
            backend=NumpyBackend(seed=1),
        )
        ids1 = np.unique(r1.ray_paths["events"]["ray_id"])
        ids2 = np.unique(r2.ray_paths["events"]["ray_id"])
        np.testing.assert_array_equal(ids1, ids2)

    def test_events_dtype_matches_contract(self):
        """The structured-array format visualization consumes must not
        change -- only the internal accumulation strategy did (D-7).
        """
        scene = _lens_scene()
        result = scene.trace(
            num_rays=500, seed=1, record_paths=True, backend=NumpyBackend(seed=1)
        )
        assert result.ray_paths["events"].dtype == _EVENT_DTYPE

    def test_full_trace_stays_full_size_with_small_subset(self):
        """The whole point of record_paths: int -- a huge trace stays
        full-size and fast; only a bounded subset is recorded.
        """
        scene = _lens_scene()
        t0 = time.perf_counter()
        result = scene.trace(
            num_rays=200_000, seed=1, record_paths=500, backend=NumpyBackend(seed=1)
        )
        elapsed = time.perf_counter() - t0
        assert result.total_flux_detected > 0.0
        n_recorded = len(np.unique(result.ray_paths["events"]["ray_id"]))
        assert n_recorded < 200_000
        assert abs(n_recorded - 500) < 500 * 0.5
        # A generous ceiling -- this used to be minutes at this scale (D-7);
        # a regression back to the old per-event Python-dict loop would
        # blow well past this.
        assert elapsed < 20.0


# ---------------------------------------------------------------------------
# TorchBackend now records hit/death events too (parity fix, PR12)
# ---------------------------------------------------------------------------


class TestTorchBackendRecordsHitsAndDeaths:
    def test_torch_backend_records_hits_and_deaths(self):
        be.set_backend("torch")
        try:
            scene = _lens_scene()
            result = scene.trace(
                num_rays=500, seed=1, record_paths=True, backend=TorchBackend(seed=1)
            )
            events = result.ray_paths["events"]
            types, counts = np.unique(events["event_type"], return_counts=True)
            counts_by_type = dict(zip(types, counts, strict=True))
            assert counts_by_type.get("birth", 0) == 500
            assert counts_by_type.get("hit", 0) > 0
            assert counts_by_type.get("death", 0) > 0
        finally:
            be.set_backend("numpy")

    def test_torch_backend_int_subset(self):
        be.set_backend("torch")
        try:
            scene = _lens_scene()
            result = scene.trace(
                num_rays=5000,
                seed=1,
                record_paths=100,
                backend=TorchBackend(seed=1),
            )
            n_recorded = len(np.unique(result.ray_paths["events"]["ray_id"]))
            assert 0 < n_recorded < 5000
        finally:
            be.set_backend("numpy")


# ---------------------------------------------------------------------------
# Visualization consumers still work against the (unchanged) format
# ---------------------------------------------------------------------------


class TestVisualizationCompatibility:
    def test_paths_from_events_smoke(self):
        from optiland.nonsequential.visualization.rays import _paths_from_events

        scene = _lens_scene()
        result = scene.trace(
            num_rays=300, seed=1, record_paths=True, backend=NumpyBackend(seed=1)
        )
        paths = _paths_from_events(result.ray_paths["events"], num_rays=50)
        assert len(paths) > 0
        for p in paths:
            assert len(p) >= 2

    def test_nsq_rays_2d_plot_smoke(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from optiland.nonsequential.visualization.rays import NSQRays2D

        scene = _lens_scene()
        result = scene.trace(
            num_rays=300, seed=1, record_paths=True, backend=NumpyBackend(seed=1)
        )
        fig, ax = plt.subplots()
        NSQRays2D(scene).plot(ax, num_rays=20, ray_paths=result.ray_paths)
        plt.close(fig)


@pytest.mark.parametrize("record_paths", [False, True, 10])
def test_record_paths_types_all_traceable(record_paths):
    """Smoke test across the full bool|int contract in one parametrization."""
    scene = _lens_scene()
    result = scene.trace(
        num_rays=200, seed=1, record_paths=record_paths, backend=NumpyBackend(seed=1)
    )
    if record_paths:
        assert result.ray_paths is not None
    else:
        assert result.ray_paths is None
