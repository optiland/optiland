"""Smoke tests for the NSQ performance benchmark harness (PR18).

Runs the harness at trivial scale to catch import/logic regressions --
this is not a throughput assertion (that would make CI flaky and
hardware-dependent), just "the harness still produces a sane record".

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from benchmarks.nonsequential.harness import (
    run_once,
    sweep_depth,
    sweep_ray_count,
    sweep_surface_count,
)
from benchmarks.nonsequential.scenes import (
    cavity_scene,
    lens_scene,
    surface_count_scene,
)


@pytest.fixture(autouse=True)
def _restore_backend():
    original = be.get_backend()
    yield
    be.set_backend(original)


class TestScenes:
    def test_surface_count_scene_has_requested_decoys(self):
        scene = surface_count_scene(5)
        # 1 real mirror-free path -> just the decoys, no other components.
        assert len(scene.surfaces) == 5

    def test_cavity_scene_has_two_mirrors(self):
        scene = cavity_scene()
        assert len(scene.surfaces) == 2

    def test_lens_scene_has_lens_surfaces(self):
        scene = lens_scene()
        assert len(scene.surfaces) >= 2


class TestRunOnce:
    def test_numpy_backend_produces_sane_record(self):
        scene = lens_scene()
        record = run_once("rays", scene, num_rays=200, backend_name="numpy")

        assert record.backend == "numpy"
        assert record.num_rays == 200
        assert record.num_surfaces == len(scene.surfaces)
        assert record.trace_time_sec > 0
        assert record.rays_per_sec > 0

    def test_torch_backend_produces_sane_record(self):
        pytest.importorskip("torch")
        scene = lens_scene()
        record = run_once("rays", scene, num_rays=200, backend_name="torch")

        assert record.backend == "torch"
        assert record.rays_per_sec > 0

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            run_once("rays", lens_scene(), num_rays=10, backend_name="bogus")

    def test_restores_original_active_backend(self):
        be.set_backend("numpy")
        run_once("rays", lens_scene(), num_rays=10, backend_name="numpy")
        assert be.get_backend() == "numpy"


class TestSweeps:
    def test_sweep_surface_count(self):
        records = sweep_surface_count([0, 2], num_rays=200, backend_names=["numpy"])
        assert [r.num_surfaces for r in records] == [0, 2]

    def test_sweep_ray_count(self):
        records = sweep_ray_count([100, 200], backend_names=["numpy"])
        assert [r.num_rays for r in records] == [100, 200]

    def test_sweep_depth(self):
        records = sweep_depth([2, 4], num_rays=200, backend_names=["numpy"])
        assert [r.max_depth for r in records] == [2, 4]
