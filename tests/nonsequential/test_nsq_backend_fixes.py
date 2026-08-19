"""Regression tests for NSQ v2 PR1 (quick-win defect fixes).

Covers D-6 (AABB attribute crash / hardcoded 100 mm escape extension) and
D-12 (ray budget can starve a source).

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.nonsequential._utils import distribute_ray_budget, estimate_bounding_scale
from optiland.nonsequential.components.geometry.base import AABB


class _FakeComponent:
    def __init__(self, box: AABB) -> None:
        self.bounding_box = box


class _FakeScene:
    def __init__(self, surfaces: list) -> None:
        self.surfaces = surfaces


class TestAABB:
    def test_min_max_properties(self):
        box = AABB(np.array([-1.0, -2.0, -3.0]), np.array([4.0, 5.0, 6.0]))
        assert box.xmin == -1.0
        assert box.xmax == 4.0
        assert box.ymin == -2.0
        assert box.ymax == 5.0
        assert box.zmin == -3.0
        assert box.zmax == 6.0


class TestEstimateBoundingScale:
    def test_scales_with_scene_extent(self):
        """D-6: escape distance must track scene scale, not a fixed 100 mm."""
        small = _FakeScene(
            [_FakeComponent(AABB(np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])))]
        )
        big = _FakeScene(
            [
                _FakeComponent(
                    AABB(np.array([0.0, 0.0, 0.0]), np.array([2000.0, 0.0, 0.0]))
                )
            ]
        )
        assert estimate_bounding_scale(big) == pytest.approx(2000.0)
        # Small scene falls back to the 100 mm floor rather than a tiny value.
        assert estimate_bounding_scale(small) == pytest.approx(100.0)

    def test_no_surfaces_falls_back(self):
        assert estimate_bounding_scale(_FakeScene([])) == 100.0


class TestDistributeRayBudget:
    def test_sums_to_total(self):
        fluxes = [1.0, 1.0, 1.0]
        counts = distribute_ray_budget(100, fluxes)
        assert sum(counts) == 100
        assert all(c >= 1 for c in counts)

    def test_many_low_flux_sources_do_not_starve_the_largest(self):
        """D-12: max(1, ...) applied to all-but-last source could drive the
        last (highest-flux) source's remaining budget to zero or negative."""
        fluxes = [1e-9] * 50 + [1.0]
        counts = distribute_ray_budget(60, fluxes)
        assert sum(counts) == 60
        assert all(c >= 1 for c in counts)
        # The dominant source should get the lion's share of the budget.
        assert counts[-1] > 0
        assert counts[-1] >= max(counts[:-1])

    def test_single_source_gets_full_budget(self):
        assert distribute_ray_budget(1000, [1.0]) == [1000]

    def test_zero_flux_splits_evenly(self):
        counts = distribute_ray_budget(10, [0.0, 0.0, 0.0])
        assert sum(counts) == 10
        assert all(c >= 1 for c in counts)

    def test_fewer_rays_than_sources(self):
        counts = distribute_ray_budget(2, [1.0, 1.0, 1.0])
        assert sum(counts) == 2
        assert counts.count(1) == 2
        assert counts.count(0) == 1
