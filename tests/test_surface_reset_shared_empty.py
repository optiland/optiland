"""Surface.reset shares one size-0 array across its records (speed PR, item 4)."""

from __future__ import annotations

import optiland.backend as be
from optiland.samples.objectives import DoubleGauss

RECORDS = ("x", "y", "z", "L", "M", "N", "intensity", "aoi", "opd", "u")


def test_reset_leaves_empty_records(set_test_backend):
    optic = DoubleGauss()
    optic.trace(0, 1, 0.55, num_rays=3, record=False)
    for surface in optic.surface_group.surfaces:
        for name in RECORDS:
            assert be.size(getattr(surface, name)) == 0


def test_recording_after_reset_does_not_alias(set_test_backend):
    optic = DoubleGauss()
    optic.trace(0, 1, 0.55, num_rays=3)
    surface = optic.surface_group.surfaces[2]
    assert be.size(surface.x) > 0
    assert surface.x is not surface.y
