"""Tests for MonteCarlo.run(..., workers=N): trials in several processes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import optiland.backend as be
from optiland.samples.objectives import CookeTriplet
from optiland.tolerancing import RangeSampler, Tolerance, ToleranceType, Tolerancing
from optiland.tolerancing.monte_carlo import MonteCarlo


class FailingTolerance(ToleranceType):
    """A tolerance type that fails when applied (module level, so it pickles)."""

    @property
    def label(self) -> str:
        """Return the label."""
        return "Failing"

    def apply(self, delta: float) -> None:
        """Fail."""
        raise RuntimeError("injected failure in a worker")

    def reset(self) -> None:
        """Do nothing."""


def _poses(optic) -> np.ndarray:
    return np.array(
        [
            [
                float(np.ravel(be.to_numpy(getattr(s.geometry.cs, a)))[0])
                for a in ("y", "z", "rx")
            ]
            for s in optic.surfaces
        ]
    )


@pytest.fixture
def tolerancing():
    be.set_backend("numpy")
    lens = CookeTriplet()
    tol = Tolerancing(lens)
    tol.add_operand(
        "rms_spot_size",
        {
            "optic": lens,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "num_rays": 3,
            "wavelength": 0.55,
        },
    )
    tol.add_tolerance("radius", surface_number=1, limit=0.2)
    tol.add_tolerance("thickness", surface_number=2, limit=0.05)
    tol.add_tolerance("surface_tilt", surface_number=3, axis="x", limit=0.05)
    tol.add_compensator("quick_focus")
    return tol


def test_parallel_equals_serial_with_seed(tolerancing):
    mc = MonteCarlo(tolerancing)
    mc.run(7, seed=11)
    serial = mc.get_results().copy()

    before = _poses(tolerancing.optic)
    mc.run(7, seed=11, workers=3)
    pd.testing.assert_frame_equal(mc.get_results(), serial)
    # the optic in this process is not touched
    np.testing.assert_array_equal(_poses(tolerancing.optic), before)


def test_parallel_without_seed_draws_distinct_trials(tolerancing):
    mc = MonteCarlo(tolerancing)
    mc.run(6, workers=2)
    values = mc.get_results()["Radius S1"].to_numpy()
    assert len(set(values)) == 6


def test_more_workers_than_trials(tolerancing):
    mc = MonteCarlo(tolerancing)
    mc.run(2, seed=0, workers=4)
    assert len(mc.get_results()) == 2


def test_worker_failure_propagates_and_keeps_results(tolerancing):
    mc = MonteCarlo(tolerancing)
    mc.run(2, seed=0)
    completed = mc.get_results()
    tolerancing.perturbations.append(
        Tolerance(FailingTolerance(tolerancing.optic), -1.0, 1.0)
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        mc.run(4, seed=0, workers=2)
    assert mc.get_results() is completed


def test_invalid_workers(tolerancing):
    with pytest.raises(ValueError, match="at least 1"):
        MonteCarlo(tolerancing).run(2, workers=0)


def test_legacy_perturbation_rejected(tolerancing):
    tolerancing.add_perturbation(
        "thickness", RangeSampler(4.0, 4.2, 3), surface_number=2
    )
    with pytest.raises(ValueError, match="add_perturbation"):
        MonteCarlo(tolerancing).run(2, workers=2)


def test_torch_backend_rejected(tolerancing):
    pytest.importorskip("torch")
    mc = MonteCarlo(tolerancing)
    be.set_backend("torch")
    try:
        with pytest.raises(ValueError, match="NumPy backend"):
            mc.run(2, workers=2)
    finally:
        be.set_backend("numpy")
