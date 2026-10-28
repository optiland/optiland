"""Golden-equivalence test for the benchmark optics (decision V22).

Real-ray coordinates, direction cosines, OPD and first-order values must match
the committed fixtures to 1e-12 relative (numpy backend) before and after every
engine PR. An intentional change is regenerated with
``python -m tests.golden.regenerate --justification "..."`` and justified in
the PR.

Kramer Harrison, 2026
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import optiland.backend as be

from .golden.regenerate import FIXTURES_DIR
from .golden.snapshot import GOLDEN_PROBLEMS, compute_snapshot

RTOL = 1e-12


def _assert_close(actual, expected, path: str) -> None:
    """Recursively compare snapshots at 1e-12 relative.

    Array entries are compared relative to the largest magnitude in the same
    array, so exact zeros (on-axis coordinates) only need to be zero to the
    rounding level of their neighbours.
    """
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{path}: key mismatch"
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        a = np.asarray(actual, dtype=float)
        e = np.asarray(expected, dtype=float)
        assert a.shape == e.shape, f"{path}: shape mismatch"
        scale = np.max(np.abs(e), initial=0.0)
        err = np.abs(a - e) / (scale if scale > 0.0 else 1.0)
        assert np.all(err <= RTOL), f"{path}: max relative error {err.max():.3e}"
    else:
        scale = max(abs(expected), 1e-300)
        assert abs(actual - expected) <= RTOL * scale, f"{path}: {actual} != {expected}"


@pytest.mark.parametrize("problem_name", GOLDEN_PROBLEMS)
def test_golden_trace(problem_name):
    be.set_backend("numpy")
    path = FIXTURES_DIR / f"{problem_name}.json"
    expected = json.loads(path.read_text(encoding="utf-8"))
    _assert_close(compute_snapshot(problem_name), expected, problem_name)
