"""Golden-value regression snapshot suite.

Traces a small set of representative Optic systems at both backends and
compares ray intercepts, OPD RMS, spot centroids, and PSF peak/FWHM against
committed fixtures. This suite gates every PR in Phases 2-7 of the tech-debt
spec (geometries, materials, psf, rays, backend) -- no such PR should merge
without a clean diff here for its area.

Run with --update-golden to deliberately regenerate the fixtures after an
intentional numerical change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import optiland.backend as be

from .golden_values import compute_snapshot
from .systems import GOLDEN_SYSTEMS

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Tolerances are looser than most unit tests: this suite exists to catch
# structural regressions across a refactor, not to pin exact float bits,
# and torch/numpy already disagree at the ~1e-6 level on some paths.
RTOL = 1e-4
ATOL = 1e-6


def _fixture_path(system_name: str) -> Path:
    return FIXTURES_DIR / f"{system_name}.json"


def _load_fixture(system_name: str) -> dict:
    with _fixture_path(system_name).open(encoding="utf-8") as f:
        return json.load(f)


def _save_fixture(system_name: str, data: dict) -> None:
    FIXTURES_DIR.mkdir(exist_ok=True)
    with _fixture_path(system_name).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _assert_snapshot_matches(actual: object, expected: object, path: str) -> None:
    """Recursively compare nested dict/list snapshot structures numerically."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual)}"
        assert actual.keys() == expected.keys(), f"{path}: key mismatch"
        for key in expected:
            _assert_snapshot_matches(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual)}"
        assert len(actual) == len(expected), f"{path}: length mismatch"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_snapshot_matches(a, e, f"{path}[{i}]")
    else:
        np.testing.assert_allclose(
            actual, expected, rtol=RTOL, atol=ATOL, err_msg=f"mismatch at {path}"
        )


@pytest.mark.parametrize("system_name", sorted(GOLDEN_SYSTEMS))
def test_golden_snapshot(system_name, set_test_backend, request):
    """Compare a system's traced snapshot against its committed golden fixture."""
    optic = GOLDEN_SYSTEMS[system_name]()
    snapshot = compute_snapshot(optic)

    if request.config.getoption("--update-golden"):
        # Only numpy is written; torch would otherwise clobber it with
        # noisier values from its autograd-enabled float64 path.
        if be.get_backend() == "numpy":
            _save_fixture(system_name, snapshot)
        pytest.skip("Fixture regenerated via --update-golden")

    if not _fixture_path(system_name).exists():
        pytest.fail(
            f"No golden fixture for '{system_name}'. Run with --update-golden "
            "to generate one after confirming the values are correct."
        )

    expected = _load_fixture(system_name)
    _assert_snapshot_matches(snapshot, expected, system_name)
