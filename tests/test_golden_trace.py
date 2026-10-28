"""Golden-equivalence test for the benchmark optics (decision V22).

Real-ray coordinates, direction cosines, OPD and first-order values must match
the committed fixtures to 1e-12 relative (numpy backend) before and after every
engine PR. An intentional change is regenerated with
``python -m tests.golden.regenerate --justification "..."`` and justified in
the PR.

OPD is the difference of two optical path lengths that are 1e4 to 3e5 waves
long, so its rounding floor scales with the path length, not with the OPD.
Platform libm/BLAS differences move it by a few ulp of the path length (up to
5 eps x OPL measured between Windows and Linux CI), which is far above 1e-12 of
a sub-wave OPD. OPD entries therefore also accept an absolute error of
:data:`OPD_PATH_RTOL` times the optical path length.

Kramer Harrison, 2026
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import optiland.backend as be
from benchmarks.optimization.problems import PROBLEMS
from optiland import _epoch

from .golden.regenerate import FIXTURES_DIR
from .golden.snapshot import GOLDEN_PROBLEMS, compute_snapshot

RTOL = 1e-12
# ~45 eps relative to the optical path the OPD is differenced from
OPD_PATH_RTOL = 1e-14


def _assert_close(actual, expected, path: str, atol: float = 0.0) -> None:
    """Recursively compare snapshots at 1e-12 relative.

    Array entries are compared relative to the largest magnitude in the same
    array, so exact zeros (on-axis coordinates) only need to be zero to the
    rounding level of their neighbours. ``atol`` adds an absolute allowance
    for quantities whose rounding floor is set by a larger intermediate.
    """
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{path}: key mismatch"
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}", atol)
    elif isinstance(expected, list):
        a = np.asarray(actual, dtype=float)
        e = np.asarray(expected, dtype=float)
        assert a.shape == e.shape, f"{path}: shape mismatch"
        scale = np.max(np.abs(e), initial=0.0)
        err = np.abs(a - e)
        tol = max(RTOL * scale, atol)
        if tol == 0.0:
            tol = RTOL
        assert np.all(err <= tol), (
            f"{path}: max error {err.max():.3e} > {tol:.3e} "
            f"(relative {err.max() / (scale if scale > 0.0 else 1.0):.3e})"
        )
    else:
        scale = max(abs(expected), 1e-300)
        assert abs(actual - expected) <= max(RTOL * scale, atol), (
            f"{path}: {actual} != {expected}"
        )


def _optical_path_waves(problem_name: str, wavelength: float) -> float:
    """Optical path length scale of an optic in waves at ``wavelength`` (um).

    The axial span of the finite surface positions (lens units, mm) bounds the
    path the OPD is differenced from; the refractive index only scales it by a
    factor of order one, which ``OPD_PATH_RTOL`` absorbs.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optic, _ = PROBLEMS[problem_name]()
    positions = be.to_numpy(optic.surfaces.positions).ravel()
    positions = positions[np.isfinite(positions)]
    return float(positions.max() - positions.min()) * 1e3 / wavelength


def _assert_matches_fixture(problem_name: str, actual: dict) -> None:
    """Compare a snapshot with the committed fixture of ``problem_name``."""
    path = FIXTURES_DIR / f"{problem_name}.json"
    expected = json.loads(path.read_text(encoding="utf-8"))

    opd_expected = expected.pop("opd")
    opd_actual = actual.pop("opd")
    _assert_close(actual, expected, problem_name)

    assert opd_actual.keys() == opd_expected.keys(), f"{problem_name}.opd: keys"
    for wavelength, fields in opd_expected.items():
        atol = OPD_PATH_RTOL * _optical_path_waves(problem_name, float(wavelength))
        _assert_close(
            opd_actual[wavelength],
            fields,
            f"{problem_name}.opd.{wavelength}",
            atol,
        )


@pytest.mark.parametrize("problem_name", GOLDEN_PROBLEMS)
def test_golden_trace(problem_name):
    be.set_backend("numpy")
    _assert_matches_fixture(problem_name, compute_snapshot(problem_name))


@pytest.mark.parametrize("problem_name", GOLDEN_PROBLEMS)
def test_golden_trace_inside_epoch(problem_name, monkeypatch):
    # Memoized paraxial, aperture and field quantities reproduce the fixtures,
    # and the debug guard confirms the snapshot leaves the optic unchanged.
    be.set_backend("numpy")
    monkeypatch.setenv(_epoch.CHECK_ENV, "1")
    actual = compute_snapshot(problem_name, in_epoch=True)
    _assert_matches_fixture(problem_name, actual)


def test_opd_allowance_stays_tight():
    # The allowance must stay far below anything physically meaningful.
    for problem_name in GOLDEN_PROBLEMS:
        atol = OPD_PATH_RTOL * _optical_path_waves(problem_name, 0.4861)
        assert atol < 1e-8, problem_name
