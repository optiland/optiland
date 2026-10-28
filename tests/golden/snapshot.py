"""Numerical snapshots of the benchmark optics for golden-equivalence tests.

The snapshot records real-ray coordinates, direction cosines, OPD and
first-order values at fixed samples for two wavelengths. Engine PRs of the
optimization v2 program must leave it unchanged (decision V22); the fixtures
are regenerated only through ``tests/golden/regenerate.py``.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Any

import numpy as np

import optiland.backend as be
from benchmarks.optimization.problems import PROBLEMS
from optiland import _epoch
from optiland.wavefront.opd import OPD

# One optic per benchmark problem; the scaling probe is represented by its
# smallest size to keep fixtures small.
GOLDEN_PROBLEMS = (
    "cooke",
    "double_gauss",
    "wide_asphere",
    "freeform",
    "scaling_10",
)

HY_SAMPLES = (0.0, 1.0)
PUPIL_SAMPLES = (
    (0.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.5, 0.5),
    (-0.3, 0.7),
)
OPD_NUM_RAYS = 3


def _to_list(value: Any) -> list[float]:
    """Flatten a backend array (or scalar) into a list of Python floats."""
    return [float(v) for v in np.atleast_1d(be.to_numpy(value)).ravel()]


def _wavelengths(optic: Any) -> list[float]:
    """The first and the primary wavelength (two entries, or one if equal)."""
    first = float(optic.wavelengths.wavelengths[0].value)
    primary = float(optic.primary_wavelength)
    return [first] if first == primary else [first, primary]


def _rays(optic: Any, wavelength: float) -> dict[str, dict[str, list[float]]]:
    """Image-plane rays at fixed field and pupil samples."""
    px = be.array([p[0] for p in PUPIL_SAMPLES])
    py = be.array([p[1] for p in PUPIL_SAMPLES])
    out = {}
    for hy in HY_SAMPLES:
        rays = optic.trace_generic(0.0, hy, px, py, wavelength)
        out[f"Hy={hy}"] = {
            name: _to_list(getattr(rays, name))
            for name in ("x", "y", "z", "L", "M", "N")
        }
    return out


def _opd(optic: Any, wavelength: float) -> dict[str, list[float]]:
    """OPD (waves) on a hexapolar pupil grid at fixed fields."""
    out = {}
    for hy in HY_SAMPLES:
        opd = OPD(
            optic,
            field=(0.0, hy),
            wavelength=wavelength,
            num_rays=OPD_NUM_RAYS,
            distribution="hexapolar",
        )
        out[f"Hy={hy}"] = _to_list(opd.get_data((0.0, hy), wavelength).opd)
    return out


def _paraxial(optic: Any) -> dict[str, float]:
    """First-order quantities at the primary wavelength."""
    paraxial = optic.paraxial
    positions = be.to_numpy(optic.surfaces.positions).ravel()
    return {
        "EPD": float(be.to_numpy(paraxial.EPD())),
        "EPL": float(be.to_numpy(paraxial.EPL())),
        "f1": float(be.to_numpy(paraxial.f1())),
        "f2": float(be.to_numpy(paraxial.f2())),
        "BFL": float(be.to_numpy(paraxial.F2())),
        "image_position": float(positions[-1]),
    }


def compute_snapshot(problem_name: str, *, in_epoch: bool = False) -> dict[str, Any]:
    """Compute the golden snapshot of one benchmark optic (numpy backend).

    Args:
        problem_name: A key of :data:`GOLDEN_PROBLEMS`.
        in_epoch: If True, build the optic first and compute the whole snapshot
            inside one evaluation epoch, so memoized quantities are reused.

    Returns:
        A JSON-serialisable dict.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optic, _ = PROBLEMS[problem_name]()
        scope = (
            _epoch.evaluation_epoch((optic,)) if in_epoch else contextlib.nullcontext()
        )
        with scope:
            return _snapshot(optic)


def _snapshot(optic: Any) -> dict[str, Any]:
    """Snapshot of a built optic; see :func:`compute_snapshot`."""
    wavelengths = _wavelengths(optic)
    return {
        "wavelengths": wavelengths,
        "paraxial": _paraxial(optic),
        "rays": {str(wl): _rays(optic, wl) for wl in wavelengths},
        "opd": {str(wl): _opd(optic, wl) for wl in wavelengths},
    }
