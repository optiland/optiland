"""Computes a serializable numerical snapshot of an Optic for regression testing.

The snapshot covers ray intercept positions, OPD/wavefront RMS, spot diagram
centroids, and PSF peak/FWHM -- the surfaces most likely to silently shift
during refactors of geometries/, materials/, psf/, rays/, or backend/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland.analysis.spot_diagram import SpotDiagram
from optiland.psf import FFTPSF
from optiland.wavefront.opd import OPD

if TYPE_CHECKING:
    from optiland.optic import Optic


def _to_list(value: Any) -> list[float]:
    """Flatten a backend array (or scalar) into a plain list of Python floats."""
    arr = np.atleast_1d(be.to_numpy(value))
    return [float(v) for v in arr.ravel()]


def _ray_intercepts(optic: Optic, wavelength: float) -> dict[str, list[float]]:
    """Trace a small ray fan and record image-surface intercept coordinates."""
    rays = optic.trace(Hx=0.0, Hy=1.0, wavelength=wavelength, num_rays=6)
    return {"x": _to_list(rays.x), "y": _to_list(rays.y), "z": _to_list(rays.z)}


def _spot_centroids(optic: Optic) -> list[dict[str, float]]:
    """Compute per-field spot diagram centroids."""
    spot = SpotDiagram(optic, num_rings=3)
    centroids = spot.centroid()
    return [
        {"x": float(be.to_numpy(x)), "y": float(be.to_numpy(y))} for x, y in centroids
    ]


def _opd_rms(optic: Optic) -> list[float]:
    """Compute OPD RMS wavefront error at the on-axis and max-field points."""
    values = []
    for hy in (0.0, 1.0):
        opd = OPD(optic, field=(0.0, hy), wavelength="primary", num_rays=8)
        values.append(float(be.to_numpy(opd.rms())))
    return values


def _psf_peak_and_fwhm(optic: Optic) -> dict[str, float]:
    """Compute the on-axis PSF peak and a pixel-count FWHM along its center row."""
    psf = FFTPSF(
        optic, field=(0.0, 0.0), wavelength="primary", num_rays=32, grid_size=64
    )
    image = be.to_numpy(psf.psf)
    peak = float(image.max())

    center_row = image[image.shape[0] // 2, :]
    half_max = center_row.max() / 2.0
    above = np.where(center_row >= half_max)[0]
    fwhm_px = float(above[-1] - above[0] + 1) if above.size else 0.0

    return {"peak": peak, "fwhm_px": fwhm_px}


def compute_snapshot(optic: Optic) -> dict[str, Any]:
    """Compute the full golden-value snapshot for a configured Optic.

    Args:
        optic: A fully configured, unmodified Optic instance.

    Returns:
        A JSON-serializable dict of numerical snapshot data.
    """
    primary_wavelength = float(optic.primary_wavelength)
    return {
        "ray_intercepts": _ray_intercepts(optic, primary_wavelength),
        "spot_centroids": _spot_centroids(optic),
        "opd_rms": _opd_rms(optic),
        "psf": _psf_peak_and_fwhm(optic),
    }
