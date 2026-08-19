"""Reciprocity: swapping a source and detector of matched extent
gives the same transferred flux.

Two identically-sized Lambertian-emitting patches face each other; one
config emits from A and detects at B, the other emits from B (with the
source flipped 180deg to face back toward A) and detects at A. For matched
-area Lambertian emitters/receivers this is a degenerate case of the
radiative-transfer reciprocity relation ``A1*F12 = A2*F21`` (equal areas
force equal transferred fractions) -- a real, non-trivial statement about
the source/detector code paths agreeing with each other, not just a
restatement of the rigid-transform invariance already checked elsewhere.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    ExtendedSourceConfig,
    IrradianceDetectorConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend


def _transferred_flux(source_z: float, detector_z: float, flip_source: bool) -> float:
    spec = Spectrum.monochromatic(0.55)
    size = 3.0
    scene = NSQScene()
    rx = np.pi if flip_source else 0.0
    scene.add_source(
        "S",
        CoordinateSystem(z=source_z, rx=rx),
        ExtendedSourceConfig(
            spectrum=spec, total_flux=1.0, width=size, height=size, half_angle_deg=90.0
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=detector_z),
        IrradianceDetectorConfig(
            width=size, height=size, num_pixels_x=1, num_pixels_y=1
        ),
    )
    result = scene.trace(num_rays=1_000_000, seed=1, backend=NumpyBackend(seed=1))
    return result.total_flux_detected


def test_swapping_matched_source_and_detector_transfers_equal_flux():
    separation = 25.0
    forward = _transferred_flux(0.0, separation, flip_source=False)
    reverse = _transferred_flux(separation, 0.0, flip_source=True)

    assert forward == pytest.approx(reverse, rel=0.05)
    assert forward > 0.0  # sanity: the geometry actually couples
