"""Batch invariance: bit-identical results across batch_size (D11).

PCG32 draws are keyed by ``(seed, ray_id, bounce, event_slot)`` -- never by
position within a batch or which other rays happen to be alive alongside a
given one -- so the recorded irradiance map must come out bit-for-bit
identical no matter how the same ray count is chopped into batches.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend


def _build_scene() -> NSQScene:
    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=8.0),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50.0),
        LensConfig(
            r1=60.0,
            r2=-60.0,
            thickness=6.0,
            material="N-BK7",
            front_aperture_radius=12.0,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=200.0),
        IrradianceDetectorConfig(
            width=40, height=40, num_pixels_x=32, num_pixels_y=32, splat="hard"
        ),
    )
    return scene


def test_bit_identical_across_batch_sizes():
    results = {}
    for batch_size in (1, 7, 1024, 16_384):
        scene = _build_scene()
        result = scene.trace(
            num_rays=3000,
            seed=42,
            batch_size=batch_size,
            backend=NumpyBackend(seed=42),
        )
        results[batch_size] = result.detectors["D1"].irradiance.copy()

    reference = results[1]
    for batch_size, data in results.items():
        assert np.array_equal(reference, data), (
            f"batch_size={batch_size} diverged from batch_size=1 -- the RNG "
            "stream must be independent of batching."
        )


def test_bounded_splitting_agrees_statistically_across_batch_sizes():
    """Bounded splitting (D2, PR11) does *not* currently carry the same
    bit-identical guarantee as the unsplit path.

    A spawned ray's id comes from a monotonic counter allocated at the
    point it is spawned, and how many *other* concurrently-live rays have
    already spawned by that point depends on ``batch_size`` (fewer rays are
    ever concurrently live in a small batch). So a given physical split
    -off photon lineage can receive a different id -- and therefore a
    different downstream PCG32 stream -- under different batch sizes. This
    is a real, discovered gap in the D11 batch-invariance guarantee (this
    test exists to keep it honestly documented, not silently rely on it);
    see ``docs/gallery/nonsequential/validation_report.rst``.

    What *does* still hold, and is checked here: the physics is still
    correct and unbiased -- results agree statistically across batch
    sizes, just not bit-for-bit.
    """
    from optiland.nonsequential.ir.scene_ir import SamplingPolicy

    totals = []
    for batch_size in (1, 7, 1024):
        scene = _build_scene()
        scene.sampling_policy = SamplingPolicy(split_depth=2, split_budget=8.0)
        result = scene.trace(
            num_rays=5_000,
            seed=3,
            batch_size=batch_size,
            backend=NumpyBackend(seed=3),
        )
        totals.append(result.total_flux_detected)

    assert max(totals) - min(totals) < 0.08 * np.mean(totals)
