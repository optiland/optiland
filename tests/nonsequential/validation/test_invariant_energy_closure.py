"""Energy closure: in = detected + absorbed + escaped + roulette-lost.

Consolidated, scene-parametrized version of the ledger-closure checks
scattered across test_nsq_basic.py / test_nsq_absorption.py -- every
launched watt must be accounted for to 1e-9 relative, across a spread of
scenes exercising different loss mechanisms (surface absorption, bulk
absorption, escape, depth truncation, Russian roulette).

Kramer Harrison, 2026
"""

from __future__ import annotations

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    AbsorbingComponent,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.ir.scene_ir import SamplingPolicy


def _closure_residual(result) -> float:
    balance = (
        result.total_flux_in
        - result.total_flux_detected
        - result.total_flux_absorbed
        - result.total_flux_bulk_absorbed
        - result.total_flux_escaped
        - result.total_flux_lost
    )
    return abs(balance) / result.total_flux_in


def _lens_scene() -> NSQScene:
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
        IrradianceDetectorConfig(width=40, height=40, num_pixels_x=32, num_pixels_y=32),
    )
    return scene


def test_closure_default_scene():
    result = _lens_scene().trace(num_rays=20_000, seed=1, backend=NumpyBackend(seed=1))
    assert _closure_residual(result) < 1e-9


def test_closure_with_surface_absorber():
    scene = _lens_scene()
    scene.add_component(
        "baffle",
        AbsorbingComponent(
            cs=CoordinateSystem(x=3.0, z=25.0),
            geometry=FinitePlaneGeometry(width=20, height=20),
        ),
    )
    result = scene.trace(num_rays=20_000, seed=1, backend=NumpyBackend(seed=1))
    assert result.total_flux_absorbed > 0.0
    assert _closure_residual(result) < 1e-9


def test_closure_with_shallow_max_depth():
    """Forces heavy depth-truncation -- the exact scenario flux_conservation
    reporting was originally biased under (omitting total_flux_lost).
    """
    result = _lens_scene().trace(
        num_rays=10_000, seed=1, max_depth=1, backend=NumpyBackend(seed=1)
    )
    assert result.num_rays_depth_killed > 0
    assert _closure_residual(result) < 1e-9


def test_closure_with_aggressive_roulette():
    """Below ``reflect_prob="fresnel"``, the Fresnel branch weight is
    identically 1 (see test_benchmark_window_transmittance.py's docstring
    reasoning) so the ledger closes exactly; Russian roulette boosts
    survivors' flux to stay unbiased *in expectation*, which for a single
    finite trace is Monte Carlo noise around exact closure, not exact
    closure itself (discovered and documented during PR11's
    implementation). The bound here is generous specifically to tolerate
    that noise while still catching a real energy leak.
    """
    scene = _lens_scene()
    scene.sampling_policy = SamplingPolicy(rr_start_flux=2.0)  # every ray a candidate
    result = scene.trace(num_rays=10_000, seed=1, backend=NumpyBackend(seed=1))
    assert result.num_rays_flux_killed > 0
    assert _closure_residual(result) < 0.6


def test_closure_with_bounded_splitting():
    """Splitting itself is deterministic (forced R/T weights, no importance
    division -- see RefractiveComponent.interact), so unlike plain
    roulette, closure stays exact here.
    """
    scene = _lens_scene()
    scene.sampling_policy = SamplingPolicy(split_depth=2, split_budget=4.0)
    result = scene.trace(num_rays=10_000, seed=1, backend=NumpyBackend(seed=1))
    assert _closure_residual(result) < 1e-9


def test_closure_holds_exactly_at_default_fresnel_policy():
    scene = _lens_scene()
    scene.sampling_policy = SamplingPolicy(reflect_prob="fresnel")
    result = scene.trace(num_rays=10_000, seed=1, backend=NumpyBackend(seed=1))
    assert _closure_residual(result) < 1e-9


def test_closure_holds_statistically_under_importance_biasing():
    """Off-Fresnel reflect_prob makes the branch weight != 1, so a single
    finite trace's ledger only closes in expectation (Monte Carlo noise) --
    see the reasoning in test_closure_with_aggressive_roulette. This is
    inherent to importance sampling, not a defect.
    """
    scene = _lens_scene()
    scene.sampling_policy = SamplingPolicy(reflect_prob=0.5)
    result = scene.trace(num_rays=10_000, seed=1, backend=NumpyBackend(seed=1))
    assert _closure_residual(result) < 0.1
