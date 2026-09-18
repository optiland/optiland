"""A transmissive detector's reading is not a flux destination.

A detector with ``absorb=False`` samples the beam and lets the ray carry on,
so the watt it reads is still in the trace and is booked a second time at
whatever finally removes it. Counting the reading in the conservation
identity booked it twice and reported an energy defect on a scene the engine
traced correctly.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.backends.torch_backend import TorchBackend


def _backend_for_current_array_library(seed: int):
    return (
        NumpyBackend(seed=seed)
        if be.get_backend() == "numpy"
        else TorchBackend(seed=seed)
    )


def _tap_then_screen(tap_absorbs: bool) -> NSQScene:
    """Collimated beam, a detector at z=10, an absorbing screen at z=20."""
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55),
            total_flux=1.0,
            aperture_radius=1.0,
        ),
    )
    scene.add_detector(
        "tap",
        CoordinateSystem(z=10),
        IrradianceDetectorConfig(
            width=10,
            height=10,
            num_pixels_x=32,
            num_pixels_y=32,
            splat="hard",
            absorb=tap_absorbs,
        ),
    )
    scene.add_detector(
        "screen",
        CoordinateSystem(z=20),
        IrradianceDetectorConfig(
            width=10,
            height=10,
            num_pixels_x=32,
            num_pixels_y=32,
            splat="hard",
            absorb=True,
        ),
    )
    return scene


def test_transmissive_detector_does_not_close_the_ledger_twice(set_test_backend):
    """1 W emitted, 2 W read, and the identity still closes.

    Before the fix this scene reported ``flux_conservation_error`` of 1.0:
    the tap's watt and the screen's watt were both subtracted from the one
    watt emitted.
    """
    scene = _tap_then_screen(tap_absorbs=False)
    result = scene.trace(
        num_rays=20_000, seed=42, backend=_backend_for_current_array_library(42)
    )

    # Both detectors read the beam; the aggregate is what they read.
    assert result.detectors["tap"].total_flux_float == pytest.approx(1.0, abs=1e-9)
    assert result.detectors["screen"].total_flux_float == pytest.approx(1.0, abs=1e-9)
    assert result.total_flux_detected == pytest.approx(2.0, abs=1e-9)

    # The tapped part is reported on its own, and the identity leaves it out.
    assert result.total_flux_tapped == pytest.approx(1.0, abs=1e-9)
    assert result.flux_conservation_error == pytest.approx(0.0, abs=1e-12)


def test_absorbing_detector_is_not_tapped(set_test_backend):
    """An ``absorb=True`` detector removes the ray, so it books nothing."""
    scene = _tap_then_screen(tap_absorbs=True)
    result = scene.trace(
        num_rays=20_000, seed=42, backend=_backend_for_current_array_library(42)
    )

    assert result.total_flux_tapped == 0.0
    assert result.total_flux_detected == pytest.approx(1.0, abs=1e-9)
    assert result.flux_conservation_error == pytest.approx(0.0, abs=1e-12)


def test_tap_with_nothing_behind_it_lets_the_flux_escape(set_test_backend):
    """The tapped watt is booked where it does leave: the escape bin."""
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55),
            total_flux=1.0,
            aperture_radius=1.0,
        ),
    )
    scene.add_detector(
        "tap",
        CoordinateSystem(z=10),
        IrradianceDetectorConfig(
            width=10,
            height=10,
            num_pixels_x=32,
            num_pixels_y=32,
            splat="hard",
            absorb=False,
        ),
    )
    result = scene.trace(
        num_rays=2_000, seed=0, backend=_backend_for_current_array_library(0)
    )

    assert result.total_flux_detected == pytest.approx(1.0, abs=1e-9)
    assert result.total_flux_tapped == pytest.approx(1.0, abs=1e-9)
    assert result.total_flux_escaped == pytest.approx(1.0, abs=1e-9)
    assert result.flux_conservation_error == pytest.approx(0.0, abs=1e-12)
