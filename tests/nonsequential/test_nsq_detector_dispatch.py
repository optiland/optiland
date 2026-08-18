"""Tests for PR10: unified detector dispatch (D-10), attached total_flux
(D-14), and Gaussian splat (D-11).

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.array_backend import ArrayBackend
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.backends.torch_backend import TorchBackend


def _collimated_scene(det_kwargs: dict) -> NSQScene:
    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=1.0),
    )
    cfg_kwargs = {
        "width": 10,
        "height": 10,
        "num_pixels_x": 32,
        "num_pixels_y": 32,
        **det_kwargs,
    }
    scene.add_detector(
        "D", CoordinateSystem(z=10), IrradianceDetectorConfig(**cfg_kwargs)
    )
    return scene


# ---------------------------------------------------------------------------
# D-10 -- duplicated dispatch is gone; both backends share one implementation
# ---------------------------------------------------------------------------


def test_intersect_detectors_not_duplicated_per_backend():
    """ArrayBackend/TorchBackend no longer define their own detector dispatch.

    Before PR10, ``ArrayBackend._intersect_detectors`` and
    ``TorchBackend._intersect_detectors`` were two independently maintained
    implementations with diverging grad-attachment semantics (D-10). Both
    should now delegate to the single shared
    ``detectors.dispatch.intersect_detectors``.
    """
    assert "_intersect_detectors" not in ArrayBackend.__dict__
    assert "_intersect_detectors" not in TorchBackend.__dict__


def test_shared_dispatch_used_by_both_backends():
    from optiland.nonsequential.detectors import dispatch

    scene = _collimated_scene({"splat": "hard"})
    result_np = scene.trace(num_rays=2000, seed=0, backend=NumpyBackend(seed=0))
    assert result_np.detectors["D"].total_flux_float > 0.9

    import optiland.backend as be

    be.set_backend("torch")
    try:
        scene_t = _collimated_scene({"splat": "hard"})
        result_t = scene_t.trace(num_rays=2000, seed=0, backend=TorchBackend(seed=0))
        assert result_t.detectors["D"].total_flux_float > 0.9
    finally:
        be.set_backend("numpy")

    assert hasattr(dispatch, "intersect_detectors")


# ---------------------------------------------------------------------------
# D-10 -- absorb=False: transmissive, mid-system beam sampling
# ---------------------------------------------------------------------------


class TestAbsorbFlag:
    def test_absorbing_detector_terminates_ray(self):
        scene = _collimated_scene({"splat": "hard", "absorb": True})
        # A second detector further downstream should see nothing: the first
        # (absorbing) detector already killed every ray.
        scene.add_detector(
            "D2",
            CoordinateSystem(z=20),
            IrradianceDetectorConfig(
                width=10, height=10, num_pixels_x=8, num_pixels_y=8
            ),
        )
        result = scene.trace(num_rays=2000, seed=0, backend=NumpyBackend(seed=0))
        assert result.detectors["D"].total_flux_float > 0.9
        assert result.detectors["D2"].total_flux_float == pytest.approx(0.0)

    def test_transmissive_detector_passes_rays_through(self):
        scene = _collimated_scene({"splat": "hard", "absorb": False})
        scene.add_detector(
            "D2",
            CoordinateSystem(z=20),
            IrradianceDetectorConfig(
                width=10, height=10, num_pixels_x=8, num_pixels_y=8
            ),
        )
        result = scene.trace(num_rays=2000, seed=0, backend=NumpyBackend(seed=0))
        # Both the transmissive detector and the one behind it record
        # (approximately) the full flux.
        assert result.detectors["D"].total_flux_float > 0.9
        assert result.detectors["D2"].total_flux_float > 0.9

    def test_transmissive_detector_does_not_deflect_rays(self):
        """A hit on an absorb=False detector must not change ray direction."""
        scene = _collimated_scene({"splat": "hard", "absorb": False})
        scene.add_detector(
            "D2",
            CoordinateSystem(z=20),
            IrradianceDetectorConfig(
                width=10, height=10, num_pixels_x=32, num_pixels_y=32
            ),
        )
        result = scene.trace(num_rays=20_000, seed=1, backend=NumpyBackend(seed=1))
        # A collimated beam that isn't deflected lands in the same footprint
        # on both planes: same flux-weighted centroid and spread.
        det_d = result.detectors["D"]
        det_d2 = result.detectors["D2"]
        irr1 = det_d.irradiance
        irr2 = det_d2.irradiance
        assert irr1.sum() > 0
        assert irr2.sum() > 0

        def _centroid(irr, coords_x, coords_y):
            xx, yy = np.meshgrid(coords_x, coords_y)
            w = irr.sum()
            return (xx * irr).sum() / w, (yy * irr).sum() / w

        cx1, cy1 = _centroid(irr1, det_d.x_coords, det_d.y_coords)
        cx2, cy2 = _centroid(irr2, det_d2.x_coords, det_d2.y_coords)
        assert cx1 == pytest.approx(cx2, abs=0.3)
        assert cy1 == pytest.approx(cy2, abs=0.3)

    def test_config_absorb_defaults_true(self):
        cfg = IrradianceDetectorConfig(width=1.0, height=1.0)
        assert cfg.absorb is True


# ---------------------------------------------------------------------------
# D-14 -- IrradianceMap.total_flux is attached; total_flux_float is detached
# ---------------------------------------------------------------------------


class TestAttachedTotalFlux:
    def test_total_flux_carries_gradient(self):
        torch = pytest.importorskip("torch")
        import optiland.backend as be

        be.set_backend("torch")
        try:
            total_flux = torch.tensor(1.0, requires_grad=True)
            spec = Spectrum.monochromatic(0.55)
            scene = NSQScene()
            scene.add_source(
                "S",
                CoordinateSystem(),
                CollimatedSourceConfig(
                    spectrum=spec, total_flux=total_flux, aperture_radius=1.0
                ),
            )
            scene.add_detector(
                "D",
                CoordinateSystem(z=10),
                IrradianceDetectorConfig(
                    width=10,
                    height=10,
                    num_pixels_x=16,
                    num_pixels_y=16,
                    splat="bilinear",
                ),
            )
            result = scene.trace(num_rays=500, seed=0, backend=TorchBackend(seed=0))
            detected = result.detectors["D"].total_flux
            assert isinstance(detected, torch.Tensor)
            detected.backward()
            assert total_flux.grad is not None
            assert total_flux.grad.item() != 0.0
        finally:
            be.set_backend("numpy")

    def test_total_flux_float_is_plain_float(self):
        scene = _collimated_scene({"splat": "hard"})
        result = scene.trace(num_rays=1000, seed=0, backend=NumpyBackend(seed=0))
        flux_float = result.detectors["D"].total_flux_float
        assert isinstance(flux_float, float)
        assert f"{flux_float:.3g}"  # must be formattable

    def test_simulation_result_total_flux_detected_stays_plain_float(self):
        scene = _collimated_scene({"splat": "hard"})
        result = scene.trace(num_rays=1000, seed=0, backend=NumpyBackend(seed=0))
        assert isinstance(result.total_flux_detected, float)


# ---------------------------------------------------------------------------
# D-11 -- Gaussian splat is a true (truncated, renormalised) kernel
# ---------------------------------------------------------------------------


class TestGaussianSplat:
    def test_gaussian_conserves_energy(self):
        scene = _collimated_scene({"splat": "gaussian", "splat_sigma": 1.5})
        result = scene.trace(num_rays=5000, seed=0, backend=NumpyBackend(seed=0))
        # Renormalised truncated kernel: total recorded flux must equal the
        # sum of per-ray flux landing on the detector (no energy lost to
        # truncation, D-9-class bias).
        assert result.detectors["D"].total_flux_float == pytest.approx(1.0, rel=0.02)

    def test_gaussian_spreads_beyond_four_neighbours(self):
        """Gaussian must differ from bilinear -- it is not a silent alias."""
        scene_g = _collimated_scene(
            {"splat": "gaussian", "splat_sigma": 2.0, "num_pixels_x": 64}
        )
        scene_b = _collimated_scene({"splat": "bilinear", "num_pixels_x": 64})
        result_g = scene_g.trace(num_rays=200, seed=3, backend=NumpyBackend(seed=3))
        result_b = scene_b.trace(num_rays=200, seed=3, backend=NumpyBackend(seed=3))
        nz_g = (result_g.detectors["D"].irradiance > 0).sum()
        nz_b = (result_b.detectors["D"].irradiance > 0).sum()
        assert nz_g > nz_b

    def test_gaussian_differentiable(self):
        torch = pytest.importorskip("torch")
        import optiland.backend as be

        be.set_backend("torch")
        try:
            total_flux = torch.tensor(1.0, requires_grad=True)
            spec = Spectrum.monochromatic(0.55)
            scene = NSQScene()
            scene.add_source(
                "S",
                CoordinateSystem(),
                CollimatedSourceConfig(
                    spectrum=spec, total_flux=total_flux, aperture_radius=1.0
                ),
            )
            scene.add_detector(
                "D",
                CoordinateSystem(z=10),
                IrradianceDetectorConfig(
                    width=10,
                    height=10,
                    num_pixels_x=16,
                    num_pixels_y=16,
                    splat="gaussian",
                    splat_sigma=1.0,
                ),
            )
            result = scene.trace(num_rays=300, seed=0, backend=TorchBackend(seed=0))
            result.detectors["D"].total_flux.backward()
            assert total_flux.grad is not None
            assert total_flux.grad.item() != 0.0
        finally:
            be.set_backend("numpy")

    def test_zero_sigma_falls_back_to_hard(self):
        scene = _collimated_scene({"splat": "gaussian", "splat_sigma": 0.0})
        result = scene.trace(num_rays=1000, seed=0, backend=NumpyBackend(seed=0))
        assert result.detectors["D"].total_flux_float == pytest.approx(1.0, rel=0.02)


# ---------------------------------------------------------------------------
# SensorIR.absorb (D-10) reflects the live detector
# ---------------------------------------------------------------------------


def test_sensor_ir_absorb_matches_detector():
    from optiland.nonsequential.ir.lower import lower

    scene = _collimated_scene({"absorb": False})
    ir = lower(scene, strict=False)
    assert ir.sensors[0].absorb is False

    scene2 = _collimated_scene({"absorb": True})
    ir2 = lower(scene2, strict=False)
    assert ir2.sensors[0].absorb is True
