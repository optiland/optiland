from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.analysis.image_simulation import (
    DistortionWarper,
    ImageSimulationEngine,
    PSFBasisGenerator,
)
from optiland.samples.objectives import CookeTriplet


class TestImageSimulation:
    @pytest.fixture
    def optic(self):
        return CookeTriplet()

    @pytest.fixture
    def source_image(self):
        # Create a small dummy image (green square in black background)
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[10:22, 10:22, 1] = 1.0
        return img

    def test_engine_init(self, optic, source_image):
        engine = ImageSimulationEngine(optic)

        assert engine.source_image is None
        assert engine.simulated_image is None
        assert engine.config is not None

    def test_engine_run(self, optic, source_image):
        # Use very loose config for speed
        config = {
            "psf_grid_shape": (3, 3),
            "psf_size": 32,
            "num_rays": 32,
            "n_components": 2,
            "oversample": 1,
            "wavelengths": [0.55],  # Mono
        }
        engine = ImageSimulationEngine(optic, config=config)
        result = engine.run(source_image)

        assert result.shape == (1, 1, 32, 32)
        assert not be.any(be.isnan(result))

        # Let's check typical RGB case
        config["wavelengths"] = [0.65, 0.55, 0.45]
        engine = ImageSimulationEngine(optic, config=config)
        result = engine.run(source_image)

        assert result.shape == (1, 3, 32, 32)
        assert not be.any(be.isnan(result))
        assert be.max(result) > 0  # Should have some signal

    def test_distortion_warper(self, optic):
        warper = DistortionWarper(optic)
        H, W = 32, 32
        dist_map = warper.generate_distortion_map(wavelength=0.55, image_shape=(H, W))
        assert dist_map.shape == (1, H, W, 2)

        # Warp a dummy image
        img = be.ones((1, H, W))
        warped = warper.warp_image(img, dist_map)
        assert warped.shape == (1, H, W)

    def test_psf_basis_generator(self, optic):
        gen = PSFBasisGenerator(
            optic, wavelength=0.55, grid_shape=(3, 3), num_rays=32, psf_grid_size=32
        )
        eigen_psfs, coeffs, mean_psf = gen.generate_basis(n_components=2)

        assert eigen_psfs.shape == (2, 32, 32)
        assert coeffs.shape == (2, 3, 3)
        assert mean_psf.shape == (32, 32)

        resized_coeffs = gen.resize_coefficient_map(coeffs, (64, 64))
        assert resized_coeffs.shape == (2, 64, 64)

    def test_engine_prepare_grayscale(self, optic):
        image = np.ones((32, 32), dtype=np.float32)

        engine = ImageSimulationEngine(optic)
        prepared = engine._prepare_source_image(image)

        assert prepared.shape == (1, 1, 32, 32)

    def test_engine_rejects_invalid_input(self, optic):
        engine = ImageSimulationEngine(optic)

        invalid_rgb = np.ones((3, 32, 32), dtype=np.float32)
        invalid_batch = np.ones((2, 2, 32, 32), dtype=np.float32)

        with pytest.raises(ValueError, match="3D source_image"):
            engine._prepare_source_image(invalid_rgb)

        with pytest.raises(ValueError, match="4D source_image"):
            engine._prepare_source_image(invalid_batch)

    def test_engine_view_validation(self, optic, source_image):
        engine = ImageSimulationEngine(optic)

        with pytest.raises(RuntimeError, match="Call run"):
            engine.view(show=False)

        config = {
            "psf_grid_shape": (3, 3),
            "psf_size": 32,
            "num_rays": 32,
            "n_components": 2,
            "oversample": 1,
            "wavelengths": [0.65, 0.55, 0.45],
        }

        engine = ImageSimulationEngine(optic, config=config)
        engine.run(source_image)

        with pytest.raises(IndexError, match="index must be between"):
            engine.view(index=1, show=False)