from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.analysis.image_simulation import (
    DistortionWarper,
    ImageSimulationEngine,
    PSFBasisGenerator,
    SpatiallyVariableSimulator,
)
from optiland.samples.objectives import ReverseTelephoto, Telephoto


def create_grid_image(size=128, step=20, thickness=2):
    """Creates a synthetic grid image for testing."""
    img = np.zeros((size, size), dtype=np.float32)
    for x in range(0, size, step):
        img[:, x : x + thickness] = 1.0
    for y in range(0, size, step):
        img[y : y + thickness, :] = 1.0
    return img


def test_distortion_warper(set_test_backend):
    """Test distortion map generation and image warping."""
    optic = ReverseTelephoto()
    warper = DistortionWarper(optic)

    img_size = 128
    wavelength = 0.55

    # Generate Map
    d_map = warper.generate_distortion_map(wavelength, (img_size, img_size))
    assert d_map.shape == (1, img_size, img_size, 2)
    # Polynomial fitting may overshoot slightly, so we allow a small margin
    assert be.max(d_map) <= 1.05
    assert be.min(d_map) >= -1.05

    # Warp Image
    input_img = be.array(create_grid_image(img_size))
    warped_img = warper.warp_image(input_img, d_map)

    assert warped_img.shape == (img_size, img_size)
    assert warped_img.dtype == input_img.dtype


def test_psf_basis_generator(set_test_backend):
    """Test PSF basis generation."""
    optic = Telephoto()
    gen = PSFBasisGenerator(
        optic=optic, wavelength=0.55, grid_shape=(3, 3), num_rays=32, psf_grid_size=32
    )

    n_components = 2
    eigen_psfs, coeffs, mean_psf = gen.generate_basis(n_components=n_components)

    # Check shapes
    assert eigen_psfs.shape == (n_components, 32, 32)
    assert coeffs.shape == (n_components, 3, 3)
    assert mean_psf.shape == (32, 32)

    # Check resizing
    target_shape = (64, 64)
    coeffs_resized = gen.resize_coefficient_map(coeffs, target_shape)
    assert coeffs_resized.shape == (n_components, 64, 64)


def test_spatially_variable_simulator(set_test_backend):
    """Test the convolution simulator."""
    sim = SpatiallyVariableSimulator()

    B = 2
    H, W = 64, 64
    P = 16
    K = 2

    source = be.ones((B, H, W))
    eigen_psfs = be.ones((K, P, P)) / (P * P)  # simple kernels
    coeffs = be.ones((K, H, W)) * 0.5
    mean_psf = be.ones((P, P)) / (P * P)

    result = sim.simulate(source, eigen_psfs, coeffs, mean_psf)

    assert result.shape == (B, H, W)
    assert be.all(result >= 0)


def test_image_simulation_engine_run(set_test_backend):
    """Test the full engine pipeline."""
    optic = Telephoto()
    source_np = create_grid_image(64)
    source_rgb = np.stack([source_np] * 3, axis=-1)  # (64, 64, 3)

    engine = ImageSimulationEngine(
        optic=optic,
        config={
            "wavelengths": [0.55, 0.55, 0.55],
            "psf_grid_shape": (3, 3),
            "psf_size": 32,
            "num_rays": 32,
            "n_components": 2,
            "oversample": 1,
            "padding": 16,
        },
    )

    result = engine.run(source_rgb)

    assert result.shape == (1, 3, 64, 64)
    assert not be.any(be.isnan(result))


def test_image_simulation_engine_run_batch(set_test_backend):
    """Test the full engine pipeline with batched input."""
    optic = Telephoto()
    source_np = create_grid_image(64)
    source_rgb = np.stack([source_np] * 3, axis=0)  # (3, 64, 64)
    source_batch = np.stack([source_rgb, source_rgb], axis=0).astype(np.float32)

    engine = ImageSimulationEngine(
        optic=optic,
        config={
            "wavelengths": [0.55, 0.55, 0.55],
            "psf_grid_shape": (3, 3),
            "psf_size": 32,
            "num_rays": 32,
            "n_components": 2,
            "oversample": 1,
            "padding": 16,
        },
    )

    result = engine.run(source_batch)

    assert result.shape == (2, 3, 64, 64)
    assert not be.any(be.isnan(result))
