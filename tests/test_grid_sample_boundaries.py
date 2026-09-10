from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.analysis.image_simulation import DistortionWarper
from tests.utils import assert_allclose


@pytest.mark.parametrize("align_corners", [False, True])
def test_bilinear_zero_padding_keeps_partial_edge_contributions(
    set_test_backend, align_corners
):
    image = be.array(np.ones((1, 1, 2, 3)))
    # Pixel coordinates: corner, edge, interior, opposite corner, outside.
    pixels = np.array([[-0.5, -0.5], [0.0, -0.5], [1.0, 0.5], [2.5, 1.5], [-1.1, 0.5]])
    if align_corners:
        normalized = 2 * pixels / np.array([2, 1]) - 1
    else:
        normalized = 2 * (pixels + 0.5) / np.array([3, 2]) - 1
    grid = be.array(normalized[None, None, ...])

    result = be.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners
    )

    assert_allclose(result, np.array([[[[0.25, 0.5, 1.0, 0.25, 0.0]]]]))


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("shape", [(3, 5), (1, 5), (3, 1)])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_bilinear_zero_padding_matches_torch(
    set_test_backend, align_corners, shape, precision
):
    torch = pytest.importorskip("torch")
    try:
        be.set_precision(precision)
        rng = np.random.default_rng(775)
        image_np = rng.normal(size=(2, 3, *shape)).astype(precision)
        grid_np = rng.uniform(-2, 2, size=(2, 4, 7, 2)).astype(precision)
        image = be.array(image_np)
        grid = be.array(grid_np)

        result = be.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        )
        expected = torch.nn.functional.grid_sample(
            torch.from_numpy(image_np),
            torch.from_numpy(grid_np),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        )

        tolerance = 1e-6 if precision == "float32" else 1e-12
        assert result.shape == (2, 3, 4, 7)
        assert result.dtype == image.dtype
        assert_allclose(result, expected.numpy(), rtol=tolerance, atol=tolerance)
    finally:
        be.set_precision("float64")


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
@pytest.mark.parametrize("padding_mode", ["border", "reflection"])
def test_other_padding_modes_keep_boundary_values(set_test_backend, mode, padding_mode):
    image = be.array(np.ones((1, 1, 2, 2)))
    grid = be.array([[[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [-1.5, 1.5]]]])

    result = be.grid_sample(
        image, grid, mode=mode, padding_mode=padding_mode, align_corners=False
    )

    assert_allclose(result, np.ones((1, 1, 1, 4)))


def test_nearest_zero_padding_keeps_existing_sampling(set_test_backend):
    image = be.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    grid = be.array([[[[-0.5, -0.5], [0.5, 0.5], [-2.0, 0.0]]]])

    result = be.grid_sample(image, grid, mode="nearest", padding_mode="zeros")

    assert_allclose(result, np.array([[[[1.0, 4.0, 0.0]]]]))


@pytest.mark.parametrize("shape", [(2, 2), (2, 2, 2), (2, 3, 2, 2)])
def test_distortion_warper_preserves_partial_edge_brightness(set_test_backend, shape):
    warper = DistortionWarper(optic=None, source_fov=(1.0, 1.0))
    image = be.array(np.ones(shape))
    grid = be.array([[[[-1.0, -1.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]]])

    result = warper.warp_image(image, grid)

    expected = np.broadcast_to([0.25, 0.5, 1.0, 0.25], (*shape[:-2], 1, 4))
    assert result.shape == expected.shape
    assert result.dtype == image.dtype
    assert_allclose(result, expected)
