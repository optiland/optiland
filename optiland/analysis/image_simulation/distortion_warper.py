"""Distortion warping for image simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be

from ..distortion_strategies import ChiefRayReferencePoint

if TYPE_CHECKING:
    from ..distortion_strategies import ReferencePointStrategy


class DistortionWarper:
    """
    Handles geometric distortion and lateral color by creating a warp map
    that transforms the ideal image coordinates to the distorted image plane.

    Args:
        optic (Optic): The optical system.
        source_fov (tuple): (max_x, max_y) of the source field in system units
                            (degrees for infinite, mm for finite).
                            If None, attempts to infer from optic.fields.max_field.
        reference_point (ReferencePointStrategy, optional): Strategy used to
            locate per-field image reference points. Defaults to
            :class:`ChiefRayReferencePoint` (chief-ray intercept). Supply a
            :class:`CentroidReferencePoint` to warp off-axis, freeform, or
            obscured systems where no chief ray can be traced.
    """

    def __init__(
        self,
        optic,
        source_fov=None,
        reference_point: ReferencePointStrategy | None = None,
    ):
        self.optic = optic
        self.reference_point = reference_point or ChiefRayReferencePoint()

        if source_fov is None:
            # Infer from optic
            # Assuming rotationally symmetric max field if single value
            max_f = self.optic.fields.max_field
            self.source_fov = (max_f, max_f)
        else:
            self.source_fov = source_fov

    def _poly_features(self, x, y, degree):
        """Generates polynomial features [1, x, y, x^2, xy, y^2, ...]"""
        features = []
        for d in range(degree + 1):
            for i in range(d + 1):
                j = d - i
                features.append((x**i) * (y**j))
        return be.stack(features, axis=1)

    def generate_distortion_map(
        self, wavelength, image_shape, num_grid_points=25, degree=5
    ):
        """
        Generates the sampling grid required by grid_sample to warp the source image
        using a polynomial fit to the distortion.
        """
        H, W = image_shape
        max_fx, max_fy = self.source_fov

        # 1. Trace Grid (normalized coordinates)
        linear = be.linspace(-1.0, 1.0, num_grid_points)
        gx, gy = be.meshgrid(linear, linear)
        gx_flat = gx.flatten()
        gy_flat = gy.flatten()

        # Physical field units
        phys_x = gx_flat * max_fx
        phys_y = gy_flat * max_fy

        # Normalize relative to Optic's full field for tracing
        optic_max = self.optic.fields.max_field
        hx_norm = phys_x / optic_max
        hy_norm = phys_y / optic_max

        # 2. Get Landing Coordinates (Real Image Plane) via the reference-point
        # strategy, so obscured/off-axis systems can be warped via the bundle
        # centroid when no chief ray exists.
        x_real, y_real = self.reference_point.locate(
            self.optic, hx_norm, hy_norm, wavelength
        )

        # Center relative to the field-center reference point
        cx, cy = self.reference_point.locate(self.optic, 0.0, 0.0, wavelength)
        x_real = x_real - cx[0]
        y_real = y_real - cy[0]

        # Normalize physical coordinates to [-1, 1] for a stable polynomial fit
        max_x = be.max(be.abs(x_real))
        max_y = be.max(be.abs(y_real))
        scale_x = max_x if float(max_x) > 0 else 1.0
        scale_y = max_y if float(max_y) > 0 else 1.0

        x_norm = x_real / scale_x
        y_norm = y_real / scale_y

        # 3. Fit Polynomial: (x_norm, y_norm) -> (gx, gy)
        X_features = self._poly_features(x_norm, y_norm, degree)

        # Solve X * c = gx  => c = lstsq(X, gx)
        c_gx = be.lstsq(X_features, gx_flat)
        c_gy = be.lstsq(X_features, gy_flat)

        # Check fit error to warn about discretization noise or poor fit
        pred_gx = be.matmul(X_features, c_gx)
        pred_gy = be.matmul(X_features, c_gy)
        err_gx = be.max(be.abs(pred_gx - gx_flat))
        err_gy = be.max(be.abs(pred_gy - gy_flat))

        max_err = float(be.max(be.array([err_gx, err_gy])))
        if max_err > 0.05:
            import warnings

            warnings.warn(
                f"Distortion mapping polynomial fit error is high (max error: "
                f"{max_err:.3f} in normalized field units). If using "
                "CentroidReferencePoint, consider increasing `num_rays` to "
                "reduce discretization noise.",
                UserWarning,
                stacklevel=2,
            )

        # 4. Evaluate on Target Grid (Detector Pixels)
        min_x_grid, max_x_grid = be.min(x_real), be.max(x_real)
        min_y_grid, max_y_grid = be.min(y_real), be.max(y_real)

        # Create target mesh (H, W)
        ty = be.linspace(max_y_grid, min_y_grid, H)
        tx = be.linspace(min_x_grid, max_x_grid, W)
        grid_x, grid_y = be.meshgrid(tx, ty)

        # Normalize target grid by the same scale factors
        grid_x_norm = grid_x.flatten() / scale_x
        grid_y_norm = grid_y.flatten() / scale_y

        X_grid = self._poly_features(grid_x_norm, grid_y_norm, degree)

        # Predict normalized coordinates for every pixel
        target_gx = be.matmul(X_grid, c_gx).reshape([H, W])
        target_gy = be.matmul(X_grid, c_gy).reshape([H, W])

        # Stack (H, W, 2) and add batch dim (1, H, W, 2)
        grid = be.stack((target_gx, -target_gy), axis=-1)
        return (
            grid.unsqueeze(0)
            if hasattr(grid, "unsqueeze")
            else be.array(grid[None, ...])
        )

    def warp_image(self, image, distortion_grid):
        """
        Warps the input image using the provided distortion grid.
        """
        image = be.array(image)
        distortion_grid = be.array(distortion_grid)

        # grid_sample expects (N, C, H, W) input
        ndim = image.ndim

        if ndim == 2:
            # (H, W) -> (1, 1, H, W)
            img_input = image[None, None, :, :]
        elif ndim == 3:
            # (B, H, W) -> (B, 1, H, W)
            img_input = image[:, None, :, :]
        elif ndim == 4:
            # (B, C, H, W)
            img_input = image
        else:
            raise ValueError(
                "image must have shape (H, W), (B, H, W), or (B, C, H, W)."
            )

        N = img_input.shape[0]
        if distortion_grid.shape[0] != N:
            # Tile the grid to match batch size
            distortion_grid = be.tile(distortion_grid, (N, 1, 1, 1))

        output = be.grid_sample(
            img_input,
            distortion_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # Restore shape
        if ndim == 2:
            return output[0, 0]
        if ndim == 3:
            return output[:, 0, :, :]
        return output
