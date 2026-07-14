"""Generates a field-dependent PSF basis for image simulation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

import optiland.backend as be
from optiland.psf.fft import FFTPSF


class PSFBasisGenerator:
    """
    Generates a basis of EigenPSFs for an optical system using SVD.

    This class computes a grid of PSFs and decomposes them to find the
    EigenPSFs that efficiently represent the spatially varying blur.

    Args:
        optic (Optic): The optical system to analyze.
        wavelength (float): The wavelength for PSF calculation.
        grid_shape (tuple, optional): (ny, nx) size of sampling grid. Default: (5, 5).
        num_rays (int, optional): Number of rays for pupil sampling. Default: 128.
        psf_grid_size (int, optional): PSF grid size (e.g., 256 for 256x256).

            If None, calculated from num_rays.
        strategy (str, optional): Wavefront reference strategy.
            Default: "chief_ray".
        remove_tilt (bool, optional): Whether to remove wavefront tilt.
            Default: False.
    """

    def __init__(
        self,
        optic,
        wavelength,
        grid_shape=(5, 5),
        num_rays=128,
        psf_grid_size=None,
        strategy="chief_ray",
        remove_tilt=False,
    ):
        self.optic = optic
        self.wavelength = wavelength
        self.grid_shape = grid_shape
        self.num_rays = num_rays
        self.psf_grid_size = psf_grid_size
        self.strategy = strategy
        self.remove_tilt = remove_tilt

    def generate_basis(self, n_components=3):
        """
        Computes the EigenPSFs and their corresponding coefficient maps.

        Args:
            n_components (int): Number of principal components (EigenPSFs) to keep.

        Returns:
        Returns:
            tuple:
                - eigen_psfs (be.ndarray): Basis PSFs, shape (n_components, H, W).
                - coefficient_grid (be.ndarray): Coefficient maps on low-res grid,
                  shape (n_components, grid_ny, grid_nx).
                - mean_psf (be.ndarray): Average PSF across field, shape (H, W).

        """
        # 1. Generate Grid of PSFs
        psf_stack = self._compute_psf_grid()
        n_psfs, h, w = psf_stack.shape

        # 2. Flatten for PCA
        # X shape: (n_samples, n_features) -> (n_psfs, h*w)
        X = be.reshape(psf_stack, (n_psfs, -1))

        # 3. Center the data
        mean_psf_flat = be.mean(X, axis=0)
        X_centered = X - mean_psf_flat

        # 4. SVD Decomposition
        # X_centered = U * S * Vt

        try:
            U, S, Vt = be.linalg.svd(X_centered, full_matrices=False)
        except AttributeError:
            # Fallback if be.linalg is not directly exposed as expected

            if be.get_backend() == "torch":
                import torch

                U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
            else:
                import numpy as np

                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 5. Extract top N components
        eigen_psfs_flat = Vt[:n_components]
        coeffs_flat = (
            U[:, :n_components] * S[:n_components]
        )  # (n_samples, n_components)

        # 6. Reshape results
        eigen_psfs = be.reshape(eigen_psfs_flat, (n_components, h, w))
        mean_psf = be.reshape(mean_psf_flat, (h, w))

        # Reshape coeffs back to the spatial grid (n_components, grid_ny, grid_nx)
        coeffs_t = be.transpose(coeffs_flat, (1, 0))
        coefficient_grid = be.reshape(
            coeffs_t, (n_components, self.grid_shape[0], self.grid_shape[1])
        )

        return eigen_psfs, coefficient_grid, mean_psf

    def _compute_psf_grid(self):
        """Generates the stack of PSFs across the field of view."""
        psfs = []
        ny, nx = self.grid_shape

        # Iterate over normalized field coordinates [-1, 1]
        ys = np.linspace(-1, 1, ny)
        xs = np.linspace(-1, 1, nx)

        for y in ys:
            for x in xs:
                field = (x, y)

                # Compute PSF
                psf_calc = FFTPSF(
                    optic=self.optic,
                    field=field,
                    wavelength=self.wavelength,
                    num_rays=self.num_rays,
                    grid_size=self.psf_grid_size,
                    strategy=self.strategy,
                    remove_tilt=self.remove_tilt,
                )

                # Normalize sum to 1 to treat as probability distribution
                raw_psf = psf_calc.psf
                norm_psf = raw_psf / be.sum(raw_psf)
                psfs.append(norm_psf)

        return be.stack(psfs)

    @staticmethod
    def resize_coefficient_map(coeff_map, target_shape):
        """
        Resizes the coefficient map to the target shape using bicubic interpolation.

        Args:
            coeff_map (be.ndarray): Input map (H_in, W_in) or (C, H_in, W_in).
            target_shape (tuple): (H_out, W_out).

        Returns:
            be.ndarray: Resized map.
        """
        # Check backend
        backend = be.get_backend()

        if backend == "torch":
            import torch.nn.functional as F

            is_3d = coeff_map.ndim == 3
            if not is_3d:
                coeff_map = coeff_map.unsqueeze(0)  # (1, H, W)
            inp = coeff_map.unsqueeze(0)

            out = F.interpolate(
                inp, size=target_shape, mode="bicubic", align_corners=False
            )

            out = out.squeeze(0)
            if not is_3d:
                out = out.squeeze(0)
            return out

        else:
            input_shape = coeff_map.shape
            if len(input_shape) == 3:
                # (C, H, W) -> zoom last two dims only
                h, w = input_shape[-2:]
                zoom_h = target_shape[0] / h
                zoom_w = target_shape[1] / w
                return be.array(
                    zoom(be.to_numpy(coeff_map), (1, zoom_h, zoom_w), order=1)
                )
            else:
                h, w = input_shape
                zoom_h = target_shape[0] / h
                zoom_w = target_shape[1] / w
                return be.array(zoom(be.to_numpy(coeff_map), (zoom_h, zoom_w), order=1))
