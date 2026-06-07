from __future__ import annotations

import optiland.backend as be


class SpatiallyVariableSimulator:
    """
    Simulates image formation with spatially variable point spread functions (PSFs)
    using the EigenPSF method.

    This simulator decomposes the spatially variant PSF into a set of basis functions
    (EigenPSFs) and coefficient maps, efficiently computing the resulting image
    via weighted sums of convolutions.

    Attributes:
        optic (Optic): The optical system to simulate.
        wavelength (float): Wavelength of operation.
    """

    def __init__(self):
        pass

    def simulate(self, source_image, eigen_psfs, coefficient_maps, mean_psf):
        """
        Simulate the image using provided EigenPSFs and Coefficient Maps.

        Args:
            source_image (be.ndarray): The high-resolution source image (B, H, W).
            eigen_psfs (be.ndarray): Basis PSFs (K, P, P).
            coefficient_maps (be.ndarray): Spatial coefficient maps (K, H, W).
            mean_psf (be.ndarray): The average PSF (P, P).

        Returns:
            be.ndarray: The simulated image (B, H, W).
        """
        source_image = be.array(source_image)
        eigen_psfs = be.array(eigen_psfs)
        coefficient_maps = be.array(coefficient_maps)
        mean_psf = be.array(mean_psf)

        if source_image.ndim != 3:
            raise ValueError("source_image must have shape (B, H, W).")

        # 1. Base term: Convolve with mean PSF
        # Corresponds to 0-th order approximation
        final_image = be.fftconvolve(
            source_image,
            mean_psf[None, :, :],
            mode="same",
        )

        # 2. Variable terms
        # Pre-multiply: (Source * Coeff) * EigenPSF
        # This correctly models field-dependent PSF weight
        weighted_source = source_image[:, None, :, :] * coefficient_maps[None, :, :, :]

        convolved = be.fftconvolve(
            weighted_source,
            eigen_psfs[None, :, :, :],
            mode="same",
        )

        final_image += be.sum(convolved, axis=1)

        return final_image
