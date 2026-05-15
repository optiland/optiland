from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

import optiland.backend as be

from .distortion_warper import DistortionWarper
from .psf_basis_generator import PSFBasisGenerator
from .simulator import SpatiallyVariableSimulator


class ImageSimulationEngine:
    """
    Master engine for performing full image simulation including spatially
    variable blur, geometric distortion, and lateral color.

    Args:
        optic (Optic): The optical system model.
        source_image (ArrayLike): The input source image (H, W, 3) or (H, W).
                                  Expected to be in RGB format if 3 channels.
        config (dict): Configuration dictionary.
            - wavelength (list[float]): List of 3 wavelengths (um) for R, G, B.
            - psf_grid_shape (tuple): (ny, nx) for PSF basis generation.
            - psf_size (int): Pixel size for PSFs.
            - num_rays (int): Number of rays for PSF generation.
            - n_components (int): Number of EigenPSFs.
            - oversample (int): Upsampling factor for simulation accuracy.
            - padding (int): Pixel padding (guard band) to avoid edge artifacts.
    """

    def __init__(self, optic, source_image, config=None):
        self.optic = optic
        self.simulated_image = None

        # Load image if path string
        if isinstance(source_image, str):
            import matplotlib.image as mpimg

            img = mpimg.imread(source_image)
            # Handle alpha channel if present (remove it for now)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
        else:
            img = source_image

        # Ensure source is (C, H, W) or (H, W) backend array
        img = be.array(img)
        if img.ndim == 3 and img.shape[2] == 3:
            # (H, W, 3) -> (3, H, W)
            img = be.transpose(img, (2, 0, 1))
        elif img.ndim == 2:
            # Monochromatic/Grayscale -> (1, H, W)
            img = img[None, :, :]

        self.source_image = img

        # Default config
        self.config = {
            "wavelengths": [0.65, 0.55, 0.45],  # R, G, B standard approx
            "psf_grid_shape": (5, 5),
            "psf_size": 128,
            "num_rays": 64,  # Optimized for performance (was 128)
            "n_components": 3,
            "oversample": 1,
            "padding": 64,
        }
        if config:
            self.config.update(config)

    def run(self):
        """
        Executes the simulation pipeline.

        Returns:
            be.ndarray: The simulated image (H, W, C) or (H, W).
                        Values defined by input dynamic range.
        """
        # 1. Preprocessing
        # Pad and Upsample
        processed_input, pad_info = self._preprocess(self.source_image)

        C, H, W = processed_input.shape
        final_output = be.zeros_like(processed_input)

        wavelengths = self.config["wavelengths"]
        # Handle grayscale input with 3 wavelengths -> treat as RGB result
        if C == 1 and len(wavelengths) == 3:
            final_output = be.zeros((3, H, W), dtype=processed_input.dtype)
            input_channels = [processed_input[0]] * 3
        else:
            # If input is RGB, match wavelengths 1-to-1
            input_channels = [
                processed_input[c] for c in range(min(C, len(wavelengths)))
            ]

        # 2. Simulation Loop per Channel
        processed_channels = []
        for _i, (wave, channel_img) in enumerate(
            zip(wavelengths, input_channels, strict=False)
        ):
            # A. Basis Generation
            gen = PSFBasisGenerator(
                self.optic,
                wavelength=wave,
                grid_shape=self.config["psf_grid_shape"],
                num_rays=self.config["num_rays"],
                psf_grid_size=self.config["psf_size"],
            )
            eigen_psfs, coeffs, mean_psf = gen.generate_basis(
                n_components=self.config["n_components"]
            )

            # Resize coeffs to image size
            coeffs_resized = gen.resize_coefficient_map(coeffs, (H, W))

            # B. Convolution (Blur)
            sim = SpatiallyVariableSimulator()
            blurred = sim.simulate(channel_img, eigen_psfs, coeffs_resized, mean_psf)

            # C. Distortion (Warp)
            warper = DistortionWarper(self.optic)
            # Generate map for current wavelength (handles lateral color)
            dist_map = warper.generate_distortion_map(wave, (H, W))
            distorted = warper.warp_image(blurred, dist_map)

            processed_channels.append(distorted)

        final_output = be.stack(processed_channels, axis=0)

        # 3. Postprocessing
        # Downsample and Crop
        result = self._postprocess(final_output, pad_info)

        # Return (H, W, C) for image format compatibility
        if result.ndim == 3:
            result = be.transpose(result, (1, 2, 0))

        self.simulated_image = result
        return result

    def view(self, force_rerun=False, *, show: bool = True):
        """
        Visualizes the original and simulated images side-by-side.
        Runs the simulation if it hasn't been run yet or if force_rerun is True.

        Args:
            force_rerun: Re-run the simulation even if results already exist.
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.
        """
        if self.simulated_image is None or force_rerun:
            self.run()

        import matplotlib.pyplot as plt

        # Prepare source for display (C, H, W) -> (H, W, C) using backend generic
        src = self.source_image
        if src.ndim == 3:
            src = be.transpose(src, (1, 2, 0))

        src_np = be.to_numpy(src)
        sim_np = be.to_numpy(self.simulated_image)

        # Ensure correct range for display
        if src_np.max() > 2.0:
            src_np = src_np / 255.0
        if sim_np.max() > 2.0:
            sim_np = sim_np / 255.0

        src_np = np.clip(src_np, 0, 1)
        sim_np = np.clip(sim_np, 0, 1)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(src_np, cmap="gray" if src_np.ndim == 2 else None)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(sim_np, cmap="gray" if sim_np.ndim == 2 else None)
        ax[1].set_title("Simulated Image")
        ax[1].axis("off")

        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def _preprocess(self, image):
        # Padding
        pad = self.config["padding"]

        # Padding: ((0,0), (pad, pad), (pad, pad)) for (C, H, W)
        image_np = be.to_numpy(image)
        padded_np = np.pad(image_np, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")

        # Upsampling
        scale = self.config["oversample"]
        if scale > 1:
            upsampled_np = zoom(padded_np, (1, scale, scale), order=1)
        else:
            upsampled_np = padded_np

        return be.array(upsampled_np), (pad, scale)

    def _postprocess(self, image, pad_info):
        """Downsamples and crops the image."""
        pad, scale = pad_info

        # Downsample
        if scale > 1:
            image_np = be.to_numpy(image)
            downsampled_np = zoom(image_np, (1, 1 / scale, 1 / scale), order=1)
            image = be.array(downsampled_np)

        target_h, target_w = self.source_image.shape[-2:]

        start_y = pad
        start_x = pad

        crop = image[:, start_y : start_y + target_h, start_x : start_x + target_w]

        # Ensure values are within valid range (prevent small negative values)
        crop = be.maximum(crop, 0.0)

        return crop
