"""Image simulation engine combining PSF convolution and distortion."""

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
        config (dict): Configuration dictionary.
            - wavelength (list[float]): List of 3 wavelengths (um) for R, G, B.
            - psf_grid_shape (tuple): (ny, nx) for PSF basis generation.
            - psf_size (int): Pixel size for PSFs.
            - num_rays (int): Number of rays for PSF generation.
            - n_components (int): Number of EigenPSFs.
            - oversample (int): Upsampling factor for simulation accuracy.
            - padding (int): Pixel padding (guard band) to avoid edge artifacts.
            - psf_strategy (str): Wavefront reference strategy.
            - psf_remove_tilt (bool): Whether to remove wavefront tilt.
            - distortion_reference (ReferencePointStrategy): Strategy used to
              locate per-field image reference points for the distortion warp.
              Defaults to None (chief-ray intercept). Supply a
              CentroidReferencePoint to simulate off-axis, freeform, or obscured
              systems where no chief ray can be traced.
    """

    def __init__(self, optic, config=None):
        self.optic = optic
        self.source_image = None
        self.simulated_image = None

        self.config = {
            "wavelengths": [0.65, 0.55, 0.45],
            "psf_grid_shape": (5, 5),
            "psf_size": 128,
            "num_rays": 64,
            "n_components": 3,
            "oversample": 1,
            "padding": 64,
            "psf_strategy": "chief_ray",
            "psf_remove_tilt": False,
            "distortion_reference": None,
        }
        if config:
            self.config.update(config)

    def _prepare_source_image(self, source_image):
        if isinstance(source_image, str):
            import matplotlib.image as mpimg

            image = mpimg.imread(source_image)
            if image.ndim == 3 and image.shape[-1] == 4:
                image = image[:, :, :3]
        else:
            image = source_image

        image = be.array(image)

        if image.ndim == 2:
            return image[None, None, :, :]

        if image.ndim == 3:
            if image.shape[-1] != 3:
                raise ValueError("3D source_image must have shape (H, W, 3).")
            return be.transpose(image, (2, 0, 1))[None, :, :, :]

        if image.ndim == 4:
            if image.shape[1] not in (1, 3):
                raise ValueError("4D source_image must have shape (B, C, H, W).")
            return image

        raise ValueError(
            "source_image must have shape (H, W), (H, W, 3), or (B, C, H, W)."
        )

    def run(self, source_image):
        """
        Executes the simulation pipeline.

        Args:
            source_image (ArrayLike): The input source image with shape (H, W),
                                      (H, W, 3), or (B, C, H, W).

        Returns:
            be.ndarray: The simulated image batch with shape (B, C, H, W).
                        Values defined by input dynamic range.
        """
        self.source_image = self._prepare_source_image(source_image)

        # 1. Preprocessing
        # Pad and Upsample
        processed_input, pad_info = self._preprocess(self.source_image)

        _, C, H, W = processed_input.shape

        wavelengths = self.config["wavelengths"]
        # Handle grayscale input with 3 wavelengths -> treat as RGB result
        if C == 1 and len(wavelengths) == 3:
            input_channels = [processed_input[:, 0, :, :]] * 3
        else:
            # If input is RGB, match wavelengths 1-to-1
            input_channels = [
                processed_input[:, c, :, :] for c in range(min(C, len(wavelengths)))
            ]

        # 2. Simulation Loop per Channel
        processed_channels = []
        sim = SpatiallyVariableSimulator()
        warper = DistortionWarper(
            self.optic, reference_point=self.config["distortion_reference"]
        )

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
                strategy=self.config["psf_strategy"],
                remove_tilt=self.config["psf_remove_tilt"],
            )
            eigen_psfs, coeffs, mean_psf = gen.generate_basis(
                n_components=self.config["n_components"]
            )

            # Resize coeffs to image size
            coeffs_resized = gen.resize_coefficient_map(coeffs, (H, W))

            # B. Convolution (Blur)
            blurred = sim.simulate(channel_img, eigen_psfs, coeffs_resized, mean_psf)

            # C. Distortion (Warp)
            # Generate map for current wavelength (handles lateral color)
            dist_map = warper.generate_distortion_map(wave, (H, W))
            distorted = warper.warp_image(blurred, dist_map)

            processed_channels.append(distorted)

        final_output = be.stack(processed_channels, axis=1)

        # 3. Postprocessing
        # Downsample and Crop
        result = self._postprocess(final_output, pad_info)

        self.simulated_image = result
        return result

    def view(self, index: int = 0, *, show: bool = True):
        """
        Visualizes one original and simulated image side-by-side from the batch.

        Args:
            index (int): Batch index to visualize.
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.
        """
        if self.source_image is None or self.simulated_image is None:
            raise RuntimeError("Call run(source_image) before view().")

        batch_size = self.source_image.shape[0]
        if index < 0 or index >= batch_size:
            raise IndexError(f"index must be between 0 and {batch_size - 1}.")

        import matplotlib.pyplot as plt

        # Prepare selected image in batch for display (C, H, W) -> (H, W, C)
        src = self.source_image[index]
        src = be.transpose(src, (1, 2, 0))

        sim = self.simulated_image[index]
        sim = be.transpose(sim, (1, 2, 0))

        src_np = be.to_numpy(src)
        sim_np = be.to_numpy(sim)

        # Ensure correct range for display
        if src_np.max() > 2.0:
            src_np = src_np / 255.0
        if sim_np.max() > 2.0:
            sim_np = sim_np / 255.0

        src_np = np.clip(src_np, 0, 1)
        sim_np = np.clip(sim_np, 0, 1)

        if src_np.shape[-1] == 1:
            src_np = src_np[:, :, 0]
        if sim_np.shape[-1] == 1:
            sim_np = sim_np[:, :, 0]

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(src_np, cmap="gray" if src_np.ndim == 2 else None)
        ax[0].set_title(f"Original Image [{index}]")
        ax[0].axis("off")

        ax[1].imshow(sim_np, cmap="gray" if sim_np.ndim == 2 else None)
        ax[1].set_title(f"Simulated Image [{index}]")
        ax[1].axis("off")

        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def _preprocess(self, image):
        # Padding
        pad = self.config["padding"]

        # Padding: ((0,0), (0,0), (pad, pad), (pad, pad)) for (B, C, H, W)
        image = be.pad(
            image,
            ((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode="reflect",
        )

        # Upsampling
        scale = self.config["oversample"]
        if scale > 1:
            image_np = be.to_numpy(image)
            upsampled_np = zoom(image_np, (1, 1, scale, scale), order=1)
            image = be.array(upsampled_np)

        return image, (pad, scale)

    def _postprocess(self, image, pad_info):
        """Downsamples and crops the image."""
        pad, scale = pad_info

        # Downsample
        if scale > 1:
            if be.get_backend().__class__.__name__ == "TorchBackend":
                import torch.nn.functional as F

                image = F.interpolate(
                    image,
                    scale_factor=1 / scale,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                image_np = be.to_numpy(image)
                downsampled_np = zoom(image_np, (1, 1, 1 / scale, 1 / scale), order=1)
                image = be.array(downsampled_np)

        target_h, target_w = self.source_image.shape[-2:]

        start_y = pad
        start_x = pad

        crop = image[
            :,
            :,
            start_y : start_y + target_h,
            start_x : start_x + target_w,
        ]

        # Ensure values are within valid range (prevent small negative values)
        crop = be.maximum(crop, 0.0)

        return crop
