"""MMDFT Point Spread Function (PSF) Module

This module provides functionality for simulating and analyzing the Point
Spread Function (PSF) of optical systems using the Matrix Multiply Discrete Fourier
Transform. It includes capabilities for generating a monochromatic PSF at a desired
image size and pixel pitch from given wavefront aberrations and calculating the
Strehl ratio. Visualization is handled by the base class.

Scott Paine, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.psf._fft_sampling import (
    pupil_direction_matrix,
    pupil_frequency_steps,
)
from optiland.psf.base import BasePSF
from optiland.psf.fft import calculate_grid_size


class MMDFTPSF(BasePSF):
    """Class representing the Matrix Multiply Discrete Fourier Transform (MMDFT) PSF.

    This class computes the PSF of an optical system by taking the Fourier
    Transform of the pupil function. It inherits common visualization and
    initialization functionalities from `BasePSF`.

    Args:
        optic (Optic): The optical system object, containing properties like
            paraxial data and surface information.
        field (tuple): The field point (e.g., (Hx, Hy) in normalized field
            coordinates) at which to compute the PSF.
        wavelength (str | float): The wavelength of light in micrometers. Can be
            'primary' or a float value.
        num_rays (int, optional): The number of rays used to sample the pupil
            plane along one dimension. The pupil will be a grid of
            `num_rays` x `num_rays`. Defaults to 128.
        image_size (int, optional): The size of the image plane, in pixels. If not
            specified, it is calculated based on `num_rays`.
        pixel_pitch (float, optional): The size of the pixels in the image plane. If
            not specified, it is calculated based on image_size.
        strategy (str): The calculation strategy to use. Supported options are
            "chief_ray", "centroid_sphere", and "best_fit_sphere".
            Defaults to "chief_ray".
        remove_tilt (bool): If True, removes tilt and piston from the OPD data.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the strategy.

    Attributes:
        pupil (be.ndarray): A complex-valued pupil function based on the wavelength
            used in the `Wavefront` parent. The pupil is a 2D array.
        psf (be.ndarray): The computed Point Spread Function. This is a 2D
            array representing the intensity distribution in the image plane,
            normalized such that a diffraction-limited system has a peak of 100.
        image_size (int): The output size of the image after DFT calculation.
        pixel_pitch (float): The resolution of the image after DFT calculation.
        num_rays (int): The number of rays used to sample the pupil. This is
                        the `num_rays` passed during initialization and used
                        by `Wavefront` for generating OPD/intensity data.
    """

    def __init__(
        self,
        optic,
        field,
        wavelength: str | float,
        num_rays=128,
        image_size=None,
        pixel_pitch=None,
        strategy="chief_ray",
        remove_tilt=False,
        **kwargs,
    ):
        if image_size is None and pixel_pitch is None:
            if num_rays < 32:
                raise ValueError(
                    "num_rays must be at least 32 if image_size and pixel_pitch are "
                    "not specified."
                )
            num_rays, grid_size = calculate_grid_size(num_rays)

        super().__init__(
            optic=optic,
            field=field,
            wavelength=wavelength,
            num_rays=num_rays,
            strategy=strategy,
            remove_tilt=remove_tilt,
            **kwargs,
        )

        # General idea for below:
        #  - If only pixel_pitch is None, calculate based off of image_size, assuming
        #    image_size is full pad size
        #  - If only image_size is None, calculate based off of pixel_pitch,
        #    truncating any trailing decimals
        #  - If both pixel_pitch and image_size are none, set image_size = grid_size
        #    and solve for pixel_pitch
        # I would implement this as a function, but since it needs access to the FNO,
        # it has to happen after super() initialization, which requires knowing
        # num_rays ahead of time
        #
        # The equivalent padded DFT size along an axis is 1 / (pixel_pitch * step),
        # with `step` the image-plane frequency of one pupil sample along that
        # axis. On axis both steps equal 1 / (wavelength * FNO_w * (num_rays - 1)).
        # Off axis they differ; the larger step sets the alias-free limit.
        # The working F/# is no longer needed here, but computing it rejects
        # systems whose marginal rays fail to trace, as before.
        self._get_working_FNO()
        max_step = max(self._frequency_steps())
        if pixel_pitch is None:
            if image_size is None:
                # Use grid_size above to set Q and therefore pixel pitch
                image_size = grid_size
            pixel_pitch = 1e3 / (image_size * max_step)

        # Below triggers only if pixel_pitch was given but image_size was not
        if image_size is None:
            # Use pixel_pitch to calculate max pad size and truncate the image
            # size to it
            # (relative slack absorbs round-off when the ratio is an integer)
            image_size = int(1e3 / (pixel_pitch * max_step) * (1 + 1e-9))

        self.image_size = image_size
        self.pixel_pitch = pixel_pitch
        self.pupil = self._generate_pupil()
        self.psf = self._compute_psf()

    def _generate_pupil(self):
        """Generates complex pupil functions for each wavelength. Copied from FFTPSF.

        For the specified wavelength, this method:
        1. Obtains wavefront data (Optical Path Difference - OPD, intensity)
           at the exit pupil using the `get_data` method from `Wavefront`.
        2. Creates a 2D grid representing the pupil, sampled with `self.num_rays`
           points across its diameter.
        3. Populates the grid with complex values: `A * exp(j * phi)`, where
           amplitude `A` is derived from ray intensity and phase `phi` from OPD.
           The OPD is converted to phase using the wavelength.

        Returns:
            be.ndarray: A complex 2D array (shape: `num_rays` x `num_rays`),
            representing the pupil function for the given wavelength.
        """
        x = be.linspace(-1, 1, self.num_rays)
        x, y = be.meshgrid(x, x)
        x = x.ravel()
        y = y.ravel()
        R2 = x**2 + y**2

        field = self.fields[0]  # PSF contains a single field.
        wl = self.wavelengths[0]  # PSF contains a single wavelength.

        wavefront_data = self.get_data(field, wl)
        P = be.to_complex(be.zeros_like(x))

        amplitude = be.sqrt(wavefront_data.intensity)
        P[R2 <= 1] = be.to_complex(
            amplitude * be.exp(-1j * 2 * be.pi * wavefront_data.opd)
        )
        pupil = be.reshape(P, (self.num_rays, self.num_rays))

        return pupil

    def _compute_psf(self):
        """Computes the PSF from the generated pupil function via matrix DFTs.

        This involves:
        1. Evaluating the image field G on a grid with spacing `pixel_pitch`
           centred on the chief ray (see `_compute_image_field`). When the
           pupil-to-direction map is diagonal, this is the matrix
           triple-product G = L g R, where g is the pupil field.
        2. Taking the absolute square of the image field G to calculate intensity.
        3. Scaling the result by a calculated normalization factor to ensure
           consistent Strehl ratio (diffraction-limited peak = 100%).

        Returns:
            be.ndarray: The computed 2D PSF (shape: `image_size` x `image_size`),
            normalized so that a diffraction-limited system's peak is 100.
        """
        image_plane = self._compute_image_field()
        psf = image_plane * be.conj(image_plane)

        return be.real(psf) * 100 / self._get_normalization()

    def _get_normalization(self):
        """Calculates the normalization factor for the PSF.

        This factor ensures that an ideal, diffraction-limited system (no
        aberrations, uniform unit amplitude transmission across the pupil)
        would have a peak PSF intensity corresponding to a Strehl ratio of 1.0
        (or 100% when scaled by `_compute_psf`).

        The normalization is based on the peak intensity of a PSF computed
        from an idealized pupil: one with uniform amplitude (1.0) and zero
        phase within the aperture defined by the actual system's first pupil,
        and zero outside.

        Since the maximum of the PSF of an ideal pupil is always centered on (0, 0),
        the value of the propagation kernel is always 1 and therefore the Fourier
        integral just becomes a sum of the input field over all points. For a
        binary-valued aperture, this is the same as just counting all the non-zero
        pixels (summation) and taking the square of the value (to go from field to PSF)

        Returns:
            float: The normalization factor.
        """
        return be.sum(be.abs(self.pupil) > 0) ** 2

    def strehl_ratio(self):
        """Computes the Strehl ratio of the PSF.

        The Strehl ratio is the ratio of the peak intensity of the aberrated
        PSF to the peak intensity of the diffraction-limited PSF.
        Assumes self.psf is normalized such that its peak would be 1.0 (or 100%)
        for a diffraction-limited system.

        Returns:
            float: The Strehl ratio.

        Raises:
            RuntimeError: If the PSF has not been computed.
        """
        if self.psf is None:
            raise RuntimeError("PSF has not been computed.")

        # Make sure to use max! The PSF may not be centered in the image plane
        return be.max(self.psf) / 100

    def _frequency_steps(self) -> tuple[float, float]:
        """Image-plane frequency of one pupil sample along pupil Y and X.

        See `optiland.psf._fft_sampling.pupil_frequency_steps`.

        Returns:
            tuple[float, float]: The steps along pupil Y and pupil X, in
            cycles/mm.
        """
        step_y, step_x = pupil_frequency_steps(
            self.optic,
            self.fields[0].coord,
            self.wavelengths[0].value,
            self.num_rays,
        )
        return float(be.to_numpy(step_y)), float(be.to_numpy(step_x))

    def _compute_image_field(self):
        """Evaluate the scalar image field on the output grid by matrix DFTs.

        In the Fraunhofer approximation the image field at offset
        ``r = (x, y)`` from the chief ray is
        ``G(r) = sum_p g(p) exp(i k r . (A p))``, where ``g`` is the pupil
        field at normalized pupil coordinates ``p = (Px, Py)``,
        ``k = 2 pi / wavelength`` and ``A`` maps ``p`` to the image-space
        direction cosines times the index (see
        `optiland.psf._fft_sampling.pupil_direction_matrix`). The output grid
        has ``image_size`` samples per axis spaced by exactly ``pixel_pitch``,
        with sample ``image_size // 2`` on the chief ray.

        When ``A`` is diagonal (fields on the x or y axis of a rotationally
        symmetric system), the sum separates into the matrix triple-product
        ``G = L g R``. Otherwise the off-diagonal terms couple the axes, and
        the field is evaluated one image row at a time, which costs about
        ``image_size`` times more. The separable form is used whenever the
        phase from the off-diagonal terms stays below 1e-6 rad over the grid.

        In order to be in line with FFTPSF, the pupil is sampled by `num_rays`
        points across its diameter, assuming all rays trace through the
        system unclipped.

        Returns:
            be.ndarray: The complex image field, shape
            ``(image_size, image_size)``; rows are image y, columns image x.

        Raises:
            ValueError: If the image grid extends beyond the alias-free extent
                set by the pupil sampling along either axis.
        """
        # The pupil field repeats in the image plane with a period of
        # 1 / step along each axis; the grid must stay within one period.
        # A relative slack of 1e-9 absorbs round-off when the default pitch
        # makes the pad size exactly `image_size`.
        pad_size = 1e3 / (self.pixel_pitch * max(self._frequency_steps())) * (1 + 1e-9)
        if self.image_size > pad_size:
            max_size = int(pad_size)  # truncate
            raise ValueError(
                f"Supplied image_size of {self.image_size} not less than or equal to "
                f"calculated pad size of {max_size}. Consider increasing num_rays."
            )

        A = pupil_direction_matrix(
            self.optic, self.fields[0].coord, self.wavelengths[0].value
        )
        k = 2 * be.pi / (self.wavelengths[0].value * 1e-3)  # 1/mm
        # Pupil sample coordinates (same linspace as `_generate_pupil`) and
        # image offsets from the chief ray in mm.
        pupil_coordinates = be.linspace(-1, 1, self.num_rays)
        image_coordinates = (
            (be.arange(self.image_size) - self.image_size // 2)
            * self.pixel_pitch
            * 1e-3
        )

        # Pupil rows are pupil Y and columns pupil X (see `_generate_pupil`).
        # right[px, x] = exp(i k A_xx Px x),  left[y, py] = exp(i k A_yy Py y)
        right_kernel = be.to_complex(
            be.exp(1j * k * A[0, 0] * be.outer(pupil_coordinates, image_coordinates))
        )

        r_max = float(be.to_numpy(be.max(be.abs(image_coordinates))))
        off_diagonal_phase = (
            k * r_max * float(be.to_numpy(be.abs(A[0, 1]) + be.abs(A[1, 0])))
        )
        if off_diagonal_phase < 1e-6:
            left_kernel = be.to_complex(
                be.exp(
                    1j * k * A[1, 1] * be.outer(image_coordinates, pupil_coordinates)
                )
            )
            return be.matmul(left_kernel, be.matmul(self.pupil, right_kernel))

        # Coupled axes: for each image row y,
        # G[y, x] = sum_py exp(i k (A_xy x + A_yy y) Py)
        #           * sum_px g[py, px] exp(i k A_yx y Px) exp(i k A_xx Px x)
        rows = []
        for y in image_coordinates:
            row_modulation = be.to_complex(
                be.exp(1j * k * A[1, 0] * y * pupil_coordinates)
            )
            partial = be.matmul(self.pupil * row_modulation[None, :], right_kernel)
            column_phase = be.to_complex(
                be.exp(
                    1j
                    * k
                    * be.outer(
                        pupil_coordinates, A[0, 1] * image_coordinates + A[1, 1] * y
                    )
                )
            )
            rows.append(be.sum(column_phase * partial, axis=0)[None, :])
        # concatenate, not stack: the torch backend's stack casts to a real dtype
        return be.concatenate(rows, axis=0)

    def _get_psf_units(self, image):
        """Calculates the physical extent (units) of the PSF image for plotting.

        This method is called by `BasePSF.view()` to determine axis labels.
        It computes the total spatial width and height (in micrometers) of the
        provided PSF image data directly using the provided pixel_pitch.

        Args:
            image (be.ndarray): The PSF image data (often a
                zoomed/cropped version from `BasePSF.view`). Its shape is used
                to determine the total extent for labeling.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the physical
            total width and total height of the PSF image area, in micrometers.
            These are returned as NumPy arrays as `BasePSF.view` expects them
            for Matplotlib's `extent` argument.
        """
        dx = self.pixel_pitch

        x = be.to_numpy(image.shape[1] * dx)
        y = be.to_numpy(image.shape[0] * dx)

        return x, y
