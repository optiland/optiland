"""FFT Point Spread Function (PSF) Module

This module provides functionality for simulating and analyzing the Point
Spread Function (PSF) of optical systems using the Fast Fourier Transform
(FFT). It includes capabilities for generating PSF from given wavefront
aberrations and calculating the Strehl ratio. Visualization is handled by
the base class.

Kramer Harrison, 2023
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.psf.base import BasePSF


def calculate_grid_size(num_rays) -> tuple[int, int]:
    """Calculates the effective pupil sampling and grid size based on the number of
    rays.

    See https://ansyshelp.ansys.com/public/account/secured?returnurl=/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/FFT_PSF.html
    for details on OpticStudio's FFT PSF sampling behavior.

    Args:
        num_rays (int): The number of rays used to sample the pupil.

    Returns:
        tuple[int, int]: A two-element tuple of

        - the effective pupil sampling size, i.e. the number of rays used to
          sample the pupil in one dimension, and
        - the grid size used for FFT computation.
    """
    effective_pupil_sampling = np.floor(32 * 2 ** ((np.log2(num_rays) - 5) / 2)).astype(
        int
    )
    grid_size = num_rays * 2

    return effective_pupil_sampling, grid_size


class ScalarFFTPSF(BasePSF):
    """Class representing the scalar Fast Fourier Transform (FFT) PSF.

    This class computes the PSF of an optical system by taking the Fourier
    Transform of the pupil function. It inherits common visualization and
    initialization functionalities from `BasePSF`.

    If no grid size is specified, OpticStudio's FFT PSF sampling behavior is
    emulated by scaling down the number of rays in the pupil and using a
    grid size of `num_rays * 2`.

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
        grid_size (int, optional): The size of the grid used for FFT
            computation (includes zero-padding). This determines the
            resolution of the PSF. Defaults to 1024. If not specified,
            it is calculated based on `num_rays`.
        strategy (str): The calculation strategy to use. Supported options are
            "chief_ray", "centroid_sphere", and "best_fit_sphere".
            Defaults to "chief_ray".
        remove_tilt (bool): If True, removes tilt and piston from the OPD data.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the strategy.

    Attributes:
        pupils (list[be.ndarray]): A list of complex-valued pupil functions,
            one for each wavelength used in the `Wavefront` parent. Each pupil
            is a 2D array.
        psf (be.ndarray): The computed Point Spread Function. This is a 2D
            array representing the intensity distribution in the image plane,
            normalized such that a diffraction-limited system has a peak of 100.
        grid_size (int): The size of the grid used for FFT computation.
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
        grid_size=None,
        strategy="chief_ray",
        remove_tilt=False,
        **kwargs,
    ):
        if grid_size is None:
            if num_rays < 32:
                raise ValueError(
                    "num_rays must be at least 32 if grid_size is not specified."
                )
            num_rays, grid_size = calculate_grid_size(num_rays)
        elif grid_size < num_rays:
            raise ValueError(
                f"Grid size ({grid_size}) must be greater than or equal to the "
                f"number of rays ({num_rays})."
            )

        super().__init__(
            optic=optic,
            field=field,
            wavelength=wavelength,
            num_rays=num_rays,
            strategy=strategy,
            remove_tilt=remove_tilt,
            **kwargs,
        )
        self.grid_size = grid_size
        self.pupils = self._generate_pupils()
        self.psf = self._compute_psf()

    def _generate_pupils(self):
        """Generates complex pupil functions for each wavelength.

        For the specified wavelength, this method:
        1. Obtains wavefront data (Optical Path Difference - OPD, intensity)
           at the exit pupil using the `get_data` method from `Wavefront`.
        2. Creates a 2D grid representing the pupil, sampled with `self.num_rays`
           points across its diameter.
        3. Populates the grid with complex values: `A * exp(j * phi)`, where
           amplitude `A` is derived from ray intensity and phase `phi` from OPD.
           The OPD is converted to phase using the wavelength.

        Returns:
            list[be.ndarray]: A list of complex 2D arrays (shape:
            `num_rays` x `num_rays`), each representing the pupil function
            for a wavelength.
        """
        x = be.linspace(-1, 1, self.num_rays)
        x, y = be.meshgrid(x, x)
        x = x.ravel()
        y = y.ravel()
        R2 = x**2 + y**2

        field = self.fields[0]  # PSF contains a single field.
        pupils = []

        for wl in self.wavelengths:
            wavefront_data = self.get_data(field, wl)
            P = be.to_complex(be.zeros_like(x))

            amplitude = be.sqrt(wavefront_data.intensity)

            P[R2 <= 1] = be.to_complex(
                amplitude * be.exp(-1j * 2 * be.pi * wavefront_data.opd)
            )
            P = be.reshape(P, (self.num_rays, self.num_rays))
            pupils.append(P)

        return pupils

    def _compute_psf(self):
        """Computes the PSF from the generated pupil functions via FFT.

        This involves:
        1. Padding the pupil functions with zeros up to `self.grid_size`.
           This padding determines the PSF's resolution.
        2. Calculating a normalization factor to ensure consistent Strehl ratio
           (diffraction-limited peak = 100%).
        3. For each (padded) pupil:
           a. Performing a 2D FFT and applying `fftshift` to center it.
           b. Calculating the squared magnitude (intensity) of the result.
        4. Normalizing the final PSF using the pre-calculated factor.

        Returns:
            be.ndarray: The computed 2D PSF (shape: `grid_size` x `grid_size`),
            normalized so that a diffraction-limited system's peak is 100.

        Raises:
            ValueError: If pupil functions have not been generated.
        """
        if not self.pupils:
            raise ValueError(
                "Pupil functions have not been generated prior to _compute_psf call."
            )

        pupils = self._pad_pupils()
        norm_factor = self._get_normalization()

        psf = []
        for pupil in pupils:
            amp = be.fft.fftshift(be.fft.fft2(pupil))
            psf.append(be.real(amp * be.conj(amp)))
        psf = be.stack(psf)

        return be.real(be.sum(psf, axis=0)) / norm_factor * 100

    def _pad_pupils(self):
        """Pads the pupil functions with zeros to match `self.grid_size`.

        Zero-padding in the spatial domain (pupil plane) before FFT leads to
        interpolation in the frequency domain (PSF plane). This effectively
        increases the sampling resolution of the computed PSF, showing more detail.

        Returns:
            list[be.ndarray]: A list of padded pupil functions (shape:
            `grid_size` x `grid_size`).

        Raises:
            ValueError: If any pupil's dimension (`num_rays`) is larger
                        than `grid_size`.
        """
        pupils_padded = []
        for pupil in self.pupils:
            pad_before = (self.grid_size - pupil.shape[0]) // 2
            pad_after = pad_before + (self.grid_size - pupil.shape[0]) % 2

            pupil = be.pad(
                pupil,
                ((pad_before, pad_after), (pad_before, pad_after)),
                mode="constant",
                constant_values=0,
            )

            pupils_padded.append(pupil)

        return pupils_padded

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
        return be.sum(be.abs(self.pupils[0]) > 0) ** 2

    def _get_psf_units(self, image):
        """Calculates the physical extent (units) of the PSF image for plotting.

        This method is called by `BasePSF.view()` to determine axis labels.
        It computes the total spatial width and height (in micrometers) of the
        provided PSF image data.

        The calculation uses:
        - Optic's effective F-number (FNO_eff).
        - Wavelength of light (using `self.wavelengths[0]`, as scale is typically
          set by one reference wavelength, units are in µm).
        - Q-factor: Ratio of `self.grid_size` (FFT grid size) to
          `self.num_rays` (pupil sampling density).

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
        FNO = self._get_working_FNO()

        Q = self.grid_size / (self.num_rays - 1)
        dx = self.wavelengths[0].value * FNO / Q

        x = be.to_numpy(image.shape[1] * dx)
        y = be.to_numpy(image.shape[0] * dx)

        return x, y


class FFTPSF:
    """Factory class for generating either a Vectorial or Scalar FFT PSF.

    This class inspects the optical system's polarization state to determine
    which FFT PSF implementation to instantiate. If polarization is enabled,
    it returns a `VectorialFFTPSF`. Otherwise, it returns a `ScalarFFTPSF`.

    Args:
        optic (Optic): The optical system object.
        field (tuple): The field point.
        wavelength (str | float): The wavelength of light.
        num_rays (int, optional): The number of rays to trace. Defaults to 128.
        grid_size (int, optional): The FFT grid size. Defaults to None.
        strategy (str): The wavefront calculation strategy. Defaults to "chief_ray".
        remove_tilt (bool): If True, removes tilt from OPD. Defaults to False.
        **kwargs: Additional keyword arguments.
    """

    def __new__(
        cls,
        optic,
        field,
        wavelength: str | float,
        num_rays=128,
        grid_size=None,
        strategy="chief_ray",
        remove_tilt=False,
        **kwargs,
    ):
        if optic.polarization_state is not None:
            from optiland.psf.vectorial_fft import VectorialFFTPSF

            return VectorialFFTPSF(
                optic=optic,
                field=field,
                wavelength=wavelength,
                num_rays=num_rays,
                grid_size=grid_size,
                strategy=strategy,
                remove_tilt=remove_tilt,
                **kwargs,
            )
        else:
            return ScalarFFTPSF(
                optic=optic,
                field=field,
                wavelength=wavelength,
                num_rays=num_rays,
                grid_size=grid_size,
                strategy=strategy,
                remove_tilt=remove_tilt,
                **kwargs,
            )
