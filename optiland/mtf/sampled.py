"""Sampled Modulation Transfer Function (MTF) Module.

This module provides the SampledMTF class for computing the MTF
of an optical system based on sampled wavefront data.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.utils import resolve_wavelength
from optiland.wavefront import Wavefront
from optiland.zernike import ZernikeFit


class SampledMTF:
    """Sampled Modulation Transfer Function (MTF) class.

    This class calculates the MTF of an optical system from sampled wavefront
    data. It utilizes Zernike polynomial fitting to represent the wavefront
    aberrations.

    Note:
        This class assumes that amplitude variations between the pupil and a
        shifted version of the pupil can be ignored.

    Args:
        optic (Optic): The optical system.
        field (tuple): The field point (Hx, Hy) at which to calculate the MTF.
        wavelength (str or float): The wavelength (in mm) at which to
            calculate the MTF. Can be 'primary' to use the optic's primary
            wavelength.
        num_rays (int, optional): The number of rings to trace for the
            wavefront analysis. Defaults to 128 in each axis.
        distribution (str, optional): The distribution of rays in the pupil.
            Defaults to 'uniform'.
        zernike_terms (int, optional): The number of Zernike terms to use for
            the wavefront fit. Defaults to 37.
        zernike_type (str, optional): The type of Zernike polynomials to use
            ('fringe', 'standard', etc.). Defaults to 'fringe'.

    Attributes:
        optic (Optic): The optical system.
        field (tuple): The field point (Hx, Hy).
        wavelength (float): The wavelength (in mm) used for calculation.
        num_rays (int): The number of rays used for wavefront analysis.
        distribution (str): The ray distribution in the pupil.
        zernike_terms (int): The number of Zernike terms for the fit.
        zernike_type (str): The type of Zernike polynomials used.
        x_norm (be.ndarray): Normalized x-coordinates of pupil samples.
        y_norm (be.ndarray): Normalized y-coordinates of pupil samples.
        opd_waves (be.ndarray): Optical Path Difference (OPD) in waves.
        intensity (be.ndarray): Intensity at each pupil sample point.
        zernike_fit (ZernikeFit): The Zernike fit object.
        P1 (be.ndarray): The complex pupil function.
        otf_at_zero (float): The value of the Optical Transfer Function (OTF)
            at zero frequency, equivalent to the sum of intensities.
    """

    def __init__(
        self,
        optic,
        field,
        wavelength: str | float,
        num_rays=128,
        distribution="uniform",
        zernike_terms=37,
        zernike_type="fringe",
    ):
        """Initializes the SampledMTF instance."""
        self.optic = optic
        self.field = field
        self.wavelength = resolve_wavelength(optic, wavelength)
        self.num_rays = num_rays
        self.distribution = distribution
        self.zernike_terms = zernike_terms
        self.zernike_type = zernike_type

        wf = Wavefront(
            optic,
            fields=[self.field],
            wavelengths=[self.wavelength],
            num_rays=self.num_rays,
            distribution=self.distribution,
        )
        wf_data = wf.get_data(self.field, self.wavelength)

        self.x_norm = wf.distribution.x
        self.y_norm = wf.distribution.y
        self.opd_waves = wf_data.opd
        self.intensity = wf_data.intensity

        # XPD/XPL are used purely as axial distances (pupil-shift ratio
        # below), never as real-space coordinates, so the unfolded axial
        # scalar remains valid on folded paths.
        self.xpd = self.optic.paraxial.XPD()
        self.xpl = -self.optic.paraxial.XPL()

        self.zernike_fit = ZernikeFit(
            self.x_norm,
            self.y_norm,
            self.opd_waves,
            self.zernike_type,
            self.zernike_terms,
        )

        self.P1 = be.sqrt(self.intensity) * be.exp(1j * 2 * be.pi * self.opd_waves)
        self.otf_at_zero = be.sum(self.intensity)

    def calculate_mtf(self, frequencies):
        """Calculates the Modulation Transfer Function (MTF) for given spatial
        frequencies.

        The method computes the MTF by determining the Optical Transfer Function (OTF)
        from the overlap integral of the pupil function with a shifted version of
        itself. The shift corresponds to the spatial frequency being evaluated.
        The MTF is the absolute value of the normalized OTF.

        Args:
            frequencies (list[tuple[float, float]] or be.ndarray): A list or array of
                tuples, where each tuple `(fx, fy)` represents a spatial frequency
                pair in cycles per mm for which to calculate the MTF.

        Returns:
            list[float]: A list of MTF values corresponding to each input frequency
            pair. The MTF values are dimensionless and range from 0 to 1.

        Notes:
            The calculation involves:

            1. Retrieving the exit pupil diameter (XPD). If XPD is near zero,
               MTF is 0 for non-zero frequencies and 1 for zero frequency.
            2. Converting the wavelength to mm.
            3. For each frequency pair (fx, fy), the OTF is built up as follows:

               a. Calculating physical shifts in the pupil based on wavelength and
                  frequency.
               b. Normalizing these shifts using the XPD radius & exit pupil position.
               c. Determining the shifted normalized coordinates for pupil evaluation.
               d. Evaluating the Optical Path Difference (OPD) at these shifted
                  coordinates using the Zernike polynomial fit of the wavefront.
               e. Masking points where the shifted evaluation falls outside the
                  unit circle (i.e., outside the pupil).
               f. Computing the complex conjugate of the pupil function at the
                  shifted coordinates (P2_conj).
               g. Calculating the OTF element as the product of the original complex
                  pupil function (P1) and P2_conj.
               h. Summing the OTF elements over all pupil sample points to get the
                  total OTF value for the given frequency.
               i. Normalizing the OTF value by otf_at_zero (the OTF at zero frequency).
               j. The MTF is the absolute value of this normalized OTF.
        """
        xpd_is_zero = self.xpd == 0.0

        wl_um = self.wavelength
        x_norm = self.x_norm
        y_norm = self.y_norm
        intensity = self.intensity
        P1 = self.P1
        zernike_fit = self.zernike_fit

        wl_mm = wl_um * 1e-3  # Convert wavelength to mm for calculation

        mtf_results = []

        for fx, fy in frequencies:
            if xpd_is_zero:
                if fx == 0.0 and fy == 0.0:
                    mtf_results.append(1.0)
                else:
                    mtf_results.append(0.0)
                continue

            delta_x_phys = wl_mm * fx
            delta_y_phys = wl_mm * fy

            if self.xpd == 0.0:
                mtf_results.append(0.0)
                continue

            delta_x_norm = self.xpl * delta_x_phys / (self.xpd / 2)
            delta_y_norm = self.xpl * delta_y_phys / (self.xpd / 2)

            eval_x_shifted_norm = x_norm - delta_x_norm
            eval_y_shifted_norm = y_norm - delta_y_norm

            r_shifted_norm = be.sqrt(eval_x_shifted_norm**2 + eval_y_shifted_norm**2)
            phi_shifted_norm = be.arctan2(eval_y_shifted_norm, eval_x_shifted_norm)

            opd_shifted_waves = zernike_fit.zernike.poly(
                r_shifted_norm, phi_shifted_norm
            )

            mask_outside_pupil = r_shifted_norm > 1.0

            P2_conj = be.sqrt(intensity) * be.exp(-1j * 2 * be.pi * opd_shifted_waves)
            P2_conj = be.where(mask_outside_pupil, 0.0 + 0.0j, P2_conj)

            otf_element = P1 * P2_conj
            otf_value = be.sum(otf_element)

            # Avoid division by zero if otf_at_zero is zero
            if self.otf_at_zero == 0:
                mtf = 0.0
            else:
                normalized_otf = otf_value / self.otf_at_zero
                mtf = be.abs(normalized_otf)

            mtf_results.append(mtf)

        return mtf_results
