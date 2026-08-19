"""Vectorial FFT Point Spread Function (PSF) Module

This module provides functionality for simulating and analyzing the Point
Spread Function (PSF) of optical systems using the Vectorial Fast Fourier Transform
(FFT). It includes capabilities for generating PSF from given wavefront
aberrations and exit pupil electric fields.

Kramer Harrison, 2026
"""

from __future__ import annotations

import optiland.backend as be
from optiland.psf.fft import ScalarFFTPSF


class VectorialFFTPSF(ScalarFFTPSF):
    """Class representing the Vectorial Fast Fourier Transform (FFT) PSF.

    This class computes the PSF of an optical system by taking the Fourier
    Transform of the vectorial pupil function, taking into account the full
    3D electric field at the exit pupil.
    """

    def _generate_pupils(self):
        """Generates complex pupil functions for each wavelength and polarization state.

        Returns:
            list[be.ndarray]: A list of complex 2D arrays, each representing a
            Cartesian component of the pupil function for a wavelength and
            incoherent polarization state.
        """
        x = be.linspace(-1, 1, self.num_rays)
        x, y = be.meshgrid(x, x)
        x = x.ravel()
        y = y.ravel()
        R2 = x**2 + y**2

        field = self.fields[0]
        pupils = []

        for wl in self.wavelengths:
            wavefront_data = self.get_data(field, wl)

            if wavefront_data.E_exits is None:
                raise ValueError(
                    "E_exits must be populated in WavefrontData for VectorialFFTPSF. "
                    "Ensure you are using PolarizedRays."
                )

            for E_exit in wavefront_data.E_exits:
                # E_exit has shape (N, 3)
                # Create Cartesian component grids
                is_valid = wavefront_data.intensity > 0
                for i in range(3):
                    P = be.to_complex(be.zeros_like(x))
                    amplitude = be.where(
                        is_valid, E_exit[..., i], be.zeros_like(E_exit[..., i])
                    )
                    P[R2 <= 1] = be.to_complex(
                        amplitude * be.exp(-1j * 2 * be.pi * wavefront_data.opd)
                    )
                    P = be.reshape(P, (self.num_rays, self.num_rays))
                    pupils.append(P)

        return pupils

    def _get_normalization(self):
        """Calculates the normalization factor for the PSF.

        The normalization factor scales the PSF such that a diffraction-limited
        system with the same aperture and amplitude distribution has a peak of 100.

        For each pupil component P_i, the ideal (unaberrated) FFT peak is
        sum(|P_i|). Summing the squares of these ideal peaks across all pupils
        gives the diffraction-limited PSF peak, which is used as the
        normalization factor.
        """
        norm = 0.0
        for pupil in self.pupils:
            norm = norm + be.sum(be.abs(pupil)) ** 2
        return norm
