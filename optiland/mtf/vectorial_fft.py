"""Vectorial FFT Modulation Transfer Function (MTF) Module.

This module provides the VectorialFFTMTF class for computing the MTF of an
optical system using the vectorial FFT method, taking into account the full
3D electric field at the exit pupil.

Kramer Harrison, 2026
"""

from __future__ import annotations

from optiland.psf.vectorial_fft import VectorialFFTPSF

from .fft import ScalarFFTMTF


class VectorialFFTMTF(ScalarFFTMTF):
    """Vectorial Fast Fourier Transform Modulation Transfer Function class.

    This class calculates the MTF of an optical system using the vectorial
    FFT method. It accounts for the full 3D electric field at the exit pupil
    and is intended for use with polarized optical systems. Use the `FFTMTF`
    factory to automatically select between scalar and vectorial
    implementations based on the optic's polarization state.

    Inherits all constructor arguments and attributes from `ScalarFFTMTF`.
    """

    def _calculate_psf(self):
        """Calculates and stores the PSF using the vectorial FFT method.

        This method uses the resolved field points and wavelength from BaseMTF,
        and explicitly uses the vectorial FFT PSF implementation to account
        for polarization effects.
        """
        self.psf = [
            VectorialFFTPSF(
                self.optic,
                field,
                self.resolved_wavelength,
                self.num_rays,
                self.grid_size,
                self.strategy,
                self.remove_tilt,
                **self.strategy_kwargs,
            ).psf
            for field in self.resolved_fields
        ]
