Wavefront Analysis
==================

This section provides an overview of the wavefront-related analyses available in Optiland,
including wavefront error, Zernike decomposition, point spread function (PSF) and modulation transfer function (MTF) calculations.

Wavefront & OPD
---------------

These modules handle optical path difference (OPD) calculations and wavefront analysis.

.. autosummary::
   :toctree: wavefront/

   optiland.wavefront.opd_fan
   optiland.wavefront.opd
   optiland.wavefront.reference_geometry
   optiland.wavefront.strategy
   optiland.wavefront.wavefront_data
   optiland.wavefront.wavefront
   optiland.wavefront.zernike_opd

Point Spread Function (PSF)
---------------------------

PSF modules calculate the point spread function using various computational methods,
from fast Fourier transforms to more rigorous diffraction calculations.

.. autosummary::
   :toctree: psf/

   psf.base
   psf.fft
   psf.huygens_fresnel
   psf.mmdft
   psf.vectorial_fft
   psf.vectorial_huygens

Modulation Transfer Function (MTF)
----------------------------------

The MTF modules provide different methods for calculating the modulation transfer function,
which characterizes the spatial frequency response of optical systems.

.. autosummary::
   :toctree: mtf/

   mtf.base
   mtf.fft
   mtf.vectorial_fft
   mtf.geometric
   mtf.huygens_fresnel
   mtf.sampled