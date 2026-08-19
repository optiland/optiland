Colorimetry
===========

This section covers Optiland's colorimetry utilities, which convert spectral data
into perceptual color quantities. These tools are used, for example, to render
extended-source or image-simulation results as realistic sRGB colors rather than
raw spectral irradiance.

The core engine converts a spectrum sampled over wavelength into CIE 1931 XYZ
tristimulus values (:func:`~optiland.colorimetry.core.spectrum_to_xyz`), which can
then be converted to chromaticity coordinates
(:func:`~optiland.colorimetry.core.xyz_to_xyY`) or to display-ready sRGB
(:func:`~optiland.colorimetry.core.xyz_to_srgb`). Standard CIE color-matching
functions and illuminants are provided in ``colorimetry.constants``, and
``colorimetry.plotting`` provides helpers for plotting results on the CIE 1931
chromaticity diagram.

.. autosummary::
   :toctree: colorimetry/
   :caption: Colorimetry Modules

   colorimetry.core
   colorimetry.constants
   colorimetry.plotting
