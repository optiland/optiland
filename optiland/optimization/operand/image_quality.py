"""Image Quality Operands Module

This module provides operands that summarize the image quality of an optical
system at a single field point and wavelength: RMS wavefront error, MTF at one
spatial frequency, and boresight error. They are intended for optimization and,
in particular, for tolerancing, where they are evaluated once per Monte Carlo
trial or sensitivity step.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import optiland.backend as be
from optiland.distribution import GaussianQuadrature
from optiland.paraxial_path import ParaxialDomainWarning
from optiland.utils import resolve_wavelength
from optiland.wavefront import Wavefront

if TYPE_CHECKING:
    from optiland._types import DistributionType
    from optiland.optic import Optic


def _boresight(
    optic: Optic, axis: str, wavelength: str | float, Hx: float, Hy: float
) -> be.ndarray:
    """Return the boresight error along ``axis`` ("x" or "y") in milliradians."""
    wavelength = resolve_wavelength(optic, wavelength)
    optic.trace_generic(Hx, Hy, 0.0, 0.0, wavelength)
    coordinate = getattr(optic.surfaces, axis)[-1, 0]
    # The EFL only scales the image offset to an angle. Its scalar paraxial
    # value ignores the tilts/decenters being toleranced, which is intended
    # here, so the domain warning is silenced instead of emitted per trial.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ParaxialDomainWarning)
        efl = optic.paraxial.f2()
    return 1e3 * coordinate / efl


class ImageQualityOperand:
    """Static methods computing image quality metrics of an optic.

    Methods:
        rms_wavefront: RMS wavefront error (piston removed) in waves.
        mtf: Modulation transfer function at one spatial frequency.
        boresight_x: Boresight error along the image x axis in milliradians.
        boresight_y: Boresight error along the image y axis in milliradians.

    """

    @staticmethod
    def rms_wavefront(
        optic: Optic,
        Hx: float,
        Hy: float,
        num_rays: int,
        wavelength: str | float,
        distribution: DistributionType = "gaussian_quad",
        remove_tilt: bool = False,
    ) -> be.ndarray:
        """Calculates the RMS wavefront error at one field and wavelength.

        The OPD is computed with :class:`optiland.wavefront.Wavefront` using
        its default chief-ray reference sphere, in waves at ``wavelength``.
        The RMS is taken about the weighted mean OPD, i.e. piston is always
        removed (the "RMS referenced to the mean" convention). Note that
        :class:`optiland.analysis.RmsWavefrontErrorVsField` instead reports
        ``sqrt(mean(opd**2))`` about the chief-ray reference, which includes
        piston, so the two agree only when the mean OPD is zero.

        Each pupil sample is weighted by its ray intensity, so vignetted or
        invalid rays (zero intensity) do not contribute, and additionally by
        the quadrature weight when the distribution provides one (as for the
        default ``'gaussian_quad'``). Otherwise samples are weighted equally.

        Args:
            optic: The optic object.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            num_rays: The pupil sampling parameter. For 'gaussian_quad' it is
                the number of rings; for 'hexapolar' the number of rings; for
                'uniform' the number of rays across the pupil diameter.
            wavelength: The wavelength in µm, or 'primary'.
            distribution: The pupil distribution. Default is 'gaussian_quad'.
            remove_tilt: If True, also remove x/y tilt with a weighted
                least-squares plane fit over the exit pupil coordinates (the
                "RMS referenced to the centroid" convention). Default is False.

        Returns:
            The RMS wavefront error in waves.

        """
        wavelength = resolve_wavelength(optic, wavelength)
        if distribution == "gaussian_quad":
            distribution = GaussianQuadrature()
            distribution.generate_points(num_rays)

        wf = Wavefront(optic, [(Hx, Hy)], [wavelength], num_rays, distribution)
        data = wf.get_data((Hx, Hy), wavelength)

        weights = data.intensity
        quad_weights = getattr(wf.distribution, "weights", None)
        if quad_weights is not None and len(quad_weights) == len(weights):
            weights = weights * quad_weights
        weights = weights / be.sum(weights)

        if remove_tilt:
            weighted = replace(data, intensity=weights)
            residual = Wavefront.fit_and_remove_tilt(weighted, remove_piston=True)
        else:
            residual = data.opd - be.sum(weights * data.opd)
        return be.sqrt(be.sum(weights * residual**2))

    @staticmethod
    def mtf(
        optic: Optic,
        Hx: float,
        Hy: float,
        wavelength: str | float,
        frequency: float,
        azimuth: Literal["tangential", "sagittal"] = "tangential",
        method: Literal["sampled", "geometric"] = "sampled",
        num_rays: int = 64,
    ) -> be.ndarray:
        """Calculates the MTF at one spatial frequency, field and wavelength.

        Following Optiland's MTF analyses, the tangential MTF is evaluated for
        a frequency along the image y axis and the sagittal MTF for a
        frequency along the image x axis, independent of the field azimuth.

        Two methods are available:

        - ``'sampled'`` (default): the diffraction MTF from
          :class:`optiland.mtf.SampledMTF`, which evaluates the pupil
          autocorrelation directly at the requested frequency. Unlike the FFT
          MTF it needs no PSF grid, so it is the cheaper diffraction method
          for a single frequency, as required per Monte Carlo trial.
        - ``'geometric'``: the geometric MTF from
          :class:`optiland.mtf.GeometricMTF` (scaled by the diffraction-limited
          MTF). It is faster for strongly aberrated systems but not
          differentiable, because it histograms the spot.

        Args:
            optic: The optic object.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            wavelength: The wavelength in µm, or 'primary'.
            frequency: The spatial frequency in cycles/mm in image space.
            azimuth: 'tangential' (frequency along image y) or 'sagittal'
                (frequency along image x). Default is 'tangential'.
            method: 'sampled' or 'geometric'. Default is 'sampled'.
            num_rays: The number of rays across the pupil diameter of the
                uniform pupil grid. Default is 64.

        Returns:
            The MTF value, dimensionless in [0, 1].

        Raises:
            ValueError: If ``azimuth`` or ``method`` is not recognized.

        """
        if azimuth not in ("tangential", "sagittal"):
            raise ValueError(
                f"Invalid azimuth '{azimuth}'. Must be 'tangential' or 'sagittal'."
            )
        wavelength = resolve_wavelength(optic, wavelength)

        if method == "sampled":
            from optiland.mtf import SampledMTF

            mtf = SampledMTF(optic, (Hx, Hy), wavelength, num_rays, "uniform")
            freq = (0.0, frequency) if azimuth == "tangential" else (frequency, 0.0)
            value = mtf.calculate_mtf([freq])[0]
            if not be.is_array_like(value):  # degenerate zero-XPD case
                value = be.array(value)
            return be.reshape(value, ())
        if method == "geometric":
            from optiland.mtf import GeometricMTF

            mtf = GeometricMTF(
                optic, [(Hx, Hy)], wavelength, num_rays, max_freq=frequency
            )
            index = 0 if azimuth == "tangential" else 1
            return mtf.mtf[0][index][-1]
        raise ValueError(
            f"Invalid method '{method}'. Must be 'sampled' or 'geometric'."
        )

    @staticmethod
    def boresight_x(
        optic: Optic, wavelength: str | float, Hx: float = 0.0, Hy: float = 0.0
    ) -> be.ndarray:
        """Calculates the boresight error along the image x axis.

        The boresight error is the x coordinate of the chief ray (pupil
        coordinates Px = Py = 0) on the image surface, in global coordinates,
        divided by the paraxial effective focal length ``f2`` and expressed in
        milliradians. The scalar paraxial ``f2`` treats tilted or decentered
        surfaces as centered, so the scale is effectively the nominal EFL.

        This follows OpticStudio's BSER operand, defined as the radial
        chief-ray image coordinate of the on-axis field divided by the
        effective focal length; ``boresight_x`` and ``boresight_y`` are its
        signed components, so ``sqrt(boresight_x**2 + boresight_y**2)`` is the
        radial boresight error. A centered system has zero boresight error;
        tilts and decenters of elements deviate the line of sight and make it
        nonzero. The sign follows that of ``f2`` (negative for systems whose
        paraxial EFL is negative, e.g. a single concave mirror).

        Args:
            optic: The optic object.
            wavelength: The wavelength in µm, or 'primary'.
            Hx: The normalized x field coordinate. Default is 0 (on axis).
            Hy: The normalized y field coordinate. Default is 0 (on axis).

        Returns:
            The boresight error along x in milliradians.

        """
        return _boresight(optic, "x", wavelength, Hx, Hy)

    @staticmethod
    def boresight_y(
        optic: Optic, wavelength: str | float, Hx: float = 0.0, Hy: float = 0.0
    ) -> be.ndarray:
        """Calculates the boresight error along the image y axis.

        Same as :meth:`boresight_x`, using the y coordinate of the chief ray
        on the image surface.

        Args:
            optic: The optic object.
            wavelength: The wavelength in µm, or 'primary'.
            Hx: The normalized x field coordinate. Default is 0 (on axis).
            Hy: The normalized y field coordinate. Default is 0 (on axis).

        Returns:
            The boresight error along y in milliradians.

        """
        return _boresight(optic, "y", wavelength, Hx, Hy)
