"""Vectorial Huygens-Fresnel Point Spread Function (PSF) Module

This module provides the ``VectorialHuygensPSF`` class, which extends
``ScalarHuygensPSF`` to perform a full vectorial diffraction calculation using
the Huygens-Fresnel principle.  The PSF is computed by summing the intensity
contributions from all three Cartesian components (Ex, Ey, Ez) of the electric
field at the exit pupil, for each incoherent polarization state.  The result
is normalized so that an equivalent diffraction-limited system has a peak
intensity of 100.

Kramer Harrison, 2026
"""

from __future__ import annotations

import optiland.backend as be
from optiland.psf.huygens_fresnel import ScalarHuygensPSF
from optiland.wavefront import Wavefront


class VectorialHuygensPSF(ScalarHuygensPSF):
    """Vectorial Huygens PSF using the full electric-field formulation.

    This class computes the PSF by integrating the three Cartesian components
    of the exit-pupil electric field (Ex, Ey, Ez) over all incoherent
    polarization states.  The total intensity at each image point is the
    incoherent sum of the squared-magnitude contributions from each component.

    The class inherits all geometry, coordinate, and visualisation logic from
    ``ScalarHuygensPSF`` and only overrides ``_compute_psf`` and
    ``_get_normalization``.

    Args:
        optic (Optic): The optical system object.
        field (tuple): The field point (Hx, Hy) in normalised coordinates.
        wavelength (str | float): Wavelength in micrometers or ``'primary'``.
        num_rays (int, optional): Pupil sampling grid size. Defaults to 128.
        image_size (int, optional): Image grid size. Defaults to 128.
        strategy (str): Wavefront reference strategy. Defaults to
            ``"chief_ray"``.
        remove_tilt (bool): Remove tilt/piston from OPD. Defaults to
            ``False``.
        oversample (float, optional): Oversampling ratio for MTF use.
            Defaults to ``None``.
        pixel_pitch (float, optional): Pixel pitch in mm. Defaults to
            ``None``.
        **kwargs: Additional keyword arguments forwarded to ``BasePSF``.
    """

    def _compute_psf(self):
        """Compute the vectorial PSF using the Huygens-Fresnel principle.

        For each incoherent polarization state and each Cartesian component of
        the exit-pupil electric field, the Huygens-Fresnel diffraction integral
        is evaluated on the image grid.  The resulting per-component intensities
        are summed incoherently to produce the final PSF.

        Returns:
            be.ndarray: 2D PSF array (shape: ``image_size`` × ``image_size``),
            normalised so that a diffraction-limited system has a peak of 100.

        Raises:
            ValueError: If ``E_exits`` is not populated in the wavefront data.
        """
        Hx, Hy = self.fields[0].coord
        wavelength_um = self.wavelengths[0].value
        wavelength_mm = wavelength_um * 1e-3
        data = self.get_data((Hx, Hy), wavelength_um)

        if data.E_exits is None:
            raise ValueError(
                "E_exits must be populated in WavefrontData for "
                "VectorialHuygensPSF. Ensure you are using PolarizedRays."
            )

        pupil_x, pupil_y, pupil_z = data.pupil_x, data.pupil_y, data.pupil_z
        pupil_opd = data.opd * wavelength_mm  # waves to mm
        Rp = data.radius
        is_valid = data.intensity > 0

        image_x, image_y, image_z = self._get_image_coordinates()

        psf = None
        for E_exit in data.E_exits:
            for i in range(3):
                amplitude = be.where(
                    is_valid, E_exit[..., i], be.zeros_like(E_exit[..., i])
                )
                component_psf = self._summation_strategy.compute(
                    image_x,
                    image_y,
                    image_z,
                    pupil_x,
                    pupil_y,
                    pupil_z,
                    amplitude,
                    pupil_opd,
                    wavelength_mm,
                    Rp,
                )
                psf = component_psf if psf is None else psf + component_psf

        # Normalize the PSF
        if self.normalization is None:
            self.normalization = self._get_normalization()
        return psf / self.normalization * 100.0

    def _get_normalization(self):
        """Compute the normalization factor for the vectorial Huygens PSF.

        The normalization equals the ideal (zero-OPD) vectorial PSF value at
        the on-axis image centre, computed by running the Huygens-Fresnel
        integral with the actual exit-pupil electric-field amplitudes but with
        all optical path differences set to zero.  Summing across all three
        Cartesian components and all incoherent polarization states gives the
        diffraction-limited peak, which maps to a Strehl ratio of 1.0 (100%).

        Returns:
            float | be.ndarray: The scalar normalization value.

        Raises:
            ValueError: If ``E_exits`` is not populated for the reference field.
        """
        if self.fields[0].coord == (0, 0):
            data = self.get_data((0, 0), self.wavelengths[0].value)
        else:
            wf = Wavefront(
                self.optic,
                distribution="uniform",
                num_rays=self.num_rays,
                fields=[(0, 0)],
                wavelengths=[self.wavelengths[0].value],
            )
            data = wf.get_data((0, 0), self.wavelengths[0].value)

        if data.E_exits is None:
            raise ValueError(
                "E_exits must be populated for VectorialHuygensPSF "
                "normalization. Ensure you are using PolarizedRays."
            )

        pupil_opd_ideal = be.zeros_like(data.opd)
        image_x = be.zeros((1, 1))
        image_y = be.zeros((1, 1))
        ideal_z = self.optic.surfaces.positions[-1]
        image_z = be.full((1, 1), ideal_z)

        is_valid = data.intensity > 0
        norm = 0.0
        for E_exit in data.E_exits:
            for i in range(3):
                amplitude = be.where(
                    is_valid, E_exit[..., i], be.zeros_like(E_exit[..., i])
                )
                component_norm = self._summation_strategy.compute(
                    image_x,
                    image_y,
                    image_z,
                    data.pupil_x,
                    data.pupil_y,
                    data.pupil_z,
                    amplitude,
                    pupil_opd_ideal,
                    self.wavelengths[0].value * 1e-3,
                    data.radius,
                )
                norm = norm + component_norm[0, 0]

        return norm
