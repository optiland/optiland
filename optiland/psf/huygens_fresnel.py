"""Huygens-Fresnel Point Spread Function (PSF) Module

This module provides functionality for simulating and analyzing the Point
Spread Function (PSF) of optical systems using the Huygens-Fresnel principle.
It includes capabilities for generating PSF for a given optical system.
Visualization and Strehl ratio calculation are primarily handled by the base
class. The PSF is normalized against the peak of an ideal diffraction-limited
system calculated using the same Huygens-Fresnel principle.

The module exposes:
- ``ScalarHuygensPSF``: scalar (intensity-only) Huygens PSF.
- ``HuygensPSF``: factory that automatically dispatches to
  ``VectorialHuygensPSF`` when a polarization state is set on the optic,
  and to ``ScalarHuygensPSF`` otherwise.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.psf.base import BasePSF
from optiland.psf.huygens_fresnel_strategies import (
    NumbaSummation,
    TorchSummation,
)
from optiland.visualization.system.utils import transform
from optiland.wavefront import Wavefront


class ScalarHuygensPSF(BasePSF):
    """Huygens PSF class using Huygens-Fresnel principle (scalar formulation).

    Computes PSF by integrating contributions from point sources across the
    exit pupil. The PSF is normalized such that the peak intensity of an
    equivalent diffraction-limited system (same pupil amplitude, zero phase error)
    would be 100.0. This makes the Strehl ratio (peak of actual PSF / 100.0,
    or value at PSF center / 100.0) meaningful.

    Inherits from `BasePSF` for common initialization (Wavefront setup) and
    visualization methods.

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
        image_size (int, optional): The size of the image grid for PSF
            calculation. Defaults to 128.
        strategy (str): The calculation strategy to use. Supported options are
            "chief_ray", "centroid_sphere", and "best_fit_sphere".
            Defaults to "chief_ray".
        remove_tilt (bool): If True, removes tilt and piston from the OPD data.
            Defaults to False.
        oversample (float, optional): The oversampling ratio with respect to the
            optical cutoff. Impacts the extent of the image and is generally
            only used for MTF calculation. Defaults to None.
        pixel_pitch (float, optional): The pixel pitch of the image plane in mm.
            If provided, this will override the automatic extent calculation.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to the strategy.
    """

    def __init__(
        self,
        optic,
        field,
        wavelength: str | float,
        num_rays=128,
        image_size=128,
        strategy="chief_ray",
        remove_tilt=False,
        oversample: float = None,
        pixel_pitch: float = None,
        normalization: float = None,
        **kwargs,
    ):
        super().__init__(
            optic=optic,
            field=field,
            wavelength=wavelength,
            num_rays=num_rays,
            strategy=strategy,
            remove_tilt=remove_tilt,
            **kwargs,
        )

        self.cx = None  # center of the image plane
        self.cy = None
        self.pixel_pitch = pixel_pitch  # pixel pitch of image plane in mm

        self.image_size = image_size
        self.oversample = oversample
        self.normalization = normalization

        self._summation_strategy = self._create_summation_strategy()
        self.psf = self._compute_psf()

    def _create_summation_strategy(self):
        """Factory method to create the appropriate summation strategy."""
        backend = be.get_backend()
        if backend == "numpy":
            return NumbaSummation()
        elif backend == "torch":
            try:
                return TorchSummation()
            except ImportError as e:
                raise ImportError(
                    "Torch backend selected, but PyTorch is not installed."
                ) from e
        else:
            raise ValueError(f"Unsupported backend for HuygensPSF: {backend}")

    def _determine_image_center(self):
        """Determine center of image via raytrace across field"""
        Hx, Hy = self.fields[0].coord
        rays = self.optic.trace(
            Hx=Hx,
            Hy=Hy,
            wavelength=self.wavelengths[0].value,
            distribution="hexapolar",
            num_rays=6,
        )

        # Use only rays with positive intensity (i > 0) to calculate the center
        valid_rays = rays.i > 0
        if not be.any(valid_rays):
            # Handle case where no rays make it to the image plane
            return be.array([0.0]), be.array([0.0])

        rx, ry, _ = transform(
            rays.x[valid_rays],
            rays.y[valid_rays],
            rays.z[valid_rays],
            self.optic.image_surface,
            is_global=True,
        )

        return rx, ry

    def _get_image_extent(self) -> tuple[float, float, float, float]:
        """Calculate the extent of the image plane in mm.

        The extent can be determined either by optical cutoff (oversample mode),
        by geometric/Airy coverage (default mode), or by a user-provided
        pixel_pitch.
        """
        # Determine image center and retrieve x, y intersections
        rx, ry = self._determine_image_center()
        self.cx = be.mean(rx)
        self.cy = be.mean(ry)

        # If pixel_pitch is provided, use it to determine the extent.
        # Otherwise, calculate it automatically.
        if self.pixel_pitch is not None:
            extent = 0.5 * self.image_size * self.pixel_pitch
        else:
            if self.oversample is not None:
                extent = self._extent_from_cutoff()
            else:
                extent = self._extent_from_geometry(rx, ry)

            # Pixel pitch is derived from the calculated extent
            self.pixel_pitch = 2 * extent / self.image_size

        # Final extents centered on chief ray intercept
        xmin = -extent + self.cx
        xmax = extent + self.cx
        ymin = -extent + self.cy
        ymax = extent + self.cy

        return xmin, xmax, ymin, ymax

    def _extent_from_cutoff(self) -> float:
        """Compute half-extent based on cutoff frequency and oversampling ratio.

        This method determines the image plane extent by enforcing sampling
        criteria relative to the optical cutoff frequency. The cutoff frequency
        is defined by the system's effective F-number and the primary wavelength.
        The oversampling factor scales the cutoff to achieve finer-than-Nyquist
        sampling, ensuring that the PSF is adequately resolved on the image grid.
        """
        f_cutoff = 1.0 / (self._get_working_FNO() * self.wavelengths[0].value * 1e-3)
        f_nyquist = self.oversample * f_cutoff
        self.pixel_pitch = 1.0 / (2 * f_nyquist)
        return 0.5 * self.image_size * self.pixel_pitch

    def _extent_from_geometry(self, rx, ry) -> float:
        """Compute half-extent based on geometric footprint and Airy disk.

        This method computes the extent of the image plane based on the geometric
        spot size, as well as a scaled ideal Airy disk at a given wavelength. The
        extent is defined as the maximum of the geometric extent and the ideal
        extent, ensuring that the PSF covers the area where the light is expected
        to be distributed.
        """
        num_Airy_disks = 5.0  # how many Airy disk radii to include in half-extent
        extent_geometric = be.max(be.hypot(rx - self.cx, ry - self.cy))
        extent_ideal = (
            num_Airy_disks
            * self._get_working_FNO()  # effective F-number
            * 1.22
            * (self.wavelengths[0].value * 1e-3)  # um --> mm
        )
        return max(extent_geometric, extent_ideal)

    def _get_image_coordinates(self):
        """Generate image coordinates for the PSF calculation.

        This method computes the coordinates of the image plane based on
        the extent of the image surface. It creates a grid of points in the
        image plane, which will be used to evaluate the PSF using the
        Huygens-Fresnel principle.
        """
        xmin, xmax, ymin, ymax = self._get_image_extent()
        image_x = be.linspace(xmin, xmax, self.image_size)
        image_y = be.linspace(ymin, ymax, self.image_size)
        image_x, image_y = be.meshgrid(image_x, image_y)

        img_geometry = self.optic.image_surface.geometry
        image_z = img_geometry.sag(image_x, image_y)

        # transform image coordinates from local to global frame
        image_x, image_y, image_z = transform(
            image_x, image_y, image_z, self.optic.image_surface, is_global=False
        )

        image_x = be.reshape(image_x, (self.image_size, self.image_size))
        image_y = be.reshape(image_y, (self.image_size, self.image_size))
        image_z = be.reshape(image_z, (self.image_size, self.image_size))

        return image_x, image_y, image_z

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

        Returns:
            float: The normalization factor.
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

        pupil_opd_ideal = be.zeros_like(data.opd)  # ideal case has no OPD
        image_x = be.zeros((1, 1))  # single point for normalization
        image_y = be.zeros((1, 1))
        ideal_z = self.optic.surfaces.positions[-1]  # image plane position
        image_z = be.full((1, 1), ideal_z)

        psf_max = self._summation_strategy.compute(
            image_x,
            image_y,
            image_z,
            data.pupil_x,
            data.pupil_y,
            data.pupil_z,
            be.ones_like(data.intensity),
            pupil_opd_ideal,
            self.wavelengths[0].value * 1e-3,
            data.radius,
        )

        return psf_max[0, 0]  # Normalize by the peak of the ideal PSF

    def _compute_psf(self):
        """Compute the PSF using the Huygens-Fresnel principle."""
        # Retrieve pupil data from Wavefront instance
        Hx, Hy = self.fields[0].coord
        wavelength_um = self.wavelengths[0].value
        wavelength_mm = wavelength_um * 1e-3
        data = self.get_data((Hx, Hy), wavelength_um)

        # Extract pupil data
        pupil_x, pupil_y, pupil_z = data.pupil_x, data.pupil_y, data.pupil_z
        pupil_amp = be.sqrt(data.intensity)
        pupil_opd = data.opd * wavelength_mm  # waves to mm
        Rp = data.radius  # Radius of curvature of exit pupil

        # Get image coordinates
        image_x, image_y, image_z = self._get_image_coordinates()

        # Compute the PSF using Huygens-Fresnel summation
        psf = self._summation_strategy.compute(
            image_x,
            image_y,
            image_z,
            pupil_x,
            pupil_y,
            pupil_z,
            pupil_amp,
            pupil_opd,
            wavelength_mm,
            Rp,
        )

        # Normalize the PSF
        if self.normalization is None:
            self.normalization = self._get_normalization()
        psf = psf / self.normalization * 100.0

        return psf

    def _get_psf_units(self, image):
        """Calculates the physical extent (units) of the PSF image for plotting.

        This method is called by `BasePSF.view()` to determine axis labels.
        It computes the total spatial width and height (in micrometers) of the
        provided PSF image data.

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
        num_x, num_y = image.shape
        dx = self.pixel_pitch
        x = be.to_numpy(num_x * dx) * 1e3  # mm --> um
        y = be.to_numpy(num_y * dx) * 1e3  # mm --> um
        return x, y


class HuygensPSF:
    """Factory class for generating either a Vectorial or Scalar Huygens PSF.

    This class inspects the optical system's polarization state to determine
    which Huygens PSF implementation to instantiate. If polarization is enabled,
    it returns a ``VectorialHuygensPSF``. Otherwise, it returns a
    ``ScalarHuygensPSF``.

    Args:
        optic (Optic): The optical system object.
        field (tuple): The field point.
        wavelength (str | float): The wavelength of light in micrometers.
        num_rays (int, optional): The number of rays to trace. Defaults to 128.
        image_size (int, optional): The size of the image grid. Defaults to 128.
        strategy (str): The wavefront calculation strategy. Defaults to
            "chief_ray".
        remove_tilt (bool): If True, removes tilt from OPD. Defaults to False.
        oversample (float, optional): Oversampling ratio for MTF use.
            Defaults to None.
        pixel_pitch (float, optional): Pixel pitch in mm. Defaults to None.
        **kwargs: Additional keyword arguments.
    """

    def __new__(
        cls,
        optic,
        field,
        wavelength: str | float,
        num_rays=128,
        image_size=128,
        strategy="chief_ray",
        remove_tilt=False,
        oversample: float = None,
        pixel_pitch: float = None,
        normalization: float = None,
        **kwargs,
    ):
        if optic.polarization_state is not None:
            from optiland.psf.vectorial_huygens import VectorialHuygensPSF

            return VectorialHuygensPSF(
                optic=optic,
                field=field,
                wavelength=wavelength,
                num_rays=num_rays,
                image_size=image_size,
                strategy=strategy,
                remove_tilt=remove_tilt,
                oversample=oversample,
                pixel_pitch=pixel_pitch,
                normalization=normalization,
                **kwargs,
            )
        else:
            return ScalarHuygensPSF(
                optic=optic,
                field=field,
                wavelength=wavelength,
                num_rays=num_rays,
                image_size=image_size,
                strategy=strategy,
                remove_tilt=remove_tilt,
                oversample=oversample,
                pixel_pitch=pixel_pitch,
                normalization=normalization,
                **kwargs,
            )
