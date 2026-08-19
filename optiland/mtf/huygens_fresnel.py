"""Huygens Modulation Transfer Function (HuygensMTF) Module.

This module provides the ScalarHuygensMTF, VectorialHuygensMTF (via factory),
and HuygensMTF factory class for computing the MTF of an optical system using
a Point Spread Function (PSF) derived from the Huygens-Fresnel principle.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.psf.huygens_fresnel import ScalarHuygensPSF
from optiland.utils import get_working_FNO

from .base import BaseMTF


class ScalarHuygensMTF(BaseMTF):
    """Scalar Huygens Modulation Transfer Function class.

    This class calculates and visualizes the Modulation Transfer Function (MTF)
    of an optic using a Point Spread Function (PSF) derived from the
    Huygens-Fresnel principle. It is intended for use with unpolarized optical
    systems. Use the ``HuygensMTF`` factory to automatically select between
    scalar and vectorial implementations based on the optic's polarization
    state.

    Supports both the NumPy and PyTorch backends. The underlying Huygens-Fresnel
    summation uses Numba (NumPy) or batched tensor operations (PyTorch)
    transparently via the strategy pattern in ``ScalarHuygensPSF``.

    Args:
        optic (Optic): The optic for which to calculate the MTF.
        fields (str or list, optional): The field coordinates for which to
            calculate the MTF. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength of light to use
            for the MTF calculation. Defaults to 'primary'.
        num_rays (int, optional): The number of rays to use for the HuygensPSF
            calculation along one dimension of the pupil grid. Defaults to 128.
        image_size (int, optional): The size of the image grid for PSF
            calculation and subsequent MTF. Defaults to 128.
        max_freq (str or float, optional): The maximum frequency for the MTF
            calculation in cycles/mm. If 'cutoff', it's determined by the
            diffraction limit (1 / (lambda * FNO)). Defaults to 'cutoff'.

    Attributes:
        num_rays (int): The number of rays used for the PSF calculation.
        image_size (int): The size of the grid used for the PSF/MTF calculation.
        max_freq (float): The maximum frequency for the MTF calculation
            (cycles/mm).
        FNO (float): The F-number of the optic.
        psf_data (list): List of 2D PSF data arrays for each field.
        psf_instances (list): List of PSF instances for each field.
        mtf (list): List of MTF data ([tangential, sagittal]) for each field.
        freq (be.ndarray): Array of frequency points for the MTF curve
            (cycles/mm).
    """

    def __init__(
        self,
        optic,
        fields: str | list = "all",
        wavelength: str | float = "primary",
        num_rays=128,
        image_size=128,
        max_freq="cutoff",
    ):
        self.num_rays = num_rays
        self.image_size = image_size
        self.psf_instances = []

        super().__init__(optic, fields, wavelength)

        self.FNO = [
            get_working_FNO(self.optic, field, self.resolved_wavelength)
            for field in self.resolved_fields
        ]

        if max_freq == "cutoff":
            on_axis_fno = self._get_fno()
            self.max_freq = 1 / (self.resolved_wavelength * 1e-3 * on_axis_fno)
        else:
            self.max_freq = max_freq

        n_fields = len(self.resolved_fields)
        self.freq_tang = [
            be.arange(self.image_size // 2) * self._get_mtf_units_tang(k)
            for k in range(n_fields)
        ]
        self.freq_sag = [
            be.arange(self.image_size // 2) * self._get_mtf_units_sag(k)
            for k in range(n_fields)
        ]
        # Backward-compatible alias (tangential is the primary reference).
        self.freq = self.freq_tang

    def _calculate_psf(self):
        """Calculates and stores the Point Spread Functions (PSFs).

        This method uses the resolved field points and wavelength from BaseMTF,
        and explicitly uses the scalar Huygens PSF implementation. It populates
        ``self.psf_data`` with 2D PSF arrays and ``self.psf_instances`` with
        the ``ScalarHuygensPSF`` objects.
        """
        self.psf_data = []
        self.psf_instances = []

        # Pre-calculate normalization once (on-axis, primary wavelength)
        # This avoiding redundant PSF peak calculations for every field.
        norm_psf = ScalarHuygensPSF(
            optic=self.optic,
            field=(0, 0),
            wavelength=self.resolved_wavelength,
            num_rays=self.num_rays,
            image_size=self.image_size,
        )
        normalization = norm_psf.normalization

        for field_coord in self.resolved_fields:
            psf_calculator = ScalarHuygensPSF(
                optic=self.optic,
                field=field_coord,
                wavelength=self.resolved_wavelength,
                num_rays=self.num_rays,
                image_size=self.image_size,
                oversample=4.0,
                normalization=normalization,
            )
            self.psf_data.append(psf_calculator.psf)
            self.psf_instances.append(psf_calculator)

    def _generate_mtf_data(self):
        """Generates the MTF data for each field from the calculated PSFs.

        The calculation involves:
        1. Taking the 2D FFT of each PSF.
        2. Shifting the zero-frequency component to the center.
        3. Taking the absolute magnitude.
        4. Extracting tangential and sagittal slices.
        5. Normalizing these slices.

        Returns:
            list: A list of MTF data for each field. Each item is a list
                  containing the normalized tangential and sagittal MTF arrays.
        """
        mtf_results = []
        for psf_array in self.psf_data:
            otf = be.fft.fftshift(be.fft.fft2(psf_array))
            mtf_abs = be.abs(otf)

            # Determine center for slicing (PSF output is image_size x image_size).
            center_idx = self.image_size // 2

            # Extract tangential and sagittal MTF
            # Tangential: Column at center_idx, from center_idx downwards
            tangential_mtf = mtf_abs[center_idx:, center_idx]
            # Sagittal: Row at center_idx, from center_idx rightwards
            sagittal_mtf = mtf_abs[center_idx, center_idx:]

            # Ensure they are 1D arrays of expected length (image_size // 2)
            tangential_mtf = tangential_mtf[: self.image_size // 2]
            sagittal_mtf = sagittal_mtf[: self.image_size // 2]

            # Normalize MTF by its DC component (value at zero frequency)
            dc_value = mtf_abs[center_idx, center_idx]
            if dc_value == 0:
                norm_tangential = be.zeros_like(tangential_mtf)
                norm_sagittal = be.zeros_like(sagittal_mtf)
            else:
                norm_tangential = tangential_mtf / dc_value
                norm_sagittal = sagittal_mtf / dc_value

            # Ensure MTF <= 1 after normalization (can exceed due to numerical
            # precision).
            norm_tangential = be.clip(norm_tangential, 0.0, 1.0)
            norm_sagittal = be.clip(norm_sagittal, 0.0, 1.0)

            mtf_results.append([norm_tangential, norm_sagittal])

        return mtf_results

    def _plot_field_mtf(self, ax, field_index, mtf_field_data, color):
        """Plots the MTF data (tangential and sagittal) for a single field.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axes object.
            field_index (int): The index of the current field in
                self.resolved_fields.
            mtf_field_data (list): A list containing normalized tangential and
                sagittal MTF data (be.ndarray) for the field.
            color (str): The color to use for plotting this field.
        """
        current_field_label_info = self.resolved_fields[field_index]

        num_mtf_points = mtf_field_data[0].shape[0]
        freq_tang = self.freq_tang[field_index][:num_mtf_points]
        freq_sag = self.freq_sag[field_index][:num_mtf_points]

        ax.plot(
            be.to_numpy(freq_tang),
            be.to_numpy(mtf_field_data[0]),
            label=(
                f"Hx: {current_field_label_info[0]:.1f}, "
                f"Hy: {current_field_label_info[1]:.1f}, Tangential"
            ),
            color=color,
            linestyle="-",
        )
        ax.plot(
            be.to_numpy(freq_sag),
            be.to_numpy(mtf_field_data[1]),
            label=(
                f"Hx: {current_field_label_info[0]:.1f}, "
                f"Hy: {current_field_label_info[1]:.1f}, Sagittal"
            ),
            color=color,
            linestyle="--",
        )

    def _get_mtf_units_tang(self, k):
        """Tangential frequency step (cycles/mm) with image-plane correction.

        The chief ray tilts in the tangential plane. Converting the per-field
        working F/# (measured in the chief-ray frame) to the flat image plane
        introduces a cos(θ_chief) ≈ FNO_on/FNO_off compression:

            df_tang = df_per_field * (FNO_on / FNO_off)

        For on-axis fields FNO_on == FNO_off and the correction is unity.

        Args:
            k (int): Field index.

        Returns:
            float: Tangential frequency step in cycles/mm.

        Raises:
            ValueError: If the PSF pixel pitch is zero or None.
        """
        pixel_pitch_mm = self.psf_instances[k].pixel_pitch
        if pixel_pitch_mm is None or pixel_pitch_mm == 0:
            raise ValueError("Pixel pitch from HuygensPSF is invalid.")

        on_axis_fno = self._get_fno()
        off_axis_fno = self.FNO[k]

        df_per_field = 1.0 / (self.image_size * pixel_pitch_mm)
        return df_per_field * (on_axis_fno / off_axis_fno)

    def _get_mtf_units_sag(self, k):
        """Sagittal frequency step (cycles/mm).

        There is no chief-ray tilt in the sagittal plane, so the per-field
        pixel-pitch-derived step is used directly without correction.

        Args:
            k (int): Field index.

        Returns:
            float: Sagittal frequency step in cycles/mm.

        Raises:
            ValueError: If the PSF pixel pitch is zero or None.
        """
        pixel_pitch_mm = self.psf_instances[k].pixel_pitch
        if pixel_pitch_mm is None or pixel_pitch_mm == 0:
            raise ValueError("Pixel pitch from HuygensPSF is invalid.")

        return 1.0 / (self.image_size * pixel_pitch_mm)


class VectorialHuygensMTF(ScalarHuygensMTF):
    """Vectorial Huygens Modulation Transfer Function class.

    This class calculates the MTF of an optical system using the vectorial
    Huygens-Fresnel method. It accounts for the full 3D electric field at the
    exit pupil and is intended for use with polarized optical systems. Use the
    ``HuygensMTF`` factory to automatically select between scalar and vectorial
    implementations based on the optic's polarization state.

    Inherits all constructor arguments and attributes from ``ScalarHuygensMTF``.
    """

    def _calculate_psf(self):
        """Calculates and stores the PSF using the vectorial Huygens method.

        This method uses the resolved field points and wavelength from BaseMTF,
        and explicitly uses the vectorial Huygens PSF implementation to account
        for polarization effects.
        """
        from optiland.psf.vectorial_huygens import VectorialHuygensPSF

        self.psf_data = []
        self.psf_instances = []

        # Pre-calculate normalization once (on-axis, primary wavelength)
        # This avoiding redundant PSF peak calculations for every field.
        norm_psf = VectorialHuygensPSF(
            optic=self.optic,
            field=(0, 0),
            wavelength=self.resolved_wavelength,
            num_rays=self.num_rays,
            image_size=self.image_size,
        )
        normalization = norm_psf.normalization

        for field_coord in self.resolved_fields:
            psf_calculator = VectorialHuygensPSF(
                optic=self.optic,
                field=field_coord,
                wavelength=self.resolved_wavelength,
                num_rays=self.num_rays,
                image_size=self.image_size,
                oversample=4.0,
                normalization=normalization,
            )
            self.psf_data.append(psf_calculator.psf)
            self.psf_instances.append(psf_calculator)


class HuygensMTF:
    """Factory class for generating either a Vectorial or Scalar Huygens MTF.

    This class inspects the optical system's polarization state to determine
    which Huygens MTF implementation to instantiate. If polarization is
    enabled, it returns a ``VectorialHuygensMTF``. Otherwise, it returns a
    ``ScalarHuygensMTF``.

    Args:
        optic (Optic): The optical system object.
        fields (str or list, optional): The field coordinates for which to
            calculate the MTF. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength of light to use.
            Defaults to 'primary'.
        num_rays (int, optional): The number of rays to use for the PSF
            calculation. Defaults to 128.
        image_size (int, optional): The size of the image grid for PSF
            calculation and subsequent MTF. Defaults to 128.
        max_freq (str or float, optional): The maximum frequency for the MTF
            calculation in cycles/mm. Defaults to 'cutoff'.
    """

    def __new__(
        cls,
        optic,
        fields: str | list = "all",
        wavelength: str | float = "primary",
        num_rays=128,
        image_size=128,
        max_freq="cutoff",
    ):
        if optic.polarization_state is not None:
            return VectorialHuygensMTF(
                optic=optic,
                fields=fields,
                wavelength=wavelength,
                num_rays=num_rays,
                image_size=image_size,
                max_freq=max_freq,
            )
        else:
            return ScalarHuygensMTF(
                optic=optic,
                fields=fields,
                wavelength=wavelength,
                num_rays=num_rays,
                image_size=image_size,
                max_freq=max_freq,
            )
