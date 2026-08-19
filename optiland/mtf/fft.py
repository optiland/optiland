"""Fast Fourier Transform Modulation Transfer Function (FFTMTF) Module.

This module provides the ScalarFFTMTF, VectorialFFTMTF (via factory), and
FFTMTF factory class for computing the MTF of an optical system using FFT
techniques.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.psf.fft import ScalarFFTPSF, calculate_grid_size
from optiland.utils import get_working_FNO

from .base import BaseMTF


class ScalarFFTMTF(BaseMTF):
    """Scalar Fast Fourier Transform Modulation Transfer Function class.

    This class calculates and visualizes the Modulation Transfer Function (MTF)
    of an optic using the scalar FFT method. It is intended for use with
    unpolarized optical systems. Use the `FFTMTF` factory to automatically
    select between scalar and vectorial implementations based on the optic's
    polarization state.

    Args:
        optic (Optic): The optic for which to calculate the MTF.
        fields (str or list, optional): The field coordinates for which to
            calculate the MTF. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength of light to use
            for the MTF calculation. Defaults to 'primary'.
        num_rays (int, optional): The number of rays to use for the PSF
            calculation, which is then used for MTF. Defaults to 128.
        grid_size (int or None, optional): The size of the grid used for the PSF/MTF
            calculation. If `None`, the grid size will be calculated from `num_rays`
            as documented in `optiland.psf.fft.ScalarFFTPSF`. Defaults to `None`.
        max_freq (str or float, optional): The maximum frequency for the MTF
            calculation. Defaults to 'cutoff'.
        strategy (str): The calculation strategy to use. Supported options are
            "chief_ray", "centroid_sphere", and "best_fit_sphere".
            Defaults to "chief_ray".
        remove_tilt (bool): If True, removes tilt and piston from the OPD data.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the strategy.

    Attributes:
        num_rays (int): The number of rays used for the MTF calculation.
        grid_size (int): The size of the grid used for the MTF calculation.
        max_freq (float): The maximum frequency for the MTF calculation.
        FNO (float): The F-number of the optic.
        psf (list): List of PSF data for each field.
        mtf (list): List of MTF data ([tangential, sagittal]) for each field.
        freq (be.ndarray): Array of frequency points for the MTF curve.
    """

    def __init__(
        self,
        optic,
        fields: str | list = "all",
        wavelength: str | float = "primary",
        num_rays=128,
        grid_size=None,
        max_freq="cutoff",
        strategy="chief_ray",
        remove_tilt=False,
        **kwargs,
    ):
        if grid_size is None:
            self.num_rays, self.grid_size = calculate_grid_size(num_rays)
        else:
            self.num_rays = num_rays
            self.grid_size = grid_size

        super().__init__(optic, fields, wavelength, strategy, remove_tilt, **kwargs)

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
            be.arange(self.grid_size // 2) * self._get_mtf_units_tang(k)
            for k in range(n_fields)
        ]
        self.freq_sag = [
            be.arange(self.grid_size // 2) * self._get_mtf_units_sag(k)
            for k in range(n_fields)
        ]
        # Backward-compatible alias (tangential is the primary reference).
        self.freq = self.freq_tang

    def _calculate_psf(self):
        """Calculates and stores the Point Spread Function (PSF).

        This method uses the resolved field points and wavelength from BaseMTF,
        and explicitly uses the scalar FFT PSF implementation.
        """
        self.psf = [
            ScalarFFTPSF(
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

    def _plot_field_mtf(self, ax, field_index, mtf_field_data, color):
        """Plots the MTF data for a single field for ScalarFFTMTF.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axes object.
            field_index (int): The index of the current field in self.resolved_fields.
            mtf_field_data (list): A list containing tangential and sagittal
                                   MTF data (be.ndarray) for the field.
            color (str): The color to use for plotting this field.
        """
        current_field_label_info = self.resolved_fields[field_index]

        # Plot tangential MTF (uses the image-plane-corrected frequency axis)
        ax.plot(
            be.to_numpy(self.freq_tang[field_index]),
            be.to_numpy(mtf_field_data[0]),
            label=(
                f"Hx: {current_field_label_info[0]:.1f}, "
                f"Hy: {current_field_label_info[1]:.1f}, Tangential"
            ),
            color=color,
            linestyle="-",
        )
        # Plot sagittal MTF (no tilt in the sagittal plane — use per-field axis)
        ax.plot(
            be.to_numpy(self.freq_sag[field_index]),
            be.to_numpy(mtf_field_data[1]),
            label=(
                f"Hx: {current_field_label_info[0]:.1f}, "
                f"Hy: {current_field_label_info[1]:.1f}, Sagittal"
            ),
            color=color,
            linestyle="--",
        )

    def _generate_mtf_data(self):
        """Generates the MTF data for each field.

        The OTF is computed as the 2D FFT of the PSF. The DC component (zero
        spatial frequency) is located at index ``(grid_size // 2, grid_size // 2)``
        after ``fftshift``. The tangential and sagittal MTF slices are extracted
        from that center outward and normalized by the DC value so that MTF(0) = 1,
        consistent with the incoherent imaging convention used by OpticStudio.

        Returns:
            list: A list of MTF data for each field. Each element is a list
                ``[tangential_mtf, sagittal_mtf]`` where each is a 1-D array of
                length ``grid_size // 2`` with values in ``[0, 1]``.
        """
        mtf_data = [be.abs(be.fft.fftshift(be.fft.fft2(psf))) for psf in self.psf]
        mtf = []
        center = self.grid_size // 2
        for data in mtf_data:
            # Extract 1-D slices from the DC bin outward, clipped to grid_size // 2
            tangential = data[center:, center][:center]
            sagittal = data[center, center:][:center]

            # Normalize by the DC value (OTF at zero frequency = total PSF power).
            # Physical MTF must satisfy MTF(0) = 1; the DC bin is always the maximum
            # for a well-behaved incoherent system.
            dc_value = data[center, center]
            if dc_value == 0:
                norm_tangential = be.zeros_like(tangential)
                norm_sagittal = be.zeros_like(sagittal)
            else:
                norm_tangential = tangential / dc_value
                norm_sagittal = sagittal / dc_value

            # Guard against floating-point overshoot beyond the physical [0, 1] range.
            norm_tangential = be.clip(norm_tangential, 0.0, 1.0)
            norm_sagittal = be.clip(norm_sagittal, 0.0, 1.0)

            mtf.append([norm_tangential, norm_sagittal])
        return mtf

    def _get_mtf_units_tang(self, k):
        """Tangential frequency step (cycles/mm) with image-plane correction.

        The chief ray tilts in the tangential plane.  Converting the per-field
        working F/# (measured in the chief-ray frame) to the flat image plane
        introduces a cos(θ_chief) ≈ FNO_on/FNO_off compression:

            df_tang = df_chief * (FNO_on / FNO_off)

        For on-axis fields FNO_on == FNO_off and the correction is unity.

        Args:
            k (int): Field index.

        Returns:
            float: Tangential frequency step in cycles/mm.
        """
        on_axis_fno = self._get_fno()
        off_axis_fno = self.FNO[k]
        df_chief = 1 / (
            (self.num_rays - 1) * self.resolved_wavelength * 1e-3 * off_axis_fno
        )
        return df_chief * (on_axis_fno / off_axis_fno)

    def _get_mtf_units_sag(self, k):
        """Sagittal frequency step (cycles/mm).

        There is no chief-ray tilt in the sagittal plane, so the per-field
        working F/# (chief-ray frame) is used directly.

        Args:
            k (int): Field index.

        Returns:
            float: Sagittal frequency step in cycles/mm.
        """
        off_axis_fno = self.FNO[k]
        return 1 / (
            (self.num_rays - 1) * self.resolved_wavelength * 1e-3 * off_axis_fno
        )


class FFTMTF:
    """Factory class for generating either a Vectorial or Scalar FFT MTF.

    This class inspects the optical system's polarization state to determine
    which FFT MTF implementation to instantiate. If polarization is enabled,
    it returns a `VectorialFFTMTF`. Otherwise, it returns a `ScalarFFTMTF`.

    Args:
        optic (Optic): The optical system object.
        fields (str or list, optional): The field coordinates for which to
            calculate the MTF. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength of light to use.
            Defaults to 'primary'.
        num_rays (int, optional): The number of rays to use for the PSF
            calculation. Defaults to 128.
        grid_size (int or None, optional): The FFT grid size. Defaults to None.
        max_freq (str or float, optional): The maximum frequency for the MTF
            calculation. Defaults to 'cutoff'.
        strategy (str): The wavefront calculation strategy. Defaults to
            "chief_ray".
        remove_tilt (bool): If True, removes tilt from OPD. Defaults to False.
        **kwargs: Additional keyword arguments passed to the strategy.
    """

    def __new__(
        cls,
        optic,
        fields: str | list = "all",
        wavelength: str | float = "primary",
        num_rays=128,
        grid_size=None,
        max_freq="cutoff",
        strategy="chief_ray",
        remove_tilt=False,
        **kwargs,
    ):
        if optic.polarization_state is not None:
            from optiland.mtf.vectorial_fft import VectorialFFTMTF

            return VectorialFFTMTF(
                optic=optic,
                fields=fields,
                wavelength=wavelength,
                num_rays=num_rays,
                grid_size=grid_size,
                max_freq=max_freq,
                strategy=strategy,
                remove_tilt=remove_tilt,
                **kwargs,
            )
        else:
            return ScalarFFTMTF(
                optic=optic,
                fields=fields,
                wavelength=wavelength,
                num_rays=num_rays,
                grid_size=grid_size,
                max_freq=max_freq,
                strategy=strategy,
                remove_tilt=remove_tilt,
                **kwargs,
            )
