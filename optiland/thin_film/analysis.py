"""Thin film analysis class.

This provides thin film analysis class for optical response calculations
using the transfer matrix method (TMM).

Corentin Nannini, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import optiland.backend as be
from optiland.colorimetry import core as color_core
from optiland.colorimetry.plotting import plot_cie_1931_chromaticity_diagram

if TYPE_CHECKING:
    from .stack import ThinFilmStack

import matplotlib.pyplot as plt

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s
ELEMENTARY_CHARGE = 1.602176634e-19  # C
PLANCK_EV = PLANCK_CONSTANT / ELEMENTARY_CHARGE  # eV⋅s ≈ 4.135667696e-15

# Type definitions
Pol = Literal["s", "p", "u"]
PolInput = Pol | list[Pol]
PlotType = Literal["R", "T", "A"]
Array: TypeAlias = Any  # be.ndarray
WavelengthUnit = Literal[
    "um", "nm", "frequency", "energy", "wavenumber", "relative_wavenumber"
]
AngleUnit = Literal["deg", "rad"]


class SpectralAnalyzer:
    """Class for analyzing thin film stacks optical response (R/T/A).

    Attributes:
        stack (ThinFilmStack): The thin film stack to be analyzed.
    """

    def __init__(self, stack: ThinFilmStack) -> None:
        """Initialize the SpectralAnalyzer with a ThinFilmStack.

        Args:
            stack (ThinFilmStack): The thin film stack to be analyzed.
        """
        self.stack = stack

    def _normalize_polarizations(self, polarization: PolInput) -> list[Pol]:
        """Convert polarization input to a list of polarizations."""
        if isinstance(polarization, str):
            return [polarization]
        return polarization

    def _get_line_style(self, pol_idx: int) -> dict[str, Any]:
        """Get line style for different polarizations."""
        styles = [
            {"linestyle": "-", "alpha": 0.8},  # solid for first
            {"linestyle": "--", "alpha": 0.8},  # dashed for second
            {"linestyle": ":", "alpha": 0.8},  # dotted for third
        ]
        return styles[pol_idx % len(styles)]

    @staticmethod
    def _plot_array(values):
        """Convert backend arrays/tensors to matplotlib-safe array-like values."""
        try:
            return be.to_numpy(values)
        except Exception:
            if hasattr(values, "detach") and hasattr(values, "cpu"):
                return values.detach().cpu().tolist()
            return values

    @staticmethod
    def _plot_scalar(value) -> float:
        """Convert backend scalar/tensor to plain python float for plotting."""
        try:
            return float(value)
        except Exception:
            arr = SpectralAnalyzer._plot_array(value)
            if isinstance(arr, list):
                return float(arr[0])
            return float(arr)

    def _convert_to_wavelength_um(
        self, values: float | Array, unit: WavelengthUnit
    ) -> Array:
        """Convert input values to wavelength in micrometers.

        Args:
            values: Input values in the specified unit
            unit: Unit of the input values

        Returns:
            Wavelength values in micrometers

        Raises:
            ValueError: If relative_wavenumber is requested but no reference
            wavelength is set
        """
        values = be.atleast_1d(values)

        if unit == "um":
            return values
        elif unit == "nm":
            return values / 1000.0
        elif unit == "frequency":  # Hz to um
            # λ (m) = c / ν, then convert to um
            return (SPEED_OF_LIGHT / values) * 1e6
        elif unit == "energy":  # eV to um
            # E = hf = hc/λ, so λ = hc/E
            # λ (m) = (h * c) / E, then convert to um
            return (PLANCK_EV * SPEED_OF_LIGHT / values) * 1e6
        elif unit == "wavenumber":  # cm⁻¹ to um
            # k = 1/λ (cm⁻¹), so λ (cm) = 1/k, then convert to um
            return 1e4 / values
        elif unit == "relative_wavenumber":
            if self.stack.reference_wl_um is None:
                raise ValueError("reference_wl_um must be set for relative_wavenumber")
            # k_rel = k / k_ref, so k = k_rel * k_ref = k_rel / λ_ref
            # λ = 1/k = λ_ref / k_rel
            return self.stack.reference_wl_um / values
        else:
            raise ValueError(f"Unknown wavelength unit: {unit}")

    def _get_wavelength_axis_label(self, unit: WavelengthUnit) -> str:
        """Get the appropriate axis label for wavelength unit."""
        labels = {
            "um": r"$\lambda$ ($\mu$m)",
            "nm": r"$\lambda$ (nm)",
            "frequency": r"$\nu$ (Hz)",
            "energy": r"$E$ (eV)",
            "wavenumber": r"$k$ (cm$^{-1}$)",
            "relative_wavenumber": r"$k/k_{\mathrm{ref}}$",
        }
        return labels[unit]

    def _convert_wavelength_for_plotting(
        self, wavelength_um: Array, unit: WavelengthUnit
    ) -> Array:
        """Convert wavelength in um to the desired unit for plotting."""
        if unit == "um":
            return wavelength_um
        elif unit == "nm":
            return wavelength_um * 1000.0
        elif unit == "frequency":  # um to Hz
            return SPEED_OF_LIGHT / (wavelength_um * 1e-6)
        elif unit == "energy":  # um to eV
            return (PLANCK_EV * SPEED_OF_LIGHT) / (wavelength_um * 1e-6)
        elif unit == "wavenumber":  # um to cm⁻¹
            return 1e4 / wavelength_um
        elif unit == "relative_wavenumber":
            if self.stack.reference_wl_um is None:
                raise ValueError("reference_wl_um must be set for relative_wavenumber")
            return self.stack.reference_wl_um / wavelength_um
        else:
            raise ValueError(f"Unknown wavelength unit: {unit}")

    def _convert_angle_to_radians(
        self, angles: float | Array, unit: AngleUnit
    ) -> Array:
        """Convert angles to radians."""
        angles = be.atleast_1d(angles)
        if unit == "rad":
            return angles
        elif unit == "deg":
            return be.deg2rad(angles)
        else:
            raise ValueError(f"Unknown angle unit: {unit}")

    def wavelength_view(
        self,
        wavelength_values: float | Array,
        wavelength_unit: WavelengthUnit = "um",
        aoi: float = 0.0,
        aoi_unit: AngleUnit = "deg",
        polarization: PolInput = "u",
        to_plot: PlotType | list[PlotType] = "R",
        ax: plt.Axes = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot R/T/A vs wavelength (or equivalent units).

        Args:
            wavelength_values: Wavelength values in the specified unit
            wavelength_unit: Unit of wavelength values
            aoi: Angle of incidence (scalar)
            aoi_unit: Unit of the angle
            polarization: Polarization type(s) - single string or list
            to_plot: Quantity(ies) to plot
            ax: Optional matplotlib Axes

        Returns:
            Tuple of (figure, axes)
        """
        # Convert inputs
        wl_um = self._convert_to_wavelength_um(wavelength_values, wavelength_unit)
        aoi_rad = float(self._convert_angle_to_radians(aoi, aoi_unit).item())

        # Normalize inputs to lists
        polarizations = self._normalize_polarizations(polarization)
        if isinstance(to_plot, str):
            to_plot = [to_plot]

        # Convert wavelength back for plotting x-axis
        x_values = self._convert_wavelength_for_plotting(wl_um, wavelength_unit)
        x_plot = self._plot_array(x_values)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Plot for each polarization and quantity combination
        for pol_idx, pol in enumerate(polarizations):
            # Compute R/T/A for this polarization
            rta_data = self.stack.compute_rtRTA(wl_um, aoi_rad, pol)

            for quantity in to_plot:
                if quantity not in ("R", "T", "A"):
                    raise ValueError("to_plot must be 'R', 'T', 'A' or a list of these")

                # Get line style for this polarization
                line_style = self._get_line_style(pol_idx)

                ax.plot(
                    x_plot,
                    self._plot_array(rta_data[quantity].flatten()),
                    label=f"{quantity}, {pol}-pol, AOI={aoi}{aoi_unit}",
                    **line_style,
                )

        ax.set_xlabel(self._get_wavelength_axis_label(wavelength_unit))
        ax.set_ylabel("Power fraction")
        ax.set_xlim(self._plot_scalar(min(x_plot)), self._plot_scalar(max(x_plot)))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig, ax

    def angular_view(
        self,
        aoi_values: float | Array,
        aoi_unit: AngleUnit = "deg",
        wavelength: float = 0.55,
        wavelength_unit: WavelengthUnit = "um",
        polarization: PolInput = "u",
        to_plot: PlotType | list[PlotType] = "R",
        ax: plt.Axes = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot R/T/A vs angle of incidence.

        Args:
            aoi_values: Angle of incidence values in the specified unit
            aoi_unit: Unit of the angle values
            wavelength: Wavelength value (scalar)
            wavelength_unit: Unit of the wavelength
            polarization: Polarization type(s) - single string or list
            to_plot: Quantity(ies) to plot
            ax: Optional matplotlib Axes

        Returns:
            Tuple of (figure, axes)
        """
        # Convert inputs
        aoi_rad = self._convert_angle_to_radians(aoi_values, aoi_unit)
        wl_um = float(
            self._convert_to_wavelength_um(wavelength, wavelength_unit).item()
        )

        # Normalize inputs to lists
        polarizations = self._normalize_polarizations(polarization)
        if isinstance(to_plot, str):
            to_plot = [to_plot]

        # Convert angles back for plotting x-axis
        x_values = be.atleast_1d(aoi_values)
        x_plot = self._plot_array(x_values)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Plot for each polarization and quantity combination
        for pol_idx, pol in enumerate(polarizations):
            # Compute R/T/A for this polarization
            rta_data = self.stack.compute_rtRTA(wl_um, aoi_rad, pol)

            for quantity in to_plot:
                if quantity not in ("R", "T", "A"):
                    raise ValueError("to_plot must be 'R', 'T', 'A' or a list of these")

                # Get line style for this polarization
                line_style = self._get_line_style(pol_idx)

                # Create label parts
                wl_axis_label = self._get_wavelength_axis_label(wavelength_unit)
                wl_symbol = wl_axis_label.split("(")[0].strip()

                ax.plot(
                    x_plot,
                    self._plot_array(rta_data[quantity].flatten()),
                    label=f"{quantity}, {pol}-pol, {wl_symbol}={wavelength}",
                    **line_style,
                )

        xlabel = r"AOI (°)" if aoi_unit == "deg" else r"AOI (rad)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Power fraction")
        ax.set_xlim(self._plot_scalar(min(x_plot)), self._plot_scalar(max(x_plot)))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig, ax

    def _resolve_map_grid(
        self,
        wavelength_values: float | Array,
        wavelength_unit: WavelengthUnit,
        aoi_values: float | Array,
        aoi_unit: AngleUnit,
        polarization: PolInput,
        to_plot: PlotType | list[PlotType],
    ) -> tuple[Array, Array, list[Pol], list[PlotType], Array, Array]:
        """Normalize map_view inputs into computation- and plot-ready arrays."""
        if aoi_values is None:
            aoi_values = (
                be.linspace(0, 80, 81)
                if aoi_unit == "deg"
                else be.linspace(0, be.deg2rad(80), 81)
            )

        wl_um = self._convert_to_wavelength_um(wavelength_values, wavelength_unit)
        aoi_rad = self._convert_angle_to_radians(aoi_values, aoi_unit)

        polarizations = self._normalize_polarizations(polarization)
        if isinstance(to_plot, str):
            to_plot = [to_plot]
        for quantity in to_plot:
            if quantity not in ("R", "T", "A"):
                raise ValueError("to_plot must be 'R', 'T', 'A' or a list of these")

        wl_plot = self._convert_wavelength_for_plotting(wl_um, wavelength_unit)
        aoi_plot = be.atleast_1d(aoi_values)

        # Create meshgrid for plotting with backend-agnostic ij indexing
        aoi_grid, wl_grid = be.meshgrid(aoi_plot, wl_plot)
        wl_mesh = self._plot_array(wl_grid)
        aoi_mesh = self._plot_array(aoi_grid)

        return wl_um, aoi_rad, polarizations, to_plot, wl_mesh, aoi_mesh

    def _prepare_map_axes(
        self,
        fig: plt.Figure,
        axs: plt.Axes | list[plt.Axes],
        nrows: int,
        ncols: int,
    ) -> tuple[plt.Figure, list[list[plt.Axes]]]:
        """Create or normalize the figure/axes grid for map_view."""
        if fig is None or axs is None:
            # Organize subplots: polarizations as columns, quantities as rows
            fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))

            # Ensure axs is always 2D array for consistent indexing
            if nrows == 1 and ncols == 1:
                axs = [[axs]]
            elif nrows == 1:
                axs = [axs]
            elif ncols == 1:
                axs = [[ax] for ax in axs]
        else:
            # If axs is provided, assume it's properly formatted
            if not isinstance(axs, list):
                axs = [[axs]]
            elif not isinstance(axs[0], list):
                axs = [axs]

        return fig, axs

    @staticmethod
    def _select_map_return(
        fig: plt.Figure,
        axs: list[list[plt.Axes]],
        to_plot: list[PlotType],
        polarizations: list[Pol],
    ) -> tuple[plt.Figure, plt.Axes | list[plt.Axes]]:
        """Pick the return shape for map_view based on the requested grid."""
        if len(to_plot) == 1 and len(polarizations) == 1:
            return fig, axs[0][0]
        elif len(to_plot) == 1:
            return fig, axs[0]  # Return list of axes for different polarizations
        elif len(polarizations) == 1:
            return fig, [
                row[0] for row in axs
            ]  # Return list of axes for different quantities
        else:
            return fig, axs  # Return 2D array of axes

    def map_view(
        self,
        wavelength_values: float | Array,
        wavelength_unit: WavelengthUnit = "um",
        aoi_values: float | Array = None,
        aoi_unit: AngleUnit = "deg",
        polarization: PolInput = "u",
        to_plot: PlotType | list[PlotType] = "R",
        colormap: str = "viridis",
        fig: plt.Figure = None,
        axs: plt.Axes | list[plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes | list[plt.Axes]]:
        """Plot 2D maps of R/T/A vs wavelength and angle of incidence.

        Args:
            wavelength_values: Wavelength values in the specified unit
            wavelength_unit: Unit of wavelength values
            aoi_values: Angle of incidence values in the specified unit
            aoi_unit: Unit of the angle values
            polarization: Polarization type(s) - single string or list
            to_plot: Quantity(ies) to plot
            fig: Optional matplotlib Figure
            axs: Optional matplotlib Axes (single or list)

        Returns:
            Tuple of (figure, axes or list of axes)
        """
        wl_um, aoi_rad, polarizations, to_plot, wl_mesh, aoi_mesh = (
            self._resolve_map_grid(
                wavelength_values,
                wavelength_unit,
                aoi_values,
                aoi_unit,
                polarization,
                to_plot,
            )
        )

        fig, axs = self._prepare_map_axes(
            fig, axs, nrows=len(to_plot), ncols=len(polarizations)
        )

        # Plot each quantity and polarization combination
        for qty_idx, quantity in enumerate(to_plot):
            for pol_idx, pol in enumerate(polarizations):
                # Compute R/T/A for this polarization
                rta_data = self.stack.compute_rtRTA(wl_um, aoi_rad, pol)

                ax_i = axs[qty_idx][pol_idx]
                im = ax_i.pcolormesh(
                    wl_mesh,
                    aoi_mesh,
                    self._plot_array(rta_data[quantity]),
                    shading="auto",
                    vmin=0,
                    vmax=1,
                    cmap=colormap,
                )

                ax_i.set_xlabel(self._get_wavelength_axis_label(wavelength_unit))
                ylabel = r"AOI (°)" if aoi_unit == "deg" else r"AOI (rad)"
                ax_i.set_ylabel(ylabel)
                ax_i.set_title(f"{quantity}, {pol}-pol")

                # Add colorbar
                cbar = fig.colorbar(im, ax=ax_i, label="Power fraction")
                cbar.set_label("Power fraction")

        fig.tight_layout()

        return self._select_map_return(fig, axs, to_plot, polarizations)

    def _get_single_polarization(self, polarization: PolInput) -> Pol:
        """Normalize and validate a single polarization selection."""
        polarizations = self._normalize_polarizations(polarization)
        if len(polarizations) != 1:
            raise ValueError("Color analysis requires a single polarization")
        return polarizations[0]

    def _get_rt_spectrum(
        self,
        wavelength_values: float | Array,
        wavelength_unit: WavelengthUnit = "um",
        aoi: float = 0.0,
        aoi_unit: AngleUnit = "deg",
        polarization: PolInput = "u",
        quantity: Literal["R", "T"] = "R",
    ) -> tuple[list[float], list[float]]:
        """Return normalized power spectrum (R or T) in nm."""
        if quantity not in ("R", "T"):
            raise ValueError("quantity must be 'R' or 'T'")

        wl_um = self._convert_to_wavelength_um(wavelength_values, wavelength_unit)
        aoi_rad = float(self._convert_angle_to_radians(aoi, aoi_unit).item())
        pol = self._get_single_polarization(polarization)

        rta_data = self.stack.compute_rtRTA(wl_um, aoi_rad, pol)

        wl_nm = self._convert_wavelength_for_plotting(wl_um, "nm")
        values = rta_data[quantity].flatten()

        return wl_nm.tolist(), values.tolist()

    def spectrum_to_xyY(
        self,
        wavelength_values: float | Array,
        wavelength_unit: WavelengthUnit = "um",
        aoi: float = 0.0,
        aoi_unit: AngleUnit = "deg",
        polarization: PolInput = "u",
        quantity: Literal["R", "T"] = "R",
        observer: Literal["2deg", "10deg"] = "2deg",
        illuminant: list[float] | None = None,
    ) -> tuple[float, float, float]:
        """Compute xyY chromaticity from a normalized power spectrum.

        Args:
            wavelength_values: Wavelength values in the specified unit.
            wavelength_unit: Unit of wavelength values.
            aoi: Angle of incidence (scalar).
            aoi_unit: Unit of the angle.
            polarization: Polarization type(s).
            quantity: Quantity to analyze ('R' or 'T').
            observer: CIE standard observer ('2deg' or '10deg').
            illuminant: Optional custom illuminant spectrum.

        Returns:
            tuple[float, float, float]: (x, y, Y) chromaticity coordinates.
        """
        wavelengths_nm, values = self._get_rt_spectrum(
            wavelength_values=wavelength_values,
            wavelength_unit=wavelength_unit,
            aoi=aoi,
            aoi_unit=aoi_unit,
            polarization=polarization,
            quantity=quantity,
        )

        X, Y, Z = color_core.spectrum_to_xyz(
            wavelengths=wavelengths_nm,
            values=values,
            illuminant=illuminant,
            observer=observer,
        )
        x, y, Y = color_core.xyz_to_xyY(X, Y, Z)
        return float(x), float(y), float(Y)

    def analyze_color(
        self,
        wavelength_values: float | Array,
        wavelength_unit: WavelengthUnit = "um",
        aoi: float = 0.0,
        aoi_unit: AngleUnit = "deg",
        polarization: PolInput = "u",
        quantity: Literal["R", "T"] = "R",
        observer: Literal["2deg", "10deg"] = "2deg",
        illuminant: list[float] | None = None,
    ) -> dict[str, tuple[float, float, float] | tuple[int, int, int]]:
        """Return XYZ, xyY, and sRGB for a thin-film spectrum.

        Args:
            wavelength_values: Wavelength values in the specified unit.
            wavelength_unit: Unit of wavelength values.
            aoi: Angle of incidence (scalar).
            aoi_unit: Unit of the angle.
            polarization: Polarization type(s).
            quantity: Quantity to analyze ('R' or 'T').
            observer: CIE standard observer ('2deg' or '10deg').
            illuminant: Optional custom illuminant spectrum.

        Returns:
            dict: Dictionary containing 'xyz', 'xyY', and 'sRGB' values.
        """
        wavelengths_nm, values = self._get_rt_spectrum(
            wavelength_values=wavelength_values,
            wavelength_unit=wavelength_unit,
            aoi=aoi,
            aoi_unit=aoi_unit,
            polarization=polarization,
            quantity=quantity,
        )

        X, Y, Z = color_core.spectrum_to_xyz(
            wavelengths=wavelengths_nm,
            values=values,
            illuminant=illuminant,
            observer=observer,
        )
        x, y, Y = color_core.xyz_to_xyY(X, Y, Z)
        r, g, b = color_core.xyz_to_srgb(X, Y, Z)

        return {
            "xyz": (float(X), float(Y), float(Z)),
            "xyY": (float(x), float(y), float(Y)),
            "sRGB": (int(r), int(g), int(b)),
        }

    def plot_color_on_cie_1931(
        self,
        wavelength_values: float | Array,
        wavelength_unit: WavelengthUnit = "um",
        aoi: float = 0.0,
        aoi_unit: AngleUnit = "deg",
        polarization: PolInput = "u",
        quantity: Literal["R", "T"] = "R",
        observer: Literal["2deg", "10deg"] = "2deg",
        illuminant: list[float] | None = None,
        ax: plt.Axes | None = None,
        color: Literal["no", "contour", "fill"] = "contour",
        marker: str = "o",
        marker_size: float = 6.0,
        marker_color: str = "black",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the chromaticity point on a CIE 1931 diagram."""
        fig, ax = plot_cie_1931_chromaticity_diagram(ax=ax, color=color)
        x, y, _ = self.spectrum_to_xyY(
            wavelength_values=wavelength_values,
            wavelength_unit=wavelength_unit,
            aoi=aoi,
            aoi_unit=aoi_unit,
            polarization=polarization,
            quantity=quantity,
            observer=observer,
            illuminant=illuminant,
        )
        x_plot = self._plot_scalar(x)
        y_plot = self._plot_scalar(y)
        ax.plot(
            x_plot, y_plot, marker=marker, markersize=marker_size, color=marker_color
        )
        ax.text(
            x_plot + 0.02,
            y_plot + 0.02,
            f"x={x_plot:.4f}, y={y_plot:.4f}",
            fontsize=9,
            ha="left",
            va="bottom",
        )
        return fig, ax
