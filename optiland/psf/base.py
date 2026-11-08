"""Base PSF Module

This module provides a base class for Point Spread Function (PSF) calculations.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal
from warnings import warn

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import zoom

import optiland.backend as be
from optiland.utils import get_working_FNO, resolve_wavelength
from optiland.wavefront import Wavefront

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D

    from optiland._types import BEArray
    from optiland.fields import Field
    from optiland.optic import Optic


def replace_nonpositive(image, min_value=1e-9):
    """
    Replace values <= 0 in the image with the smallest positive value in the image.
    If no positive value exists, use min_value.

    Args:
        image: Array (backend or numpy) to process.
        min_value: Value to use if no positive values exist (default: 1e-9).

    Returns:
        Array with non-positive values replaced.
    """
    if be.any(image <= 0):
        min_positive = be.min(image[image > 0]) if be.any(image > 0) else min_value
        return be.where(image <= 0, min_positive, image)
    return image


class BasePSF(Wavefront):
    """Base class for Point Spread Function (PSF) calculations.

    Args:
        optic (Optic): The optical system.
        field (tuple): The field as (x, y) at which to compute the PSF.
        wavelength (str | float): The wavelength of light. Can be 'primary' or a
            float value.
        num_rays (int, optional): The number of rays used for wavefront
            computation. Defaults to 128.
        strategy (str): The calculation strategy to use. Supported options are
            "chief_ray", "centroid_sphere", and "best_fit_sphere".
            Defaults to "chief_ray".
        remove_tilt (bool): If True, removes tilt and piston from the OPD data.
            Defaults to True.
        **kwargs: Additional keyword arguments passed to the strategy.

    Attributes:
        psf (ndarray): The computed PSF. This should be set by subclasses.

    Methods:
        view(projection='2d', log=False, figsize=(7, 5.5), threshold=0.05,
            num_points=128): Visualizes the PSF.
        cross_section(axis='y', through='peak'): Returns a 1D slice of the PSF.
        line_spread_function(axis='y'): Returns the line spread function.
        view_cross_section(axis='both', log=False): Plots PSF cross-sections.
    """

    def __init__(
        self,
        optic: Optic,
        field: Field,
        wavelength: str | float,
        num_rays=128,
        strategy="chief_ray",
        remove_tilt=True,
        **kwargs,
    ):
        resolved_wavelength = resolve_wavelength(optic, wavelength)
        super().__init__(
            optic=optic,
            fields=[field],
            wavelengths=[resolved_wavelength],
            num_rays=num_rays,
            distribution="uniform",
            strategy=strategy,
            remove_tilt=remove_tilt,
            **kwargs,
        )
        self.psf = None  # Subclasses must compute and set this

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        projection: str = "2d",
        log: bool = False,
        figsize: tuple = (7, 5.5),
        threshold: float = 0.05,
        num_points: int = 128,
    ) -> tuple[Figure, Axes]:
        """Visualizes the PSF.

        Args:
            projection (str, optional): The projection type. Can be '2d' or
                '3d'. Defaults to '2d'.
            log (bool, optional): Whether to use a logarithmic scale for the
                intensity. Defaults to False.
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).
            threshold (float, optional): The threshold for determining the
                bounds of the PSF for zoomed view. Defaults to 0.05.
            num_points (int, optional): The number of points used for
                interpolating the PSF for smoother visualization. Defaults to 128.
        Returns:
            tuple: A tuple containing the figure and axes objects.

        Raises:
            RunentimeError: If the PSF has not been computed.
            ValueError: If the projection is not '2d' or '3d'.
            RuntimeError: If the PSF has not been computed by the subclass.
        """
        if self.psf is None:
            raise RuntimeError(
                "PSF has not been computed. Call _compute_psf in subclass."
            )
        is_gui_embedding = fig_to_plot_on is not None
        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = (
                current_fig.add_subplot(111)
                if projection == "2d"
                else current_fig.add_subplot(111, projection="3d")
            )
        else:
            current_fig, ax = (
                plt.subplots(figsize=figsize)
                if projection == "2d"
                else plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)
            )

        psf_np = be.to_numpy(self.psf)
        min_x, min_y, max_x, max_y = self._find_bounds(psf_np, threshold)
        psf_zoomed = psf_np[min_x:max_x, min_y:max_y]

        oversampling_factor = num_points / psf_zoomed.shape[0]

        if oversampling_factor > 3:
            message = (
                f"The PSF view has a high oversampling factor "
                f"({oversampling_factor:.2f}). Results may be inaccurate."
            )
            warn(message, stacklevel=2)

        # Subclasses should implement _get_psf_units if they want physical units
        # otherwise, pixel units are used.
        if hasattr(self, "_get_psf_units"):
            x_extent, y_extent = self._get_psf_units(psf_zoomed)
            x_label, y_label = "X (µm)", "Y (µm)"
        else:
            # Default to pixel units if not implemented by subclass
            x_extent = psf_zoomed.shape[1]
            y_extent = psf_zoomed.shape[0]
            x_label, y_label = "X (pixels)", "Y (pixels)"

        psf_smooth = self._interpolate_psf(psf_zoomed, num_points)

        if projection == "2d":
            self._plot_2d(
                current_fig,
                ax,
                psf_smooth,
                log,
                x_extent,
                y_extent,
                figsize,
                x_label,
                y_label,
                psf_zoomed.shape,
            )
            return current_fig, ax
        elif projection == "3d":
            self._plot_3d(
                current_fig,
                ax,
                psf_smooth,
                log,
                x_extent,
                y_extent,
                figsize,
                x_label,
                y_label,
                psf_zoomed.shape,
            )
            return current_fig, ax
        # Raise error if projection is not recognized
        else:
            raise ValueError('Projection must be "2d" or "3d".')
        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        return current_fig, ax

    def _plot_2d(
        self,
        fig: Figure,
        ax: Axes,
        image: np.ndarray,
        log: bool,
        x_extent: float,
        y_extent: float,
        figsize: tuple,
        x_label: str,
        y_label: str,
        original_size: tuple,
    ) -> None:
        """Plots the PSF in 2D.

        Args:
            image (numpy.ndarray): The 2D image of the PSF to plot.
            log (bool): If True, apply logarithmic normalization to the image.
            x_extent (float): The extent of the x-axis.
            y_extent (float): The extent of the y-axis.
            figsize (tuple): The size of the figure.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            original_size (tuple): The original size of the PSF image before
                interpolation.
        """
        norm = LogNorm() if log else None

        # Replace values <= 0 with smallest non-zero value in image for log scale
        if log and be.any(image <= 0):
            image = replace_nonpositive(image)

        extent = [-x_extent / 2, x_extent / 2, -y_extent / 2, y_extent / 2]
        im = ax.imshow(be.to_numpy(image), norm=norm, extent=extent, origin="lower")

        self._annotate_original_size(fig, original_size)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{self.__class__.__name__.replace('PSF', ' PSF')}")

        cbar = plt.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Relative Intensity (%)", rotation=270)

    def _plot_3d(
        self,
        fig: Figure,
        ax: Axes3D,
        image: np.ndarray,
        log: bool,
        x_extent: float,
        y_extent: float,
        figsize: tuple,
        x_label: str,
        y_label: str,
        original_size: tuple,
    ) -> None:
        """Plots the PSF in 3D.

        Args:
            image (numpy.ndarray): The PSF image data.
            log (bool): Whether to apply logarithmic scaling to the image.
            x_extent (float): The extent of the x-axis.
            y_extent (float): The extent of the y-axis.
            figsize (tuple): The size of the figure.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            original_size (tuple): The original size of the PSF image before
                interpolation.
        """

        x_np = be.to_numpy(be.linspace(-x_extent / 2, x_extent / 2, image.shape[1]))
        y_np = be.to_numpy(be.linspace(-y_extent / 2, y_extent / 2, image.shape[0]))
        X_np, Y_np = np.meshgrid(x_np, y_np)

        # Replace values <= 0 with smallest non-zero value in image for log scale
        if log and be.any(image <= 0):
            image = replace_nonpositive(image)

        image_np = be.to_numpy(image)

        log_formatter = None
        if log:
            image_plot = np.log10(image_np)
            formatter = mticker.FuncFormatter(self._log_tick_formatter)
            ax.zaxis.set_major_formatter(formatter)
            ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            log_formatter = self._log_colorbar_formatter
        else:
            image_plot = image_np

        surf = ax.plot_surface(
            X_np,
            Y_np,
            image_plot,
            rstride=1,
            cstride=1,
            cmap="viridis",
            linewidth=0,
            antialiased=False,
        )

        self._annotate_original_size(fig, original_size)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel("Relative Intensity (%)")
        ax.set_title(f"{self.__class__.__name__.replace('PSF', ' PSF')}")

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.15, format=log_formatter)
        fig.tight_layout()

    def _log_tick_formatter(self, value, pos=None):
        """Formats tick labels for a logarithmic scale (Z-axis in 3D plot)."""
        return f"$10^{{{int(value)}}}$"

    def _log_colorbar_formatter(self, value, pos=None):
        """Formats tick labels for a logarithmic colorbar."""
        linear_value = 10**value
        return f"{linear_value:.1e}"

    def _annotate_original_size(self, fig: Figure, original_size):
        """Annotates the original size of the zoomed PSF in the bottom right corner."""
        text = f"Original Size: {original_size[0]}×{original_size[1]}"
        fig.text(
            0.99,
            0.01,
            text,
            transform=fig.transFigure,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    def _interpolate_psf(self, image, n=128):
        """Interpolates the PSF for visualization.

        Uses scipy.ndimage.zoom for interpolation. Converts to NumPy
        as zoom requires NumPy array.

        Args:
            image (ndarray): The input image (can be backend array).
            n (int, optional): The number of points in the interpolated
                grid. Defaults to 128.

        Returns:
            ndarray: The interpolated PSF grid (backend array).
        """
        image_np = be.to_numpy(image)
        zoom_factor = n / image_np.shape[0]

        if zoom_factor == 1:
            return image  # Return original backend array if no zoom

        interpolated_np = zoom(image_np, zoom_factor, order=3)
        return be.array(interpolated_np)

    @staticmethod
    def _find_bounds(psf, threshold=0.25):
        """Finds the bounding box coordinates for the non-zero elements in the
        PSF matrix.

        Args:
            psf (numpy.ndarray): The PSF matrix.
            threshold (float): The threshold value for determining non-zero
                elements in the PSF matrix. Default is 0.25.

        Returns:
            tuple: A tuple containing the minimum and maximum x and y
                coordinates of the bounding box.

        """
        thresholded_psf = psf > threshold
        non_zero_indices = np.argwhere(thresholded_psf)

        try:
            min_x, min_y = np.min(non_zero_indices, axis=0)
            max_x, max_y = np.max(non_zero_indices, axis=0)
        except ValueError:
            min_x, min_y = 0, 0
            max_x, max_y = psf.shape

        size = max(max_x - min_x, max_y - min_y)

        peak_x, peak_y = psf.shape[0] // 2, psf.shape[1] // 2

        min_x = peak_x - size / 2
        max_x = peak_x + size / 2
        min_y = peak_y - size / 2
        max_y = peak_y + size / 2

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(psf.shape[0], max_x)
        max_y = min(psf.shape[1], max_y)

        return int(min_x), int(min_y), int(max_x), int(max_y)

    @abstractmethod
    def _compute_psf(self):
        """Computes the PSF.

        This method must be implemented by subclasses.
        It should calculate the PSF and store it in self.psf.
        """
        raise NotImplementedError("Subclasses must implement _compute_psf.")

    def strehl_ratio(self):
        """Computes the Strehl ratio of the PSF.

        The Strehl ratio is the ratio of the peak intensity of the aberrated
        PSF to the peak intensity of the diffraction-limited PSF.
        Assumes self.psf is normalized such that its peak would be 1.0 (or 100%)
        for a diffraction-limited system.

        Returns:
            float: The Strehl ratio.

        Raises:
            RuntimeError: If the PSF has not been computed.
        """
        if self.psf is None:
            raise RuntimeError("PSF has not been computed.")

        strehl = be.max(self.psf) / 100
        return float(be.to_numpy(strehl).item())

    def cross_section(
        self,
        axis: Literal["x", "y"] = "y",
        through: Literal["peak", "center"] = "peak",
    ) -> tuple[BEArray, BEArray]:
        """Returns a 1D cross-section of the PSF.

        Rows of the PSF array run along y and columns along x, as in `view`.

        Args:
            axis (str, optional): The axis along which the slice runs, 'x' or
                'y'. Defaults to 'y'.
            through (str, optional): Where the slice is taken. 'peak' uses the
                row/column of the PSF maximum, 'center' uses the array centre.
                Defaults to 'peak'.

        Returns:
            tuple: A tuple ``(coords, values)``. ``coords`` are the pixel
            centre positions along `axis`, with pixel ``n // 2`` at 0, in µm
            if the subclass implements `_get_psf_units` and in pixels
            otherwise. ``values`` are the PSF values as stored (not
            renormalized).

        Raises:
            RuntimeError: If the PSF has not been computed.
            ValueError: If `axis` is not 'x' or 'y', or `through` is not
                'peak' or 'center'.
        """
        if self.psf is None:
            raise RuntimeError(
                "PSF has not been computed. Call _compute_psf in subclass."
            )
        if axis not in ("x", "y"):
            raise ValueError('Axis must be "x" or "y".')

        if through == "peak":
            peak = be.argwhere(self.psf == be.max(self.psf))[0]
            row, col = (int(index) for index in be.to_numpy(peak))
        elif through == "center":
            row, col = self.psf.shape[0] // 2, self.psf.shape[1] // 2
        else:
            raise ValueError('Through must be "peak" or "center".')

        values = self.psf[row, :] if axis == "x" else self.psf[:, col]
        dx, dy, _ = self._get_pixel_pitch()
        return self._cross_section_coordinates(axis, dx, dy), values

    def line_spread_function(
        self, axis: Literal["x", "y"] = "y"
    ) -> tuple[BEArray, BEArray]:
        """Returns the line spread function (LSF) of the PSF.

        The LSF along `axis` is the PSF integrated along the other axis (sum
        times pixel pitch), so the LSF integrates to the same total as the PSF.

        Args:
            axis (str, optional): The axis along which the LSF is returned,
                'x' or 'y'. Defaults to 'y'.

        Returns:
            tuple: A tuple ``(coords, lsf)``. ``coords`` are the pixel centre
            positions along `axis`, in the same units as `cross_section`.

        Raises:
            RuntimeError: If the PSF has not been computed.
            ValueError: If `axis` is not 'x' or 'y'.
        """
        if self.psf is None:
            raise RuntimeError(
                "PSF has not been computed. Call _compute_psf in subclass."
            )
        if axis not in ("x", "y"):
            raise ValueError('Axis must be "x" or "y".')

        dx, dy, _ = self._get_pixel_pitch()
        if axis == "y":
            lsf = be.sum(self.psf, axis=1) * dx
        else:
            lsf = be.sum(self.psf, axis=0) * dy
        return self._cross_section_coordinates(axis, dx, dy), lsf

    def view_cross_section(
        self,
        axis: Literal["x", "y", "both"] = "both",
        log: bool = False,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (7, 4.5),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Plots cross-sections of the PSF through its peak.

        Args:
            axis (str, optional): The cross-section(s) to plot: 'x', 'y' or
                'both'. Defaults to 'both'.
            log (bool, optional): Whether to use a logarithmic scale for the
                intensity. Defaults to False.
            fig_to_plot_on (Figure, optional): The figure to plot on. If None,
                a new figure is created. Defaults to None.
            figsize (tuple[float, float], optional): The figure size.
                Defaults to (7, 4.5).
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.

        Returns:
            tuple: A tuple containing the figure and axes objects.

        Raises:
            RuntimeError: If the PSF has not been computed.
            ValueError: If `axis` is not 'x', 'y' or 'both'.
        """
        if self.psf is None:
            raise RuntimeError(
                "PSF has not been computed. Call _compute_psf in subclass."
            )
        if axis not in ("x", "y", "both"):
            raise ValueError('Axis must be "x", "y" or "both".')

        is_gui_embedding = fig_to_plot_on is not None
        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        _, _, unit = self._get_pixel_pitch()
        axes_to_plot = ("x", "y") if axis == "both" else (axis,)
        for name in axes_to_plot:
            coords, values = self.cross_section(axis=name)
            if log:
                values = replace_nonpositive(values)
            ax.plot(be.to_numpy(coords), be.to_numpy(values), label=name)

        if log:
            ax.set_yscale("log")
        position = "Position" if axis == "both" else axis.upper()
        ax.set_xlabel(f"{position} ({unit})")
        ax.set_ylabel("Relative Intensity (%)")
        psf_name = self.__class__.__name__.replace("PSF", " PSF")
        ax.set_title(f"{psf_name} Cross-Section")
        if axis == "both":
            ax.legend()
        ax.grid(True)
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax

    def _get_pixel_pitch(self) -> tuple[float, float, str]:
        """Returns the pixel pitch of the PSF array along x and y.

        Uses `_get_psf_units` on the full PSF array (pitch = extent / shape)
        if the subclass implements it, otherwise falls back to pixel units,
        as `view` does.

        Returns:
            tuple: The pitch along x, the pitch along y, and the unit label
            ('µm' or 'pixels').
        """
        if hasattr(self, "_get_psf_units"):
            x_extent, y_extent = self._get_psf_units(self.psf)
            dx = float(x_extent) / self.psf.shape[1]
            dy = float(y_extent) / self.psf.shape[0]
            return dx, dy, "µm"
        return 1.0, 1.0, "pixels"

    def _cross_section_coordinates(
        self, axis: Literal["x", "y"], dx: float, dy: float
    ) -> BEArray:
        """Returns pixel centre positions along an axis of the PSF array.

        Pixel ``n // 2`` is at position 0, which is the centre of FFT-based
        PSF arrays (as placed by ``fftshift``).

        Args:
            axis (str): The axis, 'x' or 'y'.
            dx (float): The pixel pitch along x.
            dy (float): The pixel pitch along y.

        Returns:
            BEArray: The pixel centre positions along `axis`.
        """
        if axis == "x":
            num, pitch = self.psf.shape[1], dx
        else:
            num, pitch = self.psf.shape[0], dy
        center = num // 2
        return be.linspace(-center * pitch, (num - 1 - center) * pitch, num)

    def _get_working_FNO(self):
        """Calculates the working F-number of the optical system for the
        single defined field point and given wavelength.

        Algorithm:
            1. Retrieve the defined given wavelength and field coordinates.
            2. Determine the image-space refractive index 'n' at the given wavelength.
            3. Trace four marginal rays (top, bottom, left, right) at the pupil edges,
               as well as the chief ray.
            4. Compute the angle between each marginal ray and the chief ray.
            4. Calculate the average of the squared numerical apertures from all traced
               marginal rays.
            5. Compute the working F-number as 1 / (2 * sqrt(average_NA_squared)).
            6. Cap the calculated F/# at 10,000 if it exceeds this value.

        Returns:
            float: The working F-number.
        """
        return get_working_FNO(
            self.optic, self.fields[0].coord, self.wavelengths[0].value
        )
