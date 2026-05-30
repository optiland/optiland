"""Irradiance Analysis

This module implements the necessary logic for the
irradiance analysis in a given optical system.
*note*: for now we consider incoherent irradiance.
*note*: for now we consider incoherent irradiance.

The analysis is analogous to the SpotDiagram except that
instead of plotting the landing position of individual rays,
we accumulate their power on a detector and express the result
in W/mm^2.

Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as _np  # Use _np for plotting logic.

import optiland.backend as be
from optiland.rays import RealRays

from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class IncoherentIrradiance(BaseAnalysis):
    """Compute and visualise incoherent irradiance on the detector surface.
    For simplification, we assume that the detector surface = image surface.

     Attributes:
     ---
     optic : optiland.optic.Optic
         Reference to the optical system - must already define fields, wavelengths
         and, critically, a physical aperture on the chosen detector surface.
     res : tuple[int, int]
         Requested pixel count along (x,y) of the irradiance grid.
     px_size : tuple[float, float] | None
         Physical pixel pitch (dx,dy) in mm.  If ``None`` the pitch is
         derived from the surface aperture and `res`.
     num_rays : int
         Number of real rays launched for every (field,wavelength) pair.
     fields, wavelengths : tuple | "all"
         Convenience selectors that work exactly like those in
         `SpotDiagram` - default is to analyse all of them.
     detector_surface : int
         Index into `optic.surfaces` that designates the detector
         plane to analyse (default=`-1`->image surface).
     data : list[list[be.ndarray]]
         2-D irradiance arrays for every (field,wvl) - outer index is field,
         inner index is wavelength.  Each array has shape
         (res[0],res[1]) with X as the row index so that
         ``irr_data[f][w][i,j]`` refers to X=i, Y=j.
     user_initial_rays : RealRays | None
         Optional user-provided initial rays (at the source/object plane)
         to be traced through the whole optical system.
     source : BaseSource | None
         Optional extended source object (e.g., GaussianSource) to generate
         initial rays automatically. Cannot be used with user_initial_rays.
         When provided, num_rays determines how many rays to generate.

     Methods
     ---
     view(figsize=(6,5), cmap="inferno") → None
         Display false-colour irradiance maps three fields per row, sharing a common
         colour bar.
     peak_irradiance() → list[list[float]]
         Return the maximum pixel value for every (field,wvl) pair.
    """

    def __init__(
        self,
        optic,
        num_rays: int = 5,
        res=(128, 128),
        px_size: float | None = None,
        detector_surface: int = -1,
        *,
        fields="all",
        wavelengths="all",
        distribution: str = "random",
        user_initial_rays=None,
        source=None,
        skip_trace: bool = False,
    ):
        if fields == "all":
            self.fields = optic.fields.get_field_coords()
        else:
            self.fields = tuple(fields)

        # Handle source integration
        if source is not None and user_initial_rays is not None:
            raise ValueError("Cannot specify both 'source' and 'user_initial_rays'.")

        if source is not None:
            # Generate rays from the extended source
            self.user_initial_rays = source.generate_rays(num_rays)
            # When using a source, we treat all rays as a single "field"
            # The source emission defines the field, not optic.fields
            self.fields = [(0.0, 0.0)]  # Single dummy field for source rays
        else:
            self.user_initial_rays = user_initial_rays

        self.num_rays = num_rays
        self.npix_x, self.npix_y = res
        self.px_size = (
            None if px_size is None else (float(px_size[0]), float(px_size[1]))
        )
        self.detector_surface = int(detector_surface)
        # self.user_initial_rays = user_initial_rays
        self._initial_ray_data = None
        if self.user_initial_rays is not None:
            if not isinstance(self.user_initial_rays, RealRays):
                raise TypeError("user_initial_rays must be a RealRays object.")

            self._initial_ray_data = {
                "x": self.user_initial_rays.x,
                "y": self.user_initial_rays.y,
                "z": self.user_initial_rays.z,
                "L": self.user_initial_rays.L,
                "M": self.user_initial_rays.M,
                "N": self.user_initial_rays.N,
                "intensity": self.user_initial_rays.i,
                "wavelength": self.user_initial_rays.w,
            }
        self.distribution = distribution
        self.skip_trace = skip_trace

        # The detector surface must have a physical aperture
        surf = optic.surfaces[self.detector_surface]
        if surf.aperture is None:
            raise ValueError(
                "Detector surface has no physical aperture - set one "
                "(e.g. RectangularAperture) so that the detector size is defined."
            )

        # Override resolution if px_size is provided
        if self.px_size is not None:
            x_min, x_max, y_min, y_max = surf.aperture.extent
            detector_width = x_max - x_min
            detector_height = y_max - y_min

            # Calculate resolution from pixel size
            new_npix_x = int(round(detector_width / self.px_size[0]))
            new_npix_y = int(round(detector_height / self.px_size[1]))

            # Print warning and update resolution
            print(
                "[IncoherentIrradiance] Warning: res parameter ignored - derived "
                f"from px_size instead → ({new_npix_x},{new_npix_y}) pixels"
            )
            self.npix_x, self.npix_y = new_npix_x, new_npix_y

        super().__init__(optic, wavelengths)

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple = (6, 5),
        cmap: str | Colormap = "inferno",
        cross_section: tuple[str, int] | None = None,
        *,
        normalize: bool = True,
        show: bool = True,
    ) -> tuple[Figure, NDArray[_np.object_]] | None:
        """
        Display a false-colour irradiance map or cross-section plots for the current
        irradiance data.

        Args:
            fig_to_plot_on : plt.Figure, optional
                Existing matplotlib Figure to plot on. If None, a new figure is created.
                Default is None.
            figsize : tuple, optional
                Size of each subplot as (width, height) in inches. Default is (6, 5).
            cmap : str or Colormap, optional
                Colormap to use for the irradiance map. Default is "inferno".
            cross_section : tuple[str, int], optional
                If provided, plot a cross-section instead of a 2D map. Should be a tuple
                of ('cross-x' or 'cross-y', index), where index is the slice index along
                the specified axis.
                If None, a 2D irradiance map is plotted. Default is None.
            normalize : bool, optional
                If True, normalize irradiance maps to their peak value. If False, use
                absolute values.
                Default is True.

        Returns :
            fig : matplotlib.figure.Figure
                The matplotlib Figure object containing the plot(s).
            axs : numpy.ndarray
                Array of Axes objects for the subplots, or None if plotting on an
                existing figure.

        Notes
        -----
        - If no valid irradiance data is available, the method prints a warning
        and returns None.
        - If `cross_section` is invalid or not provided, a 2D irradiance map is
        shown.
        - The method supports plotting multiple fields and wavelengths as a grid
        of subplots.
        - Colorbars and axis labels are automatically added to each subplot.
        """
        if not self.data:
            print("No irradiance data to display.")
            return None

        cs_info = self._validate_cross_section_request(cross_section)
        vmin_plot, vmax_plot = self._calculate_plot_limits(normalize)
        fig, axs = self._setup_figure(fig_to_plot_on, figsize)

        for f_idx, field_block in enumerate(self.data):
            for w_idx, entry_data in enumerate(field_block):
                if not entry_data or entry_data[0] is None:
                    print(
                        f"Warning: No valid data for field {f_idx}, "
                        f"wavelength {w_idx}. Skipping."
                    )
                    continue

                self._plot_single_subplot(
                    axs[f_idx, w_idx],
                    fig,
                    entry_data,
                    f_idx,
                    w_idx,
                    normalize,
                    vmin_plot,
                    vmax_plot,
                    cmap,
                    cs_info,
                )

        self._finalize_figure(fig, cs_info, normalize)
        is_gui_embedding = fig_to_plot_on is not None
        if show and not is_gui_embedding:
            plt.show()
        return fig, axs

    # --- Data Generation and Access ---

    def peak_irradiance(self):
        """Maximum pixel value for each (field,wvl) pair."""
        return [[be.max(irr) for irr, *_ in fblock] for fblock in self.data]

    def _generate_data(self):
        """Generates irradiance data for all fields and wavelengths."""
        data = []
        for field in self.fields:
            f_block = []
            for wp in self.wavelengths:
                f_block.append(
                    self._generate_field_data(
                        field, wp.value, self.distribution, self.user_initial_rays
                    )
                )
            data.append(f_block)
        return data

    def _generate_field_data(self, field, wavelength, distribution, user_initial_rays):
        """
        Traces rays and bins their power. Switches between standard and
        differentiable methods based on the gradient mode.
        """
        rays_traced = None
        if not self.skip_trace:
            if user_initial_rays is None:
                Hx, Hy = field
                rays_traced = self.optic.trace(
                    Hx, Hy, wavelength, self.num_rays, distribution
                )
            else:
                rays_to_trace = RealRays(**self._initial_ray_data)
                self.optic.surfaces.trace(rays_to_trace)
                rays_traced = rays_to_trace

        surf = self.optic.surfaces[self.detector_surface]
        if rays_traced is not None:
            x_g, y_g, z_g, power = (
                rays_traced.x,
                rays_traced.y,
                rays_traced.z,
                rays_traced.i,
            )
        else:
            # Read from cache (assumes trace was done externally or skipping)
            x_g, y_g, z_g, power = surf.x, surf.y, surf.z, surf.intensity

        from optiland.visualization.system.utils import transform

        x_local, y_local, _ = transform(x_g, y_g, z_g, surf, is_global=True)

        x_min, x_max, y_min, y_max = surf.aperture.extent
        if self.px_size is None:
            x_edges = _np.linspace(x_min, x_max, self.npix_x + 1, dtype=float)
            y_edges = _np.linspace(y_min, y_max, self.npix_y + 1, dtype=float)
            pixel_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
        else:
            dx, dy = self.px_size
            x_edges = _np.arange(x_min, x_max + 0.5 * dx, dx, dtype=float)
            y_edges = _np.arange(y_min, y_max + 0.5 * dy, dy, dtype=float)
            pixel_area = dx * dy
            exp_nx, exp_ny = len(x_edges) - 1, len(y_edges) - 1
            if (exp_nx, exp_ny) != (self.npix_x, self.npix_y):
                print(
                    f"[IncoherentIrradiance] Warning: res parameter ignored - "
                    f"derived from px_size instead → ({exp_nx},{exp_ny}) pixels"
                )
                self.npix_x, self.npix_y = exp_nx, exp_ny

        # differentiable path
        if be.get_backend() == "torch" and be.grad_mode.requires_grad:
            x_edges_be = be.array(x_edges)
            y_edges_be = be.array(y_edges)
            ray_coords = be.stack([x_local, y_local], axis=1)

            if ray_coords.shape[0] == 0:
                irr = be.zeros((self.npix_x, self.npix_y))
                return irr, x_edges, y_edges

            indices, weights = be.get_bilinear_weights(
                ray_coords, (x_edges_be, y_edges_be)
            )
            power_map = be.zeros((self.npix_y, self.npix_x))
            for i in range(4):
                power_map = power_map.index_put(
                    (indices[:, i, 1].long(), indices[:, i, 0].long()),
                    weights[:, i] * power,
                    accumulate=True,
                )
            irr = power_map / pixel_area
            return irr, x_edges, y_edges
        # non-differentiable path
        else:
            x_np, y_np, power_np = (
                be.to_numpy(x_local),
                be.to_numpy(y_local),
                be.to_numpy(power),
            )

            valid = power_np > 0.0
            x_np, y_np, power_np = x_np[valid], y_np[valid], power_np[valid]

            hist, _, _ = _np.histogram2d(
                x_np, y_np, bins=[x_edges, y_edges], weights=power_np
            )
            irr = hist / pixel_area
            return be.array(irr), x_edges, y_edges

    # --- Plotting Helper Functions ---

    def _validate_cross_section_request(self, cross_section):
        """
        Validates the cross_section parameter from the view method.

        Args:
            cross_section (any): The user-provided cross_section parameter.

        Returns:
            tuple[bool, str, int]: A tuple containing:
                - A boolean indicating if a valid cross-section plot is requested.
                - The axis type ('cross-x' or 'cross-y') or None.
                - The slice index or -1.
        """
        if cross_section is None:
            return False, None, -1

        if not (isinstance(cross_section, tuple) and len(cross_section) == 2):
            print(
                "[IncoherentIrradiance] Warning: Invalid cross_section type. "
                "Expected tuple. Defaulting to 2D plot."
            )
            return False, None, -1

        axis_type_in, slice_idx_in = cross_section
        if (
            isinstance(axis_type_in, str)
            and axis_type_in.lower() in ["cross-x", "cross-y"]
            and (isinstance(slice_idx_in, int) or slice_idx_in is None)
        ):
            cs_axis_type = axis_type_in.lower()
            cs_slice_idx = slice_idx_in if slice_idx_in is not None else -1
            return True, cs_axis_type, cs_slice_idx
        else:
            print(
                "[IncoherentIrradiance] Warning: Invalid cross_section format. "
                "Expected ('cross-x' or 'cross-y', int). Defaulting to 2D plot."
            )
            return False, None, -1

    def _calculate_plot_limits(self, normalize):
        """
        Calculates the minimum and maximum values for plotting.

        Args:
            normalize (bool): If True, returns a normalized range [0, 1].
                Otherwise, computes the min and max from all irradiance data.

        Returns:
            tuple[float, float]: The minimum and maximum plot values (vmin, vmax).
        """
        if normalize:
            return 0.0, 1.0

        all_irr_values = _np.concatenate(
            [
                be.to_numpy(entry[0]).flatten()
                for field_block in self.data
                if field_block
                for entry in field_block
                if entry and entry[0] is not None
            ]
        )

        if all_irr_values.size == 0:
            print("No valid irradiance values found to determine plot limits.")
            return 0.0, 1.0

        vmin_plot, vmax_plot = _np.min(all_irr_values), _np.max(all_irr_values)
        if vmin_plot == vmax_plot:
            offset = 0.1 * abs(vmax_plot) if vmax_plot != 0 else 0.1
            vmin_plot -= offset
            vmax_plot += offset
        return vmin_plot, vmax_plot

    def _setup_figure(self, fig_to_plot_on, figsize):
        """
        Initializes the matplotlib figure and axes for plotting.

        Args:
            fig_to_plot_on (Figure | None): An existing figure to draw on.
            figsize (tuple[float, float]): The size for each subplot.

        Returns:
            tuple[Figure, _np.ndarray[Axes]]: The figure and axes array.
        """
        n_fields = len(self.fields)
        n_wavelengths = len(self.wavelengths)
        total_figsize = (figsize[0] * n_wavelengths, figsize[1] * n_fields)

        if fig_to_plot_on:
            fig = fig_to_plot_on
            fig.clear()
            fig.set_size_inches(total_figsize)
            axs = fig.subplots(
                nrows=n_fields,
                ncols=n_wavelengths,
                sharex=True,
                sharey=True,
                squeeze=False,
            )
        else:
            fig, axs = plt.subplots(
                nrows=n_fields,
                ncols=n_wavelengths,
                figsize=total_figsize,
                sharex=True,
                sharey=True,
                squeeze=False,
                tight_layout=True,
            )
        return fig, axs

    def _generate_subplot_title(self, f_idx, w_idx):
        """
        Generates the title for an individual subplot.

        Args:
            f_idx (int): The field index.
            w_idx (int): The wavelength index.

        Returns:
            str: The formatted title string for the subplot.
        """
        if self.user_initial_rays is not None:
            field_label = (
                self.fields[f_idx][0]
                if isinstance(self.fields[f_idx], tuple)
                else self.fields[f_idx]
            )
            return f"(User Rays: {field_label})"
        else:
            field_coord = self.fields[f_idx]
            wavelength_val = self.wavelengths[w_idx].value
            text = (
                f"Field {f_idx} {field_coord}, "
                f"$\\lambda_{w_idx}$ = {wavelength_val:.3f} µm"
            )
            return text

    def _plot_single_subplot(
        self,
        ax: Axes,
        fig: Figure,
        entry_data: tuple,
        f_idx: int,
        w_idx: int,
        normalize: bool,
        vmin: float,
        vmax: float,
        cmap: Colormap,
        cs_info: tuple,
    ):
        """
        Plots the data for a single subplot, either as a 2D map or a cross-section.

        Args:
            ax (Axes): The matplotlib axes to plot on.
            fig (Figure): The parent figure, for colorbar placement.
            entry_data (tuple): Tuple of (irr_map, x_edges, y_edges).
            f_idx (int): The field index.
            w_idx (int): The wavelength index.
            normalize (bool): Flag to normalize the data.
            vmin (float): Minimum value for the color scale.
            vmax (float): Maximum value for the color scale.
            cmap (str | Colormap): The colormap to use.
            cs_info (tuple): Tuple from _validate_cross_section_request.
        """
        irr_map, x_edges, y_edges = entry_data
        is_cs_plot, cs_axis, cs_idx = cs_info
        title = self._generate_subplot_title(f_idx, w_idx)

        if is_cs_plot:
            self._plot_cross_section(
                ax, irr_map, x_edges, y_edges, cs_axis, cs_idx, title, normalize
            )
        else:
            plot_map = be.to_numpy(irr_map)
            if normalize:
                peak_val = self.peak_irradiance()[f_idx][w_idx]
                if peak_val > 0:
                    plot_map = plot_map / peak_val

            im = ax.imshow(
                plot_map.T,
                aspect="auto",
                origin="lower",
                extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            cbar_lbl = "Normalized Irradiance" if normalize else "Irradiance (W/mm$^2$)"
            fig.colorbar(im, ax=ax, label=cbar_lbl)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_title(title)
            ax.set_aspect("equal")

    def _finalize_figure(self, fig, cs_info, normalize):
        """
        Adds the final touches to the figure, like the main title.

        Args:
            fig (plt.Figure): The figure to finalize.
            cs_info (tuple): The cross-section information tuple.
            normalize (bool): The normalization flag.
        """
        is_cs_plot, cs_axis, cs_idx = cs_info
        main_title = "Irradiance Analysis"
        if is_cs_plot:
            main_title += self._get_cross_section_title(cs_axis, cs_idx, normalize)
        fig.suptitle(main_title)

        if hasattr(fig, "canvas"):
            fig.canvas.draw_idle()

    def _plot_cross_section(
        self,
        ax: Axes,
        irr_map_be,
        x_edges,
        y_edges,
        axis_type: str,
        slice_idx: int,
        title_prefix: str,
        normalize: bool = True,
    ) -> None:
        """Helper method to plot a 1D cross-section of the irradiance map."""
        irr_map_np = be.to_numpy(irr_map_be)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        if axis_type == "cross-x":
            if slice_idx < 0:
                slice_idx = irr_map_np.shape[0] // 2
            if not (0 <= slice_idx < irr_map_np.shape[0]):
                print(
                    f"[IncoherentIrradiance] Warning: X-slice index {slice_idx} is out "
                    f"of bounds. Skipping."
                )
                return
            data, coords, xlabel = irr_map_np[slice_idx, :], y_centers, "Y (mm)"
        elif axis_type == "cross-y":
            if slice_idx < 0:
                slice_idx = irr_map_np.shape[1] // 2
            if not (0 <= slice_idx < irr_map_np.shape[1]):
                print(
                    f"[IncoherentIrradiance] Warning: Y-slice index {slice_idx} is out "
                    f"of bounds. Skipping."
                )
                return
            data, coords, xlabel = irr_map_np[:, slice_idx], x_centers, "X (mm)"
        else:
            return

        if normalize and data.max() > 0:
            data = data / data.max()

        ax.plot(coords, data, linestyle="-")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized Irradiance" if normalize else "Irradiance (W/mm$^2$)")
        ax.set_title(title_prefix)
        ax.grid(True)

    def _get_cross_section_title(
        self,
        axis_type: str,
        slice_idx: int,
        normalize: bool = True,
    ) -> str:
        """
        Generate a descriptive title for a cross-section plot of an irradiance map.

        Args:
            axis_type (str): The type of cross-section to plot, 'cross-x' or 'cross-y'.
            slice_idx (int): The index of the slice to use. If negative, the
                middle is used.
            normalize (bool): Whether to indicate normalization in the title.
                Default is True.

        Returns:
            str: A formatted string suitable as a plot title.
        """
        if not self.data or not self.data[0] or not self.data[0][0]:
            return ""

        irr_map_be, x_edges, y_edges = self.data[0][0]
        irr_map_np = be.to_numpy(irr_map_be)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        title = ""

        if axis_type == "cross-x":
            if slice_idx < 0:
                slice_idx = irr_map_np.shape[0] // 2
            if 0 <= slice_idx < irr_map_np.shape[0]:
                title = (
                    f" - $X$-Cross-section at $X$ ≈ {x_centers[slice_idx]:.2f} mm"
                    f" (index {slice_idx}/{irr_map_np.shape[0]})"
                )
        elif axis_type == "cross-y":
            if slice_idx < 0:
                slice_idx = irr_map_np.shape[1] // 2
            if 0 <= slice_idx < irr_map_np.shape[1]:
                title = (
                    f" - $Y$-Cross-section at $Y$ ≈ {y_centers[slice_idx]:.2f} mm"
                    f" (index {slice_idx}/{irr_map_np.shape[1]})"
                )

        if title and normalize:
            title += " (normalized)"
        return title
