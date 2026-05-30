"""Radiant Intensity Analysis

This module implements the logic for radiant intensity analysis
in an optical system, representing power per unit solid angle.


Manuel Fragata Mendes, June 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import optiland.backend as be
from optiland.analysis.base import BaseAnalysis

if TYPE_CHECKING:
    from optiland._types import BEArray, DistributionType, ScalarOrArray
    from optiland.optic import Optic


class RadiantIntensity(BaseAnalysis):
    """
    Computes and visualizes the radiant intensity distribution.

    By default, this analysis calculates radiant intensity in absolute physical
    units of Watts/steradian (W/sr). This requires that the .intensity
    attribute of the rays traced represents the physical power of each ray.

    The angles correspond to projections, similar to Zemax's
    "Angle Space" plots (Angle X, Angle Y).

    Attributes:
        optic (optiland.optic.Optic): The optical system.
        num_angular_bins_X (int): Number of bins for the X-angle.
        num_angular_bins_Y (int): Number of bins for the Y-angle.
        angle_X_min (float): Minimum X-angle in degrees for binning.
        angle_X_max (float): Maximum X-angle in degrees for binning.
        angle_Y_min (float): Minimum Y-angle in degrees for binning.
        angle_Y_max (float): Maximum Y-angle in degrees for binning.
        reference_surface_index (int): Index of the surface *after* which ray
                                       directions are considered.
        fields (list): List of field coordinates for analysis.
        wavelengths (list): List of wavelengths for analysis.
        num_rays (int): Number of rays to trace if user_initial_rays is None.
        distribution_name (str): Ray distribution if user_initial_rays is None.
        user_initial_rays (RealRays | None): Optional user-provided initial rays.
        source (BaseSource | None): Optional extended source object
            (e.g., GaussianSource) to generate initial rays automatically.
            Cannot be used with user_initial_rays. When provided, num_rays
            determines how many rays to generate.
        data (list[list[tuple]]): Stores (intensity_map,
                                          angle_X_bin_edges, angle_Y_bin_edges,
                                          angle_X_bin_centers, angle_Y_bin_centers)
                                  for each (field, wavelength).
        use_absolute_units (bool): If True (default), calculates intensity in W/sr.
                                   If False, result is a relative value normalized
                                   to the peak.
    """

    def __init__(
        self,
        optic: Optic,
        num_angular_bins_X: int = 101,
        num_angular_bins_Y: int = 101,
        angle_X_min: float = -15.0,
        angle_X_max: float = 15.0,
        angle_Y_min: float = -15.0,
        angle_Y_max: float = 15.0,
        use_absolute_units: bool = True,
        reference_surface_index: int = -1,
        fields="all",
        wavelengths="all",
        num_rays: int = 100000,
        distribution: DistributionType = "random",
        user_initial_rays=None,
        source=None,
        skip_trace: bool = False,
    ):
        if fields == "all":
            self.fields = optic.fields.get_field_coords()
        else:
            if not isinstance(fields, list):
                fields = [fields]
            self.fields = tuple(fields)

        # Handle source integration
        if source is not None and user_initial_rays is not None:
            raise ValueError("Cannot specify both 'source' and 'user_initial_rays'.")

        self.user_initial_rays = user_initial_rays
        if source is not None:
            # Generate rays from the extended source
            self.user_initial_rays = source.generate_rays(num_rays)
            # When using a source, we treat all rays as a single "field"
            # The source emission defines the field, not optic.fields
            self.fields = [(0.0, 0.0)]  # Single dummy field for source rays

        self._initial_ray_data = None
        if self.user_initial_rays is not None:
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

        self.num_angular_bins_X = num_angular_bins_X
        self.num_angular_bins_Y = num_angular_bins_Y
        self.angle_X_min, self.angle_X_max = float(angle_X_min), float(angle_X_max)
        self.angle_Y_min, self.angle_Y_max = float(angle_Y_min), float(angle_Y_max)

        # for absolute units, we need to ensure the user has provided rays
        # with 'calibrated' power
        self.use_absolute_units = use_absolute_units
        if self.use_absolute_units and self.user_initial_rays is None:
            print(
                "Warning: `use_absolute_units` is True, but no `user_initial_rays` "
                "were provided."
            )
            print(
                "         Internal ray generator may not have 'calibrated' "
                "power values."
            )
            print("         Resulting intensity map may not be in true W/sr.")

        self.reference_surface_index = int(reference_surface_index)
        self.num_rays = num_rays
        self.distribution_name: DistributionType = distribution
        self.skip_trace = skip_trace

        super().__init__(optic, wavelengths)

    def _generate_data(self):
        analysis_data = []
        for field_coord in self.fields:
            field_block = []
            for wp in self.wavelengths:
                field_block.append(
                    self._generate_field_wavelength_data(field_coord, wp.value)
                )
            analysis_data.append(field_block)
        return analysis_data

    def _generate_field_wavelength_data(
        self, field_coord: tuple[ScalarOrArray, ScalarOrArray], wavelength: float
    ) -> tuple[BEArray, BEArray, BEArray, BEArray, BEArray]:
        if not self.skip_trace:
            if self.user_initial_rays is None:
                self.optic.trace(
                    *field_coord,
                    wavelength=wavelength,
                    num_rays=self.num_rays,
                    distribution=self.distribution_name,
                )
            else:
                from optiland.rays import RealRays

                rays_to_trace = RealRays(**self._initial_ray_data)
                self.optic.surfaces.trace(rays_to_trace)

        surf_group = self.optic.surfaces
        try:
            ref_surf = surf_group.surfaces[self.reference_surface_index]
            L_all, M_all, N_all = ref_surf.L, ref_surf.M, ref_surf.N
            power_all = ref_surf.intensity
            if not (be.size(L_all) > 0):
                raise AttributeError
        except (IndexError, AttributeError):
            L_all, M_all, N_all, power_all = (be.empty(0) for _ in range(4))

        valid_mask = (
            (power_all > 1e-12)
            & ~be.isnan(L_all)
            & ~be.isnan(M_all)
            & ~be.isnan(N_all)
            & (be.abs(N_all) > 1e-9)
        )

        angle_X_bins = be.linspace(
            self.angle_X_min, self.angle_X_max, self.num_angular_bins_X + 1
        )
        angle_Y_bins = be.linspace(
            self.angle_Y_min, self.angle_Y_max, self.num_angular_bins_Y + 1
        )
        angle_X_centers = (angle_X_bins[:-1] + angle_X_bins[1:]) / 2
        angle_Y_centers = (angle_Y_bins[:-1] + angle_Y_bins[1:]) / 2

        if not be.any(valid_mask):
            power_map = be.zeros((self.num_angular_bins_Y, self.num_angular_bins_X))
        else:
            L_f, M_f, N_f, power_f = (
                arr[valid_mask] for arr in [L_all, M_all, N_all, power_all]
            )

            angle_X_deg = be.degrees(be.arctan2(L_f, N_f))
            angle_Y_deg = be.degrees(be.arctan2(M_f, N_f))

            if be.get_backend() == "torch" and be.grad_mode.requires_grad:
                ray_coords = be.stack([angle_X_deg, angle_Y_deg], axis=1)

                if ray_coords.shape[0] == 0:
                    power_map = be.zeros(
                        (self.num_angular_bins_Y, self.num_angular_bins_X)
                    )
                else:
                    # call the bilinear weights function, idea from the
                    # paper in its docstring
                    indices, weights = be.get_bilinear_weights(
                        ray_coords, (angle_X_bins, angle_Y_bins)
                    )
                    power_map = be.zeros(
                        (self.num_angular_bins_Y, self.num_angular_bins_X)
                    )
                    for i in range(4):
                        power_map = power_map.index_put(
                            (indices[:, i, 1].long(), indices[:, i, 0].long()),
                            weights[:, i] * power_f,
                            accumulate=True,
                        )
            else:
                # Use histogram2d to bin the angles, faster using torch and GPU
                power_map, _, _ = be.histogram2d(
                    angle_X_deg,
                    angle_Y_deg,
                    bins=[angle_X_bins, angle_Y_bins],
                    weights=power_f,
                )

        if self.use_absolute_units:
            # 1. Calculate basic bin sizes (radians)
            dx = be.radians(angle_X_bins[1] - angle_X_bins[0])
            dy = be.radians(angle_Y_bins[1] - angle_Y_bins[0])

            # 2. Create meshgrid of bin centers (in Radians) for
            # the Jacobian calculation
            # Note: We must be careful with tensor shapes here to match
            # power_map (Y, X)
            ax_c_rad = be.radians(angle_X_centers)
            ay_c_rad = be.radians(angle_Y_centers)

            # Create grids. Meshgrid usually returns (Y, X) with
            # indexing='ij' or 'xy' depending on backend
            # For safety, let's explicitely broadcast
            # resulting shape (Y, X) usually
            AX, AY = be.meshgrid(ax_c_rad, ay_c_rad)

            # 3. Compute Jacobian terms
            # J = (sec^2(tx) * sec^2(ty)) / (1 + tan^2(tx) + tan^2(ty))^(3/2)
            tan2_tx = be.tan(AX) ** 2
            tan2_ty = be.tan(AY) ** 2
            sec2_tx = 1.0 + tan2_tx  # Identity: sec^2 = 1 + tan^2
            sec2_ty = 1.0 + tan2_ty

            numerator = sec2_tx * sec2_ty
            denominator = (1.0 + tan2_tx + tan2_ty) ** 1.5

            jacobian_factor = numerator / denominator

            # 4. Compute true solid angle per bin
            # d_omega = J * d_theta_x * d_theta_y
            true_solid_angle_map = jacobian_factor * dx * dy

            # 5. Normalize Power Map
            final_intensity_map = be.where(
                true_solid_angle_map > 1e-12,
                power_map / true_solid_angle_map,
                be.zeros_like(power_map),
            )
        else:
            final_intensity_map = power_map

        return (
            final_intensity_map,
            angle_X_bins,
            angle_Y_bins,
            angle_X_centers,
            angle_Y_centers,
        )

    def peak_intensity_values(self):
        peaks = []
        if not self.data:
            return peaks
        for field_block in self.data:
            field_peaks = [
                be.max(entry[0]) if be.to_numpy(entry[0]).size > 0 else 0.0
                for entry in field_block
            ]
            peaks.append(field_peaks)
        return peaks

    def _plot_cross_section(
        self,
        ax,
        intensity_map,
        x_centers,
        y_centers,
        axis_type,
        slice_idx,
        title,
        style="-",
        color="red",
        ylabel="Intensity",
    ):
        """
        Helper method to plot a cross-section of the intensity map.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        intensity_map : numpy.ndarray
            The 2D intensity map.
        x_centers, y_centers : numpy.ndarray
            Center coordinates of the bins.
        axis_type : str
            Either 'cross-x' or 'cross-y' to specify direction.
        slice_idx : int
            Index along the non-plotted axis where to take the slice.
            If negative, will use the central index.
        title : str
            Title for the plot.
        style : str, optional
            Line style for the plot.
        color : str, optional
            Line color for the plot.
        ylabel : str, optional
            Label for the Y-axis.
        """
        if intensity_map.size == 0:
            ax.set_title(f"{title}\n(No valid data)")
            return

        if axis_type == "cross-x":
            # For cross-x, we take a horizontal slice (constant y)
            if slice_idx < 0 or slice_idx >= intensity_map.shape[1]:
                slice_idx = intensity_map.shape[1] // 2  # Central Y index
            data_to_plot = intensity_map[:, slice_idx]
            coords_to_plot_against = x_centers
            xlabel = "X-Angle (degrees)"
            slice_pos = y_centers[slice_idx]
            subtitle = f"Y-Angle = {slice_pos:.2f}°"
        elif axis_type == "cross-y":
            # For cross-y, we take a vertical slice (constant x)
            if slice_idx < 0 or slice_idx >= intensity_map.shape[0]:
                slice_idx = intensity_map.shape[0] // 2  # Central X index
            data_to_plot = intensity_map[slice_idx, :]
            coords_to_plot_against = y_centers
            xlabel = "Y-Angle (degrees)"
            slice_pos = x_centers[slice_idx]
            subtitle = f"X-Angle = {slice_pos:.2f}°"
        else:
            # Default to central horizontal cross-section
            slice_idx = intensity_map.shape[1] // 2
            data_to_plot = intensity_map[:, slice_idx]
            coords_to_plot_against = x_centers
            xlabel = "X-Angle (degrees)"
            subtitle = "Central Cross-Section"

        ax.plot(coords_to_plot_against, data_to_plot, linestyle=style, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{subtitle}")
        ax.grid(True, linestyle=":", alpha=0.7)

    def _get_cross_section_title(
        self,
        axis_type: str,
        slice_idx: int,
        normalize: bool = True,
    ) -> str:
        """
        Generate a descriptive title for cross-section plots.

        Parameters
        ----------
        axis_type : str
            Either 'cross-x' or 'cross-y' to specify direction.
        slice_idx : int
            Index along the non-plotted axis where to take the slice.
        normalize : bool, optional
            Whether to indicate normalization in the title.

        Returns
        -------
        str
            A formatted string for use in plot titles.
        """
        cross_section_title = ""

        if not self.data or not self.data[0] or not self.data[0][0]:
            return cross_section_title

        _, _, _, x_centers_be, y_centers_be = self.data[0][0]
        x_centers = be.to_numpy(x_centers_be)
        y_centers = be.to_numpy(y_centers_be)

        if axis_type == "cross-x":
            if slice_idx < 0:
                slice_idx = len(y_centers) // 2
            if not (0 <= slice_idx < len(y_centers)):
                return cross_section_title

            cross_section_title += (
                f" - X-Angle Cross-section at Y-Angle ≈ {y_centers[slice_idx]:.2f}°"
            )
            cross_section_title += f" (index {slice_idx}/{len(y_centers)})"

        elif axis_type == "cross-y":
            if slice_idx < 0:
                slice_idx = len(x_centers) // 2
            if not (0 <= slice_idx < len(x_centers)):
                return cross_section_title

            cross_section_title += (
                f" - Y-Angle Cross-section at X-Angle ≈ {x_centers[slice_idx]:.2f}°"
            )
            cross_section_title += f" (index {slice_idx}/{len(x_centers)})"

        if normalize:
            cross_section_title += " (normalized)"

        return cross_section_title

    def view(
        self,
        fig_to_plot_on=None,
        figsize=(8, 6),
        cmap="jet",
        cross_section=None,
        cross_section_style="-",
        cross_section_color="red",
        *,
        normalize=None,
        show: bool = True,
    ):
        """
        Display radiant intensity maps and/or cross-sections.

        Parameters
        ----------
        fig_to_plot_on : matplotlib.figure.Figure, optional
            Existing figure to plot on. If None, a new figure is created.
        figsize : tuple, optional
            Size of the figure (width, height) in inches for each subplot.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap to use for the intensity maps.
        cross_section : tuple[str, int], optional
            If provided, plot only cross-sections. Should be a tuple of
            ('cross-x' or 'cross-y', index), where index is the slice index
            along the specified axis. Default is None (plots 2D map + cross section).
        cross_section_style : str, optional
            Line style for cross-section plots.
        cross_section_color : str, optional
            Color for cross-section plots.
        normalize : bool, optional
            If True, normalize intensity to peak value.
            If False, use absolute values (W/sr).
            If None (default), use the value set in class initialization.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        axs : numpy.ndarray
            Array of Axes objects for the subplots, or single Axes if only one subplot.
        """
        import numpy as _np  # Local import for plotting

        is_gui_embedding = fig_to_plot_on is not None
        # Fix inverted normalization logic to match IncoherentIrradiance
        use_norm = not self.use_absolute_units if normalize is None else normalize

        if not self.data:
            print("No intensity data to display.")
            return None, None

        num_fields, num_wavelengths = len(self.fields), len(self.wavelengths)
        if num_fields == 0 or num_wavelengths == 0:
            return None, None

        # Process cross-section request
        plot_cross_section_requested = False
        valid_cross_section_request = False
        cs_axis_type = None
        cs_slice_idx = -1

        if cross_section is not None:
            if isinstance(cross_section, tuple) and len(cross_section) == 2:
                axis_type_in, slice_idx_in = cross_section
                if (
                    isinstance(axis_type_in, str)
                    and axis_type_in.lower() in ["cross-x", "cross-y"]
                    and (isinstance(slice_idx_in, int) or slice_idx_in is None)
                ):
                    plot_cross_section_requested = True
                    valid_cross_section_request = True
                    cs_axis_type = axis_type_in.lower()
                    cs_slice_idx = slice_idx_in if slice_idx_in is not None else -1
                    # Get cross-section title for main title
                    cross_section_title = self._get_cross_section_title(
                        cs_axis_type, cs_slice_idx, normalize=use_norm
                    )
                else:
                    print(
                        "[RadiantIntensity] Warning: Invalid cross_section format. "
                        "Expected ('cross-x' or 'cross-y', int). "
                        "Defaulting to 2D+cross plot."
                    )
            else:
                print(
                    "[RadiantIntensity] Warning: Invalid cross_section type. "
                    "Expected tuple. Defaulting to 2D+cross plot."
                )

        # Calculate global min/max for consistent colorbar
        all_peak_values = self.peak_intensity_values()
        global_max_val = 0.0

        if not use_norm:  # Using absolute units
            if all_peak_values:
                global_max_val = max(
                    max(be.to_numpy(p) for p in field_peaks)
                    for field_peaks in all_peak_values
                )
            if global_max_val == 0:
                global_max_val = 1.0
            vmin_plot, vmax_plot = 0.0, global_max_val
            cbar_label = "Radiant Intensity (W/sr)"
        else:  # Normalize to peak for relative plot
            vmin_plot, vmax_plot = 0.0, 1.0
            cbar_label = "Normalized Intensity"

        # Set up the figure and axes layout
        if is_gui_embedding:
            fig = fig_to_plot_on
            fig.clear()  # Clear the figure for new content
            if plot_cross_section_requested and valid_cross_section_request:
                axs = fig.subplots(
                    nrows=num_fields,
                    ncols=num_wavelengths,
                    squeeze=False,
                )
            else:
                # Use GridSpec for 2D map + cross section layout
                axs = _np.empty((num_fields, num_wavelengths), dtype=object)
                for f_idx in range(num_fields):
                    for w_idx in range(num_wavelengths):
                        gs = gridspec.GridSpec(
                            1, 2, width_ratios=[2.5, 1.5], figure=fig
                        )
                        axs[f_idx, w_idx] = [
                            fig.add_subplot(gs[0]),
                            fig.add_subplot(gs[1]),
                        ]
        else:
            if plot_cross_section_requested and valid_cross_section_request:
                fig, axs = plt.subplots(
                    nrows=num_fields,
                    ncols=num_wavelengths,
                    figsize=(figsize[0] * num_wavelengths, figsize[1] * num_fields),
                    squeeze=False,
                    tight_layout=True,
                )
            else:
                # For 2D map + cross section layout in a grid
                fig = plt.figure(
                    figsize=(figsize[0] * num_wavelengths, figsize[1] * num_fields)
                )
                axs = _np.empty((num_fields, num_wavelengths), dtype=object)
                for f_idx in range(num_fields):
                    for w_idx in range(num_wavelengths):
                        gs = gridspec.GridSpecFromSubplotSpec(
                            1,
                            2,
                            width_ratios=[2.5, 1.5],
                            subplot_spec=gridspec.GridSpec(
                                num_fields, num_wavelengths, figure=fig
                            )[f_idx, w_idx],
                        )
                        ax_map = fig.add_subplot(gs[0])
                        ax_cs = fig.add_subplot(gs[1])
                        axs[f_idx, w_idx] = [ax_map, ax_cs]

        # Set main title
        main_title = "Radiant Intensity Analysis"
        if plot_cross_section_requested and valid_cross_section_request:
            main_title += cross_section_title

        # Plot the data
        for f_idx in range(num_fields):
            for w_idx in range(num_wavelengths):
                intensity_map_be, x_bins_be, y_bins_be, x_centers_be, y_centers_be = (
                    self.data[f_idx][w_idx]
                )

                intensity_map = be.to_numpy(intensity_map_be)
                x_bins = be.to_numpy(x_bins_be)
                y_bins = be.to_numpy(y_bins_be)
                x_centers = be.to_numpy(x_centers_be)
                y_centers = be.to_numpy(y_centers_be)

                # Create display map with appropriate normalization
                current_display_map = intensity_map.copy()

                if use_norm:  # If we're normalizing (normalize=True)
                    peak_val = be.to_numpy(all_peak_values[f_idx][w_idx])
                    if peak_val > 1e-9:
                        current_display_map = intensity_map / peak_val

                # Get the current axes based on the plot type
                if plot_cross_section_requested and valid_cross_section_request:
                    ax = axs[f_idx, w_idx]
                    self._plot_cross_section(
                        ax=ax,
                        intensity_map=current_display_map,
                        x_centers=x_centers,
                        y_centers=y_centers,
                        axis_type=cs_axis_type,
                        slice_idx=cs_slice_idx,
                        title=f"Field: {self.fields[f_idx]}, "
                        f"λ={self.wavelengths[w_idx].value:.3f} µm",
                        style=cross_section_style,
                        color=cross_section_color,
                        ylabel=cbar_label,
                    )
                else:
                    # 2D map + cross section
                    ax_map, ax_cs = axs[f_idx, w_idx]

                    # Plot 2D intensity map
                    im = ax_map.imshow(
                        current_display_map.T,
                        aspect="auto",
                        origin="lower",
                        extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                        cmap=cmap,
                        vmin=vmin_plot,
                        vmax=vmax_plot,
                    )
                    ax_map.set_xlabel("X-Angle (degrees)")
                    ax_map.set_ylabel("Y-Angle (degrees)")
                    ax_map.set_title(
                        f"Field: {self.fields[f_idx]}, "
                        f"λ={self.wavelengths[w_idx].value:.3f} µm"
                    )
                    ax_map.grid(True, linestyle=":", alpha=0.7)  # Add grid to 2D plots
                    fig.colorbar(
                        im, ax=ax_map, label=cbar_label, fraction=0.046, pad=0.04
                    )

                    # Plot cross-section
                    central_row_index = current_display_map.shape[1] // 2
                    cross_section_data = current_display_map[:, central_row_index]
                    ax_cs.plot(
                        x_centers,
                        cross_section_data,
                        linestyle=cross_section_style,
                        color=cross_section_color,
                    )
                    ax_cs.set_xlabel("X-Angle (degrees)")
                    ax_cs.set_ylabel(cbar_label)
                    ax_cs.grid(True, linestyle=":", alpha=0.7)
                    ax_cs.set_xlim(x_centers[0], x_centers[-1])
                    ax_cs.set_ylim(bottom=-0.05 * vmax_plot, top=vmax_plot * 1.1)
                    ax_cs.set_title("Central Cross-Section")

        # Set overall title and layout
        fig.suptitle(main_title, fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if not is_gui_embedding and hasattr(fig, "canvas"):
            fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return fig, axs
