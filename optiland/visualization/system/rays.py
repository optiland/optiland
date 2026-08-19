"""Rays Visualization Module

This module contains classes for visualizing rays in an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

import numpy as np
import vtk

import optiland.backend as be
from optiland.utils import resolve_fields, resolve_wavelengths
from optiland.visualization.system.ray_bundle import RayBundle
from optiland.visualization.system.utils import transform


class Rays2D:
    """A class to represent and visualize 2D rays in an optical system.

    Args:
        optic (Optic): The optical system to be visualized.

    Attributes:
        optic (Optic): The optical system containing surfaces and fields.
        x (be.ndarray): X-coordinates of the rays.
        y (be.ndarray): Y-coordinates of the rays.
        z (be.ndarray): Z-coordinates of the rays.
        i (be.ndarray): Intensities of the rays.
        x_extent (be.ndarray): Extents of the x-coordinates for each surface.
        y_extent (be.ndarray): Extents of the y-coordinates for each surface.
        r_extent (be.ndarray): Extents of the radii for each surface.

    Methods:
        plot(ax, fields='all', wavelengths='primary', num_rays=3,
             distribution='line_y'):

    """

    def __init__(self, optic):
        self.optic = optic
        self.x = None
        self.y = None
        self.z = None
        self.i = None

        n = optic.surfaces.num_surfaces
        self.r_extent = be.zeros(n)

    def plot(
        self,
        ax,
        fields="all",
        wavelengths="primary",
        num_rays=3,
        distribution="line_y",
        reference=None,
        theme=None,
        projection="YZ",
        hide_vignetted=False,
    ):
        """Plots the rays for the given fields and wavelengths.

        Args:
            ax: The matplotlib axis to plot on.
            fields: The fields at which to trace the rays. Default is 'all'.
            wavelengths: The wavelengths at which to trace the rays.
                Default is 'primary'.
            num_rays: The number of rays to trace for each field and
                wavelength. Default is 3.
            distribution: The distribution of the rays. Default is 'line_y'.
            reference (str, optional): The reference rays to plot. Options
                include "chief" and "marginal". Defaults to None.
            theme (Theme, optional): The theme to apply. Defaults to None.
            hide_vignetted (bool, optional): If True, rays that vignette at any
                surface are not shown. Defaults to False.

        """
        # Visualization ignores weights; extract coords and values directly
        field_points = resolve_fields(self.optic, fields)
        wl_points = resolve_wavelengths(self.optic, wavelengths)
        fields_coords = [fp.coord for fp in field_points]
        wavelengths_vals = [wp.value for wp in wl_points]

        artists = {}

        for i, field in enumerate(fields_coords):
            for j, wavelength in enumerate(wavelengths_vals):
                # if only one field, use different colors for each wavelength
                color_idx = i if len(fields_coords) > 1 else j

                if distribution is None:
                    # trace only for surface extents
                    self._trace(field, wavelength, num_rays, "line_y")
                else:
                    # trace rays and plot lines
                    self._trace(field, wavelength, num_rays, distribution)
                    artists.update(
                        self._plot_lines(
                            ax,
                            color_idx,
                            field,
                            theme=theme,
                            projection=projection,
                            hide_vignetted=hide_vignetted,
                        )
                    )

                # trace reference rays and plot lines
                if reference is not None:
                    self._trace_reference(field, wavelength, reference)
                    artists.update(
                        self._plot_lines(
                            ax,
                            color_idx,
                            field,
                            linewidth=1.5,
                            theme=theme,
                            projection=projection,
                            hide_vignetted=hide_vignetted,
                        )
                    )
        return artists

    def _process_traced_rays(self):
        """Processes the traced rays and updates the surface extents."""
        self.x = self.optic.surfaces.x
        self.y = self.optic.surfaces.y
        self.z = self.optic.surfaces.z
        self.i = self.optic.surfaces.intensity

        # update surface extents
        self._update_surface_extents()

    def _trace(self, field, wavelength, num_rays, distribution):
        """Traces rays through the optical system and updates the surface extents.

        Args:
            field (tuple): The field coordinates for the ray tracing.
            wavelength (float): The wavelength of the rays.
            num_rays (int): The number of rays to trace.
            distribution (str): The distribution pattern of the rays.

        Returns:
            None

        """
        self.optic.trace(*field, wavelength, num_rays, distribution)
        self._process_traced_rays()

    def _trace_reference(self, field, wavelength, reference):
        """Traces reference rays through the optical system.

        Args:
            field (tuple): The field coordinates for the ray tracing.
            wavelength (float): The wavelength of the rays.
            reference (str): The type of reference rays to trace.

        Returns:
            None

        """
        if reference == "chief":
            self.optic.trace_generic(*field, Px=0, Py=0, wavelength=wavelength)
        elif reference == "marginal":
            self.optic.trace_generic(*field, Px=0, Py=1, wavelength=wavelength)
        else:
            raise ValueError(f"Invalid ray reference type: {reference}")

        self._process_traced_rays()

    def _update_surface_extents(self):
        """Updates the extents of the surfaces in the optic's surface group."""
        r_extent_new = be.copy(be.zeros_like(self.r_extent))
        for i, surf in enumerate(self.optic.surfaces):
            x_surf = self.x[i]
            y_surf = self.y[i]
            z_surf = self.z[i]

            # Convert to local coordinate system
            x, y, _ = transform(x_surf, y_surf, z_surf, surf, is_global=True)

            r_extent_new[i] = be.nanmax(be.hypot(x, y))
        self.r_extent = be.fmax(self.r_extent, r_extent_new)

    def _plot_lines(
        self,
        ax,
        color_idx,
        field,
        linewidth=1,
        theme=None,
        projection="YZ",
        hide_vignetted=False,
    ):
        """Plots multiple lines on the given axis.

        This method iterates through the rays stored in the object's attributes
        (self.x, self.y, self.z, self.i) and plots each valid ray on the
        provided axis. Rays that are outside the aperture (where self.i == 0)
        are excluded from the plot.

        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the lines.
            color_idx (int): The index used to determine the color of the
                lines.
            field (tuple): The field coordinates for the ray.
            linewidth (float): The width of the line.
            theme (Theme, optional): The theme to apply. Defaults to None.
            projection (str, optional): The projection plane. Must be 'XY',
                'XZ', or 'YZ'. Defaults to 'YZ'.


        Returns:
            None

        """
        artists = {}
        bundle_id = f"bundle_{color_idx}"
        # loop through rays
        for k in range(self.z.shape[1]):
            xk = be.to_numpy(self.x[:, k])
            yk = be.to_numpy(self.y[:, k])
            zk = be.to_numpy(self.z[:, k])
            ik = be.to_numpy(self.i[:, k])

            if np.any(ik == 0):
                if hide_vignetted:
                    continue
                first_zero_idx = np.where(ik == 0)[0][0]
                xk[first_zero_idx + 1 :] = np.nan
                yk[first_zero_idx + 1 :] = np.nan
                zk[first_zero_idx + 1 :] = np.nan

            artist, ray_bundle = self._plot_single_line(
                ax,
                xk,
                yk,
                zk,
                color_idx,
                field,
                linewidth,
                theme=theme,
                projection=projection,
            )
            ray_bundle.bundle_id = bundle_id
            artists[artist] = ray_bundle
        return artists

    def _plot_single_line(
        self, ax, x, y, z, color_idx, field, linewidth=1, theme=None, projection="YZ"
    ):
        """Plots a single line on the given axes.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot the line.
            x (array-like): The x-coordinates of the line.
            y (array-like): The y-coordinates of the line.
            z (array-like): The z-coordinates of the line.
            color_idx (int): The index for the color to use for the line.
            field (tuple): The field coordinates for the ray.
            linewidth (float): The width of the line. Default is 1.
            theme (Theme, optional): The theme to apply. Defaults to None.
            projection (str, optional): The projection plane. Must be 'XY',
                'XZ', or 'YZ'. Defaults to 'YZ'.

        Returns:
            None

        """
        if theme:
            ray_cycle = theme.parameters.get("ray_cycle")
            color = ray_cycle[color_idx % len(ray_cycle)]
        else:
            color = f"C{color_idx}"

        if projection == "XY":
            (line,) = ax.plot(x, y, color=color, linewidth=linewidth)
        elif projection == "XZ":
            (line,) = ax.plot(z, x, color=color, linewidth=linewidth)
        else:  # YZ
            (line,) = ax.plot(z, y, color=color, linewidth=linewidth)
        return line, RayBundle(x, y, z, field)


class Rays3D(Rays2D):
    """A class to represent 3D rays for visualization using VTK.
    Inherits from Rays2D and extends functionality to 3D.

    Methods:
        plot(ax, fields='all', wavelengths='primary', num_rays=3,
             distribution='line_y'):

    Args:
        optic: The optical system to be visualized.

    """

    def plot(
        self,
        ax,
        fields="all",
        wavelengths="primary",
        num_rays=3,
        distribution="line_y",
        reference=None,
        theme=None,
        hide_vignetted=False,
    ):
        """Plots the rays for the given fields and wavelengths.

        Args:
            ax: The matplotlib axis to plot on.
            fields: The fields at which to trace the rays. Default is 'all'.
            wavelengths: The wavelengths at which to trace the rays.
                Default is 'primary'.
            num_rays: The number of rays to trace for each field and
                wavelength. Default is 3.
            distribution: The distribution of the rays. Default is 'line_y'.
            reference (str, optional): The reference rays to plot. Options
                include "chief" and "marginal". Defaults to None.
            theme (Theme, optional): The theme to apply. Defaults to None.
            hide_vignetted (bool, optional): If True, rays that vignette at any
                surface are not shown. Defaults to False.

        """
        # Visualization ignores weights; extract coords and values directly
        field_points = resolve_fields(self.optic, fields)
        wl_points = resolve_wavelengths(self.optic, wavelengths)
        fields_coords = [fp.coord for fp in field_points]
        wavelengths_vals = [wp.value for wp in wl_points]

        for i, field in enumerate(fields_coords):
            for j, wavelength in enumerate(wavelengths_vals):
                # if only one field, use different colors for each wavelength
                color_idx = i if len(fields_coords) > 1 else j

                if distribution is None:
                    # trace only for surface extents
                    self._trace(field, wavelength, num_rays, "line_y")
                else:
                    # trace rays and plot lines
                    self._trace(field, wavelength, num_rays, distribution)
                    self._plot_lines(
                        ax,
                        color_idx,
                        field,
                        theme=theme,
                        hide_vignetted=hide_vignetted,
                    )

                # trace reference rays and plot lines
                if reference is not None:
                    self._trace_reference(field, wavelength, reference)
                    self._plot_lines(
                        ax,
                        color_idx,
                        field,
                        linewidth=1.5,
                        theme=theme,
                        hide_vignetted=hide_vignetted,
                    )

    def __init__(self, optic):
        super().__init__(optic)

        # matplotlib default colors converted to RGB
        self._rgb_colors = [
            (0.122, 0.467, 0.706),
            (1.000, 0.498, 0.055),
            (0.173, 0.627, 0.173),
            (0.839, 0.153, 0.157),
            (0.580, 0.404, 0.741),
            (0.549, 0.337, 0.294),
            (0.890, 0.467, 0.761),
            (0.498, 0.498, 0.498),
            (0.737, 0.741, 0.133),
            (0.090, 0.745, 0.812),
        ]

    def _plot_lines(
        self, ax, color_idx, field, linewidth=1, theme=None, hide_vignetted=False
    ):
        # loop through rays
        for k in range(self.z.shape[1]):
            xk = be.to_numpy(self.x[:, k])
            yk = be.to_numpy(self.y[:, k])
            zk = be.to_numpy(self.z[:, k])
            ik = be.to_numpy(self.i[:, k])

            if np.any(ik == 0):
                if hide_vignetted:
                    continue
                first_zero_idx = np.where(ik == 0)[0][0]
                xk[first_zero_idx + 1 :] = np.nan
                yk[first_zero_idx + 1 :] = np.nan
                zk[first_zero_idx + 1 :] = np.nan

            self._plot_single_line(
                ax, xk, yk, zk, color_idx, field, linewidth, theme=theme
            )

    def _plot_single_line(
        self, renderer, x, y, z, color_idx, field, linewidth=1, theme=None
    ):
        """Plots a single line in 3D space using VTK with the specified
        coordinates and color index.

        Args:
            renderer (vtkRenderer): The VTK renderer to add the line actor to.
            x (list of float): The x-coordinates of the line.
            y (list of float): The y-coordinates of the line.
            z (list of float): The z-coordinates of the line.
            color_idx (int): The index of the color to use from the
                _rgb_colors list.
            field (tuple): The field coordinates for the ray.
            linewidth (float): The width of the line. Default is 1.
            theme (Theme, optional): The theme to apply. Defaults to None.

        """
        if theme:
            from matplotlib.colors import to_rgb

            ray_cycle = theme.parameters.get("ray_cycle")
            color = to_rgb(ray_cycle[color_idx % len(ray_cycle)])
        else:
            color = self._rgb_colors[color_idx % 10]

        for k in range(1, len(x)):
            if np.isnan(x[k - 1]) or np.isnan(x[k]):
                continue

            p0 = [x[k - 1], y[k - 1], z[k - 1]]
            p1 = [x[k], y[k], z[k]]

            line_source = vtk.vtkLineSource()
            line_source.SetPoint1(p0)
            line_source.SetPoint2(p1)

            line_mapper = vtk.vtkPolyDataMapper()
            line_mapper.SetInputConnection(line_source.GetOutputPort())
            line_actor = vtk.vtkActor()
            line_actor.SetMapper(line_mapper)
            line_actor.GetProperty().SetLineWidth(linewidth)
            line_actor.GetProperty().SetColor(color)

            renderer.AddActor(line_actor)
