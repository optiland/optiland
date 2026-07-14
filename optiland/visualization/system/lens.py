"""Lens Visualization Module

This module contains classes for visualizing lenses in 2D and 3D.

Kramer Harrison, 2024
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import vtk
from matplotlib.patches import Polygon

import optiland.backend as be
from optiland.visualization.system.utils import revolve_contour, transform, transform_3d


class Lens2D:
    """A class to represent a 2D lens and provide methods for plotting it.

    Args:
        surfaces (list): A list of surface objects that make up the lens.

    Attributes:
        surfaces (list): A list of surface objects that make up the lens.

    Methods:
        plot(ax):
            Plots the lens on the given matplotlib axis.

    """

    def __init__(self, surfaces):
        # TODO: raise warning when lens surfaces overlap
        self.surfaces = surfaces

    def plot(self, ax, theme=None, projection="YZ"):
        """Plots the lens on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the
                lens will be plotted.
            theme (Theme, optional): The theme to use for plotting.
                Defaults to None.
            projection (str, optional): The projection plane. Must be 'XY',
                'XZ', or 'YZ'. Defaults to 'YZ'.

        """
        if projection == "XY":
            # For XY projection, draw a circle representing the lens aperture
            max_extent = self._get_max_extent()

            # Get the center of the first surface in local coordinates (the origin)
            # and transform it to global coordinates to find the lens center.
            x_local, y_local, z_local = be.array([0]), be.array([0]), be.array([0])
            first_surface = self.surfaces[0].surf
            center_x_global, center_y_global, _ = transform(
                x_local, y_local, z_local, first_surface, is_global=False
            )

            facecolor = (0.8, 0.8, 0.8, 0.6)
            edgecolor = (0.5, 0.5, 0.5)
            if theme:
                facecolor = theme.parameters.get("lens.color", facecolor)
                edgecolor = theme.parameters.get("axes.edgecolor", edgecolor)

            circle = plt.Circle(
                (
                    float(be.to_numpy(center_x_global).item()),
                    float(be.to_numpy(center_y_global).item()),
                ),
                float(be.to_numpy(max_extent).item()),
                facecolor=facecolor,
                edgecolor=edgecolor,
                label="Lens",
            )
            ax.add_patch(circle)
            return {circle: self}
        else:
            sags = self._compute_sag(projection=projection)
            return self._plot_lenses(ax, sags, theme=theme, projection=projection)

    def _compute_sag(self, apply_transform=True, projection="YZ"):
        """Computes the sag of the lens in local coordinates and handles
        clipping due to physical apertures.

        Returns:
            list: A list of tuples containing arrays of x, y, and z
                coordinates.

        """
        max_extent = self._get_max_extent()
        sags = []
        for surf in self.surfaces:
            x, y, z = surf._compute_sag(projection)

            # extend surface to max extent
            if surf.extent < max_extent:
                x, y, z = self._extend_surface(
                    x, y, z, surf.surf, max_extent, projection
                )

            # convert to global coordinates
            if apply_transform:
                x, y, z = transform(x, y, z, surf.surf, is_global=False)

            sags.append((x, y, z))

        return sags

    def _get_max_extent(self):
        """Gets the maximum radial extent of all surfaces in the lens in global
        coordinates.

        Returns:
            float: The maximum radial extent of all surfaces in the lens.

        """
        extents = be.array([surf.extent for surf in self.surfaces])
        return be.nanmax(extents, axis=0)

    def _extend_surface(self, x, y, z, surface, extent, projection="YZ"):
        """Extends the surface to the maximum extent.

        Args:
            x (numpy.ndarray): The x coordinates of the surface.
            y (numpy.ndarray): The y coordinates of the surface.
            z (numpy.ndarray): The z coordinates of the surface.
            surface (Surface): The surface object.
            extent (numpy.ndarray): The maximum extent of the surface.
            projection (str, optional): The projection plane. Must be 'XY',
                'XZ', or 'YZ'. Defaults to 'YZ'.

        Returns:
            tuple: A tuple containing the extended x, y, and z coordinates.

        """
        if projection == "XZ":
            x_new = be.array([extent])
            x = be.concatenate([-x_new, x, x_new])
            y = be.concatenate([be.array([0]), y, be.array([0])])
            z = be.concatenate([be.array([z[0]]), z, be.array([z[-1]])])
        else:  # YZ
            y_new = be.array([extent])
            x = be.concatenate([be.array([0]), x, be.array([0])])
            y = be.concatenate([-y_new, y, y_new])
            z = be.concatenate([be.array([z[0]]), z, be.array([z[-1]])])

        surface.extent = extent

        return x, y, z

    def _plot_single_lens(self, ax, x, y, z, theme=None, projection="YZ"):
        """Plot a single lens on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the
                lens will be plotted.
            x (numpy.ndarray): The x coordinates of the lens.
            y (numpy.ndarray): The y coordinates of the lens.
            z (numpy.ndarray): The z coordinates of the lens.
            theme (Theme, optional): The theme to use for plotting.
                Defaults to None.
            projection (str, optional): The projection plane. Must be 'XY',
                'XZ', or 'YZ'. Defaults to 'YZ'.

        """
        facecolor = (0.8, 0.8, 0.8, 0.6)
        edgecolor = (0.5, 0.5, 0.5)
        if theme:
            facecolor = theme.parameters.get("lens.color", facecolor)
            edgecolor = theme.parameters.get("axes.edgecolor", edgecolor)

        if projection == "XZ":
            vertices = be.to_numpy(be.column_stack((z, x)))
        else:
            vertices = be.to_numpy(be.column_stack((z, y)))

        polygons = []
        for segment in self._finite_vertex_segments(vertices):
            polygon = Polygon(
                segment,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                label="Lens",
            )
            ax.add_patch(polygon)
            polygons.append(polygon)
        return polygons

    def _finite_vertex_segments(self, vertices):
        """Split closed polygon vertices into finite contiguous segments."""
        finite = np.isfinite(vertices).all(axis=1)
        if finite.all():
            return [vertices]

        finite_indices = np.flatnonzero(finite)
        if finite_indices.size == 0:
            return []

        breaks = np.where(np.diff(finite_indices) > 1)[0] + 1
        segments = np.split(finite_indices, breaks)

        # The polygon is closed, so finite runs at the start and end are
        # contiguous across the closing edge and should form one patch.
        if finite[0] and finite[-1] and len(segments) > 1:
            segments[0] = np.concatenate([segments[-1], segments[0]])
            segments = segments[:-1]

        return [vertices[segment] for segment in segments if len(segment) > 2]

    def _plot_lenses(self, ax, sags, theme=None, projection="YZ"):
        """Plot the lenses on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the
                lenses will be plotted.
            sags (list): A list of tuples containing arrays of x, y, and z
                coordinates for each surface.
            theme (Theme, optional): The theme to use for plotting.
                Defaults to None.
            projection (str, optional): The projection plane. Must be 'XY',
                'XZ', or 'YZ'. Defaults to 'YZ'.

        """
        artists = {}
        for k in range(len(sags) - 1):
            x1, y1, z1 = sags[k]
            x2, y2, z2 = sags[k + 1]

            # plot lens
            x = be.concatenate([x1, be.flip(x2)])
            y = be.concatenate([y1, be.flip(y2)])
            z = be.concatenate([z1, be.flip(z2)])

            artists_for_lens = self._plot_single_lens(
                ax, x, y, z, theme=theme, projection=projection
            )
            if artists_for_lens is None:
                continue
            if not isinstance(artists_for_lens, list):
                artists_for_lens = [artists_for_lens]
            for artist in artists_for_lens:
                artists[artist] = self
        return artists


class Lens3D(Lens2D):
    """A class used to represent a 3D Lens, inheriting from Lens2D.

    Args:
        surfaces (list): A list of surfaces that make up the lens.
                         Each element is expected to be an object (e.g., Surface3D)
                         that has a `surf` attribute (the actual Surface object)
                         and an `extent` attribute.

    Attributes:
        surfaces (list): A list of surfaces that make up the lens.

    Methods:
        is_symmetric:
            Checks if the lens is rotationally symmetric.
        plot(renderer):
            Plots the lens using the given VTK renderer.

    """

    def __init__(self, surfaces):
        super().__init__(surfaces)

    @property
    def is_symmetric(self):
        """Check if all surfaces in the lens are symmetric.

        This method iterates through each surface in the lens and checks if the
        geometry of the surface is symmetric. A surface is considered symmetric
        if its geometry's `is_symmetric` attribute is True and both `rx` and
        `ry` attributes of its coordinate system (`cs`) are zero.

        Returns:
            bool: True if all surfaces are symmetric, False otherwise.

        """
        for surf in self.surfaces:
            geometry = surf.surf.geometry
            if not geometry.is_symmetric:
                return False
            if (
                geometry.cs.rx != 0
                or geometry.cs.ry != 0
                or geometry.cs.x != 0
                or geometry.cs.y != 0
            ):
                return False
        return True

    def plot(self, renderer, theme=None, *args, **kwargs):
        """Plots the lens or surfaces using the provided renderer.

        Args:
            renderer: The rendering engine used to plot the lens or surfaces.
            theme (Theme, optional): The theme to use for plotting.
                Defaults to None.

        """
        if self.is_symmetric:
            sags = self._compute_sag()  # sags are in global coordinates
            self._plot_lenses(renderer, sags, theme=theme)
        else:
            self._plot_surfaces(renderer, theme=theme)
            self._plot_surface_edges(renderer, theme=theme)

    def _plot_single_lens(self, renderer, x, y, z, theme=None, *args, **kwargs):
        """Plots a single lens by revolving a contour and configuring its
        material. The input coordinates are expected to be in global space.

        Args:
            renderer (vtkRenderer): The renderer to which the lens actor will
                be added.
            x (numpy.ndarray): The x-coordinates of the contour points.
            y (numpy.ndarray): The y-coordinates of the contour points.
            z (numpy.ndarray): The z-coordinates of the contour points.
            theme (Theme, optional): The theme to use for plotting.
                Defaults to None.

        """
        # Points are already global from _compute_sag with apply_transform=True
        actor = revolve_contour(be.to_numpy(x), be.to_numpy(y), be.to_numpy(z))
        actor = self._configure_material(actor, theme=theme)
        renderer.AddActor(actor)

    def _configure_material(self, actor, theme=None):
        """Configures the material properties of a given VTK actor.
        This method sets the opacity, color, specular, and specular power
        properties of the provided VTK actor to predefined values.

        Args:
            actor (vtkActor): The VTK actor whose material properties are to
                be configured.
            theme (Theme, optional): The theme to use for plotting.
                Defaults to None.

        Returns:
            vtkActor: The VTK actor with updated material properties.

        """
        color = (0.9, 0.9, 1.0)
        if theme:
            from matplotlib.colors import to_rgb

            color_hex = theme.parameters.get("lens.color", "#E5E5FF")
            color = to_rgb(color_hex)

        prop = actor.GetProperty()
        prop.SetOpacity(0.5)
        prop.SetColor(color)
        prop.SetSpecular(1.0)
        prop.SetSpecularPower(50.0)
        return actor

    def _plot_surfaces(self, renderer, theme=None):
        """Plots the non-symmetric surfaces of the lens in the given renderer.

        This method retrieves a pre-transformed actor for each surface,
        configures its material, and adds it to the renderer.
        If a surface's extent is less than the maximum lens extent,
        an annulus is plotted to fill the gap.

        Args:
            renderer (vtkRenderer): The VTK renderer where the surfaces will
                be added.
            theme (Theme, optional): The theme to use for plotting.
                Defaults to None.

        """
        max_extent = self._get_max_extent()
        for (
            surface_3d_obj
        ) in self.surfaces:  # surface_3d_obj is an instance of e.g. Surface3D
            actor = (
                surface_3d_obj.get_surface()
            )  # retrieves actor from Surface3D (already transformed)
            actor = self._configure_material(actor, theme=theme)
            renderer.AddActor(actor)

            # Add annulus if surface extent does not extend to lens edge
            if surface_3d_obj.extent < max_extent:
                self._plot_annulus(
                    surface_3d_obj, renderer, theme=theme
                )  # Pass renderer

    def _plot_annulus(self, surface_3d_obj, renderer, theme=None):
        """Plots a VTK annulus for a given surface.

        The annulus extends from the surface's current extent to the maximum
        lens extent. It is defined in the local coordinate system of the
        surface and then transformed to its global position.

        Args:
            surface_3d_obj: The surface object (e.g., an instance of Surface3D)
                            for which to plot the annulus. It should have `surf`
                            (Surface object with geometry and cs) and `extent`
                            attributes.
            renderer (vtkRenderer): The VTK renderer to add the annulus actor to.
        """
        surf_props = surface_3d_obj.surf  # Actual Surface object with .geometry
        surf_geom = surf_props.geometry

        inner_radius = surface_3d_obj.extent
        outer_radius = self._get_max_extent()

        if outer_radius <= inner_radius:
            # No annulus needed if the surface already extends to or beyond max_extent
            return

        num_theta_points = 64
        # Do not take endpoint that overlaps with first point
        theta = be.linspace(0, 2 * be.pi, num_theta_points + 1)[:-1]

        # --- Define annulus geometry in local coordinates ---
        # Inner circle points
        x_inner_local = inner_radius * be.cos(theta)
        y_inner_local = inner_radius * be.sin(theta)
        z_inner_local = surf_geom.sag(x_inner_local, y_inner_local)

        # Outer circle points
        x_outer_local = outer_radius * be.cos(theta)
        y_outer_local = outer_radius * be.sin(theta)
        z_outer_local = surf_geom.sag(x_outer_local, y_outer_local)

        # Convert to NumPy arrays for VTK
        points_data = [
            be.to_numpy(arr)
            for arr in [
                x_inner_local,
                y_inner_local,
                z_inner_local,
                x_outer_local,
                y_outer_local,
                z_outer_local,
            ]
        ]
        x_inner_np, y_inner_np, z_inner_np, x_outer_np, y_outer_np, z_outer_np = (
            points_data
        )

        # Create vtkPoints object
        vtk_pts = vtk.vtkPoints()
        for i in range(num_theta_points):
            vtk_pts.InsertNextPoint(x_inner_np[i], y_inner_np[i], z_inner_np[i])
        for i in range(num_theta_points):
            vtk_pts.InsertNextPoint(x_outer_np[i], y_outer_np[i], z_outer_np[i])

        # Create vtkCellArray for the quadrilaterals forming the annulus surface
        quads = vtk.vtkCellArray()
        for i in range(num_theta_points):
            quad = vtk.vtkQuad()
            p_inner_curr = i
            p_inner_next = (i + 1) % num_theta_points
            # Offset for outer circle points in vtk_pts array
            p_outer_curr = num_theta_points + i
            p_outer_next = num_theta_points + ((i + 1) % num_theta_points)

            quad.GetPointIds().SetId(0, p_inner_curr)
            quad.GetPointIds().SetId(1, p_inner_next)
            quad.GetPointIds().SetId(2, p_outer_next)
            quad.GetPointIds().SetId(3, p_outer_curr)
            quads.InsertNextCell(quad)

        # Create vtkPolyData
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_pts)
        polydata.SetPolys(quads)

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        annulus_actor = vtk.vtkActor()
        annulus_actor.SetMapper(mapper)

        # Configure material properties
        annulus_actor = self._configure_material(annulus_actor, theme=theme)
        annulus_actor = transform_3d(annulus_actor, surf_props)
        renderer.AddActor(annulus_actor)

    def _get_edge_surface(self, circle1, circle2, theme=None):
        """Generates a VTK actor representing the surface between two circles.

        Args:
            circle1 (list of tuple): List of points representing the first
                circle.
            circle2 (list of tuple): List of points representing the second
                circle.
            theme (Theme, optional): The theme to use for plotting.
                Defaults to None.

        Returns:
            vtk.vtkActor: VTK actor representing the surface between the two
                circles.

        """
        num_points = len(circle1)

        # Create vtkPoints object to hold all the points from both circles
        points = vtk.vtkPoints()
        for p in circle1:
            points.InsertNextPoint(p)
        for p in circle2:
            points.InsertNextPoint(p)

        # Create vtkCellArray to hold the quadrilaterals forming the surface
        polys = vtk.vtkCellArray()

        # Loop over the points and connect points from the two circles
        for i in range(num_points):
            next_i = (i + 1) % num_points  # Wrap around for the last point

            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, i)
            quad.GetPointIds().SetId(1, next_i)
            quad.GetPointIds().SetId(2, num_points + next_i)
            quad.GetPointIds().SetId(3, num_points + i)

            polys.InsertNextCell(quad)

        # Create the vtkPolyData object
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polys)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor = self._configure_material(actor, theme=theme)

        return actor

    def _plot_surface_edges(self, renderer, theme=None):
        """Plots the edges of surfaces in a 3D renderer.

        This method generates circular edges for each surface. It then
        transforms these edges into the appropriate coordinate system and adds
        them to the renderer.

        Args:
            renderer: The 3D renderer object where the surface edges will be
                added.

        """
        circles = []
        for surface in self.surfaces:
            x, y, z = self._get_edge_points(surface)
            x, y, z = transform(x, y, z, surface.surf, is_global=False)
            x = be.to_numpy(x)
            y = be.to_numpy(y)
            z = be.to_numpy(z)
            circles.append(np.stack((x, y, z), axis=-1))

        for k in range(len(circles) - 1):
            circle1 = circles[k]
            circle2 = circles[k + 1]
            actor = self._get_edge_surface(circle1, circle2, theme=theme)
            renderer.AddActor(actor)

    def _get_edge_points(self, surface_obj):
        """Computes the (x, y, z) local coordinates of the edges of the lens.

        Args:
            surface_obj (Surface): The surface object itself (not Surface3D).

        Returns:
            tuple: A tuple containing arrays of x, y, and z
                coordinates in the local coordinate system of surface_obj.
        """
        max_extent_lens = self._get_max_extent()
        theta = be.linspace(0, 2 * be.pi, 256)  # 256 points for smooth edge

        x_local = max_extent_lens * be.cos(theta)
        y_local = max_extent_lens * be.sin(theta)
        z_local = surface_obj.surf.geometry.sag(x_local, y_local)

        return x_local, y_local, z_local
