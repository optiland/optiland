"""Surface Visualization Module

This module contains classes for visualizing optical surfaces in 2D and 3D.

Kramer Harrison, 2024
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

import optiland.backend as be
from optiland.physical_apertures import RadialAperture
from optiland.rays import RealRays
from optiland.visualization.system.utils import revolve_contour, transform, transform_3d

# Surfaces tilted more than 60° toward the viewing axis are rendered as a
# boundary ellipse rather than a cross-section line to avoid artifacts.
_FACE_ON_THRESHOLD = 0.5  # |cos(60°)|


def _finite_aperture_extent(surface) -> float | None:
    """Return a finite physical drawing bound in the surface's local frame."""
    if surface.aperture is not None:
        bounds = be.to_numpy(be.array(surface.aperture.extent))
        if np.all(np.isfinite(bounds)):
            return float(np.max(np.abs(bounds)))
    # semi_aperture can be a paraxial ray-height estimate; it does not establish
    # a physical aperture or detector dimension.
    return None


def _schematic_image_extent(surface) -> float:
    """Choose a drawing scale, never a physical image or detector dimension."""
    previous = surface.previous_surface
    while previous is not None:
        radius = _finite_aperture_extent(previous)
        if radius is not None and radius > 0:
            return radius
        previous = previous.previous_surface
    return 1.0  # mm: deterministic schematic radius without any finite aperture


class Surface2D:
    """A class used to represent a 2D surface for visualization.

    Args:
        surface (Surface): The surface object containing the geometry.
        ray_extent (float): Local radial extent of valid incident ray samples.
        is_image_plane (bool): Use a schematic marker if this image endpoint
            has no finite physical aperture. Defaults to False.

    Attributes:
        surf (Surface): The surface object containing the geometry.
        extent (float): Resolved local radial drawing extent.
        extent_source (str): Physical aperture, ray samples or schematic sizing.

    Methods:
        plot(ax):
            Plots the surface on the given matplotlib axis.

    """

    def __init__(self, surface, ray_extent, *, is_image_plane: bool = False):
        self.surf = surface
        self.is_image_plane = is_image_plane
        extent = _finite_aperture_extent(surface)
        if extent is not None:
            self.extent = extent
            self.extent_source = "physical_aperture"
        elif is_image_plane:
            self.extent = _schematic_image_extent(surface)
            self.extent_source = "schematic"
        else:
            self.extent = ray_extent if be.isfinite(be.array(ray_extent)) else 0.0
            self.extent_source = "ray_samples"

    # Maps projection name to the index of the viewing axis in global normal.
    # e.g. for "YZ" the view is along X (index 0); "XZ" along Y (index 1); etc.
    _VIEWING_AXIS_IDX: dict[str, int] = {"YZ": 0, "XZ": 1, "XY": 2}

    def _is_face_on(self, projection: str) -> bool:
        """Return True if the surface normal is >60° toward the viewing axis.

        When True, rendering a cross-section line would create an artifact;
        drawing the aperture boundary ellipse is cleaner instead.
        """
        _, rot_mat = self.surf.geometry.cs.get_effective_transform()
        rot = be.to_numpy(rot_mat)
        axis_idx = self._VIEWING_AXIS_IDX[projection]
        return abs(float(rot[axis_idx, 2])) > _FACE_ON_THRESHOLD

    def plot(self, ax, theme=None, projection="YZ"):
        """Plots the surface on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the
                surface will be plotted.
            theme (Theme, optional): The theme to use for plotting.
                Defaults to None.
            projection (str, optional): The projection plane. Must be 'XY',
                'XZ', or 'YZ'. Defaults to 'YZ'.

        """
        # For surfaces strongly tilted toward the viewing axis the normal
        # cross-section line becomes a misleading vertical artifact.  Draw the
        # aperture boundary circle instead so the projection stays clean.
        sag_projection = "XY" if self._is_face_on(projection) else projection
        x, y, z = self._compute_sag(sag_projection)

        # convert to global coordinates and return
        x, y, z = transform(x, y, z, self.surf, is_global=False)

        x = be.to_numpy(x)
        y = be.to_numpy(y)
        z = be.to_numpy(z)

        color = "gray"
        if theme:
            color = theme.parameters.get("axes.edgecolor", color)

        label = f"Surface {self.surf.comment}"
        if self.extent_source == "schematic":
            label += " (schematic image plane)"
        if projection == "XY":
            (line,) = ax.plot(x, y, color=color, label=label)
        elif projection == "XZ":
            (line,) = ax.plot(z, x, color=color, label=label)
        else:  # YZ
            (line,) = ax.plot(z, y, color=color, label=label)
        return {line: self}

    def _compute_sag(self, projection="YZ"):
        """Computes the sag of the surface in local coordinates and handles
        clipping due to physical apertures.

        Returns:
            tuple: A tuple containing arrays of x, y, and z coordinates.

        """
        if projection == "XY":
            # local coordinates for XY circular aperture view
            theta = be.linspace(0, 2 * be.pi, 128)
            x = self.extent * be.cos(theta)
            y = self.extent * be.sin(theta)
            z = self.surf.geometry.sag(x, y)
            # No aperture clipping needed here as we are plotting the boundary
            return x, y, z

        # local coordinates for XZ or YZ cross-section
        if projection == "XZ":
            y = be.zeros(128)
            x = be.linspace(-self.extent, self.extent, 128)
        else:  # YZ
            x = be.zeros(128)
            y = be.linspace(-self.extent, self.extent, 128)
        z = self.surf.geometry.sag(x, y)

        # handle physical apertures for line cross-sections
        if self.surf.aperture:
            if projection == "XZ":
                x = be.copy(x)
            else:  # YZ
                y = be.copy(y)  # required to maintain gradient for torch backend
            intensity = be.ones_like(x)  # works for both cases
            rays = RealRays(x, y, x, x, x, x, intensity, x)
            self.surf.aperture.clip(rays)
            if projection == "XZ":
                x[rays.i == 0] = be.nan
            else:  # YZ
                y[rays.i == 0] = be.nan

        return x, y, z


class Surface3D(Surface2D):
    """A class used to represent a 3D surface for visualization.

    Args:
        surf (Surface): The surface object containing the geometry.
        extent (tuple): The extent of the surface in the x and y directions.

    Attributes:
        surf (Surface): The surface object containing the geometry.
        extent (tuple): The extent of the surface in the x and y directions.

    Methods:
        plot(renderer):
            Plots the 3D surface using the provided VTK renderer.

    """

    def __init__(self, surface, extent, *, is_image_plane: bool = False):
        super().__init__(surface, extent, is_image_plane=is_image_plane)

    def plot(self, renderer, theme=None, *args, **kwargs):
        """Plots the surface on the given renderer.

        Args:
            renderer (vtkRenderer): The renderer to which the surface actor
                will be added.
            theme (Theme, optional): The theme to use for plotting.
                Defaults to None.

        """
        actor = self.get_surface(theme=theme)
        self._configure_material(actor, theme=theme)
        renderer.AddActor(actor)

    def get_surface(self, theme=None):
        """Retrieves the surface actor based on the symmetry of the surface
        geometry.

        If the surface geometry is symmetric, it retrieves a symmetric surface
        actor. Otherwise, it retrieves an asymmetric surface actor.

        Returns:
            actor: The surface actor, either symmetric or asymmetric, based on
                the surface geometry.

        """
        has_symmetric_aperture = (
            type(self.surf.aperture) is RadialAperture
            or self.surf.aperture is None  # "no aperture" is symmetric
        )
        is_symmetric = self.surf.geometry.is_symmetric
        if is_symmetric and has_symmetric_aperture:
            actor = self._get_symmetric_surface()
        else:
            actor = self._get_asymmetric_surface()
        actor = self._configure_material(actor, theme=theme)
        return actor

    def _get_symmetric_surface(self):
        """Generates a symmetric surface actor by computing the sag, revolving
        the contour, transforming it in 3D, and configuring its material
        properties.

        Returns:
            vtkActor: The configured 3D actor representing the symmetric
                surface.

        """
        x, y, z = self._compute_sag()
        actor = revolve_contour(x, y, z)
        actor = transform_3d(actor, self.surf)
        return actor

    def _get_asymmetric_surface(self):
        """Build a clipped quad grid with bulk VTK point/connectivity arrays.

        Returns:
            vtk.vtkActor: A VTK actor representing the asymmetric surface.

        """
        x, y, z = self._compute_sag_3d()
        x = be.to_numpy(x)
        y = be.to_numpy(y)
        z = be.to_numpy(z)

        # Apply aperture filtering to the grid of points
        if self.surf.aperture is not None:
            mask = self.surf.aperture.contains(x, y)
        else:
            r = np.hypot(x, y)
            mask = r <= be.to_numpy(self.extent)

        # Preserve the original row-major vertex order and vtkPoints float32
        # precision, but avoid one Python/VTK call per coordinate and quad.
        points = vtk.vtkPoints()
        num_rows, num_cols = x.shape
        coordinates = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        points.SetData(numpy_to_vtk(coordinates.astype(np.float32), deep=True))
        point_ids = np.arange(num_rows * num_cols, dtype=np.int64).reshape(x.shape)
        mask = np.asarray(be.to_numpy(mask), dtype=bool)
        inside = mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]
        quads = np.column_stack(
            (
                point_ids[:-1, :-1][inside],
                point_ids[1:, :-1][inside],
                point_ids[1:, 1:][inside],
                point_ids[:-1, 1:][inside],
            )
        ).ravel()
        offsets = np.arange(0, len(quads) + 1, 4, dtype=np.int64)
        cells = vtk.vtkCellArray()
        cells.SetData(
            numpy_to_vtkIdTypeArray(offsets, deep=True),
            numpy_to_vtkIdTypeArray(quads, deep=True),
        )

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Convert to global coordinates
        actor = transform_3d(actor, self.surf)

        return actor

    def _configure_material(self, actor, theme=None):
        """Configures the material properties of a given actor.

        This method sets the color, ambient, diffuse, specular, and specular
        power properties of the actor's material.

        Args:
            actor: The actor whose material properties are to be configured.
            theme (Theme, optional): The theme to use for plotting.
                Defaults to None.

        Returns:
            The actor with updated material properties.

        """
        color = (1, 1, 1)
        if theme:
            from matplotlib.colors import to_rgb

            color_hex = theme.parameters.get("lens.color", "#FFFFFF")
            color = to_rgb(color_hex)

        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetAmbient(0.5)
        actor.GetProperty().SetDiffuse(0.05)
        actor.GetProperty().SetSpecular(1.0)
        actor.GetProperty().SetSpecularPower(100)

        return actor

    def _compute_sag_3d(self):
        """Computes the 3D sag (surface height) of the optical surface within the
        given extent.

        This method calculates the sag of the optical surface over a 2D grid
        of points. The sag is computed using the surface's geometry.

        Returns:
            tuple: A tuple containing three numpy arrays (x, y, z)
                representing the coordinates of the points on the surface
                within the maximum radial extent.

        """
        if self.surf.aperture is not None:
            x_min, x_max, y_min, y_max = self.surf.aperture.extent
            x = be.linspace(x_min, x_max, 256)
            y = be.linspace(y_min, y_max, 256)
            x, y = be.meshgrid(x, y)
        else:
            x = be.linspace(-self.extent, self.extent, 256)
            x, y = be.meshgrid(x, x)

        z = self.surf.geometry.sag(x, y)
        return x, y, z
