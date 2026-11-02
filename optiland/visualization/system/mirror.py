"""Mirror Visualization Module

This module contains the 2D and 3D mirror visualization classes.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vtk
from matplotlib.patches import Polygon

import optiland.backend as be
from optiland.physical_apertures import RadialAperture
from optiland.visualization.system.surface import Surface2D, Surface3D
from optiland.visualization.system.utils import (
    revolve_contour,
    transform,
    transform_3d,
)

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes


class Mirror2D(Surface2D):
    """Draw a reflective surface and its optional mechanical backing in 2D."""

    component_kind = "mirror"

    def __init__(
        self, surface: Any, ray_extent: float, *, backing_sign: float = 1.0
    ) -> None:
        """Set the side of the reflective surface occupied by the substrate.

        Args:
            surface: Reflective optical surface.
            ray_extent: Local radial drawing extent in millimeters.
            backing_sign: Sign of the incoming ray's local axial direction.
        """
        super().__init__(surface, ray_extent)
        self.backing_sign = backing_sign

    def plot(
        self, ax: Axes, theme: Any = None, projection: str = "YZ"
    ) -> dict[Artist, Mirror2D]:
        """Plot the mirror cross-section and a flat back at its center thickness.

        Args:
            ax: Matplotlib axes receiving the mirror artists.
            theme: Optional visualization theme.
            projection: Projection plane, either ``YZ``, ``XZ``, or ``XY``.

        Returns:
            Mapping from artists to this mirror component.
        """
        artists = {}
        thickness = self.surf.mirror_thickness
        if (
            thickness
            and projection in ("YZ", "XZ")
            and not self._is_face_on(projection)
        ):
            x, y, z = self._compute_sag(projection)
            front = transform(x, y, z, self.surf, is_global=False)
            back = transform(
                x,
                y,
                be.ones_like(z) * (self.backing_sign * thickness),
                self.surf,
                is_global=False,
            )
            front_axis = front[0] if projection == "XZ" else front[1]
            back_axis = back[0] if projection == "XZ" else back[1]
            front_points = be.column_stack((front[2], front_axis))
            back_points = be.column_stack((back[2], back_axis))
            finite = (
                be.isfinite(front[2])
                & be.isfinite(front_axis)
                & be.isfinite(back[2])
                & be.isfinite(back_axis)
            )
            facecolor = (0.6, 0.6, 0.6, 0.35)
            edgecolor = "gray"
            if theme:
                facecolor = theme.parameters.get("lens.color", facecolor)
                edgecolor = theme.parameters.get("axes.edgecolor", edgecolor)
            facecolor = self.face_color() or facecolor
            start = None
            for index in range(len(finite) + 1):
                if index < len(finite) and bool(finite[index]):
                    if start is None:
                        start = index
                    continue
                if start is None:
                    continue
                if index - start >= 2:
                    vertices = be.concatenate(
                        (front_points[start:index], be.flip(back_points[start:index]))
                    )
                    patch = Polygon(
                        vertices.tolist(),
                        closed=True,
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        label="Mirror backing",
                    )
                    ax.add_patch(patch)
                    artists[patch] = self
                start = None

        artists.update(super().plot(ax, theme=theme, projection=projection))
        return artists


class Mirror3D(Surface3D):
    """Render a reflective face and its optional mechanical substrate in 3D.

    Args:
        surface: Reflective optical surface.
        extent: Local radial drawing extent in millimeters.
        backing_sign: Sign of incoming travel along the local surface axis.
    """

    component_kind = "mirror"

    def __init__(
        self, surface: Any, extent: float, *, backing_sign: float = 1.0
    ) -> None:
        super().__init__(surface, extent)
        self.backing_sign = backing_sign

    def plot(self, renderer: vtk.vtkRenderer, theme: Any = None, **kwargs: Any) -> None:
        """Add the optical face and, when requested, its back and side walls.

        Args:
            renderer: VTK renderer receiving the mirror actors.
            theme: Optional visualization theme.
            **kwargs: Ignored projection settings from the system plotter.
        """
        is_annular = (
            type(self.surf.aperture) is RadialAperture and self.surf.aperture.r_min > 0
        )
        if is_annular:
            front = self._get_asymmetric_surface()
            renderer.AddActor(self._configure_material(front, theme=theme))
        else:
            super().plot(renderer, theme=theme)
        if not self.surf.mirror_thickness:
            return

        has_circular_aperture = self.surf.aperture is None or (
            type(self.surf.aperture) is RadialAperture and not is_annular
        )
        if self.surf.geometry.is_symmetric and has_circular_aperture:
            backing = self._revolved_backing()
        else:
            backing = self._gridded_backing()
        backing.GetProperty().SetColor(0.7, 0.7, 0.7)
        backing.GetProperty().SetAmbient(0.35)
        backing.GetProperty().SetDiffuse(0.65)
        renderer.AddActor(backing)

    def _revolved_backing(self) -> vtk.vtkActor:
        """Revolve a radial back profile for a circular mirror."""
        inner_radius = (
            self.surf.aperture.r_min
            if type(self.surf.aperture) is RadialAperture
            else 0.0
        )
        outer_radius = float(self.extent)
        radii = be.array([outer_radius, outer_radius, inner_radius, inner_radius])
        zeros = be.zeros(1)
        edge_sag = self.surf.geometry.sag(be.array([outer_radius]), zeros)
        inner_sag = self.surf.geometry.sag(be.array([inner_radius]), zeros)
        back_z = self.backing_sign * self.surf.mirror_thickness
        heights = be.concatenate((edge_sag, be.ones(2) * back_z, inner_sag))
        actor = revolve_contour(be.zeros(4).tolist(), radii.tolist(), heights.tolist())
        return transform_3d(actor, self.surf)

    def _add_quad(
        self, cells: vtk.vtkCellArray, ids: tuple[int, int, int, int]
    ) -> None:
        """Add one outward-facing quadrilateral to a VTK cell array."""
        cells.InsertNextCell(4)
        for point_id in ids if self.backing_sign > 0 else reversed(ids):
            cells.InsertCellPoint(point_id)

    def _gridded_backing(self) -> vtk.vtkActor:
        """Build a flat back and walls along an arbitrary aperture boundary."""
        x, y, z = self._compute_sag_3d()
        if self.surf.aperture is None:
            inside = be.hypot(x, y) <= self.extent
        else:
            inside = self.surf.aperture.contains(x, y)
        inside = inside & be.isfinite(z)
        valid_cells = (
            inside[:-1, :-1] & inside[1:, :-1] & inside[1:, 1:] & inside[:-1, 1:]
        )
        back_z = self.backing_sign * self.surf.mirror_thickness
        front_points = be.column_stack(
            (be.ravel(x), be.ravel(y), be.ravel(be.where(be.isfinite(z), z, 0)))
        )
        back_points = be.column_stack(
            (be.ravel(x), be.ravel(y), be.ravel(be.ones_like(z) * back_z))
        )
        points = vtk.vtkPoints()
        for coordinates in be.concatenate((front_points, back_points)).tolist():
            points.InsertNextPoint(*coordinates)

        rows, cols = x.shape
        back_offset = rows * cols
        cells = vtk.vtkCellArray()
        for row in range(rows - 1):
            for col in range(cols - 1):
                if not valid_cells[row, col]:
                    continue
                a = row * cols + col
                b = a + 1
                c = a + cols + 1
                d = a + cols
                self._add_quad(
                    cells,
                    (
                        a + back_offset,
                        b + back_offset,
                        c + back_offset,
                        d + back_offset,
                    ),
                )
                if row == 0 or not valid_cells[row - 1, col]:
                    self._add_quad(cells, (a, b, b + back_offset, a + back_offset))
                if col == cols - 2 or not valid_cells[row, col + 1]:
                    self._add_quad(cells, (b, c, c + back_offset, b + back_offset))
                if row == rows - 2 or not valid_cells[row + 1, col]:
                    self._add_quad(cells, (c, d, d + back_offset, c + back_offset))
                if col == 0 or not valid_cells[row, col - 1]:
                    self._add_quad(cells, (d, a, a + back_offset, d + back_offset))

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return transform_3d(actor, self.surf)

    def _configure_material(self, actor, theme=None):
        """Configures the material properties of the mirror surface.

        Args:
            actor (vtkActor): The actor representing the mirror surface.

        """
        override = self.face_color()
        actor.GetProperty().SetColor(*(override[:3] if override else (1, 1, 1)))
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.1)
        actor.GetProperty().SetSpecular(1.0)
        actor.GetProperty().SetSpecularPower(100)

        return actor
