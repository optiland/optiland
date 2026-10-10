"""Small planar display meshes with explicit aperture boundaries and holes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

import optiland.backend as be
from optiland.geometries.plane import Plane
from optiland.geometries.standard import StandardGeometry
from optiland.physical_apertures import (
    EllipticalAperture,
    OffsetRadialAperture,
    RadialAperture,
    RectangularAperture,
)
from optiland.visualization.system.utils import transform_3d

if TYPE_CHECKING:
    from optiland.visualization.system.surface import Surface3D


def compact_planar_face(view: Surface3D) -> vtk.vtkActor | None:
    """Return a planar actor, or None when the existing mesh must be retained.

    A plane needs its boundary and planar cells, not a 256-by-256 sag grid.
    Curved/custom geometries and composite apertures retain their normal meshes.
    Circular boundaries use the existing revolution's 256 angular intervals.
    """
    geometry = view.surf.geometry
    if not (
        type(geometry) is Plane
        or (
            type(geometry) is StandardGeometry
            and bool(np.isinf(be.to_numpy(geometry.radius)).all())
        )
    ):
        return None
    aperture = view.surf.aperture
    if type(aperture) is RectangularAperture:
        left, right, bottom, top = map(float, aperture.extent)
        coordinates = np.array(
            [(left, bottom, 0), (right, bottom, 0), (right, top, 0), (left, top, 0)]
        )
        connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)
    else:
        center = np.zeros(2)
        inner_radius = 0.0
        if aperture is None:
            a = b = float(view.extent)
        elif type(aperture) in (RadialAperture, OffsetRadialAperture):
            a = b = float(aperture.r_max)
            inner_radius = float(aperture.r_min)
            if type(aperture) is OffsetRadialAperture:
                center[:] = float(aperture.offset_x), float(aperture.offset_y)
        elif type(aperture) is EllipticalAperture:
            if aperture.offset_x != 0 or aperture.offset_y != 0:
                return None
            a, b = float(aperture.a), float(aperture.b)
        else:
            return None
        if not np.isfinite([a, b, inner_radius, *center]).all() or min(a, b) <= 0:
            return None
        if not 0 <= inner_radius < min(a, b):
            return None
        theta = np.arange(256) * (2 * np.pi / 256)
        unit = np.column_stack((np.cos(theta), np.sin(theta)))
        outer = unit * (a, b) + center
        indices = np.arange(256, dtype=np.int64)
        following = (indices + 1) % 256
        if inner_radius:
            xy = np.vstack((outer, unit * inner_radius + center))
            connectivity = np.column_stack(
                (indices, following, following + 256, indices + 256)
            )
        else:
            xy = np.vstack((outer, center))
            connectivity = np.column_stack(
                (np.full(256, 256, dtype=np.int64), indices, following)
            )
        coordinates = np.column_stack((xy, np.zeros(len(xy))))
    if not np.isfinite(coordinates).all():
        return None
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(coordinates.astype(np.float32), deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(
            np.arange(len(connectivity) + 1, dtype=np.int64) * connectivity.shape[1],
            deep=True,
        ),
        numpy_to_vtkIdTypeArray(connectivity.ravel(), deep=True),
    )
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    view._configure_material(actor)
    return transform_3d(actor, view.surf)
