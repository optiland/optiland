"""Visualization Utilities Module

This module contains utility functions for visualization tasks.

Kramer Harrison, 2024
"""

from __future__ import annotations

import numpy as np  # noqa: TC002
import vtk

import optiland.backend as be
from optiland.rays import RealRays


def transform(x, y, z, surface, is_global=True):
    """Transforms the coordinates (x, y, z) based on the surface geometry.

    Args:
        x (numpy.ndarray): The x-coordinates of the points.
        y (numpy.ndarray): The y-coordinates of the points.
        z (numpy.ndarray): The z-coordinates of the points.
        surface (Surface): The surface object that contains the geometry for
            transformation.
        is_global (bool, optional): If True, localize the points to the
            surface geometry. If False, globalize the points. Defaults to True.

    Returns:
        tuple: Transformed x, y, and z coordinates as numpy arrays.

    """
    if be.isscalar(x):
        x = be.array([x])
    if be.isscalar(y):
        y = be.array([y])
    if be.isscalar(z):
        z = be.array([z])

    t = be.zeros(x.shape[0])
    points = RealRays(x, y, z, t, t, t, t, t)
    if is_global:
        surface.geometry.localize(points)
    else:
        surface.geometry.globalize(points)

    return points.x, points.y, points.z


def transform_3d(actor, surface):
    """Applies the effective rotation and translation of a surface to an actor.

    Args:
        actor: The actor to be transformed. This is typically an instance of a
            VTK actor.
        surface: The surface whose transformation will be applied to the actor.

    Returns:
        The transformed actor with the applied rotation and translation.

    """
    # Get the effective rotation and translation of the surface
    cs = surface.geometry.cs
    rx, ry, rz = cs.get_effective_rotation_euler()
    translation, _ = cs.get_effective_transform()
    dx, dy, dz = translation

    # Apply the rotation and translation to the actor
    actor.RotateZ(be.degrees(rz))
    actor.RotateY(be.degrees(ry))
    actor.RotateX(be.degrees(rx))
    actor.SetPosition(dx, dy, dz)

    return actor


def project_rays(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    projection: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D ray coordinates onto the chosen 2D plane.

    Args:
        xs: X coordinates, shape (N,).
        ys: Y coordinates, shape (N,).
        zs: Z coordinates, shape (N,).
        projection: One of ``'YZ'``, ``'XZ'``, ``'XY'``.

    Returns:
        (horizontal, vertical) coordinate arrays for the projection plane.
        YZ → (z, y), XZ → (z, x), XY → (x, y).
    """
    proj = projection.upper()
    if proj == "YZ":
        return zs, ys
    if proj == "XZ":
        return zs, xs
    # XY
    return xs, ys


def revolve_contour(x, y, z):
    """Revolves a contour defined by the input points (x, y, z) around the z-axis
    to create a 3D surface and returns the corresponding VTK actor.

    Args:
        x (list of float): List of x-coordinates of the contour points.
        y (list of float): List of y-coordinates of the contour points.
        z (list of float): List of z-coordinates of the contour points.

    Returns:
        vtk.vtkActor: VTK actor representing the revolved 3D surface.

    """
    pts = [(xi, yi, zi) for xi, yi, zi in zip(x, y, z, strict=False)]

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for pt in pts:
        pt_id = points.InsertNextPoint(pt)
        if pt_id < len(pts) - 1:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pt_id)
            line.GetPointIds().SetId(1, pt_id + 1)
            lines.InsertNextCell(line)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)

    revolution = vtk.vtkRotationalExtrusionFilter()
    revolution.SetInputData(poly_data)
    revolution.SetResolution(256)

    surface_mapper = vtk.vtkPolyDataMapper()
    surface_mapper.SetInputConnection(revolution.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(surface_mapper)

    return actor
