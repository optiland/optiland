"""Prepare owned layout geometry in the calculation process.

Existing plotters run against an isolated, noninteractive Matplotlib figure or
render-free VTK collector. Only copied numerical arrays and plain metadata leave
the process: no figure, artist, actor, render context, or Optic is transported.
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland_gui.services.job_records import check_cancelled


def _surface_indices(component, indices):
    if hasattr(component, "surfaces"):
        return tuple(indices[id(surface.surf)] for surface in component.surfaces)
    surface = getattr(component, "surf", component)
    index = indices.get(id(surface))
    return () if index is None else (index,)


def prepare_2d(snapshot, parameters, progress, cancelled):
    """Return themed-independent 2D primitives and surface/body ownership."""
    from matplotlib.figure import Figure
    from matplotlib.text import Annotation

    from optiland.visualization.system.lens import Lens2D
    from optiland.visualization.system.ray_bundle import RayBundle
    from optiland.visualization.system.rays import Rays2D
    from optiland.visualization.system.surface import Surface2D
    from optiland.visualization.system.system import OpticalSystem
    from optiland.visualization.system.utils import transform

    progress("Restoring optical snapshot")
    optic = snapshot.restore()
    indices = {id(surface): i for i, surface in enumerate(optic.surfaces)}
    figure = Figure()
    axes = figure.add_subplot()
    rays = Rays2D(optic)
    progress("Tracing layout rays")
    ray_artists = rays.plot(
        axes,
        fields="all",
        wavelengths="primary",
        num_rays=parameters["num_rays"],
        distribution=parameters["distribution"],
    )
    check_cancelled(cancelled)
    progress("Preparing optical outlines")
    system = OpticalSystem(optic, rays, projection="2d")
    artists = {**ray_artists, **system.plot(axes)}
    primitives = []
    boundaries = {}
    references = {}
    surface_views = {}
    for component in system.components:
        if isinstance(component, Lens2D):
            surface_views.update((id(item.surf), item) for item in component.surfaces)
        elif isinstance(component, Surface2D):
            surface_views[id(component.surf)] = component
    for line in axes.lines:
        component = artists.get(line)
        role = "ray" if isinstance(component, RayBundle) else "surface"
        if id(component) in indices:
            role = "aperture"
        primitives.append(
            {
                "kind": "line",
                "xy": line.get_xydata().copy(),
                "role": role,
                "surfaces": _surface_indices(component, indices),
                "color_index": int(component.bundle_id.rsplit("_", 1)[-1])
                if role == "ray"
                else 0,
                "linewidth": line.get_linewidth(),
                "linestyle": line.get_linestyle(),
                "label": line.get_label(),
            }
        )
    for patch in axes.patches:
        component = artists.get(patch)
        if not hasattr(patch, "get_xy"):
            raise ValueError("This component needs a numerical 2D layout adapter.")
        owned = _surface_indices(component, indices)
        pairs = getattr(component, "artist_surfaces", {})
        if patch in pairs:
            owned = tuple(indices[id(surface)] for surface in pairs[patch])
        primitives.append(
            {
                "kind": "polygon",
                "xy": patch.get_xy().copy(),
                "role": "lens",
                "surfaces": owned,
                "linewidth": patch.get_linewidth(),
            }
        )
    for component in system.components:
        if isinstance(component, Lens2D):
            sags = component._compute_sag()
            for surface, (_x, y, z) in zip(component.surfaces, sags, strict=True):
                boundaries[indices[id(surface.surf)]] = (
                    be.to_numpy(z).copy(),
                    be.to_numpy(y).copy(),
                )
    # Reference markers are available only on demand for editor highlighting.
    # They are never added to the ordinary physical layout or ray path.
    for index, surface in enumerate(optic.surfaces):
        if index in boundaries or getattr(surface, "is_infinite", False):
            continue
        item = surface_views.get(id(surface))
        if item is None:
            item = Surface2D(surface, rays.r_extent[index])
        projection = "XY" if item._is_face_on("YZ") else "YZ"
        x, y, z = item._compute_sag(projection)
        _, y, z = transform(x, y, z, surface, is_global=False)
        references[index] = (be.to_numpy(z).copy(), be.to_numpy(y).copy())
    annotations = [
        {
            "xy": tuple(text.xy),
            "xytext": tuple(text.get_position()),
            "arrowprops": dict(text.arrowprops),
        }
        for text in axes.texts
        if isinstance(text, Annotation) and text.arrowprops
    ]
    check_cancelled(cancelled)
    return {
        "name": optic.name,
        "primitives": primitives,
        "boundaries": boundaries,
        "references": references,
        "annotations": annotations,
        "extent": be.to_numpy(rays.r_extent).copy(),
        "extent_sources": {
            indices[identity]: item.extent_source
            for identity, item in surface_views.items()
            if hasattr(item, "extent_source")
        },
    }


class _ActorCollector:
    """Collect CPU-side actors without constructing a renderer or render window."""

    def __init__(self):
        self.actors = []

    def AddActor(self, actor):  # noqa: N802
        self.actors.append(actor)


def prepare_3d(snapshot, parameters, progress, cancelled):
    """Return VTK polydata arrays without any native rendering context."""
    from vtk.util.numpy_support import vtk_to_numpy

    from optiland.visualization.system.rays import Rays3D
    from optiland.visualization.system.system import OpticalSystem

    progress("Restoring optical snapshot")
    optic = snapshot.restore()
    indices = {id(surface): i for i, surface in enumerate(optic.surfaces)}
    rays = Rays3D(optic)
    collector = _ActorCollector()
    progress("Tracing 3D rays")
    rays.plot(
        collector, fields="all", wavelengths="primary", num_rays=24, distribution="ring"
    )
    groups = [(actor, "ray", ()) for actor in collector.actors]
    system = OpticalSystem(optic, rays, projection="3d")
    system._identify_components()
    for index, component in enumerate(system.components):
        progress("Preparing 3D components", index, len(system.components))
        collector.actors = []
        component.plot(collector)
        groups.extend(
            (actor, "lens", _surface_indices(component, indices))
            for actor in collector.actors
        )
    meshes = []
    for index, (actor, role, surfaces) in enumerate(groups):
        progress("Preparing display arrays", index, len(groups))
        mapper = actor.GetMapper()
        mapper.Update()
        data = mapper.GetInput()
        if data is None or data.GetPoints() is None:
            continue
        cells = {}
        for name, source in (
            ("polys", data.GetPolys()),
            ("lines", data.GetLines()),
            ("verts", data.GetVerts()),
            ("strips", data.GetStrips()),
        ):
            if source.GetNumberOfCells():
                cells[name] = (
                    vtk_to_numpy(source.GetOffsetsArray()).copy(),
                    vtk_to_numpy(source.GetConnectivityArray()).copy(),
                )
        matrix = actor.GetMatrix()
        prop = actor.GetProperty()
        color = tuple(prop.GetColor())
        color_index = min(
            range(len(rays._rgb_colors)),
            key=lambda i: np.linalg.norm(np.asarray(rays._rgb_colors[i]) - color),
        )
        meshes.append(
            {
                "points": vtk_to_numpy(data.GetPoints().GetData()).copy(),
                "cells": cells,
                "matrix": np.array(
                    [[matrix.GetElement(i, j) for j in range(4)] for i in range(4)]
                ),
                "color": color,
                "color_index": color_index,
                "opacity": prop.GetOpacity(),
                "ambient": prop.GetAmbient(),
                "diffuse": prop.GetDiffuse(),
                "specular": prop.GetSpecular(),
                "power": prop.GetSpecularPower(),
                "linewidth": prop.GetLineWidth(),
                "role": role,
                "surfaces": surfaces,
                "normals": vtk_to_numpy(data.GetPointData().GetNormals()).copy()
                if data.GetPointData().GetNormals() is not None
                else None,
            }
        )
    check_cancelled(cancelled)
    return {"name": optic.name, "meshes": _batch_ray_meshes(meshes)}


def _batch_ray_meshes(meshes):
    """Pack equal-style ray segments without changing vertices or connectivity.

    The public renderer can produce one actor per segment. Sending one polydata
    per style avoids hundreds of GUI-side actor allocations and draw calls.
    Optical body ownership and surface meshes remain separate.
    """
    groups = {}
    result = []
    for mesh in meshes:
        if mesh["role"] != "ray" or set(mesh["cells"]) != {"lines"}:
            result.append(mesh)
            continue
        key = tuple(
            mesh[name]
            for name in (
                "color",
                "color_index",
                "opacity",
                "ambient",
                "diffuse",
                "specular",
                "power",
                "linewidth",
            )
        ) + (mesh["matrix"].tobytes(),)
        if key not in groups:
            groups[key] = []
            result.append(groups[key])
        groups[key].append(mesh)
    batched = []
    for item in result:
        if isinstance(item, dict):
            batched.append(item)
            continue
        points, offsets, connectivity = [], [np.array([0], dtype=np.int64)], []
        point_count = cell_size = 0
        for mesh in item:
            mesh_offsets, mesh_cells = mesh["cells"]["lines"]
            points.append(mesh["points"])
            offsets.append(mesh_offsets[1:] + cell_size)
            connectivity.append(mesh_cells + point_count)
            point_count += len(mesh["points"])
            cell_size += len(mesh_cells)
        batched.append(
            {
                **item[0],
                "points": np.concatenate(points),
                "cells": {
                    "lines": (np.concatenate(offsets), np.concatenate(connectivity))
                },
            }
        )
    return batched


def prepare_sag(snapshot, parameters, progress, cancelled):
    """Compute sag grid/profile arrays while keeping the embedded figure GUI-owned."""
    progress("Computing surface sag")
    optic = snapshot.restore()
    surface = optic.surfaces[parameters["surface_index"]]
    extent = parameters["max_extent"]
    coordinates = be.linspace(-extent, extent, 50)
    x, y = be.meshgrid(coordinates, coordinates)
    sag = surface.geometry.sag(x, y)
    profile_x = surface.geometry.sag(
        coordinates, be.full_like(coordinates, parameters["y_cross_section"])
    )
    profile_y = surface.geometry.sag(
        be.full_like(coordinates, parameters["x_cross_section"]), coordinates
    )
    check_cancelled(cancelled)
    return {
        "coordinates": be.to_numpy(coordinates).copy(),
        "sag": be.to_numpy(sag).copy(),
        "profile_x": be.to_numpy(profile_x).copy(),
        "profile_y": be.to_numpy(profile_y).copy(),
        "parameters": parameters,
    }
