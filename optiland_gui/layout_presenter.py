"""Install prepared numerical layout data exclusively on the GUI thread."""

from __future__ import annotations

import matplotlib
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon

from optiland.visualization.themes import get_active_theme
from optiland_gui import gui_plot_utils


def present_2d(viewer, data, context, restyle=False):
    """Reconstruct the 2D canvas and retain surface identities for highlighting."""
    gui_plot_utils.apply_gui_matplotlib_styles(viewer.current_theme)
    theme = get_active_theme().parameters
    viewer._is_plotting = True
    try:
        same_document = (
            getattr(viewer, "_scene_document_id", None) == context["document_id"]
        )
        preserve = same_document and (
            restyle or viewer._preserve_next or viewer._user_initiated_view_change
        )
        xlim, ylim = viewer.ax.get_xlim(), viewer.ax.get_ylim()
        if hasattr(viewer, "_finish_default_pan"):
            viewer._finish_default_pan()
        if hasattr(viewer, "clear_2d_highlights"):
            viewer.clear_2d_highlights()
        viewer.ax.clear()
        background = matplotlib.rcParams["figure.facecolor"]
        viewer.figure.set_facecolor(background)
        viewer.ax.set_facecolor(background)
        body_artists, surface_artists = {}, {}
        identities = context["surface_identities"]
        for item in data["primitives"]:
            owned = tuple(identities[i] for i in item["surfaces"])
            if item["kind"] == "polygon":
                patch = Polygon(
                    item["xy"],
                    closed=True,
                    facecolor=theme.get("lens.color", (0.8, 0.8, 0.8, 0.6)),
                    edgecolor=theme.get("axes.edgecolor", "gray"),
                    linewidth=item["linewidth"],
                )
                viewer.ax.add_patch(patch)
                body_artists[patch] = owned
            else:
                role = item["role"]
                if role == "ray":
                    cycle = theme.get("ray_cycle", ["C0", "C1", "C2"])
                    color = cycle[item["color_index"] % len(cycle)]
                else:
                    color = (
                        "black"
                        if role == "aperture"
                        else theme.get("axes.edgecolor", "gray")
                    )
                (line,) = viewer.ax.plot(
                    item["xy"][:, 0],
                    item["xy"][:, 1],
                    color=color,
                    linewidth=item["linewidth"],
                    linestyle=item["linestyle"],
                    label=item["label"],
                )
                if role == "surface" and len(owned) == 1:
                    surface_artists[line] = owned[0]
        for annotation in data["annotations"]:
            viewer.ax.annotate(
                "",
                xy=annotation["xy"],
                xytext=annotation["xytext"],
                arrowprops=annotation["arrowprops"],
            )
        viewer.ax.set_title(f"System: {data['name']} (2D)")
        viewer.ax.set_xlabel("Z-axis (mm)")
        viewer.ax.set_ylabel("Y-axis (mm)")
        viewer.ax.grid(True, linestyle="--", alpha=0.7)
        viewer.ax.autoscale_view()
        if preserve:
            viewer.ax.set_xlim(xlim)
            viewer.ax.set_ylim(ylim)
            viewer.ax.set_aspect("auto")
        else:
            viewer.ax.set_aspect("equal", adjustable="box")
        # The interaction feature provides this seam; its absence requires no
        # calculation fallback or dependence on the combined feature branch.
        if hasattr(viewer, "install_2d_highlight_bindings"):
            viewer.install_2d_highlight_bindings(
                body_artists,
                surface_artists,
                {identities[i]: xy for i, xy in data["boundaries"].items()},
                {identities[i]: xy for i, xy in data["references"].items()},
            )
        viewer.canvas.draw_idle()
        viewer._scene_document_id = context["document_id"]
    finally:
        viewer._is_plotting = False


def present_3d(viewer, data, context, restyle=False):
    """Install bulk polydata and preserve the user's existing camera."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    gui_plot_utils.apply_gui_matplotlib_styles(viewer.current_theme)
    theme = get_active_theme().parameters
    if not viewer._initialized:
        viewer.iren.Initialize()
        viewer._initialized = True
    viewer.renderer.SetBackground(*to_rgb(theme.get("axes.facecolor", "#202020")))
    if restyle and hasattr(viewer, "_scene_actor_specs"):
        for actor, mesh in viewer._scene_actor_specs:
            _style_3d_actor(actor, mesh, theme)
        viewer.vtkWidget.GetRenderWindow().Render()
        return
    actors = []
    for mesh in data["meshes"]:
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(mesh["points"], deep=True))
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        if mesh.get("normals") is not None:
            polydata.GetPointData().SetNormals(numpy_to_vtk(mesh["normals"], deep=True))
        for name, (offsets, connectivity) in mesh["cells"].items():
            cells = vtk.vtkCellArray()
            cells.SetData(
                numpy_to_vtkIdTypeArray(offsets, deep=True),
                numpy_to_vtkIdTypeArray(connectivity, deep=True),
            )
            {
                "polys": polydata.SetPolys,
                "lines": polydata.SetLines,
                "verts": polydata.SetVerts,
                "strips": polydata.SetStrips,
            }[name](cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        matrix = vtk.vtkMatrix4x4()
        matrix.DeepCopy(mesh["matrix"].ravel())
        actor.SetUserMatrix(matrix)
        _style_3d_actor(actor, mesh, theme)
        actors.append(actor)
    viewer._scene_actor_specs = list(zip(actors, data["meshes"], strict=True))
    viewer.renderer.RemoveAllViewProps()
    for actor in actors:
        viewer.renderer.AddActor(actor)
    if (
        not viewer._has_scene
        or getattr(viewer, "_scene_document_id", None) != context["document_id"]
    ):
        viewer.renderer.ResetCamera()
    viewer._has_scene = True
    viewer._scene_document_id = context["document_id"]
    viewer.renderer.ResetCameraClippingRange()
    viewer.vtkWidget.GetRenderWindow().Render()


def _style_3d_actor(actor, mesh, theme):
    prop = actor.GetProperty()
    if mesh["role"] == "ray":
        cycle = theme.get("ray_cycle")
        color = (
            to_rgb(cycle[mesh["color_index"] % len(cycle)]) if cycle else mesh["color"]
        )
    else:
        color = to_rgb(theme.get("lens.color", "white"))
    prop.SetColor(color)
    prop.SetOpacity(mesh["opacity"])
    prop.SetAmbient(mesh["ambient"])
    prop.SetDiffuse(mesh["diffuse"])
    prop.SetSpecular(mesh["specular"])
    prop.SetSpecularPower(mesh["power"])
    prop.SetLineWidth(mesh["linewidth"])


def present_sag(viewer, data, context, restyle=False):
    """Plot already computed sag values without calling surface geometry."""
    import numpy as np
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    gui_plot_utils.apply_gui_matplotlib_styles(viewer.current_theme)
    viewer.figure.clear()
    axes = viewer.figure.add_subplot(111)
    divider = make_axes_locatable(axes)
    bottom = divider.append_axes("bottom", size="25%", pad=0.25, sharex=axes)
    right = divider.append_axes("right", size="25%", pad=0.25, sharey=axes)
    color_axes = divider.append_axes("top", size="5%", pad=0.25)
    x, y = np.meshgrid(data["coordinates"], data["coordinates"])
    contours = axes.contourf(x, y, data["sag"], levels=50)
    viewer.figure.colorbar(contours, cax=color_axes, orientation="horizontal")
    parameters = data["parameters"]
    axes.axhline(parameters["y_cross_section"], color="red", linestyle="--")
    axes.axvline(parameters["x_cross_section"], color="blue", linestyle="--")
    axes.set_aspect("equal")
    axes.set_title(
        f"Surface S{parameters['surface_index']} | "
        f"View: ±{parameters['max_extent']:.2f} mm",
        pad=65,
    )
    axes.set_ylabel("Y-coordinate (mm)")
    bottom.plot(data["coordinates"], data["profile_x"], color="red")
    right.plot(data["profile_y"], data["coordinates"], color="blue")
    bottom.set_xlabel("X-coordinate (mm)")
    bottom.set_ylabel("Sag (z)")
    right.set_xlabel("Sag (z)")
    viewer.canvas.draw_idle()
