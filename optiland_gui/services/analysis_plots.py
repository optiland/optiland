"""Plain plot data for the finite set of two-dimensional GUI analyses.

The worker executes numerical ``view`` preparation. Only arrays, text and style
values cross the process boundary; no Figure, Artist or optical object does.
Unsupported artist families fail explicitly rather than silently dropping data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection, PathCollection, QuadMesh
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.colorizer import ColorizingArtist
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from matplotlib.text import Text
    from matplotlib.transforms import Transform


def _array(value: Any) -> np.ndarray:
    return np.array(np.ma.filled(value, np.nan), copy=True)


def _style(artist: Artist) -> dict:
    return {
        "alpha": artist.get_alpha(),
        "zorder": artist.get_zorder(),
        "label": artist.get_label(),
    }


def _text(text: Text, transform: Transform) -> dict:
    position = transform.inverted().transform(
        text.get_transform().transform(text.get_position())
    )
    return {
        "position": tuple(position),
        "text": text.get_text(),
        "fontsize": text.get_fontsize(),
        "ha": text.get_ha(),
        "va": text.get_va(),
        "rotation": text.get_rotation(),
    }


def _normalization(artist: ColorizingArtist) -> dict:
    norm = artist.norm
    if type(norm) not in (colors.Normalize, colors.LogNorm):
        raise ValueError(
            f"Unsupported analysis color normalization: {type(norm).__name__}"
        )
    return {
        "log": isinstance(norm, colors.LogNorm),
        "vmin": norm.vmin,
        "vmax": norm.vmax,
        "cmap": artist.get_cmap().name,
    }


def _legend(legend: Legend | None, transform: Transform) -> dict | None:
    if legend is None:
        return None
    handles = []
    for handle, label in zip(legend.legend_handles, legend.get_texts(), strict=True):
        style = {"label": label.get_text()}
        if isinstance(handle, Line2D):
            style.update(
                color=handle.get_color(),
                linestyle=handle.get_linestyle(),
                linewidth=handle.get_linewidth(),
                marker=handle.get_marker(),
                markersize=handle.get_markersize(),
            )
        elif isinstance(handle, PathCollection):
            style.update(color=handle.get_facecolors()[0], marker="o", linestyle="None")
        else:
            raise ValueError(f"Unsupported analysis legend: {type(handle).__name__}")
        handles.append(style)
    anchor = legend.get_bbox_to_anchor().transformed(transform.inverted())
    return {
        "handles": handles,
        "location": legend._loc,
        "anchor": tuple(anchor.bounds),
        "ncols": legend._ncols,
        "title": legend.get_title().get_text(),
    }


def capture_analysis_plot(figure: Figure) -> dict:
    """Extract the registered analyses' line, spot, pupil and image plot families."""
    figure.canvas.draw()
    result = {
        "figsize": tuple(figure.get_size_inches()),
        "axes": [],
        "texts": [_text(t, figure.transFigure) for t in figure.texts],
        "legends": [_legend(legend, figure.transFigure) for legend in figure.legends],
    }
    for ax in figure.axes:
        if ax.name != "rectilinear":
            raise ValueError("Analysis pages currently support 2D projections only.")
        axis = {
            "position": tuple(ax.get_position().bounds),
            "xlim": ax.get_xlim(),
            "ylim": ax.get_ylim(),
            "xscale": ax.get_xscale(),
            "yscale": ax.get_yscale(),
            "aspect": ax.get_aspect(),
            "title": ax.get_title(),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "visible": ax.axison,
            "grid": any(line.get_visible() for line in ax.get_xgridlines()),
            "lines": [],
            "patches": [],
            "collections": [],
            "images": [],
            "texts": [_text(t, ax.transAxes) for t in ax.texts],
            "legend": _legend(ax.get_legend(), ax.transAxes),
            "colorbar": hasattr(ax, "_colorbar"),
        }
        for line in ax.lines:
            if not line.get_visible():
                continue
            xy = ax.transData.inverted().transform(
                line.get_transform().transform(line.get_xydata())
            )
            axis["lines"].append(
                {
                    "xy": xy.copy(),
                    "color": line.get_color(),
                    "linewidth": line.get_linewidth(),
                    "linestyle": line.get_linestyle(),
                    "marker": line.get_marker(),
                    "markersize": line.get_markersize(),
                    **_style(line),
                }
            )
        for patch in ax.patches:
            if not patch.get_visible():
                continue
            path = patch.get_path().transformed(patch.get_transform() - ax.transData)
            axis["patches"].append(
                {
                    "vertices": path.vertices.copy(),
                    "codes": None if path.codes is None else path.codes.copy(),
                    "facecolor": patch.get_facecolor(),
                    "edgecolor": patch.get_edgecolor(),
                    "linewidth": patch.get_linewidth(),
                    "linestyle": patch.get_linestyle(),
                    **_style(patch),
                }
            )
        for collection in ax.collections:
            if not collection.get_visible():
                continue
            item = {
                **_style(collection),
                "facecolors": collection.get_facecolors().copy(),
                "edgecolors": collection.get_edgecolors().copy(),
                "linewidths": collection.get_linewidths().copy(),
            }
            if isinstance(collection, QuadMesh):
                item.update(
                    kind="mesh",
                    coordinates=collection.get_coordinates().copy(),
                    values=_array(collection.get_array()),
                    **_normalization(collection),
                )
            elif isinstance(collection, PathCollection):
                item.update(
                    kind="scatter",
                    offsets=_array(collection.get_offsets()),
                    sizes=collection.get_sizes().copy(),
                    paths=[
                        (p.vertices.copy(), None if p.codes is None else p.codes.copy())
                        for p in collection.get_paths()
                    ],
                )
            elif isinstance(collection, LineCollection):
                item.update(
                    kind="segments",
                    segments=[_array(s) for s in collection.get_segments()],
                    # get_linestyles() has already scaled dashes by linewidth;
                    # constructors require the original patterns to avoid scaling twice.
                    linestyles=collection._us_linestyles,
                )
            else:
                raise ValueError(
                    f"Unsupported analysis artist: {type(collection).__name__}"
                )
            axis["collections"].append(item)
        for im in ax.images:
            if not im.get_visible():
                continue
            axis["images"].append(
                {
                    "values": _array(im.get_array()),
                    "extent": tuple(im.get_extent()),
                    "origin": im.origin,
                    "interpolation": im.get_interpolation(),
                    **_normalization(im),
                    **_style(im),
                }
            )
        result["axes"].append(axis)
    return result


def _norm(item: dict) -> colors.Normalize:
    cls = colors.LogNorm if item["log"] else colors.Normalize
    return cls(vmin=item["vmin"], vmax=item["vmax"])


def draw_analysis_plot(figure: Figure, prepared: dict, theme: str = "dark") -> None:
    """Install detached geometry on the GUI-owned figure without optical work."""
    from optiland_gui.gui_plot_utils import apply_gui_matplotlib_styles

    apply_gui_matplotlib_styles(theme)
    figure.clear()
    dark = "dark" in theme.lower()
    foreground = "#dddddd" if dark else "#222222"
    background = "#1e1e1e" if dark else "white"
    figure.set_facecolor(background)
    for data in prepared["axes"]:
        ax = figure.add_axes(data["position"])
        ax.set_facecolor(background)
        for item in data["lines"]:
            style = {k: v for k, v in item.items() if k != "xy"}
            ax.plot(*item["xy"].T, **style)
        for item in data["patches"]:
            style = {k: v for k, v in item.items() if k not in ("vertices", "codes")}
            ax.add_patch(PathPatch(Path(item["vertices"], item["codes"]), **style))
        for item in data["collections"]:
            style = {
                k: item[k]
                for k in (
                    "alpha",
                    "zorder",
                    "label",
                    "facecolors",
                    "edgecolors",
                    "linewidths",
                )
            }
            if item["kind"] == "scatter":
                collection = PathCollection(
                    [Path(*p) for p in item["paths"]],
                    sizes=item["sizes"],
                    offsets=item["offsets"],
                    offset_transform=ax.transData,
                    **style,
                )
                # Scatter marker paths are in points; offsets are data coordinates.
                from matplotlib.transforms import IdentityTransform

                collection.set_transform(IdentityTransform())
                ax.add_collection(collection)
            elif item["kind"] == "segments":
                ax.add_collection(
                    LineCollection(
                        item["segments"], linestyles=item["linestyles"], **style
                    )
                )
            else:
                coords = item["coordinates"]
                ax.pcolormesh(
                    coords[..., 0],
                    coords[..., 1],
                    item["values"],
                    cmap=item["cmap"],
                    norm=_norm(item),
                    shading="flat",
                    alpha=item["alpha"],
                    zorder=item["zorder"],
                    label=item["label"],
                )
        for item in data["images"]:
            ax.imshow(
                item["values"],
                extent=item["extent"],
                origin=item["origin"],
                interpolation=item["interpolation"],
                cmap=item["cmap"],
                norm=_norm(item),
                alpha=item["alpha"],
                zorder=item["zorder"],
                label=item["label"],
            )
        for item in data["texts"]:
            style = {k: v for k, v in item.items() if k not in ("position", "text")}
            ax.text(
                *item["position"],
                item["text"],
                transform=ax.transAxes,
                color=foreground,
                **style,
            )
        ax.set(
            xscale=data["xscale"],
            yscale=data["yscale"],
            xlim=data["xlim"],
            ylim=data["ylim"],
            aspect=data["aspect"],
            title=data["title"],
            xlabel=data["xlabel"],
            ylabel=data["ylabel"],
        )
        ax.tick_params(colors=foreground)
        ax.xaxis.label.set_color(foreground)
        ax.yaxis.label.set_color(foreground)
        ax.title.set_color(foreground)
        if not data["visible"]:
            ax.set_axis_off()
        ax.grid(data["grid"])
        if data["legend"]:
            legend = data["legend"]
            ax.legend(
                handles=[Line2D([], [], **style) for style in legend["handles"]],
                loc=legend["location"],
                bbox_to_anchor=legend["anchor"],
                ncols=legend["ncols"],
                title=legend["title"],
            )
        if data["colorbar"]:
            ax.yaxis.set_ticks_position("right")
            ax.yaxis.set_label_position("right")
            ax.set_xticks([])
    for item in prepared["texts"]:
        style = {k: v for k, v in item.items() if k not in ("position", "text")}
        figure.text(*item["position"], item["text"], color=foreground, **style)
    for legend in prepared["legends"]:
        figure.legend(
            handles=[Line2D([], [], **style) for style in legend["handles"]],
            loc=legend["location"],
            bbox_to_anchor=legend["anchor"],
            ncols=legend["ncols"],
            title=legend["title"],
        )
