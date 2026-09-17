"""Prepared analysis plots retain explanatory keys and optical plot styling."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from optiland_gui.services.analysis_plots import (
    capture_analysis_plot,
    draw_analysis_plot,
)


def figure_with_canvas():
    figure = Figure(figsize=(7, 5))
    FigureCanvasAgg(figure)
    return figure


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_figure_and_axes_legends_preserve_labels_columns_and_anchor(theme):
    original = figure_with_canvas()
    ax = original.subplots()
    ax.plot([0, 1], [0, 1], label="Short wavelength", color="blue")
    ax.scatter([0, 1], [1, 0], label="Long wavelength", color="orange")
    ax.legend(ncols=2, title="Axis key", loc="upper left")
    original.legend(
        ncols=2, title="Wavelengths", loc="lower center", bbox_to_anchor=(0.5, 0.02)
    )
    prepared = capture_analysis_plot(original)
    restored = figure_with_canvas()
    draw_analysis_plot(restored, prepared, theme)
    restored.canvas.draw()
    assert len(restored.legends) == 1
    for actual, expected in (
        (restored.legends[0], original.legends[0]),
        (restored.axes[0].get_legend(), ax.get_legend()),
    ):
        assert [t.get_text() for t in actual.get_texts()] == [
            t.get_text() for t in expected.get_texts()
        ]
        assert actual._ncols == 2
        assert actual.get_title().get_text() == expected.get_title().get_text()
        np.testing.assert_allclose(
            actual.get_bbox_to_anchor().bounds, expected.get_bbox_to_anchor().bounds
        )


def test_dashed_collections_image_opacity_and_patch_paths_survive_transfer():
    original = figure_with_canvas()
    ax = original.subplots()
    segments = [[[0, 0], [1, 1]], [[1, 1], [2, 0]]]
    lines = LineCollection(
        segments,
        linewidths=[2, 3],
        linestyles=["--", ":"],
        colors=["blue", "orange"],
        alpha=0.7,
    )
    ax.add_collection(lines)
    ax.add_patch(Circle((0.5, 0.5), 0.2, fill=False, linestyle="--"))
    image = ax.imshow([[0, 1], [2, 3]], alpha=0.4, zorder=5, label="Intensity")
    hidden = ax.imshow([[0, 1], [2, 3]])
    hidden.set_visible(False)
    prepared = capture_analysis_plot(original)
    restored = figure_with_canvas()
    draw_analysis_plot(restored, prepared)
    actual = restored.axes[0]
    assert len(actual.images) == 1
    assert actual.images[0].get_alpha() == image.get_alpha()
    assert actual.images[0].get_zorder() == image.get_zorder()
    assert actual.images[0].get_label() == image.get_label()
    for actual_style, expected in zip(
        actual.collections[0].get_linestyles(), lines.get_linestyles(), strict=True
    ):
        assert actual_style[0] == expected[0]
        np.testing.assert_allclose(actual_style[1], expected[1])
    np.testing.assert_allclose(actual.collections[0].get_segments(), segments)
    assert actual.collections[0].get_alpha() == 0.7
    expected_path = (
        ax.patches[0]
        .get_path()
        .transformed(ax.patches[0].get_transform() - ax.transData)
    )
    np.testing.assert_allclose(
        actual.patches[0].get_path().vertices, expected_path.vertices
    )


def test_quadmesh_retains_normalization_and_transparency():
    original = figure_with_canvas()
    ax = original.subplots()
    mesh = ax.pcolormesh([[1, 2], [3, 4]], alpha=0.6, zorder=4, cmap="viridis")
    prepared = capture_analysis_plot(original)
    restored = figure_with_canvas()
    draw_analysis_plot(restored, prepared)
    actual = restored.axes[0].collections[0]
    assert actual.get_alpha() == mesh.get_alpha()
    assert actual.get_zorder() == mesh.get_zorder()
    np.testing.assert_allclose(actual.get_array(), mesh.get_array())
    assert actual.norm.vmin == mesh.norm.vmin and actual.norm.vmax == mesh.norm.vmax


def test_annotations_survive_and_hidden_artists_remain_hidden():
    original = figure_with_canvas()
    ax = original.subplots()
    ax.plot([0, 1], [0, 1], visible=False)
    patch = Circle((0, 0), 1, visible=False)
    ax.add_patch(patch)
    ax.scatter([0], [0], visible=False)
    ax.text(0.2, 0.7, "Reference focus", transform=ax.transAxes)
    ax.set_axis_off()
    restored = figure_with_canvas()
    draw_analysis_plot(restored, capture_analysis_plot(original))
    actual = restored.axes[0]
    assert not actual.lines and not actual.collections and not actual.patches
    assert not actual.axison
    assert actual.texts[0].get_text() == "Reference focus"
    np.testing.assert_allclose(actual.texts[0].get_position(), (0.2, 0.7))


def test_unsupported_legend_is_reported():
    original = figure_with_canvas()
    ax = original.subplots()
    ax.add_patch(Circle((0, 0), 1, label="Unsupported filled legend"))
    ax.legend()
    with pytest.raises(ValueError, match="Unsupported analysis legend"):
        capture_analysis_plot(original)


@pytest.mark.parametrize("unsupported", ["projection", "collection", "normalization"])
def test_unsupported_plot_payload_fails_instead_of_silently_dropping_content(
    unsupported,
):
    original = figure_with_canvas()
    ax = original.add_subplot(
        projection="polar" if unsupported == "projection" else None
    )
    if unsupported == "collection":
        ax.add_collection(PolyCollection([[(0, 0), (1, 0), (0, 1)]]))
    elif unsupported == "normalization":
        ax.imshow([[1, 2], [3, 4]], norm=BoundaryNorm([0, 2, 4], 2))
    with pytest.raises(ValueError, match="2D projections|Unsupported analysis"):
        capture_analysis_plot(original)
