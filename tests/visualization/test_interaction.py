"""Unit tests for the visualization interaction module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.backend_bases import MouseEvent

from optiland.samples import CookeTriplet
from optiland.visualization.system.optic_viewer import OpticViewer
from optiland.visualization.system.surface import Surface2D


@pytest.fixture
def optic_viewer():
    """Returns an OpticViewer instance for a Cooke Triplet."""
    optic = CookeTriplet()
    viewer = OpticViewer(optic)
    original_view = viewer.view

    def view_wrapper():
        fig, ax, im = original_view()
        return fig, ax, im

    viewer.view = view_wrapper
    return viewer


def test_interaction_manager_initialization(optic_viewer):
    """Tests that the InteractionManager is initialized with the correct figure and axes."""
    fig, ax, im = optic_viewer.view()
    assert im.fig == fig
    assert im.ax == ax
    plt.close(fig)


def test_hover_highlight_and_tooltip(optic_viewer):
    """Tests that hovering over a surface highlights it and shows a tooltip."""
    fig, ax, im = optic_viewer.view()

    # Find a surface artist to hover over
    surface_artist = None
    for artist, obj in im.artist_registry.items():
        if isinstance(obj, Surface2D):
            surface_artist = artist
            break
    assert surface_artist is not None

    # Get the center of the artist
    x_data = surface_artist.get_xdata()
    y_data = surface_artist.get_ydata()
    x, y = x_data[len(x_data) // 2], y_data[len(y_data) // 2]
    x_pix, y_pix = ax.transData.transform((x, y))

    # Simulate a hover event
    event = MouseEvent("motion_notify_event", fig.canvas, x_pix, y_pix)
    event.inaxes = ax
    im.show_tooltip(surface_artist, event)

    # Check that the artist is highlighted
    assert surface_artist.get_linewidth() > 1

    # Check that the tooltip is visible and has the correct text
    assert im.tooltip.get_visible()
    assert "Surface" in im.tooltip.get_text()

    plt.close(fig)
