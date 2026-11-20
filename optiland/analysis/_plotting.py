"""Shared figure handling for analysis ``view()`` methods.

An analysis either draws on a figure the GUI passes in, or creates its own.
These helpers set up the single axes for both cases and finish the figure.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def prepare_axes(
    fig_to_plot_on: Figure | None, figsize: tuple[float, float]
) -> tuple[Figure, Axes, bool]:
    """Returns a figure with one cleared axes to draw on.

    Args:
        fig_to_plot_on: A figure to draw on (GUI embedding), or None to create
            a new figure.
        figsize: The size of a new figure.

    Returns:
        tuple: The figure, its axes, and whether the figure is embedded.
    """
    is_gui_embedding = fig_to_plot_on is not None
    if is_gui_embedding:
        current_fig = fig_to_plot_on
        current_fig.clear()
        ax = current_fig.add_subplot(111)
    else:
        current_fig, ax = plt.subplots(figsize=figsize)
    return current_fig, ax, is_gui_embedding


def finalize_figure(current_fig: Figure, is_gui_embedding: bool, show: bool) -> None:
    """Redraws an embedded figure, or shows a new one when requested.

    Args:
        current_fig: The figure to finish.
        is_gui_embedding: Whether the figure is embedded in the GUI.
        show: Whether to call ``plt.show()`` for a figure that is not embedded.
    """
    if is_gui_embedding and hasattr(current_fig, "canvas"):
        current_fig.canvas.draw_idle()
    if show and not is_gui_embedding:
        plt.show()
