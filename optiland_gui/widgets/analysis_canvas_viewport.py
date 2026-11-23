"""Scroll a native subplot grid when its labels need more room than the dock."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QSizePolicy

if TYPE_CHECKING:
    from .retained_analysis_canvas import RetainedAnalysisCanvas


class AnalysisCanvasViewport(QScrollArea):
    """Keep the dock shrinkable and leave plot sizing and export to Matplotlib."""

    def __init__(self, canvas: RetainedAnalysisCanvas, rows: int, columns: int):
        super().__init__()
        self.canvas = canvas
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Analysis plot viewport")
        self.setAccessibleDescription(
            "Focus this area to scroll with arrow or page keys."
        )
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setWidgetResizable(True)

        # Measure once per prepared figure in logical pixels. Native layout owns
        # every subsequent resize. Leave space for tick labels and a useful data
        # rectangle even when a cell has a short title or a multiline heading.
        renderer = canvas.get_renderer()
        axes = canvas.figure.axes
        em = max(renderer.points_to_pixels(ax.xaxis.label.get_size()) for ax in axes)
        titles = [ax.title.get_window_extent(renderer) for ax in axes]
        ylabels = [ax.yaxis.label.get_window_extent(renderer) for ax in axes]
        cell_width = max(12 * em, max(title.width for title in titles) + 5 * em)
        cell_height = max(title.height for title in titles) + max(
            7 * em, max(label.height for label in ylabels) + 3 * em
        )
        if any(ax.get_aspect() != "auto" for ax in axes):
            # Square data rectangles still need space for coordinate ticks in
            # both directions, including a multi-line defocus heading.
            cell_width = max(cell_width, 22 * em)
            cell_height = max(
                cell_height, max(title.height for title in titles) + 14 * em
            )
        width, height = columns * cell_width, rows * cell_height
        for legend in canvas.figure.legends:
            bounds = legend.get_window_extent(renderer)
            width = max(width, bounds.width + 2 * em)
            height += bounds.height + em
        ratio = canvas.device_pixel_ratio
        canvas.setMinimumSize(ceil(width / ratio), ceil(height / ratio))
        self.setWidget(canvas)
