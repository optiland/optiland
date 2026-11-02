"""Allow measured plot overflow without enlarging an analysis dock."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import QFrame, QScrollArea, QSizePolicy

from .analysis_figure_layout import AnalysisFigureLayout

if TYPE_CHECKING:
    from matplotlib.backend_bases import DrawEvent
    from PySide6.QtGui import QCloseEvent, QResizeEvent
    from PySide6.QtWidgets import QWidget

    from .retained_analysis_canvas import RetainedAnalysisCanvas


class AnalysisCanvasViewport(QScrollArea):
    """Scroll only when a compatible grid cannot fit its scientific labels.

    The canvas retains plot-wheel zoom and normal export of the full figure.
    The visible scrollbar provides access to overflowing fields and legends.
    """

    _minimum_changed = Signal()

    def __init__(
        self, canvas: RetainedAnalysisCanvas, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.canvas = canvas
        self._outer_size = (-1, -1)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAccessibleName("Analysis plot viewport")
        self.setAccessibleDescription(
            "Use the scrollbars or focus this viewport and use arrow or page keys "
            "to inspect overflowing fields. The plot retains wheel zoom."
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setWidget(canvas)
        self._minimum_changed.connect(
            self._apply_minimum, Qt.ConnectionType.QueuedConnection
        )
        self._draw_cid = canvas.mpl_connect("draw_event", self._rendered)

    def _rendered(self, event: DrawEvent) -> None:
        if event.canvas is not self.canvas or self.canvas.is_saving():
            return
        # Applying child geometry inside draw_event can suppress the final
        # normal Matplotlib idle draw while it is already drawing. Queue the
        # Qt geometry change after the completed renderer has been published.
        self._minimum_changed.emit()

    @Slot()
    def _apply_minimum(self) -> None:
        if not self.isVisible() or self.canvas.is_saving():
            return
        engine = self.canvas.figure.get_layout_engine()
        width = height = 0
        if isinstance(engine, AnalysisFigureLayout):
            height = ceil(
                engine.minimum_height_inches
                * self.canvas.figure.dpi
                / self.canvas.device_pixel_ratio
            )
            width = ceil(
                engine.minimum_width_inches
                * self.canvas.figure.dpi
                / self.canvas.device_pixel_ratio
            )
        if width != self.canvas.minimumWidth() or height != self.canvas.minimumHeight():
            self.canvas.setMinimumSize(width, height)

    def resizeEvent(self, event: QResizeEvent) -> None:
        # A changed viewport gets a fresh opportunity to fit. Changing only
        # the child minimum does not resize this ignored-size-hint viewport.
        size = (self.width(), self.height())
        if size != self._outer_size:
            self._outer_size = size
            self.canvas.setMinimumSize(0, 0)
            engine = self.canvas.figure.get_layout_engine()
            if isinstance(engine, AnalysisFigureLayout):
                engine.minimum_height_inches = 0.0
                engine.minimum_width_inches = 0.0
        super().resizeEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.canvas.mpl_disconnect(self._draw_cid)
        self.canvas.close()
        super().closeEvent(event)
