"""Keep expensive retained analysis plots usable while dock sizes change."""

from __future__ import annotations

import time

from matplotlib.backend_bases import ResizeEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import QEvent, QTimer
from PySide6.QtGui import QImage, QPainter, QResizeEvent
from PySide6.QtWidgets import QWidget


class RetainedAnalysisCanvas(FigureCanvasQTAgg):
    """Coalesce only expensive resize renders, retaining a scaled finished frame.

    A measured draw of at least 50 ms enables resize-only deferral. Cheap plots
    continue using ordinary Matplotlib idle drawing. Figure size and public
    resize callbacks always update immediately; after 75 ms without resizing,
    render the final size. A 250 ms timer also requests periodic progress during
    sustained resizing. These are requested timer intervals, not latency bounds
    when the GUI thread is busy rendering.

    Explicit draws, scientific result changes and interaction redraws retain
    normal Matplotlib behavior. Plot input first finishes a pending retained
    frame so hit testing uses the current axes transforms and device pixels.
    """

    _EXPENSIVE_DRAW_MS = 50
    _RESIZE_QUIET_MS = 75
    _RESIZE_PROGRESS_MS = 250
    _PLOT_INPUT = frozenset(
        {
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.MouseButtonDblClick,
            QEvent.Type.MouseMove,
            QEvent.Type.Wheel,
            QEvent.Type.KeyPress,
        }
    )

    def __init__(self, figure):
        self._render_cost_ms = 0.0
        self._has_rendered = False
        self._handling_resize = False
        self._resize_frame = None
        super().__init__(figure)
        self._resize_quiet = QTimer(self)
        self._resize_quiet.setSingleShot(True)
        self._resize_quiet.setInterval(self._RESIZE_QUIET_MS)
        self._resize_quiet.timeout.connect(self.draw_idle)
        self._resize_progress = QTimer(self)
        self._resize_progress.setSingleShot(True)
        self._resize_progress.setInterval(self._RESIZE_PROGRESS_MS)
        self._resize_progress.timeout.connect(self.draw_idle)
        self._resize_followup = QTimer(self)
        self._resize_followup.setSingleShot(True)
        self._resize_followup.timeout.connect(self._apply_followup_resize)

    def _apply_followup_resize(self):
        """Apply the latest geometry after a resize callback resizes this widget."""
        self.resizeEvent(QResizeEvent(self.size(), self.size()))

    def _cancel_resize_frame(self):
        self._resize_quiet.stop()
        self._resize_progress.stop()
        self._resize_frame = None

    def draw(self):
        self._cancel_resize_frame()
        started = time.perf_counter()
        super().draw()
        self._render_cost_ms = 1000 * (time.perf_counter() - started)
        self._has_rendered = True

    def draw_idle(self):
        self._cancel_resize_frame()
        super().draw_idle()

    def print_figure(self, *args, **kwargs):
        had_resize_frame = self._resize_frame is not None
        self._cancel_resize_frame()
        try:
            return super().print_figure(*args, **kwargs)
        finally:
            if had_resize_frame:
                # Vector/alternate-DPI export restores the display canvas and
                # DPI, but may leave its old renderer until another paint event.
                self.draw_idle()
                self.update()

    def blit(self, bbox=None):
        if self._resize_frame is not None:
            self.draw()
        super().blit(bbox)

    def resizeEvent(self, event):
        if self.figure is None:
            return
        if self._handling_resize:
            self._resize_followup.start(0)
            return
        self._handling_resize = True
        try:
            defer = (
                self.isVisible()
                and self._has_rendered
                and self._render_cost_ms >= self._EXPENSIVE_DRAW_MS
            )
            if defer and self._resize_frame is None:
                rgba = self.buffer_rgba()
                # Copy the Agg memory: subsequent renders replace its storage.
                self._resize_frame = QImage(
                    rgba,
                    rgba.shape[1],
                    rgba.shape[0],
                    rgba.strides[0],
                    QImage.Format.Format_RGBA8888,
                ).copy()
            size = event.size()
            ratio = self.device_pixel_ratio
            self.figure.set_size_inches(
                size.width() * ratio / self.figure.dpi,
                size.height() * ratio / self.figure.dpi,
                forward=False,
            )
            QWidget.resizeEvent(self, event)
            if defer:
                self._resize_quiet.start()
                if not self._resize_progress.isActive():
                    self._resize_progress.start()
                self.update()
            else:
                self.draw_idle()
            # A callback may request a normal draw, cancelling the deferral.
            self.callbacks.process("resize_event", ResizeEvent("resize_event", self))
        finally:
            self._handling_resize = False

    def paintEvent(self, event):
        if self._resize_frame is None:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        try:
            painter.eraseRect(self.rect())
            painter.drawImage(self.rect(), self._resize_frame)
        finally:
            painter.end()

    def event(self, event):
        if self._resize_frame is not None and event.type() in self._PLOT_INPUT:
            self.draw()
        return super().event(event)

    def hideEvent(self, event):
        self._cancel_resize_frame()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.draw_idle()
