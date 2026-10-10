"""Consistent direct mouse navigation for the GUI's Matplotlib 2D plots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backend_bases import MouseButton, cursors
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QEvent, Qt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QWidget


def valid_plot_view(x: tuple[float, float], y: tuple[float, float]) -> bool:
    """Reject nonfinite or collapsed limits in axis-scaled coordinates."""
    return all(
        all(math.isfinite(v) and abs(v) < 1e100 for v in limits)
        and abs(limits[1] - limits[0])
        > max(abs(limits[0]), abs(limits[1]), 1.0) * 1e-12
        for limits in (x, y)
    )


@dataclass
class _PointerPan:
    views: dict[Axes, tuple[tuple[float, float], tuple[float, float]]]
    button: MouseButton
    owns_lock: bool


class PlotNavigationToolbar(NavigationToolbar2QT):
    """Right/middle drag pans; wheel zooms independently of the selected tool.

    The explicit Pan and Zoom to rectangle tools retain left-drag interaction.
    Navigation uses Matplotlib's axis transforms, including logarithmic axes.
    The toolbar's standard file actions can be filtered by individual panels.
    """

    def __init__(
        self, canvas: FigureCanvasQTAgg, parent: QWidget, coordinates: bool = True
    ) -> None:
        self._pointer_pan: _PointerPan | None = None
        super().__init__(canvas, parent, coordinates=coordinates)
        self._actions["pan"].setToolTip(
            "Pan with left-drag. Right-drag and middle-drag always pan."
        )
        canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        canvas.installEventFilter(self)
        canvas.mpl_connect("scroll_event", self._on_scroll)
        canvas.setToolTip(
            "Right-drag or middle-drag to pan; scroll to zoom at the pointer. "
            "Pan and Zoom to rectangle tools use left-drag. Escape exits the tool."
        )

    @property
    def is_dragging(self) -> bool:
        return any((self._pointer_pan, self._pan_info, self._zoom_info))

    def _remember_view(self) -> None:
        """Record a baseline after programmatic view changes without duplicates."""
        current = self._nav_stack()
        if current is None or any(
            ax not in current or current[ax][0] != ax._get_view()
            for ax in self.canvas.figure.axes
            if ax.name != "3d"
        ):
            self.push_current()

    def _zoom_pan_handler(self, event: MouseEvent) -> None:
        if self._pointer_pan is not None:
            if (
                event.name == "button_release_event"
                and event.button == self._pointer_pan.button
            ):
                self.finish_navigation()
            return
        if event.name == "button_press_event" and event.button in (
            MouseButton.RIGHT,
            MouseButton.MIDDLE,
        ):
            axes = event.inaxes
            if axes is not None and axes.name == "3d":
                super()._zoom_pan_handler(event)
                return
            if (
                self.is_dragging
                or axes is None
                or not axes.get_navigate()
                or not axes.can_pan()
                or not self.canvas.widgetlock.available(self)
            ):
                return
            self._remember_view()
            targets = self._start_event_axes_interaction(event, method="pan")
            views = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in targets}
            owns_lock = not self.canvas.widgetlock.isowner(self)
            self.canvas.widgetlock(self)
            for ax in targets:
                ax.start_pan(event.x, event.y, MouseButton.LEFT)
            self._pointer_pan = _PointerPan(
                views,
                event.button,
                owns_lock,
            )
            self._last_cursor = cursors.MOVE
            self.canvas.set_cursor(cursors.MOVE)
            return
        # Right/middle release must not complete an unrelated left-button tool.
        if event.button == MouseButton.LEFT or (
            event.inaxes is not None and event.inaxes.name == "3d"
        ):
            if (
                event.name == "button_press_event"
                and self.mode
                and not self.is_dragging
            ):
                self._remember_view()
            super()._zoom_pan_handler(event)

    def mouse_move(self, event: MouseEvent) -> None:
        pan = self._pointer_pan
        if pan is None:
            super().mouse_move(event)
            return
        if event.buttons != {pan.button} or not self.canvas.widgetlock.isowner(self):
            self.finish_navigation()
            return
        for ax in pan.views:
            # Use the frozen press-time transform and native axis constraints.
            ax.drag_pan(MouseButton.LEFT, event.key, event.x, event.y)
        self.canvas.draw_idle()

    def finish_navigation(self) -> None:
        """End gestures before focus changes, scene replacement or tool changes."""
        if self._pan_info is not None:
            self.release_pan(None)
        zoom, self._zoom_info = self._zoom_info, None
        if zoom is not None:
            # Cancel without applying a rectangle or relying on Matplotlib's
            # version-specific private zoom-cleanup helper.
            self.canvas.mpl_disconnect(zoom.cid)
            self.remove_rubberband()
            self.canvas.draw_idle()
        pan, self._pointer_pan = self._pointer_pan, None
        if pan is None:
            return
        for ax in pan.views:
            ax.end_pan()
        if any(
            (ax.get_xlim(), ax.get_ylim()) != view for ax, view in pan.views.items()
        ):
            self.push_current()
        if pan.owns_lock and self.canvas.widgetlock.isowner(self):
            self.canvas.widgetlock.release(self)
        self._last_cursor = cursors.POINTER
        self.canvas.set_cursor(cursors.POINTER)

    def pan(self, *args: object) -> None:
        self.finish_navigation()
        super().pan(*args)

    def zoom(self, *args: object) -> None:
        self.finish_navigation()
        super().zoom(*args)

    def update(self) -> None:
        self.finish_navigation()
        super().update()

    def home(self, *args: object) -> None:
        self.finish_navigation()
        super().home(*args)

    def back(self, *args: object) -> None:
        self.finish_navigation()
        super().back(*args)

    def forward(self, *args: object) -> None:
        self.finish_navigation()
        super().forward(*args)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.canvas and (
            event.type()
            in (QEvent.Type.FocusOut, QEvent.Type.Hide, QEvent.Type.WindowDeactivate)
            or (
                event.type() == QEvent.Type.KeyPress
                and event.key() == Qt.Key.Key_Escape
            )
        ):
            handled = self.is_dragging or bool(self.mode)
            self.finish_navigation()
            if event.type() == QEvent.Type.KeyPress:
                if self._actions["pan"].isChecked():
                    self.pan()
                elif self._actions["zoom"].isChecked():
                    self.zoom()
                if handled:
                    return True
        return super().eventFilter(watched, event)

    def _on_scroll(self, event: MouseEvent) -> None:
        axes = event.inaxes
        if (
            self.is_dragging
            or axes is None
            or axes.name == "3d"
            or not axes.get_navigate()
            or not axes.can_zoom()
            or not self.canvas.widgetlock.available(self)
            or event.xdata is None
            or event.ydata is None
            or not all(math.isfinite(v) for v in (event.xdata, event.ydata, event.step))
            or event.step == 0
        ):
            return
        scale = 1.2 ** -max(-10, min(10, event.step))
        bounds = []
        scaled_bounds = []
        for axis, limits, anchor in (
            (axes.xaxis, axes.get_xlim(), event.xdata),
            (axes.yaxis, axes.get_ylim(), event.ydata),
        ):
            transform = axis.get_transform()
            lower, upper, center = transform.transform([*limits, anchor])
            new = center + (np.array([lower, upper]) - center) * scale
            # Extreme wheel events must not overflow log transforms or collapse
            # a view. A rejected gesture leaves limits and history untouched.
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                raw = transform.inverted().transform(new)
            if not all(math.isfinite(v) and abs(v) < 1e100 for v in raw):
                return
            if axis.get_scale() == "log" and min(raw) <= 0:
                return
            bounds.append(raw)
            scaled_bounds.append(new)
        if not valid_plot_view(*scaled_bounds):
            return
        self._remember_view()
        axes.set_xlim(bounds[0])
        axes.set_ylim(bounds[1])
        self.push_current()
        self.canvas.draw_idle()
