"""Separate, read-only presentation of an optimizer's prepared candidate scene."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QWidget

from .layout_presenter import present_2d


class OptimizationPreview(QDialog):
    """Candidate-only canvas that never replaces the editable document's scene."""

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Optimization candidate preview")
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Calculation candidate — current document remains editable.")
        )
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self._is_plotting = False
        self.preserve_zoom = True
        self._preserve_next = True
        self._user_initiated_view_change = False
        self.current_theme = "dark"
        self._data = None
        self._context = None

    def install(self, data: dict, context: dict) -> None:
        self._data, self._context = data, context
        present_2d(self, data, context)

    def update_theme(self, theme_name: str) -> None:
        self.current_theme = theme_name
        if self._data is not None:
            present_2d(self, self._data, self._context, restyle=True)
