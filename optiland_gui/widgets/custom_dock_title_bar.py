"""Provides a custom title bar for QDockWidgets.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPalette
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QToolButton,
    QWidget,
)

from optiland_gui.resources import resources_rc  # noqa: F401


class CustomDockTitleBar(QWidget):
    """A custom title bar for :class:`~PySide6.QtWidgets.QDockWidget` instances.

    Provides float/dock and close buttons. Qt owns title-bar gestures
    for both docked and floating panels, including the drag threshold,
    docking preview and double-click behavior.

    Args:
        parent_dock: The dock widget that owns this title bar.
        title: Display text shown in the title bar label.
    """

    def __init__(self, parent_dock: QDockWidget, title: str = "") -> None:
        super().__init__(parent_dock)
        self.setObjectName("CustomDockTitleBar")
        self.dock_widget = parent_dock

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 6, 3)
        layout.setSpacing(2)

        self.title_label = QLabel(title)
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.title_label, 1)

        self.undock_btn = QToolButton(self)
        self.undock_btn.setObjectName("DockUndockButton")
        self.undock_btn.clicked.connect(self._toggle_floating)

        self.close_btn = QToolButton(self)
        self.close_btn.setObjectName("DockCloseButton")
        self.close_btn.setToolTip("Close panel")
        self.close_btn.setAccessibleName("Close panel")
        self.close_btn.setAccessibleDescription("Hide this panel from the workspace.")
        self.close_btn.clicked.connect(parent_dock.close)

        for button in (self.undock_btn, self.close_btn):
            button.setFixedSize(26, 26)
            button.setIconSize(QSize(16, 16))
            button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            layout.addWidget(button)

        parent_dock.topLevelChanged.connect(self._update_floating_state)
        parent_dock.featuresChanged.connect(self._update_features)
        parent_dock.windowTitleChanged.connect(self._update_title)
        self._update_title(title)
        self._update_floating_state(parent_dock.isFloating())
        self._update_features(parent_dock.features())
        self._update_theme_icons()

    def _update_title(self, title: str) -> None:
        """Keep the full panel title available when the header clips its label."""
        self.title_label.setText(title)
        self.setToolTip(title)

    def _update_floating_state(self, floating: bool) -> None:
        """Name the action this control will perform in the current state."""
        action = "Dock panel" if floating else "Float panel"
        self.undock_btn.setToolTip(action)
        self.undock_btn.setAccessibleName(action)
        self.undock_btn.setAccessibleDescription(
            "Return this panel to its docked location."
            if floating
            else "Move this panel to a separate window."
        )

    def _toggle_floating(self) -> None:
        """Keep the requested panel visible and its action reachable."""
        keep_focus = self.undock_btn.hasFocus()
        redocking = self.dock_widget.isFloating()
        self.dock_widget.setFloating(not redocking)
        if redocking:
            # Qt can return a dock behind the peer selected while it floated.
            self.dock_widget.raise_()
        if keep_focus:
            self.dock_widget.activateWindow()
            self.undock_btn.setFocus(Qt.FocusReason.OtherFocusReason)

    def _update_theme_icons(self) -> None:
        """Keep the bundled action glyphs legible when the theme changes."""
        background = self.palette().color(QPalette.ColorRole.Window)
        theme = "light" if background.lightness() > 128 else "dark"
        self.undock_btn.setIcon(QIcon(f":/icons/{theme}/maximize_restore.svg"))
        self.close_btn.setIcon(QIcon(f":/icons/{theme}/close.svg"))

    def changeEvent(self, event: QEvent) -> None:
        """Update glyph contrast with Qt's stylesheet-driven palette."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "close_btn"):
            self._update_theme_icons()

    def _update_features(self, features: QDockWidget.DockWidgetFeatures) -> None:
        """Expose only actions allowed by the owning dock widget."""
        self.undock_btn.setVisible(
            bool(features & QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        )
        self.close_btn.setVisible(
            bool(features & QDockWidget.DockWidgetFeature.DockWidgetClosable)
        )

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw the background according to the active stylesheet."""
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(
            QStyle.PrimitiveElement.PE_Widget, opt, painter, self
        )
        super().paintEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        """Pass title-bar presses to QDockWidget's gesture handler."""
        event.ignore()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        """Let Qt move the panel and calculate its docking preview together."""
        event.ignore()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        """Let Qt commit the previewed docking location on release."""
        event.ignore()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: ANN001
        """Let Qt toggle floating state while respecting dock feature flags."""
        event.ignore()
