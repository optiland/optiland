"""Matplotlib theme utilities for the Optiland GUI.

Provides :func:`apply_plot_theme` which updates ``matplotlib.rcParams`` to
match the application's dark or light theme whenever the user switches themes.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import matplotlib

_DARK_RCPARAMS: dict[str, object] = {
    "axes.facecolor": "#1E1E1E",
    "figure.facecolor": "#202020",
    "axes.edgecolor": "#555555",
    "axes.labelcolor": "#E0E0E0",
    "xtick.color": "#BBBBBB",
    "ytick.color": "#BBBBBB",
    "text.color": "#E0E0E0",
    "grid.color": "#3A3A3A",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    # App accent palette for default colour cycle
    "axes.prop_cycle": matplotlib.cycler(
        color=["#007ACC", "#FF6B35", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]
    ),
    "image.cmap": "inferno",
}

_LIGHT_RCPARAMS: dict[str, object] = {
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#F5F5F5",
    "axes.edgecolor": "#CCCCCC",
    "axes.labelcolor": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "text.color": "#333333",
    "grid.color": "#DDDDDD",
    "grid.linestyle": "--",
    "grid.alpha": 0.8,
    "axes.prop_cycle": matplotlib.cycler(
        color=["#007ACC", "#FF6B35", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]
    ),
    "image.cmap": "viridis",
}


def apply_plot_theme(is_dark: bool) -> None:
    """Update ``matplotlib.rcParams`` to match the current app theme.

    This function applies a conservative set of overrides so that embedded
    plots remain readable in both dark and light mode.  It should be called
    whenever the user switches themes.

    Args:
        is_dark: ``True`` for dark theme, ``False`` for light theme.
    """
    params = _DARK_RCPARAMS if is_dark else _LIGHT_RCPARAMS
    valid = {k: v for k, v in params.items() if k in matplotlib.rcParams}
    matplotlib.rcParams.update(valid)
