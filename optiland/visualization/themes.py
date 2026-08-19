"""Visualization Themes Module

This module provides a framework for managing visualization themes in Optiland.
It allows users to switch between different visual styles for plots, ensuring
a consistent and customizable appearance. The framework is designed to be
extensible, allowing for the creation of new themes and the modification of
existing ones.

The core components of this module are:
- The `Theme` class, which encapsulates all styling parameters.
- A registry of predefined themes, such as 'light' and 'dark'.
- A global state that holds the currently active theme.
- Functions to set the global theme and to use themes temporarily via a
  context manager.

This system is inspired by matplotlib's rcParams, but with a more
structured, theme-based approach.

Kramer Harrison, 2025
"""

from __future__ import annotations

import contextlib
import copy

from optiland.visualization.palettes import (
    dark_palette,
    light_palette,
    midnight_palette,
    solarized_dark_palette,
    solarized_light_palette,
)


class Theme:
    """A class that defines the visual parameters for Optiland plots."""

    def __init__(self, name: str, description: str, palette: dict, **kwargs):
        self.name = name
        self.description = description
        self.palette = palette
        self.parameters = {
            "figure.figsize": (10, 4),
            "figure.dpi": 100,
            "figure.facecolor": palette["background"],
            "axes.facecolor": palette["background"],
            "axes.edgecolor": palette["edges"],
            "axes.labelcolor": palette["text"],
            "xtick.color": palette["axis"],
            "ytick.color": palette["axis"],
            "grid.color": palette["grid"],
            "grid.alpha": 0.25,
            "text.color": palette["text"],
            "lens.color": palette["lens"],
            "ray_cycle": palette["ray_cycle"],
            "font.family": "sans-serif",
            "font.size": 10,
            "font.weight": "normal",
        }
        self.parameters.update(kwargs)

    def to_dict(self):
        """Return a dictionary representation of the theme."""
        return {
            "name": self.name,
            "description": self.description,
            "palette": self.palette,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, theme_dict: dict) -> Theme:
        """Create a Theme object from a dictionary."""
        return cls(
            name=theme_dict["name"],
            description=theme_dict["description"],
            palette=theme_dict["palette"],
            **theme_dict["parameters"],
        )


# Predefined themes
_themes = {
    "light": Theme(
        "light",
        "A theme with a light background, suitable for presentations.",
        light_palette,
    ),
    "dark": Theme(
        "dark",
        "A theme with a dark background, suitable for screen viewing.",
        dark_palette,
    ),
    "solarized_light": Theme(
        "solarized_light",
        "A light theme based on the Solarized color palette.",
        solarized_light_palette,
    ),
    "solarized_dark": Theme(
        "solarized_dark",
        "A dark theme based on the Solarized color palette.",
        solarized_dark_palette,
    ),
    "midnight": Theme(
        "midnight",
        "A dark theme with vibrant colors for better visibility.",
        midnight_palette,
    ),
}

# Global state
_active_theme = _themes["light"]


def get_active_theme() -> Theme:
    """Return a copy of the active theme."""
    return copy.deepcopy(_active_theme)


def list_themes() -> list[str]:
    """Return a list of available theme names."""
    return list(_themes.keys())


def register_theme(name: str, theme: Theme):
    """Register a new theme.

    Args:
        name: The name of the theme.
        theme: The Theme object to register.

    Raises:
        ValueError: If a theme with the same name already exists.
    """
    if name in _themes:
        raise ValueError(f"A theme with the name '{name}' already exists.")
    _themes[name] = theme


def set_theme(theme: str | Theme):
    """Set the global theme for all plots.

    Args:
        theme: The theme to set. Can be a string identifier (e.g., 'light',
            'dark', 'solarized_light', 'solarized_dark') or a Theme object.

    Raises:
        ValueError: If the theme identifier is not found.
        TypeError: If the theme is not a string or a Theme object.
    """
    global _active_theme
    if isinstance(theme, str):
        if theme not in _themes:
            raise ValueError(f"Theme '{theme}' not found.")
        _active_theme = _themes[theme]
    elif isinstance(theme, Theme):
        _active_theme = theme
    else:
        raise TypeError("theme must be a string or a Theme object.")


@contextlib.contextmanager
def theme_context(theme: str | Theme):
    """A context manager to temporarily set the theme.

    This is useful for applying a theme to a specific block of code without
    changing the global theme.

    Args:
        theme: The theme to set. Can be a string identifier (e.g., 'light',
            'dark', 'solarized_light', 'solarized_dark') or a Theme object.
    """
    global _active_theme
    original_theme = _active_theme
    try:
        set_theme(theme)
        yield
    finally:
        _active_theme = original_theme
