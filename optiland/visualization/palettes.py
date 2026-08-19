"""Palettes Module

This module defines color palettes for Optiland's visualization themes.
It provides a structured way to manage colors, ensuring consistency
and ease of customization. The palettes are designed to be aesthetically
pleasing and adaptable to different visualization contexts, such as 'light'
and 'dark' modes.

Each palette is a dictionary containing specific color definitions for various
plot elements, including background, axes, text, grid, lenses, and rays.
This modular approach allows for easy extension with new palettes and ensures
that themes can be built on a consistent color foundation.

Kramer Harrison, 2025
"""

# Base palette for the 'light' theme
from __future__ import annotations

light_palette = {
    "background": "#FFFFFF",
    "axis": "#333333",
    "text": "#333333",
    "grid": "#CCCCCC",
    "lens": "#E0E0E0",
    "edges": "#808080",
    "ray_cycle": [  # default Matplotlib color cycle
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
}

# Base palette for the 'dark' theme
dark_palette = {
    "background": "#1B1B1B",
    "axis": "#E6E6E6",
    "text": "#E6E6E6",
    "grid": "#3C3C3C",
    "lens": "#5E5E5E",
    "edges": "#CFCFCF",
    "ray_cycle": [
        "#2F89E2",  # brighter blue
        "#FFA340",  # warm orange
        "#34C759",  # Apple green
        "#FF4F4F",  # vibrant red
        "#A07FFF",  # lavender
        "#C47A54",  # warm brown
        "#FF7BD1",  # pink
        "#9A9A9A",  # neutral gray
        "#D1D64E",  # yellow-green
        "#25C8D8",  # cyan
    ],
}


# Solarized light palette
solarized_light_palette = {
    "background": "#fdf6e3",
    "axis": "#657b83",
    "text": "#586e75",
    "grid": "#eee8d5",
    "lens": "#93a1a1",
    "edges": "#657b83",
    "ray_cycle": [
        "#268bd2",
        "#2aa198",
        "#859900",
        "#d33682",
        "#cb4b16",
        "#6c71c4",
    ],
}

# Solarized dark palette
solarized_dark_palette = {
    "background": "#002b36",
    "axis": "#839496",
    "text": "#93a1a1",
    "grid": "#073642",
    "lens": "#586e75",
    "edges": "#839496",
    "ray_cycle": [
        "#2aa198",
        "#268bd2",
        "#859900",
        "#d33682",
        "#cb4b16",
        "#6c71c4",
    ],
}


midnight_palette = {
    "background": "#0B0C0F",
    "axis": "#DCDCDC",
    "text": "#DCDCDC",
    "grid": "#262626",
    "lens": "#404040",
    "edges": "#A0A0A0",
    "ray_cycle": [
        "#3DA5FF",
        "#FF9F43",
        "#5AFF6E",
        "#FF6464",
        "#C79AFF",
        "#FF8AC6",
        "#FFD666",
        "#00D2D2",
    ],
}
