"""Configuration constants for the Optiland GUI.

This module holds shared constants (file paths and application settings)
to avoid circular import issues between other modules.
"""

from __future__ import annotations

import os

# --- Theme and Style Paths ---
GUI_BASE_DIR = os.path.dirname(__file__)
RESOURCES_DIR = os.path.join(GUI_BASE_DIR, "resources")
STYLES_DIR = os.path.join(RESOURCES_DIR, "styles")

THEME_DARK_PATH = os.path.join(STYLES_DIR, "dark_theme.qss")
THEME_LIGHT_PATH = os.path.join(STYLES_DIR, "light_theme.qss")
SIDEBAR_QSS_PATH = os.path.join(STYLES_DIR, "sidebar.qss")
ICONS_DIR = os.path.join(RESOURCES_DIR, "icons")
OPTILAND_ICON_PATH = ":/icons/optiland_icon.png"


# --- Application Info ---
ORGANIZATION_NAME = "OptilandProject"
APPLICATION_NAME = "OptilandGUI"
