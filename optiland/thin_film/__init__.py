"""Thin-film multilayer calculations (Transfer Matrix Method).

Public API:
- ``Layer``: one thin-film layer (material + thickness)
- ``ThinFilmStack``: stack structure and TMM computations (r, t, R, T, A)

Units: wavelength in µm, thickness in µm (nm helpers), AOI in radians (deg helpers).
"""

from __future__ import annotations

# flake8: noqa

import importlib

from .analysis import SpectralAnalyzer
from .layer import Layer
from .stack import ThinFilmStack


def __getattr__(name):
    if name == "optimization":
        return importlib.import_module(".optimization", __name__)
    if name == "tolerancing":
        return importlib.import_module(".tolerancing", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SpectralAnalyzer",
    "Layer",
    "ThinFilmStack",
    "optimization",
    "tolerancing",
]
