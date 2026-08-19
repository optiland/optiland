"""
Wavefront Modules.

This subpackage contains modules for wavefront analysis, including OPD (Optical
Path Difference) calculations, Zernike polynomial fitting, and related data
structures.

Public API:
- `Wavefront`: The main class for wavefront analysis.
- `WavefrontData`: A data container for wavefront results.
- `OPD`: A class for calculating and visualizing OPD maps.
- `OPDFan`: A class for creating OPD fan plots.
- `ZernikeOPD`: A class for Zernike decomposition of OPD data.
"""

from __future__ import annotations

from .opd import OPD
from .opd_fan import OPDFan
from .reference_geometry import PlanarReference, ReferenceGeometry, SphericalReference
from .wavefront import Wavefront
from .wavefront_data import WavefrontData
from .zernike_opd import ZernikeOPD

__all__ = [
    "Wavefront",
    "WavefrontData",
    "OPD",
    "OPDFan",
    "ZernikeOPD",
    "ReferenceGeometry",
    "SphericalReference",
    "PlanarReference",
]
