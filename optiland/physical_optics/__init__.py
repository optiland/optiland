"""Scalar physical-optics field models and propagation algorithms."""

from __future__ import annotations

from .diagnostics import (
    BoundaryDiagnostic,
    WraparoundDiagnostic,
    boundary_diagnostic,
    wraparound_diagnostic,
)
from .field import ScalarField, gaussian_field
from .gaussian_beam import GaussianBeam
from .propagation import angular_spectrum

__all__ = [
    "BoundaryDiagnostic",
    "GaussianBeam",
    "ScalarField",
    "WraparoundDiagnostic",
    "angular_spectrum",
    "boundary_diagnostic",
    "gaussian_field",
    "wraparound_diagnostic",
]
