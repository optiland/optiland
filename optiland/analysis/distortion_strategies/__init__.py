"""Pluggable strategies for distortion analysis.

This subpackage decomposes distortion computation into small, single-purpose,
interchangeable strategies so that distortion can be evaluated for classical
(paraxial, rotationally symmetric) systems as well as for off-axis, freeform,
or obscured systems where no chief ray can be traced and no scalar focal length
exists.

Two orthogonal abstractions are provided:

* :class:`ReferencePointStrategy` — locates the per-field image reference point
  (the "chief-ray replacement"). :class:`ChiefRayReferencePoint` traces a single
  chief ray (classical behavior); :class:`CentroidReferencePoint` traces a pupil
  bundle and returns the transmitted-energy centroid, which is obscuration-proof
  and works when no chief ray exists.

* :class:`DistortionModel` — defines the distortion-free reference mapping and the
  residual. :class:`RotationalDistortionModel` reproduces the classical
  ``f-tan`` / ``f-theta`` model; :class:`AffineDistortionModel` implements the
  axis-free best-fit affine (plate-scale).

The :func:`create_distortion_model` factory wires these together from a simple
``method`` string, while still allowing fully custom strategy objects to be
injected (dependency inversion).

Kramer Harrison, 2026
"""

from __future__ import annotations

from .model import (
    AffineDistortionModel,
    DistortionModel,
    DistortionResult,
    RotationalDistortionModel,
    create_distortion_model,
)
from .reference_point import (
    CentroidReferencePoint,
    ChiefRayReferencePoint,
    ReferencePointStrategy,
)

__all__ = [
    "AffineDistortionModel",
    "CentroidReferencePoint",
    "ChiefRayReferencePoint",
    "DistortionModel",
    "DistortionResult",
    "ReferencePointStrategy",
    "RotationalDistortionModel",
    "create_distortion_model",
]
