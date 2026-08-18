"""MediumIR -- data-only description of an optical medium.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MediumIR:
    """A single optical medium, as plain data.

    ``n_model`` (and the reserved ``k_model``, D-13 / PR8) describe *how* to
    evaluate dispersion without embedding a live evaluator: a dict tagged by
    ``"kind"`` rather than a bound method, so the description survives a JSON
    round-trip and a non-Python backend can interpret it.

    Two ``n_model`` kinds are supported in this revamp:

    - ``{"kind": "constant", "n": <float>}`` -- wavelength-independent index
      (vacuum is ``n=1.0``).
    - ``{"kind": "catalog", "name": <str>}`` -- a glass catalog name (e.g.
      ``"N-BK7"``), resolved via :meth:`NSQMaterial.from_glass` the same way
      :mod:`optiland.nonsequential.serialization` already round-trips
      materials.

    A custom dispersion model with no catalog name cannot be lowered losslessly
    and raises at lowering time (see :func:`~optiland.nonsequential.ir.lower.lower`)
    rather than silently dropping to a constant approximation.

    Attributes:
        id: Index into ``SceneIR.media``.
        name: Human-readable label (glass name, or ``"vacuum"``).
        n_model: Refractive-index model descriptor, as above.
        k_model: Extinction-coefficient model descriptor. ``None`` means
            non-absorbing. Reserved for Beer-Lambert absorption (D-13); no
            kind is populated by this revamp's lowering yet.
    """

    id: int
    name: str
    n_model: dict[str, Any]
    k_model: dict[str, Any] | None = field(default=None)
