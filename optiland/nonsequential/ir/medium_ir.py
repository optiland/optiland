"""MediumIR -- data-only description of an optical medium.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MediumIR:
    """A single optical medium, as plain data.

    ``n_model``/``k_model`` describe *how* to evaluate dispersion/absorption
    without embedding a live evaluator: a dict tagged by ``"kind"`` rather
    than a bound method, so the description survives a JSON round-trip and a
    non-Python backend can interpret it. Both use the same three kinds and
    are always populated together by :func:`~optiland.nonsequential.ir.lower.lower`:

    - ``{"kind": "constant", "n": <float>}`` / ``{"kind": "constant", "k":
      <float>}`` -- wavelength-independent (vacuum is ``n=1.0``, ``k=0.0``).
    - ``{"kind": "catalog", "name": <str>}`` -- a glass catalog name (e.g.
      ``"N-BK7"``), resolved via :meth:`NSQMaterial.from_glass` the same way
      :mod:`optiland.nonsequential.serialization` already round-trips
      materials.
    - ``{"kind": "opaque"}`` -- a non-catalog material kept only for
      dispatch (``lower(scene, strict=False)``); not losslessly
      serializable.

    A custom dispersion model with no catalog name cannot be lowered losslessly
    and raises at lowering time (see :func:`~optiland.nonsequential.ir.lower.lower`)
    rather than silently dropping to a constant approximation.

    Note that ``k_model`` here is descriptive only, matching the existing
    ``n_model`` precedent: the interpreter's Beer-Lambert absorption (D-13,
    see ``optiland.nonsequential.backends.array_backend.ArrayBackend.trace``)
    reads ``k(wavelength)`` from the live ``NSQMaterial`` on
    ``rays.k_current``, never from this IR.

    Attributes:
        id: Index into ``SceneIR.media``.
        name: Human-readable label (glass name, or ``"vacuum"``).
        n_model: Refractive-index model descriptor, as above.
        k_model: Extinction-coefficient model descriptor, as above.
    """

    id: int
    name: str
    n_model: dict[str, Any]
    k_model: dict[str, Any] | None = field(default=None)
