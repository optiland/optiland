"""Unified non-finite / trace-failure policy (D16, §3.5).

A single policy shared by every evaluator and managed adapter governs what
happens when a merit or residual evaluation fails — either by raising (e.g. a
ray-trace error: TIR, missed surface) or by producing a non-finite value
(``NaN`` / ``inf``).

Modes (``on_failure``):

* ``"reject"`` (default) — the evaluation returns a **non-finite** merit /
  ``NaN`` residual.  Native stepped controllers already treat a non-finite
  trial as a rejected step, so the *step* is rejected, not the whole run.
* ``"raise"`` — re-raise as :class:`OptimizationFailure`.
* ``"penalty"`` — return a finite, **scale-proportional** penalty (a multiple
  of a reference merit, never a hardcoded ``1e10``).  This is what managed
  SciPy adapters use internally because ``scipy`` cannot accept ``NaN``.

Managed SciPy adapters always map a non-finite evaluation to a penalty (because
the solver cannot consume ``NaN``) unless ``on_failure="raise"``.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import Any

import optiland.backend as be
from optiland.optimization.errors import ConfigurationError, OptimizationFailure

#: Valid ``on_failure`` policy strings.
VALID_FAILURE_MODES: tuple[str, ...] = ("reject", "raise", "penalty")

#: A penalty merit is this multiple of the reference (last-finite) merit.
PENALTY_FACTOR: float = 1e6

#: Floor on the reference merit so a near-zero reference still repels strongly.
PENALTY_FLOOR: float = 1.0


def normalize_failure_mode(mode: str | None) -> str:
    """Validate and normalize an ``on_failure`` policy string.

    Args:
        mode: One of ``"reject"``, ``"raise"``, ``"penalty"`` (case-insensitive),
            or ``None`` (resolves to the default ``"reject"``).

    Returns:
        The normalized lowercase mode.

    Raises:
        ConfigurationError: If ``mode`` is not a recognized policy.
    """
    if mode is None:
        return "reject"
    norm = str(mode).lower()
    if norm not in VALID_FAILURE_MODES:
        raise ConfigurationError(
            f"Unknown on_failure policy {mode!r}. "
            f"Valid options: {list(VALID_FAILURE_MODES)}."
        )
    return norm


def is_nonfinite(value: Any) -> bool:
    """Return True if ``value`` is non-finite or cannot be coerced to a float."""
    try:
        f = float(be.to_numpy(value))
    except Exception:  # noqa: BLE001
        return True
    return not math.isfinite(f)


def penalty_merit(reference: float) -> float:
    """Return a finite penalty merit proportional to a reference merit."""
    return PENALTY_FACTOR * max(abs(float(reference)), PENALTY_FLOOR)


def penalty_residual(reference: float, n: int) -> float:
    """Return the per-residual magnitude whose squares sum to ``penalty_merit``.

    Args:
        reference: Reference merit (e.g. the last finite ``sum_squared``).
        n: Number of residual entries.

    Returns:
        ``sqrt(penalty_merit(reference) / n)`` (or ``sqrt(penalty_merit)`` when
        ``n <= 0``).
    """
    total = penalty_merit(reference)
    return math.sqrt(total / n) if n > 0 else math.sqrt(total)


__all__ = [
    "VALID_FAILURE_MODES",
    "PENALTY_FACTOR",
    "PENALTY_FLOOR",
    "OptimizationFailure",
    "normalize_failure_mode",
    "is_nonfinite",
    "penalty_merit",
    "penalty_residual",
]
