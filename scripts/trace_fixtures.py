"""Shared system and bundle fixtures for the fused-trace tests and harnesses.

WP0 ships the stub: ``CATALOG`` (every shipped sample system, built at import
from ``optiland.samples.__all__`` plus the four classes ``samples.objectives``
does not export), empty ``KNOWN_INELIGIBLE`` and ``REFUSAL_FIXTURES`` maps, and
a ``register`` that does nothing yet.  WP5 fills them from I0 on.

It lives under ``scripts/`` rather than ``tests/`` on purpose: ``tests`` is
excluded from ruff, and this module is imported by tests *and* by
``scripts/metal_oracle_e2e.py`` / ``metal_suite.py`` / ``metal_benchmark.py``,
so it has to stay lint-clean.

Import it with ``scripts/`` on ``sys.path``::

    import sys; sys.path.insert(0, "scripts")
    import trace_fixtures
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

import optiland.samples as _samples
from optiland.samples import objectives as _objectives

__all__ = [
    "CATALOG",
    "KNOWN_INELIGIBLE",
    "REFUSAL_FIXTURES",
    "UNEXPORTED_OBJECTIVES",
    "build",
    "register",
]

#: Sample classes that ``optiland.samples.__all__`` omits but that the
#: conformance sweep must still cover (design 8.5).
UNEXPORTED_OBJECTIVES: tuple[str, ...] = (
    "WideAngle100FOV",
    "ProjectionLens120FOV",
    "ProjectionLens160FOV",
    "WideAngle170FOV",
)


def _build_catalog() -> dict[str, type]:
    """Every shipped sample system, keyed by class name."""
    catalog: dict[str, type] = {}
    for name in _samples.__all__:
        catalog[name] = getattr(_samples, name)
    for name in UNEXPORTED_OBJECTIVES:
        catalog[name] = getattr(_objectives, name)
    return catalog


#: name -> zero-argument class that builds the optic.
CATALOG: dict[str, type] = _build_catalog()

#: name -> the ``FusedTraceSkip`` reason a catalog system is expected to hit.
#: Empty at WP0: the conformance sweep asserts every catalog system is
#: eligible unless it is listed here, so WP5 adding a row is a recorded
#: decision rather than a silent fallback.
KNOWN_INELIGIBLE: dict[str, str] = {}

#: reason -> zero-argument builder of a system/bundle that must be refused
#: with exactly that reason.  Filled by WP5.
REFUSAL_FIXTURES: dict[str, Callable[[], Any]] = {}


def build(name: str) -> Any:
    """Instantiate the catalog system called ``name``."""
    return CATALOG[name]()


def register(*args: Any, **kwargs: Any) -> None:
    """No-op registration hook; WP5 replaces it.

    Present so that adapter modules written in wave 1 can already call it.
    """
    return None
