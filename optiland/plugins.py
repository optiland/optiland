"""Discovers and loads third-party Optiland plugins via entry points.

A plugin package declares one or more of the ``optiland.surfaces``,
``optiland.materials``, or ``optiland.analyses`` entry-point groups in its
own ``pyproject.toml`` and exposes a zero-argument callable that registers
itself against the relevant Optiland registry -- typically
``GeometryFactory.register`` for a new surface geometry, or
``MaterialRegistry.register``/``register_file`` for a glass catalog. This
module does not define new registration APIs: it only makes the existing
``register()`` calls (added by the prior SOLID pass) discoverable from
outside the package, without requiring a source edit to Optiland itself.

Loading is lazy and memoized per group -- each group's entry points are
resolved and invoked at most once per process, triggered by the first
factory access rather than at import time, so installing Optiland with no
plugins present pays no cost.
"""

from __future__ import annotations

import importlib.metadata
import warnings

SURFACES_GROUP = "optiland.surfaces"
MATERIALS_GROUP = "optiland.materials"
ANALYSES_GROUP = "optiland.analyses"

_loaded_groups: set[str] = set()


def load_plugins(group: str) -> None:
    """Load and invoke every entry point registered under *group*, once.

    Args:
        group: One of ``SURFACES_GROUP``, ``MATERIALS_GROUP``, or
            ``ANALYSES_GROUP``.
    """
    if group in _loaded_groups:
        return
    _loaded_groups.add(group)

    for entry_point in importlib.metadata.entry_points(group=group):
        try:
            register = entry_point.load()
            register()
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not break Optiland
            warnings.warn(
                f"Failed to load Optiland plugin '{entry_point.name}' "
                f"from group '{group}': {exc}",
                UserWarning,
                stacklevel=2,
            )
