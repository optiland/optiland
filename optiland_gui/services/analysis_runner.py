"""Analysis runner service for the Optiland GUI.

Handles analysis class discovery via :mod:`optiland_gui.registry` and provides
execution lifecycle stubs that forward to the :class:`AnalysisPanel`.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)


class AnalysisRunner:
    """Manages analysis discovery and the execution lifecycle.

    Analysis classes are loaded lazily from
    :data:`optiland_gui.registry.ANALYSIS_REGISTRY` via
    :func:`importlib.import_module`.  The resolved
    ``(category, name, class)`` tuples are cached after the first call to
    :meth:`get_analysis_registry`.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`
            instance that owns this service.
    """

    def __init__(self, connector: object) -> None:
        self._connector = connector
        self._registry_cache: list[tuple[str, str, type]] | None = None

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def get_analysis_registry(self) -> list[tuple[str, str, type]]:
        """Return the resolved analysis registry.

        Each entry is a ``(category, display_name, cls)`` tuple where *cls*
        is the live Python class loaded via :func:`importlib.import_module`.
        Entries whose class path cannot be imported are silently omitted and
        a warning is logged.

        The result is cached after the first call.

        Returns:
            A list of ``(category, display_name, cls)`` tuples.
        """
        if self._registry_cache is not None:
            return self._registry_cache

        from optiland_gui.registry import ANALYSIS_REGISTRY

        resolved: list[tuple[str, str, type]] = []
        for category, name, class_path in ANALYSIS_REGISTRY:
            try:
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                resolved.append((category, name, cls))
            except (ImportError, AttributeError) as exc:
                logger.warning(
                    "AnalysisRunner: could not load '%s' (%s): %s",
                    name,
                    class_path,
                    exc,
                )

        self._registry_cache = resolved
        return self._registry_cache

    # ------------------------------------------------------------------
    # Execution lifecycle stubs
    # ------------------------------------------------------------------

    def run(
        self,
        analysis_name: str,
        params: dict,
        optic: object,
    ) -> None:
        """Execute a named analysis with the given parameters.

        Args:
            analysis_name: The display name of the analysis as it appears in
                the registry.
            params: A dict of parameter name → value pairs to pass to the
                analysis class constructor.
            optic: The :class:`~optiland.optic.Optic` instance to analyse.
        """

    def stop(self) -> None:
        """Request cancellation of an in-progress analysis run."""

    def get_result(self) -> object | None:
        """Return the result of the most recent analysis run.

        Returns:
            ``None`` until background analysis execution is fully wired up.
        """
        return None
