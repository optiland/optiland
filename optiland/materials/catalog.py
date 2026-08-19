"""MaterialCatalog — read-only view into one catalog within the MaterialRegistry.

Provides discovery and search UX for a specific manufacturer catalog.

Kramer Harrison, 2025
"""

from __future__ import annotations

import pathlib
from importlib import resources

from optiland.materials.material_spec import MatchPolicy
from optiland.materials.registry import MaterialRegistry


class MaterialCatalog:
    """Read-only view into one glass manufacturer catalog.

    Usage::

        MaterialCatalog.groups()  # ['3d', 'glass', 'main', 'organic', 'other']
        MaterialCatalog.available()  # glass manufacturers only
        MaterialCatalog("schott").list()  # glass names in catalog
        MaterialCatalog("schott").search("bk7")  # substring search
        MaterialCatalog("schott").get("N-BK7")  # instantiate material

    Args:
        catalog: Glass manufacturer catalog name (e.g. ``"schott"``, ``"ohara"``).
    """

    def __init__(self, catalog: str) -> None:
        self._catalog = catalog

    @classmethod
    def groups(cls) -> list[str]:
        """Return sorted list of all group names in the database.

        Groups reflect the top-level organisation of the refractiveindex.info
        database: ``'glass'`` (manufacturer catalogs), ``'main'`` (elements and
        compounds), ``'organic'`` (organic species), ``'other'`` (miscellaneous),
        and ``'3d'`` (metamaterials/plasmonics).

        Non-glass groups are not exposed through :class:`MaterialCatalog`; use
        :meth:`~optiland.materials.registry.MaterialRegistry.list_catalogs` with
        a ``group`` argument to enumerate them.
        """
        return MaterialRegistry.instance().list_groups()

    @classmethod
    def available(cls) -> list[str]:
        """Return sorted list of glass manufacturer catalog names.

        The list is derived directly from the subdirectory names under the
        built-in ``data-nk/glass/`` directory, which is the authoritative
        source of manufacturer catalogs.  To list species in non-glass groups
        use :meth:`~optiland.materials.registry.MaterialRegistry.list_catalogs`
        directly with ``group='main'``, ``'organic'``, or ``'other'``.
        """
        glass_dir = pathlib.Path(
            str(resources.files("optiland.database").joinpath("data-nk/glass"))
        )
        return sorted(d.name for d in glass_dir.iterdir() if d.is_dir())

    def list(self) -> list[str]:
        """Return sorted list of material names in this catalog."""
        return MaterialRegistry.instance().list_materials(self._catalog)

    def search(self, query: str, n: int | None = None) -> list[str]:
        """Return material names whose name contains ``query`` (case-insensitive).

        Args:
            query: Substring to search for within glass names.
            n: Maximum number of results to return.  ``None`` returns all
                matches.

        Returns:
            Sorted list of matching material names.
        """
        q = query.lower()
        matches = [name for name in self.list() if q in name.lower()]
        return matches[:n] if n is not None else matches

    def get(self, name: str, match_policy: MatchPolicy = MatchPolicy.WARN) -> object:
        """Instantiate a :class:`~optiland.materials.material.Material` from this
        catalog.

        Args:
            name: Exact (or close) material name.
            match_policy: Controls fuzzy-match behavior.

        Returns:
            A :class:`~optiland.materials.material.Material` instance.
        """
        from optiland.materials.material import Material

        return Material(name, catalog=self._catalog, match_policy=match_policy)

    def __repr__(self) -> str:
        return f"MaterialCatalog('{self._catalog}')"
