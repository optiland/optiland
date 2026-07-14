"""MaterialRegistry — global singleton for material lookup.

Owns the complete lookup table for both built-in and user-registered materials.
Wraps the existing ``catalog_nk.csv`` load and adds user catalog state.

The hot path (built-in CSV lookup) adds zero overhead compared to PR1: the
built-in DataFrame is lazy-loaded once per process and cached, exactly as before.

Kramer Harrison, 2025
"""

from __future__ import annotations

import contextlib
import pathlib
import tempfile
import warnings
from importlib import resources

import pandas as pd
import yaml

import optiland.plugins as plugins
from optiland.materials.material_spec import MatchPolicy
from optiland.materials.warnings import OptilandMaterialWarning

_CATALOG_CSV = str(resources.files("optiland.database").joinpath("catalog_nk.csv"))
_DATA_NK_DIR = str(resources.files("optiland.database").joinpath("data-nk"))


def _levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    rows, cols = len(s1) + 1, len(s2) + 1
    dist = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        dist[i][0] = i
    for j in range(1, cols):
        dist[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )
    return dist[-1][-1]


def _catalog_dir_from_filename(filename: str, group: str = "") -> str:
    """Extract the manufacturer catalog name from a filename path.

    For glass entries whose path starts with ``glass/``, the manufacturer is
    always the second path segment (``glass/{manufacturer}/...``), regardless
    of nesting depth.  For all other entries the species name is the
    second-to-last segment.
    """
    parts = filename.replace("\\", "/").split("/")
    if group.lower() == "glass" and parts[0].lower() == "glass" and len(parts) >= 3:
        return parts[1]
    return parts[-2] if len(parts) >= 3 else ""


def _extract_wavelength_range(data: dict) -> tuple[float, float]:
    """Extract min/max wavelength (µm) from a refractiveindex.info YAML payload."""
    for item in data.get("DATA", []):
        raw = item.get("data", "")
        if not raw:
            continue
        wls: list[float] = []
        for line in raw.strip().splitlines():
            parts = line.split()
            if parts:
                with contextlib.suppress(ValueError):
                    wls.append(float(parts[0]))
        if wls:
            return min(wls), max(wls)
        # Formula entries may have a wavelength_range field
        wl_range = item.get("wavelength_range", "")
        if wl_range:
            parts = str(wl_range).split()
            if len(parts) >= 2:
                try:
                    return float(parts[0]), float(parts[1])
                except ValueError:
                    pass
    return 0.0, 100.0


class MaterialRegistry:
    """Global singleton owning all material lookup state (built-in + user).

    Access via :meth:`MaterialRegistry.instance`.  All methods are safe to
    call from any thread after the first :meth:`instance` call completes.

    On the first call to :meth:`instance` the registry checks for
    ``~/.optiland/catalogs/``.  Each subdirectory there is ingested via
    :meth:`load_catalog` as a best-effort operation; failures emit
    :class:`~optiland.materials.warnings.OptilandMaterialWarning` and are
    skipped, never raising.

    Args:
        None — instantiate via :meth:`instance`.
    """

    _instance: MaterialRegistry | None = None

    def __init__(self) -> None:
        self.__built_in_df: pd.DataFrame | None = None
        self._user_entries: list[dict] = []
        self._user_temp_files: list[str] = []
        self._combined_cache: pd.DataFrame | None = None

    @classmethod
    def instance(cls) -> MaterialRegistry:
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            obj = object.__new__(cls)
            obj.__init__()  # type: ignore[misc]
            cls._instance = obj
            cls._instance._auto_discover()
        return cls._instance

    # ------------------------------------------------------------------
    # Built-in catalog
    # ------------------------------------------------------------------

    @property
    def built_in_df(self) -> pd.DataFrame:
        """Lazy-loaded, cached built-in catalog DataFrame."""
        if self.__built_in_df is None:
            self.__built_in_df = pd.read_csv(_CATALOG_CSV)
        return self.__built_in_df

    # ------------------------------------------------------------------
    # User catalog
    # ------------------------------------------------------------------

    def register(self, name: str, catalog: str, data: dict) -> None:
        """Register a material programmatically.

        The ``data`` dict must follow the refractiveindex.info YAML schema.
        If a built-in or previously-registered entry with the same
        ``(name, catalog)`` key exists, it is shadowed and a warning is
        emitted.

        Args:
            name: Material name (used for lookup).
            catalog: Catalog name (e.g. ``"internal"``).
            data: refractiveindex.info YAML payload as a Python dict.
        """
        # Write data to a named temp file so MaterialFile can read it.
        # NamedTemporaryFile with delete=False is used so the path persists
        # after close; SIM115 does not apply here because we need tmp.name.
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
            delete=False,
            suffix=".yml",
            prefix=f"{name}_",
            mode="w",
            encoding="utf-8",
        )
        yaml.dump(data, tmp)
        tmp.flush()
        tmp.close()
        self._user_temp_files.append(tmp.name)

        min_wl, max_wl = _extract_wavelength_range(data)
        reference = data.get("REFERENCE", catalog)

        entry = {
            "group": "user",
            "category_name": catalog,
            "category_name_full": catalog,
            "reference": reference,
            "name": name,
            "filename": tmp.name,
            "min_wavelength": min_wl,
            "max_wavelength": max_wl,
            "filename_no_ext": name,
            "catalog_dir": catalog.lower(),
        }

        self._warn_if_shadow(name, catalog)
        self._user_entries.append(entry)
        self._combined_cache = None  # invalidate cache

    def register_file(self, path: str | pathlib.Path) -> None:
        """Load a single refractiveindex.info-format YAML file.

        The catalog name is inferred from the parent directory name.  The
        material name is the filename stem (without extension).

        Args:
            path: Path to a refractiveindex.info YAML file.
        """
        p = pathlib.Path(path)
        name = p.stem
        catalog = p.parent.name

        with open(p, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        reference = data.get("REFERENCE", catalog) if data else catalog
        min_wl, max_wl = _extract_wavelength_range(data or {})

        entry = {
            "group": "user",
            "category_name": catalog,
            "category_name_full": catalog,
            "reference": reference,
            "name": name,
            "filename": str(p.resolve()),
            "min_wavelength": min_wl,
            "max_wavelength": max_wl,
            "filename_no_ext": name,
            "catalog_dir": catalog.lower(),
        }

        self._warn_if_shadow(name, catalog)
        self._user_entries.append(entry)
        self._combined_cache = None

    def load_catalog(self, directory: str | pathlib.Path) -> None:
        """Load all YAML files found in ``directory``.

        If a ``catalog.csv`` index exists (same schema as the built-in
        ``catalog_nk.csv``), it is used directly.  Otherwise each ``.yml``
        file is registered via :meth:`register_file`.

        Args:
            directory: Path to a directory of refractiveindex.info YAML files.
        """
        d = pathlib.Path(directory)
        if not d.is_dir():
            return

        index_csv = d / "catalog.csv"
        if index_csv.exists():
            extra_df = pd.read_csv(index_csv)
            catalog_name = d.name.lower()
            if "catalog_dir" not in extra_df.columns:
                extra_df["catalog_dir"] = catalog_name

            # Resolve relative filenames against the directory
            def _resolve_fn(fn: str) -> str:
                if pathlib.Path(fn).is_absolute():
                    return fn
                return str((d / fn).resolve())

            extra_df["filename"] = extra_df["filename"].apply(_resolve_fn)
            for _, row in extra_df.iterrows():
                self._warn_if_shadow(row.get("name", ""), row.get("catalog_dir", ""))
            self._user_entries.extend(extra_df.to_dict("records"))
            self._combined_cache = None
        else:
            for yml_file in sorted(d.glob("*.yml")):
                try:
                    self.register_file(yml_file)
                except Exception as exc:
                    warnings.warn(
                        f"Failed to load material file '{yml_file}': {exc}",
                        OptilandMaterialWarning,
                        stacklevel=2,
                    )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        name: str,
        catalog: str | None = None,
        reference: str | None = None,
        match_policy: MatchPolicy = MatchPolicy.WARN,
        min_wavelength: float | None = None,
        max_wavelength: float | None = None,
    ) -> str:
        """Return the absolute path to the resolved YAML data file.

        Args:
            name: Material name to search for.
            catalog: Manufacturer catalog to restrict lookup to.
            reference: Citation string for further disambiguation.
            match_policy: Controls fuzzy-match warnings/errors.
            min_wavelength: Minimum wavelength filter (µm).
            max_wavelength: Maximum wavelength filter (µm).

        Returns:
            Absolute path to the resolved YAML data file.

        Raises:
            ValueError: If no match found, or if ``match_policy='strict'``
                and the match is not exact / is ambiguous.
        """
        plugins.load_plugins(plugins.MATERIALS_GROUP)
        path, _ = self._resolve_with_row(
            name, catalog, reference, match_policy, min_wavelength, max_wavelength
        )
        return path

    def _resolve_with_row(
        self,
        name: str,
        catalog: str | None,
        reference: str | None,
        match_policy: MatchPolicy,
        min_wavelength: float | None,
        max_wavelength: float | None,
    ) -> tuple[str, dict]:
        """Resolve a material and return ``(path, metadata_row_dict)``."""
        df = self._prefilter_by_catalog(self._get_combined_df(), catalog)

        filtered_df = self._find_matches(
            df, name, reference, min_wavelength, max_wavelength
        )
        self._raise_if_no_matches(filtered_df, name, catalog, reference)

        self._apply_match_policy(filtered_df, name, catalog, match_policy)

        row = filtered_df.iloc[0].to_dict()
        return self._row_to_path(row), row

    def _prefilter_by_catalog(
        self, df: pd.DataFrame, catalog: str | None
    ) -> pd.DataFrame:
        """Restrict ``df`` to a single manufacturer catalog, if given."""
        if catalog is None:
            return df
        catalog_lower = catalog.lower()
        filtered = df[df["catalog_dir"].str.lower() == catalog_lower].copy()
        if filtered.empty:
            raise ValueError(f"No catalog '{catalog}' found in material database.")
        return filtered

    def _raise_if_no_matches(
        self,
        filtered_df: pd.DataFrame,
        name: str,
        catalog: str | None,
        reference: str | None,
    ) -> None:
        """Raise ``ValueError`` if no candidate rows survived filtering."""
        if filtered_df.empty:
            msg = f"No matches found for material '{name}'"
            if catalog:
                msg += f" in catalog '{catalog}'"
            if reference:
                msg += f" with reference '{reference}'"
            raise ValueError(msg)

    def _apply_match_policy(
        self,
        filtered_df: pd.DataFrame,
        name: str,
        catalog: str | None,
        match_policy: MatchPolicy,
    ) -> None:
        """Enforce ``match_policy`` for a non-exact or ambiguous top match."""
        best_score = filtered_df["similarity_score"].iloc[0]
        exact_mask = filtered_df["similarity_score"] == 0
        n_exact_files = (
            int(filtered_df.loc[exact_mask, "filename"].nunique())
            if exact_mask.any()
            else 0
        )
        ambiguous_exact = best_score == 0 and n_exact_files > 1

        if best_score <= 0 and not ambiguous_exact:
            return

        if catalog is not None:
            self._apply_match_policy_with_catalog(
                filtered_df, name, catalog, match_policy, best_score
            )
        else:
            self._apply_match_policy_without_catalog(
                filtered_df, name, match_policy, best_score
            )

    def _apply_match_policy_with_catalog(
        self,
        filtered_df: pd.DataFrame,
        name: str,
        catalog: str,
        match_policy: MatchPolicy,
        best_score: float,
    ) -> None:
        """Apply ``match_policy`` when resolution was scoped to one catalog."""
        if match_policy == MatchPolicy.STRICT:
            raise ValueError(
                f"No exact match for '{name}' in catalog '{catalog}'. "
                "Use the exact name or a less strict match_policy."
            )
        if best_score > 0:
            resolved = filtered_df.iloc[0]["name"]
            warnings.warn(
                f"No exact match for '{name}' in catalog '{catalog}'; "
                f"resolved to '{resolved}'. Use exact name to silence.",
                OptilandMaterialWarning,
                stacklevel=6,
            )

    def _apply_match_policy_without_catalog(
        self,
        filtered_df: pd.DataFrame,
        name: str,
        match_policy: MatchPolicy,
        best_score: float,
    ) -> None:
        """Apply ``match_policy`` when resolution spans all catalogs."""
        if match_policy == MatchPolicy.STRICT:
            top = filtered_df.head(5)["name"].tolist()
            raise ValueError(
                f"No exact match for material '{name}'. "
                f"Top candidates: {top}. "
                "Use match_policy='warn' or 'best' for fuzzy matching."
            )
        if match_policy == MatchPolicy.WARN and best_score > 0:
            resolved = filtered_df.iloc[0]["name"]
            warnings.warn(
                f"Material '{name}' resolved to '{resolved}' via fuzzy match.",
                OptilandMaterialWarning,
                stacklevel=6,
            )

    def _row_to_path(self, row: dict) -> str:
        """Resolve a matched row's ``filename`` to an absolute path."""
        filename = row["filename"]
        if not pathlib.Path(filename).is_absolute():
            return str(pathlib.Path(_DATA_NK_DIR) / filename)
        return filename

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_groups(self) -> list[str]:
        """Return sorted unique group names (built-in + user-registered)."""
        df = self._get_combined_df()
        return sorted(df["group"].dropna().unique().tolist())

    def list_catalogs(self, group: str | None = None) -> list[str]:
        """Return sorted unique catalog names, optionally filtered by group.

        Args:
            group: If given, restrict results to this group (e.g. ``'glass'``,
                ``'main'``, ``'organic'``, ``'other'``, ``'3d'``).
        """
        df = self._get_combined_df()
        if group is not None:
            df = df[df["group"].str.lower() == group.lower()]
        return sorted(df["catalog_dir"].dropna().unique().tolist())

    def list_materials(self, catalog: str | None = None) -> list[str]:
        """Return sorted material names, optionally filtered to one catalog.

        Args:
            catalog: If given, restrict results to this catalog name.

        Returns:
            Sorted list of material ``filename_no_ext`` values.
        """
        df = self._get_combined_df()
        if catalog is not None:
            df = df[df["catalog_dir"].str.lower() == catalog.lower()]
        return sorted(df["filename_no_ext"].dropna().unique().tolist())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_combined_df(self) -> pd.DataFrame:
        """Return built-in + user entries as a single DataFrame (cached)."""
        if self._combined_cache is not None:
            return self._combined_cache

        built_in = self.built_in_df.copy()
        built_in["catalog_dir"] = built_in.apply(
            lambda r: _catalog_dir_from_filename(r["filename"], r.get("group", "")),
            axis=1,
        )

        if not self._user_entries:
            self._combined_cache = built_in
            return self._combined_cache

        user_df = pd.DataFrame(self._user_entries)

        # User entries shadow built-ins with the same (filename_no_ext, catalog_dir)
        shadow_keys = set(
            zip(
                user_df["filename_no_ext"].str.lower(),
                user_df["catalog_dir"].str.lower(),
                strict=False,
            )
        )
        mask = ~built_in.apply(
            lambda r: (
                (
                    r["filename_no_ext"].lower(),
                    r.get("catalog_dir", "").lower(),
                )
                in shadow_keys
            ),
            axis=1,
        )
        self._combined_cache = pd.concat([built_in[mask], user_df], ignore_index=True)
        return self._combined_cache

    def _find_matches(
        self,
        df: pd.DataFrame,
        name: str,
        reference: str | None,
        min_wavelength: float | None,
        max_wavelength: float | None,
    ) -> pd.DataFrame:
        """Find and score candidate rows for the given name."""
        name_lower = name.lower()

        dfi = df[
            df["category_name"].str.lower().str.contains(name_lower, na=False)
            | df["name"].str.lower().str.contains(name_lower, na=False)
            | df["filename_no_ext"].str.lower().str.contains(name_lower, na=False)
        ].copy()

        if reference:
            ref_lower = reference.lower()
            dfi = dfi[
                dfi["category_name"].str.lower().str.contains(ref_lower, na=False)
                | dfi["category_name_full"]
                .str.lower()
                .str.contains(  # noqa: E501
                    ref_lower, na=False
                )
                | dfi["reference"].str.lower().str.contains(ref_lower, na=False)
                | dfi["name"].str.lower().str.contains(ref_lower, na=False)
                | dfi["filename"].str.lower().str.contains(ref_lower, na=False)
            ]

        if min_wavelength is not None:
            dfi = dfi[
                (dfi["min_wavelength"] <= min_wavelength)
                & (dfi["max_wavelength"] >= min_wavelength)
            ]
        if max_wavelength is not None:
            dfi = dfi[
                (dfi["min_wavelength"] <= max_wavelength)
                & (dfi["max_wavelength"] >= max_wavelength)
            ]

        if dfi.empty:
            return pd.DataFrame()

        dfi["similarity_score"] = dfi.apply(
            lambda row: min(
                _levenshtein(name_lower, row["category_name"].lower()),
                _levenshtein(name_lower, row["name"].lower()),
                _levenshtein(name_lower, row["filename_no_ext"].lower()),
            ),
            axis=1,
        )

        return dfi.sort_values("similarity_score").reset_index(drop=True)

    def _warn_if_shadow(self, name: str, catalog: str) -> None:
        """Emit a warning if (name, catalog) shadows an existing entry."""
        name_lower = name.lower()
        catalog_lower = catalog.lower()

        # Check built-in
        bi = self.built_in_df
        bi_catalog_dirs = bi.apply(
            lambda r: _catalog_dir_from_filename(r["filename"], r.get("group", "")),
            axis=1,
        ).str.lower()
        if (
            (bi["filename_no_ext"].str.lower() == name_lower)
            & (bi_catalog_dirs == catalog_lower)
        ).any():
            warnings.warn(
                f"User-registered material '{name}' (catalog='{catalog}') "
                "shadows a built-in entry.",
                OptilandMaterialWarning,
                stacklevel=3,
            )
            return

        # Check existing user entries
        for entry in self._user_entries:
            if (
                entry.get("filename_no_ext", "").lower() == name_lower
                and entry.get("catalog_dir", "").lower() == catalog_lower
            ):
                warnings.warn(
                    f"User-registered material '{name}' (catalog='{catalog}') "
                    "overwrites a previously registered user entry.",
                    OptilandMaterialWarning,
                    stacklevel=3,
                )
                return

    def _auto_discover(self) -> None:
        """Check ~/.optiland/catalogs/ and ingest any subdirectories found."""
        user_catalogs = pathlib.Path.home() / ".optiland" / "catalogs"
        if not user_catalogs.is_dir():
            return
        for subdir in sorted(user_catalogs.iterdir()):
            if subdir.is_dir():
                try:
                    self.load_catalog(subdir)
                except Exception as exc:
                    warnings.warn(
                        f"Auto-discovery: failed to load catalog "
                        f"'{subdir.name}': {exc}",
                        OptilandMaterialWarning,
                        stacklevel=1,
                    )
