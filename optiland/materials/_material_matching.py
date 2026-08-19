"""Material name resolution engine: filtering, scoring, and match_policy.

Split out of ``optiland.materials.registry`` — everything here is a pure
function of a candidate DataFrame and the caller's search parameters, with
no dependency on `MaterialRegistry` instance state.

Kramer Harrison, 2025
"""

from __future__ import annotations

import pathlib
import warnings
from importlib import resources

import pandas as pd

from optiland._suggest import did_you_mean
from optiland.materials._catalog_parsing import _levenshtein
from optiland.materials.material_spec import MatchPolicy
from optiland.materials.warnings import OptilandMaterialWarning

_DATA_NK_DIR = str(resources.files("optiland.database").joinpath("data-nk"))


def resolve_with_row(
    df: pd.DataFrame,
    name: str,
    catalog: str | None,
    reference: str | None,
    match_policy: MatchPolicy,
    min_wavelength: float | None,
    max_wavelength: float | None,
) -> tuple[str, dict]:
    """Resolve a material against ``df`` and return ``(path, metadata_row_dict)``."""
    df = _prefilter_by_catalog(df, catalog)

    filtered_df = _find_matches(df, name, reference, min_wavelength, max_wavelength)
    _raise_if_no_matches(filtered_df, name, catalog, reference, df)

    _apply_match_policy(filtered_df, name, catalog, match_policy)

    row = filtered_df.iloc[0].to_dict()
    return _row_to_path(row), row


def _prefilter_by_catalog(df: pd.DataFrame, catalog: str | None) -> pd.DataFrame:
    """Restrict ``df`` to a single manufacturer catalog, if given."""
    if catalog is None:
        return df
    catalog_lower = catalog.lower()
    filtered = df[df["catalog_dir"].str.lower() == catalog_lower].copy()
    if filtered.empty:
        known = sorted(df["catalog_dir"].dropna().unique())
        raise ValueError(
            f"No catalog '{catalog}' found in the material database."
            f"{did_you_mean(catalog, known)} "
            "List the available catalogs with "
            "MaterialRegistry.instance().list_catalogs()."
        )
    return filtered


def _raise_if_no_matches(
    filtered_df: pd.DataFrame,
    name: str,
    catalog: str | None,
    reference: str | None,
    candidates: pd.DataFrame | None = None,
) -> None:
    """Raise ``ValueError`` if no candidate rows survived filtering.

    Args:
        filtered_df: The rows that survived name/reference filtering.
        name: The requested material name.
        catalog: The catalog the search was scoped to, if any.
        reference: The reference the search was scoped to, if any.
        candidates: The rows searched, used to build a "did you mean"
            suggestion from their ``name`` column.
    """
    if not filtered_df.empty:
        return
    msg = f"No matches found for material '{name}'"
    if catalog:
        msg += f" in catalog '{catalog}'"
    if reference:
        msg += f" with reference '{reference}'"
    msg += "."
    if candidates is not None and "name" in candidates:
        msg += did_you_mean(name, candidates["name"].dropna().unique())
    raise ValueError(msg)


def _apply_match_policy(
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
        _apply_match_policy_with_catalog(
            filtered_df, name, catalog, match_policy, best_score
        )
    else:
        _apply_match_policy_without_catalog(
            filtered_df, name, match_policy, best_score, ambiguous_exact
        )


def _apply_match_policy_with_catalog(
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
    filtered_df: pd.DataFrame,
    name: str,
    match_policy: MatchPolicy,
    best_score: float,
    ambiguous_exact: bool = False,
) -> None:
    """Apply ``match_policy`` when resolution spans all catalogs."""
    if match_policy == MatchPolicy.STRICT:
        if ambiguous_exact:
            catalogs = sorted(
                filtered_df.loc[
                    filtered_df["similarity_score"] == 0, "catalog_dir"
                ].unique()
            )
            raise ValueError(
                f"Material '{name}' matches exactly in multiple catalogs: "
                f"{catalogs}. Pass catalog=<name> to disambiguate."
            )
        top = filtered_df.head(5)["name"].tolist()
        raise ValueError(
            f"No exact match for material '{name}'. "
            f"Top candidates: {top}. "
            "Use match_policy='warn' or 'best' for fuzzy matching."
        )
    if match_policy == MatchPolicy.WARN:
        if best_score > 0:
            resolved = filtered_df.iloc[0]["name"]
            warnings.warn(
                f"Material '{name}' resolved to '{resolved}' via fuzzy match.",
                OptilandMaterialWarning,
                stacklevel=6,
            )
        elif ambiguous_exact:
            catalogs = sorted(
                filtered_df.loc[
                    filtered_df["similarity_score"] == 0, "catalog_dir"
                ].unique()
            )
            resolved_catalog = filtered_df.iloc[0]["catalog_dir"]
            warnings.warn(
                f"Material '{name}' matches exactly in multiple catalogs "
                f"{catalogs}; resolved to catalog '{resolved_catalog}'. "
                "Pass catalog=<name> to disambiguate and silence this "
                "warning.",
                OptilandMaterialWarning,
                stacklevel=6,
            )


def _row_to_path(row: dict) -> str:
    """Resolve a matched row's ``filename`` to an absolute path."""
    filename = row["filename"]
    if not pathlib.Path(filename).is_absolute():
        return str(pathlib.Path(_DATA_NK_DIR) / filename)
    return filename


def _find_matches(
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
            | dfi["category_name_full"].str.lower().str.contains(ref_lower, na=False)
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
