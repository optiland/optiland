"""Material

This module contains the Material class, which represents a generic material
used in the Optiland system. This class identifies the correct material given
the material name and (optionally) the reference, which is generally the
manufacturer name or the author name. This is the primary material class used
to define the optical properties of a material (or glass) in Optiland.

Kramer Harrison, 2024
"""

# import pkg_resources
from __future__ import annotations

import warnings
from importlib import resources

import pandas as pd

from optiland.materials.material_file import MaterialFile
from optiland.materials.material_spec import MatchPolicy
from optiland.materials.registry import MaterialRegistry


class Material(MaterialFile):
    """Represents a generic material used in the Optiland system.
    This class identifies the correct material given the material name and
    (optionally) the reference, which is generally the manufacturer name or
    the author name.

    Note:
        The material database is stored in the file `catalog_nk.csv` in the
        `database` directory. This contains the names, references, and
        filenames of the materials.

    Args:
        name (str): The name of the material to search for.
        reference (str, optional): The reference for the material, typically
            the manufacturer or author name. This helps disambiguate materials
            with similar names. Defaults to None.
        robust_search (bool | None, optional): Deprecated. Use ``match_policy``
            instead.  ``True`` maps to ``MatchPolicy.BEST``; ``False`` maps to
            ``MatchPolicy.STRICT``.  Passing this argument emits a
            ``DeprecationWarning``.  Defaults to None.
        min_wavelength (float, optional): Minimum wavelength in microns for
            filtering materials based on their valid range. Defaults to None.
        max_wavelength (float, optional): Maximum wavelength in microns for
            filtering materials based on their valid range. Defaults to None.
        catalog (str, optional): Manufacturer catalog to restrict lookup to
            (e.g. ``"schott"``, ``"ohara"``).  Keyword-only.  Defaults to None.
        match_policy (MatchPolicy, optional): Controls fuzzy-match behavior.
            ``"warn"`` (default) emits ``OptilandMaterialWarning`` on fuzzy
            match; ``"best"`` silently takes the best match; ``"strict"``
            raises ``ValueError`` on any non-exact match.  Keyword-only.

    Attributes:
        name (str): The name of the material.
        reference (str): The reference for the material.

    """

    _df = None
    _filename = str(resources.files("optiland.database").joinpath("catalog_nk.csv"))

    def __init__(
        self,
        name: str,
        reference: str | None = None,
        robust_search: bool | None = None,
        min_wavelength: float | None = None,
        max_wavelength: float | None = None,
        propagation_model=None,
        *,
        catalog: str | None = None,
        match_policy: MatchPolicy = MatchPolicy.WARN,
    ) -> None:
        self.name = name
        self.reference = reference
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self._catalog = catalog

        # Handle deprecated robust_search parameter
        if robust_search is not None:
            warnings.warn(
                "robust_search is deprecated; use match_policy='strict' or "
                "match_policy='best'.",
                DeprecationWarning,
                stacklevel=2,
            )
            match_policy = MatchPolicy.BEST if robust_search else MatchPolicy.STRICT

        self._match_policy = match_policy
        # Keep self.robust for backward compatibility
        self.robust = match_policy != MatchPolicy.STRICT

        file, self.material_data = self._retrieve_file()
        super().__init__(file, propagation_model=propagation_model)

    @classmethod
    def _load_dataframe(cls):
        """Load the catalog DataFrame (delegates to MaterialRegistry)."""
        return MaterialRegistry.instance().built_in_df

    @staticmethod
    def _levenshtein_distance(s1, s2):
        """Calculates the Levenshtein distance between two strings.

        Args:
            s1 (str): The first string.
            s2 (str): The second string.

        Returns:
            int: The Levenshtein distance between the two strings.

        """
        # Initialize matrix of zeros
        rows = len(s1) + 1
        cols = len(s2) + 1
        distance_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

        # Populate matrix with initial values
        for i in range(1, rows):
            distance_matrix[i][0] = i
        for j in range(1, cols):
            distance_matrix[0][j] = j

        # Calculate the distance
        for i in range(1, rows):
            for j in range(1, cols):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                distance_matrix[i][j] = min(
                    distance_matrix[i - 1][j] + 1,
                    distance_matrix[i][j - 1] + 1,
                    distance_matrix[i - 1][j - 1] + cost,
                )

        return distance_matrix[-1][-1]

    def _find_material_matches(self, df):
        """Finds material matches in a DataFrame based on the given name and
        reference.

        Args:
            df (pandas.DataFrame): The DataFrame containing the materials.

        Returns:
            pandas.DataFrame: A DataFrame containing materials that match the
            search criteria, sorted by similarity score. Returns an empty
            DataFrame if no potential matches are found.

        """
        # Make input name lowercase
        name = self.name.lower()

        # Filter rows where input string is substring of category_name or name
        dfi = df[
            df["category_name"].str.lower().str.contains(name)
            | df["name"].str.lower().str.contains(name)
            | df["filename_no_ext"].str.lower().str.contains(name)
        ].copy()

        # If reference given, filter rows non-matching rows
        if self.reference:
            reference = self.reference.lower()
            dfi = dfi[
                dfi["category_name"].str.lower().str.contains(reference)
                | dfi["category_name_full"].str.lower().str.contains(reference)
                | dfi["reference"].str.lower().str.contains(reference)
                | dfi["name"].str.lower().str.contains(reference)
                | dfi["filename"].str.lower().str.contains(reference)
            ]

        # Filter rows based on wavelength range
        if self.min_wavelength:
            dfi = dfi[
                (dfi["min_wavelength"] <= self.min_wavelength)
                & (dfi["max_wavelength"] >= self.min_wavelength)
            ]
        if self.max_wavelength:
            dfi = dfi[
                (dfi["min_wavelength"] <= self.max_wavelength)
                & (dfi["max_wavelength"] >= self.max_wavelength)
            ]

        # If no rows match, return an empty DataFrame
        if dfi.empty:
            return pd.DataFrame()

        # Calculate similarity scores using Levenshtein distance
        dfi["similarity_score"] = dfi.apply(
            lambda row: min(
                self._levenshtein_distance(name, row["category_name"].lower()),
                self._levenshtein_distance(name, row["name"].lower()),
                self._levenshtein_distance(name, row["filename_no_ext"].lower()),
            ),
            axis=1,
        )

        # Sort by similarity score in ascending order
        dfi = dfi.sort_values(by="similarity_score").reset_index(drop=True)

        return dfi

    def _raise_material_error(self, no_matches=False, multiple_matches=False):
        """Raises an error if no matches or multiple matches are found for the
        material.

        Args:
            no_matches (bool): Indicates if no matches were found.
            multiple_matches (bool): Indicates if multiple matches were found.

        Raises:
            ValueError: If no matches or multiple matches are found for the
                material.

        """
        if no_matches:
            message = f"No matches found for material {self.name}"
        elif multiple_matches:
            message = f"Multiple matches found for material {self.name}"
        else:
            message = f"Error finding material {self.name}"

        if self.reference:
            message += f" with reference {self.reference}"

        if self._catalog:
            message += f" in catalog '{self._catalog}'"

        if self.min_wavelength or self.max_wavelength:
            wavelength_range = f"({self.min_wavelength}, {self.max_wavelength}) µm"
            message += f" within wavelength range {wavelength_range}"

        raise ValueError(message)

    @staticmethod
    def _catalog_from_filename(filename: str) -> str:
        """Extract the manufacturer catalog name from a material filename path.

        The filename follows the pattern ``group/catalog/name.yml``, so the
        manufacturer catalog is the second-to-last path segment.

        Args:
            filename: The filename string from the catalog DataFrame.

        Returns:
            str: The catalog name, or an empty string if not determinable.
        """
        parts = filename.split("/")
        return parts[-2] if len(parts) >= 3 else ""

    def _retrieve_file(self):
        """Retrieve the file path for the material by delegating to MaterialRegistry.

        Returns:
            tuple[str, dict]: A tuple containing:
                - The full file path to the material data file.
                - A dictionary containing the material's metadata from the catalog.

        Raises:
            ValueError: If no matches are found for the material.
            ValueError: If match_policy is STRICT and the match is not exact.

        """
        return MaterialRegistry.instance()._resolve_with_row(
            self.name,
            self._catalog,
            self.reference,
            self._match_policy,
            self.min_wavelength,
            self.max_wavelength,
        )

    def to_dict(self):
        """Converts the material to a dictionary.

        Returns:
            dict: A dictionary representation of the Material instance's
            configuration, not the material data itself.

        """
        material_dict = super().to_dict()
        material_dict.update(
            {
                "name": self.name,
                "reference": self.reference,
                "catalog": self._catalog,
                "match_policy": self._match_policy.value,
                "robust_search": None,
                "min_wavelength": self.min_wavelength,
                "max_wavelength": self.max_wavelength,
            },
        )

        return material_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a material from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the material.

        Returns:
            Material: The material created from the dictionary.

        """
        if "name" not in data:
            raise ValueError("Missing required key: name")

        # Warn when loading a file that has no catalog field (legacy format).
        if "catalog" not in data or data["catalog"] is None:
            warnings.warn(
                f"Material '{data['name']}' loaded from file has no 'catalog' "
                "field. Re-save the lens file to record catalog information. "
                "Lookup will fall back to fuzzy search.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Translate legacy robust_search to match_policy without triggering
        # the deprecation warning — from_dict is the known old-format handler.
        if "robust_search" in data and data["robust_search"] is not None:
            rs = data["robust_search"]
            match_policy = MatchPolicy.BEST if rs else MatchPolicy.STRICT
        else:
            mp_value = data.get("match_policy", MatchPolicy.WARN.value)
            match_policy = MatchPolicy(mp_value)

        return cls(
            data["name"],
            data.get("reference", None),
            None,  # robust_search=None avoids re-triggering DeprecationWarning
            data.get("min_wavelength", None),
            data.get("max_wavelength", None),
            catalog=data.get("catalog", None),
            match_policy=match_policy,
        )

    def __repr__(self) -> str:
        catalog_str = f", catalog='{self._catalog}'" if self._catalog else ""
        wl_range = ""
        md = getattr(self, "material_data", None)
        if md:
            min_wl = md.get("min_wavelength")
            max_wl = md.get("max_wavelength")
            if min_wl is not None and max_wl is not None:
                wl_range = f", λ=[{min_wl:.2f}µm, {max_wl:.2f}µm]"
        return f"Material(name='{self.name}'{catalog_str}{wl_range})"
