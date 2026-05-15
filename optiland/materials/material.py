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
        robust_search (bool, optional): If True, the search attempts to find the
            closest match even if an exact match isn't found, and returns the
            first one based on similarity scoring. If False, an error is raised
            if multiple close matches are found. Defaults to True.
        min_wavelength (float, optional): Minimum wavelength in microns for
            filtering materials based on their valid range. Defaults to None.
        max_wavelength (float, optional): Maximum wavelength in microns for
            filtering materials based on their valid range. Defaults to None.

    Attributes:
        name (str): The name of the material.
        reference (str): The reference for the material.

    """

    _df = None
    _filename = str(resources.files("optiland.database").joinpath("catalog_nk.csv"))

    def __init__(
        self,
        name,
        reference=None,
        robust_search=True,
        min_wavelength=None,
        max_wavelength=None,
        propagation_model=None,
    ):
        self.name = name
        self.reference = reference
        self.robust = robust_search
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        file, self.material_data = self._retrieve_file()
        super().__init__(file, propagation_model=propagation_model)

    @classmethod
    def _load_dataframe(cls):
        """Load the DataFrame if not yet loaded."""
        if cls._df is None:
            cls._df = pd.read_csv(cls._filename)
        return cls._df

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

        # Warning if no exact matches found
        if dfi["similarity_score"].iloc[0] > 0:
            warnings.warn(
                f"No exact matches found for material {self.name}. "
                "Material may be invalid.",
                UserWarning,
                stacklevel=2,
            )

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

        if self.min_wavelength or self.max_wavelength:
            wavelength_range = f"({self.min_wavelength}, {self.max_wavelength}) µm"
            message += f" within wavelength range {wavelength_range}"

        raise ValueError(message)

    def _retrieve_file(self):
        """Retrieves the file path for the material based on the given criteria.

        Returns:
            tuple[str, dict]: A tuple containing:
                - The full file path to the material data file.
                - A dictionary containing the material's metadata from the catalog.

        Raises:
            ValueError: If no matches are found for the material.
            ValueError: If multiple matches are found for the material.

        """
        df = self._load_dataframe()
        filtered_df = self._find_material_matches(df)

        if filtered_df.empty:
            self._raise_material_error(no_matches=True)

        if len(filtered_df) > 1 and not self.robust:
            self._raise_material_error(multiple_matches=True)

        material_data = filtered_df.loc[0].to_dict()
        filename = filtered_df.loc[0, "filename"]

        full_filename = str(
            resources.files("optiland.database").joinpath("data-nk", filename),
        )

        return full_filename, material_data

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
                "robust_search": self.robust,
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

        return cls(
            data["name"],
            data.get("reference", None),
            data.get("robust_search", True),
            data.get("min_wavelength", None),
            data.get("max_wavelength", None),
            # propagation_model is handled by MaterialFile.from_dict
        )
