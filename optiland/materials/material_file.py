"""Material File

This module contains a class for representing a material based on a material
YAML file from the refractiveindex.info database.

Kramer Harrison, 2024
"""

from __future__ import annotations

import contextlib
import os
from io import StringIO

import numpy as np
import yaml

import optiland.backend as be
from optiland.materials.base import BaseMaterial


class MaterialFile(BaseMaterial):
    """Represents a material based on a material YAML file from the
    refractiveindex.info database.

    Material refractive indices are based on various dispersion formulas or
    tabulated data. The material file contains the coefficients for the
    dispersion formulas and/or tabulated data.

    See https://refractiveindex.info/database/doc/Dispersion%20formulas.pdf

    Args:
        filename (str): The path to the material file.
        propagation_model (BasePropagationModel, optional): The propagation
            model to use. Defaults to None, which creates a
            HomogeneousPropagation model.

    Attributes:
        filename (str): The filename of the material file.
        coefficients (list): A list of coefficients for calculating the
            refractive index.

    Methods:
        n(wavelength, temperature, pressure): Calculates the refractive index of the
            material at a given wavelength, temperature and pressure.
        k(wavelength): Retrieves the extinction coefficient of the material at
            a given wavelength.

    """

    def __init__(self, filename, propagation_model=None):
        super().__init__(propagation_model)
        self.filename = filename
        self._k_warning_printed = False
        self.coefficients = []
        self.thermdispcoef = []
        self._k_wavelength = None
        self._k = None
        self._n_formula = None
        self._n_wavelength = None
        self._n = None
        self._t0 = None
        self.reference_data = None

        self.formula_map = {
            "formula 1": self._formula_1,
            "formula 2": self._formula_2,
            "formula 3": self._formula_3,
            "formula 4": self._formula_4,
            "formula 5": self._formula_5,
            "formula 6": self._formula_6,
            "formula 7": self._formula_7,
            "formula 8": self._formula_8,
            "formula 9": self._formula_9,
            "tabulated n": self._tabulated_n,
            "tabulated nk": self._tabulated_n,
        }

        data = self._read_file()
        self._parse_file(data)

    def _calculate_n(self, wavelength, **kwargs):
        """Calculates the refractive index of the material at given wavelengths.

        The method first calculates the refractive index from the dispersion formula,
        which is assumed to be relative to air at a reference temperature.
        If system temperature and pressure are provided, it applies a
        correction based on the material's thermal dispersion coefficients.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) in microns.
            temperature (float): The material temperature in Celsius. If None, no
                thermal correction is applied.
            pressure (float): The ambient pressure in atmospheres (atm). Defaults to
                1.0 atm in calculations if not provided but temperature is.

        Returns:
            float or be.ndarray: The refractive index(s) of the material.
        """
        temperature = kwargs.get("temperature")
        pressure = kwargs.get("pressure")
        # Apply environmental corrections only if temperature data is available.
        if (
            temperature is not None
            and self._t0 is not None
            and be.any(be.array(self.thermdispcoef))
        ):
            pressure = 1.0 if pressure is None else pressure

            # Calculate the 'relative' wavelength which is input wavelength scaled
            # to reference temperature and pressure
            waverel = (
                wavelength
                * self._nair(wavelength, temperature, pressure)
                / self._nair(wavelength, self._t0, 1.0)
            )
            # Calculate the baseline refractive index. This is relative to air at the
            # reference temperature (self._t0) and for 'relative' wavelength.
            base_relative_n = self.formula_map[self._n_formula](waverel)
            return self._apply_environmental_correction(
                base_relative_n, wavelength, temperature, pressure
            )
        else:
            # If no temperature is given or material has no thermal data,
            # return the catalog value.
            # Calculate the baseline refractive index. This is relative to air at the
            # reference temperature (self._t0).
            base_relative_n = self.formula_map[self._n_formula](wavelength)
            return base_relative_n

    def _apply_environmental_correction(
        self, base_relative_n, wavelength, temp_c, pressure_atm
    ):
        """Applies temperature and pressure corrections to a refractive index.

        This method encapsulates the physics of converting from a relative
        index to absolute, applying thermal changes, and converting back to the
        new relative index.

        Args:
            base_relative_n (float or be.ndarray): The catalog refractive index
                relative to air at reference temperature.
            wavelength (float or be.ndarray): The wavelength(s) in microns.
            temp_c (float): The system temperature in Celsius.
            pressure_atm (float): The system pressure in atmospheres.

        Returns:
            float or be.ndarray: The corrected refractive index relative to the
            ambient conditions.
        """
        # If pressure is not specified, assume standard atmosphere
        if pressure_atm is None:
            pressure_atm = 1.0

        # Compute the index of air at the material's reference temperature and standard
        # pressure (1 atm)
        n_air_reference = self._nair(wavelength, self._t0, 1.0)

        # Compute the absolute index of the material (relative to vacuum) at its
        # reference temperature
        n_absolute_reference = base_relative_n * n_air_reference

        # Compute the change in the absolute index due to the temperature difference
        c = self.thermdispcoef
        delta_t = temp_c - self._t0

        # This is the Schott formula for the change in absolute refractive index
        term1 = c[0] + c[1] * delta_t + c[2] * delta_t**2
        term2 = (c[3] + c[4] * delta_t) / (wavelength**2 - c[5] ** 2)
        dn_abs = (
            (n_absolute_reference**2 - 1.0)
            / (2.0 * n_absolute_reference)
            * (term1 + term2)
            * delta_t
        )

        n_absolute_corrected = n_absolute_reference + dn_abs

        # Compute the index of air at the new system temperature and pressure.
        n_air_system = self._nair(wavelength, temp_c, pressure_atm)

        # Compute the final index of the material relative to the ambient air at the
        # new system conditions.
        final_relative_n = n_absolute_corrected / n_air_system

        return final_relative_n

    def _nair(self, wavelength_um, temp_c, pressure_atm=1.0):
        """Computes the refractive index of air

        This formula is a variant of the Edlén equation for the dispersion of air.

        Args:
            wavelength_um (float or be.ndarray): The wavelength(s) in microns.
            temp_c (float): The temperature of air in degrees Celsius.
            pressure_atm (float): The relative pressure in atmospheres (atm).
                Defaults to 1.0.

        Returns:
            float or be.ndarray: The refractive index of air.
        """
        AIR_REF_TEMP_C = 15.0
        AIR_THERMAL_COEFF = 0.0034785  # Corresponds to 3.4785 / 1000

        # Calculate (n-1) for air at the reference temperature (15°C)
        # and 1 atm pressure using the dispersion formula.
        w2 = wavelength_um**2
        n_ref_minus_1 = (
            6432.8 + (2949810 * w2) / (146 * w2 - 1) + (25540 * w2) / (41 * w2 - 1)
        ) * 1e-8

        # Adjust for the actual system temperature and pressure.
        n_air = 1.0 + (n_ref_minus_1 * pressure_atm) / (
            1.0 + (temp_c - AIR_REF_TEMP_C) * AIR_THERMAL_COEFF
        )

        return n_air

    def _calculate_k(self, wavelength, **kwargs):
        """Retrieves the extinction coefficient of the material at a
        given wavelength. If no exxtinction coefficient data is found, it is
        assumed to be 0 and prints a warning message, only once.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) in microns.

        Returns:
            float or be.ndarray: The extinction coefficient of the material. If no
            data is available, returns 0.0 or an array of zeros.

        """
        # If the extinction coefficient is missing from the file, return 0
        if self._k is None or self._k_wavelength is None:
            if not self._k_warning_printed:
                material_name = os.path.basename(self.filename)
                print(
                    f"WARNING: No extinction coefficient data found "
                    f"for {material_name}. Assuming it is 0.",
                )

                # we set it to True to avoid printing the warning again
                self._k_warning_printed = True

            if be.is_array_like(wavelength):
                return be.zeros_like(wavelength)
            return 0.0

        return be.interp(wavelength, self._k_wavelength, self._k)

    def _formula_1(self, w):
        """Calculate the refractive index using dispersion formula 1 from
        refractiveindex.info (Sellmeier formula).

        Args:
            w (float or be.ndarray): The wavelength(s) in microns.

        Returns:
            float or be.ndarray: The refractive index(s) of the material.

        """
        c = self.coefficients
        try:
            n = 1 + c[0]
            for k in range(1, len(c), 2):
                n = n + c[k] * w**2 / (w**2 - c[k + 1] ** 2)
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 1.") from err
        return be.sqrt(n)

    def _formula_2(self, w):
        """Calculate the refractive index using dispersion formula 2 from
        refractiveindex.info (Sellmeier-2 formula).

        Args:
            w (float or be.ndarray): The wavelength(s) in microns.

        Returns:
            float or be.ndarray: The refractive index(s) of the material.

        """
        c = self.coefficients
        try:
            n = 1 + c[0]
            for k in range(1, len(c), 2):
                n = n + c[k] * w**2 / (w**2 - c[k + 1])
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 2.") from err
        return be.sqrt(n)

    def _formula_3(self, w):
        """Calculate the refractive index using dispersion formula 3 from
        refractiveindex.info (Polynomial formula).

        Args:
            w (float or be.ndarray): The wavelength(s) in microns.

        Returns:
            float or be.ndarray: The refractive index(s) of the material.

        """
        c = self.coefficients
        try:
            n = c[0]
            for k in range(1, len(c), 2):
                n = n + c[k] * w ** c[k + 1]
            return be.sqrt(n)
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 3.") from err

    def _formula_4(self, w):
        """Calculate the refractive index using dispersion formula 4 from
        refractiveindex.info (RefractiveIndex.INFO formula).

        Args:
            w (float or be.ndarray): The wavelength(s) in microns.

        Returns:
            float or be.ndarray: The refractive index(s) of the material.

        """
        c = self.coefficients
        try:
            n = (
                c[0]
                + c[1] * w ** c[2] / (w**2 - c[3] ** c[4])
                + c[5] * w ** c[6] / (w**2 - c[7] ** c[8])
            )
            for k in range(9, len(c), 2):
                n = n + c[k] * w ** c[k + 1]
            return be.sqrt(n)
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 4.") from err

    def _formula_5(self, w):
        """Calculate the refractive index using dispersion formula 5 from
        refractiveindex.info (Cauchy formula).

        Args:
            w (float or be.ndarray): The wavelength(s) in microns.

        Returns:
            float or be.ndarray: The refractive index(s) of the material.

        """
        c = self.coefficients
        try:
            n = c[0]
            for k in range(1, len(c), 2):
                n = n + c[k] * w ** c[k + 1]
            return n
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 5.") from err

    def _formula_6(self, w):
        """Calculate the refractive index using dispersion formula 6 from
        refractiveindex.info (Gases formula).

        Args:
            w (float or be.ndarray): The wavelength(s) in microns.

        Returns:
            float or be.ndarray: The refractive index(s) of the material.

        """
        c = self.coefficients
        try:
            n = 1 + c[0]
            for k in range(1, len(c), 2):
                n = n + c[k] / (c[k + 1] - w**-2)
            return n
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 6.") from err

    def _formula_7(self, w):
        """Calculate the refractive index using dispersion formula 7 from
        refractiveindex.info (Herzberger formula).

        Args:
            w (float or be.ndarray): The wavelength(s) in microns.

        Returns:
            float or be.ndarray: The refractive index(s) of the material.

        """
        c = self.coefficients
        try:
            n = c[0] + c[1] / (w**2 - 0.028) + c[2] * (1 / (w**2 - 0.028)) ** 2
            for k in range(3, len(c)):
                n = n + c[k] * w ** (2 * (k - 2))
            return n
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 7.") from err

    def _formula_8(self, w):
        """Calculate the refractive index using dispersion formula 8 from
        refractiveindex.info (Retro formula).

        Args:
            w (float or be.ndarray): The wavelength(s) in microns.

        Returns:
            float or be.ndarray: The refractive index(s) of the material.

        """
        c = self.coefficients
        if len(c) != 4:
            raise ValueError("Invalid coefficients for dispersion formula 8.")

        b = c[0] + c[1] * w**2 / (w**2 - c[2]) + c[3] * w**2
        return be.sqrt((1 + 2 * b) / (1 - b))

    def _formula_9(self, w):
        """Calculate the refractive index using dispersion formula 9 from
        refractiveindex.info (Exotic formula).

        Args:
            w (float or be.ndarray): The wavelength(s) in microns.

        Returns:
            float or be.ndarray: The refractive index(s) of the material.

        """
        c = self.coefficients
        if len(c) != 6:
            raise ValueError("Invalid coefficients for dispersion formula 9.")

        n = c[0] + c[1] / (w**2 - c[2]) + c[3] * (w - c[4]) / ((w - c[4]) ** 2 + c[5])
        return be.sqrt(n)

    def _tabulated_n(self, w):
        """Calculate the refractive index using tabulated data.

        Args:
            w (float or be.ndarray): The wavelength(s) in microns.

        Returns:
            float or be.ndarray: Interpolated refractive index(s).
        """
        try:
            return be.interp(w, self._n_wavelength, self._n)
        except ValueError as err:  # Typically if _n_wavelength or _n is None or empty
            raise ValueError(
                "No tabular refractive index data found or data is invalid."
            ) from err

    def _read_file(self) -> dict:
        """Read the material YAML file.

        Returns:
            dict: Parsed YAML data.
        """
        with open(self.filename) as stream:
            return yaml.safe_load(stream)

    def _set_formula_type(self, formula_type):
        """Set the refractive index formula type."""
        if self._n_formula is None:
            self._n_formula = formula_type
        else:
            raise ValueError("Multiple refractive index formulas found.")

    def _parse_file(self, data: dict) -> None:
        """Parse a material file's structured data.

        Args:
            data (dict): Dictionary containing parsed JSON/YAML material data.
        """
        for sub_data in data.get("DATA", []):
            self._parse_sub_data(sub_data)

        self._parse_thermal_dispersion(data)
        self._parse_reference(data)

    def _parse_sub_data(self, sub_data: dict) -> None:
        """Parse a single DATA block from the material file."""
        sub_data_type = sub_data.get("type", "")

        if sub_data_type.startswith("formula "):
            self._parse_formula_data(sub_data, sub_data_type)
        elif sub_data_type.startswith("tabulated"):
            self._parse_tabulated_data(sub_data, sub_data_type)

    def _parse_formula_data(self, sub_data: dict, sub_data_type: str) -> None:
        """Parse formula-based material data."""
        coeff_values = [float(k) for k in sub_data.get("coefficients", "").split()]
        # Build a 2D column array directly from Python values so the result is
        # always a graph-leaf (avoids a non-leaf view from reshape on a 1-D tensor).
        self.coefficients = be.array([[c] for c in coeff_values])
        self._set_formula_type(sub_data_type)

    def _parse_tabulated_data(self, sub_data: dict, sub_data_type: str) -> None:
        """Parse tabulated material data."""
        data_file = StringIO(sub_data.get("data", ""))
        numpy_arr = np.loadtxt(data_file)
        arr = be.asarray(numpy_arr)

        # Ensure at least 2D shape for consistent indexing
        if arr.ndim == 1:
            arr = be.reshape(arr, (1, -1) if arr.shape[0] > 0 else (0, 0))

        if sub_data_type == "tabulated n":
            self._n_wavelength, self._n = arr[:, 0], arr[:, 1]
            self._set_formula_type(sub_data_type)
        elif sub_data_type == "tabulated k":
            self._k_wavelength, self._k = arr[:, 0], arr[:, 1]
        elif sub_data_type == "tabulated nk":
            self._n_wavelength = self._k_wavelength = arr[:, 0]
            self._n, self._k = arr[:, 1], arr[:, 2]
            self._set_formula_type(sub_data_type)

    def _parse_thermal_dispersion(self, data: dict) -> None:
        """Parse thermal dispersion and reference temperature data."""
        try:
            coeff = data["SPECS"]["thermal_dispersion"][0]
            if coeff.get("type", "").startswith("Schott"):
                td_values = [float(k) for k in coeff.get("coefficients", "").split()]
                self.thermdispcoef = be.array([[c] for c in td_values])

            self._t0 = float(data["SPECS"]["temperature"].split(" ")[0])
        except KeyError:
            pass  # Optional field, safe to ignore

    def _parse_reference(self, data: dict) -> None:
        """Parse optional reference information."""
        with contextlib.suppress(KeyError):
            self.reference_data = data["REFERENCE"]

    def to_dict(self):
        """Returns the material data as a dictionary.

        Returns:
            dict: The material data.

        """
        material_dict = super().to_dict()
        material_dict.update(
            {
                "filename": self.filename,
            },
        )

        return material_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a material from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the material.

        Returns:
            MaterialFile: An instance of MaterialFile.

        """
        if "filename" not in data:
            raise ValueError("Material file data missing filename.")

        return cls(data["filename"])
