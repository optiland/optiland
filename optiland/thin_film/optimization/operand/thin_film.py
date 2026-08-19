"""Thin Film Operand Module

This module contains the ThinFilmOperand class with static methods for computing
reflectance, transmittance, and absorptance values for thin film stacks.
These operands are designed to work with the optimization framework.

Corentin Nannini, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.thin_film import ThinFilmStack


class ThinFilmOperand:
    """Operand functions for thin film stack optimization.

    This class provides static methods that compute optical properties
    (reflectance, transmittance, absorptance) of thin film stacks.
    These methods are designed to be used as operand functions in the
    optimization framework.
    """

    @staticmethod
    def reflectance(
        stack: ThinFilmStack,
        wavelength_nm: float | list[float],
        aoi_deg: float = 0.0,
        polarization: str = "u",
    ) -> float:
        """Compute reflectance for optimization.

        Args:
            stack: The thin film stack to analyze.
            wavelength_nm: Wavelength(s) in nanometers.
            aoi_deg: Angle of incidence in degrees. Defaults to 0.0.
            polarization: Polarization state ('s', 'p', 'u'). Defaults to 'u'.

        Returns:
            float: The reflectance value. If multiple wavelengths are provided,
                returns the mean reflectance.
        """
        # Convert to arrays for consistency
        wl_nm = be.atleast_1d(wavelength_nm)
        aoi = be.atleast_1d(aoi_deg)

        # Compute reflectance
        R = stack.reflectance_nm_deg(wl_nm, aoi, polarization)

        # Return scalar if single point, mean if multiple points
        if R.size == 1:
            return float(R.item())
        else:
            return float(be.mean(R))

    @staticmethod
    def transmittance(
        stack: ThinFilmStack,
        wavelength_nm: float | list[float],
        aoi_deg: float = 0.0,
        polarization: str = "u",
    ) -> float:
        """Compute transmittance for optimization.

        Args:
            stack: The thin film stack to analyze.
            wavelength_nm: Wavelength(s) in nanometers.
            aoi_deg: Angle of incidence in degrees. Defaults to 0.0.
            polarization: Polarization state ('s', 'p', 'u'). Defaults to 'u'.

        Returns:
            float: The transmittance value. If multiple wavelengths are provided,
                returns the mean transmittance.
        """
        # Convert to arrays for consistency
        wl_nm = be.atleast_1d(wavelength_nm)
        aoi = be.atleast_1d(aoi_deg)

        # Compute transmittance
        T = stack.transmittance_nm_deg(wl_nm, aoi, polarization)

        # Return scalar if single point, mean if multiple points
        if T.size == 1:
            return float(T.item())
        else:
            return float(be.mean(T))

    @staticmethod
    def absorptance(
        stack: ThinFilmStack,
        wavelength_nm: float | list[float],
        aoi_deg: float = 0.0,
        polarization: str = "u",
    ) -> float:
        """Compute absorptance for optimization.

        Args:
            stack: The thin film stack to analyze.
            wavelength_nm: Wavelength(s) in nanometers.
            aoi_deg: Angle of incidence in degrees. Defaults to 0.0.
            polarization: Polarization state ('s', 'p', 'u'). Defaults to 'u'.

        Returns:
            float: The absorptance value. If multiple wavelengths are provided,
                returns the mean absorptance.
        """
        # Convert to arrays for consistency
        wl_nm = be.atleast_1d(wavelength_nm)
        aoi = be.atleast_1d(aoi_deg)

        # Compute absorptance
        A = stack.absorptance_nm_deg(wl_nm, aoi, polarization)

        # Return scalar if single point, mean if multiple points
        if A.size == 1:
            return float(A.item())
        else:
            return float(be.mean(A))

    @staticmethod
    def reflectance_weighted(
        stack: ThinFilmStack,
        wavelengths_nm: list[float],
        weights: list[float],
        aoi_deg: float = 0.0,
        polarization: str = "u",
    ) -> float:
        """Compute weighted reflectance over multiple wavelengths.

        Args:
            stack: The thin film stack to analyze.
            wavelengths_nm: List of wavelengths in nanometers.
            weights: List of weights for each wavelength.
            aoi_deg: Angle of incidence in degrees. Defaults to 0.0.
            polarization: Polarization state ('s', 'p', 'u'). Defaults to 'u'.

        Returns:
            float: The weighted reflectance value.
        """
        wl_nm = be.array(wavelengths_nm)
        w = be.array(weights)
        aoi = be.atleast_1d(aoi_deg)

        # Normalize weights
        w = w / be.sum(w)

        # Compute reflectance
        R = stack.reflectance_nm_deg(wl_nm, aoi, polarization)

        # Return weighted average
        return float(be.sum(R.flatten() * w))

    @staticmethod
    def transmittance_weighted(
        stack: ThinFilmStack,
        wavelengths_nm: list[float],
        weights: list[float],
        aoi_deg: float = 0.0,
        polarization: str = "u",
    ) -> float:
        """Compute weighted transmittance over multiple wavelengths.

        Args:
            stack: The thin film stack to analyze.
            wavelengths_nm: List of wavelengths in nanometers.
            weights: List of weights for each wavelength.
            aoi_deg: Angle of incidence in degrees. Defaults to 0.0.
            polarization: Polarization state ('s', 'p', 'u'). Defaults to 'u'.

        Returns:
            float: The weighted transmittance value.
        """
        wl_nm = be.array(wavelengths_nm)
        w = be.array(weights)
        aoi = be.atleast_1d(aoi_deg)

        # Normalize weights
        w = w / be.sum(w)

        # Compute transmittance
        T = stack.transmittance_nm_deg(wl_nm, aoi, polarization)

        # Return weighted average
        return float(be.sum(T.flatten() * w))

    @staticmethod
    def absorptance_weighted(
        stack: ThinFilmStack,
        wavelengths_nm: list[float],
        weights: list[float],
        aoi_deg: float = 0.0,
        polarization: str = "u",
    ) -> float:
        """Compute weighted absorptance over multiple wavelengths.

        Args:
            stack: The thin film stack to analyze.
            wavelengths_nm: List of wavelengths in nanometers.
            weights: List of weights for each wavelength.
            aoi_deg: Angle of incidence in degrees. Defaults to 0.0.
            polarization: Polarization state ('s', 'p', 'u'). Defaults to 'u'.

        Returns:
            float: The weighted absorptance value.
        """
        wl_nm = be.array(wavelengths_nm)
        w = be.array(weights)
        aoi = be.atleast_1d(aoi_deg)

        # Normalize weights
        w = w / be.sum(w)

        # Compute absorptance
        A = stack.absorptance_nm_deg(wl_nm, aoi, polarization)

        # Return weighted average
        return float(be.sum(A.flatten() * w))
