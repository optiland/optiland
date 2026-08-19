"""Base Zernike Module

This module contains the abstract base class for all zernike-related classes.
The Zernike implementation in this module is based on  Niu, K., & Tian, C. (2022).
Zernike polynomials and their applications. Journal of Optics, 24(12), 123001.
https://doi.org/10.1088/2040-8986/ac9e08

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    from optiland._types import ScalarOrArray

_ZernikeIndex = np.dtype([("n", int), ("m", int)])


class BaseZernike(ABC):
    """
    Abstract base class for Zernike polynomials.

    Args:
        coeffs (array-like): The Zernike coefficients. Defaults to None.
        num_terms (int): the maximum number of terms. Only used if coeffs is None.
            Defaults to 36.
    """

    _indices_cache: ClassVar[np.ndarray | None] = None

    def __init__(self, coeffs=None, num_terms=36):
        self.coeffs = be.zeros([num_terms]) if coeffs is None else coeffs
        self.indices = self._generate_indices(len(self.coeffs))

    def get_term(
        self,
        coeff: ScalarOrArray = 0,
        n: int = 0,
        m: int = 0,
        r: ScalarOrArray = 0,
        phi: ScalarOrArray = 0,
    ) -> ScalarOrArray:
        """Calculate the Zernike term for given coefficients and parameters.

        Args:
            coeff (float): Coefficient value for the Zernike term.
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.
            r (float): Radial distance from the origin.
            phi (float): Azimuthal angle in radians.

        Returns:
            float: The calculated value of the Zernike term.

        """
        return (
            coeff
            * self._norm_constant(n, m)
            * self._radial_term(n, m, r)
            * self._azimuthal_term(m, phi)
        )

    def terms(self, r: ScalarOrArray = 0, phi: ScalarOrArray = 0) -> list:
        """Calculate the Zernike terms for given radial distance and azimuthal
        angle.

        Args:
            r (float): Radial distance from the origin.
            phi (float): Azimuthal angle in radians.

        Returns:
            list: List of calculated Zernike term values.

        """
        val = []

        for coeff, idx in zip(self.coeffs, self.indices, strict=True):
            n, m = idx
            val.append(self.get_term(coeff, n, m, r, phi))

        return val

    def poly(self, r: ScalarOrArray = 0, phi: ScalarOrArray = 0) -> float:
        """Calculate the Zernike polynomial for given radial distance and
        azimuthal angle.

        Args:
            r (float): Radial distance from the origin.
            phi (float): Azimuthal angle in radians.

        Returns:
            float: The calculated value of the Zernike polynomial.

        """
        return sum(self.terms(r, phi))

    def get_derivative(self, n=0, m=0, r=0, phi=0):
        """Calculate the derivative of the Zernike polynomial for the given
        coefficients and parameters.

        Returns a tuple of the radial (dZ / dr) and azimuthal (dZ / dphi) partial
        derivatives of the Zernike polynomial.

        Args:
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.
            r (float): Radial distance from the origin.
            phi (float): Azimuthal angle in radians.

        Returns:
            tuple[float, float]: The radial and azimuthal derivatives of the
                Zernike polynomial.
        """
        radial_term = self._radial_term(n, abs(m), r)
        radial_derivative = self._radial_derivative(n, abs(m), r)

        if m == 0:
            partial_radial_derivative = radial_derivative
            partial_azimuthal_derivative = 0.0
        elif m > 0:
            partial_radial_derivative = radial_derivative * be.cos(m * phi)
            partial_azimuthal_derivative = -m * radial_term * be.sin(m * phi)
        else:  # m < 0
            partial_radial_derivative = radial_derivative * be.sin(be.abs(m) * phi)
            partial_azimuthal_derivative = (
                be.abs(m) * radial_term * be.cos(be.abs(m) * phi)
            )

        return partial_radial_derivative, partial_azimuthal_derivative

    @classmethod
    def _generate_indices(cls, n_indices: int) -> np.ndarray:
        """Generate the indices for Zernike terms.

        Args:
            n_indices (int): The number of indices to generate.

        Returns:
            list: List of tuples representing the indices (n, m) of the
                Zernike terms.
        """
        if cls._indices_cache is not None and len(cls._indices_cache) >= n_indices:
            return cls._indices_cache[:n_indices]

        numbers_present = np.full(n_indices + 1, False)
        # Set the first element to True if the notation is one-indexed
        numbers_present[0] = cls._index_to_number(0, 0) != 0
        number = []
        indices = []

        n = 0
        m = -n

        while not all(numbers_present):
            _number = cls._index_to_number(n, m)

            if _number is not None:
                number.append(_number)
                indices.append((n, m))

                if _number <= n_indices:
                    numbers_present[_number] = True

            if m == n:
                n += 1
                m = -n
            else:
                m += 1

        # sort indices according to scheme-specific coefficient number
        indices_sorted = np.array(
            [element for _, element in sorted(zip(number, indices, strict=False))],
            dtype=_ZernikeIndex,
        )[:n_indices]

        cls._indices_cache = indices_sorted
        return indices_sorted

    @staticmethod
    @abstractmethod
    def _index_to_number(n: int, m: int) -> int | None:
        """Convert Zernike indices (n, m) to a coefficient number.

        Args:
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.

        Returns:
            int: The coefficient number corresponding to the Zernike indices.
        """
        # pragma: no cover

    @staticmethod
    @abstractmethod
    def _norm_constant(n: int, m: int) -> float:
        """Calculate the normalization constant of the Zernike polynomial.

        Args:
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.

        Returns:
            float: The calculated value of the normalization constant.

        """
        # pragma: no cover

    @staticmethod
    def _radial_term(n, m, r):
        """Calculate the radial term of the Zernike polynomial."""
        n = int(n)
        m_abs = abs(int(m))

        s_max = (n - m_abs) // 2 + 1
        r = be.array(r)

        # Initialize value with correct backend
        value = be.zeros_like(r) if not isinstance(r, int | float) else 0.0

        for k in range(s_max):
            num = be.factorial(n - k)
            denom = (
                be.factorial(k)
                * be.factorial((n + m_abs) // 2 - k)
                * be.factorial((n - m_abs) // 2 - k)
            )
            coeff = (-1) ** k * num / denom
            term = coeff * (r ** (n - 2 * k))
            value = value + term

        return value

    @staticmethod
    def _azimuthal_term(m: int = 0, phi: ScalarOrArray = 0):
        """Calculate the azimuthal term of the Zernike polynomial.

        Args:
            m (int): Azimuthal order of the Zernike term.
            phi (float): Azimuthal angle in radians.

        Returns:
            float: The calculated value of the azimuthal term.

        """
        m = be.array(m)
        phi = be.array(phi)

        if m >= 0:
            return be.cos(m * phi)
        return be.sin(be.abs(m) * phi)

    @staticmethod
    def _radial_derivative(n, m, r):
        """Calculate the derivative of the radial term with respect to r.

        R_n^m(rho) = sum_{k=0}^{(n - m)/2} (-1)^k * (n-k)! /
                     [k! ((n+m)/2 - k)! ((n-m)/2 - k)!] * rho^(n - 2k)

        Args:
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.
            r (float): Radial distance from the origin.

        Returns:
            float: The calculated value of the radial derivative.
        """
        n = int(n)
        m = abs(int(m))

        s_max = (n - m) // 2 + 1
        r = be.array(r)

        # Initialize value with correct backend
        value = be.zeros_like(r) if not isinstance(r, int | float) else 0.0

        for k in range(s_max):
            numerator = be.factorial(n - k)
            denominator = (
                be.factorial(k)
                * be.factorial((n + m) // 2 - k)
                * be.factorial((n - m) // 2 - k)
            )
            factor = n - 2 * k

            if factor < 0:
                continue

            power_term = r ** (n - 2 * k - 1) if (n - 2 * k - 1) >= 0 else 0
            value = value + (-1) ** k * (numerator / denominator) * factor * power_term

        return value
