"""Noll Zernike Module

This module contains the NollZernike class, which defines Zernike polynomials
based on the "Noll" indexing scheme. Note that the Noll notation is used for the
"Zernike Standard Coefficients" in Ansys Zemax OpticStudio.

References:
    1. https://en.wikipedia.org/wiki/
       Zernike_polynomials#Noll's_sequential_indices
    2. Noll, R. J. (1976). "Zernike polynomials and atmospheric
       turbulence". J. Opt. Soc. Am. 66 (3): 207

Kramer Harrison, 2025
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.zernike.base import BaseZernike


class ZernikeNoll(BaseZernike):
    """Zernike Coefficients - Noll Standard

    This class represents Zernike Noll Coefficients. It is a subclass of the
    BaseZernike class. Note that the Noll notation is used for the
    "Zernike Standard Coefficients" in Ansys Zemax OpticStudio.

    Args:
        coeffs (list): the coefficient list for the Zernike polynomials.
            Defaults to all zeros (36 elements total)

    References:
        1. https://en.wikipedia.org/wiki/
           Zernike_polynomials#Noll's_sequential_indices
        2. Noll, R. J. (1976). "Zernike polynomials and atmospheric
           turbulence". J. Opt. Soc. Am. 66 (3): 207

    """

    @staticmethod
    def _norm_squared(n: int = 0, m: int = 0) -> float:
        """Return the squared normalization constant: n + 1 if m = 0, else 2 n + 2."""
        denominator = 2 if m == 0 else 1
        return (2 * n + 2) / denominator

    @staticmethod
    def _norm_constant(n=0, m=0):
        """Calculate the normalization constant for a given Zernike polynomial.

        Args:
            n (int): The radial order of the Zernike polynomial.
            m (int): The azimuthal order of the Zernike polynomial.

        Returns:
            float: The normalization constant for the Zernike polynomial.

        """
        return be.sqrt(be.array(ZernikeNoll._norm_squared(n, m)))

    @staticmethod
    def _index_to_number(n: int, m: int) -> int | None:
        """Convert Zernike indices (n, m) to a coefficient number.

        Args:
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.

        Returns:
            int: The coefficient number corresponding to the Zernike indices.
        """
        if (n - m) % 2 == 0:
            mod = n % 4
            if (m > 0 and mod <= 1) or (m < 0 and mod >= 2):
                c = 0
            elif (m >= 0 and mod >= 2) or (m <= 0 and mod <= 1):
                c = 1

            return int(n * (n + 1) / 2 + np.abs(m) + c)

        return None
