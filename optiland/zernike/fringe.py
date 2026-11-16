"""Fringe Zernike Module

This module contains the FringeZernike class, which defines Zernike polynomials
based on the "fringe", or "University of Arizona", indexing scheme.

References:
    1. https://en.wikipedia.org/wiki/Zernike_polynomials#Fringe/
       University_of_Arizona_indices

Kramer Harrison, 2025
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.zernike.base import BaseZernike


class ZernikeFringe(BaseZernike):
    """Zernike Fringe Coefficients

    This class represents Zernike Fringe Coefficients. It is a subclass of the
    BaseZernike class.

    Args:
        coeffs (list): the coefficient list for the Zernike polynomials.
            Defaults to all zeros (36 elements total)

    References:
        1. https://en.wikipedia.org/wiki/Zernike_polynomials#Fringe/
           University_of_Arizona_indices

    """

    @staticmethod
    def _norm_squared(n: int = 0, m: int = 0) -> float:
        """Return the square of the normalization constant: 1 for all terms."""
        return 1.0

    @staticmethod
    def _norm_constant(n=0, m=0):
        """Calculate the normalization constant for a given Zernike polynomial.
        Note that this is 1 for all terms for Zernike Fringe polynomials.

        Args:
            n (int): The radial order of the Zernike polynomial.
            m (int): The azimuthal order of the Zernike polynomial.

        Returns:
            float: The normalization constant for the Zernike polynomial.

        """
        return be.array(1)

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
            return int(
                (1 + (n + np.abs(m)) / 2) ** 2 - 2 * np.abs(m) + (1 - np.sign(m)) / 2
            )

        return None
