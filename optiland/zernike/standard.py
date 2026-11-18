"""Standard Zernike Module

This module contains the ZernikeStandard class, which defines Zernike polynomials
based on the "standard" indexing scheme.

References:
    1. https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices
    2. Thibos LN, Applegate RA, Schwiegerling JT, Webb R; VSIA Standards
       Taskforce Members. Vision science and its applications. Standards
       for reporting the optical aberrations of eyes. J Refract Surg. 2002
       Sep-Oct;18(5):S652-60. doi: 10.3928/1081-597X-20020901-30. PMID:
       12361175.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.zernike.base import BaseZernike


class ZernikeStandard(BaseZernike):
    """OSA/ANSI Standard Zernike

    This class represents the OSA/ANSI Standard Zernike polynomials.
    It provides methods to calculate the Zernike terms, Zernike polynomials,
    and other related functions.

    Args:
        coeffs (array-like): The Zernike coefficients. Defaults to None.
        num_terms (int): the maximum number of terms. Only used if coeffs is None.
            Defaults to 36.

    References:
        1. https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/
           ANSI_standard_indices
        2. Thibos LN, Applegate RA, Schwiegerling JT, Webb R; VSIA Standards
           Taskforce Members. Vision science and its applications. Standards
           for reporting the optical aberrations of eyes. J Refract Surg. 2002
           Sep-Oct;18(5):S652-60. doi: 10.3928/1081-597X-20020901-30. PMID:
           12361175.

    """

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
            return (n * (n + 2) + m) // 2

        return None

    @staticmethod
    def _norm_squared(n: int = 0, m: int = 0) -> float:
        """Return the squared normalization constant: n + 1 if m = 0, else 2 n + 2."""
        denominator = 2 if m == 0 else 1
        return (2 * n + 2) / denominator

    @staticmethod
    def _norm_constant(n=0, m=0):
        """Calculate the normalization constant of the Zernike polynomial.

        Args:
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.

        Returns:
            float: The calculated value of the normalization constant.

        """
        return be.sqrt(be.array(ZernikeStandard._norm_squared(n, m)))
