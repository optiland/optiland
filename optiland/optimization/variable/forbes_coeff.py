"""Specialized variable handlers for Forbes polynomial coefficients.

This module provides handlers for Forbes polynomial coefficients,
distinguishing between rotationally symmetric (slope-orthogonal Q) and
freeform (Q-2D) surfaces.

Manuel Fragata Mendes, August 2025
"""

from __future__ import annotations

import warnings

from optiland.geometries.forbes.geometry import (
    ForbesQ2dGeometry,
    ForbesQNormalSlopeGeometry,
)
from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable.base import VariableBehavior


class ForbesQNormalSlopeCoeffVariable(VariableBehavior):
    """Represents a variable for a Forbes Q (slope-orthogonal) coefficient.

    This variable targets a coefficient for a specific radial order `n`.

    Attributes:
        coeff_number (int): The radial order 'n' of the coefficient.
    """

    def __init__(self, optic, surface_number, coeff_number, scaler=None, **kwargs):
        """Initializes the ForbesQNormalSlopeCoeffVariable.

        Args:
            optic: The optical system.
            surface_number (int): The surface number to which the coefficient belongs.
            coeff_number (int): The radial order 'n' of the coefficient.
            scaler (Scaler, optional): The scaler to use. Defaults to IdentityScaler().
            **kwargs: Additional keyword arguments.
        """
        if scaler is None:
            scaler = IdentityScaler()
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)
        self.coeff_number = coeff_number  # the radial order 'n'

    def get_value(self):
        """Gets the value of the nth Q (slope-orthogonal) coefficient.

        Returns:
            float: The value of the coefficient.

        Raises:
            TypeError: If the geometry is not a ForbesQNormalSlopeGeometry.
        """
        surf = self._surfaces[self.surface_number]
        geom = surf.geometry
        if not isinstance(geom, ForbesQNormalSlopeGeometry):
            raise TypeError("This variable is only for ForbesQNormalSlopeGeometry.")

        return geom.radial_terms.get(self.coeff_number, 0.0)

    def update_value(self, new_value):
        """Updates the value of the nth Q (slope-orthogonal) coefficient.

        Args:
            new_value (float): The new value for the coefficient.
        """
        surf = self.optic.surfaces[self.surface_number]
        geom = surf.geometry

        geom.radial_terms[self.coeff_number] = new_value
        geom._prepare_coeffs()

    def __str__(self):
        return f"Forbes Q Coeff n={self.coeff_number}, Surface {self.surface_number}"


class ForbesQ2dCoeffVariable(VariableBehavior):
    """Represents a variable for a Forbes Q-2D (freeform) coefficient.

    This variable targets the coefficient `c` corresponding to a specific
    (type, m, n) term, following the Zemax convention.

    Attributes:
        attribute (str): The name of the attribute holding the coefficients.
        m (int): The azimuthal frequency `m`.
        n (int): The radial order `n`.
        term_type (str): The type of term ('cos' or 'sin').
        coeff_tuple (tuple): The identifier for the coefficient.
    """

    def __init__(self, optic, surface_number, coeff_tuple, scaler=None, **kwargs):
        """Initializes the ForbesQ2dCoeffVariable.

        Args:
            optic: The optical system.
            surface_number (int): The surface number to which the coefficient belongs.
            coeff_tuple (tuple): The identifier for the coefficient, following the
                Zemax convention: `('a', m, n)` for a cosine term a_n^m, or
                `('b', m, n)` for a sine term b_n^m.
            scaler (Scaler, optional): The scaler to use. Defaults to IdentityScaler().
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If `coeff_tuple` is not in the correct format.
        """
        if scaler is None:
            scaler = IdentityScaler()
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)
        self.attribute = "freeform_coeffs"

        # Parse the Zemax-style key for correct string representation and validation
        try:
            term_char, m, n = coeff_tuple
            self.m = m
            self.n = n
            if term_char.lower() == "a":
                self.term_type = "cos"
            elif term_char.lower() == "b":
                self.term_type = "sin"
            else:
                raise ValueError("Term type in coeff_tuple must be 'a' or 'b'.")
        except (ValueError, TypeError, IndexError) as err:
            raise ValueError(
                "coeff_tuple for ForbesQ2dCoeffVariable must be a tuple of the "
                "form ('a', m, n) or ('b', m, n)."
            ) from err

        self.coeff_tuple = coeff_tuple

    def get_value(self):
        """Gets the value of the coefficient for the specified term.

        Returns:
            float: The value of the coefficient.

        Raises:
            TypeError: If the geometry is not a ForbesQ2dGeometry.
        """
        surf = self._surfaces[self.surface_number]
        geom = surf.geometry
        if not isinstance(geom, ForbesQ2dGeometry):
            raise TypeError("This variable is only for ForbesQ2dGeometry.")

        return geom.freeform_coeffs.get(self.coeff_tuple, 0.0)

    def update_value(self, new_value):
        """Updates the value of the coefficient for the specified term.

        Args:
            new_value (float): The new value for the coefficient.
        """
        surf = self.optic.surfaces[self.surface_number]
        geom = surf.geometry

        geom.freeform_coeffs[self.coeff_tuple] = new_value
        geom._prepare_coeffs()

    def __str__(self):
        """Returns a string representation of the variable."""
        return (
            f"Forbes Q-2D Coeff (n={self.n}, m={self.m}, {self.term_type}), "
            f"Surface {self.surface_number}"
        )


class ForbesQbfsCoeffVariable(ForbesQNormalSlopeCoeffVariable):
    """Deprecated alias for ForbesQNormalSlopeCoeffVariable.

    .. deprecated::
        Use :class:`ForbesQNormalSlopeCoeffVariable` instead.
    """

    def __init__(self, optic, surface_number, coeff_number, scaler=None, **kwargs):
        warnings.warn(
            "ForbesQbfsCoeffVariable is deprecated; use "
            "ForbesQNormalSlopeCoeffVariable instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(optic, surface_number, coeff_number, scaler, **kwargs)

    def __str__(self):
        return (
            f"Forbes Q-bfs Coeff n={self.coeff_number}, Surface {self.surface_number}"
        )
