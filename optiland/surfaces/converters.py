"""Converters

This module contains classes that convert between different surface types.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.interactions.thin_lens_interaction_model import ThinLensInteractionModel
from optiland.materials.base import BaseMaterial

if TYPE_CHECKING:
    from optiland.optic.optic import Optic
    from optiland.surfaces.standard_surface import Surface


class ParaxialToThickLensConverter:
    """
    Converts a ParaxialSurface into an equivalent thick lens composed of two
    real surfaces.

    Args:
        paraxial_surface: The ParaxialSurface to convert.
        optic: The parent Optic instance containing the paraxial surface.
        material: The lens material. Can be:
            - A string (e.g., "N-BK7", resolved via Material lookup).
            - A float (refractive index, creates an IdealMaterial).
            - A BaseMaterial instance.
        center_thickness: The desired center thickness of the thick lens.
    """

    def __init__(
        self,
        paraxial_surface: Surface,
        optic: Optic,
        material: str | float | BaseMaterial = "N-BK7",
        center_thickness: float = 3.0,  # Default center thickness in mm
    ):
        if not isinstance(paraxial_surface.interaction_model, ThinLensInteractionModel):
            raise TypeError("paraxial_surface must have a ThinLensInteractionModel.")

        self.paraxial_surface = paraxial_surface
        self.optic = optic
        self.original_focal_length = paraxial_surface.interaction_model.f
        self.center_thickness = center_thickness

        self._material_instance = self._resolve_material(material)

    def _resolve_material(
        self, material_input: str | float | BaseMaterial
    ) -> BaseMaterial:
        """Resolves the material input to a BaseMaterial instance."""
        from optiland.materials.ideal import IdealMaterial
        from optiland.materials.material import Material

        if isinstance(material_input, BaseMaterial):
            return material_input
        elif isinstance(material_input, str):
            try:
                return Material(material_input)
            except Exception as e:
                raise ValueError(
                    f"Could not resolve material string '{material_input}': {e}"
                ) from e
        elif isinstance(material_input, int | float):
            return IdealMaterial(n=float(material_input))
        else:
            raise TypeError(
                "Invalid material type. Must be BaseMaterial, str, or float."
            )

    def convert(self):
        """
        Performs the conversion from paraxial to thick lens.

        This method will:
        1. Calculate the front and back radii of the thick lens.
        2. Remove the original paraxial surface from the optic.
        3. Create and add two new surfaces to the optic.
        """
        r1, r2 = self._calculate_radii()

        # Store original index before removal
        original_index = self._get_paraxial_surface_index()
        if original_index is None:
            raise RuntimeError("Original paraxial surface not found in optic.")

        self._remove_paraxial_surface(original_index)
        self._add_surfaces(r1, r2, original_index)

    def _get_paraxial_surface_index(self):
        """Finds the index of the self.paraxial_surface in the optic's surface list."""
        for i, s in enumerate(self.optic.surfaces):
            if s is self.paraxial_surface:
                return i
        return None

    def _paraxial_index(self) -> float:
        """Resolves the lens material's refractive index at the primary
        wavelength to a plain Python float."""
        n = self._material_instance.n(self.optic.primary_wavelength)
        if hasattr(n, "item"):  # If n is a 0-dim array/tensor
            n = n.item()
        return n

    @staticmethod
    def _solve_symmetric_radius(
        a_quad: float, b_quad: float, c_quad: float, positive: bool, label: str
    ) -> float:
        """Solves P*n*R1^2 - 2*n*(n-1)*R1 + (n-1)^2*d = 0 for R1, selecting the
        root with the sign convention required by `positive` (True for
        biconvex R1 > 0, False for biconcave R1 < 0).

        Args:
            a_quad, b_quad, c_quad: Quadratic coefficients.
            positive: Whether the desired root must be positive (else negative).
            label: "Biconvex" or "Biconcave", used in error messages.

        Returns:
            float: The selected root R1.
        """

        def _matches(x: float) -> bool:
            return x > 0 if positive else x < 0

        if abs(a_quad) < 1e-9:
            if abs(b_quad) < 1e-9:
                raise ValueError(f"Cannot solve for R1 in {label.lower()} (P=0, n=1).")
            return -c_quad / b_quad  # Linear case

        discriminant = b_quad**2 - 4 * a_quad * c_quad
        if discriminant < 0:
            raise ValueError(f"{label}: discriminant < 0, cannot find real R1.")

        sol1 = (-b_quad + be.sqrt(discriminant)) / (2 * a_quad)
        sol2 = (-b_quad - be.sqrt(discriminant)) / (2 * a_quad)
        r1 = sol1 if _matches(sol1) else sol2
        if not _matches(r1):
            r1 = sol2 if _matches(sol2) else sol1
            if not _matches(r1):
                sign_word = "positive" if positive else "negative"
                raise ValueError(f"{label}: No {sign_word} R1 solution found.")
        return r1

    def _calculate_radii(self):
        """
        Calculates the front (R1) and back (R2) radii of curvature for the
        thick lens using the Lensmaker's equation.

        P = (n_lens - n_medium) * (1/R1 - 1/R2 +
                                   (n_lens - n_medium)*d / (n_lens*R1*R2))
        where P = 1/f (power), n_lens is lens refractive index, n_medium is
        surrounding medium refractive index (assumed air, n_medium=1), and d is
        center thickness.

        For a target focal length f_target (self.original_focal_length),
        and assuming n_medium = 1 (air):
        1/f_target = (n - 1) * (1/R1 - 1/R2 + (n - 1)*d / (n*R1*R2))

        This method uses a biconvex lens for positive focal lengths and a biconcave
        lens for negative focal lengths.
        - biconvex: R1 > 0, R2 < 0. Assume R1 = -R2 for simplicity.
        - biconcave: R1 < 0, R2 > 0. Assume R1 = -R2.

        Returns:
            tuple[float, float]: (R1, R2)
        """
        n = self._paraxial_index()
        f_target = self.original_focal_length
        d = self.center_thickness

        if abs(f_target) < 1e-9:
            return be.inf, be.inf

        # Biconvex/biconcave: P*n*R1^2 - 2*n*(n-1)*R1 + (n-1)^2*d = 0. For R1 = -R2.
        P = 1.0 / f_target  # Power
        a_quad = P * n
        b_quad = -2 * n * (n - 1)
        c_quad = (n - 1) ** 2 * d

        if f_target > 0:
            r1 = self._solve_symmetric_radius(
                a_quad, b_quad, c_quad, positive=True, label="Biconvex"
            )
        else:
            r1 = self._solve_symmetric_radius(
                a_quad, b_quad, c_quad, positive=False, label="Biconcave"
            )

        return float(r1), float(-r1)

    def _add_surfaces(self, r1: float, r2: float, original_index: int):
        """
        Creates the two new standard Surface instances.
        Materials pre/post are set based on original paraxial surface context
        and the new lens material.
        """
        original_material_post = self.paraxial_surface.material_post

        # Surface 1: front surface of the thick lens
        self.optic.surfaces.add(
            index=original_index,
            radius=r1,
            material=self._material_instance,
            is_stop=self.paraxial_surface.is_stop,
            thickness=self.center_thickness,
            comment="Thick Lens - Surface 1",
        )

        # Surface 2: back surface of the thick lens
        self.optic.surfaces.add(
            index=original_index + 1,
            radius=r2,
            material=original_material_post,
            is_stop=False,  # Stop, if any, is on the first surface
            thickness=self.paraxial_surface.thickness,
            comment="Thick Lens - Surface 2",
        )

    def _remove_paraxial_surface(self, original_index: int):
        """
        Removes the original ParaxialSurface from the parent optic's
        surface_group using its index.
        """
        if not (0 < original_index < len(self.optic.surfaces)):
            raise IndexError(
                f"Invalid index {original_index} for removing paraxial surface."
            )
        self.optic.surfaces.remove(original_index)


def convert_to_thick_lens(lens: Optic):
    """
    Converts all paraxial surfaces in a lens into thick lenses

    Args:
        lens (Optic): the lens to be converted

    Returns:
        Optic: the converted lens
    """
    for surf in lens.surfaces:
        if isinstance(surf.interaction_model, ThinLensInteractionModel):
            converter = ParaxialToThickLensConverter(surf, lens)
            converter.convert()
    return lens
