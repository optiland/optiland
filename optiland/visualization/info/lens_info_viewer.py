"""Lens Info Viewer Module

Provides a class for viewing detailed information about an optical system.
This module contains the `LensInfoViewer` class, which is designed to
present a tabular summary of the properties of each surface in an optical
system. This includes surface type, radius, thickness, material, conic
constant, and semi-aperture.

Kramer Harrison, 2024

re-worked by Manuel Fragata Mendes, june 2025
"""

from __future__ import annotations

import pandas as pd

import optiland.backend as be
from optiland.geometries import (
    ChebyshevPolynomialGeometry,
    EvenAsphere,
    OddAsphere,
    PolynomialGeometry,
    ZernikePolynomialGeometry,
)
from optiland.physical_apertures import RadialAperture
from optiland.visualization.base import BaseViewer
from optiland.visualization.info.material_formatter import MaterialFormatter


class LensInfoViewer(BaseViewer):
    """A class for viewing information about a lens.

    Args:
        optic (Optic): The optic object containing the lens information.

    Attributes:
        optic (Optic): The optic object containing the lens information.

    Methods:
        view(): Prints the lens information in a tabular format.

    """

    def __init__(self, optic):
        self.optic = optic

    def view(self):
        """Prints the lens information in a tabular format.

        The lens information includes the surface type, radius, thickness,
        material, conic, and semi-aperture of each surface.
        """
        self.optic.updater.update_paraxial()

        surf_type = self._get_surface_types()
        comments = self._get_comments()
        radii = be.to_numpy(self.optic.surfaces.radii)
        thicknesses = self._get_thicknesses()
        conic = be.to_numpy(self.optic.surfaces.conic)
        semi_aperture = self._get_semi_apertures()
        mat = self._get_materials()

        self.optic.updater.update_paraxial()

        df = pd.DataFrame(
            {
                "Type": surf_type,
                "Comment": comments,
                "Radius": radii,
                "Thickness": thicknesses,
                "Material": mat,
                "Conic": conic,
                "Semi-aperture": semi_aperture,
            },
        )
        print(df.to_markdown(headers="keys", tablefmt="fancy_outline"))

        # Aspheric coefficients
        rows, headers = self._get_aspheric_coefficients()
        df = pd.DataFrame(rows, columns=headers)
        print(df.to_markdown(index=False, tablefmt="fancy_outline", floatfmt=".4g"))

    def _get_surface_types(self):
        """Extracts and formats the surface types."""
        surf_type = []
        for surf in self.optic.surfaces:
            g = surf.geometry

            # Check if __str__ method exists
            if type(g).__dict__.get("__str__"):
                surf_type.append(str(surf.geometry))
            else:
                raise ValueError("Unknown surface type")

            if surf.is_stop:
                surf_type[-1] = "Stop - " + surf_type[-1]
        return surf_type

    def _get_comments(self):
        """Extracts comments for each surface."""
        return [surf.comment for surf in self.optic.surfaces]

    def _get_thicknesses(self):
        """Calculates thicknesses between surfaces."""
        thicknesses = be.diff(
            be.ravel(self.optic.surfaces.positions), append=be.array([be.nan])
        )
        return be.to_numpy(thicknesses)

    def _get_semi_apertures(self):
        """Extracts semi-aperture values for each surface."""
        semi_apertures = []
        for surf in self.optic.surfaces:
            if isinstance(surf.aperture, RadialAperture):
                semi_apertures.append(be.to_numpy(surf.aperture.r_max))
            else:
                semi_apertures.append(be.to_numpy(surf.semi_aperture))
        return semi_apertures

    def _get_materials(self):
        """Determines the material for each surface."""
        return [MaterialFormatter.format(surf) for surf in self.optic.surfaces]

    def _get_aspheric_coefficients(self):
        """Extracts the aspheric coefficients for each surface."""
        valid_geometry_types = (
            EvenAsphere,
            OddAsphere,
            PolynomialGeometry,
            ZernikePolynomialGeometry,
            ChebyshevPolynomialGeometry,
        )

        surface_coeffs = []
        for i, surface in enumerate(self.optic.surfaces):
            if isinstance(surface.geometry, valid_geometry_types):
                coefficients = list(surface.geometry.coefficients)
                if coefficients:
                    rows = [f"Surface {i}"] + coefficients
                    surface_coeffs.append(rows)

        if surface_coeffs:
            max_coef_num = len(max(surface_coeffs, key=len))
            headers = ["Surface"] + [f"c{i}" for i in range(max_coef_num - 1)]
            return surface_coeffs, headers
        else:
            return None, None
