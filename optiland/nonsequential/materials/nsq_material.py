"""NSQ Material adapter.

Wraps optiland.materials.BaseMaterial for use in the non-sequential tracer.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial
    from optiland.nonsequential.bsdf.base import BaseBSDF


@dataclass
class NSQMaterial:
    """Thin adapter over optiland.materials for non-sequential raytracing.

    Attributes:
        optiland_material: Refractive index provider. None means vacuum (n=1).
        bsdf: Optional surface scatter model for volumetric effects.
        absorption_coeff: Beer-Lambert absorption coefficient [1/mm].
            0.0 means lossless (MVP: always 0).
    """

    optiland_material: BaseMaterial | None = None
    bsdf: BaseBSDF | None = None
    absorption_coeff: float = 0.0

    def n(self, wavelength_nm: float | np.ndarray) -> float | np.ndarray:
        """Refractive index at the given wavelength(s).

        Args:
            wavelength_nm: Wavelength(s) in nanometres.

        Returns:
            Refractive index. Returns 1.0 for vacuum.
        """
        if self.optiland_material is None:
            if np.ndim(wavelength_nm) == 0:
                return 1.0
            return np.ones_like(np.asarray(wavelength_nm, dtype=float))
        # BaseMaterial.n() takes wavelength in microns
        wl_um = np.asarray(wavelength_nm, dtype=float) / 1000.0
        result = self.optiland_material.n(wl_um)
        # Convert backend array to numpy scalar/array
        try:
            return (
                float(result)
                if np.ndim(result) == 0
                else np.asarray(result, dtype=float)
            )
        except Exception:
            return np.asarray(result, dtype=float)


# Module-level vacuum constant — refractive index 1.0, no scatter, lossless.
VACUUM: NSQMaterial = NSQMaterial(optiland_material=None)
