"""Ray database result for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class RayDatabase:
    """Phase-space record of individual rays at a detector surface.

    Attributes:
        x: Ray positions x [mm].
        y: Ray positions y [mm].
        z: Ray positions z [mm].
        dx: Ray directions x (unit vector).
        dy: Ray directions y.
        dz: Ray directions z.
        flux: Per-ray flux [W].
        wavelength: Per-ray wavelength [nm].
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray,
        flux: np.ndarray,
        wavelength: np.ndarray,
    ) -> None:
        """Initialize RayDatabase.

        Args:
            x: Positions x [mm].
            y: Positions y [mm].
            z: Positions z [mm].
            dx: Directions x.
            dy: Directions y.
            dz: Directions z.
            flux: Per-ray flux [W].
            wavelength: Per-ray wavelength [nm].
        """
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.flux = flux
        self.wavelength = wavelength

    @property
    def num_rays(self) -> int:
        """Number of stored rays."""
        return len(self.x)

    def to_dataframe(self):
        """Return ray data as a pandas DataFrame.

        Returns:
            DataFrame with columns x, y, z, dx, dy, dz, flux, wavelength.

        Raises:
            ImportError: If pandas is not installed.
        """
        import pandas as pd  # noqa: PLC0415

        return pd.DataFrame(
            {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "dx": self.dx,
                "dy": self.dy,
                "dz": self.dz,
                "flux": self.flux,
                "wavelength": self.wavelength,
            }
        )

    def save(self, path: str | Path) -> None:
        """Save ray database to a .npz file.

        Args:
            path: Output file path.
        """
        np.savez(
            path,
            x=self.x,
            y=self.y,
            z=self.z,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            flux=self.flux,
            wavelength=self.wavelength,
        )
