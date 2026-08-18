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
        L: Ray direction cosine x (unit vector).
        M: Ray direction cosine y.
        N: Ray direction cosine z.
        flux: Per-ray flux [W].
        wavelength: Per-ray wavelength [µm].
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        L: np.ndarray,
        M: np.ndarray,
        N: np.ndarray,
        flux: np.ndarray,
        wavelength: np.ndarray,
    ) -> None:
        """Initialize RayDatabase.

        Args:
            x: Positions x [mm].
            y: Positions y [mm].
            z: Positions z [mm].
            L: Direction cosine x.
            M: Direction cosine y.
            N: Direction cosine z.
            flux: Per-ray flux [W].
            wavelength: Per-ray wavelength [µm].
        """
        self.x = x
        self.y = y
        self.z = z
        self.L = L
        self.M = M
        self.N = N
        self.flux = flux
        self.wavelength = wavelength

    @property
    def num_rays(self) -> int:
        """Number of stored rays."""
        return len(self.x)

    @property
    def total_flux(self) -> float:
        """Total flux recorded at this detector [W].

        Exposed so a ray-database detector contributes to the tracer's flux
        ledger like every other detector; without it the detected flux is
        undercounted and ``flux_conservation_error`` misreports.
        """
        return float(np.sum(self.flux))

    def to_dataframe(self):
        """Return ray data as a pandas DataFrame.

        Returns:
            DataFrame with columns x, y, z, L, M, N, flux, wavelength.

        Raises:
            ImportError: If pandas is not installed.
        """
        import pandas as pd  # noqa: PLC0415

        return pd.DataFrame(
            {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "L": self.L,
                "M": self.M,
                "N": self.N,
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
            L=self.L,
            M=self.M,
            N=self.N,
            flux=self.flux,
            wavelength=self.wavelength,
        )
