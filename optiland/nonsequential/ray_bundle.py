"""Ray Bundle for Non-Sequential Raytracing.

Defines NSQRayBundle -- the core in-memory ray state.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    import numpy as np


@dataclass
class NSQRayBundle:
    """Central in-memory object carrying all live ray state.

    All arrays are shape (N,) or (N, 3). Arrays may be NumPy ndarray or
    torch Tensor depending on the active TracerBackend.

    Attributes:
        x: Position x-component [mm], shape (N,).
        y: Position y-component [mm], shape (N,).
        z: Position z-component [mm], shape (N,).
        L: Direction x-component (unit vector), shape (N,).
        M: Direction y-component (unit vector), shape (N,).
        N: Direction z-component (unit vector), shape (N,).
        flux: Current flux / throughput weight, shape (N,).
        wavelength: Wavelength [µm], shape (N,).
        n_current: Refractive index of current medium, shape (N,).
        bounce: Number of surface hits, shape (N,).
        alive: Boolean mask -- False for dead/terminated rays, shape (N,).
        ray_id: Unique ray identifier, shape (N,). None if not assigned.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    L: np.ndarray
    M: np.ndarray
    N: np.ndarray
    flux: np.ndarray
    wavelength: np.ndarray
    n_current: np.ndarray
    bounce: np.ndarray
    alive: np.ndarray
    ray_id: np.ndarray | None = None

    @property
    def num_rays(self) -> int:
        """Total number of rays (alive + dead)."""
        return int(self.x.shape[0])

    @property
    def num_rays_alive(self) -> int:
        """Number of alive rays."""
        return int(be.sum(self.alive))

    @property
    def positions(self) -> np.ndarray:
        """Ray positions as (N, 3) array [mm]."""
        return be.stack([self.x, self.y, self.z], axis=1)

    @property
    def directions(self) -> np.ndarray:
        """Ray directions as (N, 3) unit-vector array."""
        return be.stack([self.L, self.M, self.N], axis=1)

    def compact(self) -> NSQRayBundle:
        """Return a new bundle containing only alive rays.

        Note: compaction is disabled in TorchBackend (alive rays carry
        zero-weight rather than being removed, to keep the graph fixed-shape).
        This method is used only in the NumPy forward fast path.
        """
        mask = self.alive
        kwargs: dict = dict(
            x=self.x[mask],
            y=self.y[mask],
            z=self.z[mask],
            L=self.L[mask],
            M=self.M[mask],
            N=self.N[mask],
            flux=self.flux[mask],
            wavelength=self.wavelength[mask],
            n_current=self.n_current[mask],
            bounce=self.bounce[mask],
            alive=self.alive[mask],
        )
        if self.ray_id is not None:
            kwargs["ray_id"] = self.ray_id[mask]
        return NSQRayBundle(**kwargs)

    def advance(self, t: np.ndarray) -> None:
        """Advance ray positions along their directions by distance t.

        Args:
            t: Per-ray distances [mm], shape (N,).
        """
        self.x = self.x + t * self.L
        self.y = self.y + t * self.M
        self.z = self.z + t * self.N
