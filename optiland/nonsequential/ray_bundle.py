"""Ray Bundle for Non-Sequential Raytracing.

Defines NSQRayBundle -- the core in-memory ray state.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # noqa: TC002

from optiland.nonsequential._utils import get_xp


@dataclass
class NSQRayBundle:
    """Central in-memory object carrying all live ray state.

    All arrays are shape (N,) or (N, 3). Arrays may be NumPy or CuPy
    depending on the active TracerBackend.

    Attributes:
        x: Position x-component [mm], shape (N,).
        y: Position y-component [mm], shape (N,).
        z: Position z-component [mm], shape (N,).
        L: Direction x-component (unit vector), shape (N,).
        M: Direction y-component (unit vector), shape (N,).
        N: Direction z-component (unit vector), shape (N,).
        flux: Current flux weight [W or normalized], shape (N,).
        wavelength: Wavelength [µm], shape (N,).
        n_current: Refractive index of current medium, shape (N,).
        bounce: Number of surface hits, shape (N,).
        alive: Boolean mask -- False for dead/terminated rays, shape (N,).
        ray_id: Unique ray identifier, shape (N,). None if not assigned.
        s0: Stokes S0 parameter, shape (N,). None if polarization disabled.
        s1: Stokes S1 parameter, shape (N,). None if polarization disabled.
        s2: Stokes S2 parameter, shape (N,). None if polarization disabled.
        s3: Stokes S3 parameter, shape (N,). None if polarization disabled.
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
    s0: np.ndarray | None = None
    s1: np.ndarray | None = None
    s2: np.ndarray | None = None
    s3: np.ndarray | None = None

    @property
    def num_rays(self) -> int:
        """Total number of rays (alive + dead)."""
        return int(self.x.shape[0])

    @property
    def num_rays_alive(self) -> int:
        """Number of alive rays."""
        xp = get_xp(self.x)
        return int(xp.sum(self.alive))

    @property
    def positions(self) -> np.ndarray:
        """Ray positions as (N, 3) array [mm]."""
        xp = get_xp(self.x)
        return xp.stack([self.x, self.y, self.z], axis=1)

    @property
    def directions(self) -> np.ndarray:
        """Ray directions as (N, 3) unit-vector array."""
        xp = get_xp(self.L)
        return xp.stack([self.L, self.M, self.N], axis=1)

    def compact(self) -> NSQRayBundle:
        """Return a new bundle containing only alive rays."""
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
        if self.s0 is not None:
            kwargs["s0"] = self.s0[mask]
            kwargs["s1"] = self.s1[mask]
            kwargs["s2"] = self.s2[mask]
            kwargs["s3"] = self.s3[mask]
        return NSQRayBundle(**kwargs)

    def advance(self, t: np.ndarray) -> None:
        """Advance ray positions along their directions by distance t.

        Args:
            t: Per-ray distances [mm], shape (N,).
        """
        self.x = self.x + t * self.L
        self.y = self.y + t * self.M
        self.z = self.z + t * self.N


def _get_xp(arr: np.ndarray):
    """Backward-compatible alias for get_xp."""
    return get_xp(arr)
