"""Ray Bundle for Non-Sequential Raytracing.

Defines NSQRayBundle (the core in-memory ray state) and IntersectionResult.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NSQRayBundle:
    """Central in-memory object carrying all live ray state.

    All arrays are shape (N,) or (N, 3). Arrays may be NumPy or CuPy
    depending on the active TracerBackend. Dead rays (alive=False) remain
    in the array and are skipped by masked operations.

    Attributes:
        x: Position x-component [mm], shape (N,).
        y: Position y-component [mm], shape (N,).
        z: Position z-component [mm], shape (N,).
        dx: Direction x-component (unit vector), shape (N,).
        dy: Direction y-component (unit vector), shape (N,).
        dz: Direction z-component (unit vector), shape (N,).
        flux: Current flux weight [W or normalized], shape (N,).
        wavelength: Wavelength [nm], shape (N,).
        n_current: Refractive index of current medium, shape (N,).
        bounce: Number of surface hits, shape (N,).
        alive: Boolean mask -- False for dead/terminated rays, shape (N,).
        s0: Stokes S0 parameter, shape (N,). None if polarization disabled.
        s1: Stokes S1 parameter, shape (N,). None if polarization disabled.
        s2: Stokes S2 parameter, shape (N,). None if polarization disabled.
        s3: Stokes S3 parameter, shape (N,). None if polarization disabled.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    flux: np.ndarray
    wavelength: np.ndarray
    n_current: np.ndarray
    bounce: np.ndarray
    alive: np.ndarray
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
        xp = _get_xp(self.x)
        return int(xp.sum(self.alive))

    @property
    def positions(self) -> np.ndarray:
        """Ray positions as (N, 3) array [mm]."""
        xp = _get_xp(self.x)
        return xp.stack([self.x, self.y, self.z], axis=1)

    @property
    def directions(self) -> np.ndarray:
        """Ray directions as (N, 3) unit-vector array."""
        xp = _get_xp(self.dx)
        return xp.stack([self.dx, self.dy, self.dz], axis=1)

    def compact(self) -> NSQRayBundle:
        """Return a new bundle containing only alive rays.

        Removes dead rays to improve GPU occupancy. Call when the dead
        fraction exceeds 50%.

        Returns:
            A new NSQRayBundle with only the alive rays.
        """
        mask = self.alive
        kwargs: dict = dict(
            x=self.x[mask],
            y=self.y[mask],
            z=self.z[mask],
            dx=self.dx[mask],
            dy=self.dy[mask],
            dz=self.dz[mask],
            flux=self.flux[mask],
            wavelength=self.wavelength[mask],
            n_current=self.n_current[mask],
            bounce=self.bounce[mask],
            alive=self.alive[mask],
        )
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
        self.x = self.x + t * self.dx
        self.y = self.y + t * self.dy
        self.z = self.z + t * self.dz


@dataclass
class IntersectionResult:
    """Result of a ray-component intersection test.

    Attributes:
        t: Per-ray distance to nearest hit [mm], shape (N,). inf if no hit.
        normals: Hit surface normals in global frame, shape (N, 3).
            Normals point toward the incoming ray (outward from surface).
        hit_mask: True where the ray actually intersected, shape (N,).
        component_index: Index of the hit component/detector in the scene
            list. -1 if no hit. Shape (N,).
        is_detector: True if the hit object is a detector (not a component).
            Shape (N,).
    """

    t: np.ndarray
    normals: np.ndarray
    hit_mask: np.ndarray
    component_index: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    is_detector: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))


def _get_xp(arr: np.ndarray):
    """Return the array module (numpy or cupy) for the given array.

    Args:
        arr: A NumPy or CuPy array.

    Returns:
        The array module (numpy or cupy).
    """
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy
    except ImportError:
        pass
    return np
