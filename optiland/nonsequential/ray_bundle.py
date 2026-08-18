"""Ray Bundle for Non-Sequential Raytracing.

Defines NSQRayBundle -- the core in-memory ray state.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import optiland.backend as be

# Maximum simultaneous medium nesting depth a ray's medium_stack can record
# (e.g. a cemented triplet in an immersion fluid inside a sealed housing is
# 4). A push past this depth raises MediumStackOverflowError rather than
# silently wrapping or dropping the entry.
MEDIUM_STACK_MAX_DEPTH = 8

# Sentinel medium id for "no medium" / unused stack slots.
MEDIUM_STACK_EMPTY = -1


class MediumStackOverflowError(Exception):
    """A ray's medium nesting exceeded ``MEDIUM_STACK_MAX_DEPTH``.

    Raised rather than silently wrapping or truncating: this indicates
    either a pathologically deep (past any realistic optical assembly)
    volume nesting, or a geometry defect that pushes without ever popping.
    """


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
        k_current: Extinction coefficient of the current medium at each
            ray's wavelength, shape (N,). 0 for a non-absorbing medium
            (vacuum, or any material with no measured extinction data).
            Feeds Beer-Lambert bulk absorption over the distance a
            ray travels before its next hit; updated alongside ``n_current``
            wherever a ray crosses into a new medium.
        medium_stack: Nested medium ids the ray has entered but not yet
            exited, shape (N, ``MEDIUM_STACK_MAX_DEPTH``), int64. Slots at
            or beyond ``medium_depth`` for a given row are
            ``MEDIUM_STACK_EMPTY``. Plain NumPy always (bookkeeping only,
            never differentiated): pushed/popped by
            ``RefractiveComponent.interact`` alongside ``n_current``.
        medium_depth: Stack pointer -- number of valid entries in
            ``medium_stack`` for each ray, shape (N,), int32. Plain NumPy.
        medium_stack_underflows: Cumulative count of pop attempts on an
            empty ``medium_stack`` for each ray (a ray exiting a volume it
            never entered -- a geometry defect), shape (N,), int32. Plain
            NumPy; summed across all rays into
            ``Diagnostics.medium_stack_underflows`` at the end of a trace.
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
    k_current: np.ndarray | None = None
    medium_stack: np.ndarray | None = None
    medium_depth: np.ndarray | None = None
    medium_stack_underflows: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.k_current is None:
            # n_current is plain NumPy at construction time (sources always
            # build a NumPy bundle; TorchBackend promotes every field to a
            # tensor afterward via _ensure_torch_bundle), so this stays
            # NumPy too rather than dispatching through the active backend.
            self.k_current = np.zeros_like(self.n_current)
        if self.medium_stack is None:
            self.medium_stack = np.full(
                (self.num_rays, MEDIUM_STACK_MAX_DEPTH),
                MEDIUM_STACK_EMPTY,
                dtype=np.int64,
            )
        if self.medium_depth is None:
            self.medium_depth = np.zeros(self.num_rays, dtype=np.int32)
        if self.medium_stack_underflows is None:
            self.medium_stack_underflows = np.zeros(self.num_rays, dtype=np.int32)

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
            k_current=self.k_current[mask],
            medium_stack=self.medium_stack[mask],
            medium_depth=self.medium_depth[mask],
            medium_stack_underflows=self.medium_stack_underflows[mask],
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

    def select(self, idx: np.ndarray, ray_id: np.ndarray | None = None) -> NSQRayBundle:
        """Return a new, independent bundle containing rays at ``idx``.

        NumPy-only (fancy indexing on plain ndarrays): used by the bounded
        -splitting orchestration (D2, PR11;
        :mod:`optiland.nonsequential.ir.interpreter`) to snapshot the
        pre-interaction state of a set of rays before mutating the original
        bundle in place, so the transmit child of a split can be built from
        the same starting point as the reflect child.

        Args:
            idx: Integer index array selecting rows to copy.
            ray_id: If given, overrides ``self.ray_id[idx]`` in the returned
                bundle -- used to assign the spawned rays fresh identities
                so their RNG stream (keyed by ``ray_id``) is independent of
                the sibling ray that stayed at the original id.

        Returns:
            A new :class:`NSQRayBundle`, alive on every row (splitting only
            ever snapshots rays that are alive and mid-interaction).
        """
        kwargs: dict = dict(
            x=self.x[idx].copy(),
            y=self.y[idx].copy(),
            z=self.z[idx].copy(),
            L=self.L[idx].copy(),
            M=self.M[idx].copy(),
            N=self.N[idx].copy(),
            flux=self.flux[idx].copy(),
            wavelength=self.wavelength[idx].copy(),
            n_current=self.n_current[idx].copy(),
            bounce=self.bounce[idx].copy(),
            alive=np.ones(len(idx), dtype=bool),
            k_current=self.k_current[idx].copy(),
            medium_stack=self.medium_stack[idx].copy(),
            medium_depth=self.medium_depth[idx].copy(),
            medium_stack_underflows=self.medium_stack_underflows[idx].copy(),
        )
        if ray_id is not None:
            kwargs["ray_id"] = ray_id
        elif self.ray_id is not None:
            kwargs["ray_id"] = self.ray_id[idx].copy()
        return NSQRayBundle(**kwargs)

    @staticmethod
    def concat(bundles: list[NSQRayBundle]) -> NSQRayBundle:
        """Concatenate several bundles into one (NumPy-only).

        Used by the bounded-splitting orchestration to merge
        spawned transmit-branch children back into the live bundle at the
        end of a bounce, and to merge per-primitive spawn batches within a
        single bounce.

        Args:
            bundles: Non-empty list of bundles to concatenate, in order.

        Returns:
            A new :class:`NSQRayBundle` with every field concatenated along
            axis 0.
        """
        kwargs: dict = dict(
            x=np.concatenate([b.x for b in bundles]),
            y=np.concatenate([b.y for b in bundles]),
            z=np.concatenate([b.z for b in bundles]),
            L=np.concatenate([b.L for b in bundles]),
            M=np.concatenate([b.M for b in bundles]),
            N=np.concatenate([b.N for b in bundles]),
            flux=np.concatenate([b.flux for b in bundles]),
            wavelength=np.concatenate([b.wavelength for b in bundles]),
            n_current=np.concatenate([b.n_current for b in bundles]),
            bounce=np.concatenate([b.bounce for b in bundles]),
            alive=np.concatenate([b.alive for b in bundles]),
            k_current=np.concatenate([b.k_current for b in bundles]),
            medium_stack=np.concatenate([b.medium_stack for b in bundles]),
            medium_depth=np.concatenate([b.medium_depth for b in bundles]),
            medium_stack_underflows=np.concatenate(
                [b.medium_stack_underflows for b in bundles]
            ),
        )
        if all(b.ray_id is not None for b in bundles):
            kwargs["ray_id"] = np.concatenate([b.ray_id for b in bundles])
        return NSQRayBundle(**kwargs)
