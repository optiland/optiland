"""Pupil Map Module

This module implements the per-field affine launch model used by the
chief-ray calibrated robust ray aimer (see ``robust.py``), together with a
warm-start cache keyed by ``(Hx, Hy, wavelength)``.

The pupil map is a cheap seed generator only: it is exact at the chief ray
and four cardinal edge probes, and a good linear approximation elsewhere.
The final Newton/Broyden polish (in ``iterative.py``) makes every ray exact,
so the map never needs to carry gradient information -- it is stored as
plain Python floats, which are inherently detached from any autograd graph.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.optic import Optic


def to_float(x: Any) -> float:
    """Extract a plain Python float from a scalar or length-1 backend array."""
    return float(be.to_numpy(x).reshape(-1)[0])


class PupilMap:
    """Per-field affine launch model: ``launch(Px, Py) = c + A @ [Px, Py]``.

    For infinite-conjugate systems ``launch`` is the ray origin ``(x, y)``;
    for finite conjugates it is the launch direction ``(L, M)``. The
    remaining launch components (the ones not solved by the 2-DOF Newton
    loop) are fixed at their chief-ray values and reused for every pupil
    point of this field.

    Attributes:
        c: Chief launch ``(p1, p2)`` -- ``(x, y)`` or ``(L, M)``.
        A: Affine matrix ``((a11, a12), (a21, a22))``.
        is_infinite: Whether the object is at infinity.
        fixed: The non-solved launch components, evaluated at the chief ray:
            ``(z, L, M, N)`` for infinite conjugates, or
            ``(z, x, y, N)`` for finite conjugates.
    """

    __slots__ = ("c", "A", "is_infinite", "fixed")

    def __init__(
        self,
        c: tuple[float, float],
        A: tuple[tuple[float, float], tuple[float, float]],
        is_infinite: bool,
        fixed: tuple[float, float, float, float],
    ) -> None:
        self.c = c
        self.A = A
        self.is_infinite = is_infinite
        self.fixed = fixed

    def seed(self, Px: Any, Py: Any) -> tuple:
        """Evaluate the affine model for pupil coordinates (Px, Py).

        Args:
            Px: Normalized pupil x-coordinates.
            Py: Normalized pupil y-coordinates.

        Returns:
            tuple: Full launch guess ``(x, y, z, L, M, N)``.
        """
        Px = be.as_array_1d(Px)
        Py = be.as_array_1d(Py)

        c1, c2 = self.c
        (a11, a12), (a21, a22) = self.A
        p1 = c1 + a11 * Px + a12 * Py
        p2 = c2 + a21 * Px + a22 * Py

        z_fixed, o1, o2, o3 = self.fixed
        z = be.ones_like(Px) * z_fixed

        if self.is_infinite:
            x, y = p1, p2
            L = be.ones_like(Px) * o1
            M = be.ones_like(Px) * o2
            N = be.ones_like(Px) * o3
        else:
            L, M = p1, p2
            x = be.ones_like(Px) * o1
            y = be.ones_like(Px) * o2
            N = be.ones_like(Px) * o3

        return x, y, z, L, M, N


class PupilMapCache:
    """Warm-start cache of :class:`PupilMap`, keyed by ``(Hx, Hy, wavelength)``.

    The cache is never cleared on a system change. Instead, a lightweight
    fingerprint of the aiming-relevant system state is tracked per entry:
    a cache hit is only "fresh" (reusable without recomputation) if the
    fingerprint at store time still matches the current one. A stale or
    missing entry is always still usable as a warm-start *seed* for a fresh
    calibration -- correctness never depends on the cache, only speed.
    """

    def __init__(self, precision: int = 6) -> None:
        self.precision = precision
        self._store: dict[tuple[float, float, float], PupilMap] = {}
        self._fingerprint_at_store: dict[tuple[float, float, float], Any] = {}
        self._current_fingerprint: Any = None

    def _key(self, Hx: float, Hy: float, wl: float) -> tuple[float, float, float]:
        p = self.precision
        return (round(float(Hx), p), round(float(Hy), p), round(float(wl), p))

    def sync(self, optic: Optic) -> None:
        """Recompute the current system fingerprint (once per aiming call)."""
        self._current_fingerprint = self.fingerprint(optic)

    def get_fresh(self, Hx: float, Hy: float, wl: float) -> PupilMap | None:
        """Return the cached map only if the system hasn't changed since it
        was stored -- an exact-reuse hit that skips recalibration entirely.
        """
        key = self._key(Hx, Hy, wl)
        pmap = self._store.get(key)
        if pmap is None:
            return None
        if self._fingerprint_at_store.get(key) != self._current_fingerprint:
            return None
        return pmap

    def get_stale(self, Hx: float, Hy: float, wl: float) -> PupilMap | None:
        """Return the map for this exact key regardless of freshness."""
        return self._store.get(self._key(Hx, Hy, wl))

    def nearest(self, Hx: float, Hy: float) -> PupilMap | None:
        """Return the cached map whose field is nearest in (Hx, Hy).

        Used for field-marching warm starts (D8): a newly requested field
        with no cached entry seeds its chief solve from the closest field
        already solved, rather than a cold paraxial guess.
        """
        if not self._store:
            return None
        target = (float(Hx), float(Hy))
        best_key = min(
            self._store,
            key=lambda k: (k[0] - target[0]) ** 2 + (k[1] - target[1]) ** 2,
        )
        return self._store[best_key]

    def put(self, Hx: float, Hy: float, wl: float, pmap: PupilMap) -> None:
        key = self._key(Hx, Hy, wl)
        self._store[key] = pmap
        self._fingerprint_at_store[key] = self._current_fingerprint

    def fingerprint(self, optic: Optic) -> Any:
        """Compute a lightweight hash of aiming-relevant system state.

        Only surfaces up to and including the stop matter for aiming, so
        post-stop surfaces (and unrelated optic metadata) are excluded to
        keep this cheap enough to call on every aiming request.
        """
        stop_index = optic.surfaces.stop_index
        surf_data = tuple(
            str(optic.surfaces[i].to_dict()) for i in range(stop_index + 1)
        )
        aperture_data = str(optic.aperture.to_dict()) if optic.aperture else None
        fields_data = str(optic.fields.to_dict())
        wl_data = str(optic.wavelengths.to_dict())
        return hash((surf_data, aperture_data, fields_data, wl_data))
