"""Cached Ray Aiming Module

This module implements a caching wrapper for ray aiming algorithms.
It stores previous results to speed up repetitive calculations, especially
during optimization or tolerance analysis where system changes might be small.

Retained for backward compatibility (explicit ``cache=True``) but no longer
the default warm-start mechanism for ``"robust"``, which has its own
intrinsic, always-on ``PupilMapCache`` (see ``pupil_map.py`` and
``robust.py``).

Kramer Harrison, 2025
"""

from __future__ import annotations

import hashlib
import pickle
from typing import TYPE_CHECKING, Any

from optiland.rays.ray_aiming.base import BaseRayAimer

if TYPE_CHECKING:
    from optiland.optic import Optic


class CachedRayAimer(BaseRayAimer):
    """Cached ray aiming strategy.

    This class wraps another ray aimer and caches its results. It checks
    if the inputs and the optical system state have changed. If they match
    a cached entry, the result is returned immediately. If the system has
    changed but inputs match, the previous result is used as a starting guess.

    Attributes:
        optic (Optic): The optical system being traced.
        wrapped_aimer (BaseRayAimer): The actual aiming strategy being cached.
        max_cache_size (int): Maximum number of entries in the cache.
    """

    def __init__(
        self,
        optic: Optic,
        wrapped_aimer: BaseRayAimer,
        max_cache_size: int = 128,
        **kwargs: Any,
    ) -> None:
        """Initialize the CachedRayAimer.

        Args:
            optic (Optic): The optical system instance.
            wrapped_aimer (BaseRayAimer): The aimer instance to wrap.
            max_cache_size (int, optional): Max cache entries. Defaults to 128.
            **kwargs: Additional arguments passed to BaseRayAimer.
        """
        super().__init__(optic, **kwargs)
        self.wrapped_aimer = wrapped_aimer
        self.max_cache_size = max_cache_size
        self._cache: dict[str, tuple[str, tuple]] = {}

    def aim_rays(
        self,
        fields: tuple,
        wavelengths: Any,
        pupil_coords: tuple,
        initial_guess: tuple | None = None,
    ) -> tuple:
        """Calculate ray starting coordinates, using cache if available.

        Args:
            fields (tuple): Field coordinates.
            wavelengths (Any): Wavelengths.
            pupil_coords (tuple): Pupil coordinates.
            initial_guess (tuple | None, optional): Explicit starting guess.

        Returns:
            tuple: Ray parameters (x, y, z, L, M, N).
        """
        # If an explicit initial guess is provided, we skip the cache check
        if initial_guess is not None:
            return self.wrapped_aimer.aim_rays(
                fields, wavelengths, pupil_coords, initial_guess
            )

        # 1. Generate Input Hash
        input_key = self._get_input_hash(fields, wavelengths, pupil_coords)

        # 2. Generate System Hash
        current_sys_hash = self._get_system_hash()

        # 3. Check Cache
        cached_entry = self._cache.get(input_key)

        guess = None
        if cached_entry:
            cached_sys_hash, cached_result = cached_entry
            if cached_sys_hash == current_sys_hash:
                # Exact match: inputs same, system same
                return cached_result
            else:
                # System changed, but inputs same. Use cached result as guess.
                guess = cached_result

        # 4. Delegate to wrapped aimer
        result = self.wrapped_aimer.aim_rays(
            fields, wavelengths, pupil_coords, initial_guess=guess
        )

        # 5. Update Cache
        self._cache[input_key] = (current_sys_hash, result)

        # 6. Manage Cache Size
        if len(self._cache) > self.max_cache_size:
            # Simple FIFO removal (Python dicts preserve insertion order)
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        return result

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()

    def _get_input_hash(
        self, fields: tuple, wavelengths: Any, pupil_coords: tuple
    ) -> str:
        """Generate a hash for the input parameters."""

        def _to_hashable(obj: Any) -> Any:
            if hasattr(obj, "tobytes"):
                return obj.tobytes()
            elif isinstance(obj, list | tuple):
                return tuple(_to_hashable(x) for x in obj)
            return obj

        data = (
            _to_hashable(fields),
            _to_hashable(wavelengths),
            _to_hashable(pupil_coords),
        )
        return hashlib.md5(pickle.dumps(data)).hexdigest()

    def _get_system_hash(self) -> str:
        """Generate a hash for the current state of the optical system."""
        data = (
            self.optic.surfaces.to_dict(),
            self.optic.fields.to_dict(),
            self.optic.wavelengths.to_dict(),
            self.optic.aperture.to_dict() if self.optic.aperture else None,
            self.optic.ray_tracer.ray_aiming_config,
        )
        return hashlib.md5(str(data).encode("utf-8")).hexdigest()
