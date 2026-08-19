"""
Ray Aimer Registry Module

This module provides a registry for managing available ray aiming algorithms.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.optic import Optic
    from optiland.rays.ray_aiming.base import BaseRayAimer

# Registry storage
_AIMER_REGISTRY: dict[str, type[BaseRayAimer]] = {}


def register_aimer(name: str) -> Callable[[type[BaseRayAimer]], type[BaseRayAimer]]:
    """
    Decorator to register a ray aiming class.

    Args:
        name (str): The name to register the aimer under.

    Returns:
        Callable[[Type[BaseRayAimer]], Type[BaseRayAimer]]: The decorated class.
    """

    def decorator(cls: type[BaseRayAimer]) -> type[BaseRayAimer]:
        if name in _AIMER_REGISTRY:
            raise ValueError(f"Ray aimer '{name}' is already registered.")
        _AIMER_REGISTRY[name] = cls
        return cls

    return decorator


def create_ray_aimer(name: str, optic: Optic, **kwargs: Any) -> BaseRayAimer:
    """
    Factory function to create a ray aimer instance.

    Args:
        name (str): The name of the ray aimer to create.
        optic (Optic): The optical system instance.
        **kwargs: Additional arguments for the aimer constructor.

    Returns:
        BaseRayAimer: An instance of the requested ray aimer.

    Raises:
        ValueError: If the requested ray aimer is not registered.
    """
    if name not in _AIMER_REGISTRY:
        available = ", ".join(sorted(_AIMER_REGISTRY.keys()))
        raise ValueError(f"Unknown ray aimer '{name}'. Available aimers: {available}")

    aimer_cls = _AIMER_REGISTRY[name]
    aimer = aimer_cls(optic, **kwargs)

    cache = kwargs.get("cache", False)
    if cache:
        # Avoid circular import
        from optiland.rays.ray_aiming.cached import CachedRayAimer

        # Max cache size can be passed in kwargs as "max_cache_size"
        max_cache_size = kwargs.get("max_cache_size", 128)
        aimer = CachedRayAimer(optic, aimer, max_cache_size=max_cache_size)

    return aimer
