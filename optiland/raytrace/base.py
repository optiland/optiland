"""Base Ray Tracer

Abstract base class for all ray tracer implementations.

Kramer Harrison, 2025
"""

from __future__ import annotations

import abc


class BaseRayTracer(abc.ABC):
    """Abstract base class for ray tracers.

    Defines the common interface that all ray tracer implementations must
    satisfy.

    Args:
        optic (Optic): The optical system to be traced.
    """

    def __init__(self, optic):
        self.optic = optic

    @abc.abstractmethod
    def trace(self, *args, **kwargs):
        """Trace rays through the optical system.

        Args:
            *args: Positional arguments specific to the tracer implementation.
            **kwargs: Keyword arguments specific to the tracer implementation.

        Returns:
            The traced rays.
        """
