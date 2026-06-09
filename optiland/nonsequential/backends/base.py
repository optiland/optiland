"""TracerBackend ABC for Non-Sequential Raytracing.

The backend owns the entire Monte Carlo trace loop, enabling future
alternative backends (e.g. an OptiX-based backend from ``optiland-rt``)
to plug in without modifying NSQTracer or NSQScene.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.tracer import SimulationResult


class TracerBackend(ABC):
    """Abstract backend for the NSQ Monte Carlo trace loop.

    The full simulation loop -- ray generation, intersection, interaction,
    detection -- is delegated to the backend implementation.
    ``NumpyBackend`` provides the default CPU implementation.  A future
    ``OptiXBackend`` from ``optiland-rt`` would replace the entire loop
    with NVIDIA OptiX kernel dispatch.

    New backends implement :meth:`trace` and are passed to
    ``NSQScene.trace(backend=...)`` or ``NSQTracer.trace(backend=...)``.
    """

    @abstractmethod
    def trace(
        self,
        scene: NSQScene,
        num_rays: int,
        max_depth: int = 16,
        min_flux_fraction: float = 1e-6,
        batch_size: int = 1_000_000,
        seed: int | None = None,
        record_paths: bool = False,
    ) -> SimulationResult:
        """Run the full simulation and return results.

        Args:
            scene: The NSQScene to trace (provides flat surface/source/detector
                lists via :attr:`~NSQScene.surfaces`,
                :attr:`~NSQScene.sources`, :attr:`~NSQScene.detectors`).
            num_rays: Total number of rays to launch.
            max_depth: Maximum surface interactions per ray before
                termination.
            min_flux_fraction: Rays whose flux drops below
                ``min_flux_fraction * (total_flux / num_rays)`` are killed.
            batch_size: Number of rays per processing batch.
            seed: RNG seed for reproducibility.
            record_paths: If True, record the full phase-space path of
                every ray, bounce-by-bounce, for visualization.

        Returns:
            :class:`~optiland.nonsequential.tracer.SimulationResult` with
            per-detector results and global statistics.
        """
