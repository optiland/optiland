"""NSQTracer -- thin coordinator for Non-Sequential Raytracing.

NSQTracer holds backend configuration and delegates the full simulation loop
to the backend. The Monte Carlo loop has moved to
:class:`~optiland.nonsequential.backends.numpy_backend.NumpyBackend`.

Backward compatibility: the old constructor signature
``NSQTracer(scene, num_rays, max_bounces, ..., seed)`` with
``tracer.trace(backend=None)`` is preserved.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from optiland.nonsequential.backends.numpy_backend import NumpyBackend

if TYPE_CHECKING:
    from optiland.nonsequential.backends.base import TracerBackend
    from optiland.nonsequential.scene import NSQScene


@dataclass
class SimulationResult:
    """Top-level result returned by NSQTracer.trace() / NSQScene.trace().

    Attributes:
        detectors: Per-detector result objects, keyed by detector name
            (or ``'detector_N'`` if unnamed).
        num_rays_total: Total number of rays launched.
        num_rays_absorbed: Rays terminated by absorbing components.
        num_rays_escaped: Rays that ran out of bounces or fell below flux
            threshold.
        total_flux_in: Total flux launched by all sources [W].
        total_flux_detected: Total flux recorded on all detectors [W].
        flux_conservation_error:
            ``|flux_in - (detected + absorbed + escaped)| / flux_in``.
        trace_time_sec: Wall-clock time for the trace [s].
        ray_paths: Optional dictionary containing ray history matrices.
    """

    detectors: dict[str, object] = field(default_factory=dict)
    num_rays_total: int = 0
    num_rays_absorbed: int = 0
    num_rays_escaped: int = 0
    total_flux_in: float = 0.0
    total_flux_detected: float = 0.0
    flux_conservation_error: float = 0.0
    trace_time_sec: float = 0.0
    ray_paths: dict[str, list] | None = None


class NSQTracer:
    """Thin coordinator: holds backend configuration, delegates trace loop.

    The coordinator pattern decouples scene construction from backend
    selection.  The full Monte Carlo loop lives in the backend.

    Supports two call patterns:

    **New (preferred)**::

        tracer = NSQTracer(scene)
        result = tracer.trace(num_rays=1_000_000, seed=42)

    **Legacy (backward-compatible)**::

        tracer = NSQTracer(scene, num_rays=1_000_000, seed=42)
        result = tracer.trace()

    Attributes:
        scene: The NSQScene to simulate.
        backend: TracerBackend instance (defaults to NumpyBackend).
        num_rays: Total rays (legacy constructor only).
        max_bounces: Max surface hits per ray (legacy constructor only).
        min_flux_fraction: Kill threshold (legacy constructor only).
        batch_size: Rays per batch (legacy constructor only).
        seed: RNG seed (legacy constructor only).
    """

    def __init__(
        self,
        scene: NSQScene,
        num_rays: int | None = None,
        max_bounces: int = 200,
        min_flux_fraction: float = 1e-6,
        batch_size: int = 1_000_000,
        polarization: bool = False,
        seed: int | None = None,
        backend: TracerBackend | None = None,
    ) -> None:
        """Initialize NSQTracer.

        Args:
            scene: The NSQScene to trace.
            num_rays: Total rays to launch. Stored for legacy
                ``tracer.trace()`` call. Not needed when calling
                ``tracer.trace(num_rays=...)``.
            max_bounces: Maximum surface hits per ray (legacy path).
            min_flux_fraction: Kill threshold (legacy path).
            batch_size: Rays per processing batch (legacy path).
            polarization: Reserved for future polarization tracking.
            seed: RNG seed (legacy path).
            backend: Backend to use (both old and new paths).
        """
        self.scene = scene
        # Legacy constructor params (kept for backward compat)
        self.num_rays = int(num_rays) if num_rays is not None else None
        self.max_bounces = int(max_bounces)
        self.min_flux_fraction = float(min_flux_fraction)
        self.batch_size = int(batch_size)
        self.polarization = polarization
        self.seed = seed
        self.backend = backend

    def trace(
        self,
        num_rays: int | None = None,
        max_bounces: int | None = None,
        min_flux_fraction: float | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        backend: TracerBackend | None = None,
        record_paths: bool = False,
    ) -> SimulationResult:
        """Run the simulation and return results.

        Args:
            num_rays: Total rays. Uses constructor value if not given.
            max_bounces: Max bounces. Uses constructor value if not given.
            min_flux_fraction: Kill threshold. Uses constructor value if
                not given.
            batch_size: Rays per batch. Uses constructor value if not given.
            seed: RNG seed. Uses constructor value if not given.
            backend: Backend override. Uses constructor backend if not given.
            record_paths: If True, tracks node points of bouncing rays.

        Returns:
            SimulationResult.

        Raises:
            ValueError: If num_rays is not available from any source.
        """
        resolved_n = num_rays if num_rays is not None else self.num_rays
        if resolved_n is None:
            raise ValueError(
                "num_rays must be supplied either to NSQTracer() or to trace()."
            )

        resolved_seed = seed if seed is not None else self.seed
        resolved_backend = backend or self.backend or NumpyBackend(seed=resolved_seed)

        return resolved_backend.trace(
            self.scene,
            num_rays=int(resolved_n),
            max_bounces=max_bounces if max_bounces is not None else self.max_bounces,
            min_flux_fraction=(
                min_flux_fraction
                if min_flux_fraction is not None
                else self.min_flux_fraction
            ),
            batch_size=batch_size if batch_size is not None else self.batch_size,
            seed=seed if seed is not None else self.seed,
            record_paths=record_paths,
        )
