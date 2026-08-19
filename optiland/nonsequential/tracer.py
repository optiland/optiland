"""NSQTracer -- thin coordinator for Non-Sequential Raytracing.

NSQTracer holds backend configuration and delegates the full simulation loop
to the backend.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from optiland.nonsequential._utils import DEFAULT_BATCH_SIZE
from optiland.nonsequential.diagnostics import Diagnostics

if TYPE_CHECKING:
    from optiland.nonsequential.backends.base import TracerBackend
    from optiland.nonsequential.scene import NSQScene


@dataclass
class SimulationResult:
    """Top-level result returned by NSQTracer.trace() / NSQScene.trace().

    Attributes:
        detectors: Per-detector result objects, keyed by detector name.
        num_rays_total: Total number of rays launched.
        num_rays_absorbed: Rays terminated by absorbing components.
        num_rays_escaped: Rays that left the scene with no hit.
        num_rays_flux_killed: Rays killed for falling below flux threshold.
        num_rays_depth_killed: Rays killed for exceeding max_depth.
        total_flux_in: Total flux launched by all sources [W].
        total_flux_detected: Total flux recorded on all detectors [W].
        total_flux_absorbed: Flux absorbed by AbsorbingComponents [W].
        total_flux_bulk_absorbed: Flux lost to Beer-Lambert bulk absorption
            while travelling through an absorbing medium (k > 0),
            e.g. tinted glass -- distinct from ``total_flux_absorbed``,
            which is surface (AbsorbingComponent) absorption only [W].
        total_flux_escaped: Flux carried by escaped rays [W].
        total_flux_lost: Flux lost to flux/depth kill [W].
        flux_conservation_error:
            ``|flux_in - detected - absorbed - bulk_absorbed - escaped
            - lost| / flux_in``.
        trace_time_sec: Wall-clock time for the trace [s].
        ray_paths: Optional per-ray event log dict (``{"events":
            structured_array}``), populated when ``record_paths`` is
            truthy -- see :mod:`optiland.nonsequential.path_recording`.
        diagnostics: Self-diagnosing summary of this trace --
            depth truncation, roulette loss, unreached geometry, per
            -detector sampling quality, and a threshold-based warning list.
            See :meth:`report` and :mod:`optiland.nonsequential.diagnostics`.
    """

    detectors: dict[str, object] = field(default_factory=dict)
    num_rays_total: int = 0
    num_rays_absorbed: int = 0
    num_rays_escaped: int = 0
    num_rays_flux_killed: int = 0
    num_rays_depth_killed: int = 0
    total_flux_in: float = 0.0
    total_flux_detected: float = 0.0
    total_flux_absorbed: float = 0.0
    total_flux_bulk_absorbed: float = 0.0
    total_flux_escaped: float = 0.0
    total_flux_lost: float = 0.0
    flux_conservation_error: float = 0.0
    trace_time_sec: float = 0.0
    ray_paths: dict | None = None
    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    def report(self) -> str:
        """Full human-readable diagnostic report for this trace.

        Returns:
            A multi-line string -- see :meth:`Diagnostics.report`.
        """
        return self.diagnostics.report()

    def __repr__(self) -> str:
        """Concise summary instead of dumping every detector/array field.

        Includes a warning count so a warning-bearing result is visible
        even when only skimmed in a REPL or notebook cell -- call
        :meth:`report` for the full diagnostic text.
        """
        n_warnings = len(self.diagnostics.warnings())
        warn_note = f", {n_warnings} diagnostic warning(s)" if n_warnings else ""
        return (
            f"SimulationResult(num_rays_total={self.num_rays_total}, "
            f"detectors={list(self.detectors)}, "
            f"total_flux_in={self.total_flux_in:.6g}, "
            f"total_flux_detected={self.total_flux_detected:.6g}, "
            f"flux_conservation_error={self.flux_conservation_error:.3e}"
            f"{warn_note})"
        )


class NSQTracer:
    """Thin coordinator: holds backend configuration, delegates trace loop.

    The coordinator pattern decouples scene construction from backend
    selection. The full simulation loop lives in the backend.

    Usage::

        tracer = NSQTracer(scene)
        result = tracer.trace(num_rays=1_000_000, seed=42)

    Attributes:
        scene: The NSQScene to simulate.
        backend: TracerBackend instance (defaults to NumpyBackend or
            TorchBackend based on the active ``optiland.backend``).
    """

    def __init__(
        self,
        scene: NSQScene,
        backend: TracerBackend | None = None,
    ) -> None:
        """Initialize NSQTracer.

        Args:
            scene: The NSQScene to trace.
            backend: Backend to use. If None, selected automatically from
                the active ``optiland.backend`` (numpy → NumpyBackend,
                torch → TorchBackend).
        """
        self.scene = scene
        self.backend = backend

    def trace(
        self,
        num_rays: int,
        max_depth: int = 16,
        min_flux_fraction: float = 1e-6,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int | None = None,
        backend: TracerBackend | None = None,
        record_paths: bool | int = False,
    ) -> SimulationResult:
        """Run the simulation and return results.

        Args:
            num_rays: Total rays to launch.
            max_depth: Maximum surface interactions per ray before
                termination.
            min_flux_fraction: Kill threshold relative to per-ray initial
                flux.
            batch_size: Rays per processing batch. Does not change the result,
                only the speed; see ``DEFAULT_BATCH_SIZE``.
            seed: RNG seed for reproducibility.
            backend: Backend override. Uses constructor backend if not given.
                Auto-selects from active ``optiland.backend`` if neither
                is provided.
            record_paths: ``False`` records nothing, ``True`` records every
                ray's path, and a positive ``int`` records an approximately
                that-many-ray subset selected deterministically by
                ``ray_id`` hash -- see
                :mod:`optiland.nonsequential.path_recording`.

        Returns:
            SimulationResult.
        """
        resolved_backend = backend or self.backend or _default_backend(seed)

        return resolved_backend.trace(
            self.scene,
            num_rays=int(num_rays),
            max_depth=max_depth,
            min_flux_fraction=min_flux_fraction,
            batch_size=batch_size,
            seed=seed,
            record_paths=record_paths,
        )


def _default_backend(seed: int | None) -> TracerBackend:
    """Select a backend based on the active optiland.backend.

    Args:
        seed: RNG seed forwarded to the backend.

    Returns:
        NumpyBackend or TorchBackend depending on the active backend.
    """
    import optiland.backend as be  # noqa: PLC0415

    if be.get_backend() == "torch":
        from optiland.nonsequential.backends.torch_backend import (  # noqa: PLC0415
            TorchBackend,
        )

        return TorchBackend(seed=seed)

    from optiland.nonsequential.backends.numpy_backend import (  # noqa: PLC0415
        NumpyBackend,
    )

    return NumpyBackend(seed=seed)
