"""Self-diagnosing simulation results.

``SimulationResult.diagnostics`` turns several silent failure modes the NSQ
engine could previously produce -- a scene that depth-truncates most of its
flux, a detector whose map is pure shot noise, a surface no ray ever
touches, roulette eating an unexpected fraction of the beam -- into an
explicit, inspectable object with a threshold-based warning list, rather
than numbers a user has to know to go looking for.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.scene import NSQScene

# Warn when flux killed by the hard max_depth cutoff exceeds this fraction
# of total launched flux: the result is truncated and max_depth should be
# raised, or the scene is leaking flux into paths this deep on purpose and
# the user should know that.
_DEPTH_TRUNCATION_WARN = 1e-3

# Warn when the flux ledger (in - detected - absorbed - bulk - escaped -
# lost) fails to close by more than this fraction of total_flux_in. Under
# reflect_prob="fresnel" (the default) this should be ~machine epsilon;
# under importance biasing or heavy roulette it is Monte Carlo noise that
# shrinks with ray count, so this threshold is deliberately loose enough
# to tolerate that while still catching a genuine energy leak.
_FLUX_CONSERVATION_WARN = 0.05

# Warn when Russian-roulette-killed flux exceeds this fraction of
# total launched flux. Roulette is unbiased in expectation, so a nonzero
# value is not itself wrong, but a large one means rr_start_flux is
# aggressive relative to this scene and the per-trace estimator is noisy.
_RR_KILLED_WARN = 1e-3

# Below this many mean hits per detector pixel, shot noise dominates the
# map (relative Poisson error 1/sqrt(hits) ~= 32% at 10 hits).
_UNDERSAMPLED_HITS_PER_PIXEL = 10.0

# Hits/pixel needed for ~5% relative Poisson error (1 / 0.05**2), used to
# estimate the ray count a target-quality map would need.
_TARGET_HITS_PER_PIXEL = 400.0


@dataclass(frozen=True)
class DetectorDiagnostic:
    """Per-detector sampling-quality diagnostic.

    Attributes:
        name: Detector's registry name.
        num_rays_hit: Rays recorded on this detector.
        num_pixels: Pixel/bin count of the detector's map, or ``None`` for
            detector kinds with no grid (e.g. ``RayDatabaseDetector``).
        mean_hits_per_pixel: ``num_rays_hit / num_pixels``, or ``None`` when
            ``num_pixels`` is ``None``.
        undersampled: True when ``mean_hits_per_pixel`` is below
            :data:`_UNDERSAMPLED_HITS_PER_PIXEL` -- shot noise dominates the
            map. Always False when ``num_pixels`` is ``None`` (nothing to
            under-sample against).
        rays_needed_for_5pct: Estimated total ray count (scaled from this
            trace's ``num_rays_total``) that would bring this detector to
            ~5% relative Poisson error, assuming ray count scales linearly
            with hits (true when flux/geometry are unchanged). ``None``
            when ``num_pixels`` is ``None`` or no rays hit at all.
    """

    name: str
    num_rays_hit: int
    num_pixels: int | None
    mean_hits_per_pixel: float | None
    undersampled: bool
    rays_needed_for_5pct: int | None


@dataclass(frozen=True)
class Diagnostics:
    """Self-diagnosing summary of one trace.

    Every field is computed during the trace at negligible extra cost (a
    few running counters and one pass over ``scene.detectors`` at the end)
    -- this is diagnosis, not a second simulation.

    Attributes:
        depth_truncated_flux_fraction: Fraction of ``total_flux_in`` killed
            by the hard ``max_depth`` cutoff. This is the one loss
            mechanism that is an *inherent, reported bias* -- not
            something roulette or importance sampling can fix -- so a
            nonzero value is a direct signal to raise ``max_depth`` if the
            deep paths matter.
        rr_killed_flux_fraction: Fraction of ``total_flux_in`` killed by
            Russian roulette, including bounded-splitting budget
            culling. Unbiased in expectation; large values mean the
            per-trace estimator is noisy, not necessarily wrong.
        flux_conservation_error: Copied from
            :attr:`~optiland.nonsequential.tracer.SimulationResult.flux_conservation_error`
            for convenience -- see that field's docstring.
        unreached_geometry: Names of scene components no ray ever hit
            (nearest-hit or otherwise) over the whole trace. Usually a
            misplaced or mis-oriented surface, but can be a deliberately
            unused spare aperture -- reported, not assumed to be a bug.
        detectors: Per-detector sampling-quality diagnostics, in
            ``scene.detectors`` order.
        medium_stack_underflows: Total pop-on-empty-stack events across all
            rays this trace (see ``NSQRayBundle.medium_stack``/
            ``RefractiveComponent.interact``). This is a diagnostic
            cross-check layered on top of the geometric ``n_geom``
            sidedness resolution -- it never affects the physics (n1/n2 are
            always resolved geometrically, never from the stack) -- so a
            nonzero value flags a likely geometry defect (a volume boundary
            surface reused inconsistently, or two separately constructed
            ``NSQMaterial`` instances standing in for what should be one
            physical medium) without itself changing any traced result.
        split_budget_saturated: True if bounded splitting ever
            hit its ``split_budget`` cap during this trace, meaning some
            spawned ghost-path rays were roulette-terminated rather than
            all being kept. NumPy backend only; always False otherwise.
    """

    depth_truncated_flux_fraction: float = 0.0
    rr_killed_flux_fraction: float = 0.0
    flux_conservation_error: float = 0.0
    unreached_geometry: tuple[str, ...] = ()
    detectors: tuple[DetectorDiagnostic, ...] = field(default_factory=tuple)
    medium_stack_underflows: int = 0
    split_budget_saturated: bool = False

    def warnings(self) -> list[str]:
        """Human-readable warnings for every diagnostic past its threshold.

        Returns:
            A list of one-line warning strings; empty if nothing is flagged.
        """
        msgs: list[str] = []
        if self.depth_truncated_flux_fraction > _DEPTH_TRUNCATION_WARN:
            msgs.append(
                f"{self.depth_truncated_flux_fraction:.2%} of launched flux "
                "was truncated by max_depth -- results are incomplete for "
                "paths deeper than max_depth; raise it if those paths matter."
            )
        if self.flux_conservation_error > _FLUX_CONSERVATION_WARN:
            msgs.append(
                f"flux_conservation_error is {self.flux_conservation_error:.2%} "
                "-- higher than expected Monte Carlo noise; check for an "
                "energy-accounting bug before trusting this trace."
            )
        if self.rr_killed_flux_fraction > _RR_KILLED_WARN:
            msgs.append(
                f"Russian roulette killed {self.rr_killed_flux_fraction:.2%} "
                "of launched flux -- consider a lower rr_start_flux if this "
                "trace's variance looks high."
            )
        if self.unreached_geometry:
            names = ", ".join(self.unreached_geometry)
            msgs.append(
                f"{len(self.unreached_geometry)} component(s) were never hit "
                f"by any ray: {names}. Check placement/orientation, or "
                "ignore if intentionally unused."
            )
        for det in self.detectors:
            if det.undersampled:
                needed = (
                    f"; ~{det.rays_needed_for_5pct:,} rays would bring it to "
                    "~5% relative error"
                    if det.rays_needed_for_5pct is not None
                    else ""
                )
                msgs.append(
                    f"Detector '{det.name}' is undersampled: "
                    f"{det.mean_hits_per_pixel:.1f} mean hits/pixel "
                    f"(< {_UNDERSAMPLED_HITS_PER_PIXEL:g}), shot noise "
                    f"dominates the map{needed}."
                )
        if self.medium_stack_underflows:
            msgs.append(
                f"{self.medium_stack_underflows} medium-stack underflow(s) "
                "detected -- a ray exited a volume it never entered; this "
                "indicates a geometry defect."
            )
        if self.split_budget_saturated:
            msgs.append(
                "Bounded splitting hit its split_budget cap at least once "
                "-- some spawned ghost-path rays were roulette-terminated "
                "rather than kept; raise split_budget for lower variance."
            )
        return msgs

    def report(self) -> str:
        """Full human-readable diagnostic report.

        Returns:
            A multi-line string: every diagnostic value, followed by a
            "Warnings" section (or "No warnings." if none fired).
        """
        lines = [
            "NSQ trace diagnostics:",
            f"  depth_truncated_flux_fraction: "
            f"{self.depth_truncated_flux_fraction:.4%}",
            f"  rr_killed_flux_fraction:       {self.rr_killed_flux_fraction:.4%}",
            f"  flux_conservation_error:       {self.flux_conservation_error:.4%}",
            f"  unreached_geometry:            "
            f"{list(self.unreached_geometry) if self.unreached_geometry else 'none'}",
            f"  medium_stack_underflows:       {self.medium_stack_underflows}",
            f"  split_budget_saturated:        {self.split_budget_saturated}",
        ]
        if self.detectors:
            lines.append("  detectors:")
            for det in self.detectors:
                mh = (
                    f"{det.mean_hits_per_pixel:.2f}"
                    if det.mean_hits_per_pixel is not None
                    else "n/a"
                )
                lines.append(
                    f"    {det.name}: {det.num_rays_hit} hits, "
                    f"{mh} mean hits/pixel"
                    + (" [undersampled]" if det.undersampled else "")
                )
        warns = self.warnings()
        if warns:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in warns)
        else:
            lines.append("No warnings.")
        return "\n".join(lines)


def _detector_pixel_count(result: object) -> int | None:
    """Pixel/bin count of a detector result, or ``None`` if it has no grid.

    Args:
        result: A detector's ``get_result()`` output (``IrradianceMap``,
            ``SpectralResult``, ``FarFieldPattern``, or ``RayDatabase``).

    Returns:
        ``len(x) * len(y)`` for a spatial grid (``x_coords``/``y_coords``)
        or an angular grid (``theta``/``phi``); ``None`` for a result with
        neither (``RayDatabase``).
    """
    x = getattr(result, "x_coords", None)
    y = getattr(result, "y_coords", None)
    if x is not None and y is not None:
        return len(x) * len(y)
    theta = getattr(result, "theta", None)
    phi = getattr(result, "phi", None)
    if theta is not None and phi is not None:
        return len(theta) * len(phi)
    return None


def _detector_diagnostic(
    name: str, result: object, num_rays_total: int
) -> DetectorDiagnostic:
    """Build one :class:`DetectorDiagnostic` from a detector's result.

    Args:
        name: Detector's registry name.
        result: The detector's ``get_result()`` output.
        num_rays_total: Total rays launched this trace (for the
            ``rays_needed_for_5pct`` estimate).

    Returns:
        The diagnostic.
    """
    num_rays_hit = int(getattr(result, "num_rays_hit", 0))
    num_pixels = _detector_pixel_count(result)
    if not num_pixels:
        return DetectorDiagnostic(
            name=name,
            num_rays_hit=num_rays_hit,
            num_pixels=num_pixels,
            mean_hits_per_pixel=None,
            undersampled=False,
            rays_needed_for_5pct=None,
        )
    mean_hits = num_rays_hit / num_pixels
    undersampled = mean_hits < _UNDERSAMPLED_HITS_PER_PIXEL
    rays_needed = (
        int(num_rays_total * (_TARGET_HITS_PER_PIXEL / mean_hits))
        if mean_hits > 0
        else None
    )
    return DetectorDiagnostic(
        name=name,
        num_rays_hit=num_rays_hit,
        num_pixels=num_pixels,
        mean_hits_per_pixel=mean_hits,
        undersampled=undersampled,
        rays_needed_for_5pct=rays_needed,
    )


def build_diagnostics(
    scene: NSQScene,
    hit_component_ids: set[int],
    num_rays_total: int,
    total_flux_in: float,
    total_flux_depth_killed: float,
    total_flux_rr_killed: float,
    flux_conservation_error: float,
    split_budget_saturated: bool,
    detector_results: dict[str, object],
    medium_stack_underflows: int = 0,
) -> Diagnostics:
    """Assemble a :class:`Diagnostics` from one trace's bookkeeping.

    Shared by both reference backends so the diagnostic definitions cannot
    drift between them.

    Args:
        scene: The traced scene.
        hit_component_ids: Indices into ``scene.surfaces`` that were the
            nearest hit for at least one ray at least once.
        num_rays_total: Total rays launched this trace.
        total_flux_in: Total launched flux [W].
        total_flux_depth_killed: Flux killed by the hard ``max_depth`` cutoff.
        total_flux_rr_killed: Flux killed by Russian roulette (including
            split-budget culling).
        flux_conservation_error: The trace's flux-ledger closure error.
        split_budget_saturated: Whether bounded splitting ever hit its cap.
        detector_results: ``{name: get_result()}`` for every detector, in
            ``scene.detectors`` order.
        medium_stack_underflows: Total medium-stack pop-on-empty events
            across the trace (see :class:`Diagnostics`).

    Returns:
        The assembled diagnostics.
    """
    unreached = tuple(
        (c.name or f"component_{i}")
        for i, c in enumerate(scene.surfaces)
        if i not in hit_component_ids
    )
    detector_diags = tuple(
        _detector_diagnostic(name, result, num_rays_total)
        for name, result in detector_results.items()
    )
    depth_frac = total_flux_depth_killed / total_flux_in if total_flux_in > 0 else 0.0
    rr_frac = total_flux_rr_killed / total_flux_in if total_flux_in > 0 else 0.0

    return Diagnostics(
        depth_truncated_flux_fraction=depth_frac,
        rr_killed_flux_fraction=rr_frac,
        flux_conservation_error=flux_conservation_error,
        unreached_geometry=unreached,
        detectors=detector_diags,
        medium_stack_underflows=medium_stack_underflows,
        split_budget_saturated=split_budget_saturated,
    )
