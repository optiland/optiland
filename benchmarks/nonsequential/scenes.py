"""Parametrized NSQ scenes for the performance benchmark harness.

Each builder isolates one performance axis (surface count, ray count, trace
depth) so the harness can vary that axis alone while holding the others
fixed.

Kramer Harrison, 2026
"""

from __future__ import annotations

from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    MirrorConfig,
    NSQMaterial,
    NSQScene,
    Spectrum,
)

_SPECTRUM = Spectrum.monochromatic(0.55)
# A constant-index material (rather than a catalog glass) so the ray count
# sweep exercises the same code path on both backends -- catalog dispersion
# formulas are not yet guaranteed torch-tensor-safe, which is an unrelated,
# pre-existing gap outside this harness's scope.
_LENS_MATERIAL = NSQMaterial(optiland_material=IdealMaterial(n=1.5))


def surface_count_scene(num_surfaces: int, beam_radius: float = 5.0) -> NSQScene:
    """Scene whose primitive count is the only thing that varies.

    A collimated beam travels down +z toward a detector. ``num_surfaces``
    decoy flat mirrors are placed along the way, offset far enough off-axis
    (``x = 500 * i`` mm) that no ray ever hits one -- their only effect is to
    make ``intersect_scene`` walk a longer component list every bounce, which
    is exactly the O(rays x surfaces) cost this harness measures.

    Args:
        num_surfaces: Number of decoy mirrors to add ahead of the detector.
        beam_radius: Collimated beam semi-diameter [mm].

    Returns:
        A validated NSQScene ready to trace.
    """
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-10.0),
        CollimatedSourceConfig(
            spectrum=_SPECTRUM, total_flux=1.0, aperture_radius=beam_radius
        ),
    )
    for i in range(num_surfaces):
        scene.add_mirror(
            f"decoy_{i}",
            CoordinateSystem(x=500.0 * (i + 1), z=10.0),
            MirrorConfig(radius=float("inf"), reflectance=0.9, aperture_radius=5.0),
        )
    scene.add_detector(
        "D",
        CoordinateSystem(z=100.0),
        IrradianceDetectorConfig(
            width=4 * beam_radius,
            height=4 * beam_radius,
            num_pixels_x=64,
            num_pixels_y=64,
        ),
    )
    return scene


def cavity_scene(reflectance: float = 0.98, beam_radius: float = 3.0) -> NSQScene:
    """Two-mirror cavity that forces many bounces per ray.

    A collimated beam launched from between two facing flat mirrors
    retroreflects back and forth at normal incidence. A transmissive
    (``absorb=False``) detector samples the beam on every pass without
    terminating it, so the ray keeps bouncing until Russian roulette or
    ``max_depth`` ends it -- letting the harness isolate per-bounce cost by
    sweeping ``max_depth`` alone on a fixed two-surface scene.

    Args:
        reflectance: Mirror reflectance (constant, both mirrors).
        beam_radius: Collimated beam semi-diameter [mm].

    Returns:
        A validated NSQScene ready to trace.
    """
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=0.0),
        CollimatedSourceConfig(
            spectrum=_SPECTRUM, total_flux=1.0, aperture_radius=beam_radius
        ),
    )
    scene.add_mirror(
        "M_far",
        CoordinateSystem(z=50.0),
        MirrorConfig(
            radius=float("inf"), reflectance=reflectance, aperture_radius=20.0
        ),
    )
    scene.add_mirror(
        "M_near",
        CoordinateSystem(z=-50.0),
        MirrorConfig(
            radius=float("inf"), reflectance=reflectance, aperture_radius=20.0
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=25.0),
        IrradianceDetectorConfig(
            width=4 * beam_radius,
            height=4 * beam_radius,
            num_pixels_x=32,
            num_pixels_y=32,
            absorb=False,
        ),
    )
    return scene


def lens_scene(aperture_radius: float = 12.5) -> NSQScene:
    """Single lens + detector, used to sweep ray count at fixed geometry.

    Args:
        aperture_radius: Front/back semi-diameter of the lens [mm].

    Returns:
        A validated NSQScene ready to trace.
    """
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-25.0),
        CollimatedSourceConfig(
            spectrum=_SPECTRUM,
            total_flux=1.0,
            aperture_radius=aperture_radius * 0.8,
        ),
    )
    scene.add_lens(
        "L",
        CoordinateSystem(z=0.0),
        LensConfig(
            r1=100.0,
            r2=-100.0,
            thickness=5.0,
            material=_LENS_MATERIAL,
            front_aperture_radius=aperture_radius,
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=150.0),
        IrradianceDetectorConfig(
            width=4 * aperture_radius,
            height=4 * aperture_radius,
            num_pixels_x=64,
            num_pixels_y=64,
        ),
    )
    return scene
