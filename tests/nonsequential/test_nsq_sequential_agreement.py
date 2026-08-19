"""Cross-validation of the NSQ tracer against the sequential tracer.

The sequential tracer is the mature, independently validated engine in
Optiland. NSQ is a separate implementation — its own intersection routines,
its own refraction, its own coordinate transforms — so agreement between the
two on the same optical prescription is the strongest available evidence
that the NSQ physics is right. A unit test of Snell's law confirms one
formula; this confirms the whole chain.

Two quantities are compared for an on-axis collimated beam:

- **Marginal ray height** at the image plane: the edge-of-aperture ray,
  deterministic in the sequential engine and the outermost sampled ray in
  NSQ, so the two must agree tightly (measured: within 0.02%). It is
  sensitive to surface sag, surface normals, and refraction at both faces,
  the conic normal formula in particular.
- **RMS spot radius**. Compared loosely: the engines sample the pupil
  differently (hexapolar grid vs. uniform Monte Carlo), which changes the
  radial weighting, so a few percent of difference is expected and fine.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    DoubletConfig,
    LensConfig,
    NSQScene,
    RayDatabaseConfig,
    Spectrum,
)

_WAVELENGTH = 0.55
_EPD = 10.0
_LENS_Z = 10.0


def _sequential_spot(
    surfaces: list[dict], back_focus: float, epd: float = _EPD
) -> tuple[float, float]:
    """Trace an on-axis collimated beam with the sequential engine.

    Args:
        surfaces: Surface dicts (radius/thickness/material) after the object.
        back_focus: Distance from the last surface to the image plane [mm].
        epd: Entrance pupil diameter [mm].

    Returns:
        ``(rms_radius, marginal_radius)`` in mm.
    """
    from optiland.optic import Optic

    optic = Optic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        optic.add_surface(index=0, thickness=float("inf"))
        for i, surf in enumerate(surfaces, start=1):
            optic.add_surface(index=i, is_stop=(i == 1), **surf)
        optic.add_surface(index=len(surfaces) + 1)
        optic.set_aperture(aperture_type="EPD", value=epd)
        optic.set_field_type(field_type="angle")
        optic.add_field(y=0.0)
        optic.add_wavelength(value=_WAVELENGTH, is_primary=True)

        optic.trace(
            Hx=0, Hy=0, wavelength=_WAVELENGTH, num_rays=64, distribution="hexapolar"
        )
        x = np.asarray(be.to_numpy(optic.surfaces.x[-1])).ravel()
        y = np.asarray(be.to_numpy(optic.surfaces.y[-1])).ravel()

    radius = np.hypot(x, y)
    return float(np.sqrt(np.mean(radius**2))), float(radius.max())


def _nsq_spot(scene: NSQScene, num_rays: int = 20_000) -> tuple[float, float]:
    """Trace the equivalent NSQ scene and measure the spot.

    A few ghost rays (ones whose Fresnel branch sampled reflection at an
    interface) reach the detector far from focus. They cannot be separated
    by flux: the detached-sample / attached-weight estimator keeps every
    ray's forward flux at its full value and represents reflection by
    *sampling* the branch, so a ghost carries exactly the same flux as a
    direct ray. That is the correct unbiased behaviour, and it means ghosts
    must be rejected geometrically instead.

    Direct rays form a dense cluster that runs continuously out to the
    marginal ray, while ghosts sit several times further out, so a gap
    criterion separates them cleanly: anything beyond 1.5x the 99th
    percentile radius is a ghost.

    The marginal ray height is then the *maximum* radius of the direct
    cluster, matching the statistic taken from the sequential tracer. A
    percentile must not be used here: for a uniformly sampled disk the 99.5th
    percentile corresponds to pupil radius sqrt(0.995) * R rather than the
    edge, and under strong spherical aberration (image radius growing roughly
    as pupil cubed) that undershoots the true marginal ray by ~0.7% -- an
    artefact of the statistic, not a disagreement between the engines.

    Args:
        scene: Scene whose detector "D1" is a RayDatabaseDetector.
        num_rays: Rays to launch.

    Returns:
        ``(rms_radius, marginal_radius)`` in mm.
    """
    result = scene.trace(num_rays=num_rays, seed=1, max_depth=8)
    db = result.detectors["D1"]
    radius = np.hypot(np.asarray(db.x), np.asarray(db.y))
    assert radius.size > 0.5 * num_rays, "Most rays should reach the detector"

    ghost_cut = 1.5 * float(np.percentile(radius, 99.0))
    direct = radius <= ghost_cut
    assert direct.sum() > 0.95 * radius.size, (
        "Ghost rejection discarded too many rays; the direct cluster and the "
        "ghost population are not cleanly separated in this scene"
    )

    rms = float(np.sqrt(np.mean(radius[direct] ** 2)))
    return rms, float(radius[direct].max())


def _add_source_and_detector(
    scene: NSQScene, image_z: float, epd: float = _EPD
) -> None:
    """Attach the standard collimated source and a ray-database detector."""
    scene.add_source(
        "S1",
        CoordinateSystem(z=0.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(_WAVELENGTH),
            total_flux=1.0,
            aperture_radius=epd / 2.0,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=image_z),
        RayDatabaseConfig(width=40.0, height=40.0),
    )


class TestSequentialAgreement:
    """NSQ must reproduce the sequential tracer's spot for the same optic."""

    def setup_method(self):
        be.set_backend("numpy")

    def test_singlet_matches_sequential(self):
        """Biconvex N-BK7 singlet, on-axis collimated beam."""
        r1, r2, thickness, back_focus = 50.0, -50.0, 5.0, 45.0

        rms_seq, marginal_seq = _sequential_spot(
            [
                {"radius": r1, "thickness": thickness, "material": "N-BK7"},
                {"radius": r2, "thickness": back_focus},
            ],
            back_focus,
        )

        scene = NSQScene()
        scene.add_lens(
            "L1",
            CoordinateSystem(z=_LENS_Z),
            LensConfig(
                r1=r1,
                r2=r2,
                thickness=thickness,
                material="N-BK7",
                front_aperture_radius=_EPD / 2.0 + 1.0,
            ),
        )
        _add_source_and_detector(scene, _LENS_Z + thickness + back_focus)
        rms_nsq, marginal_nsq = _nsq_spot(scene)

        assert marginal_nsq == pytest.approx(marginal_seq, rel=2e-3), (
            f"Marginal ray height disagrees: sequential {marginal_seq * 1e3:.3f} um "
            f"vs NSQ {marginal_nsq * 1e3:.3f} um"
        )
        assert rms_nsq == pytest.approx(rms_seq, rel=0.03), (
            f"RMS spot disagrees: sequential {rms_seq * 1e3:.3f} um "
            f"vs NSQ {rms_nsq * 1e3:.3f} um"
        )

    def test_doublet_matches_sequential(self):
        """Cemented doublet — three refracting surfaces and two glasses."""
        r1, r2, r3 = 60.0, -30.0, -80.0
        t1, t2, back_focus = 6.0, 2.0, 50.0

        rms_seq, marginal_seq = _sequential_spot(
            [
                {"radius": r1, "thickness": t1, "material": "N-BK7"},
                {"radius": r2, "thickness": t2, "material": "N-F2"},
                {"radius": r3, "thickness": back_focus},
            ],
            back_focus,
        )

        scene = NSQScene()
        scene.add_doublet(
            "D_LENS",
            CoordinateSystem(z=_LENS_Z),
            DoubletConfig(
                r1=r1,
                r2=r2,
                r3=r3,
                thickness1=t1,
                thickness2=t2,
                material1="N-BK7",
                material2="N-F2",
                aperture_radius=_EPD / 2.0 + 1.0,
            ),
        )
        _add_source_and_detector(scene, _LENS_Z + t1 + t2 + back_focus)
        rms_nsq, marginal_nsq = _nsq_spot(scene)

        assert marginal_nsq == pytest.approx(marginal_seq, rel=2e-3), (
            f"Marginal ray height disagrees: sequential {marginal_seq * 1e3:.3f} um "
            f"vs NSQ {marginal_nsq * 1e3:.3f} um"
        )
        assert rms_nsq == pytest.approx(rms_seq, rel=0.03), (
            f"RMS spot disagrees: sequential {rms_seq * 1e3:.3f} um "
            f"vs NSQ {rms_nsq * 1e3:.3f} um"
        )

    def test_aspheric_singlet_matches_sequential(self):
        """Strongly aspheric, fast front surface — exercises the conic normal.

        A wrong conic normal is invisible on a sphere (K = 0) and nearly
        invisible on a slow one, but tilts every refracted ray once the
        surface is both fast and strongly conic, with the error growing
        toward the aperture edge. This configuration is chosen to have that
        sensitivity: the previous (incorrect) normal expression deviates
        from the sequential tracer by 5.6% here, against 0.02% for the
        analytic curvature form, so the tolerance below separates them by a
        wide margin.
        """
        epd = 20.0
        r1, conic, r2, thickness, back_focus = 30.0, 2.5, -50.0, 5.0, 45.0

        rms_seq, marginal_seq = _sequential_spot(
            [
                {
                    "radius": r1,
                    "thickness": thickness,
                    "material": "N-BK7",
                    "conic": conic,
                },
                {"radius": r2, "thickness": back_focus},
            ],
            back_focus,
            epd=epd,
        )

        scene = NSQScene()
        scene.add_lens(
            "L1",
            CoordinateSystem(z=_LENS_Z),
            LensConfig(
                r1=r1,
                r2=r2,
                conic1=conic,
                thickness=thickness,
                material="N-BK7",
                front_aperture_radius=epd / 2.0 + 1.0,
            ),
        )
        _add_source_and_detector(scene, _LENS_Z + thickness + back_focus, epd=epd)
        rms_nsq, marginal_nsq = _nsq_spot(scene)

        assert marginal_nsq == pytest.approx(marginal_seq, rel=2e-3), (
            f"Marginal ray height disagrees for K={conic}: sequential "
            f"{marginal_seq * 1e3:.3f} um vs NSQ {marginal_nsq * 1e3:.3f} um"
        )
        assert rms_nsq == pytest.approx(rms_seq, rel=0.03), (
            f"RMS spot disagrees for K={conic}: sequential {rms_seq * 1e3:.3f} um "
            f"vs NSQ {rms_nsq * 1e3:.3f} um"
        )
