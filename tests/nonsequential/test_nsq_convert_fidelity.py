"""Round-trip fidelity tests for sequential_to_nonsequential (PR15).

For each reference Optic system, the converted NSQ scene's flux-weighted
irradiance centroid and core spot size must agree with the sequential
engine's own (geometric) spot diagram to a stated tolerance. This is the
converter's actual "does this produce the same optics" contract -- the
structural unit tests in test_nsq_convert.py check that the right *kind* of
NSQ objects come out, this checks that the *numbers* agree.

Core spot size uses the radius enclosing 50% of detected flux (r50), not
the RMS radius: NSQ physically models the Fresnel reflection the sequential
engine's geometric spot diagram doesn't trace at all (a real, if usually
faint, ghost -- e.g. a back-surface reflection re-emerging annulus-shaped
several mm from the primary image). That ghost is a few percent of total
flux, but RMS is quadratic in radius, so it dominates the RMS of an
otherwise-tight core spot; r50 is a standard, much more robust encircled
-energy statistic for exactly this "small tail, sharp core" shape.

Tolerances are loose enough to absorb the two engines' different sampling
schemes (hexapolar deterministic grid vs. Monte Carlo) and NSQ's ghost flux
while still catching an actual geometry-conversion bug (which would show up
as an O(1) mismatch, not a percent-level one).

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from optiland.nonsequential.convert import sequential_to_nonsequential


def _sequential_spot(optic, hx, hy, wavelength, num_rays=300):
    """Geometric spot centroid and r50 (median ray radius) at the image plane.

    Args:
        optic: Sequential Optic.
        hx: Normalized x field coordinate.
        hy: Normalized y field coordinate.
        wavelength: Wavelength [um].
        num_rays: Ray count for the hexapolar pupil sampling.

    Returns:
        ``(centroid_x, centroid_y, r50)`` [mm].
    """
    import optiland.backend as be

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rays = optic.trace(
            Hx=hx,
            Hy=hy,
            wavelength=wavelength,
            num_rays=num_rays,
            distribution="hexapolar",
        )
    x = be.to_numpy(rays.x)
    y = be.to_numpy(rays.y)
    cx, cy = float(x.mean()), float(y.mean())
    r50 = float(np.median(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)))
    return cx, cy, r50


def _nsq_spot(scene, num_rays=200_000, seed=1):
    """Flux-weighted irradiance centroid and r50 (50%-encircled-energy
    radius) on detector D1.

    Args:
        scene: Converted NSQScene.
        num_rays: Rays to trace.
        seed: RNG seed.

    Returns:
        ``(centroid_x, centroid_y, r50)`` [mm].
    """
    result = scene.trace(num_rays=num_rays, seed=seed)
    irr = result.detectors["D1"]
    xx, yy = np.meshgrid(irr.x_coords, irr.y_coords)
    w = irr.irradiance
    total = w.sum()
    cx = float((xx * w).sum() / total)
    cy = float((yy * w).sum() / total)

    r_flat = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
    w_flat = w.ravel()
    order = np.argsort(r_flat)
    cum = np.cumsum(w_flat[order])
    r50 = float(r_flat[order][np.searchsorted(cum, 0.5 * total)])
    return cx, cy, r50


def _assert_spots_agree(seq_spot, nsq_spot, *, centroid_abs=0.05, r50_rel=0.6):
    """Assert two (cx, cy, r50) spots agree to a stated tolerance.

    Args:
        seq_spot: Sequential engine's ``(cx, cy, r50)``.
        nsq_spot: NSQ engine's ``(cx, cy, r50)``.
        centroid_abs: Absolute+relative tolerance on centroid position [mm]
            (``pytest.approx`` combines them: passes within
            ``max(centroid_abs, 0.1 * |expected|)``).
        r50_rel: Relative tolerance on r50 -- loose (allows up to 60%) since
            r50 for a sub-millimetre spot is itself close to one detector
            pixel's width, so pixelation/binning noise is a large fraction
            of the value being compared, on top of Monte Carlo noise.
    """
    seq_cx, seq_cy, seq_r50 = seq_spot
    nsq_cx, nsq_cy, nsq_r50 = nsq_spot
    assert nsq_cx == pytest.approx(seq_cx, abs=centroid_abs, rel=0.1)
    assert nsq_cy == pytest.approx(seq_cy, abs=centroid_abs, rel=0.1)
    assert nsq_r50 == pytest.approx(seq_r50, rel=r50_rel)


# ---------------------------------------------------------------------------
# Reference systems
# ---------------------------------------------------------------------------


def _singlet_optic(field_y: float = 0.0):
    """A single-field singlet -- one field only, so the converted scene's
    lone detector unambiguously corresponds to this field's spot (adding a
    second field would mix both fields' flux onto the same detector,
    contaminating the centroid/RMS comparison).
    """
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=float("inf"))
    optic.add_surface(
        index=1, radius=50.0, thickness=5.0, material="N-BK7", is_stop=True
    )
    optic.add_surface(index=2, radius=-50.0, thickness=50.0)
    optic.add_surface(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="angle")
    optic.add_field(y=field_y)
    optic.add_wavelength(value=0.55, is_primary=True)
    return optic


def _doublet_optic():
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=float("inf"))
    optic.add_surface(
        index=1, radius=60.0, thickness=6.0, material="N-BK7", is_stop=True
    )
    optic.add_surface(index=2, radius=-30.0, thickness=2.0, material="N-F2")
    optic.add_surface(index=3, radius=-80.0, thickness=50.0)
    optic.add_surface(index=4)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="angle")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)
    return optic


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------


class TestSingletFidelity:
    def test_on_axis_spot_agrees(self):
        optic = _singlet_optic()
        seq_spot = _sequential_spot(optic, hx=0, hy=0, wavelength=0.55)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene = sequential_to_nonsequential(
                optic, detector_pixels=(128, 128), detector_width=4, detector_height=4
            )
        nsq_spot = _nsq_spot(scene)

        _assert_spots_agree(seq_spot, nsq_spot)

    def test_off_axis_spot_agrees(self):
        optic = _singlet_optic(field_y=5.0)
        seq_spot = _sequential_spot(optic, hx=0, hy=1, wavelength=0.55)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene = sequential_to_nonsequential(
                optic,
                detector_pixels=(128, 128),
                detector_width=20,
                detector_height=20,
            )
        nsq_spot = _nsq_spot(scene)

        _assert_spots_agree(seq_spot, nsq_spot)


class TestDoubletFidelity:
    def test_on_axis_spot_agrees(self):
        optic = _doublet_optic()
        seq_spot = _sequential_spot(optic, hx=0, hy=0, wavelength=0.55)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene = sequential_to_nonsequential(
                optic, detector_pixels=(128, 128), detector_width=4, detector_height=4
            )
        nsq_spot = _nsq_spot(scene)

        _assert_spots_agree(seq_spot, nsq_spot)


class TestFidelityWithCoating:
    def test_ar_coated_singlet_flux_matches_expected_transmission(self):
        """With AR coatings on both lens faces carried over, the
        detected flux must be close to (1-R1)(1-R2) of the launched flux --
        not the much larger loss a bare-Fresnel (uncoated) conversion would
        show. This is the direct test that the convert.py:187-era advice
        ("add coatings via SurfaceConfig.coating") is now something the
        converter itself honours automatically when the sequential system
        already has them.
        """
        from optiland.coatings import SimpleCoating
        from optiland.optic import Optic

        r1, t1 = 0.005, 0.995
        r2, t2 = 0.005, 0.995

        optic = Optic()
        optic.add_surface(index=0, thickness=float("inf"))
        optic.add_surface(
            index=1,
            radius=50.0,
            thickness=5.0,
            material="N-BK7",
            is_stop=True,
            coating=SimpleCoating(reflectance=r1, transmittance=t1),
        )
        optic.add_surface(
            index=2,
            radius=-50.0,
            thickness=50.0,
            coating=SimpleCoating(reflectance=r2, transmittance=t2),
        )
        optic.add_surface(index=3)
        optic.set_aperture(aperture_type="EPD", value=10.0)
        optic.set_field_type(field_type="angle")
        optic.add_field(y=0.0)
        optic.add_wavelength(value=0.55, is_primary=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene = sequential_to_nonsequential(
                optic, detector_pixels=(64, 64), detector_width=4, detector_height=4
            )
        assert scene.conversion_report.coated_surfaces == ["L1.front", "L1.back"]

        result = scene.trace(num_rays=100_000, seed=2)
        expected_transmission = t1 * t2
        actual_transmission = result.total_flux_detected / result.total_flux_in
        assert actual_transmission == pytest.approx(expected_transmission, rel=0.05)
