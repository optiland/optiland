"""Cross-tool validation against an independently developed optical design code.

Every other numerical test in this suite compares Optiland against Optiland: a
golden value captured from a previous run, or a round-trip through a file
format. Those catch regressions, but they cannot catch a quantity that was
wrong the first time it was computed — they only prove Optiland agrees with
itself.

The values in this file come from outside. Each one was read out of Ansys Zemax
OpticStudio 2023 R1.02 tracing the same prescription, as part of the validation
drive in #710. They are what makes this file worth its maintenance cost, and
also what makes it fragile in an unusual way:

    ⚠  These constants must NEVER be regenerated from Optiland.

    Regenerating them turns this file into an expensive self-consistency test
    and silently destroys the only independent check in the suite. If a value
    here starts failing, the possibilities are (a) a real regression, (b) a
    deliberate change in a definition or convention, or (c) the reference was
    wrong. Work out which. Do not "refresh" the number.

    This is also why the file lives in ``tests/`` rather than
    ``tests/regression/``: nothing here should ever be swept up by
    ``--update-golden``.

Provenance
----------
Tool          Ansys Zemax OpticStudio 2023 R1.02
Measured      2026-08-06 by @RayPragma, reported in
              https://github.com/optiland/optiland/issues/710
Settings      mm / degrees, angle fields, vignetting factors all zero, ray
              aiming off, 20 °C / 1 atm, unpolarized, no coatings. Spot
              diagrams referenced to the chief ray on a hexapolar grid with 6
              rings. Zernike listing set to Fringe (not Annular, despite the
              obscured pupil), 37 terms, no tilt or piston removal.
Source file   ``hubble.zmx`` exported by ``save_zemax_file`` after #712. The
              pre-#712 writer emitted 9 significant digits, which shifted ray
              intercepts by ~1.8e-06 mm and would have made several of the
              assertions below fail for reasons unrelated to tracing.

Tolerances follow the acceptance criteria stated in #710: 1e-4 relative on
first-order quantities, 1e-6 mm on real ray intercepts, 1% on spot radii, 1% or
0.01 waves on wavefront error, and 1% on Seidel sums ignoring sign convention.

Not covered: RMS wavefront error
--------------------------------
Deliberately absent, because the quantity is dominated by quadrature error
rather than by anything the two tools disagree about.

The hexapolar average converges as O(1/N) in the ring count. On the Hubble
at full field it reads 0.319178 / 0.303402 / 0.298539 / 0.296086 / 0.294854
waves at 12 / 32 / 64 / 128 / 256 rings; on a Double Gauss with a clear
circular pupil it reads 0.250448 / 0.243480 / 0.239942 / 0.238160 at
32 / 64 / 128 / 256. Successive differences halve as the count doubles in
both cases, so the central obscuration is not the cause.

Richardson-extrapolating the Double Gauss lands on the OpticStudio value to
every printed digit, on axis (0.236378 against 0.2364) and at full field
(1.218146 against 1.2182). So for a clear pupil the entire gap is the
reduction, not the wavefront.

The Hubble is the exception and is why this stays out of the suite: its
extrapolation converges to a stable 0.29362 against OpticStudio's 0.294380,
leaving 7.6e-04 waves, 0.26%, unexplained. That is inside the 0.01-wave
criterion, so an assertion would pass — but it would pass while sitting on
top of both an unconverged reduction and a residual nobody has localised
yet. It is the second Hubble-specific anomaly, alongside the tangential
focus shift in ``test_hubble_field_curvature_tangential`` below.

The rim is where the O(1/N) comes from: a hexapolar set with N rings puts
the outermost ring exactly at r = 1, but that ring covers half an annulus
and so wants half weight as a composite trapezoid endpoint. Giving it that
takes the on-axis Double Gauss error at 32 rings from 5.9% to 0.5% for the
same number of traced rays. It helps much less off axis, so the endpoint is
the dominant term only for a rotationally symmetric wavefront and a proper
Gauss-Legendre radial rule would be the real fix.

Worth adding here once the reduction is converged, and then as a convergence
assertion (successive ring counts must agree to X%) rather than a single
value, since a single value silently encodes whatever sampling produced it.
#710 has the full data.

Adding a system
---------------
Only the Hubble (``optiland.samples.telescopes.HubbleTelescope``) has been
measured so far. #710 also covers a singlet, a Cooke triplet and a Double
Gauss. To add one, trace it in a trusted tool under the settings above and add
a block of constants — the test bodies are written to generalize.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.analysis import FieldCurvature, SpotDiagram
from optiland.samples.telescopes import HubbleTelescope
from optiland.wavefront import ZernikeOPD

HUBBLE_WAVELENGTH = 0.55

# ---------------------------------------------------------------------------
# Reference values — OpticStudio 2023 R1.02, Hubble Space Telescope
# ---------------------------------------------------------------------------

# Read from the cardinal-point block of the Prescription Data report (12
# digits) where available, otherwise from System Data (6-7 digits).
#
# Each entry carries its own absolute tolerance rather than sharing the 1e-4
# relative criterion from #710. That criterion is the right one for judging a
# hand-written cross-tool report, but as a regression bound it is far too loose
# here: 1e-4 of a 57600 mm focal length is 5.8 mm, so an EFL that drifted by
# half a millimetre would sail through. The bounds below are set by how much
# precision each reference number actually carries.
#
# For five of the six that is the printed resolution — half of the last digit
# shown. `f2` is the exception: it is the only value read at 12 digits, so the
# limiting error is not the printout but the ~1e-3 mm the pre-#712 writer lost
# when the file was exported for measurement. Re-measuring against a file
# written by the current writer would allow this one to tighten by ~3 orders.
HUBBLE_FIRST_ORDER = {
    # quantity: (reference value, absolute tolerance)
    "f2": (57600.079948, 2e-3),
    "EPD": (2400.0, 1e-9),
    "EPL": (4910.01, 5e-3),
    "XPD": (289.9321, 5e-5),
    "XPL": (-6958.364, 1e-3),
    "FNO": (24.00003, 1e-5),
}

# From the "Seidel Aberration Coefficients" section (columns SPHA S1 ... DIST
# S5), TOT row. NOT the "Seidel Aberration Coefficients in Waves" section
# immediately below it, whose W-coefficients differ by a factor of ~200.
#
# OpticStudio reports every one of these with the opposite sign to Optiland.
# That is an overall sign-convention difference, not a disagreement, so the
# test compares magnitudes.
# The listing prints six decimals, so S1 carries a single significant digit
# and the absolute tolerance below is set by that, not by physics.
HUBBLE_SEIDEL_MAGNITUDES = [0.000003, 0.000337, 0.001452, 0.012841, 0.000760]
HUBBLE_SEIDEL_PRINT_RESOLUTION = 5e-7  # half of the last printed digit

# Image-surface row of the Real Ray Trace Data listing: Y-coordinate -> y,
# X-cosine -> L, Y-cosine -> M. Traced on the post-#712 file.
# (Hx, Hy, Px, Py) -> (x, y, L, M)
HUBBLE_RAYS = [
    ((0.0, 0.0, 0.0, 1.0), (0.0, 0.000292798, 0.0, -0.020832193)),
    ((0.0, 0.0, 0.0, 0.7), (0.0, 0.000224337, 0.0, -0.014582932)),
    ((0.0, 1.0, 0.0, 0.0), (0.0, 150.423380108, 0.0, 0.021668821)),
    ((0.0, 1.0, 0.0, 1.0), (0.0, 150.415481283, 0.0, 0.000731952)),
    ((0.0, 1.0, 0.0, -1.0), (0.0, 150.483684128, 0.0, 0.042499115)),
    ((0.0, 0.7, 0.7, 0.0), (0.011252785, 105.43303112, -0.014598926, 0.015153183)),
]

# Standard Spot Diagram, reported by OpticStudio in µm; converted to mm.
# "Max Spot Radius" is Optiland's GEO radius.
HUBBLE_SPOT_MM = {
    # field index -> (rms radius, geometric radius)
    0: (2.33518114e-01 / 1000, 2.92798267e-01 / 1000),
    1: (2.94822866e01 / 1000, 6.03040204e01 / 1000),
}

# Zernike Fringe Coefficient listing at H_y = 1.0, in waves.
HUBBLE_ZERNIKE_FRINGE = {
    1: 0.00042375,  # piston
    3: -0.22194993,  # y tilt
    4: -0.00051683,  # defocus
    5: -0.65888665,  # 0 deg astigmatism
    8: -0.11101475,  # y coma
    9: -0.00094328,  # spherical
}

# Field Curvature / Distortion listing, f-tan(theta), at maximum field.
HUBBLE_MAX_DISTORTION_PCT = -0.24752433
HUBBLE_SAGITTAL_SHIFT_MM = 1.65273862
HUBBLE_ZERO_FIELD_SHIFT_MM = 0.01686423

# One quantity does not currently agree; it is marked xfail below rather than
# omitted, so that it is re-checked on every run and announces itself via
# XPASS once resolved.
HUBBLE_TANGENTIAL_SHIFT_MM = -1.66509538  # Optiland gives -1.67515169 (0.6%)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def hubble(set_test_backend):
    """Built *after* the backend fixture, so the optic's arrays match it."""
    return HubbleTelescope()


def _scalar(value) -> float:
    """Collapse a backend scalar/0-d array to a plain float."""
    array = be.to_numpy(value).reshape(-1)
    return float(array[-1])


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quantity", "expected", "tolerance"),
    [(name, value, tol) for name, (value, tol) in sorted(HUBBLE_FIRST_ORDER.items())],
)
def test_hubble_first_order(hubble, quantity, expected, tolerance):
    """First-order quantities, bounded by each reference's own precision."""
    actual = _scalar(getattr(hubble.paraxial, quantity)())
    assert actual == pytest.approx(expected, abs=tolerance)


def test_hubble_seidel_sums(hubble):
    """Seidel sums S1-S5, compared as magnitudes.

    OpticStudio's sign convention is the opposite of Optiland's on all five
    terms. Comparing signed values here would report a convention difference as
    five failures.
    """
    actual = [abs(_scalar(s)) for s in hubble.aberrations.seidels()]
    assert len(actual) == len(HUBBLE_SEIDEL_MAGNITUDES)
    for got, expected in zip(actual, HUBBLE_SEIDEL_MAGNITUDES, strict=True):
        assert got == pytest.approx(
            expected, rel=1e-2, abs=HUBBLE_SEIDEL_PRINT_RESOLUTION
        )


@pytest.mark.parametrize(("pupil_field", "expected"), HUBBLE_RAYS)
def test_hubble_real_ray_intercepts(hubble, pupil_field, expected):
    """Real ray intercepts at the image surface.

    #710 calls this the most valuable row to validate and asks for agreement to
    1e-6 mm. The measured agreement is around 3e-9; the tolerance here is the
    stated criterion rather than the observed one, to leave room for the
    reference's own print precision.
    """
    rays = hubble.trace_generic(*pupil_field, wavelength=HUBBLE_WAVELENGTH)
    actual = tuple(_scalar(getattr(rays, attr)) for attr in "xyLM")
    for got, want, name in zip(actual, expected, "xyLM", strict=True):
        assert got == pytest.approx(want, abs=1e-6), f"component {name}"


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_index", sorted(HUBBLE_SPOT_MM))
def test_hubble_spot_radii(hubble, field_index):
    """RMS and geometric spot radii, chief-ray referenced, hexapolar, 6 rings.

    These are the defaults of ``SpotDiagram``, and they are also what the
    reference was measured with — OpticStudio defaults to a centroid reference
    and has to be switched over explicitly.
    """
    spot = SpotDiagram(hubble, fields="all", wavelengths=[HUBBLE_WAVELENGTH])
    expected_rms, expected_geo = HUBBLE_SPOT_MM[field_index]

    actual_rms = _scalar(spot.rms_spot_radius()[field_index][0])
    actual_geo = _scalar(spot.geometric_spot_radius()[field_index][0])

    assert actual_rms == pytest.approx(expected_rms, rel=1e-2)
    assert actual_geo == pytest.approx(expected_geo, rel=1e-2)


@pytest.mark.parametrize(("term", "expected"), sorted(HUBBLE_ZERNIKE_FRINGE.items()))
def test_hubble_zernike_fringe(hubble, term, expected):
    """Fringe Zernike coefficients at full field, piston and tilt retained.

    Note that the correct OpticStudio listing for this system is Zernike
    *Fringe*, not Zernike *Annular*, even though the pupil is obscured:
    Optiland normalizes over the unit circle.
    """
    opd = ZernikeOPD(
        hubble,
        field=(0.0, 1.0),
        wavelength=HUBBLE_WAVELENGTH,
        zernike_type="fringe",
        num_terms=37,
        remove_tilt=False,
    )
    actual = float(be.to_numpy(opd.zernike.coeffs[term - 1]))
    assert actual == pytest.approx(expected, abs=1e-5)


def test_hubble_max_distortion(hubble):
    """Maximum f-tan(theta) distortion, in percent."""
    from optiland.analysis import Distortion

    curve = Distortion(hubble, wavelengths=[HUBBLE_WAVELENGTH]).data[0]
    actual = float(be.to_numpy(curve).reshape(-1)[-1])
    assert actual == pytest.approx(HUBBLE_MAX_DISTORTION_PCT, rel=1e-2)


def _field_curvature_shifts(optic) -> tuple[float, float, float]:
    """Return (zero-field shift, tangential @ max field, sagittal @ max field)."""
    curves = FieldCurvature(optic, wavelengths=[HUBBLE_WAVELENGTH]).data[0]
    tangential = be.to_numpy(curves[0]).reshape(-1)
    sagittal = be.to_numpy(curves[1]).reshape(-1)
    return float(tangential[0]), float(tangential[-1]), float(sagittal[-1])


def test_hubble_field_curvature_datum(hubble):
    """On axis the two branches coincide, at the back focal distance.

    This pins the reference plane, which is what makes the sagittal/tangential
    comparison below meaningful rather than a datum argument.
    """
    zero_field, _, _ = _field_curvature_shifts(hubble)
    assert zero_field == pytest.approx(HUBBLE_ZERO_FIELD_SHIFT_MM, abs=1e-6)


def test_hubble_field_curvature_sagittal(hubble):
    """Sagittal focus shift at maximum field."""
    _, _, sagittal = _field_curvature_shifts(hubble)
    assert sagittal == pytest.approx(HUBBLE_SAGITTAL_SHIFT_MM, abs=1e-6)


@pytest.mark.xfail(
    reason=(
        "Open discrepancy from #710: the tangential focus shift disagrees by "
        "0.6% (Optiland -1.67515169, OpticStudio -1.66509538) while the "
        "sagittal branch agrees to 2e-08 at the same field, on the same "
        "wavefront, against the same datum. Ruled out on the Optiland side: "
        "field sampling (num_points 128 and 512 agree to 8 decimals) and the "
        "parabasal step size (stable for delta from 1e-3 to 1e-5). Remove this "
        "marker when resolved."
    ),
    strict=True,
)
def test_hubble_field_curvature_tangential(hubble):
    """Tangential focus shift at maximum field."""
    _, tangential, _ = _field_curvature_shifts(hubble)
    assert tangential == pytest.approx(HUBBLE_TANGENTIAL_SHIFT_MM, abs=1e-6)
