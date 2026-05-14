"""Analysis registry for the Optiland GUI.

This module is the single source of truth for which core analysis classes
the GUI exposes.  Adding a new analysis to the core requires only a new
line in ``ANALYSIS_REGISTRY``; no other GUI code needs to change.

Each entry is a 3-tuple:
    ``(category, display_name, dotted_class_path)``

- *category*: Group header shown in the analysis selector (non-selectable).
- *display_name*: Label shown for the analysis item itself (selectable).
- *dotted_class_path*: Fully-qualified class path used by
  :func:`importlib.import_module` to load the class at runtime.
"""

from __future__ import annotations

ANALYSIS_REGISTRY: list[tuple[str, str, str]] = [
    # ------------------------------------------------------------------ #
    # Spot & Ray                                                           #
    # ------------------------------------------------------------------ #
    ("Spot & Ray", "Spot Diagram", "optiland.analysis.SpotDiagram"),
    ("Spot & Ray", "Ray Fan", "optiland.analysis.RayFan"),
    ("Spot & Ray", "Best-Fit Ray Fan", "optiland.analysis.BestFitRayFan"),
    (
        "Spot & Ray",
        "Through-Focus Spot",
        "optiland.analysis.ThroughFocusSpotDiagram",
    ),
    ("Spot & Ray", "Encircled Energy", "optiland.analysis.EncircledEnergy"),
    (
        "Spot & Ray",
        "RMS Spot Size vs Field",
        "optiland.analysis.RmsSpotSizeVsField",
    ),
    # ------------------------------------------------------------------ #
    # Wavefront                                                            #
    # ------------------------------------------------------------------ #
    ("Wavefront", "OPD", "optiland.wavefront.OPD"),
    ("Wavefront", "OPD Fan", "optiland.wavefront.OPDFan"),
    ("Wavefront", "Zernike OPD", "optiland.wavefront.ZernikeOPD"),
    (
        "Wavefront",
        "RMS Wavefront vs Field",
        "optiland.analysis.RmsWavefrontErrorVsField",
    ),
    # ------------------------------------------------------------------ #
    # PSF                                                                  #
    # ------------------------------------------------------------------ #
    ("PSF", "FFT PSF", "optiland.psf.FFTPSF"),
    ("PSF", "Huygens PSF", "optiland.psf.HuygensPSF"),
    ("PSF", "MMDFT PSF", "optiland.psf.MMDFTPSF"),
    # ------------------------------------------------------------------ #
    # MTF                                                                  #
    # ------------------------------------------------------------------ #
    ("MTF", "Geometric MTF", "optiland.mtf.GeometricMTF"),
    ("MTF", "FFT MTF", "optiland.mtf.FFTMTF"),
    # ------------------------------------------------------------------ #
    # Aberrations                                                          #
    # ------------------------------------------------------------------ #
    ("Aberrations", "YYbar", "optiland.analysis.YYbar"),
    ("Aberrations", "Pupil Aberration", "optiland.analysis.PupilAberration"),
    (
        "Aberrations",
        "Angle vs Height (Pupil)",
        "optiland.analysis.PupilIncidentAngleVsHeight",
    ),
    (
        "Aberrations",
        "Angle vs Height (Field)",
        "optiland.analysis.FieldIncidentAngleVsHeight",
    ),
    ("Aberrations", "Field Curvature", "optiland.analysis.FieldCurvature"),
    # ------------------------------------------------------------------ #
    # Distortion                                                           #
    # ------------------------------------------------------------------ #
    ("Distortion", "Distortion", "optiland.analysis.Distortion"),
    ("Distortion", "Grid Distortion", "optiland.analysis.GridDistortion"),
]
