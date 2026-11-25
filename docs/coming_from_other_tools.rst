.. _coming_from_other_tools:

.. meta::
   :description: OpticStudio and CODE V terms mapped to their Optiland names: system setup, surfaces, solves, operands, analyses, tolerancing and file formats.

Coming from OpticStudio or CODE V
=================================

A term-by-term map for designers who know OpticStudio (Zemax) or CODE V. Each row
gives the name of the same thing in each tool. In the Optiland column, ``lens`` is an
:class:`~optic.optic.Optic`.

- An empty cell means there is no direct equivalent, or none that could be checked.
- Matching names do not promise matching numbers. Units, sign conventions and default
  ray sampling can differ between tools. :doc:`validation` lists the differences
  found so far.
- To bring a whole design across, see :ref:`coming_from_file_formats`.

----

System setup
------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - OpticStudio
     - CODE V
     - Optiland
   * - Aperture type: Entrance Pupil Diameter
     - ``EPD``
     - ``lens.set_aperture(aperture_type="EPD", value=...)``,
       :class:`~aperture.EPDAperture`
   * - Aperture type: Image Space F/#
     - ``FNO``
     - ``aperture_type="imageFNO"``, :class:`~aperture.ImageFNOAperture`
   * - Aperture type: Object Space NA
     - ``NAO``
     - ``aperture_type="objectNA"``, :class:`~aperture.ObjectNAAperture`
   * - Aperture type: Float By Stop Size
     -
     - ``aperture_type="float_by_stop_size"``, :class:`~aperture.FloatByStopAperture`
   * - Field type: Angle
     - Object angle (``XAN``, ``YAN``)
     - ``lens.fields.set_type("angle")``, then ``lens.fields.add(y=...)``,
       :class:`~fields.field_types.AngleField`
   * - Field type: Object Height
     - Object height (``XOB``, ``YOB``)
     - ``"object_height"``, :class:`~fields.field_types.ObjectHeightField`
   * - Field type: Paraxial Image Height
     - Paraxial image height (``XIM``, ``YIM``)
     - ``"paraxial_image_height"``,
       :class:`~fields.field_types.ParaxialImageHeightField`
   * - Field type: Real Image Height
     - Real image height (``XRI``, ``YRI``)
     - ``"real_image_height"``, :class:`~fields.field_types.RealImageHeightField`
   * - Vignetting factors VCX, VCY
     -
     - ``vx`` and ``vy`` in ``lens.fields.add()``
   * - Wavelengths, primary wavelength and weights
     - ``WL``, ``REF``, ``WTW``
     - ``lens.wavelengths.add(value=..., is_primary=True, weight=...)``,
       :class:`~wavelength.WavelengthGroup`
   * - Ray Aiming
     -
     - ``lens.ray_tracer.set_aiming(mode)`` with ``"paraxial"``, ``"iterative"`` or
       ``"robust"``

----

Surfaces
--------

Pass the Optiland name as ``surface_type`` to ``lens.surfaces.add()``.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - OpticStudio
     - CODE V
     - Optiland
   * - Standard
     - Sphere, Conic (``SPH``)
     - ``"standard"`` with ``radius`` and ``conic``,
       :class:`~geometries.standard.StandardGeometry`
   * - Even Asphere
     - Asphere (``ASP``)
     - ``"even_asphere"``, :class:`~geometries.even_asphere.EvenAsphere`
   * - Odd Asphere
     -
     - ``"odd_asphere"``, :class:`~geometries.odd_asphere.OddAsphere`
   * - Toroidal
     -
     - ``"toroidal"``, :class:`~geometries.toroidal.ToroidalGeometry`
   * - Biconic
     -
     - ``"biconic"``, :class:`~geometries.biconic.BiconicGeometry`
   * - Zernike Standard Sag
     -
     - ``"zernike"`` with ``zernike_type="noll"``,
       :class:`~geometries.zernike.ZernikePolynomialGeometry`. Optiland's
       ``"standard"`` is the OSA/ANSI ordering, not OpticStudio's.
   * - Zernike Fringe Sag
     -
     - ``"zernike"`` with ``zernike_type="fringe"``
   * - Chebyshev Polynomial
     -
     - ``"chebyshev"``, :class:`~geometries.chebyshev.ChebyshevPolynomialGeometry`
   * - Q-Type Asphere
     -
     - ``"forbes_qbfs"``, :class:`~geometries.forbes.ForbesQNormalSlopeGeometry`
   * - Q-Type Freeform
     -
     - ``"forbes_q2d"``, :class:`~geometries.forbes.ForbesQ2dGeometry`
   * - Grid Sag
     -
     - ``"grid_sag"``
   * - Paraxial
     - ``MFL``
     - ``"paraxial"`` with the focal length ``f``
   * - Coordinate Break
     - Decenter and tilt (``XDE``, ``YDE``, ``ZDE``, ``ADE``, ``BDE``, ``CDE``)
     - ``dx``, ``dy``, ``rx``, ``ry``, ``rz`` in ``lens.surfaces.add()``. See
       :doc:`Tutorial 4a <examples/Tutorial_4a_Tilts_Decenters_and_Asymmetric_Systems>`.
   * - Diffraction Grating
     -
     - ``"grating"``, :class:`~geometries.standard_grating.StandardGratingGeometry`
   * - Binary 2
     -
     - ``interaction_type="phase"`` with :class:`~optiland.phase.RadialPhaseProfile`
       (phase in even powers of r)

----

Solves and pickups
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - OpticStudio
     - CODE V
     - Optiland
   * - Thickness solve: Marginal Ray Height
     - ``PIM`` (paraxial image solve, height 0 at the image)
     - ``lens.solves.add("marginal_ray_height", surface_idx, height=...)``,
       :class:`~solves.thickness.MarginalRayHeightThicknessSolve`
   * - Thickness solve: Chief Ray Height
     -
     - ``"chief_ray_height"``, :class:`~solves.thickness.ChiefRayHeightThicknessSolve`
   * - Curvature solve: Marginal Ray Angle
     -
     - ``lens.solves.add("marginal_ray_angle", surface_idx, angle=...)``,
       :class:`~solves.curvature.MarginalRayAngleCurvatureSolve`
   * - Curvature solve: Chief Ray Angle
     -
     - ``"chief_ray_angle"``, :class:`~solves.curvature.ChiefRayAngleCurvatureSolve`
   * - Quick Focus
     -
     - ``lens.solves.add("quick_focus")``, :class:`~solves.quick_focus.QuickFocusSolve`
   * - Pickup
     - Pickup
     - ``lens.pickups.add(source_surface_idx, attr_type, target_surface_idx,
       scale, offset)``, :class:`~pickup.Pickup`

----

Optimization operands
---------------------

Operands go into an :class:`~optimization.problem.OptimizationProblem` with
``problem.add_operand(operand_type=..., target=..., weight=..., input_data=...)``.
Give ``min_val`` or ``max_val`` instead of ``target`` for a bound. The Optiland
names below are the keys of the operand registry
(``optiland.optimization.operand.operand_registry``).

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - OpticStudio
     - CODE V
     - Optiland
   * - EFFL
     - EFL (specific constraint)
     - ``"f2"``
   * - PMAG
     -
     - ``"magnification"``
   * - ENPP
     -
     - ``"EPL"`` (from the first surface)
   * - EPDI
     -
     - ``"EPD"``
   * - EXPP
     -
     - ``"XPL"`` (from the image surface)
   * - EXPD
     -
     - ``"XPD"``
   * - TOTR
     -
     - ``"total_track"``
   * - ISFN
     -
     -
   * - WFNO
     -
     -
   * - TTHI
     -
     - ``"center_thickness"`` for a single gap
   * - CTVA, CTGT, CTLT
     - Minimum and maximum center thickness (general constraints)
     - ``"center_thickness"`` with ``target``, ``min_val`` or ``max_val``
   * - ETVA, ETGT, ETLT
     - Minimum edge thickness (general constraints)
     - ``"edge_thickness"`` with ``target``, ``min_val`` or ``max_val``
   * - REAX, REAY, REAZ
     -
     - ``"real_x_intercept_lcs"``, ``"real_y_intercept_lcs"``,
       ``"real_z_intercept_lcs"`` (relative to the surface vertex)
   * - RAGX, RAGY, RAGZ
     -
     - ``"real_x_intercept"``, ``"real_y_intercept"``, ``"real_z_intercept"``
       (global coordinates)
   * - REAR
     -
     -
   * - TRAC
     -
     -
   * - RAID
     -
     - ``"AOI"``
   * - RSCE
     -
     - ``"rms_spot_size"`` (about the centroid)
   * - RWCE
     -
     - ``"rms_wavefront"`` with ``remove_tilt=True`` (piston and tilt removed);
       without it, piston only
   * - SPHA, COMA, ASTI, FCUR, DIST
     -
     - ``"seidel"`` with ``seidel_number`` 1 to 5
   * - DIMX
     -
     -
   * - MTFT, MTFS
     -
     - ``"mtf"`` with ``azimuth="tangential"`` or ``"sagittal"``
   * - BSER
     -
     - ``"boresight_x"``, ``"boresight_y"`` (signed components, in
       milliradians)
   * - ZERN
     -
     -

Optiland operands with no row above: ``"f1"``, ``"f2_range"``, ``"F1"``, ``"F2"``,
``"P1"``, ``"P2"``, ``"N1"``, ``"N2"``, ``"real_L"``, ``"real_M"``, ``"real_N"``,
``"clearance"``, ``"OPD_difference"``, and the per-surface third-order terms
``"TSC"``, ``"SC"``, ``"CC"``, ``"TCC"``, ``"TAC"``, ``"AC"``, ``"TPC"``, ``"PC"``,
``"DC"``, ``"TAchC"``, ``"LchC"``, ``"TchC"`` with their ``_sum`` forms.

----

Analyses
--------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - OpticStudio
     - CODE V
     - Optiland
   * - Spot Diagram
     - Spot Diagram
     - :class:`~analysis.spot_diagram.SpotDiagram`
   * - Ray Fan
     - Ray Aberration Curves
     - :class:`~analysis.ray_fan.RayFan`
   * - OPD Fan
     - OPD Aberration Curves
     - :class:`~optiland.wavefront.opd_fan.OPDFan`
   * - Field Curvature and Distortion
     -
     - :class:`~analysis.field_curvature.FieldCurvature`,
       :class:`~analysis.distortion.Distortion`
   * - Grid Distortion
     -
     - :class:`~analysis.grid_distortion.GridDistortion`
   * - Relative Illumination
     -
     - :class:`~analysis.relative_illumination.RelativeIllumination`
   * - FFT MTF
     - MTF (diffraction)
     - :class:`~mtf.fft.FFTMTF`
   * - Huygens MTF
     -
     - :class:`~mtf.huygens_fresnel.HuygensMTF`
   * - Geometric MTF
     - MTF (geometrical)
     - :class:`~mtf.geometric.GeometricMTF`
   * - MTF vs. Field
     -
     - :class:`~analysis.mtf_vs_field.MTFvsField`
   * - Through Focus MTF
     -
     - :class:`~analysis.through_focus_mtf.ThroughFocusMTF`
   * - FFT PSF
     - Point Spread Function
     - :class:`~psf.fft.FFTPSF`
   * - Huygens PSF
     -
     - :class:`~psf.huygens_fresnel.HuygensPSF`
   * - Geometric Encircled Energy
     -
     - :class:`~analysis.encircled_energy.EncircledEnergy`
   * - Wavefront Map
     - Pupil Map
     - :class:`~optiland.wavefront.opd.OPD`
   * - RMS vs. Field
     - Wavefront Analysis (RMS wavefront error)
     - :class:`~analysis.rms_vs_field.RmsWavefrontErrorVsField`,
       :class:`~analysis.rms_vs_field.RmsSpotSizeVsField`
   * - Zernike Standard Coefficients
     -
     - :class:`~optiland.wavefront.zernike_opd.ZernikeOPD` with
       ``zernike_type="noll"``
   * - Zernike Fringe Coefficients
     -
     - :class:`~optiland.wavefront.zernike_opd.ZernikeOPD` with
       ``zernike_type="fringe"``
   * - Seidel Coefficients, Seidel Diagram
     - Third Order Aberrations
     - ``lens.aberrations`` (:class:`~aberrations.Aberrations`),
       :class:`~analysis.seidel_diagram.SeidelDiagram`
   * - Longitudinal Aberration
     -
     - :class:`~analysis.longitudinal_aberration.LongitudinalAberration`
   * - Lateral Color
     -
     - :class:`~analysis.lateral_color.LateralColor`
   * - Chromatic Focal Shift
     -
     - :class:`~analysis.chromatic_focal_shift.ChromaticFocalShift`
   * - Footprint Diagram
     -
     - :class:`~analysis.footprint.FootprintDiagram`
   * - Prescription Data
     - List Lens Data
     - :class:`~prescription.prescription.Prescription`, ``lens.info()``

----

Tolerancing
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - OpticStudio
     - CODE V
     - Optiland
   * - Tolerance Data Editor
     -
     - :class:`~tolerancing.core.Tolerancing` with ``add_tolerance()``,
       ``add_perturbation()`` and ``add_operand()``. See :doc:`tolerancing_guide`.
   * - Tolerance Wizard
     -
     - ``add_default_tolerances(table)`` with your own table; no grade values
       are built in
   * - TRAD
     -
     - ``add_tolerance("radius", ...)``
   * - TFRN
     -
     - ``add_tolerance("fringes", ...)``
   * - TCON
     -
     - ``add_tolerance("conic", ...)``
   * - TTHI
     -
     - ``add_tolerance("thickness", ..., adjust=...)``
   * - TSDX, TSDY
     -
     - ``add_tolerance("surface_decenter", ..., axis="x")``, ``axis="y"``
   * - TSTX, TSTY
     -
     - ``add_tolerance("surface_tilt", ..., axis="x")``, ``axis="y"`` (degrees)
   * - TEDX, TEDY
     -
     - ``add_tolerance("element_decenter", ..., first=..., last=..., axis="x")``
   * - TETX, TETY
     -
     - ``add_tolerance("element_tilt", ..., axis="x", pivot=...)`` (degrees)
   * - TETZ
     -
     - ``add_tolerance("element_roll", ...)`` (degrees)
   * - TIND, TABB
     -
     - ``add_tolerance("index", ...)``, ``add_tolerance("abbe", ...)``
   * - TIRR
     -
     - ``add_tolerance("irregularity", ...)``. Kind ``"B"`` is pure
       astigmatism and kind ``"C"`` spherical; TIRR mixes the two equally.
   * - Compensators
     -
     - ``add_compensator()``, :class:`~tolerancing.compensator.CompensatorOptimizer`
   * - Compensator setting: Paraxial Focus
     -
     - ``add_compensator("quick_focus")``, a fast refocus without an optimizer
       (from real rays, not paraxial)
   * - Sensitivity
     -
     - :class:`~tolerancing.sensitivity_analysis.SensitivityAnalysis`
   * - Estimated change (RSS)
     -
     - ``SensitivityAnalysis.rss_estimate()``
   * - Worst offenders
     -
     - ``SensitivityAnalysis.view_tornado()``
   * - Inverse sensitivity, inverse limit
     -
     - :class:`~tolerancing.inverse_sensitivity.InverseSensitivity`
   * - Monte Carlo
     -
     - :class:`~tolerancing.monte_carlo.MonteCarlo`, ``run(n, seed=..., workers=...)``
   * - Monte Carlo statistics (yield, percent of trials better than)
     -
     - ``MonteCarlo.yield_fraction()``, ``MonteCarlo.percentiles()``
   * - Statistics: Normal, Uniform, Parabolic
     -
     - ``distribution="normal"`` (truncated at the limits, limits at ±2σ),
       ``"uniform"``, ``"parabolic"``; also ``"end_point"``
   * - ISO element drawing tolerances
     -
     - ``add_tolerances_from_drawing()``, ``drawing_values_from_tolerancing()``

----

.. _coming_from_file_formats:

File formats
------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - OpticStudio
     - CODE V
     - Optiland
   * - ``.zmx``
     -
     - ``load_zemax_file()``, :func:`~optiland.fileio.zemax.save_zemax_file`.
       See :doc:`Tutorial 1d <examples/Tutorial_1d_Saving_and_Loading>` and
       :doc:`zemax_import`.
   * -
     - ``.seq``
     - ``load_codev_file()``, :func:`~optiland.fileio.codev.save_codev_file`.
       See :doc:`gallery/miscellaneous/codev_import_demo`.

Optiland also reads and writes OSLO ``.len`` files (``load_oslo_file()``,
:func:`~optiland.fileio.oslo.save_oslo_file`, see :doc:`oslo_import`), and saves
any system as JSON with :func:`~optiland.fileio.save_optiland_file` and
:func:`~optiland.fileio.load_optiland_file`. All of these are imported from
``optiland.fileio``.
