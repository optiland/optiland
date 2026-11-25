.. _tolerancing_guide:

.. meta::
   :description: Tolerance an Optiland design: tolerances and perturbations, tolerance types and units, distributions, compensators, sensitivity, inverse sensitivity and Monte Carlo analyses, tolerance tables and the ISO 10110 bridge.

Tolerancing Guide
=================

Tolerancing answers the question every design has to face before it is built:
will the lens still meet its specification once every radius, thickness,
glass and mounting is slightly off? This guide covers the whole workflow in
:mod:`optiland.tolerancing`. For a worked example from start to finish, see
:doc:`Tutorial 6d <examples/Tutorial_6d_Complete_Tolerance_Study>`.

A tolerance study has four parts, all held by a
:class:`~tolerancing.core.Tolerancing` object:

1. **Operands**: the performance criteria, such as the RMS spot size or the RMS
   wavefront error. They are the same operands as in optimization.
2. **Tolerances** (or the lower-level *perturbations*): what may change, and by
   how much.
3. **Compensators** (optional): what is adjusted after assembly, such as the
   focus.
4. **An analysis**: :class:`~tolerancing.sensitivity_analysis.SensitivityAnalysis`,
   :class:`~tolerancing.inverse_sensitivity.InverseSensitivity` or
   :class:`~tolerancing.monte_carlo.MonteCarlo`.

.. code-block:: python

   from optiland.samples.objectives import CookeTriplet
   from optiland.tolerancing import SensitivityAnalysis, Tolerancing
   from optiland.tolerancing.monte_carlo import MonteCarlo

   lens = CookeTriplet()
   tolerancing = Tolerancing(lens)

   # 1. performance criteria
   tolerancing.add_operand(
       "rms_wavefront",
       {"optic": lens, "Hx": 0.0, "Hy": 1.0, "num_rays": 6, "wavelength": 0.55},
       target=0.0,
   )

   # 2. tolerances: deviations from nominal
   tolerancing.add_tolerance("radius", 0.1, surface_number=1)           # ±0.1 mm
   tolerancing.add_tolerance("thickness", minus=-0.02, plus=0.05, surface_number=3)
   tolerancing.add_tolerance("element_tilt", 0.03, first=1, last=2, axis="x")  # deg

   # 3. compensator: refocus by changing the last air gap
   tolerancing.add_compensator("thickness", surface_number=6)

   # 4. analyses
   sensitivity = SensitivityAnalysis(tolerancing)
   sensitivity.run()
   print(sensitivity.rss_estimate())

   monte_carlo = MonteCarlo(tolerancing)
   monte_carlo.run(500, seed=1)
   print(monte_carlo.yield_fraction({0: ("<=", 0.25)}))

Operands
--------

``Tolerancing.add_operand(operand_type, input_data, target=..., min_val=...,
max_val=...)`` takes any optimization operand (see
:doc:`optimization_merit_constraints`). The operand values are what the
analyses report, and they are also the merit function of the optimizer
compensators:

- With an explicit ``target``, the compensators drive the operand towards it.
  ``target=0.0`` on an error metric means "minimize".
- With no ``target`` and no bounds, the value at the time the operand is added
  becomes the target, so the compensators restore the nominal value.
- With ``min_val`` or ``max_val``, any value inside the range is acceptable and
  the compensators only act outside it.

Four operands were added with the tolerancing rework, because tolerance
specifications are usually stated with them. They are computed by real ray
tracing at one field point and wavelength:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Operand
     - Value
   * - ``"rms_wavefront"``
     - RMS wavefront error in waves, piston removed. ``remove_tilt=True`` also
       removes tilt (RMS referenced to the centroid).
   * - ``"mtf"``
     - MTF at one spatial frequency (cycles/mm), ``azimuth="tangential"`` or
       ``"sagittal"``.
   * - ``"boresight_x"``, ``"boresight_y"``
     - Image position of the chief ray of a field point, as an angle in
       milliradians: the pointing error of the system.

All four are in :mod:`optiland.optimization.operand.image_quality`.

Tolerances and perturbations
----------------------------

There are two ways to state what may change.

**Tolerances**, added with ``Tolerancing.add_tolerance``, are signed deviations
from the nominal design, between two limits, the way a drawing or an optical
shop states them:

.. code-block:: python

   tolerancing.add_tolerance("radius", 0.1, surface_number=1)          # [-0.1, +0.1]
   tolerancing.add_tolerance("fringes", minus=-1, plus=3, surface_number=2)
   tolerancing.add_tolerance("index", 0.0005, surface_number=1, distribution="uniform")

``limit`` gives the symmetric range ``[-limit, +limit]``; ``minus`` and
``plus`` give an asymmetric one. The deviation is always added to the nominal
value, so a tolerance does not need to know the design values, and the same
tolerance works on every design. Each call returns the
:class:`~tolerancing.tolerance.Tolerance`, whose ``minus``, ``plus`` and
``distribution`` can be changed later.

**Perturbations**, added with ``Tolerancing.add_perturbation(variable_type,
sampler, ...)``, set *absolute values* of an optimization variable, in the
variable's own units (for example a tilt in radians). Their values come from a
sampler: ``ScalarSampler`` (one value), ``RangeSampler`` (a sweep) or
``DistributionSampler`` (random draws). Perturbations can change any parameter
that has an optimization variable, which makes them the tool for parameters
that no tolerance type covers, and for sweeps. Tutorials
:doc:`6a <examples/Tutorial_6a_Tolerancing_Sensitivity_Analysis>` and
:doc:`6b <examples/Tutorial_6b_Monte_Carlo_Tolerancing_Analysis>` use them.

Both can be used in the same problem. The newer analysis features (the RSS
estimate per tolerance limit, inverse sensitivity, seeded and parallel Monte
Carlo) are built for tolerances; the notes below say where perturbations are
not supported.

Tolerance types
---------------

The registered types, their arguments and units:

.. list-table::
   :header-rows: 1
   :widths: 19 31 16 34

   * - Type
     - Arguments
     - Units
     - Deviation
   * - ``"radius"``
     - ``surface_number``
     - lens units
     - Radius of curvature. Not defined on a flat surface; use ``"fringes"``.
   * - ``"fringes"``
     - ``surface_number``, ``test_wavelength`` (µm, default 0.6328),
       ``semi_aperture``
     - fringes
     - Power: one fringe is half a test wavelength of sag at the edge of the
       clear aperture, applied as an exact change of radius. A positive value
       makes the surface more strongly curved.
   * - ``"conic"``
     - ``surface_number``
     - (none)
     - Conic constant.
   * - ``"thickness"``
     - ``surface_number``, ``adjust``
     - lens units
     - Thickness after the surface. With ``adjust``, the thickness after that
       surface changes by the opposite amount, so later surfaces stay in place.
   * - ``"index"``
     - ``surface_number``
     - Δn\ :sub:`d`
     - d-line index of the material after the surface. The Abbe number and the
       relative partial dispersions are kept.
   * - ``"abbe"``
     - ``surface_number``
     - Δν\ :sub:`d`
     - Abbe number of the material after the surface. The d-line index and the
       relative partial dispersions are kept. Needs a dispersive material.
   * - ``"irregularity"``
     - ``surface_number``, ``kind`` (``"B"`` or ``"C"``), ``angle`` (degrees,
       kind B), ``test_wavelength``, ``semi_aperture``
     - fringes
     - Form error as a Zernike sag error, with a peak-to-valley sag of the
       deviation times half the test wavelength. Kind ``"B"`` (default) is
       astigmatism, kind ``"C"`` rotationally symmetric (spherical, power
       removed), as in the ISO 10110-5 code ``3/A(B/C)``.
   * - ``"surface_decenter"``
     - ``surface_number``, ``axis`` (``"x"`` or ``"y"``)
     - lens units
     - Decenter of one surface.
   * - ``"surface_tilt"``
     - ``surface_number``, ``axis`` (``"x"`` or ``"y"``)
     - degrees
     - Tilt of one surface about its vertex.
   * - ``"element_decenter"``
     - ``first``, ``last``, ``axis`` (``"x"`` or ``"y"``)
     - lens units
     - Decenter of surfaces ``first`` to ``last`` as one rigid body.
   * - ``"element_tilt"``
     - ``first``, ``last``, ``axis`` (``"x"`` or ``"y"``), ``pivot``
     - degrees
     - Tilt of the element about the pivot.
   * - ``"element_roll"``
     - ``first``, ``last``, ``pivot``
     - degrees
     - Rotation of the element about its own axis through the pivot.

Lens units are those of the optic, usually millimetres. Element types move the
surfaces ``first`` to ``last`` (inclusive); the surfaces after the element do
not move. ``pivot`` is ``"center"`` (default, midway between the first and the
last vertex), ``"first"``, ``"last"``, or a distance from the first vertex
along the element axis. Elements can also be optimized: the variables
``"element_decenter"``, ``"element_tilt"`` and ``"element_roll"`` take the
same arguments.

The index and Abbe tolerances never modify the glass: they replace the
material with an :class:`~materials.offset.OffsetMaterial` that wraps it, and
restore the original object on reset. The irregularity tolerance likewise wraps
the surface geometry in a :class:`~geometries.sag_error.SagErrorGeometry`.

``tolerance_registry.names()`` lists the registered types. To add a type,
subclass :class:`~tolerancing.registry.ToleranceType` and register it; see
:doc:`developers_guide/tolerancing_framework`.

Order of application
--------------------

In a trial, tolerances are applied in *phases*, not in the order they were
added:

1. **Structure**: radius, fringes, conic, thickness.
2. **Material**: index, Abbe number.
3. **Form**: irregularity.
4. **Rigid body**: surface and element decenters, tilts and rolls.

Within a phase, the order of insertion is kept. A thickness change moves every
surface after it, so rigid-body moves come last and are applied to the
surfaces where they already are; reset runs in the reverse order and restores
the nominal design exactly. ``Tolerancing.ordered_perturbations()`` returns the
order used. Perturbations added with ``add_perturbation`` belong to the
structure phase.

Distributions
-------------

Monte Carlo analyses draw each deviation from the tolerance's distribution,
set with ``distribution=``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Name
     - Distribution
   * - ``"normal"`` (default)
     - Normal distribution truncated at the limits, centred between them, with
       the limits at ±2σ. Use
       :class:`~tolerancing.distributions.TruncatedNormal` with ``num_sigma``
       for another width.
   * - ``"uniform"``
     - Uniform between the limits.
   * - ``"parabolic"``
     - Density proportional to the squared distance from the centre: values
       near the limits are the most likely.
   * - ``"end_point"``
     - One of the two limits, with equal probability.

A :class:`~tolerancing.distributions.Distribution` instance can be passed
instead of a name. Sampling always uses NumPy's random generator, so seeded
results are the same on both backends. Sensitivity and inverse sensitivity
analyses do not sample: they use the limits.

.. _tolerancing_compensators:

Compensators
------------

A compensator models an adjustment made after assembly. Optiland has two kinds.

**Optimizer compensators**: ``add_compensator(variable_type, **kwargs)`` adds an
optimization variable, such as ``"thickness"`` (refocus), ``"decenter"`` or
``"tilt"``. On every trial, the compensator optimizer adjusts the variables to
minimize the operands (see *Operands* above). It runs through
:func:`optiland.optimization.minimize`; ``Tolerancing(optic, method=...,
tol=...)`` chooses L-BFGS-B (``"generic"``, default) or Levenberg–Marquardt
(``"least_squares"``). A trial that cannot be traced is left uncompensated
instead of stopping the analysis. The compensated values are reported in the
result tables, in columns named ``"C0: ..."``, ``"C1: ..."``.

**Quick focus**: ``add_compensator("quick_focus", Hx=0, Hy=0, wavelength="primary",
num_rays=5, distribution="hexapolar")`` moves the image surface along z with
the closed-form :class:`~solves.quick_focus.QuickFocusSolve` instead of an
optimizer, which is much faster. It runs before the optimizer compensators, and
adding it again replaces it. Its result column is ``"quick_focus: image
distance"``, the distance from the last surface to the image.

.. caution::

   ``QuickFocusSolve`` places the image at the mean of the axial positions
   where each traced ray comes closest to the optical axis. This is close to,
   but not exactly, the minimum RMS spot position, and it is measured from the
   optical axis. It is meant for coaxial systems: when decenters or tilts move
   the image off the axis, use an optimizer compensator, such as the
   thickness of the last air gap.

Analyses
--------

Sensitivity analysis
~~~~~~~~~~~~~~~~~~~~

:class:`~tolerancing.sensitivity_analysis.SensitivityAnalysis` applies each
tolerance alone, at its lower and at its upper limit (a perturbation over every
value of its ``RangeSampler``), applies the compensators and evaluates the
operands. ``get_results()`` returns one row per trial. After the trials, the
operands are evaluated once more at nominal, with compensators, which is the
reference for the summaries:

- ``rss_estimate()`` takes, for each tolerance, the larger change of each
  operand at its two limits, and combines the changes by root sum of squares
  (RSS). It returns the compensated ``nominal``, the ``rss_change`` and the
  ``estimate`` (their sum) per operand. RSS assumes small, independent
  changes; use Monte Carlo to check it.
- ``view_tornado(operand_index=0, top=None)`` plots the change at each end of
  every tolerance, sorted with the largest on top.
- ``view()`` plots each operand against each perturbation value.

A sensitivity analysis takes at most six operands and six perturbations or
tolerances, a limit of its plot grid. For a full tolerance set, split it over
several runs, or use inverse sensitivity or Monte Carlo, which have no limit.

Inverse sensitivity
~~~~~~~~~~~~~~~~~~~

:class:`~tolerancing.inverse_sensitivity.InverseSensitivity` turns the question
around: given an allowed change of some operands, how large may each tolerance
be? ``max_change`` maps an operand index to the allowed absolute change from
the compensated nominal value:

.. code-block:: python

   from optiland.tolerancing import InverseSensitivity

   inverse = InverseSensitivity(tolerancing, max_change={0: 0.05})
   inverse.run()
   inverse.get_results()   # inverse_minus, inverse_plus, limiting_*, capped_*
   inverse.apply_limits()  # optional: write the limits into the tolerances

Each side of each tolerance is searched on its own: the deviation is applied
alone and compensated, and the search finds the largest deviation that keeps
every listed operand within its change (bracketing from the current limit,
then Brent's method). A side that still meets the budget at ``max_factor``
(default 100) times the current limit is reported as ``capped``. Only
tolerances are searched; perturbations stay at nominal.

Monte Carlo
~~~~~~~~~~~

:class:`~tolerancing.monte_carlo.MonteCarlo` applies all tolerances at once,
each drawn from its distribution, compensates and evaluates, for
``num_iterations`` trials:

.. code-block:: python

   monte_carlo = MonteCarlo(tolerancing)
   monte_carlo.run(1000, seed=42)

   monte_carlo.percentiles()                       # 50, 90 and 98 % by default
   monte_carlo.yield_fraction({0: ("<=", 0.25),    # operand index ...
                               "C0: Thickness, Surface 6": (">", 41.5)})  # ... or column
   monte_carlo.view_histogram()
   monte_carlo.view_cdf()
   monte_carlo.view_heatmap()

- ``seed`` makes a run reproducible: each trial draws from its own generator,
  spawned from ``numpy.random.SeedSequence(seed)``. Perturbations keep their
  own samplers and are not seeded.
- ``yield_fraction(spec)`` returns the fraction of trials that meet every
  condition. A trial with a NaN value (one that could not be traced) fails.
- ``percentiles(q)`` gives the operand and compensator values per percentile,
  leaving out NaN trials.
- ``workers=n`` splits the trials over ``n`` processes. The results equal a
  serial run with the same seed. It needs the NumPy backend and tolerances
  only (no ``add_perturbation``), and on Windows and macOS the call must sit
  under ``if __name__ == "__main__":``, as for any Python multiprocessing.

.. _tolerancing_tables:

Tolerance tables
----------------

``Tolerancing.add_default_tolerances(table, surfaces=None)`` applies a table of
tolerances to every matching part of the design, the way a shop's standard
tolerances apply to every lens:

.. code-block:: python

   shop_table = {
       "radius": 0.05,
       "fringes": 2,
       "irregularity": 0.5,
       "thickness": 0.05,
       "surface_tilt": 0.017,                      # 1 arc minute
       "element_decenter": 0.02,
       "index": {"limit": 0.0005, "distribution": "uniform"},
       "abbe": 0.5,
   }
   added = tolerancing.add_default_tolerances(shop_table)

An entry is a number (a symmetric limit) or a dictionary of ``limit``, or
``minus`` and ``plus``, and optionally ``distribution``. Types are placed as
follows:

- surface types (radius, fringes, conic, irregularity, surface decenter and
  tilt): every refracting or reflecting surface; radius and conic only on
  curved ones;
- ``"thickness"``: every gap next to such a surface, except the gap before the
  image, which is usually the focus compensator;
- element types: every run of surfaces with glass between them;
- ``"index"`` and ``"abbe"``: every surface followed by glass;
- decenters and tilts once about x and once about y.

``surfaces=`` restricts the table to some surface numbers (an element is kept
when its first surface is listed). Types without a default placement, such as
``"element_roll"``, are added with ``add_tolerance``.

**Optiland ships no grade tables.** The "commercial", "precision" and "high
precision" values found in textbooks and vendor catalogs differ from vendor to
vendor and change with the part size and the process, and no standard defines
them: ISO 10110 defines how a tolerance is written on a drawing, not its value.
Built-in values would look authoritative without being so. Use the table your
shop or vendor quotes, and keep it with your project.

.. _tolerancing_iso10110:

ISO 10110 drawings
------------------

:mod:`optiland.tolerancing.iso10110_bridge` connects tolerancing with the ISO
10110 drawing specification of :mod:`optiland.iso10110`
(:doc:`Tutorial 11a <examples/Tutorial_11a_ISO_10110_Drawings>`):

- ``Tolerancing.add_tolerances_from_drawing(drawing)`` (or
  ``tolerances_from_drawing(tolerancing, drawing)``) adds the tolerances stated
  by a :class:`~iso10110.spec.DrawingSpec`.
- ``drawing_values_from_tolerancing(tolerancing)`` reports the drawing fields
  implied by the tolerances, as keyword arguments for
  ``DrawingSpec.set_surface`` (under ``"surfaces"``) and
  ``DrawingSpec.set_element`` (under ``"elements"``).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Drawing field
     - Tolerance
   * - ``r_tolerance`` (``±a``, ``a`` or ``+a/-b``, mm)
     - ``"radius"``. Flat surfaces are skipped.
   * - ``3/A(B/C)``: ``sag_tolerance`` (A), ``irregularity`` (B),
       ``rotationally_symmetric`` (C)
     - ``"fringes"``, ``"irregularity"`` kind B, ``"irregularity"`` kind C,
       at the drawing's ``test_wavelength`` (546.07 nm when unset). With
       ``form_unit="nm"`` the values are converted to fringes.
   * - ``4/σ'(δ)``: ``centration`` σ (arc minutes),
       ``centration_decentration`` δ (mm)
     - ``"surface_tilt"`` about x and y, in degrees, and ``"surface_decenter"``
       in x and y. ``centering="per_axis"`` (default) gives each axis ±σ;
       ``centering="rss"`` gives ±σ/√2, so the combined tilt is σ.
   * - ``ct_tolerance`` of a singlet (mm)
     - ``"thickness"`` after its first surface. With
       ``adjust_thickness=True`` (default) the following air space absorbs the
       change. Cemented elements are skipped.

ISO 10110 has no field for an index or Abbe-number offset (the homogeneity
classes are not offsets), so these, air-space thicknesses and element
decenters, tilts and rolls are not read from or written to drawings. A field
that cannot be mapped is skipped with a ``UserWarning``, or raises with
``strict=True``.

See also
--------

- :doc:`Tutorial 6d: A Complete Tolerance Study <examples/Tutorial_6d_Complete_Tolerance_Study>`
- :doc:`gallery/tolerancing`
- :doc:`api/api_tolerancing`
- :ref:`coming_from_other_tools` for the OpticStudio names of these features
