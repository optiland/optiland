.. _validation:

.. meta::
   :description: How Optiland's results compare with Zemax OpticStudio and CODE V on four reference optical systems.

Validation
==========

Optiland's own test suite shows that Optiland agrees with *itself* over time. It
cannot show that the physics is right. The only way to show that is to trace the
same system in independently developed, widely used tools and compare numbers.

This page reports that comparison. Four reference systems were traced in
**Zemax OpticStudio** and **CODE V** and compared with Optiland, quantity by
quantity.

.. admonition:: Summary
   :class: tip

   On all four systems, **first-order properties, third-order (Seidel)
   aberrations and real-ray intercepts agree with both OpticStudio and CODE V**,
   to the last printed digit in most cases and to better than 1 × 10\ :sup:`-6` mm
   on real-ray intercepts. Image-quality metrics (spot size, wavefront error,
   MTF) have been compared in more limited form and agree within the stated
   tolerances, with a few open items listed :ref:`below <validation_open>`.

.. note::

   This is a comparison on four systems, not a certification. It does not cover
   every surface type, material model or analysis in Optiland. The
   :ref:`scope section <validation_scope>` says exactly what has and has not been
   checked.

----

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 22 22 32

   * - System
     - OpticStudio
     - CODE V
     - What it exercises
   * - **Singlet**, stop on rear surface
     - Tier 1 agrees
     - Tier 1 agrees
     - Real-ray and first-order correctness on a strongly aberrated lens
   * - **Cooke triplet**
     - Tier 1 and Tier 2 agree
     - Tier 1 agrees (see :ref:`glass note <validation_glass>`)
     - Interior stop, three-element design, glass catalog handling
   * - **Double Gauss**, F/5
     - Tier 1 agrees
     - Tier 1 agrees
     - Aperture defined by image-space F/#, cemented doublets, pupil solving
   * - **Hubble Space Telescope**
     - Tier 1 agrees, Tier 2 mostly (one open item)
     - Tier 1 agrees
     - Mirrors, conics, negative thicknesses, central obscuration, curved image

*Tier 1* is first-order properties, Seidel sums and single real-ray intercepts.
*Tier 2* is spot radii, wavefront error, distortion, field curvature, Zernike
coefficients and MTF.

----

What was compared
-----------------

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: First order
      :class-card: sd-shadow-sm

      Effective focal length, entrance and exit pupil size and position,
      image-space F/#, back focal point, principal plane, Lagrange invariant.

   .. grid-item-card:: Third-order aberrations
      :class-card: sd-shadow-sm

      Seidel sums S1 to S5 plus longitudinal axial and lateral color.

   .. grid-item-card:: Real rays
      :class-card: sd-shadow-sm

      Six individual rays per system (on axis, full field, 70 % field, several
      pupil positions): image-plane position and direction cosines.

Agreement criteria
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Quantity
     - Counted as agreement when
   * - First-order properties
     - Within 1 × 10\ :sup:`-4` relative
   * - Real-ray intercepts
     - Within 1 × 10\ :sup:`-6` mm
   * - Seidel sums
     - Within 1 %, ignoring the overall sign convention
   * - Spot radii
     - Within 1 %
   * - RMS wavefront error
     - Within 1 % or 0.01 waves, whichever is larger
   * - MTF
     - Within 0.01 absolute

----

Results
-------

Tool versions: OpticStudio 2023 R1.02, 2025 R2.02 and 2026 R1.00 (depending on
the system) and CODE V 2025.03.

Real rays
~~~~~~~~~

Real-ray intercepts are the most demanding test, because small errors in
surface intersection, refraction or reflection show up directly.

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - System
     - OpticStudio
     - CODE V
   * - Singlet
     - Agrees to every printed digit
     - Agrees to every printed digit
   * - Cooke triplet
     - Agrees to every printed digit (< 1 × 10\ :sup:`-6` mm)
     - Up to 2.3 × 10\ :sup:`-5` mm, **entirely from a glass substitution**;
       agrees to 5 × 10\ :sup:`-7` mm with the exact glass data
   * - Double Gauss
     - Within 5 × 10\ :sup:`-10` mm
     - Within 5 × 10\ :sup:`-7` mm
   * - Hubble
     - Within 5 × 10\ :sup:`-9` mm
     - Within 1 × 10\ :sup:`-8` mm

First-order properties
~~~~~~~~~~~~~~~~~~~~~~

Effective focal length, as an example of the first-order comparison:

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - System
     - Optiland
     - OpticStudio
     - CODE V
   * - Singlet
     - 101.501826
     - 101.501826
     - 101.501826
   * - Cooke triplet
     - 49.999783
     - 49.999783
     - 49.999598 [#cooke]_
   * - Double Gauss
     - 100.003748
     - 100.003748
     - 100.003748
   * - Hubble
     - 57600.080998
     - 57600.079948 [#hubble]_
     - 57600.080998

All other first-order quantities follow the same pattern; the full tables are in
the discussion linked :ref:`below <validation_full>`.

Seidel aberrations
~~~~~~~~~~~~~~~~~~

Magnitudes agree on every system with both tools. Two conventions to know:

- **OpticStudio reports S1 to S5 and lateral color with the opposite sign** to
  Optiland. This is an overall sign-convention difference, not a disagreement.
  CODE V's signs match Optiland's.
- Use the plain *Seidel Aberration Coefficients* listing in OpticStudio, **not**
  the separate *in Waves* listing, which is a different normalization and
  differs by roughly a factor of 200.

Image quality (Tier 2)
~~~~~~~~~~~~~~~~~~~~~~

Compared with OpticStudio on the Cooke triplet and, in part, Hubble and the
Double Gauss. Spot radii agree within 1 % (largest deviation 0.77 %), RMS
wavefront error within 1 % (largest 0.13 %), and distortion and sagittal field
curvature to the printed digits. Zernike coefficients (Fringe, 37 terms) agree
to about 1 × 10\ :sup:`-5` waves on the Cooke triplet, and to 2.7 × 10\ :sup:`-5`
waves at full field on the Double Gauss, where the residual is stable under
pupil-sampling refinement. CODE V Tier 2 results are still to come.

----

.. _validation_open:

Open items
----------

These are known, tracked, and not hidden. None affects the Tier 1 results above.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Item
     - Status
   * - Hubble tangential field-curvature shift differs by 0.6 % from OpticStudio
       (sagittal agrees to 2 × 10\ :sup:`-8`)
     - Unexplained. Checked against sampling and finite-difference step
   * - Double Gauss polychromatic on-axis MTF: gap grows with spatial frequency
       (about +0.01 at 20 cycles/mm to -0.05 at 50 cycles/mm)
     - Unresolved. Does not shrink with finer sampling. Suspected pupil-extent
       or apodization convention
   * - Cooke triplet FFT MTF: one low-contrast point (H\ :sub:`y` = 1.0, 40 lp/mm,
       tangential)
     - A sampling effect: converges to OpticStudio as Optiland's pupil sampling
       increases
   * - Double Gauss Tier 2 wavefront and MTF not yet compared against any tool
       (Zernike coefficients are compared with OpticStudio, see above)
     - Awaiting contributions
   * - CODE V Tier 2 not yet reported
     - Awaiting contributions

----

.. _validation_scope:

Scope and limits
----------------

**Covered:** sequential ray tracing of spherical, conic and mirror systems;
paraxial properties; Seidel aberrations; real-ray intercepts; spot, wavefront,
distortion, field curvature, Zernike and FFT MTF on the systems noted above.

**Not covered:** aspheric and freeform surfaces, gradient-index media,
diffractive surfaces, tilted and decentered systems, polarization and
coatings, non-sequential tracing, tolerancing and optimization. Absence from
this page means *not yet compared*, not *known to differ*.

**Single reporter per tool.** Each tool and version was run by one person. The
numbers are most useful when someone else with the same tool reproduces them.
Anyone with OpticStudio, CODE V or another tool is invited to contribute.

.. _validation_glass:

**Glass data.** Optiland resolves glasses from
`refractiveindex.info <https://refractiveindex.info>`_. The Cooke triplet uses
Hikari SK16, which CODE V's catalog does not carry. The closest CODE V glass
shifts EFL by 4 parts per million, and every Cooke residual in CODE V follows
from that. With a private glass carrying Optiland's exact indices, CODE V
reproduces Optiland's EFL, back focal point, color and rays.

----

.. _validation_full:

Reproduce it, or add your own
-----------------------------

- **Full tables, prescriptions and tool settings:**
  `GitHub issue #710 <https://github.com/optiland/optiland/issues/710>`_.
  It also has ready-to-open ``.zmx`` files, and the template for reporting your
  own comparison.
- **Optiland side:** the four systems ship with Optiland.

  .. code-block:: python

     from optiland.samples.simple import SingletStopSurf2
     from optiland.samples.objectives import CookeTriplet, DoubleGauss
     from optiland.samples.telescopes import HubbleTelescope

- **Regression test:** the OpticStudio Hubble reference values are encoded in
  ``tests/test_cross_tool_validation.py`` and run with the test suite, so a
  change that breaks agreement is caught.

.. rubric:: Notes

.. [#cooke] CODE V has no Hikari SK16 and the closest catalog glass was used;
   with Optiland's exact SK16 indices CODE V gives 49.999783.
.. [#hubble] This value matches Optiland run on a ``.zmx`` written with
   9-significant-digit curvatures (the writer before #712), and CODE V
   reproduces it from that same 9-digit lens. It is expected to match
   57600.080998 when re-measured on the current file; that re-measurement is
   pending.
