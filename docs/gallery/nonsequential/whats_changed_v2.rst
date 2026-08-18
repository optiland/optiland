.. _nsq_whats_changed_v2:

What Changed in NSQ Schema v2
==============================

NSQ has never been officially released, so there is no v1 compatibility
mode and no auto-migration path: a scene file written with
``nsq_schema_version: 1`` (the pre-revamp beta) is **refused** on load with
an error pointing here. Carrying two physics paths at once inside the
engine would defeat the point of the revamp. If you have a v1 file, rebuild
the scene from its original construction code, or re-run
``sequential_to_nonsequential(optic)`` if it was converted, against the
current API, then re-export.

This page lists every physics-affecting change between v1 and v2, with its
expected magnitude and direction, so a v1-vs-v2 flux discrepancy in your own
before/after comparison is diagnosable rather than mysterious. See
:ref:`nsq_limitations_and_roadmap` for the current v2 capability envelope and
:ref:`nsq_validation_report` for what v2 is checked against in CI.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Change
     - Expected effect
     - PR
   * - PCG32 counter-based RNG (D11)
     - **Every ray-for-ray output changes**, even for scenes whose physics
       is otherwise identical. Aggregate statistics (total flux, detector
       maps) converge to the same answer; individual ray paths and
       per-ray flux values do not match v1 bit-for-bit. This is the
       largest single driver of "my numbers changed but I didn't change my
       scene."
     - PR2
   * - Geometric medium sidedness (D-1)
     - Fixes silently-wrong refraction at index-matched/cemented interfaces
       that the old index-proximity heuristic misclassified. Scenes with
       such interfaces (e.g. cemented doublets) see a **correctness** fix,
       not just a numeric shift -- v1 output at these interfaces should not
       be trusted.
     - PR5
   * - Coating-driven Fresnel R/T (D-2/D-3)
     - Ghost flux on **coated** surfaces drops roughly **16x** (v1 used bare
       Fresnel, ~4% per surface, regardless of any coating; v2 uses the
       coating's actual reflectance, ~0.25% for a typical single-layer AR
       coat). Uncoated surfaces are unaffected.
     - PR7
   * - Required mirror reflectance (D-3)
     - Mirror throughput now drops by ``1 - reflectance`` per bounce.
       ``reflectance`` was previously optional and defaulted to perfect
       (100%) reflection; it is now a required, explicit field. Any scene
       relying on the old implicit-100% default sees **lower** flux after
       every mirror bounce.
     - PR7
   * - Beer-Lambert bulk absorption (D-13)
     - Absorbing glasses (nonzero extinction coefficient :math:`k`) now lose
       flux over path length via :math:`\exp(-4\pi k L/\lambda)`. v1 media
       were perfectly transparent regardless of :math:`k`. Effect scales
       with path length and :math:`k`; visually clear glasses (small
       :math:`k`) see a negligible change, absorptive/tinted glasses see a
       **substantial** drop.
     - PR8
   * - Explicit BSDF reflect/transmit lobes (D-4)
     - Transmissive diffusers (e.g. ground-glass diffusion in transmission)
       are now physically representable; direction and medium-side state no
       longer disagree after a scatter event. Scenes using only reflective
       scattering (the common case) are unaffected.
     - PR9
   * - ``scatter_fraction`` gradient (D-5)
     - Gradient-mode (PyTorch) optimization against ``scatter_fraction`` now
       actually produces a nonzero gradient; it was silently zero in v1
       despite the branch draw depending on it. Forward-mode (NumPy) flux
       values are unaffected -- this is a differentiability fix, not a
       physics fix.
     - PR9/PR16
   * - Detector unification, ``absorb`` flag, attached ``total_flux`` (D-10/D-11/D-14)
     - Same physics, corrected bookkeeping: ``absorb=True`` detectors now
       correctly remove flux from the ledger, and ``total_flux`` is an
       attached (differentiable) value rather than a detached float. A true
       Gaussian splat replaced an approximate one for
       ``IrradianceDetector``, changing the sub-pixel spatial distribution
       of a detector map (not its total flux) by a small amount near sharp
       features.
     - PR10
   * - Sampling policy: importance biasing, Russian roulette, bounded splitting (D-9/D-2)
     - Replaces v1's biased flux-truncation approach to rare paths with an
       unbiased Russian-roulette estimator and true (bounded) path
       splitting. Expected flux is unchanged in aggregate (checked by the
       "estimator unbiasedness" and "splitting agreement" invariants in
       :ref:`nsq_validation_report`); variance and convergence behavior on
       rare high-reflectance/high-order-ghost paths improve.
     - PR11
   * - Vectorized path recording (D-7)
     - Same output format, substantially faster on the NumPy backend. The
       ``record_paths`` field now also accepts an ``int`` to request a
       subset sample of paths rather than only ``bool`` all-or-nothing --
       an additive API change, not a behavior change for existing
       ``bool`` usage.
     - PR12
   * - ``RayDatabaseConfig.max_rays`` now honored
     - v1 accepted this field but silently ignored it, always recording
       every ray. v2 actually caps the recorded ray count at ``max_rays``.
       Scenes that set this field see **fewer** rows in the recorded ray
       database (and lower memory use); scenes that never set it are
       unaffected.
     - PR13
   * - Photometric units layer added (PR14)
     - Additive: ``total_flux_lumens`` is a new optional source field
       converted internally to radiometric watts via CIE V(:math:`\lambda`).
       Scenes specifying flux in watts (the v1-only option) are unaffected.
     - PR14
   * - Coating carry-over in ``sequential_to_nonsequential`` (D-2)
     - Converted scenes with AR-coated sequential surfaces now carry the
       coating into NSQ instead of falling back to bare Fresnel -- the same
       ~16x ghost-flux drop as the direct coating change above, but for
       *converted* scenes specifically. A ``ConversionReport`` is now
       attached to ``scene.conversion_report`` documenting what was and
       was not carried over.
     - PR15

Not a physics change
---------------------

- **Schema bump itself.** Bumping ``nsq_schema_version`` to 2 and refusing
  v1 files is a loader policy, not a physics change -- it exists so a v1
  file is never silently misinterpreted under v2 physics.
- **Diagnostics (PR13) and validation suite (PR16).** New introspection and
  test coverage; no effect on traced flux.
