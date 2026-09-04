.. _nsq_validation_report:

NSQ Validation Report
======================

This page summarizes the non-sequential (NSQ) engine's validation suite
(``tests/nonsequential/validation/``), run in CI alongside the rest of the
test suite. It exists so a user integrating NSQ into their own QA process
has a single place pointing at exactly what is checked, how, and to what
tolerance -- rather than having to read test source to find out.

Closed-form analytic benchmarks
--------------------------------

Each benchmark traces a scene whose answer is known exactly (or to a
well-defined statistical tolerance) from closed-form physics, independent
of the NSQ engine itself.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Benchmark
     - Asserted quantity
     - Module
   * - Point source -> small flat patch
     - Exact inverse-square law, :math:`E = F / (4\pi d^2)`
     - ``test_benchmark_inverse_square.py``
   * - Small Lambertian disc -> parallel plane
     - :math:`\cos^4\theta` off-axis falloff
     - ``test_benchmark_lambertian_cos4.py``
   * - Uncoated plane-parallel window
     - Total transmittance incl. all internal reflections:
       :math:`T = 2n/(n^2+1)`
     - ``test_benchmark_window_transmittance.py``
   * - Single interface, swept incidence angle
     - Unpolarized Fresnel :math:`R_\text{unpol}(\theta)` vs. the analytic
       curve
     - ``test_benchmark_fresnel_sweep.py``
   * - Total internal reflection
     - Sharp cutoff at :math:`\theta_c = \arcsin(n_2/n_1)`
     - ``test_benchmark_tir_critical_angle.py``
   * - Absorbing slab
     - Beer-Lambert :math:`\exp(-4\pi k L/\lambda)`, swept over :math:`k`
       and :math:`L`
     - ``test_benchmark_absorbing_slab.py``
   * - Thin lens
     - Focal spot sharpest near the paraxial thin-lens prediction
     - ``test_benchmark_thin_lens_focus.py``
   * - AR-coated interface
     - Reflectance/transmittance match ``optiland.coatings`` directly (D-2)
     - ``test_benchmark_ar_coating.py``

**Deliberately deferred, not silently missing:**

- **Prism at minimum deviation.** Requires a two-plane wedge geometry and a
  minimum-deviation angle search; not yet built. The single-interface
  Fresnel and TIR benchmarks above already exercise the same underlying
  refraction code a prism would.
- **Integrating sphere.** Requires a detector patch conformal to a curved
  wall (a flat ``IrradianceDetector`` tangent to a sphere does not
  intersect the interior rays correctly) and enough bounce depth /
  Russian-roulette tuning for near-unity-reflectance convergence. The
  underlying mechanism (deterministic per-bounce reflectance loss via
  ``ReflectiveComponent`` + a Lambertian BSDF) is exercised by
  ``test_nsq_coatings.py`` and the roulette invariant tests, just not
  assembled into the full sphere-multiplier benchmark.

Invariants
----------

Checked as properties that must hold across a family of scenes and
settings, rather than against one closed-form number.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Invariant
     - Statement
     - Module
   * - Energy closure
     - ``in = detected + absorbed + escaped + lost``, to 1e-9 relative at
       the default (Fresnel) sampling policy; to a looser statistical bound
       under importance biasing or aggressive roulette, where the ledger
       only closes in expectation (see the module docstring)
     - ``test_invariant_energy_closure.py``
   * - Batch invariance
     - Bit-identical results for ``batch_size`` in ``{1, 7, 1024, 16384}``
       (D11/PCG32) on the unsplit path. **Known gap:** bounded splitting
       (D2/PR11) does not currently carry this guarantee -- a spawned
       ray's id depends on how many other rays had already split by that
       point, which is itself batch-size dependent. Physics stays unbiased
       (checked statistically) even though the RNG stream is not
       reproduced bit-for-bit.
     - ``test_invariant_batch_independence.py``
   * - Convergence
     - Error shrinks as :math:`N^{-1/2}` over a swept ray count (fitted
       exponent :math:`-0.5 \pm 0.1`)
     - ``test_invariant_convergence_rate.py``
   * - Rigid invariance
     - A global rotation + translation of the entire scene (via a shared
       ``reference_cs``) leaves the detector's own-frame map unchanged
     - ``test_invariant_rigid_transform.py``
   * - Reciprocity
     - Swapping a matched-extent Lambertian source and detector transfers
       the same flux in both directions
     - ``test_invariant_reciprocity.py``
   * - Estimator unbiasedness
     - ``reflect_prob`` in ``{fresnel, 0.25, 0.5, auto}`` converge to the
       same detected flux
     - ``tests/nonsequential/test_nsq_sampling_policy.py``
   * - Splitting agreement
     - ``split_depth`` in ``{0, 1, 2, 3}`` converge to the same answer, with
       variance shrinking as depth increases
     - ``tests/nonsequential/test_nsq_sampling_policy.py``
   * - Sequential agreement
     - NSQ matches Optiland's sequential tracer for singlet/doublet/aspheric
       systems
     - ``tests/nonsequential/test_nsq_sequential_agreement.py``

Gradient validation
--------------------

Finite-difference checks for parameters documented as differentiable.
Beyond the pre-existing coverage (source ``total_flux``, BSDF reflectance,
lens/mirror geometry, detector ``total_flux``), this pass added:

- **``scatter_fraction`` (D-5).** Writing this check caught that D-5 was
  never actually fixed despite being marked done: ``BaseComponent``
  forced ``scatter_fraction`` through a bare ``float()`` at construction
  (detaching any tensor immediately), and neither ``RefractiveComponent``
  nor ``ReflectiveComponent`` applied a compensating attached weight to
  either branch of the scatter/specular split. Both are now fixed with the
  same detached-sample / attached-weight estimator used for the Fresnel
  branch. See ``test_gradient_scatter_fraction.py``.
- **Beer-Lambert ``k`` (D-13).** ``test_gradient_absorption_k.py``.

Existing coverage retained as part of this suite's contract:

- ``test_nsq_geometric_gradients.py``'s assertion that *visibility*
  gradients are zero -- a known, documented v1 limitation (see
  :ref:`nsq_limitations_and_roadmap`), not a regression to silently fix
  here.
- ``test_nsq_geometric_gradients.py``'s ``TestDifferentiableParameterContract``:
  every parameter *not* on the differentiable list raises via
  ``as_detached_param`` rather than silently detaching -- no third
  category of dead variables.

Running the suite
------------------

.. code-block:: bash

   .venv/Scripts/python.exe -m pytest -v tests/nonsequential/validation/

Individual benchmarks are ordinary parametrized pytest tests; failures
report which ray count / angle / material combination diverged and by how
much, the same as any other test in the suite.
