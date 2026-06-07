Migration Guide: Deprecated Optimizers → minimize()
=====================================================

.. deprecated::
   The legacy optimizer classes listed below emit ``DeprecationWarning`` on
   construction and will be removed in v0.7.0.  Migrate using the table below.

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Old (deprecated)
     - New
   * - ``OptimizerGeneric(problem).optimize()``
     - ``minimize(problem, "l-bfgs-b")``
   * - ``LeastSquares(problem).optimize()``
     - ``minimize(problem, "least_squares")``
   * - ``DualAnnealing(problem).optimize()``
     - ``minimize(problem, "dual_annealing")``
   * - ``DifferentialEvolution(problem).optimize()``
     - ``minimize(problem, "differential_evolution")``
   * - ``BasinHopping(problem).optimize()``
     - ``minimize(problem, "basin_hopping")``
   * - ``SHGO(problem).optimize()``
     - ``minimize(problem, "shgo")``
   * - ``TorchAdamOptimizer(problem).optimize()``
     - ``minimize(problem, "adam")``
   * - ``TorchSGDOptimizer(problem).optimize()``
     - ``minimize(problem, "sgd")``

Code Examples
-------------

**Before:**

.. code-block:: python

   from optiland.optimization import OptimizerGeneric
   opt = OptimizerGeneric(problem)
   opt.optimize()

**After:**

.. code-block:: python

   from optiland.optimization import minimize
   result = minimize(problem, "l-bfgs-b")
   print(result.value, result.success)

Result Object Changes
---------------------

The old scipy optimizers returned a raw ``scipy.optimize.OptimizeResult``.
``minimize()`` always returns an :class:`~optiland.optimization.state.OptimizationResult`
with richer fields:

* ``.value``: final scalar merit value
* ``.x``: final parameter vector (copy)
* ``.success``: bool
* ``.method``: the resolved method string
* ``.status``: why the run ended (or SciPy-compatible ``.message``)
* ``.improvement_pct``: percent improvement over the starting merit
* ``.wall_time_s``: elapsed wall-clock seconds
* ``.history``: per-iteration merit history (if :class:`~optiland.optimization.observers.history.HistoryObserver` was attached)
* ``.fun``, ``.message``: SciPy-compatible duck-typing aliases
* ``.multipliers``, ``.active_set``, ``.constraint_report``, ``.max_constraint_violation``
  -- constraint diagnostics (non-None only when hard constraints were declared)
* ``.grad_norm``, ``.lambda_final``, ``.cond_estimate`` -- solver diagnostics
* ``.resolved_from`` -- ``"auto"`` when the method was chosen by ``"auto"`` routing

Mutation Contract
-----------------

``minimize()`` mutates the optic **in place**. On return, ``problem``'s optic
holds the optimized design. To preserve the starting state:

.. code-block:: python

   import copy
   starting = copy.deepcopy(optic)
   result = minimize(problem, "dls")
   # optic now holds the result; starting is untouched

Residual-Weighting Behavior Change
------------------------------------

The reworked optimization subsystem changes residual-vector semantics for
multi-field and multi-wavelength problems. The ``residual_vector()`` /
``weighted_residuals()`` path now scales each residual by
``sqrt(eff_weight) * delta`` so that ``sum(weighted_residuals**2) == sum_squared()``.

**Who is affected:** users who compared raw residual *vectors* across versions
(e.g. recorded per-residual values for post-processing or custom stopping logic).
The scalar merit value ``sum_squared()`` is the same; only the residual vector
components change for multi-field/multi-wavelength problems.

**What to do:** if you were relying on the old unweighted residual vector in
custom stopping criteria or post-processing, verify your logic against the
updated values after upgrading to the reworked subsystem.

Standalone Optimizers Are Not Migrated
---------------------------------------

``GlassExpert``, ``OrthogonalDescent``, and ``ParticleSwarm`` are **not**
deprecated and are **not** accessible via ``minimize()`` by design.
Continue using them directly:

.. code-block:: python

   from optiland.optimization import GlassExpert
   expert = GlassExpert(problem, glasses=[...])
   expert.run()

See Also
--------

* :doc:`method_selection` -- choosing the right method for your problem
* :doc:`/api/api_optimization` -- full API reference
