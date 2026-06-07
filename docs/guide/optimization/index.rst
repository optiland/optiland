Optimization Guide
==================

Optiland provides a unified optimization framework for optical system design.
This section covers the architecture, usage patterns, and extension points.

**The single rule:** does Optiland run the loop, or do you?

* **Optiland runs it** -> :func:`optiland.optimization.minimize`
* **You run it** (custom loss, batched data, optic as a differentiable layer) -> :class:`optiland.ml.OpticalSystemModule`

At a Glance
-----------

Method Cheat Table
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Method string
     - Family
     - Backend
     - Use when
   * - ``"dls"`` / ``"lm"``
     - Native LM (stepped)
     - NumPy or Torch
     - Classical lens, equality targets, hard constraints; CODE V / Zemax style.
   * - ``"gauss_newton"``
     - Native GN (stepped)
     - NumPy or Torch
     - Already near a solution; no damping overhead.
   * - ``"adam"``
     - Torch first-order (stepped)
     - Torch only
     - Differentiable, GPU, single scalar merit; Optiland owns the loop.
   * - ``"sgd"``
     - Torch first-order (stepped)
     - Torch only
     - Same as adam; momentum-free variant.
   * - ``"l-bfgs-b"``
     - SciPy local (managed)
     - NumPy
     - Bounds/inequality-dominated; general-purpose quasi-Newton.
   * - ``"bfgs"`` / ``"cg"`` / ``"powell"`` / ...
     - SciPy local (managed)
     - NumPy
     - Standard unconstrained local search variants.
   * - ``"slsqp"`` / ``"trust-constr"``
     - SciPy local (managed)
     - NumPy
     - Soft inequality constraints passed to SciPy.
   * - ``"least_squares"``
     - SciPy LS (managed)
     - NumPy
     - Residual-based with trust-region bounds; trf/dogbox/lm sub-methods.
   * - ``"differential_evolution"`` / ``"dual_annealing"`` / ``"shgo"`` / ``"basin_hopping"``
     - SciPy global (managed)
     - NumPy
     - Global / derivative-free search; population-based or stochastic.
   * - ``"auto"`` (default)
     - Resolved at runtime
     - Any
     - Picks a sensible default; prints its choice when ``disp=True``.

``"auto"`` resolution: hard constraints present -> ``"dls"``; torch backend -> ``"adam"``; numpy with all equality targets and m >= n -> ``"dls"``; otherwise -> ``"l-bfgs-b"``.

Key kwargs at a Glance
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Kwarg
     - What it does
   * - ``method``
     - Method string (see table above). Default ``"auto"``.
   * - ``constraints``
     - A :class:`~optiland.optimization.constraints.base.ConstraintStrategy` or list. For hard equality/inequality constraints, use ``problem.add_constraint(...)`` instead.
   * - ``bounds``
     - ``True`` (default) to honor variable bounds via ``BoxBoundsStrategy``.
   * - ``stop``
     - A :class:`~optiland.optimization.stopping.criteria.StoppingCriterion` or composed combination.
   * - ``observers``
     - A list of :class:`~optiland.optimization.observers.base.Observer` instances.
   * - ``on_failure``
     - ``"reject"`` (default), ``"raise"``, or ``"penalty"``. Controls what happens on ray-trace failures.
   * - ``tol``
     - Convergence tolerance (semantics depend on method family). Default ``1e-3``.
   * - ``maxiter``
     - Maximum iterations before stopping. Default ``1000``.
   * - ``disp``
     - Print per-step progress. Default ``True``.

Hard Constraints
~~~~~~~~~~~~~~~~

To add hard equality or inequality constraints (enforced via KKT active-set):

.. code-block:: python

   problem.add_constraint(operand_type="f2", target=50.0, input_data={"optic": lens})
   problem.add_constraint(operand_type="total_track", max_val=80.0, input_data={"optic": lens})
   result = minimize(problem, "dls")  # or "auto" -- both select the KKT path

See :ref:`hard_constraints` in the framework guide for the full workflow.

Where to Read Results
~~~~~~~~~~~~~~~~~~~~~

The returned :class:`~optiland.optimization.state.OptimizationResult` exposes:

* ``result.value``, ``result.x``, ``result.success``, ``result.status``
* ``result.improvement_pct``, ``result.wall_time_s``, ``result.method``
* **Constraint diagnostics:** ``result.multipliers``, ``result.active_set``,
  ``result.constraint_report``, ``result.max_constraint_violation``
* **Solver diagnostics:** ``result.grad_norm``, ``result.lambda_final``,
  ``result.cond_estimate``
* **History:** ``result.history`` (attach a ``HistoryObserver`` to populate)

See Tutorial 3d for a hands-on walkthrough of all diagnostic fields.

.. toctree::
   :maxdepth: 2

   method_selection
   framework
   operands_variables
   migration

See Also
--------

* :doc:`/api/api_optimization` -- full API reference
* :doc:`/learning_guide` -- step-by-step tutorials (section 3)
