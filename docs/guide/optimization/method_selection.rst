Optimization Method Selection Guide
====================================

.. note::
   The canonical entry point for optimization in Optiland is :func:`optiland.optimization.minimize`.

The Optiland Rule
-----------------

**Does Optiland run the loop, or do you?**

* **Optiland runs the loop** -> use :func:`~optiland.optimization.minimize`.
* **You own the loop** (custom loss, batched data, optic as a layer in a bigger network) -> use :class:`~optiland.ml.OpticalSystemModule`.

Choosing the Right Entry Point
------------------------------

Optiland provides multiple entry points tailored to different design and modeling tasks:

* **minimize() Facade**: The primary, unified interface for general lens optimization. It supports classical damped least squares (DLS/LM), gradient-free solvers, and PyTorch-based first-order optimizers (``Adam``, ``SGD``).
* **OpticalSystemModule**: A PyTorch ``nn.Module`` wrapper that integrates Optiland optics directly into deep learning training loops. Ideal for neural-optical co-design, custom loss functions, and processing batched training data where you manage the optimization loop.
* **Standalone Optimizers**: Specialized tools with unique interfaces or discrete search routines that operate outside the standard gradient facade (e.g., :class:`~optiland.optimization.GlassExpert` for discrete glass catalogs, :class:`~optiland.optimization.OrthogonalDescent` for null-space design exploration, and :class:`~optiland.optimization.optimizer.custom.particle_swarm.ParticleSwarm` for derivative-free global swarm search).

Everything below applies to :func:`~optiland.optimization.minimize`.

Method Selection Decision Table
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Situation
     - Recommended Method
     - Why
   * - Classical lens, all equality targets, ``m >= n``, ``numpy``
     - ``"dls"`` (native LM)
     - CODE V / Zemax-style damped least squares; robust, fast on dense small problems.
   * - Same, need exact equality constraints (hard constraints)
     - ``"dls"`` + ``problem.add_constraint``
     - KKT active-set enforcement; use ``add_constraint`` with ``target=`` for exact equalities.
   * - Close to a solution (no damping needed)
     - ``"gauss_newton"``
     - No ``\lambda``-damping overhead; converges faster when already near the minimum.
   * - Bounds / inequality-dominated, ``numpy``
     - ``"l-bfgs-b"`` / ``"slsqp"``
     - Box/constraint-aware quasi-Newton; managed by SciPy.
   * - Residual-based, trust-region bounds, ``numpy``
     - ``"least_squares"``
     - ``scipy.optimize.least_squares`` (TRF / dogbox); supports bounds directly.
   * - Differentiable, GPU, single merit, Optiland owns the loop
     - ``"adam"`` (torch)
     - On-device autograd first-order; Optiland owns the loop.
   * - Optic as a layer in a bigger net / custom loss / batched data
     - :class:`~optiland.ml.OpticalSystemModule`
     - You own the loop; custom loss, batched data, optic composed in a larger ``nn.Module``.
   * - Global / derivative-free search
     - ``"differential_evolution"`` / ``"dual_annealing"`` / ...
     - Out of the gradient facade by design; population-based or stochastic.
   * - Discrete glass search
     - :class:`~optiland.optimization.GlassExpert`
     - Enumerates the glass catalog; not gradient-based.
   * - Just exploring / teaching
     - ``"auto"`` (default)
     - Picks a sensible default, prints what it chose when ``disp=True``.

``tol`` Semantics by Method Family
----------------------------------

The ``tol`` argument to :func:`~optiland.optimization.minimize` controls different criteria depending on the method family:

* **Native LM / DLS / GaussNewton** and **SciPy managed (local, LS)**:
  ``tol`` is passed to :class:`~optiland.optimization.stopping.criteria.CostTolerance` (stops when the relative change in merit falls below ``tol``).
* **Torch first-order (adam, sgd)**:
  ``tol`` is passed to :class:`~optiland.optimization.stopping.criteria.GradNormTolerance` (stops when the gradient L2 norm falls below ``tol``).

The active criterion and the threshold that fired are always recorded in ``result.status`` (e.g., ``"cost_tol=1.00e-03 (rel_change=8.21e-04)"``), so you can inspect which criterion terminated the run without parsing console output.

Hard Constraints and Method Routing
-------------------------------------

When ``problem.constraints`` is non-empty (hard constraints declared via
``problem.add_constraint``), ``method="auto"`` automatically selects ``"dls"``
regardless of backend. Hard constraints require ``method="dls"`` or
``method="lm"`` -- any other method raises
:class:`~optiland.optimization.errors.ConfigurationError`.

.. code-block:: python

   problem.add_constraint(operand_type="f2", target=50.0, input_data={"optic": lens})
   result = minimize(problem)        # "auto" -> "dls" (KKT path)
   result = minimize(problem, "lm") # also valid

For soft operand-based targets (no strict enforcement), continue using
``problem.add_operand`` with ``target=``. See :ref:`hard_constraints` for the
full workflow and diagnostics.

``on_failure`` and Numerical Robustness
----------------------------------------

The ``on_failure`` kwarg controls what happens when a ray trace fails or the
merit function returns a non-finite value:

* ``"reject"`` (default, **Stable**) -- the step is rejected; the optimizer
  backs off. Safest for all methods.
* ``"raise"`` -- raises an exception immediately. Useful for debugging.
* ``"penalty"`` -- returns a large finite penalty. SciPy managed methods
  always use this internally (SciPy cannot consume NaN).

For ``"central"`` finite differences (``scheme="central"``), accuracy improves
at roughly 2x the cost of ``"forward"`` (the default). Prefer ``"central"``
when tight convergence is needed and the merit landscape is smooth.

``auto`` Resolution
-------------------

When ``method="auto"`` (the default):

* **Hard constraints present** -> ``"dls"`` (KKT path).
* Under the **torch** backend -> ``"adam"``.
* Under **numpy**, all operands have targets, and ``m >= n`` -> ``"dls"``.
* Under **numpy**, otherwise -> ``"l-bfgs-b"``.

The resolved method is printed at the start of the run when ``disp=True`` (e.g., ``auto -> dls``). It is also available as ``result.method`` and ``result.resolved_from == "auto"`` for programmatic inspection.

Side Effects / Mutation Contract
--------------------------------

:func:`~optiland.optimization.minimize` **mutates the optic in place**. On return, the optic holds the optimized design. ``result.x`` is a convenience copy of the final parameter vector. To preserve the starting design, snapshot the optic yourself before calling ``minimize()``.

The same side-effect contract applies to :class:`~optiland.ml.OpticalSystemModule`: each ``forward()`` call pushes the current ``nn.Parameter`` values into the optic's surface attributes.

Deprecated Entry Points
-----------------------

The following classes are **deprecated** (emit ``DeprecationWarning`` on construction) and will be removed in **v0.7.0**:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Deprecated Class
     - Replacement
   * - ``OptimizerGeneric``
     - ``minimize(problem, method='l-bfgs-b')``
   * - ``LeastSquares``
     - ``minimize(problem, method='least_squares')``
   * - ``DualAnnealing``
     - ``minimize(problem, method='dual_annealing')``
   * - ``DifferentialEvolution``
     - ``minimize(problem, method='differential_evolution')``
   * - ``BasinHopping``
     - ``minimize(problem, method='basin_hopping')``
   * - ``SHGO``
     - ``minimize(problem, method='shgo')``
   * - ``TorchAdamOptimizer``
     - ``minimize(problem, method='adam')``
   * - ``TorchSGDOptimizer``
     - ``minimize(problem, method='sgd')``

Non-Deprecated Standalone Optimizers
------------------------------------

These are **not** accessible via :func:`~optiland.optimization.minimize` by design. Use them directly:

* :class:`~optiland.optimization.GlassExpert`: discrete glass search.
* :class:`~optiland.optimization.OrthogonalDescent`: descent in the operand null-space.
* :class:`~optiland.optimization.optimizer.custom.particle_swarm.ParticleSwarm`: particle-swarm global optimizer.

