.. _optimization_guide:

.. meta::
   :description: Choose an Optiland optimizer, run it with minimize(), watch and cancel a run, make it fast, and understand what happens when a ray trace fails.

Optimization Guide
==================

This guide covers how to *run* an optimization: which method to pick, how to
call it, how to follow and stop a run, how to keep it fast, and what to expect
when something goes wrong. How to *build* the problem (variables, operands,
merit functions, constraints) is in :doc:`optimization_merit_constraints` and
in the :doc:`tutorials <learning_guide>` starting at
:doc:`Tutorial 3a <examples/Tutorial_3a_Simple_Optimization>`.

.. note::

   ``minimize()``, the native ``"dls"`` optimizer, observers, hard constraints
   and the default merit are **experimental**: they are public and tested, but
   their details may still change. The legacy optimizer classes are unchanged.

The problem in four lines
-------------------------

Every optimizer works on an :class:`~optiland.optimization.OptimizationProblem`:
variables to change, and operands (or residual blocks) that define the merit
function. The merit is the sum of squared residuals,
``sum(r ** 2)``, with each operand residual ``r = effective_weight * delta``.

.. code-block:: python

   from optiland.optimization import OptimizationProblem, minimize
   from optiland.samples.objectives import CookeTriplet

   lens = CookeTriplet()
   problem = OptimizationProblem()
   problem.add_operand("f2", target=50.0, weight=10.0, input_data={"optic": lens})
   problem.add_default_merit(lens)                      # see the merit page
   problem.add_variable(lens, "radius", surface_number=1)
   problem.add_variable(lens, "thickness", surface_number=1, min_val=0.5)

   result = minimize(problem)       # the lens now holds the best design found
   problem.info()

``effective_weight`` is the operand weight times the field and wavelength
weights of the optic, so those weights count in every optimizer.

Choosing a method
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 28 42

   * - Situation
     - Use
     - Notes
   * - A normal lens design: continuous variables, residual-style merit
     - ``minimize(problem)`` (``"auto"`` picks ``"dls"``)
     - Native damped least squares. Fastest on small and medium problems.
   * - Hard limits on thicknesses or other operands
     - ``minimize(problem, "dls")`` after ``add_constraint`` or
       ``add_physical_constraints``
     - Only ``"dls"`` enforces hard constraints; every other method raises
       ``ConfigurationError``. Penalty operands (``min_val``/``max_val``) work
       everywhere.
   * - Same, from the legacy API
     - ``LeastSquares(problem).optimize(method_choice="dls")``
     - Also honors variable bounds. ``"lm"``, ``"trf"`` and ``"dogbox"`` still
       mean the SciPy solvers.
   * - Merit is a scalar, not a residual vector, or you want a SciPy solver
     - ``minimize(problem, "l-bfgs-b")`` (also ``"bfgs"``, ``"slsqp"``,
       ``"trust-constr"``, ``"cobyla"``, ``"nelder-mead"``, ``"powell"``, …)
     - Minimize ``sum(r ** 2)``. ``"least_squares"`` calls
       ``scipy.optimize.least_squares`` on the residuals.
   * - Torch backend, gradients through the whole model
     - ``minimize(problem, "adam")`` or ``"sgd"``; or
       :doc:`Tutorial 7b <examples/Tutorial_7b_Differentiable_Lens_Optimization>`
     - Options ``lr`` and ``gamma``; ``maxiter`` is the step count.
   * - Stuck in a local minimum
     - ``DualAnnealing``, ``DifferentialEvolution``, ``BasinHopping``,
       ``SHGO``, ``ParticleSwarm``
     - Legacy classes only. See :doc:`gallery/optimization/global`.
   * - Pick glasses from a catalog
     - ``GlassExpert``
     - See :doc:`Tutorial 3e <examples/Tutorial_3e_Glass_Expert_Categorical_Optimization>`.

Rule of thumb: start with ``minimize(problem)``. If your variables include
categorical ones (glass names), ``"auto"`` falls back to ``"l-bfgs-b"``; use
``GlassExpert`` for those.

minimize()
----------

.. code-block:: python

   result = minimize(
       problem,
       method="auto",       # "dls", a SciPy name, "least_squares", "adam", "sgd"
       maxiter=None,        # residual evaluations for "dls" and "least_squares"
       tol=1e-3,            # ftol for "dls"; SciPy tol for the local solvers
       stop=None,           # extra stopping criterion
       observers=None,      # list of observers
       cancel_token=None,   # cancel the run from another thread
       disp=False,          # print progress
       **options,           # per-method options (below)
   )

Things worth knowing:

* **The optic is changed in place.** On return it holds the best accepted
  design, ``result.x``. Copy the optic first (``copy.deepcopy(lens)``) if you
  need the starting point back.
* **``maxiter`` counts residual evaluations, Jacobian columns included.** One
  Jacobian of a 12-variable problem costs 12 evaluations, so the default
  budgets are not "iterations".
* **Unknown options raise.** ``minimize(problem, "dls", sheme="central")``
  raises ``ConfigurationError`` listing the valid names, instead of being
  silently ignored.
* The result is a SciPy ``OptimizeResult`` with extra fields: ``stop_reason``,
  ``n_evals``, ``history`` (with a ``HistoryObserver``), ``method``, and for
  ``"dls"`` also ``multipliers``, ``active_set``, ``max_constraint_violation``
  and ``constraint_report``. ``result.fun`` is the final merit.

Options of ``"dls"``:

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Option
     - Values
     - Meaning
   * - ``scheme``
     - ``"forward"`` (default), ``"central"``
     - Finite-difference scheme. ``"central"`` costs twice as many evaluations
       per Jacobian; use it for tight convergence on smooth merits.
   * - ``rel_step``
     - float
     - Relative finite-difference step.
   * - ``aim_policy``
     - ``"exact"`` (default), ``"frozen"``
     - Whether ray aiming is recomputed for every Jacobian column or reused
       from the baseline design. ``"frozen"`` is faster for systems with
       expensive aiming (wide field, telecentric); try ``"exact"`` first.
   * - ``jacobian_update``
     - ``"exact"`` (default), ``"auto"``, ``"broyden"``
     - Recompute the Jacobian after every step, or reuse it with rank-one
       updates after good steps. Cuts evaluations on large problems, at some
       risk to robustness.
   * - ``on_failure``
     - ``"reject"`` (default), ``"raise"``
     - See :ref:`optimization_failures`.

``minimize()`` and ``LeastSquares``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The classes you already use keep their signatures, and no deprecation
warnings are issued. ``LeastSquares(problem).optimize(method_choice="dls")`` runs
the same native core as ``minimize(problem, "dls")``; ``callback``, ``plot`` and
``disp`` work as before. Use ``minimize()`` when you want observers, a stopping
criterion, a cancel token or hard constraints on a typical run.

For ``method_choice="lm"``, ``"trf"`` and ``"dogbox"``, SciPy receives its
Jacobians from the same finite-difference evaluator, with the steps of SciPy's
own ``"2-point"`` scheme, so a run takes the same path as before while the
columns reuse the baseline traces. Pass ``aim_policy="frozen"`` to
``optimize()`` to reuse the baseline ray aiming in every column; with
iterative ray aiming this makes each Jacobian several times cheaper.

Stopping, observers and cancellation
------------------------------------

*Stopping criteria* decide when a run ends. ``"dls"`` always stops on ``tol``,
on a vanishing step or gradient, on a collapsed trust region and on
``maxiter``; ``stop=`` adds your own tests, checked after every accepted step.
Combine criteria with ``|`` (any) and ``&`` (all, each latched once true):

.. code-block:: python

   from optiland.optimization import CostTolerance, MaxEvals, StepStall

   stop = MaxEvals(2000) | (CostTolerance(1e-6) & StepStall(1e-6))
   result = minimize(problem, "dls", stop=stop)
   print(result.stop_reason)

Also available: ``MaxIter`` (accepted steps) and ``GradNormTolerance``.

*Observers* watch a run without changing it. They receive ``on_start``,
``on_step`` (after each accepted step) and ``on_end``, and the current
:class:`~optiland.optimization.state.OptimizationState` (``x``, ``value``,
``iteration``, ``n_evals``, …), which they must treat as read-only.

.. code-block:: python

   from optiland.optimization import ConsoleObserver, HistoryObserver, CheckpointObserver

   history = HistoryObserver()
   result = minimize(
       problem,
       "dls",
       observers=[history, ConsoleObserver(log_every=5), CheckpointObserver("runs", every=10)],
   )
   merit = [r["value"] for r in history.records]    # also result.history

``disp=True`` is shorthand for adding a ``ConsoleObserver``. Any object with
the three methods is an observer; see
:class:`~optiland.optimization.observers.base.Observer`. SciPy-based methods
call observers at the start and end of the run, and per step where SciPy offers
a callback (the local solvers do, ``"least_squares"`` does not).

*Cancellation* is cooperative. Set a :class:`~optiland.optimization.CancelToken`
from any thread (a GUI button, a timer); the run ends cleanly after the current
step, the optic holds the best design so far, and ``result.stop_reason`` ends with
``"cancelled by user"``.

.. code-block:: python

   import threading
   from optiland.optimization import CancelToken

   token = CancelToken()
   threading.Timer(30.0, token.cancel).start()      # give up after 30 s
   result = minimize(problem, "dls", cancel_token=token)

Making it fast
--------------

Most of the cost of an optimization is ray tracing, and in a finite-difference
Jacobian, tracing the same rays again and again. Optiland removes much of that
for you; these are the levers that remain.

* **Use ``"dls"`` on the NumPy backend.** It reuses the ray traces of the
  baseline design for every Jacobian column (only surfaces at or after the
  changed one are re-traced), and per-evaluation work such as material
  indices, paraxial data and ray aiming is computed once per design. On the
  benchmark set of ``python -m benchmarks.optimization.run`` it converges two
  to ten times faster than SciPy ``"lm"`` on a Cooke triplet, a Double Gauss
  and a ten-variable scaling probe, and sometimes ends at a slightly higher
  merit, so compare the final ``problem.sum_squared()`` for your own design.
* **Keep batching on** (the default). Ray operands that share a field,
  wavelength and pupil are traced together. Call
  ``problem.disable_batching()`` only when debugging.
* **Trace fewer rays.** Cost scales with ray count. The
  :ref:`default merit <default_merit>` needs ``rings * arms`` rays per field and
  wavelength (18 for the default 3 × 6); ``rms_spot_size`` operands with
  ``num_rays=100`` trace 30 301. Start coarse, finish fine.
* **Use bounds instead of penalty operands** where you can; they cost nothing
  extra.
* **Reduce Jacobian cost on large problems** with
  ``jacobian_update="auto"``, or ``aim_policy="frozen"`` when aiming dominates.
* **Do not tighten ``tol`` blindly.** The default ``1e-3`` is a relative merit
  change; much smaller values mostly add evaluations.
* **Torch backend**: correctness is guaranteed, but the speed work targets
  NumPy. Prefer ``"adam"`` / ``"sgd"`` there, with gradients from autograd.

To see where time goes, print ``result.n_evals`` and time ``problem.sum_squared()``
once: runtime is roughly ``n_evals`` times that.

.. _optimization_failures:

When a ray trace fails
----------------------

During an optimization a trial design can be untraceable: a ray misses a
surface, hits total internal reflection, or a thickness goes negative. Optiland
treats this the same way everywhere:

* A failed trace gives ``NaN`` in the affected residual rows. It never raises
  out of an evaluation, and one bad operand does not poison the others.
* **Native ``"dls"``**: a trial with any non-finite residual is *rejected*;
  the optimizer shrinks its step and tries again from the last good design.
  The set of rays stays fixed during an iteration, and rays that keep failing
  are dropped from the merit between iterations (their rows read zero) and
  restored at the end of the run.
* **SciPy-based methods** cannot digest ``NaN``, so failed rows are replaced by
  a penalty that scales with the last finite residual norm, which keeps their
  finite-difference Jacobians usable. The best finite design seen is
  restored if the final point is non-finite.
* **The starting design must be traceable.** A non-finite initial residual
  raises ``ConfigurationError`` naming the failing operands or blocks.
* ``on_failure="raise"`` (``"dls"``, SciPy local solvers) raises
  ``OptimizationFailure`` at the first failure instead. Use it to find out
  *which* operand fails.

If a run keeps rejecting steps and ends with ``success=False``: loosen
variable bounds that allow impossible geometry, check the starting design with
``lens.info()`` and ``lens.draw()``, add thickness limits
(:ref:`physical constraints <physical_constraints>`), and read
``result.stop_reason``.

Migration notes
---------------

**Weights are no longer dropped in ``LeastSquares``.** The legacy
``LeastSquares`` optimizer used to build residuals from
``weight * delta`` and so ignored the field and wavelength weights, while
every other optimizer and ``problem.sum_squared()`` used
``effective_weight * delta``. They now agree. If you set non-uniform field or
wavelength weights, a ``LeastSquares`` run converges to a different (now
consistent) design than before. Equal weights give the same results.

**SciPy ``LeastSquares`` no longer sees ``NaN``.** A failed trace used to feed
``NaN`` into SciPy's Jacobian and could end the run at a non-finite point. It
now uses the penalty described above and restores the best finite design.

**The default of ``method_choice`` is still ``"lm"``.** The native ``"dls"`` is
opt-in: pass ``method_choice="dls"`` (or use ``minimize``). The default will
change to ``"dls"`` only after it is at least as fast and as good as SciPy on
the whole benchmark set; until then, nothing changes for existing scripts.
Pin ``method_choice="lm"`` if you want to be unaffected when it does.

**``maxiter`` for ``"dls"`` counts residual evaluations**, Jacobian columns
included, the same as ``maxiter`` of ``"lm"`` and ``"trf"``.

**Hard constraints are new and ``"dls"``-only.** Existing penalty operands
(``min_val`` / ``max_val``) behave as before on every optimizer.
