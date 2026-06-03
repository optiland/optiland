Optimization Framework Architecture
=====================================

This document describes the internal architecture of Optiland's optimization
subsystem, the six protocol extension points, worked extension recipes, and
advanced topics including batched ray evaluation and the GlassExpert algorithm.

.. contents::
   :local:
   :depth: 2


Overview and Mental Model
--------------------------

Optiland's optimization subsystem is organized around a two-driver model that
cleanly separates *who controls the iteration loop* from *how each step is
computed*:

.. code-block:: text

   minimize()
       │
       ├── SteppedDriver          (native LM, Gauss-Newton, Torch)
       │       ├── Evaluator      reads/writes parameter vector; computes J, grad
       │       ├── Optimizer      computes the raw parameter update Δx
       │       ├── StepController applies damping, line search, or identity scaling
       │       ├── ConstraintStrategy  projects Δx to satisfy constraints
       │       ├── StoppingCriterion   decides when to halt
       │       └── Observer[]     hook callbacks at each iteration
       │
       └── ManagedDriver          (all SciPy methods)
               ├── Evaluator      same interface as above
               ├── ConstraintStrategy  translated to SciPy constraint dicts
               ├── StoppingCriterion   translated to SciPy callback
               └── Observer[]     wrapped in SciPy callback

**SteppedDriver** owns the loop.  It calls ``optimizer.step()``, then
``controller.apply()``, then ``strategy.project()``, checks
``criterion.should_stop()``, and fires ``observer.on_step()``.  This is used
for ``"dls"``, ``"lm"``, ``"gauss_newton"``, ``"adam"``, and ``"sgd"``.

**ManagedDriver** delegates the loop to ``scipy.optimize``.  It wraps the
evaluator callbacks into the SciPy interface.  This is used for ``"l-bfgs-b"``,
``"bfgs"``, ``"slsqp"``, ``"trust-constr"``, ``"least_squares"``,
``"differential_evolution"``, ``"dual_annealing"``, ``"shgo"``, and
``"basin_hopping"``.

:func:`optiland.optimization.minimize` selects the correct driver, wires up
default components for any not supplied by the caller, and returns an
:class:`~optiland.optimization.state.OptimizationResult`.

.. figure:: ../../_static/cooke_triplet_lens_optimization_evolution.gif
   :width: 60%
   :align: center

.. figure:: ../../_static/cooke_triplet_merit_function_evolution.gif
   :width: 60%
   :align: center


The Six Extension Protocols
-----------------------------

Each of the following is a formal Python protocol (or abstract base class).
Pass instances to :func:`~optiland.optimization.minimize` via the corresponding
keyword arguments, or wire them directly into the driver when building a custom
pipeline.

1. Evaluator
~~~~~~~~~~~~~

The ``Evaluator`` is the bridge between the optimizer's numerical world and
the ``OptimizationProblem``.  It reads and writes the parameter vector,
computes the merit value, gradient, and Jacobian.

**Capability flags** (``EvalCapability`` enum):

* ``VALUE`` — can compute a scalar merit value
* ``GRADIENT`` — can compute ``∇f`` (requires autograd or finite differences)
* ``JACOBIAN`` — can compute the residual Jacobian ``J`` (for LM/GN)

**Key methods:**

.. code-block:: python

   evaluator.read_x()        # → np.ndarray current parameter vector
   evaluator.write_x(x)      # push new x into the optic's attributes
   evaluator.value()         # → float scalar merit
   evaluator.gradient(x)     # → np.ndarray ∇f
   evaluator.jacobian(x)     # → np.ndarray J (m × n)

**Built-in evaluators:**

* ``FiniteDifferenceEvaluator`` — NumPy backend; computes ``∇f`` and ``J``
  via forward finite differences.  Default for all SciPy and native NumPy
  methods.
* ``AutogradEvaluator`` — PyTorch backend; uses ``torch.autograd`` to compute
  exact gradients.  Default when the torch backend is active.

**Experimental** ``jacobian_mode`` kwarg on ``minimize()``:

* ``"stateful"`` (default) — evaluator uses the problem's current state;
  compatible with all backends.
* ``"functional"`` — wraps each evaluation as a pure function call; useful
  for functional-style transforms but carries higher overhead.
* ``"compiled"`` — traces the evaluation through ``torch.compile``; reduces
  Python overhead on repeated calls.  Only meaningful on PyTorch backend; not
  in the all-backend guarantee.

2. Optimizer (step computation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``Optimizer`` receives the current parameter vector and evaluator, and
returns a raw (unconstrained, undamped) step ``Δx``.

**Built-in optimizers:**

* :class:`~optiland.optimization.optimizer.native.levenberg.LevenbergMarquardt`
  — Damped least-squares LM.  Calls ``evaluator.jacobian()``.  Accepts a
  :class:`~optiland.optimization.controller.levenberg.LevenbergController`.
* :class:`~optiland.optimization.optimizer.native.gauss_newton.GaussNewton`
  — Undamped Gauss-Newton.  Converges faster near the minimum when the
  residuals are small, but can diverge far from it.
* :class:`~optiland.optimization.optimizer.torch_opt.TorchOptimizer`
  — Wraps a ``torch.optim.Optimizer`` (Adam, SGD, …).  Calls
  ``evaluator.gradient()`` via autograd.

**Protocol (minimal interface):**

.. code-block:: python

   class MyOptimizer:
       def step(self, x: np.ndarray, evaluator) -> np.ndarray:
           """Return the update vector Δx."""
           ...

3. StepController
~~~~~~~~~~~~~~~~~~

A ``StepController`` post-processes the raw step ``Δx`` before it is applied.
This is where damping and line search live.

**Built-in controllers:**

* ``IdentityController`` — passes ``Δx`` through unchanged.  Used with
  GaussNewton by default.
* :class:`~optiland.optimization.controller.levenberg.LevenbergController`
  — Adjusts the LM damping parameter λ after each step.  Increases λ when
  the step worsens the merit, decreases λ when it improves it (trust-region
  style).
* :class:`~optiland.optimization.controller.line_search.LineSearchController`
  — Scales ``Δx`` by ``α ∈ (0, 1]`` chosen by Armijo backtracking.  Useful
  for Gauss-Newton or custom optimizers when steps overshoot.

**Protocol:**

.. code-block:: python

   class MyController:
       def apply(self, x: np.ndarray, dx: np.ndarray, evaluator) -> np.ndarray:
           """Return the controlled (possibly scaled) update vector."""
           ...

4. ConstraintStrategy
~~~~~~~~~~~~~~~~~~~~~~

A ``ConstraintStrategy`` projects or modifies the step to satisfy constraints.
It reports whether it requires per-step application via the
``PER_STEP_CONSTRAINTS`` capability flag.

**Built-in strategies:**

* ``BoxBoundsStrategy`` — clips ``x + Δx`` to ``[min_val, max_val]`` for
  each variable.  Automatically used when any variable has bounds and the
  method is native (SteppedDriver).
* :class:`~optiland.optimization.constraint.null_space.NullSpaceStrategy`
  — Projects ``Δx`` into the null space of the active equality constraint
  Jacobian.  Enforces exact equality constraints at each LM step (CODE-V
  style).  Pass via ``minimize(problem, "dls", constraint_strategy=NullSpaceStrategy(...))``.
* ``ScipyNativeStrategy`` — Translates constraints to SciPy ``constraints``
  dict for use in ManagedDriver methods (``"slsqp"``, ``"trust-constr"``).
* ``CompositeStrategy`` — Chains multiple strategies in sequence.

**Protocol:**

.. code-block:: python

   class MyStrategy:
       def project(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
           """Return the projected update vector."""
           ...

       @property
       def capability(self) -> set:
           return {"PER_STEP_CONSTRAINTS"}   # or empty set

5. StoppingCriterion
~~~~~~~~~~~~~~~~~~~~~

A ``StoppingCriterion`` is polled after each step to decide whether to halt.
Criteria compose with the ``|`` operator.

**Built-in criteria:**

* ``MaxIterCriterion(n)`` — stops after ``n`` steps.
* ``CostTolerance(tol)`` — stops when the relative change in merit falls
  below ``tol`` (``Δmerit / merit_prev < tol``).
* ``GradNormTolerance(tol)`` — stops when ``‖∇f‖₂ < tol``.  Default
  termination for torch methods.

**Composition:**

.. code-block:: python

   from optiland.optimization.stopping import MaxIterCriterion, CostTolerance

   stop = MaxIterCriterion(500) | CostTolerance(1e-6)
   result = minimize(problem, "dls", stop=stop)

**Protocol:**

.. code-block:: python

   class MyStoppingCriterion:
       def should_stop(self, state) -> bool:
           """Return True to halt the iteration."""
           ...

       def __or__(self, other):
           return CompositeCriterion(self, other)

6. Observer
~~~~~~~~~~~~

An ``Observer`` is notified at four hook points during a run:

* ``on_start(state)`` — called once before the first step
* ``on_step(state)`` — called after each step (including failed steps)
* ``on_stop(state)`` — called once after the last step
* ``on_error(exc, state)`` — called if the driver raises an exception

**Built-in observers:**

* :class:`~optiland.optimization.observers.history.HistoryObserver` — appends
  ``(iteration, merit)`` to ``result.history`` after each step.
* ``ConsoleObserver`` — prints per-step merit and parameter summary.
  Activated when ``disp=True`` is passed to ``minimize()``.
* :class:`~optiland.optimization.observers.checkpoint.CheckpointObserver`
  — saves the optic state to disk at configurable intervals.  Useful for long
  runs.
* ``CancelObserver`` — checks a thread-safe :class:`~optiland.optimization.cancel.CancelToken`
  and raises a soft cancellation exception.  See *Experimental features*.

**Protocol:**

.. code-block:: python

   class MyObserver:
       def on_start(self, state) -> None: ...
       def on_step(self, state) -> None: ...
       def on_stop(self, state) -> None: ...
       def on_error(self, exc, state) -> None: ...


Extension Recipes
-----------------

.. _extension_recipes:

The following minimal examples show how to extend each protocol.

Custom Operand
~~~~~~~~~~~~~~

Add a new function to ``optiland/optimization/operand/operand.py`` and
register it in ``METRIC_DICT``:

.. code-block:: python

   # optiland/optimization/operand/my_operands.py
   from __future__ import annotations

   class MyOperand:
       @staticmethod
       def back_focal_distance(optic):
           """Distance from rear vertex to rear focal point."""
           return optic.paraxial.F2() - optic.surfaces.get_thickness(-2)[0]

Then in ``operand.py``:

.. code-block:: python

   from optiland.optimization.operand.my_operands import MyOperand

   METRIC_DICT["back_focal_distance"] = MyOperand.back_focal_distance

Usage:

.. code-block:: python

   problem.add_operand(
       operand_type="back_focal_distance",
       target=0.0,
       weight=1.0,
       input_data={"optic": lens},
   )

Custom Variable
~~~~~~~~~~~~~~~

Subclass :class:`~optiland.optimization.variable.base.VariableBehavior`:

.. code-block:: python

   from __future__ import annotations
   from optiland.optimization.variable.base import VariableBehavior

   class SemiDiameterVariable(VariableBehavior):
       """Optimizes the semi-diameter of a surface."""

       def get_value(self) -> float:
           return self._surfaces[self.surface_number].semi_diameter

       def update_value(self, new_value: float) -> None:
           self._surfaces[self.surface_number].semi_diameter = float(new_value)

       def __str__(self) -> str:
           return f"SemiDiameter, Surface {self.surface_number}"

Register in the ``Variable`` class dispatch table
(``optiland/optimization/variable/variable.py``) so it can be used by string:

.. code-block:: python

   # Inside Variable.__init__ type dispatch dict:
   "semi_diameter": SemiDiameterVariable,

Custom Observer
~~~~~~~~~~~~~~~

.. code-block:: python

   from __future__ import annotations

   class PlottingObserver:
       """Records merit history for live plotting."""

       def __init__(self):
           self.iterations = []
           self.merits = []

       def on_start(self, state) -> None:
           self.iterations.clear()
           self.merits.clear()

       def on_step(self, state) -> None:
           self.iterations.append(state.iteration)
           self.merits.append(state.merit)

       def on_stop(self, state) -> None:
           print(f"Final merit: {state.merit:.6g} after {state.iteration} steps")

       def on_error(self, exc, state) -> None:
           print(f"Run failed at iteration {state.iteration}: {exc}")

   obs = PlottingObserver()
   result = minimize(problem, "dls", observers=[obs])

Custom StoppingCriterion
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from __future__ import annotations

   class AbsoluteMeritTolerance:
       """Stop when merit drops below an absolute threshold."""

       def __init__(self, threshold: float):
           self.threshold = threshold

       def should_stop(self, state) -> bool:
           return state.merit < self.threshold

       def __or__(self, other):
           from optiland.optimization.stopping import CompositeCriterion
           return CompositeCriterion(self, other)

   stop = AbsoluteMeritTolerance(1e-4) | MaxIterCriterion(1000)
   result = minimize(problem, "dls", stop=stop)

Custom StepController
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from __future__ import annotations
   import numpy as np

   class ClampedStepController:
       """Clips any single parameter update to ±max_step."""

       def __init__(self, max_step: float = 0.1):
           self.max_step = max_step

       def apply(self, x: np.ndarray, dx: np.ndarray, evaluator) -> np.ndarray:
           return np.clip(dx, -self.max_step, self.max_step)

   from optiland.optimization.optimizer.native.levenberg import LevenbergMarquardt
   result = minimize(
       problem,
       "dls",
       step_controller=ClampedStepController(max_step=0.05),
   )

Custom ConstraintStrategy
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from __future__ import annotations
   import numpy as np

   class SymmetryConstraintStrategy:
       """Forces surface 2 and surface 4 to have the same radius."""

       def project(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
           dx_out = dx.copy()
           # Assume variables 0 and 2 are radii of surfaces 2 and 4
           mean_update = 0.5 * (dx_out[0] + dx_out[2])
           dx_out[0] = mean_update
           dx_out[2] = mean_update
           return dx_out

       @property
       def capability(self) -> set:
           return {"PER_STEP_CONSTRAINTS"}

   result = minimize(problem, "dls", constraint_strategy=SymmetryConstraintStrategy())


Batched Ray Evaluation
-----------------------

The batching path is implemented in
:class:`optiland.optimization.batched_evaluator.BatchedRayEvaluator` and is
integrated into ``OptimizationProblem`` by default.  You can opt out with
``problem.disable_batching()`` and re-enable with ``problem.enable_batching()``.

**Algorithm:**

1. Group compatible operands that can share the same trace call.
2. Execute the minimum required set of ``trace_generic`` and ``trace`` calls.
3. Extract per-operand values from shared traced arrays while preserving
   backend behavior and autograd.

**Currently batched operand families:**

* Single-ray (``trace_generic``) operands: ``real_x_intercept``,
  ``real_y_intercept``, ``real_z_intercept``, local-frame intercept variants
  (``_lcs``), direction cosines (``real_L``, ``real_M``, ``real_N``),
  ``clearance``, and ``AOI``.
* Distribution (``trace``) operands: ``rms_spot_size`` when trace parameters
  match across operands.

Operands that are not currently batchable are evaluated through the standard
direct path, so mixed merit functions are fully supported.

For PyTorch workflows, this design keeps gradients valid because values are
extracted by tensor indexing from traced data without detaching.  For NumPy
workflows, behavior remains numerically equivalent to the standard per-operand
evaluation path.

.. figure:: ../../_static/cooke_triplet_starting_point.png
   :width: 80%
   :align: center

.. figure:: ../../_static/cooke_triplet_optimized.png
   :width: 80%
   :align: center


Experimental Features
----------------------

jacobian_mode
~~~~~~~~~~~~~

The ``jacobian_mode`` kwarg on :func:`~optiland.optimization.minimize` is
experimental and controls how the evaluator computes the Jacobian:

* ``"stateful"`` (default) — standard forward-difference evaluation; supported
  on both backends.
* ``"functional"`` — wraps each merit evaluation as a pure function call.
  Useful for functional-style transforms but carries overhead from repeated
  state serialization.
* ``"compiled"`` — traces the forward evaluation through ``torch.compile``
  before differentiating.  Can significantly reduce per-step overhead on
  repeated calls, but requires the PyTorch backend and is not part of the
  all-backend compatibility guarantee.

These modes do not affect the public result interface.

Cancel Token and CancelObserver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For long-running jobs (GUI integration, threaded workers), a
:class:`~optiland.optimization.cancel.CancelToken` can be set from another
thread to request soft cancellation:

.. code-block:: python

   from optiland.optimization.cancel import CancelToken
   from optiland.optimization.observers.cancel import CancelObserver
   import threading

   token = CancelToken()
   obs = CancelObserver(token)

   def run_opt():
       result = minimize(problem, "dls", observers=[obs])

   t = threading.Thread(target=run_opt)
   t.start()

   # From another thread or GUI callback:
   token.cancel()
   t.join()

On cancellation, the driver finishes the current step cleanly, fires
``observer.on_stop()``, and returns a result with ``result.success=False``
and ``result.stop_reason="cancelled"``.


GlassExpert: Categorical Glass Optimization
---------------------------------------------

:class:`~optiland.optimization.GlassExpert` is a specialized optimizer for
problems that include categorical material variables.  It is **not** accessible
via :func:`~optiland.optimization.minimize` — by design — because it manages
a hybrid discrete/continuous search.

Architecture
~~~~~~~~~~~~

GlassExpert separates continuous variables (radii, thicknesses) from
categorical glass variables.  It runs a greedy nearest-neighbour search
over the glass catalog, interleaving discrete glass substitutions with
continuous local optimizations.

The algorithm operates in five phases:

1. **Initialization** — set up the problem with both continuous variables and
   categorical ``"material"`` variables.  Each material variable holds a list
   of candidate glasses:

   .. code-block:: python

      from optiland.optimization import GlassExpert
      from optiland import material_utils

      glasses = material_utils.glasses_selection(0.4, 0.7, catalogs=["schott", "ohara"])
      problem.add_variable(lens, "material", surface_number=1, glass_selection=glasses)

      optimizer = GlassExpert(problem)

2. **Global exploration** — for each glass variable, GlassExpert downsamples
   the full catalog using K-Means clustering (controlled by ``pool_size``),
   retaining a diverse representative subset.  Each candidate is temporarily
   substituted into the design, followed by a continuous optimization.

   .. figure:: ../../_static/glass_map_global_exploration_space.png
      :width: 60%
      :align: center

      Glass map (n_d vs. V_d) showing candidates selected for global search.

3. **Local exploration** — after global search, for each glass variable,
   the ``num_neighbours`` nearest materials in (n_d, V_d) space are trialed.

   .. figure:: ../../_static/glass_map_local_exploration_space.png
      :width: 60%
      :align: center

      Glass map showing candidates selected for local search.

4. **Evaluation and refinement** — for each candidate glass, a continuous
   local optimization is run on all continuous variables.  If the new glass
   produces a lower merit, it is kept; otherwise the design reverts.

5. **Final polish** — a final local continuous optimization with the selected
   glass combination.

Running GlassExpert
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   res = optimizer.run(
       num_neighbours=7,    # glasses to try in local exploration phase
       maxiter=100,         # max iterations for each inner local opt
       tol=1e-6,            # convergence tolerance for inner local opt
       verbose=True,        # print progress
       plot_glass_map=False,
   )

The merit function trace during a GlassExpert run typically shows step
discontinuities — these are normal and correspond to the design being restored
to its best known state when a trialed glass worsens the merit:

.. figure:: ../../_static/glass_expert_error_function.png
   :width: 60%
   :align: center

   Merit function (log scale) during a GlassExpert run with 7 neighbours.

Run duration scales with the number of lens elements and ``num_neighbours``.

Key Implementation Notes
~~~~~~~~~~~~~~~~~~~~~~~~~

* **optiland.optimization.glass_expert.GlassExpert** — main class.
* Glasses are identified by name (string), but their (n_d, V_d) properties
  are used for neighborhood searches and K-Means downsampling (via
  ``optiland.materials.get_nd_vd`` and ``get_neighbour_glasses``).
* GlassExpert temporarily separates continuous and categorical variables.
  Inner continuous optimizations only act on the continuous set.

Extending GlassExpert
~~~~~~~~~~~~~~~~~~~~~

Developers can extend GlassExpert by:

* **Customizing the search strategy** — subclass ``GlassExpert`` and override
  ``_global_exploration()`` or ``_local_exploration()`` to use alternative
  discrete search heuristics (e.g., simulated annealing over the glass map).
* **Integrating additional material properties** — override the distance
  metric used in ``_local_exploration()`` to account for cost, thermal
  coefficient, or transmission band.
* **Reducing local optimization calls** — use a surrogate model to predict
  merit for a given glass, reserving full optimization for top candidates.

For a practical example, see
:doc:`/examples/Tutorial_3f_Standalone_and_Global_Optimizers`.


See Also
--------

* :doc:`method_selection` — choosing the right method
* :doc:`operands_variables` — full operand and variable catalog
* :doc:`migration` — migrating from deprecated optimizer classes
* :doc:`/api/api_optimization` — full API reference
