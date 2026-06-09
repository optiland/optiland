Optimization Framework Architecture
=====================================

This document describes the internal architecture of Optiland's optimization
subsystem, the extension protocols, worked extension recipes, advanced knobs,
hard constraints, and specialized features including batched ray evaluation and
the GlassExpert algorithm.

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
       |
       +-- SteppedDriver          (native LM, Gauss-Newton, Torch)
       |       +-- Evaluator      reads/writes parameter vector; computes J, grad
       |       +-- Optimizer      computes the raw parameter update dx
       |       +-- StepController applies damping, line search, or KKT projection
       |       +-- ConstraintStrategy  applies per-step bounds/projection
       |       +-- StoppingCriterion   decides when to halt
       |       +-- Observer[]     hook callbacks at each iteration
       |
       +-- ManagedDriver          (all SciPy methods)
               +-- Evaluator      same interface as above
               +-- ConstraintStrategy  translated to SciPy constraint dicts
               +-- StoppingCriterion   translated to SciPy callback
               +-- Observer[]     wrapped in SciPy callback

**SteppedDriver** owns the loop. It calls ``optimizer.step()``, the
``StepController`` transforms the raw step, ``ConstraintStrategy.apply_to_step()``
projects it, checks ``criterion.should_stop()``, and fires ``observer.on_step()``.
This driver is used for ``"dls"``, ``"lm"``, ``"gauss_newton"``, ``"adam"``,
and ``"sgd"``.

**ManagedDriver** delegates the loop to ``scipy.optimize``. It wraps the
evaluator callbacks into the SciPy interface. This is used for ``"l-bfgs-b"``,
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


The Seven Extension Protocols
------------------------------

Each of the following is a formal Python protocol (or abstract base class).
Pass instances to :func:`~optiland.optimization.minimize` via the corresponding
keyword arguments, or wire them directly into the driver when building a custom
pipeline.

1. Evaluator
~~~~~~~~~~~~~

The ``Evaluator`` is the bridge between the optimizer's numerical world and
the ``OptimizationProblem``. It reads and writes the parameter vector,
computes the merit value, gradient, and Jacobian.

**Capability flags** (``EvalCapability`` enum):

* ``VALUE`` -- can compute a scalar merit value
* ``RESIDUALS`` -- can compute the residual vector
* ``GRADIENT`` -- can compute ``grad_f`` (requires autograd or finite differences)
* ``JACOBIAN`` -- can compute the residual Jacobian ``J`` (for LM/GN)

**Key methods:**

.. code-block:: python

   evaluator.read_x()        # -> np.ndarray current parameter vector
   evaluator.write_x(x)      # push new x into the optic's attributes
   evaluator.value(x)        # -> float scalar merit
   evaluator.gradient(x)     # -> np.ndarray grad_f
   evaluator.jacobian(x)     # -> np.ndarray J (m x n)

**Built-in evaluators:**

* ``FiniteDiffEvaluator`` -- NumPy backend; computes ``grad_f`` and ``J``
  via forward or central finite differences. Default for all SciPy and native
  NumPy methods. Constructor accepts ``rel_step``, ``abs_step``, ``scheme``,
  and ``on_failure``.
* ``AutogradEvaluator`` -- PyTorch backend; uses ``torch.autograd`` to compute
  exact gradients. Default when the torch backend is active.

**Experimental -- Autograd Jacobian modes:**

The torch Jacobian strategy can be configured by passing ``jacobian_mode`` to :func:`~optiland.optimization.minimize` or directly to ``AutogradEvaluator``:

* ``"stateful"`` (default) -- standard reverse-mode autograd; compatible with
  all torch workflows. Best when the number of residuals ``m`` is small relative
  to the number of variables ``n``.
* ``"functional"`` (**Experimental**) -- wraps each evaluation as a pure
  function call via ``torch.func.jacrev``; useful for functional transforms but
  carries state-serialization overhead.
* ``"forward"`` (**Experimental**) -- forward-mode ``torch.func.jacfwd`` over a
  functional closure. Beats reverse-mode when ``m ≫ n`` (many residuals, few
  variables — the common lens design case). Falls back to ``"stateful"`` if not
  compatible.
* ``"compiled"`` (**Experimental**) -- traces through ``torch.compile`` before
  differentiating; can reduce per-step overhead on repeated calls.
* ``"auto"`` (**Experimental**) -- automatically selects ``"forward"`` when
  ``n_vars < m``, otherwise uses the verified ``"stateful"`` path.

2. Optimizer (step computation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``Optimizer`` computes the raw (unconstrained, undamped) step ``dx`` from the
current state.

**Interface (``SteppedOptimizer`` ABC):**

.. code-block:: python

   class MyOptimizer(SteppedOptimizer):
       capabilities: frozenset  # declares EvalCapability requirements
       requires_backend: str    # "numpy", "torch", or "any"

       def initialize(self, evaluator, x0, *, controller, constraints) -> state:
           """Set up internal state; return initial OptimizationState."""
           ...

       def step(self, state) -> state:
           """Compute and apply one step; return updated state."""
           ...

       def converged(self, state) -> bool:
           """Return True if the optimizer's own convergence criterion is met."""
           ...

**Built-in optimizers:**

* :class:`~optiland.optimization.native.least_squares.LevenbergMarquardt`
  -- Damped least-squares LM. Calls ``evaluator.jacobian()``. Accepts a
  :class:`~optiland.optimization.control.levenberg.LevenbergController` or
  :class:`~optiland.optimization.control.kkt.KKTController`.
* :class:`~optiland.optimization.native.least_squares.GaussNewton`
  -- Undamped Gauss-Newton with Armijo line-search fallback. Converges faster
  near the minimum when residuals are small.
* :class:`~optiland.optimization.native.torch_opt.TorchOptimizer`
  -- Wraps ``torch.optim.Adam`` or ``torch.optim.SGD``. Calls
  ``evaluator.gradient()`` via autograd.

3. StepController
~~~~~~~~~~~~~~~~~~

A ``StepController`` post-processes the raw step ``dx`` before it is applied.
This is where damping and line search live.

**Interface:**

.. code-block:: python

   class MyController:
       def reset(self) -> None:
           """Reset internal state (called at initialize)."""
           ...

       def transform(self, step_info: StepInfo) -> StepOutcome:
           """Return a StepOutcome with the controlled dx."""
           ...

**Built-in controllers:**

* ``IdentityController`` -- passes ``dx`` through unchanged. Used with
  GaussNewton by default.
* :class:`~optiland.optimization.control.levenberg.LevenbergController`
  -- Nielsen gain-ratio damping (**Stable**). Adjusts the LM damping
  parameter ``lambda`` using the gain-ratio between predicted and actual
  improvement. Increases ``lambda`` when the step worsens the merit, decreases
  it when improvement is good. This is the default controller for ``"dls"``
  and ``"lm"`` without hard constraints.
* :class:`~optiland.optimization.control.kkt.KKTController`
  -- Active-set KKT controller for hard-constrained problems (**Stable** for
  the constrained LM path). Solves the augmented KKT system at each step,
  manages the working set of active inequality constraints, and falls back to
  SVD-based least-squares on singular systems. Selected automatically when
  ``problem.constraints`` is non-empty.
* :class:`~optiland.optimization.control.line_search.LineSearchController`
  -- Armijo backtracking line search. Scales ``dx`` by ``alpha in (0, 1]``.
  Useful for GaussNewton or custom optimizers when steps overshoot.

4. ConstraintStrategy
~~~~~~~~~~~~~~~~~~~~~~

A ``ConstraintStrategy`` applies per-step modifications to keep the iterate
within soft bounds or project it to satisfy soft constraints. This is
**distinct** from hard constraints (see `Hard Constraints (KKT active-set)`_
below): a ``ConstraintStrategy`` is for bounds and SciPy-native soft
constraints, while hard equality/inequality constraints use ``add_constraint``
and the KKT path.

**Interface:**

.. code-block:: python

   class MyStrategy:
       def prepare(self, evaluator, variables) -> None:
           """One-time setup called before the loop."""
           ...

       def apply_to_step(self, x_proposed, state):
           """Modify x_proposed in-place or return a new feasible x."""
           ...

       def to_scipy(self):
           """Return SciPy constraint dicts for ManagedDriver methods."""
           ...

       def is_feasible(self, x, tol: float = 1e-8) -> bool:
           """Return True if x satisfies the constraint within tol."""
           ...

**Built-in strategies:**

* ``BoxBoundsStrategy`` -- clips ``x + dx`` to ``[min_val, max_val]`` for
  each variable. Automatically used when any variable has bounds and the
  method is a native SteppedDriver method.
* ``ScipyNativeStrategy`` -- translates constraints to SciPy ``constraints``
  dict for ManagedDriver methods (``"slsqp"``, ``"trust-constr"``).
* ``CompositeStrategy`` -- chains multiple strategies in sequence.

5. StoppingCriterion
~~~~~~~~~~~~~~~~~~~~~

A ``StoppingCriterion`` is polled after each step to decide whether to halt.
Criteria compose with the ``|`` (OR) and ``&`` (AND) operators.

**Interface:**

.. code-block:: python

   class MyStoppingCriterion:
       def reset(self) -> None:
           """Reset internal state at the start of a run."""
           ...

       def should_stop(self, state) -> tuple[bool, str | None]:
           """Return (halt, reason_string) where reason is None if not halting."""
           ...

**Built-in criteria (all in** ``optiland.optimization.stopping.criteria`` **):**

* ``MaxIter(n)`` -- stops after ``n`` accepted steps.
* ``MaxEvals(n)`` -- stops after ``n`` value evaluations (including trial steps).
* ``CostTolerance(tol)`` -- stops when the relative change in merit falls
  below ``tol``.
* ``GradNormTolerance(tol)`` -- stops when ``||grad_f||_2 < tol``. Default
  termination for torch methods.
* ``StepStall(tol)`` -- stops when ``||delta_x|| < tol`` (step too small).
* ``RelImprovement(tol)`` -- stops when cumulative improvement from initial
  merit drops below ``tol``.

**Composition:**

.. code-block:: python

   from optiland.optimization.stopping.criteria import MaxIter, CostTolerance

   stop = MaxIter(500) | CostTolerance(1e-6)
   result = minimize(problem, "dls", stop=stop)

6. Observer
~~~~~~~~~~~~

An ``Observer`` is notified at three hook points during a run:

* ``on_start(state)`` -- called once before the first step
* ``on_step(state)`` -- called after each accepted step
* ``on_end(state, result)`` -- called once after the run completes

**Built-in observers:**

* :class:`~optiland.optimization.observers.history.HistoryObserver` -- appends
  per-iteration records to ``result.history`` after each step.
* :class:`~optiland.optimization.observers.logging.ConsoleObserver` -- prints
  per-step merit and parameter summary. Activated when ``disp=True`` is passed
  to ``minimize()``.
* :class:`~optiland.optimization.observers.checkpoint.CheckpointObserver`
  -- saves the optic state to disk at configurable intervals. Useful for long
  runs.
* :class:`~optiland.optimization.observers.cancel.CancelObserver` -- checks a
  thread-safe :class:`~optiland.optimization.observers.cancel.CancelToken`
  and requests soft cancellation. See `Cancel Token and CancelObserver`_.

**Protocol:**

.. code-block:: python

   class MyObserver:
       def on_start(self, state) -> None: ...
       def on_step(self, state) -> None: ...
       def on_end(self, state, result) -> None: ...

7. KKT Controller (Hard Constraints)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``problem.constraints`` is non-empty, :func:`~optiland.optimization.minimize`
replaces the default ``LevenbergController`` with a
:class:`~optiland.optimization.control.kkt.KKTController`. This controller
solves the full KKT system at each LM step, enforcing hard equality and
inequality constraints without trading them off against the merit function.
See `Hard Constraints (KKT active-set)`_ for the full workflow.


.. _hard_constraints:

Hard Constraints (KKT active-set)
-----------------------------------

Optiland supports two ways to add constraints to an optimization problem:

* **Soft constraints** (``add_operand``): the constraint enters the merit
  function as a weighted residual. The optimizer can trade off constraint
  satisfaction against other operands. Use this for targets that are
  preferences, not requirements.
* **Hard constraints** (``add_constraint``): the constraint is enforced at
  every step via the KKT active-set method. The optimizer cannot trade these
  off -- equality constraints are met exactly at convergence, and inequality
  constraints remain feasible throughout. Use this for absolute requirements
  such as exact focal length, minimum edge thickness, or max chief-ray angle.

.. admonition:: Soft vs. hard decision rule

   Use ``add_operand`` when violating the constraint by a small amount is
   acceptable. Use ``add_constraint`` when the constraint must hold to high
   accuracy regardless of the merit value.

Declaring Hard Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``OptimizationProblem.add_constraint`` takes the same spec as ``add_operand``:

.. code-block:: python

   from optiland.optimization import minimize, OptimizationProblem

   problem = OptimizationProblem()
   # ... add variables and operands ...

   # Equality constraint: EFL must equal 50 mm
   problem.add_constraint(
       operand_type="f2",
       target=50.0,
       input_data={"optic": lens},
       weight=1.0,
   )

   # Inequality constraint: total track <= 80 mm
   problem.add_constraint(
       operand_type="total_track",
       max_val=80.0,
       input_data={"optic": lens},
   )

   result = minimize(problem, "dls")

* ``target=v`` declares an **equality** constraint: the operand must equal ``v``
  at convergence.
* ``min_val=lo`` / ``max_val=hi`` declare **inequality** constraints.
* ``weight``, ``scale``, ``tol``, and ``input_data`` work the same as
  ``add_operand``.

Method Requirement
~~~~~~~~~~~~~~~~~~

Hard constraints require ``method="dls"`` or ``method="lm"`` (both are
KKT-capable). Any other method raises
:class:`~optiland.optimization.errors.ConfigurationError` with guidance.
``method="auto"`` automatically selects ``"dls"`` when
``problem.constraints`` is non-empty.

.. code-block:: python

   # "auto" picks "dls" when constraints are present
   result = minimize(problem, "auto")   # -> "dls"

   # Explicit:
   result = minimize(problem, "dls")

Constraint Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~

The returned :class:`~optiland.optimization.state.OptimizationResult` carries
full constraint diagnostics:

.. code-block:: python

   result = minimize(problem, "dls")

   print(result.multipliers)              # KKT multipliers (equality + inequality)
   print(result.active_set)              # indices of active inequality constraints
   print(result.max_constraint_violation) # max |c_i| or max(g_i, 0) at solution
   print(result.constraint_report)       # per-constraint dict: residual/violation/feasible

   # Programmatic feasibility check:
   mgr = problem.constraints
   print(mgr.report())        # tabular per-row summary
   print(mgr.max_violation()) # scalar max violation
   print(mgr.n_equality, mgr.n_inequality)


Advanced Knobs
---------------

These are ``**method_options`` passed to :func:`~optiland.optimization.minimize`
alongside ``method``, ``stop``, ``observers``, and the other named kwargs.

on_failure -- Ray-trace failure policy (**Stable**)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``on_failure`` controls what happens when a ray trace fails or the merit
function returns a non-finite value:

* ``"reject"`` (default) -- the step is rejected; the optimizer backs off and
  tries again. Safe for all methods.
* ``"raise"`` -- raises an exception immediately. Useful for debugging.
* ``"penalty"`` -- returns a large finite penalty value. Required internally
  for SciPy managed methods (SciPy cannot consume NaN); set explicitly when
  you want the stepped path to behave similarly.

.. code-block:: python

   result = minimize(problem, "dls", on_failure="reject")   # default
   result = minimize(problem, "dls", on_failure="penalty")  # accept failed steps with penalty

scheme -- Finite-difference scheme (**Advanced**)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``scheme`` selects the finite-difference scheme used by ``FiniteDiffEvaluator``:

* ``"forward"`` (default) -- one extra function evaluation per parameter.
  Fast; less accurate.
* ``"central"`` -- two extra evaluations per parameter; second-order accurate
  at roughly 2x cost. Prefer when tight convergence is needed and the merit
  is smooth.

.. code-block:: python

   result = minimize(problem, "dls", scheme="central")

rel_step / abs_step (**Advanced**)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Control the finite-difference step size used by ``FiniteDiffEvaluator``:

* ``rel_step`` (default ``1e-5``) -- step as a fraction of the parameter value.
* ``abs_step`` (default ``1e-8``) -- absolute floor for the step.

linear_solver -- Linear Solve Strategy (**Advanced**)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``linear_solver`` selects how the step is solved during native ``dls``/``lm`` runs:

* ``"normal"`` (default) -- normal-equations solve. Fast; forms the normal
  equations ``JᵀJ + λD``. Can lose accuracy for ill-conditioned Jacobians.
* ``"qr"`` -- QR-based solve. Solves the augmented least-squares system
  without forming ``JᵀJ`` (robust for ill-conditioned Jacobians). Applies to both
  the unconstrained ``LevenbergController`` and the hard-constrained ``KKTController``
  (where it uses the Schur-complement range-space method).

.. code-block:: python

   result = minimize(problem, "dls", linear_solver="qr")

Geodesic Acceleration (**Experimental**)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Geodesic acceleration is a second-order correction to the LM step that can
significantly reduce the number of iterations near the solution by correcting
for surface curvature of the residual manifold. It is opt-in and experimental.
Consult the ``LevenbergMarquardt`` and ``KKTController`` source for the current
opt-in mechanism; this feature may change in future releases.

Auto-scaling (**Stable**)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameter scaling is applied automatically before each run. The scaler
normalizes variable magnitudes so that the optimizer operates in a
well-conditioned space regardless of the physical units of the design
parameters (mm radii vs. dimensionless conic constants, for example). This
is on by default and requires no user action.

lr / gamma -- Torch first-order (**Stable**)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``method="adam"`` or ``method="sgd"``:

* ``lr`` (default ``1e-2``) -- learning rate.
* ``gamma`` (default ``0.99``) -- multiplicative LR decay per step.

method_choice -- SciPy least_squares sub-method (**Stable**)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``method="least_squares"``:

* ``method_choice`` (default ``"lm"``) -- passes through to
  ``scipy.optimize.least_squares`` as the ``method`` argument. Options:
  ``"lm"`` (Levenberg-Marquardt), ``"trf"`` (trust-region reflective),
  ``"dogbox"``.


Extension Recipes
------------------

.. _opt_extension_recipes:

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
           self.merits.append(state.value)

       def on_end(self, state, result) -> None:
           print(f"Final merit: {state.value:.6g} after {state.iteration} steps")

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

       def reset(self) -> None:
           pass

       def should_stop(self, state) -> tuple[bool, str | None]:
           if state.value < self.threshold:
               return True, f"abs_merit_tol={self.threshold:.2e}"
           return False, None

   from optiland.optimization.stopping.criteria import MaxIter

   stop = AbsoluteMeritTolerance(1e-4) | MaxIter(1000)
   result = minimize(problem, "dls", stop=stop)

Custom StepController
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from __future__ import annotations
   import numpy as np
   from optiland.optimization.control.base import StepInfo, StepOutcome

   class ClampedStepController:
       """Clips any single parameter update to +/-max_step."""

       def __init__(self, max_step: float = 0.1):
           self.max_step = max_step

       def reset(self) -> None:
           pass

       def transform(self, step_info: StepInfo) -> StepOutcome:
           dx_clamped = np.clip(step_info.direction, -self.max_step, self.max_step)
           return StepOutcome(delta_x=dx_clamped, accepted=True, info={})

   result = minimize(
       problem,
       "dls",
       controller=ClampedStepController(max_step=0.05),
   )

Custom ConstraintStrategy
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from __future__ import annotations
   import numpy as np

   class SymmetryStrategy:
       """Forces surface 2 and surface 4 to move by equal amounts."""

       def prepare(self, evaluator, variables) -> None:
           pass

       def apply_to_step(self, x_proposed, state):
           # Assume variables 0 and 2 are radii of surfaces 2 and 4
           mean_x = 0.5 * (x_proposed[0] + x_proposed[2])
           x_proposed[0] = mean_x
           x_proposed[2] = mean_x
           return x_proposed

       def to_scipy(self):
           return []

       def is_feasible(self, x, tol=1e-8) -> bool:
           return abs(x[0] - x[2]) < tol

   result = minimize(problem, "dls", constraints=SymmetryStrategy())


Batched Ray Evaluation
-----------------------

The batching path is implemented in
:class:`optiland.optimization.batched_evaluator.BatchedRayEvaluator` and is
integrated into ``OptimizationProblem`` by default. You can opt out with
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
extracted by tensor indexing from traced data without detaching. For NumPy
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

Cancel Token and CancelObserver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For long-running jobs (GUI integration, threaded workers), a
:class:`~optiland.optimization.observers.cancel.CancelToken` can be set from
another thread to request soft cancellation:

.. code-block:: python

   from optiland.optimization.observers.cancel import CancelToken, CancelObserver
   import threading

   token = CancelToken()

   def run_opt():
       result = minimize(problem, "dls", cancel_token=token)

   t = threading.Thread(target=run_opt)
   t.start()

   # From another thread or GUI callback:
   token.cancel()
   t.join()

On cancellation, the driver finishes the current step cleanly, fires
``observer.on_end()``, and returns a result with ``result.success=False``
and ``result.status="cancelled"``.


GlassExpert: Categorical Glass Optimization
---------------------------------------------

:class:`~optiland.optimization.GlassExpert` is a specialized optimizer for
problems that include categorical material variables. It is **not** accessible
via :func:`~optiland.optimization.minimize` by design, because it manages
a hybrid discrete/continuous search.

Architecture
~~~~~~~~~~~~

GlassExpert separates continuous variables (radii, thicknesses) from
categorical glass variables. It runs a greedy nearest-neighbour search
over the glass catalog, interleaving discrete glass substitutions with
continuous local optimizations.

The algorithm operates in five phases:

1. **Initialization** -- set up the problem with both continuous variables and
   categorical ``"material"`` variables. Each material variable holds a list
   of candidate glasses:

   .. code-block:: python

      from optiland.optimization import GlassExpert
      from optiland import material_utils

      glasses = material_utils.glasses_selection(0.4, 0.7, catalogs=["schott", "ohara"])
      problem.add_variable(lens, "material", surface_number=1, glass_selection=glasses)

      optimizer = GlassExpert(problem)

2. **Global exploration** -- for each glass variable, GlassExpert downsamples
   the full catalog using K-Means clustering (controlled by ``pool_size``),
   retaining a diverse representative subset. Each candidate is temporarily
   substituted into the design, followed by a continuous optimization.

   .. figure:: ../../_static/glass_map_global_exploration_space.png
      :width: 60%
      :align: center

      Glass map (n_d vs. V_d) showing candidates selected for global search.

3. **Local exploration** -- after global search, for each glass variable,
   the ``num_neighbours`` nearest materials in (n_d, V_d) space are trialed.

   .. figure:: ../../_static/glass_map_local_exploration_space.png
      :width: 60%
      :align: center

      Glass map showing candidates selected for local search.

4. **Evaluation and refinement** -- for each candidate glass, a continuous
   local optimization is run on all continuous variables. If the new glass
   produces a lower merit, it is kept; otherwise the design reverts.

5. **Final polish** -- a final local continuous optimization with the selected
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
discontinuities -- these are normal and correspond to the design being restored
to its best known state when a trialed glass worsens the merit:

.. figure:: ../../_static/glass_expert_error_function.png
   :width: 60%
   :align: center

   Merit function (log scale) during a GlassExpert run with 7 neighbours.

Run duration scales with the number of lens elements and ``num_neighbours``.

Key Implementation Notes
~~~~~~~~~~~~~~~~~~~~~~~~~

* **optiland.optimization.glass_expert.GlassExpert** -- main class.
* Glasses are identified by name (string), but their (n_d, V_d) properties
  are used for neighborhood searches and K-Means downsampling (via
  ``optiland.materials.get_nd_vd`` and ``get_neighbour_glasses``).
* GlassExpert temporarily separates continuous and categorical variables.
  Inner continuous optimizations only act on the continuous set.

Extending GlassExpert
~~~~~~~~~~~~~~~~~~~~~

Developers can extend GlassExpert by:

* **Customizing the search strategy** -- subclass ``GlassExpert`` and override
  ``_global_exploration()`` or ``_local_exploration()`` to use alternative
  discrete search heuristics (e.g., simulated annealing over the glass map).
* **Integrating additional material properties** -- override the distance
  metric used in ``_local_exploration()`` to account for cost, thermal
  coefficient, or transmission band.
* **Reducing local optimization calls** -- use a surrogate model to predict
  merit for a given glass, reserving full optimization for top candidates.

For a practical example, see
:doc:`/examples/Tutorial_3f_Standalone_and_Global_Optimizers`.


See Also
--------

* :doc:`method_selection` -- choosing the right method
* :doc:`operands_variables` -- full operand and variable catalog
* :doc:`migration` -- migrating from deprecated optimizer classes
* :doc:`/api/api_optimization` -- full API reference
