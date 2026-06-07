Optimization
============

.. note::
   For tutorials and architecture documentation, see :doc:`/guide/optimization/index`.

The optimization module is organized as follows:

Entry Points
------------

.. autosummary::
   :toctree: optimization/
   :nosignatures:

   optimization.api
   optimization.problem

Results & Errors
----------------

.. autosummary::
   :toctree: optimization/
   :nosignatures:

   optimization.state
   optimization.errors

Native Solvers
--------------

.. autosummary::
   :toctree: optimization/native/
   :nosignatures:

   optimization.native.base
   optimization.native.least_squares
   optimization.native.torch_opt

Managed Adapters
----------------

.. autosummary::
   :toctree: optimization/managed/
   :nosignatures:

   optimization.managed.scipy_local
   optimization.managed.scipy_least_squares
   optimization.managed.scipy_global

Drivers
-------

.. autosummary::
   :toctree: optimization/
   :nosignatures:

   optimization.drivers

Evaluators
----------

.. autosummary::
   :toctree: optimization/evaluators/
   :nosignatures:

   optimization.evaluators.base
   optimization.evaluators.finite_difference
   optimization.evaluators.autograd

Step Controllers
----------------

.. autosummary::
   :toctree: optimization/control/
   :nosignatures:

   optimization.control.base
   optimization.control.identity
   optimization.control.levenberg
   optimization.control.kkt
   optimization.control.line_search

Stopping Criteria
-----------------

.. autosummary::
   :toctree: optimization/stopping/
   :nosignatures:

   optimization.stopping.base
   optimization.stopping.criteria
   optimization.stopping.composite

Observers
---------

.. autosummary::
   :toctree: optimization/observers/
   :nosignatures:

   optimization.observers.base
   optimization.observers.history
   optimization.observers.logging
   optimization.observers.checkpoint
   optimization.observers.cancel

Hard Constraints
----------------

Hard equality and inequality constraints are declared via
``OptimizationProblem.add_constraint`` and enforced at every step via the KKT
active-set method. See :ref:`hard_constraints` in the framework guide.

.. autosummary::
   :toctree: optimization/constraint/
   :nosignatures:

   optimization.constraint.constraint
   optimization.constraint.manager

Soft Constraint Strategies
---------------------------

``ConstraintStrategy`` objects apply per-step modifications (bounds projection,
SciPy-native constraints). These are distinct from hard constraints above.

.. autosummary::
   :toctree: optimization/constraints/
   :nosignatures:

   optimization.constraints.base
   optimization.constraints.bounds
   optimization.constraints.scipy_native

Standalone Optimizers
---------------------

These optimizers are not accessible via :func:`~optimization.api.minimize` by design.
Use them directly.

.. autosummary::
   :toctree: optimization/optimizer/
   :nosignatures:

   optimization.optimizer.scipy.glass_expert
   optimization.optimizer.scipy.orthogonal_descent
   optimization.optimizer.custom.particle_swarm

Operands
--------

.. autosummary::
   :toctree: optimization/operand/
   :nosignatures:
   :recursive:

   optimization.operand.aberration
   optimization.operand.operand
   optimization.operand.paraxial
   optimization.operand.ray

Variables
---------

.. autosummary::
   :toctree: optimization/variable/
   :nosignatures:
   :recursive:

   optimization.variable.asphere_coeff
   optimization.variable.base
   optimization.variable.chebyshev_coeff
   optimization.variable.conic
   optimization.variable.decenter
   optimization.variable.forbes_coeff
   optimization.variable.grid_sag
   optimization.variable.index
   optimization.variable.material
   optimization.variable.norm_radius
   optimization.variable.nurbs
   optimization.variable.polynomial_coeff
   optimization.variable.radius
   optimization.variable.reciprocal_radius
   optimization.variable.thickness
   optimization.variable.tilt
   optimization.variable.torch
   optimization.variable.variable
   optimization.variable.variable_manager
   optimization.variable.zernike_coeff

Scaling
-------

.. autosummary::
   :toctree: optimization/scaling/
   :nosignatures:
   :recursive:

   optimization.scaling.base
   optimization.scaling.identity
   optimization.scaling.linear
   optimization.scaling.log
   optimization.scaling.power
   optimization.scaling.reciprocal

Deprecated Classes (v0.7.0 removal)
-------------------------------------

.. deprecated::
   The following classes emit ``DeprecationWarning`` on construction and will be
   removed in **v0.7.0**: ``OptimizerGeneric``, ``LeastSquares``,
   ``DualAnnealing``, ``DifferentialEvolution``, ``SHGO``, ``BasinHopping``,
   ``TorchAdamOptimizer``, ``TorchSGDOptimizer``.

   Use :func:`~optimization.api.minimize` with the equivalent ``method=`` string
   instead. See :doc:`/guide/optimization/migration` for the full migration table.
