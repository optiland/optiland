Optimization
============

This section covers the optimization functionality of the Optiland package. The optimization module
provides a set of tools for optimizing optical systems. It includes a variety of optimization algorithms,
such as gradient-based and evolutionary algorithms, as well as tools for defining optimization variables
and objectives.

For thin film and coating specific optimization (e.g., Needle Synthesis), see the :doc:`api_thin_film` module.

The optimization module is divided into four subcategories:

1. Core Functionalities - Problem definition and optimizers
2. Operands - Functions to compute optical performance metrics
3. Variables - Properties of optical elements that can be optimized
4. Scaling - Methods to scale optimization variables for better performance

Core Functionalities
--------------------

.. autosummary::
   :toctree: optimization/
   :caption: Optimization Modules

   optimization.problem
   optimization.optimizer.scipy.base
   optimization.optimizer.scipy.basin_hopping
   optimization.optimizer.scipy.differential_evolution
   optimization.optimizer.scipy.dual_annealing
   optimization.optimizer.scipy.least_squares
   optimization.optimizer.scipy.orthogonal_descent
   optimization.optimizer.scipy.shgo
   optimization.optimizer.scipy.glass_expert
   optimization.optimizer.torch.base
   optimization.optimizer.torch.adam
   optimization.optimizer.torch.sgd


Operands
--------

The `optimization.operand` subpackage contains the following modules:

.. autosummary::
   :toctree: optimization/operand/
   :caption: Operand Modules
   :recursive:

   optimization.operand.aberration
   optimization.operand.operand
   optimization.operand.paraxial
   optimization.operand.ray


Variables
---------

The `optimization.variable` subpackage contains the following modules:

.. autosummary::
   :toctree: optimization/variable/
   :caption: Variable Submodules
   :recursive:

   optimization.variable.asphere_coeff
   optimization.variable.base
   optimization.variable.chebyshev_coeff
   optimization.variable.conic
   optimization.variable.decenter
   optimization.variable.forbes_coeff
   optimization.variable.index
   optimization.variable.material
   optimization.variable.norm_radius
   optimization.variable.nurbs
   optimization.variable.polynomial_coeff
   optimization.variable.radius
   optimization.variable.reciprocal_radius
   optimization.variable.thickness
   optimization.variable.tilt
   optimization.variable.variable_manager
   optimization.variable.variable
   optimization.variable.zernike_coeff


Scaling
-------

The `optimization.scaling` subpackage contains the following modules:

.. autosummary::
   :toctree: optimization/scaling/
   :caption: Scaling Modules
   :recursive:

   optimization.scaling.base
   optimization.scaling.identity
   optimization.scaling.linear
   optimization.scaling.log
   optimization.scaling.power
   optimization.scaling.reciprocal