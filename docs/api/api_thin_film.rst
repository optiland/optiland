Thin Film
=========

This section covers the thin film functionality of the Optiland package. The thin film module
provides a set of tools for defining, analyzing, and optimizing multilayer thin film stacks.

Core Functionalities
--------------------

.. autosummary::
   :toctree: thin_film/
   :caption: Core Modules

   thin_film.analysis
   thin_film.core
   thin_film.layer
   thin_film.stack


Optimization
------------

.. autosummary::
   :toctree: thin_film/optimization/
   :caption: Optimization Modules
   :recursive:

   thin_film.optimization.needle
   thin_film.optimization.optimizer
   thin_film.optimization.report
   thin_film.optimization.operand.core
   thin_film.optimization.operand.plotter
   thin_film.optimization.operand.thin_film
   thin_film.optimization.variable.layer_thickness


Tolerancing
-----------

.. autosummary::
   :toctree: thin_film/tolerancing/
   :caption: Tolerancing Modules
   :recursive:

   thin_film.tolerancing.core
   thin_film.tolerancing.monte_carlo
   thin_film.tolerancing.perturbation
   thin_film.tolerancing.sensitivity_analysis
