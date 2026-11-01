.. _gallery_optimization:

Optimization
============

Optiland supports optimization through different backends.
The **NumPy backend** relies on Optiland's native damped least squares and on
SciPy optimizers, while the **Torch backend** uses PyTorch's native
optimization tools. Which one to use, and how to run and cancel it, is
explained in the :doc:`../optimization_guide`; one-call merit functions and hard
constraints are in :doc:`../optimization_merit_constraints`.

NumPy (SciPy) Optimization
--------------------------

These examples show how Optiland integrates with SciPy's optimizers
to improve optical systems using classical numerical methods.

.. nbgallery::
    optimization/rms_spot_size
    optimization/wavefront_error
    optimization/asphere
    optimization/beam_expander
    optimization/freeform
    optimization/glass_expert_example
    optimization/undo
    optimization/pickups
    optimization/constrained
    optimization/bounded_operands
    optimization/orthogonal_descent
    optimization/global
    optimization/basin_hopping
    optimization/particle_swarm_optimization
    optimization/shgo
    optimization/custom_scaler
    optimization/reciprocal_radii_optimization

Torch Optimization
------------------

These examples demonstrate optimization using the Torch backend,
taking advantage of PyTorch's autograd and optimizers for differentiable design.

.. nbgallery::
    optimization/torch_rms_spot_size
    optimization/torch_constrained
    optimization/torch_module_rms_spot
    optimization/torch_module_custom_objective