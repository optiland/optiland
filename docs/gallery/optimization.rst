.. _gallery_optimization:

Optimization
============

Examples illustrating Optiland's unified optimization framework.
For architecture details and tutorials, see :doc:`/guide/optimization/index`.

Getting Started
---------------

.. nbgallery::
    optimization/rms_spot_size
    optimization/wavefront_error
    optimization/beam_expander

Classical / Stepped (DLS)
-------------------------

.. nbgallery::
    optimization/constrained
    optimization/bounded_operands

Variables & Surfaces
--------------------

.. nbgallery::
    optimization/asphere
    optimization/freeform
    optimization/reciprocal_radii_optimization
    optimization/pickups
    optimization/custom_scaler
    optimization/undo

Controlling a Run
-----------------

.. nbgallery::
    optimization/history_observer
    optimization/custom_stopping
    optimization/checkpoint_observer

Differentiable (Torch)
-----------------------

.. nbgallery::
    optimization/torch_rms_spot_size
    optimization/torch_constrained
    optimization/torch_module_rms_spot
    optimization/torch_module_custom_objective

Global & Standalone
--------------------

.. nbgallery::
    optimization/global
    optimization/basin_hopping
    optimization/shgo
    optimization/particle_swarm_optimization
    optimization/orthogonal_descent
    optimization/glass_expert_example
