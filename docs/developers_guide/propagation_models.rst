.. _propagation_models:

Propagation Models
==================

Propagation models define how rays propagate through a medium. Each `Material` object has a `PropagationModel` that determines how the ray's position changes as it travels.

BasePropagationModel
--------------------

The `BasePropagationModel` is an abstract base class that defines the interface for all propagation models. It has one main method:

- `propagate(rays, t)`: Propagates rays a distance `t` through the medium.

Homogeneous
-----------

The `HomogeneousPropagation` model is used for materials with a uniform refractive index. It's the most common propagation model.

GRIN - gradient-index
---------------------

The `GRINPropagation` model is used for gradient-index materials, where the refractive index varies with position.
