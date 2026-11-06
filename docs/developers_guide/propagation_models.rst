.. _propagation_models:

Propagation Models
==================

Propagation models define how rays propagate through a medium. Each `Material` object has a `PropagationModel` that determines how the ray's position changes as it travels. The model also owns every other quantity that depends on position inside the medium.

BasePropagationModel
--------------------

The `BasePropagationModel` is an abstract base class that defines the interface for all propagation models. It has one main method:

- `propagate(rays, t)`: Propagates rays a distance `t` through the medium.

Tracing calls the model through four hooks. Their defaults implement straight,
homogeneous travel, so a model only overrides what its medium changes:

- `propagate_to_surface(rays, surface)`: moves rays, in the local frame of
  `surface`, onto `surface` and adds the optical path. Default: geometric
  distance, `propagate(rays, t)`, then `t * n`.
- `boundary_index(rays, surface, entrance)`: the medium index where rays meet
  `surface`, used for refraction. `entrance` is the surface where the medium
  starts. Default: `n(wavelength)`.
- `paraxial_transfer(y, u, t)`: paraxial heights and slopes after an axial
  distance `t`. Default: `(y + t * u, u)`.
- `display_path(rays, surface)`: points to draw between two surface hits.
  Default: `None`, a straight segment.

Homogeneous
-----------

The `HomogeneousPropagation` model is used for materials with a uniform refractive index. It's the most common propagation model.

GRIN - gradient-index
---------------------

The `GRINPropagation` model is used for gradient-index materials, where the refractive index varies with position (see `GradientMaterial`).

It integrates the ray equation in Sharma's form, with state `(r, T = n t)`,
`dr/dτ = T`, `dT/dτ = n ∇n` and optical path `dOPL/dτ = n²`, using fixed RK4
steps of about `step_size` mm. Integration runs in the local frame of the
entrance surface, where the index is defined, so tilted and decentered media
work. A ray stops on the step where it crosses the next surface's sag, and a
straight chord from there lands it on the surface. Updates use `be.where`
instead of masked writes, so the model works and differentiates on both
backends.

Rays that do not reach the next surface within `max_steps` steps, or leave the
region where the index is valid, are vignetted. `propagate(rays, t)` and
`paraxial_transfer` raise `NotImplementedError`. `display_path` integrates only
the rays being drawn, so analysis traces never store intermediate points.
