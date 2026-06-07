.. _interaction_models:

Interaction Models
==================

Interaction models define how rays interact with a surface. Each `Surface` object has an `InteractionModel` that determines whether a ray is refracted, reflected, or diffracted.

BaseInteractionModel
--------------------

The `BaseInteractionModel` is an abstract base class that defines the interface for all interaction models. It has two main methods:

- `interact_real_rays(rays)`: Interacts with real rays.
- `interact_paraxial_rays(rays)`: Interacts with paraxial rays.

RefractiveReflectiveModel
-------------------------

The `RefractiveReflectiveModel` is the most common interaction model. It handles both refraction and reflection based on the `is_reflective` flag.

ThinLensInteractionModel
------------------------

The `ThinLensInteractionModel` is used for paraxial surfaces. It simplifies the surface to an ideal thin lens with a given focal length.

DiffractiveModel
----------------

The `DiffractiveModel` is used for surfaces with diffraction gratings. It calculates the new direction of the ray based on the grating equation.

PhaseInteractionModel
---------------------

The `PhaseInteractionModel` is a powerful and flexible model that can be used to create surfaces with arbitrary phase profiles. It uses the Strategy pattern to delegate the phase calculation to a `BasePhaseProfile` object. This allows you to create custom phase profiles by subclassing `BasePhaseProfile` and implementing the `get_phase` and `get_gradient` methods.

The `PhaseInteractionModel` can be used to create a wide variety of optical components, such as:

- Lenses with complex aspheric or freeform surfaces
- Diffractive optical elements (DOEs)
- Metasurfaces

To create a surface with a phase profile, you need to create a `BasePhaseProfile` object and pass it to the `SurfaceFactory` using the `phase_profile` argument. The `interaction_type` will be automatically set to `phase`.

How to Extend This
------------------

**Scenario:** Add a new surface interaction model to Optiland.

**Step 1:** Create a new file in ``optiland/interactions/my_interaction.py``.
**Step 2:** Subclass ``BaseInteractionModel`` and implement ``interact_real_rays(rays, surface)``
and ``interact_paraxial_rays(rays, surface)``.
**Step 3:** Register in ``optiland/interactions/__init__.py``.
**Step 4:** Add tests in ``tests/test_interactions/test_my_interaction.py``.

For step-by-step guidance, see :ref:`extension_recipes`.
