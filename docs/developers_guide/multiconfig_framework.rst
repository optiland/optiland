Multi-Configuration Framework
=============================

The Multi-Configuration framework in Optiland is designed to manage optical systems with multiple states, such as zoom lenses, thermal soaks, or scanning systems.

Architecture
------------

The core of the framework is the :class:`optiland.multiconfig.multi_configuration.MultiConfiguration` class.

This class maintains a list of :class:`optiland.optic.Optic` instances, where each instance represents a full, independent state of the optical system.
Configuration 0 is always the "base" optic from which others are typically derived.

Configuration Creation
----------------------

When `add_configuration(source_config_idx)` is called, the source optic is deep-copied to create a new configuration.
This ensures that each configuration has its own independent set of Surfaces, Fields, and other properties.

Data Linking (Pickups)
----------------------

To maintain the relationships required for a multi-configuration system (where most parameters are shared across configurations), Optiland uses :class:`optiland.pickup.Pickup`.

When a new configuration is added, the framework automatically establishes pickups for:
- Surface Radii
- Surface Conic Constants
- Surface Thicknesses (except the last surface)
- Material properties

This means that by default, changing a parameter in the Base Configuration (Config 0) will propagate to all other configurations.

Breaking Links
--------------

To create a "Zoom" parameter (a parameter that varies between configurations), the specific variable must be decoupled from the base configuration.
This is handled automatically by the :meth:`set_property` method (and its convenience wrappers `set_thickness`, `set_radius`, etc.).

When a value is set for a specific configuration index (other than 0), the framework:
1. Removes the existing Pickup for that parameter on that specific configuration.
2. Sets the new value directly on the `Optic` instance of that configuration.

This allows for flexible definition of both shared and unique parameters.

Tolerancing a multi-configuration system
----------------------------------------

The configurations of a zoom lens are one physical lens: a radius error or a
decentered element is the same in every zoom position. A plain
:class:`optiland.tolerancing.Tolerancing` perturbs one optic only, so a
tolerance on configuration 0 leaves the other configurations at nominal, even
for linked parameters (operand evaluation does not apply pickups).
:class:`optiland.tolerancing.MultiConfigTolerancing` tolerances all
configurations together:

.. code-block:: python

   from optiland.tolerancing import MultiConfigTolerancing
   from optiland.tolerancing.monte_carlo import MonteCarlo

   tol = MultiConfigTolerancing(mc)
   tol.add_operand("rms_spot_size", spot_args, configuration="all")
   tol.add_tolerance("radius", 0.05, surface_number=1)  # every configuration
   tol.add_tolerance("thickness", 0.02, surface_number=4, configurations=[1])
   tol.add_compensator("quick_focus", configuration="all")

   monte_carlo = MonteCarlo(tol)
   monte_carlo.run(500, seed=1)
   monte_carlo.yield_fraction({0: ("<", 0.02), 1: ("<", 0.02)})  # all pass

- ``add_tolerance(..., configurations="all" | [i, ...])`` builds one tolerance
  type per selected configuration and groups them in a
  :class:`~optiland.tolerancing.multiconfig.ConfigurationGroupTolerance`, which
  applies the same deviation to every member. Label, phase and units are those
  of the first member, and the group sits in an ordinary ``Tolerance``, so
  ``SensitivityAnalysis``, ``MonteCarlo`` (including ``workers=``) and
  ``InverseSensitivity`` run unchanged. ``add_default_tolerances`` and
  ``add_tolerances_from_drawing`` add to all configurations.
- **Pickups win.** The deviation is applied directly only to members whose
  attribute is not driven by a pickup from another selected configuration. A
  pickup-driven member gets the perturbed value through its pickup, so a
  pickup with a scale and an offset gives ``scale * (v0 + delta) + offset``.
  Tolerance types state their pickup-linkable attributes with
  ``ToleranceType.pickup_targets()`` (radius, fringes, conic, thickness, index
  and Abbe); the default, an empty list, means "always apply". Decenters,
  tilts and irregularity are not pickup-linked and are applied to every
  configuration. Selecting a configuration whose attribute is picked up from
  a configuration that is not selected raises ``ValueError``.
- After every apply and reset of a group with pickup targets, the pickups of
  the non-master configurations are applied (``updater.update()``), so no
  evaluation sees stale pickups. Groups without pickup targets skip this:
  applying the thickness pickups rebuilds the axial surface positions, which
  would drop the axial part of a rigid-body move.
- Form tolerances (fringes, irregularity) take the clear semi-aperture of the
  first selected configuration, so every configuration sees the same surface
  error.
- ``add_operand(..., configuration=i | "all")`` sets ``input_data["optic"]``.
  With ``"all"``, one operand per configuration is added and the result
  columns end with ``" (config i)"``; other operand names do not change.
- ``add_compensator(variable_type, configuration=i | "all")`` puts the
  variable on that configuration, for example one image-gap thickness per
  zoom position. ``"quick_focus"`` is kept per configuration and runs before
  the optimizer compensators; its result column ends with ``" (config i)"``
  for configurations other than 0. A thickness compensator on a gap that is
  picked up from configuration 0 is overwritten by the pickup, so unlink the
  gap (``mc.set_thickness(k, value, [i])``) to refocus a configuration on its
  own.
- All configurations must have the same number of surfaces.
