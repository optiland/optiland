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

Derived Configurations
~~~~~~~~~~~~~~~~~~~~~~

A configuration can be derived from any existing configuration, not only from Config 0
(for example ``add_configuration(1)``). The new configuration is linked to its source for
radius, conic and thickness. Pickups it inherits from the source keep pointing at their real
source configurations: the deep copy does not duplicate the other configurations. An
inherited pickup that drives the same surface attribute as one of the new links is dropped,
so each attribute has exactly one driver. Other inherited links are kept, for example a
material link to Config 0 created by ``set_material(..., configurations="all")``.

Breaking Links
--------------

To create a "Zoom" parameter (a parameter that varies between configurations), the specific variable must be decoupled from the base configuration.
This is handled automatically by the :meth:`set_property` method (and its convenience wrappers `set_thickness`, `set_radius`, etc.).

When a value is set for a specific configuration index (other than 0), the framework:
1. Removes the existing Pickup for that parameter on that specific configuration.
2. Sets the new value directly on the `Optic` instance of that configuration.

This allows for flexible definition of both shared and unique parameters.
