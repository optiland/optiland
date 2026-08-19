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
