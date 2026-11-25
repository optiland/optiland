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

Saving and Loading
------------------

A single configuration's ``Optic.to_dict()`` can not record a pickup that takes its value
from another optic, so it would load as an unlinked optic. Save the whole system instead:
:meth:`~optiland.multiconfig.multi_configuration.MultiConfiguration.to_dict` stores every
configuration with ``Optic.to_dict()`` and lists the cross-configuration pickups
separately:

.. code-block:: python

   {
       "version": 1,
       "configurations": [optic.to_dict(), ...],
       "links": [[config_index, pickup_index, source_config_index], ...],
   }

:meth:`~optiland.multiconfig.multi_configuration.MultiConfiguration.from_dict` rebuilds
each configuration with ``Optic.from_dict()`` and then restores the source of every listed
pickup, so linked parameters keep propagating after loading. ``to_dict`` raises a ``ValueError`` if a pickup takes its value from an optic
that is not one of the configurations. Use the generic JSON helpers to save to a file:

.. code-block:: python

   from optiland.fileio import load_obj_from_json, save_obj_to_json
   from optiland.multiconfig import MultiConfiguration

   save_obj_to_json(mc, "zoom.json")
   mc = load_obj_from_json(MultiConfiguration, "zoom.json")

Analysing configurations side by side
-------------------------------------

:meth:`~optiland.multiconfig.multi_configuration.MultiConfiguration.analyze` runs one analysis on every configuration (or on a
list of indices) and returns a :class:`optiland.multiconfig.analysis.MultiConfigAnalysis`.
The positional and keyword arguments after the analysis class are passed to it, after
the optic:

.. code-block:: python

    from optiland.analysis import SpotDiagram

    spots = mc.analyze(SpotDiagram, num_rings=6)            # all configurations
    spots.results[1].rms_spot_radius()                      # an ordinary SpotDiagram
    figures = spots.view(show=False)                        # one figure per configuration

Each result is an ordinary analysis bound to ``mc.configurations[i]``.
``MultiConfigAnalysis.view(*args, show=True, **kwargs)`` calls each result's ``view``
and returns their outputs as a list. The figures are not merged: analyses build their
own layout and most clear a figure passed to them through ``fig_to_plot_on``, so one
figure per configuration is the only layout that works for every analysis without
changing their signatures. Each figure's ``suptitle`` is prefixed with
``"Configuration i"``. ``show`` is passed as ``show=False`` to every analysis whose
``view`` accepts it, and ``plt.show()`` is called once at the end when ``show`` is True.

:meth:`~optiland.multiconfig.multi_configuration.MultiConfiguration.summary` returns a ``pandas.DataFrame`` with one column per
configuration (``"Config 0"``, ...), the multi-configuration editor view. The first rows
are always ``EFL``, ``F/#``, ``Image height`` (paraxial chief ray at the image, maximum
field), ``Total track`` and ``EPD``. They are followed by every property whose *value*
differs between configurations, whether or not a pickup links it: aperture type and
value, field type and coordinates (``"Field 1 y"``), wavelengths (``"Wavelength 0
(µm)"``) and, per surface, radius, conic (curved surfaces only), thickness, material,
decenters and tilts (``"Surface 2 thickness"``). Surface indices are those of
``optic.surfaces`` (0 is the object surface). A quantity that can not be computed for a
configuration, such as a paraxial value of a folded system, is NaN rather than an error.
The summary reads the current state and does not run the pickups; call
``optic.updater.update()`` first if a configuration may be stale.

Prescription of a multi-configuration system
--------------------------------------------

:class:`optiland.prescription.Prescription` describes one optic and raises for a
``MultiConfiguration``. Use :class:`optiland.prescription.MultiConfigPrescription`
instead; it has the same ``build()``, ``view()`` and ``save(path, renderer=None)`` and
uses the same document model and renderers (console, plain text, PDF):

.. code-block:: python

    from optiland.prescription import MultiConfigPrescription

    MultiConfigPrescription(mc).view()
    MultiConfigPrescription(mc).save("zoom.txt")

The default report has the system overview of configuration 0, a configuration table
(the rows of ``mc.summary()``), the first-order properties of every configuration and
the surface tables of configuration 0.

Sections that need the whole system subclass
:class:`optiland.prescription.BaseMultiConfigSection` and implement
``build(multiconfig) -> Section``. Two are provided:
:class:`optiland.prescription.ConfigurationTableSection` and
:class:`optiland.prescription.PerConfigurationSection`, an adapter that renders any
single-optic section once per configuration, prefixing its block titles with
``"Configuration i"``. A plain single-optic section passed in ``sections`` is built on
configuration 0, so custom reports mix both kinds:

.. code-block:: python

    from optiland.prescription import (
        ConfigurationTableSection,
        PerConfigurationSection,
        SeidelAberrationSection,
        SystemOverviewSection,
    )

    MultiConfigPrescription(
        mc,
        sections=[
            SystemOverviewSection(),                            # configuration 0
            ConfigurationTableSection(),
            PerConfigurationSection(SeidelAberrationSection()),  # every configuration
        ],
    ).view()
