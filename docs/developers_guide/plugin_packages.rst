.. _plugin_packages:

Plugin Packages
===============

:doc:`extension_recipes` shows how to add a new surface geometry, material, or analysis by
editing Optiland's own source tree. This page instead walks through shipping each of those as a
**separate, pip-installable package** that Optiland discovers automatically at runtime, with no
edit to Optiland itself, using Python `entry points
<https://packaging.python.org/en/latest/specifications/entry-points/>`_.

Optiland defines three entry-point groups: ``optiland.surfaces``, ``optiland.materials``, and
``optiland.analyses``. A plugin package declares one or more of them in its own
``pyproject.toml`` and points each at a zero-argument callable. ``optiland/plugins.py`` resolves
and calls every registered callable, at most once per process:

.. code-block:: python

   import optiland.plugins as plugins
   plugins.load_plugins(plugins.SURFACES_GROUP)

- For **surfaces** and **materials**, this happens lazily, triggered by the first access to
  ``GeometryFactory`` / ``MaterialRegistry`` respectively — installing Optiland with no plugins
  present pays no import-time cost.
- For **analyses**, there is no factory to hang a lazy trigger off (analyses are instantiated
  directly, e.g. ``SpotDiagram(optic)``, not looked up by string), so ``optiland.analysis``
  loads the ``optiland.analyses`` group eagerly on package import instead.

A failing plugin only produces a ``UserWarning`` — it never breaks Optiland for everyone else
installed alongside it.

----

Walkthrough 1: A Surface Geometry Plugin
-----------------------------------------

Package layout for a standalone ``optiland-my-surface`` distribution:

.. code-block:: text

   optiland-my-surface/
     pyproject.toml
     my_surface_plugin/
       __init__.py
       geometry.py
       register.py

``my_surface_plugin/geometry.py`` — the geometry itself, following :doc:`extension_recipes`
Recipe 1 (subclass ``BaseGeometry`` or ``NewtonRaphsonGeometry``):

.. code-block:: python

   from __future__ import annotations
   from optiland.geometries.newton_raphson import NewtonRaphsonGeometry

   class MyGeometry(NewtonRaphsonGeometry):
       def sag(self, x, y):
           ...

       def _surface_normal(self, x, y):
           ...

``my_surface_plugin/register.py`` — the entry-point target. It registers a factory function and
config dataclass against Optiland's ``GeometryFactory``, exactly as an in-tree geometry would:

.. code-block:: python

   from dataclasses import dataclass

   from optiland.surfaces.factories.geometry_factory import GeometryFactory


   @dataclass
   class MyGeometryConfig:
       surface_type = "my_surface"
       radius: float = float("inf")


   def _create_my_geometry(cs, config):
       from .geometry import MyGeometry

       return MyGeometry(cs, radius=config.radius)


   def register() -> None:
       GeometryFactory.register("my_surface", _create_my_geometry, MyGeometryConfig)

``pyproject.toml`` of the plugin package:

.. code-block:: toml

   [project.entry-points."optiland.surfaces"]
   my_surface = "my_surface_plugin.register:register"

Once ``pip install optiland-my-surface`` is run alongside Optiland,
``optic.surfaces.add(surface_type="my_surface", ...)`` works with no changes to Optiland's
source.

----

Walkthrough 2: A Material Catalog Plugin
------------------------------------------

**Scenario:** ship a proprietary or vendor glass catalog as an installable package rather than
asking every user to load a CSV/YAML file by hand.

Package layout:

.. code-block:: text

   optiland-acme-glass/
     pyproject.toml
     acme_glass_plugin/
       __init__.py
       register.py
       data/
         acme_catalog.yml

``acme_glass_plugin/register.py``:

.. code-block:: python

   from importlib import resources

   from optiland.materials.registry import MaterialRegistry


   def register() -> None:
       catalog_path = resources.files("acme_glass_plugin.data") / "acme_catalog.yml"
       MaterialRegistry.instance().register_file(str(catalog_path))

``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."optiland.materials"]
   acme_glass = "acme_glass_plugin.register:register"

After installing ``optiland-acme-glass``, materials from ``acme_catalog.yml`` resolve through
the normal ``Material("...")`` lookup path the first time any material is resolved — the
catalog file never has to live inside the user's own project.

----

Walkthrough 3: An Analysis Plugin
------------------------------------

**Scenario:** ship a custom performance metric (e.g. a proprietary stray-light or ghost-image
analysis) as an installable package.

Package layout:

.. code-block:: text

   optiland-ghost-analysis/
     pyproject.toml
     ghost_analysis_plugin/
       __init__.py
       analysis.py
       register.py

``ghost_analysis_plugin/analysis.py`` — following :doc:`extension_recipes` Recipe 4:

.. code-block:: python

   from __future__ import annotations
   from optiland.analysis.base import BaseAnalysis

   class GhostImageAnalysis(BaseAnalysis):
       def _generate_data(self):
           ...

       def view(self, show: bool = True):
           ...

Analyses have no central factory, so the plugin's ``register()`` makes the class importable
from Optiland's own namespace, matching how in-tree analyses are exposed:

.. code-block:: python

   import sys

   def register() -> None:
       from .analysis import GhostImageAnalysis

       setattr(sys.modules["optiland.analysis"], "GhostImageAnalysis", GhostImageAnalysis)

``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."optiland.analyses"]
   ghost_analysis = "ghost_analysis_plugin.register:register"

After installing ``optiland-ghost-analysis``, ``from optiland.analysis import
GhostImageAnalysis`` works without Optiland ever having imported the plugin package by name.
(A plugin analysis is also perfectly usable without registering into ``optiland.analysis`` at
all — nothing stops a caller from doing ``from ghost_analysis_plugin.analysis import
GhostImageAnalysis`` directly. The entry point only buys the convenience of the familiar
``optiland.analysis`` import path.)

----

See :doc:`extension_recipes` for the in-tree-edit versions of these same three recipes, and
``CONTRIBUTING.md`` for the dead-code-audit and tooling conventions that apply to plugin
development the same as to the core package.
