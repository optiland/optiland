Environment
===========

This section covers Optiland's environment subpackage, which calculates the
refractive index of air. This is used to account for the effect of ambient
temperature, pressure, humidity, and CO2 concentration on an optical system's
performance, as opposed to the fixed reference-air assumption used elsewhere
in the package.

The main entry point is :func:`~optiland.environment.air_index.refractive_index_air`,
which dispatches to one of several well-established empirical models based on an
:class:`~optiland.environment.conditions.EnvironmentalConditions` instance. Each
model - Ciddor, Edlén, Birch & Downs, and Kohlrausch - is implemented in its own
module under ``environment.models`` and can also be called directly.

.. code-block:: python

   from optiland.environment import EnvironmentalConditions, refractive_index_air

   conditions = EnvironmentalConditions(
       temperature=15.0,
       pressure=101325.0,
       relative_humidity=0.0,
       co2_ppm=400.0,
   )
   n = refractive_index_air(0.55, conditions, model="ciddor")

.. autosummary::
   :toctree: environment/
   :caption: Environment Modules

   environment.air_index
   environment.conditions

.. autosummary::
   :toctree: environment/
   :caption: Environment Models

   environment.models.ciddor
   environment.models.edlen
   environment.models.birch_downs
   environment.models.kohlrausch
