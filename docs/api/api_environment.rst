Environment
===========

This section covers Optiland's environment subpackage, which calculates the
refractive index of air. This is used to account for the effect of ambient
temperature, pressure, humidity, and CO2 concentration on an optical system's
performance, as opposed to the fixed reference-air assumption used elsewhere
in the package.

The main entry point for air is :func:`~optiland.environment.air_index.refractive_index_air`,
which dispatches to one of several well-established empirical models based on an
:class:`~optiland.environment.conditions.EnvironmentalConditions` instance. Each
model - Ciddor, Edlén, Birch & Downs, and Kohlrausch - is implemented in its own
module under ``environment.models`` and can also be called directly.

To evaluate a lens at another temperature, use
:func:`~optiland.environment.thermal.optic_at_temperature`. It returns a deep
copy whose glasses, radii, thicknesses and air gaps have been updated for the
temperature change; the input optic is left unchanged.

.. code-block:: python

   from optiland.environment import EnvironmentalConditions, refractive_index_air

   conditions = EnvironmentalConditions(
       temperature=15.0,
       pressure=101325.0,
       relative_humidity=0.0,
       co2_ppm=400.0,
   )
   n = refractive_index_air(0.55, conditions, model="ciddor")

The Kohlrausch model uses the OpticStudio dry-air convention. Its pressure input
is in pascals; the evaluator converts it to atmospheres before applying the
reference-temperature scaling. Humidity and CO2 do not enter this model.
At 0.55 µm it reproduces 1.00027308 at 20 °C / 101325 Pa and 1.00052810 at
30 °C / 202650 Pa, to the precision published in the
`Ansys worked example <https://optics.ansys.com/hc/en-us/articles/42661799783443-How-OpticStudio-calculates-refractive-index-at-arbitrary-temperatures-and-pressures>`_.
The evaluator accepts NumPy/Torch wavelength and temperature arrays and preserves
their gradients; other environmental models retain their own input contracts.
It rejects non-finite or non-positive wavelengths consistently for scalar and
array inputs, and rejects a non-positive temperature scaling denominator.
Changing this formula
does not switch the dispatcher's default model or connect global environment
settings to material evaluation automatically.

.. autosummary::
   :toctree: environment/
   :caption: Environment Modules

   environment.air_index
   environment.conditions
   environment.thermal

.. autosummary::
   :toctree: environment/
   :caption: Environment Models

   environment.models.ciddor
   environment.models.edlen
   environment.models.birch_downs
   environment.models.kohlrausch
