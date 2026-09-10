.. _try_it:

.. meta::
   :description: Run Optiland in your browser with an in-page Python kernel. No installation required.

Try Optiland in Your Browser
============================

The shell below is a complete Python kernel running in your browser
(`xeus-python <https://github.com/jupyter-xeus/xeus-python>`_ compiled to
WebAssembly and served by `JupyterLite <https://jupyterlite.readthedocs.io/>`_)
with Optiland preinstalled. Nothing is sent to a server and nothing needs to be
installed on your machine.

.. note::

   The kernel and its packages are downloaded on first use, which can take
   anywhere from a few seconds to a minute depending on your connection.
   3D plotting (``draw3D``, VTK) is not available in the browser; run those
   examples in a local Python environment instead (see :doc:`installation`).

.. replite::
   :kernel: xpython
   :toolbar: True
   :width: 100%
   :height: 600px

   from optiland.samples.objectives import CookeTriplet

   lens = CookeTriplet()
   lens.draw()

Ideas to explore
----------------

Not sure what to type? Paste any of these into the shell after the first
example has run:

.. code-block:: python

   lens.draw()  # Visualize the optical layout

.. code-block:: python

   effl = lens.paraxial.f2()  # Retrieve the effective focal length

.. code-block:: python

   # Trace 1024 random rays for the on-axis field point, then use rays.x, rays.y
   # for the intersection points
   rays = lens.trace(Hx=0, Hy=0, wavelength=0.55, num_rays=1024, distribution="random")

.. code-block:: python

   # Run a spot diagram analysis
   from optiland.analysis import SpotDiagram

   spot = SpotDiagram(lens)
   spot.view()

Next steps
----------

- :doc:`quickstart` — from a fresh install to an optimized system in five minutes.
- :doc:`installation` — install Optiland locally, with or without PyTorch.
- :doc:`learning_guide` — the complete tutorial series.
