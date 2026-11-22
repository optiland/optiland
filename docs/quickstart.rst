.. _quickstart:

Quickstart — Your First 5 Minutes
==================================

This page takes you from a fresh install to a working, optimized optical system. Run the blocks
in order, in a Jupyter notebook or a Python script: later blocks reuse the lens built in section 3.

----

1. Install
----------

.. code-block:: bash

   pip install optiland

For GPU-accelerated differentiable ray tracing also install PyTorch:

.. code-block:: bash

   pip install optiland[torch]           # CPU-only PyTorch
   # or, for an NVIDIA GPU, a CUDA build of PyTorch (here CUDA 12.6):
   pip install torch --index-url https://download.pytorch.org/whl/cu126

See :doc:`installation` for the details.

----

2. Hello, World
---------------

Load and visualize a Cooke Triplet in 3D — two lines of code:

.. code-block:: python

   from optiland.samples.objectives import CookeTriplet

   lens = CookeTriplet()
   lens.draw3D()

.. figure:: images/cooke.png
   :alt: Cooke Triplet 3D visualization
   :align: center

   3D visualization of the Cooke Triplet lens system.

Print the surface table (similar to a Lens Data Editor):

.. code-block:: python

   lens.info()

----

3. Build from Scratch
---------------------

.. tip::

   The surface parameters below follow a specific model — see :doc:`conventions` for why
   ``thickness`` and ``material`` describe the space *after* a surface, not the surface itself.

Create a simple biconvex singlet in 8 lines:

.. code-block:: python

   from optiland import optic

   lens = optic.Optic(name="Singlet")
   lens.surfaces.add(index=0, radius=float("inf"), thickness=float("inf"))  # object at infinity
   lens.surfaces.add(index=1, radius=50.0, thickness=5.0, material="N-BK7", is_stop=True)
   lens.surfaces.add(index=2, radius=-50.0, thickness=0.0)
   lens.surfaces.add(index=3)  # image plane
   lens.set_aperture(aperture_type="EPD", value=10.0)
   lens.fields.set_type("angle")
   lens.fields.add(y=0.0)
   lens.wavelengths.add(value=0.5876, is_primary=True)
   lens.updater.image_solve()   # moves image surface to paraxial focus

----

4. Trace Rays
-------------

Trace a bundle of rays and inspect the image-plane coordinates. For the ``hexapolar``
distribution, ``num_rays`` is the number of rings N, so 1 + 3N(N+1) rays are traced
(12481 for N = 64):

.. code-block:: python

   rays = lens.trace(Hx=0, Hy=0, wavelength=0.5876, num_rays=64, distribution="hexapolar")
   print("x range:", rays.x.min(), "to", rays.x.max())
   print("y range:", rays.y.min(), "to", rays.y.max())

Trace a single ray specified by normalized field and pupil coordinates:

.. code-block:: python

   # chief ray for the on-axis field
   ray = lens.trace_generic(Hx=0, Hy=0, Px=0, Py=0, wavelength=0.5876)

----

5. Spot Diagram
---------------

Visualize the geometric ray spread at the image plane:

.. code-block:: python

   from optiland.analysis import SpotDiagram

   spot = SpotDiagram(lens)
   spot.view()

The resulting plot shows the ray scatter for each field and wavelength. A tighter cluster indicates
better image quality.

----

6. One-Step Optimization
------------------------

Let the optimizer bend the lens: vary both radii to minimize the on-axis RMS spot size while holding
the effective focal length at 50 mm.

.. code-block:: python

   from optiland.optimization import OptimizationProblem, LeastSquares

   problem = OptimizationProblem()
   problem.add_variable(lens, "radius", surface_number=1)
   problem.add_variable(lens, "radius", surface_number=2)
   problem.add_operand(operand_type="f2", target=50.0, weight=1, input_data={"optic": lens})
   problem.add_operand(
       operand_type="rms_spot_size",
       target=0.0,
       weight=10,
       input_data={"optic": lens, "Hx": 0, "Hy": 0, "wavelength": 0.5876,
                   "distribution": "hexapolar", "num_rays": 6, "surface_number": -1},
   )
   optimizer = LeastSquares(problem)
   result = optimizer.optimize()
   print("Final merit:", result.cost)
   problem.info()                       # operands and variables, before and after

The image plane stayed where it was, so refocus with ``lens.updater.image_solve()`` and redraw
with ``lens.draw()`` to see the new shape. The full story is in
:doc:`Tutorial 3a <examples/Tutorial_3a_Simple_Optimization>` and the :doc:`optimization_guide`.

----

7. Save and Load
----------------

Serialize the optimized design to JSON and reload it in a new session:

.. code-block:: python

   from optiland.fileio import save_optiland_file, load_optiland_file

   save_optiland_file(lens, "singlet.json")
   lens2 = load_optiland_file("singlet.json")
   lens2.info()

----

8. Beyond Imaging: Non-Sequential & Illumination
------------------------------------------------

Optiland also ships a differentiable **non-sequential (NSQ)** engine for
illumination design, stray-light, and ghost analysis - where light is free to
take any path through a 3-D scene. You can even convert an existing sequential
lens in one line::

   from optiland.nonsequential import sequential_to_nonsequential
   scene = sequential_to_nonsequential(lens)

See the :ref:`NSQ gallery <gallery_nonsequential>` overview and the on-ramp
tutorial :doc:`Tutorial 10a <examples/Tutorial_10a_Non_Sequential_and_Illumination>`.

----

9. What Next?
-------------

That's the whole workflow: install, build, trace, analyze, optimize and save. Everything else in
Optiland is a deeper version of one of these steps.

Choose where to go next based on your goals:

- :doc:`how_do_i` — a task-shaped index: find your question, get the notebook that answers it
- :ref:`start_here` — persona-based routing for students, engineers, researchers, and contributors
- :doc:`conventions` — one page on how a lens is described; worth reading next
- :doc:`cheat_sheet` — copy-paste snippets for the most common tasks
- :doc:`learning_guide` — 40+ tutorials, from beginner to advanced
- :doc:`Example Gallery <gallery/introduction>` — visual showcase of designs and analyses
