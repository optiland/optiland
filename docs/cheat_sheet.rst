.. _cheat_sheet:

API Cheat Sheet
===============

Copy-paste snippets for the 20 most common Optiland tasks.
New to these concepts? See the :ref:`glossary` first.

----

1. Install and import
---------------------

.. code-block:: bash

   pip install optiland

.. code-block:: python

   from optiland import optic
   import optiland.backend as be

----

2. Load a sample lens
---------------------

.. code-block:: python

   from optiland.samples.objectives import CookeTriplet, ReverseTelephoto

   lens = CookeTriplet()
   lens.info()   # print surface table

----

3. Build a simple singlet from scratch
---------------------------------------

.. code-block:: python

   from optiland import optic

   lens = optic.Optic()
   lens.surfaces.add(index=0, radius=float("inf"), thickness=float("inf"))  # object at infinity
   lens.surfaces.add(index=1, radius=50.0, thickness=5.0, material="N-BK7", is_stop=True)
   lens.surfaces.add(index=2, radius=-50.0, thickness=45.0)
   lens.surfaces.add(index=3)  # image plane
   lens.set_aperture(aperture_type="EPD", value=10.0)
   lens.fields.set_type("angle")
   lens.fields.add(y=0.0)
   lens.wavelengths.add(value=0.5876, is_primary=True)
   lens.updater.image_solve()

----

4. Add a surface
-----------------

.. code-block:: python

   # Insert a surface at index 2 with radius, thickness, and material
   lens.surfaces.add(index=2, radius=-435.76, thickness=6.0, material=("F2", "schott"))

----

5. Set aperture, field, and wavelength
---------------------------------------

.. code-block:: python

   lens.set_aperture(aperture_type="EPD", value=10.0)
   # alternatives: "imageFNO", "objectNA", "float_by_stop_size"

   lens.fields.set_type("angle")   # or "object_height"
   lens.fields.add(y=0.0)
   lens.fields.add(y=14.0)
   lens.fields.add(y=20.0)

   lens.wavelengths.add(value=0.4861)                    # F-line
   lens.wavelengths.add(value=0.5876, is_primary=True)   # d-line
   lens.wavelengths.add(value=0.6563)                    # C-line

----

6. Switch backend (NumPy ↔ PyTorch)
-------------------------------------

.. code-block:: python

   import optiland.backend as be

   be.set_backend("torch")    # enable PyTorch (autograd, GPU)
   be.set_backend("numpy")    # revert to NumPy (default)

   # GPU and precision (PyTorch only)
   be.set_device("cuda")
   be.set_precision("float64")

----

7. Draw the lens (2D)
----------------------

.. code-block:: python

   lens.draw(num_rays=5, distribution="line_y")

----

8. Draw the lens (3D)
----------------------

.. code-block:: python

   lens.draw3D(num_rays=24, distribution="ring")

----

9. Trace rays manually
-----------------------

.. code-block:: python

   # Trace a distribution of rays for a given field and wavelength
   rays = lens.trace(Hx=0, Hy=0, wavelength=0.5876, num_rays=64, distribution="hexapolar")
   print(rays.x, rays.y)   # image-plane x, y coordinates

   # Trace a single ray defined by normalized field + pupil coordinates
   ray = lens.trace_generic(Hx=0, Hy=1, Px=0, Py=0, wavelength=0.5876)

----

10. Spot diagram
-----------------

.. code-block:: python

   from optiland.analysis import SpotDiagram

   spot = SpotDiagram(lens)
   spot.view()

----

11. Ray fan plot
-----------------

.. code-block:: python

   from optiland.analysis import RayFan

   fan = RayFan(lens)
   fan.view()

----

12. Wavefront / Zernike decomposition
---------------------------------------

.. code-block:: python

   from optiland.wavefront import Wavefront, ZernikeOPD

   wf = Wavefront(lens, field=(0, 0), wavelength="primary")
   wf.view()

   zfit = ZernikeOPD(lens, field=(0, 0), wavelength="primary", num_terms=37)
   zfit.view()

----

13. PSF and MTF
----------------

.. code-block:: python

   from optiland.psf import FFTPSF
   from optiland.mtf import FFTMTF

   psf = FFTPSF(lens, field=(0, 0), wavelength="primary")
   psf.view()

   mtf = FFTMTF(lens)
   mtf.view()

----

14. Paraxial properties (EFL, f/#, pupil positions)
-----------------------------------------------------

.. code-block:: python

   print("EFL:", lens.paraxial.f2())
   print("f/#:", lens.paraxial.FNO())
   print("EPD:", lens.paraxial.EPD())
   print("EPL:", lens.paraxial.EPL())   # entrance pupil location
   print("XPL:", lens.paraxial.XPL())   # exit pupil location
   print("Magnification:", lens.paraxial.magnification())

----

15. Define an optimization variable
-------------------------------------

.. code-block:: python

   from optiland.optimization import OptimizationProblem

   problem = OptimizationProblem()
   problem.add_variable(lens, "radius", surface_number=1)
   problem.add_variable(lens, "thickness", surface_number=1)

----

16. Define an operand
----------------------

.. code-block:: python

   input_data = {"optic": lens}
   problem.add_operand(operand_type="f2", target=50.0, weight=1, input_data=input_data)
   problem.add_operand(operand_type="rms_spot_size", target=0.0, weight=1,
                       input_data={"optic": lens, "field_index": 1, "wavelength_index": 0,
                                   "distribution": "hexapolar", "num_rays": 100})

----

17. Run local optimization
---------------------------

.. code-block:: python

   from optiland.optimization import LeastSquares

   optimizer = LeastSquares(problem)
   result = optimizer.optimize()
   print(result)

----

18. Run global optimization
----------------------------

.. code-block:: python

   from optiland.optimization import DualAnnealing

   optimizer = DualAnnealing(problem)
   result = optimizer.optimize()
   print(result)

----

19. Save / load system (JSON)
------------------------------

.. code-block:: python

   from optiland.fileio import save_optiland_file, load_optiland_file

   save_optiland_file(lens, "my_lens.json")
   lens2 = load_optiland_file("my_lens.json")

----

20. Generate a prescription report
------------------------------------

.. code-block:: python

   from optiland.prescription import Prescription

   p = Prescription(lens)
   p.view()                         # Rich console output
   p.save("prescription.txt")       # plain text
   p.save("prescription.pdf")       # PDF (requires reportlab)
