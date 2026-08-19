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
   print("EPL:", lens.paraxial.EPL())   # entrance pupil location, relative to surface 1
   print("XPL:", lens.paraxial.XPL())   # exit pupil location, relative to the image surface
   print("Magnification:", lens.paraxial.magnification())

   # EFL of a lens group, i.e. surfaces 1 through 2 only
   print("Group EFL:", lens.paraxial.f2_range(1, 2))

.. note::

   ``f2_range(start, end)`` treats the surface range as a lens group **in
   isolation**, with its own conjugates. It is not a decomposition of the
   system's power, so the group focal lengths of a design will not add back up
   to ``f2()``. The underlying 2x2 ray-transfer matrix of a surface range is
   available as ``lens.paraxial.ray_transfer_matrix(start, end)``.

.. note::

   ``EPL()`` is measured **relative to the first physical surface** (surface 1),
   matching the convention of ``XPL()`` (relative to the image surface) and the
   other first-order quantities. If you need the entrance pupil on the same
   axial coordinate as surface positions — e.g. to compare against object or
   surface positions — use ``lens.paraxial.entrance_pupil_axial_position()``
   (``entrance_pupil_z()`` is a deprecated alias). For the pupil's real-space
   location in a folded system, use
   ``lens.paraxial.entrance_pupil_point_gcs()``.

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

----

21. Non-sequential (illumination / stray-light)
------------------------------------------------

.. code-block:: python

   from optiland.coordinate_system import CoordinateSystem
   from optiland.nonsequential import (
       NSQScene, Spectrum,
       CollimatedSourceConfig, LensConfig, IrradianceDetectorConfig,
   )

   scene = NSQScene()
   spec = Spectrum.monochromatic(0.55)            # wavelengths in µm
   scene.add_source("S", CoordinateSystem(z=0),
                    CollimatedSourceConfig(spectrum=spec, total_flux=1.0,
                                           aperture_radius=10))
   scene.add_lens("L", CoordinateSystem(z=50),
                  LensConfig(r1=100, r2=-100, thickness=5, material="N-BK7",
                             front_aperture_radius=12.5))
   scene.add_detector("D", CoordinateSystem(z=150),
                      IrradianceDetectorConfig(width=20, height=20))

   result = scene.trace(num_rays=1_000_000, max_depth=16, seed=42)
   result.detectors["D"].plot()                   # irradiance map

----

22. Convert a sequential lens to non-sequential
------------------------------------------------

.. code-block:: python

   from optiland.nonsequential import sequential_to_nonsequential

   scene = sequential_to_nonsequential(lens)      # for stray-light / ghost analysis

----

23. Differentiable non-sequential (gradients)
----------------------------------------------

.. code-block:: python

   import optiland.backend as be
   be.set_backend("torch")                        # build scene AFTER this

   # ... build scene with torch.Tensor params, then:
   result = scene.trace(num_rays=2_000, max_depth=16, seed=42)
   loss = ((result.detectors["D"].data - target) ** 2).mean()
   loss.backward()                                # gradients flow to scene params
