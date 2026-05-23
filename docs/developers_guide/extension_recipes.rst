.. _extension_recipes:

Extension Recipes
=================

Step-by-step recipes for the 8 most common Optiland contribution scenarios. Each recipe lists the
file to create, what to subclass, which methods to implement, and where to register.

For the full architecture context behind each recipe, follow the cross-links to the relevant
Developer's Guide section.

.. seealso::
   :doc:`architecture` · :doc:`surface_overview` · :doc:`geometry_overview` ·
   :doc:`interaction_models` · :doc:`analysis_framework` · :doc:`optimization_framework` ·
   :doc:`tolerancing_framework` · :doc:`configurable_backend` · :doc:`visualization_framework`

----

Recipe 1: Add a New Surface Geometry
--------------------------------------

**Scenario:** Add a new parametric surface shape (e.g., a Chebyshev variant, a metasurface sag).

**Step 1:** Create ``optiland/geometries/my_geometry.py``.

**Step 2:** Subclass ``BaseGeometry`` (closed-form) or ``NewtonRaphsonGeometry`` (iterative):

.. code-block:: python

   from __future__ import annotations
   import optiland.backend as be
   from optiland.geometries.base import BaseGeometry

   class MyGeometry(BaseGeometry):
       def distance(self, rays):
           # return propagation distance along ray to surface intersection
           ...

       def sag(self, x, y):
           # return sag (z deviation from vertex plane) at (x, y)
           ...

       def surface_normal(self, rays):
           # return (nx, ny, nz) unit normal at intersection
           ...

**Step 3:** Register in ``optiland/geometries/__init__.py``:

.. code-block:: python

   from .my_geometry import MyGeometry
   __all__ = [..., "MyGeometry"]

**Step 4:** Add the geometry type string to ``optiland/surfaces/factories/geometry_factory.py``
so it can be created from a string identifier (needed for JSON serialisation).

**Step 5:** Add tests in ``tests/test_geometries/test_my_geometry.py`` using both backends.

See :doc:`../examples/Tutorial_8a_Custom_Surface_Types` for a worked example.

----

Recipe 2: Add a New Surface Interaction Model
----------------------------------------------

**Scenario:** Add a custom ray-surface interaction (e.g., a birefringent crystal, a holographic
element with a non-standard diffraction law).

**Step 1:** Create ``optiland/interactions/my_interaction.py``.

**Step 2:** Subclass ``BaseInteractionModel``:

.. code-block:: python

   from __future__ import annotations
   from optiland.interactions.base import BaseInteractionModel

   class MyInteractionModel(BaseInteractionModel):
       def interact_real_rays(self, rays, surface):
           # modify rays.L, rays.M, rays.N, rays.intensity in-place
           ...

       def interact_paraxial_rays(self, rays, surface):
           # modify paraxial ray height and angle
           ...

**Step 3:** Register in ``optiland/interactions/__init__.py``.

**Step 4:** Add tests in ``tests/test_interactions/test_my_interaction.py``.

See :doc:`interaction_models` for the full interaction-model architecture.

----

Recipe 3: Add a New Geometry (Shortcut Path)
----------------------------------------------

If you only need to define a sag function and let Newton-Raphson handle the intersection:

**Step 1:** Subclass ``NewtonRaphsonGeometry`` from ``optiland/geometries/newton_raphson.py``:

.. code-block:: python

   from optiland.geometries.newton_raphson import NewtonRaphsonGeometry
   import optiland.backend as be

   class MyNRGeometry(NewtonRaphsonGeometry):
       def sag(self, x, y):
           # your sag formula; return be.array of same shape as x
           ...

       def _surface_normal(self, x, y):
           # return (nx, ny, nz) at (x, y) — used internally
           ...

**Step 2 onward:** Same as Recipe 1, steps 3-5.

----

Recipe 4: Add a New Analysis Class
------------------------------------

**Scenario:** Add a new optical performance metric (e.g., ghost image intensity, scatter PSF).

**Step 1:** Create ``optiland/analysis/my_analysis.py``.

**Step 2:** Subclass ``BaseAnalysis``:

.. code-block:: python

   from __future__ import annotations
   from optiland.analysis.base import BaseAnalysis

   class MyAnalysis(BaseAnalysis):
       def _generate_data(self):
           # trace rays, compute metric, store results on self
           ...

       def view(self):
           # plot or print results
           ...

**Step 3:** Register in ``optiland/analysis/__init__.py``.

**Step 4:** Add tests in ``tests/test_analysis/test_my_analysis.py``.

See :doc:`analysis_framework` and :doc:`../examples/Tutorial_2c_Aberration_Analyses` for context.

----

Recipe 5: Add a New Optimization Operand
------------------------------------------

**Scenario:** Add a custom merit-function term (e.g., chief-ray angle at a specific surface,
image distortion at a given field).

**Step 1:** Add a new function to ``optiland/optimization/operand/operand.py`` and register it
in the operand registry at the bottom of that file:

.. code-block:: python

   def my_operand(optic, surface_number, **kwargs):
       # compute and return a scalar value
       ...

   # Register:
   OPERAND_REGISTRY["my_operand"] = my_operand

**Step 2:** Add tests in ``tests/test_optimization/test_operand.py``.

See :doc:`optimization_framework` and :doc:`../examples/Tutorial_3c_User_Defined_Optimization` for context.

----

Recipe 6: Add a Custom Tolerance Sensitivity Class
----------------------------------------------------

**Scenario:** Model a non-standard manufacturing error (e.g., index inhomogeneity, surface
irregularity described by Zernike coefficients).

**Step 1:** Create or extend ``optiland/tolerancing/perturbations.py`` with a new perturbation
class. Perturbations must implement ``apply(optic)`` and ``revert(optic)``:

.. code-block:: python

   class MyPerturbation:
       def __init__(self, surface_number, magnitude):
           self._surface_number = surface_number
           self._magnitude = magnitude
           self._original = None

       def apply(self, optic):
           s = optic.surfaces.surfaces[self._surface_number]
           self._original = s.geometry.radius
           s.geometry.radius += self._magnitude

       def revert(self, optic):
           s = optic.surfaces.surfaces[self._surface_number]
           s.geometry.radius = self._original

**Step 2:** Pass an instance of your perturbation to ``Tolerancing.add_perturbation()``.

**Step 3:** Add tests in ``tests/test_tolerancing/``.

See :doc:`tolerancing_framework` for the complete workflow.

----

Recipe 7: Add a Backend-Agnostic Utility Function
---------------------------------------------------

**Scenario:** Add a numerical helper that must work with both NumPy and PyTorch tensors.

**Step 1:** Create your utility using only ``optiland.backend`` operations:

.. code-block:: python

   from __future__ import annotations
   import optiland.backend as be

   def my_metric(x, y):
       """Compute something using the active backend."""
       dot_xy = be.sum(x * y)
       norm_x = be.sqrt(be.sum(x * x))
       norm_y = be.sqrt(be.sum(y * y))
       return dot_xy / (norm_x * norm_y)

**Step 2:** If the function requires a backend-specific API (e.g., ``torch.autograd.grad``),
add abstract and concrete implementations to ``AbstractBackend``, ``NumpyBackend``, and
``TorchBackend`` in ``optiland/backend/``.

**Step 3:** Test on both backends using the ``set_test_backend`` fixture from ``tests/conftest.py``
and ``assert_allclose`` from ``tests/utils.py``.

See :doc:`configurable_backend` for the full backend architecture.

----

Recipe 8: Add a New 2D or 3D Renderer
----------------------------------------

**Scenario:** Add a new visualization component (e.g., plot a custom ray bundle, render a
focal-plane heat map, add a new 3D actor).

**For 2D (Matplotlib):**

**Step 1:** Create a component class in ``optiland/visualization/``:

.. code-block:: python

   class MyComponent2D:
       def __init__(self, optic, ax):
           self._optic = optic
           self._ax = ax

       def draw(self):
           # add matplotlib artists to self._ax
           ...

**Step 2:** Integrate with ``OpticViewer`` in ``optiland/visualization/optic_viewer.py`` if
the component should appear in the standard ``lens.draw()`` output.

**For 3D (VTK):**

**Step 1:** Create a component class that generates VTK actors:

.. code-block:: python

   import vtk

   class MyActor3D:
       def get_actor(self):
           # return a vtkActor (or vtkAssembly)
           ...

**Step 2:** Integrate with ``OpticViewer3D`` in ``optiland/visualization/optic_viewer_3d.py``.

**Step 3:** Add tests in ``tests/test_visualization/``.

See :doc:`visualization_framework` for the full viewer architecture.
