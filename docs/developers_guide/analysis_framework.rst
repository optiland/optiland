Analysis Framework
==================

Optiland provides a variety of analysis tools to evaluate the performance of optical systems. Each analysis tool focuses
on specific optical characteristics, such as image quality, aberrations, or system distortion. These analyses use ray
tracing results to generate meaningful visual and numerical outputs.

Supported Analysis Types
------------------------

Optiland includes several built-in analysis tools, each tailored to a specific aspect of optical system performance:

- **Spot Diagrams**: Evaluate the focusing quality by plotting ray intersection points on the image plane.
- **Ray Fans**: Visualize transverse ray error versus relative position in the pupil, useful for assessing aberrations.
- **Distortion**: Quantify image distortion caused by the optical system. Standard distortion and grid distortion are supported.
- **Field Curvature**: Examine variations in focus across the field of view.
- **Encircled Energy**: Compute the integrated energy as a function of radius from the image center.
- **Pupil Aberration**: Assess difference between the paraxial and real ray intersection point at the aperture stop.
- **RMS vs. Field**: Plot the root mean square (RMS) wavefront error or RMS spot size as a function of field position.
- **Through Focus**: Analyze the optical system's performance as a function of focus position, including spot diagrams.
- **Y Y-bar**: Compare the chief and marginal ray heights at each surface.
- **PSF**: Compute the point spread function (PSF) of the optical system using either the FFT or direct integration (Huygens-Fresnel) method.
- **MTF**: Calculate the modulation transfer function (MTF) of the optical system via both geometric and diffraction (FFT) methods.
- **Wavefront**: Compute the wavefront error across the field of view and pupil. Supports both focal systems (spherical reference) and afocal systems (planar reference), with selectable reference strategies (Chief Ray, Centroid, Best Fit).
- **Zernike Polynomials**: Decompose the wavefront error into Zernike polynomials.
- **Image Simulation**: Simulate the image formation process, accounting for spatially variable blur, distortion, and lateral color.

Weighted Analysis
-----------------

Optiland analyses can optionally aggregate results over multiple fields and wavelengths based on user-defined weights. When creating ``Field`` and ``Wavelength`` objects, you can assign a ``weight`` attribute (defaulting to ``1.0``). Analyses that generate aggregated system metrics (e.g., MTF, PSF, Spot Diagram, RMS vs. Field) will automatically compute weighted averages across the specified fields and wavelengths to reflect these importance values. Assigning a weight of ``0.0`` will exclude the field or wavelength from the weighted mean, but it will still be evaluated individually in the standalone data arrays.

Analysis Workflow
-----------------

Each analysis follows a similar workflow:

1. An **Optic** instance is passed to an analysis class.
2. Ray tracing is performed as needed, with relevant rays being traced to specific surfaces or planes.
3. Results are computed based on the traced rays and the optical system configuration.
4. The computed results are stored in the analysis instance and can be visualized or exported.

Example: Spot Diagram Analysis
-------------------------------

For example, a spot diagram is computed as follows:

- The `SpotDiagram` class is instantiated with an **Optic** instance.
- Rays are traced to the image plane.
- The intersection points are recorded and visualized.

.. note::
   For details on the ray tracing framework and structure, see the :ref:`ray_overview` section.

Running an Analysis
-------------------

To run an analysis, follow these general steps:

1. **Select an Analysis Type**: Import the relevant analysis class for the desired evaluation. For example:

.. code:: python

   from optiland.analysis import SpotDiagram

2. **Create the Analysis Instance**: Pass your optical system (Optic instance) to the analysis class. When the instance is created, the required data is typically created. Optionally select specific parameters for the analysis.

.. code:: python

   spot_diagram = SpotDiagram(optic, fields='all', wavelengths='all', num_rings=6, distribution='hexapolar')

3. **Visualize the Result**: Use visualization tools to view the results.

.. code:: python

   spot_diagram.view()

Extending the Analysis Framework
--------------------------------

Adding a new analysis type is straightforward:

- Create a new analysis class with methods to perform ray tracing and compute results.
- Use Optiland's ray tracing framework as needed to simplify implementation.

.. tip::
   See individual analysis class documentation for parameter details or the :ref:`example_gallery` example code.

How to Extend This
------------------

**Scenario:** Add a new analysis class to Optiland.

**Step 1:** Create a new file in ``optiland/analysis/my_analysis.py``.
**Step 2:** Subclass ``BaseAnalysis`` and implement ``_generate_data()`` (perform ray tracing and
store results on ``self``) and ``view()`` (plot or print the results).
**Step 3:** Register in ``optiland/analysis/__init__.py``.
**Step 4:** Add tests in ``tests/test_analysis/test_my_analysis.py``.

For step-by-step guidance, see :ref:`extension_recipes`.
