.. _functionalities:

Functionalities
===============

Optiland offers a powerful and extensible platform for the design, analysis, and optimization of optical systems. With a fully differentiable architecture and high-performance backend support, it is suitable for both classical optical engineering and modern machine learning applications. Below is an overview of its main capabilities.

Design Tools
------------

- **Differentiable and Backend-Configurable Architecture**:
  Optiland supports both NumPy and PyTorch backends for all core computations. This flexibility enables easy integration with scientific computing pipelines and deep learning frameworks.
- **Sequential Ray Tracing**:
  Trace rays through traditional or advanced systems, including asymmetric and freeform designs.
- **Lens System Modeling**:
  Built-in support for spherical, conic, aspheric, and fully freeform surfaces.
- **Aperture, Field, and Wavelength Configuration**:
  Configure optical systems for diverse apertures, fields, and wavelengths.

Analysis Tools
--------------

- **Real and Paraxial Ray Tracing**:
  Perform precise ray-based evaluations for both idealized and physically realistic systems.
- **Polarization Ray Tracing**:
  Model vectorial light propagation, including polarization effects and birefringent materials.
- **Jones Pupil Analysis**:
  Visualize spatially resolved Jones matrix elements at the exit pupil to assess polarization properties.
- **Comprehensive Optical Analysis**:
  Generate spot diagrams, ray aberration fans, OPD maps, distortion plots, and more.
- **Wavefront Analysis**:
  Decompose wavefronts into Zernike polynomials, compute RMS/peak error, and visualize wavefront error maps, with support for both focal (spherical reference) and afocal (planar reference) systems.
- **PSF and MTF Computation**:
  Evaluate image quality and spatial frequency response in imaging systems.
- **BSDF and Scattering Models**:
  Simulate surface scattering using measured or analytical BSDF models.

Optimization and Tolerancing
----------------------------

- **Flexible Optimization Framework**:
  Includes gradient-based solvers, global search algorithms, and support for automatic differentiation.
- **Operand-Based Merit Function Design**:
  Define custom merit functions using symbolic operands, easily extended with user-defined metrics.
- **Tolerancing and Sensitivity Analysis**:
  Perform Monte Carlo simulations and parametric sweeps to assess manufacturability and robustness.
- **Categorical Variable Optimization (Glass Expert)**:
  Optimize lens materials (categorical variables) alongside continuous parameters using the specialized `GlassExpert` tool, which intelligently searches glass catalogs.
- **Extensible Framework**:
  Add new optimization variables, constraints, or algorithms with minimal overhead.

Extended Source Modeling
-----------------------

- **Extended Source Ray Tracing**:
  Model spatially and angularly extended light sources and trace them through optical systems using the ``ExtendedSourceOptic`` wrapper.
- **Single-Mode Fiber (SMF) Source**:
  Generate physically accurate ray bundles representing fiber outputs, with Gaussian spatial and angular distributions and quasi-random Sobol sampling.
- **Custom Source Support**:
  Define custom source types by inheriting from ``BaseSource`` and implementing the ``generate_rays`` method.
- **Source Visualization**:
  Validate source definitions with multi-panel diagnostic plots showing spatial distributions, angular distributions, cross-sections, and ray propagation paths.
- **Irradiance Analysis Compatibility**:
  Compute incoherent irradiance distributions at detector surfaces from extended source illumination.

Material Database
-----------------

- **Integrated Refractive Index Library**:
  Access data from refractiveindex.info directly within the package.
- **User-Defined Materials**:
  Create and register new materials with custom dispersion models.

Coating and Thin Film Modeling
------------------------------

- **Multilayer Thin Film Stacks**:
  Define, analyze, and optimize thin film stacks using the Transfer Matrix Method.
- **Needle Synthesis**:
  An iterative layer-insertion optimizer based on the Tikhonravov & Trubetskov algorithm. Configurable candidate materials and spectral/angular targets let you automatically design optimal thin-film coatings.
- **Thin Film Tolerancing**:
  Perform sensitivity analysis and Monte Carlo simulations to assess how manufacturing variations in layer thickness and refractive index affect spectral performance.

Visualization
-------------

- **Interactive 2D and 3D Visualization**:
  Plot optical layouts, surface properties, and ray traces using Matplotlib (2D) and VTK (3D). The 2D plots are fully interactive, providing tooltips with detailed information when you hover over optical components.
- **Customizable Visualization Themes**:
  Optiland includes a flexible theme system to control the appearance of 2D plots. You can easily switch between predefined themes (e.g., ``light``, ``dark``) or create your own. For a hands-on example, see the :doc:`gallery/miscellaneous/themes` notebook.
- **Optiland GUI**:
  A fully functional graphical user interface (GUI) for interactive design, analysis, and visualization of optical systems. Features include a VS Code-style **Command Palette** (Ctrl+K) for quick access to actions and tools. See the :ref:`gui_quickstart` for more details on using the GUI.

Interoperability and Scripting
------------------------------

- **Zemax File Support**:
  Import and adapt existing designs from Zemax .zmx files. Write Optiland designs to .zmx format.
- **CODE V File Support**:
  Import and adapt existing designs from CODE V .seq files. Write Optiland designs to .seq format.
- **JSON-Based I/O**:
  Save and load optical systems in a human-readable JSON format.
- **Python API**:
  Build and control optical systems programmatically for scripting, automation, and integration.

Performance
-----------

- **High-Speed Ray Tracing**:
  Optiland achieves real-time performance on modern hardware, with ray tracing speeds of (approx.):
  
  - **150–200 million ray surfaces per second** or more using the PyTorch backend on GPU.
  - **5–10 million ray surfaces per second** on CPU with the NumPy backend.

- **GPU Acceleration**:
  The PyTorch backend enables seamless GPU acceleration via PyTorch.
- **ML/DL Integration**:
  The differentiable PyTorch backend makes Optiland compatible with deep learning pipelines and allows gradient-based optimization of optical systems within broader ML frameworks.
- **JIT Compilation with Numba**:
  The NumPy backend uses Numba where appropriate to speed up CPU-bound calculations.

.. note::
   Have suggestions or feature requests? Feel free to open an issue on our GitHub repository. We welcome contributions and ideas from the community.
