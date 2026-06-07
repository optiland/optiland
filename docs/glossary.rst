.. _glossary:

Glossary
========

This page defines the core concepts in Optiland. Want runnable examples? See the :ref:`cheat_sheet`.

Optic
-----

The central container for an entire optical system. An ``Optic`` instance holds all surfaces, aperture
definitions, field points, and wavelength information. It also exposes paraxial analysis, aberration
computations, the ray tracer, and polarization state.

Example: ``lens = optic.Optic(name="My System")``

SurfaceGroup
------------

Manages the ordered collection of ``Surface`` objects within an ``Optic``. It propagates rays through
the system by invoking surface-specific logic at each step, aggregates ray-trace history, and exposes
methods for adding, removing, and modifying surfaces.

Surface
-------

Represents a single optical interface — a lens element, mirror, image plane, grating, etc. Each
surface is composed of:

- **Geometry**: The mathematical shape (e.g., planar, spherical, aspheric, freeform).
- **Material**: Refractive index and extinction coefficient on each side.
- **Coating**: Optional thin-film stack modifying reflection, transmission, or polarization.
- **Interaction model**: How rays interact — refraction, reflection, diffraction, or a custom phase profile.
- **Physical aperture**: Optional mask defining the clear aperture.
- **BSDF**: Bidirectional scattering distribution function for surface scatter.

Special surface types include ``ObjectSurface`` (first surface, object plane), ``ImageSurface``
(final surface, image plane), and surfaces carrying a ``ThinLensInteractionModel`` (paraxial thin-lens
approximation).

Material
--------

Defines the optical medium between surfaces. Optiland supports:

- ``MaterialFile``: Loads dispersion data from the `refractiveindex.info <https://refractiveindex.info>`_
  database, which is bundled with the package.
- ``IdealMaterial``: A wavelength-independent medium specified by a single refractive index.
- ``AbbeMaterial``: A medium specified by its refractive index at the d-line (nd) and Abbe number (Vd),
  using a Buchdahl dispersion model (recommended) or a legacy polynomial model.
- User-registered materials via ``MaterialRegistry``.

Geometry
--------

Defines the mathematical shape of a surface and provides two critical operations: ray–surface intersection
and surface normal computation. Built-in geometries include:

- ``StandardGeometry``: Spherical and conic surfaces.
- ``EvenAsphere``, ``OddAsphere``: Polynomial aspheres.
- ``Biconic``, ``Toroidal``, ``Polynomial``, ``ChebyshevPolynomialGeometry``, ``ZernikePolynomialGeometry``: Freeforms.
- ``PlaneGrating``, ``StandardGrating``: Diffraction gratings.
- ``NURBSGeometry``: Non-Uniform Rational B-Splines.
- ``ForbesGeometry``: Q-polynomial freeform surfaces (Forbes convention).

Custom geometries can be added by subclassing ``BaseGeometry`` (closed-form) or
``NewtonRaphsonGeometry`` (iterative).

Aperture
--------

Defines the system's limiting aperture. The aperture type determines how the entrance-pupil size is
specified:

- ``EPD``: Entrance pupil diameter in mm.
- ``imageFNO``: Image-space f-number.
- ``objectNA``: Object-space numerical aperture.
- ``float_by_stop_size``: The aperture stop physical diameter drives the pupil size.

Fields
------

Define the points in the object plane (or angular directions) being imaged. Fields can be specified by
angle (degrees) or object height (mm). Vignetting factors can be applied per field. Each ``Field`` has an
optional ``weight`` used in weighted analysis aggregation.

Wavelengths
-----------

Specify the wavelengths of light used for analysis. All values are stored internally in microns (µm).
One wavelength is designated as the primary wavelength, used for paraxial calculations and single-wavelength
analyses. Each ``Wavelength`` has an optional ``weight`` used in weighted analysis aggregation.

Coordinate System
-----------------

Each surface has its own Local Coordinate System (LCS) defined by position (x, y, z) and rotation
(rx, ry, rz) relative to a reference. Key conventions:

- Light propagates from **left to right** along the **+z axis**.
- Surface 1 is typically at the global origin (z = 0).
- **Thickness** is the axial separation to the *next* surface; positive means to the right.
- **Radius of curvature**: positive means center of curvature to the right (convex to the incoming
  beam); negative means to the left.
- Tilts and decenters are applied as ``R = Rz @ Ry @ Rx``.

Apodization
-----------

Defines the intensity (amplitude) distribution across the entrance pupil. The default is
``UniformApodization`` (flat). ``GaussianApodization`` applies a Gaussian intensity profile across the
pupil, modelling a Gaussian input beam.

Backend
-------

Optiland routes all numerical operations through a unified backend abstraction (``optiland.backend``),
allowing transparent switching between **NumPy** (default, CPU) and **PyTorch** (GPU and autograd). All
Optiland code uses ``import optiland.backend as be`` instead of importing NumPy or PyTorch directly.
Switch backends with ``be.set_backend("torch")`` or ``be.set_backend("numpy")``.

See :ref:`configurable_backend` in the Developer's Guide for the full backend architecture.
