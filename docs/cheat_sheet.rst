Optiland Cheat Sheet
====================

Optiland is a Python library for designing, analyzing, and optimizing optical systems. This guide covers the fundamental concepts to get you started.

Core Concepts
-------------

At its heart, Optiland revolves around a few key components:

* **Optic** Object: This is the main container for your entire optical system. It holds all the surfaces, aperture definitions, field points, and wavelength information.
  
  * Example: ``my_system = optic.Optic()``

* **SurfaceGroup** (``optiland.surfaces.SurfaceGroup``): Manages the collection of surfaces within an ``Optic`` object.

* **Surfaces** (``Surface``): These represent the individual optical surfaces (lenses, mirrors, image planes, etc.). Each surface has a geometry, material properties on either side, and can optionally have coatings, physical apertures, or be designated as the system's aperture stop.
  
  * **Object Surface** (``ObjectSurface``): The first surface in your system, representing the object being imaged. It can be at a finite distance or at infinity.
  * **Image Surface** (``ImageSurface``): The final surface where the image is formed.
  * **Standard Surface** (``StandardGeometry``): Spherical or conic surfaces.
  * **Aspheric & Freeform Surfaces**: Optiland supports various complex geometries like ``EvenAsphere``, ``OddAsphere``, ``PolynomialGeometry``, ``ChebyshevPolynomialGeometry``, ``ZernikePolynomialGeometry``, and more...
  * **Paraxial Surface** (``ParaxialSurface``): A thin lens approximation defined by its focal length, useful for initial layouts.

* **Materials** (``Material``): Define the optical properties of the media between surfaces, primarily the refractive index (``n``) and optionally the extinction coefficient (``k``) as a function of wavelength.
  
  * Optiland can use data from the `refractiveindex.info <https://refractiveindex.info>`_ database (``MaterialFile``) or allow you to define ideal materials (``IdealMaterial``) or materials based on index (nd) and Abbe number (Vd) (``AbbeMaterial``). The ``AbbeMaterial`` supports both a legacy polynomial model and a new, more accurate Buchdahl model (recommended).

* **Geometries** (``BaseGeometry``): These define the mathematical shape of a surface (e.g., plane, sphere, asphere).

* **Aperture** (``BaseSystemAperture``): Defines the system's limiting aperture. This can be specified as Entrance Pupil Diameter (EPD), Image Space F-number (imageFNO), Object Space Numerical Aperture (objectNA), or Float by Stop Size (float_by_stop_size).

* **Fields** (``Field``, ``FieldGroup``): Define the points in the object plane that are being imaged. Can be specified by angle or object height. Vignetting can also be applied.

* **Wavelengths** (``Wavelength``, ``WavelengthGroup``): Specify the wavelengths of light used for analysis, including a primary wavelength. All wavelengths are internally converted to microns (µm).

* **Coordinate Systems** (``CoordinateSystem``): Each surface has its own local coordinate system (LCS) defined by its position (x, y, z) and rotation (rx, ry, rz) relative to a reference system.

* **Apodization** (``BaseApodization``): Defines the intensity distribution within the pupil. This is uniform (``UniformApodization``) by default, but other options include Gaussian (``GaussianApodization``).

**Numerical Backend**

* Optiland can use **NumPy** (default) or **PyTorch**.
* PyTorch allows for automatic differentiation.
    * Switch with:

        .. code-block:: python

            import optiland.backend as be
            be.set_backend("torch")  # or "numpy"

Basic Workflow: Defining an Optical System
------------------------------------------

1.  **Import Optiland:**

    .. code-block:: python

        from optiland import optic
        from optiland import materials # if using specific materials
        import optiland.backend as be  # backend for numerical operations - either numpy or torch

2.  **Create an** ``Optic`` **Instance:**

    .. code-block:: python

        my_lens = optic.Optic(name="My Cooke Triplet")

3.  **Add Surfaces** (``add_surface``): Surfaces are added sequentially.
    * The **first surface (index 0)** is typically the object surface.

        .. code-block:: python

            my_lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf) # Object at infinity

    * Add optical surfaces with their properties:

        .. code-block:: python

            my_lens.surfaces.add(index=1, radius=22.01359, thickness=3.25896, material="SK16")
            my_lens.surfaces.add(index=2, radius=-435.76044, thickness=6.00755) # Air gap by default
            my_lens.surfaces.add(index=3, radius=-22.21328, thickness=0.99997, material=("F2", "schott"), is_stop=True) # Stop surface
            # ... more surfaces ...

    * The **last surface** is the image plane.

        .. code-block:: python

            my_lens.surfaces.add(index=N) # N is the index after the last optical surface

4.  **Set System Aperture** (``set_aperture``):

    .. code-block:: python

        my_lens.set_aperture(aperture_type="EPD", value=10.0)  # Entrance Pupil Diameter of 10 mm
        # Or: my_lens.set_aperture(aperture_type="imageFNO", value=5.0)
        # Or: my_lens.set_aperture(aperture_type="float_by_stop_size", value=7.6), specifies diameter of the stop surface

5.  **Define Field of View** (``set_field_type``, ``add_field``):

    .. code-block:: python

        my_lens.fields.set_type(field_type="angle") # Field specified by angle
        my_lens.fields.add(y=0.0)  # On-axis field
        my_lens.fields.add(y=14.0) # Off-axis field at 14 degrees
        my_lens.fields.add(y=20.0)
        # Or for object height:
        # my_lens.fields.set_type(field_type="object_height")
        # my_lens.fields.add(y=10.0) # Object height of 10 mm

6.  **Define Wavelengths** (``add_wavelength``):

    .. code-block:: python

        my_lens.wavelengths.add(value=0.4861) # F-line (blue) in µm
        my_lens.wavelengths.add(value=0.5876, is_primary=True) # d-line (yellow), primary
        my_lens.wavelengths.add(value=0.6563) # C-line (red)

7.  **(Optional) Image Plane Solve** (``image_solve``): Moves the image surface to the paraxial focus.

    .. code-block:: python

        my_lens.image_solve()

Coordinate System & Sign Conventions
------------------------------------

Understanding Optiland's coordinate system and sign conventions is crucial:

* **Global Coordinate System (GCS)**: A fixed reference frame.
* **Local Coordinate System (LCS)**: Each surface has its own LCS.
* **Light Propagation**: From **left to right**, along the positive **z-axis**.
* **Surface Vertex**: Surface 1 typically at GCS origin (z=0). Others at their LCS origin.
* **Thickness**: Axial separation to the *next* surface. **Positive** means to the right.
* **Radius of Curvature (R)**:
    * **Positive R**: Center of curvature to the **right** (convex to left).
    * **Negative R**: Center of curvature to the **left** (concave to left).
    * **Infinite R**: Planar surface.
* **Tilts and Decenters**: The rotation matrix (of the global CS) is given by ``R = Rz @ Ry @ Rx``.
* **Ray Parameters**:
    * **Height (y)**: Positive above the optical axis.
    * **Slope (u - paraxial)**: Positive if traveling upwards.
    * **Direction Cosines (L, M, N - real)**: Components of the unit vector.
* **Angles**: Positive clockwise.

Ray Tracing
-----------

Optiland uses *normalized coordinates* for both the field and pupil to define rays in a general, system-independent way:

- **Field Coordinates** (`Hx`, `Hy`): Define the ray's starting field position. `(0, 0)` corresponds to the optical axis, and `(±1, ±1)` spans the full normalized field of view.
- **Pupil Coordinates** (`Px`, `Py`): Define the ray's position in the entrance pupil. `(0, 0)` corresponds to the chief ray, and `(±1, ±1)` spans the full normalized entrance pupil.

Optiland can trace both paraxial and real rays.

* **Paraxial Rays**:

    * For first-order calculations. Access through ``optic.paraxial``.
    * Example:

        .. code-block:: python

            heights, slopes = lens.paraxial.trace(Hy, Py)

* **Real Rays**:

    * For detailed analysis, including aberrations.

    * Example: Trace a bundle of rays

        .. code-block:: python

            optic.trace(Hx, Hy, wavelength, num_rays, distribution)

    * Example: Trace a specific ray, defined by the normalized field and pupil coordinates.

        .. code-block:: python

            optic.trace_generic(Hx, Hy, Px, Py, wavelength)

* **Advanced Ray Tracing** (``RealRays``, ``surface_group.trace``): For more control, create a ``RealRays`` object and trace using ``optic.surfaces.trace(rays)``.

    * Example:

        .. code-block:: python

            from optiland.rays import RealRays
            import optiland.backend as be
            # Assume 'my_lens' is an existing Optic object
            # Create a grid of rays at z=0 (e.g., entrance pupil plane)
            x_coords = be.linspace(-5.0, 5.0, 3) # Adjust range based on EPD
            y_coords = be.linspace(-5.0, 5.0, 3)
            X, Y = be.meshgrid(x_coords, y_coords)
            # Create a collimated ray bundle (traveling along +z)
            x_in = X.reshape(-1)
            y_in = Y.reshape(-1)
            z_in = be.zeros_like(x_in)
            L_in = be.zeros_like(x_in)
            M_in = be.zeros_like(x_in)
            N_in = be.ones_like(x_in)
            intensity = be.ones_like(x_in)
            # Create the RealRays object
            primary_wl = my_lens.wavelengths.primary_wavelength.value
            rays_in = RealRays(x=x_in, y=y_in, z=z_in,
                               L=L_in, M=M_in, N=N_in,
                               wavelength=primary_wl, intensity=intensity)
            # Trace the manually created rays
            rays_out = my_lens.surfaces.trace(rays_in)
            # Get x, y coordinates at the image plane (last surface)
            x_image = my_lens.surfaces.x[-1,:]
            y_image = my_lens.surfaces.y[-1,:]

* **Ray Distributions** (``distribution.py``): Specify pupil distribution (e.g., ``'hexapolar'``, ``'uniform'``, ``'random'``).

Analysis Tools
--------------

Optiland offers a suite of tools to evaluate performance:

* ``Aberrations``: Seidel & chromatic. (``my_lens.aberrations.seidels()``)
* ``SpotDiagram``: Geometric ray spread.
* ``RayFan``: Transverse ray aberrations.
* ``OPD``: Wavefront errors.
* ``MTF``: Image contrast vs. frequency.
* ``PSF``: Point spread function.
* ``FieldCurvature``, ``Distortion``: Field performance.
* *(Many classes have a ``.view()`` method for plotting)*.

See the :ref:`Example Gallery <example_gallery>` for a full overview of available analysis tools and their usage.

Visualization
-------------

* **2D Layout** (``optic.draw()``):

    .. code-block:: python

        my_lens.draw(num_rays=5, distribution='line_y')

* **3D Layout** (``optic.draw3D()``):

    .. code-block:: python

        my_lens.draw3D(num_rays=24, distribution='ring')

* **Lens Data Table** (``optic.info()``): Prints surface data in a tabular format, resembling the commonly found Lens Data Editor (LDE).

    .. code-block:: python

        my_lens.info()

GUI Keyboard Shortcuts
----------------------

* **Ctrl+K**: Open the VS Code-style **Command Palette** for quick access to actions and tools.
* **1 / 2**: Load layout presets 1 and 2, respectively.

Advanced Features (Brief Overview)
----------------------------------

* **Coatings** (``coatings.py``): Model anti-reflection or reflective coatings (``SimpleCoating``, ``FresnelCoating``).
* **Thin Films** (``thin_film/*``): Define and optimize multilayer thin-film stacks, including tolerancing and Needle Synthesis.
* **Polarization** (``polarized_rays.py``, ``jones.py``): Trace polarized light and apply Jones calculus for polarizing elements.
* **Pickups** (``pickup.py``): Link a parameter of one surface to another (e.g., make radius of S2 = -radius of S1).
* **Solves** (``solves``): Automatically adjust parameters to meet certain conditions (e.g., ``QuickFocusSolve`` adjusts image plane for best focus).
* **Optimization** (``optimization/*``): Define merit functions with operands and variables to optimize system designs.
* **Tolerancing** (``tolerancing/*``): Analyze the impact of manufacturing errors using sensitivity analysis and Monte Carlo simulations.

This cheat sheet should provide a solid starting point. Happy designing! ✨