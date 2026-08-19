.. _getting_started:

Getting Starting with the Codebase
===================================

This section will help you explore the Optiland codebase and run your first simulation. 

Exploring the Codebase
----------------------

Optiland is designed with modularity in mind. Here are the key components you’ll want to explore first:

- **optic**: Defines the core `Optic` class for building optical systems.
- **rays**: Defines ray objects and and ray generators for simulating light propagation.
- **surfaces**: The building blocks for defining optical elements like lenses and mirrors.
- **optimization**: Tools for optimizing optical systems based on user-defined objectives and variables.

We recommend starting with the **optic** module, as it provides the primary interface for defining optical systems.
You may also explore the **analysis** module to see how simulations are generally performed.

The code is thoroughly documented, so use the inline comments and function/class docstrings to understand how
each module works.

Running Your First Simulation
-----------------------------

Here’s a simple example to get you started. This script simulates a basic singlet lens,
visualizes the system, and traces some rays.

.. code-block:: python

    import numpy as np
    from optiland import optic

    # Initialize an optic system
    system = optic.Optic()

    # add surfaces to the system
    system.surfaces.add(index=0, thickness=np.inf)
    system.surfaces.add(index=1, thickness=7, radius=20.0, is_stop=True, material='N-SF11')
    system.surfaces.add(index=2, thickness=23.0)
    system.surfaces.add(index=3)

    # add aperture
    system.set_aperture(aperture_type='EPD', value=20)

    # add field
    system.fields.set_type(field_type='angle')
    system.fields.add(y=0)

    # add wavelength
    system.wavelengths.add(value=0.587, is_primary=True)

    # draw the system in 2D
    system.draw(num_rays=5)

    # trace some rays (~32x32 rays total)
    rays = system.trace(Hx=0, Hy=0, wavelength=0.587, num_rays=32, distribution='uniform')

    # print the first 10 ray y intersection points on the image plane
    print(rays.y[:10])

This script demonstrates three important aspects of Optiland:

1. **Defining an optical system**: The `Optic` object is the starting point for all simulations. Aperture, field, and wavelength are set using the `set_aperture`, `set_field_type`, `add_field` and `add_wavelength` methods.
2. **Visualization**: Calling the `draw` method generates a 2D plot of the system.
3. **Tracing rays**: Calling the `trace` method generates rays and traces them through the system. This returns the rays at the image plane.

Save this script as `example_simulation.py` and run it using:

.. code-block:: bash

   python example_simulation.py

Alternatively, you can run the script in an interactive Python environment like Jupyter Notebook.

You should see a 2D plot of rays passing through the lens. Experiment with different parameters (e.g., radius of curvature, conic) to observe how the system behaves.

Next Steps
----------

Once you’ve run the example simulation, explore the following areas:

- **Modify optical systems**: Add more lenses or mirrors to build complex systems.
- **Use the analysis tools**: Evaluate system performance with tools like spot diagrams or wavefront analysis.
- **Experiment with optimization**: Learn how to optimize lens thickness, curvature, and other parameters for specific design goals.

For more examples, see the :ref:`example_gallery` or learn Optiland step-by-step with the :ref:`learning_guide`.
