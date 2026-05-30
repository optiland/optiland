Visualization Framework
=======================

The visualization framework in Optiland provides robust tools for both **2D** and **3D** visualization of optical systems,
allowing users to analyze system components, ray paths, and overall design layouts interactively. It is modular,
extensible, and designed for easy integration into workflows.

Overview
--------

The framework supports:

- **2D Visualization**: Built with Matplotlib, this mode is ideal for quick visual representations of optical systems and ray paths.
- **3D Visualization**: Powered by VTK, this mode provides detailed, interactive views of optical systems in three dimensions. A dark mode is available for enhanced viewing aesthetics.

Core Classes
------------

The visualization framework is centered around the following key classes:

- **OpticViewer**: Handles 2D visualization of an `Optic` instance and its components. This is the primary class for generating 2D plots.

- **OpticViewer3D**: Handles 3D visualization of an `Optic` instance and its components, enabling interactive exploration of the system layout in three dimensions.

- **InteractionManager**: Manages all interactive features in 2D Matplotlib plots, such as hover-over tooltips for optical components.

- **themes** and **palettes**: Modules that define the styling system. `themes.py` contains the core logic for applying styles, while `palettes.py` defines the color schemes.

Theming and Palettes
--------------------

A key feature of the 2D visualization framework is its powerful theming engine, which allows for easy customization of plot aesthetics. The system is managed by two core modules:

- **palettes.py**: Defines various color palettes (e.g., 'light', 'dark', 'solarized_light'). Each palette is a dictionary specifying colors for different plot elements like lenses, rays, and backgrounds.
- **themes.py**: Contains the `set_theme()` and `theme_context()` functions that apply a chosen palette to the Matplotlib rcParams, ensuring all subsequent plots follow the specified style.

This architecture makes it simple to create visually consistent and publication-ready figures.

Interactive Pop-Up Info
-----------------------

The 2D visualization is now fully interactive. The **InteractionManager** class connects to the Matplotlib figure's event loop and listens for mouse-over events. When the cursor hovers over a plotted artist (like a surface or a ray bundle), the manager identifies the corresponding optical component and displays a pop-up annotation with relevant information.

This feature is enabled by default in `OpticViewer` and provides a powerful way to inspect system properties directly from the plot.

- **OpticalSystem**: Orchestrates the plotting of system components and rays. It identifies system elements such as lenses and mirrors and delegates rendering to specialized component classes.

- **Rays2D and Rays3D**: Responsible for plotting ray paths in 2D and 3D, respectively. These classes are called by the `OpticalSystem` to handle ray visualization.

Component Classes
-----------------

Each optical component, such as lenses or mirrors, has specialized classes for 2D and 3D visualization:

- **Lens2D** and **Lens3D**: Deal with rendering lenses in their respective projections, taking into account geometry, thickness, and positioning, etc.

- **Surface2D** and **Surface3D**: Handle rendering of surfaces or mirrors, accounting for their orientations and coordinate systems.

Workflow
--------

The typical workflow for using the visualization framework is as follows:

1. **Initialize Viewer**: Create an instance of `OpticViewer` or `OpticViewer3D` and provide an `Optic` instance.

2. **Customize if Needed**: While users typically interact with high-level viewers, the framework supports fine-grained customization by extending component classes or altering plotting parameters.

3. **Render the Plot**: Generate the desired visualization, whether in 2D or 3D, using the `view` method. For 3D plots, additional options like dark mode can be applied for enhanced aesthetics.

Dark Mode for VTK
-----------------

The `OpticViewer3D` class includes an optional **dark mode** for 3D plots, offering a modern and visually appealing interface for exploring optical systems.

.. note::
   The image in the :ref:`developers_guide` shows an example of the dark mode 3D visualization output for a lithographic lens system.

How to Extend This
------------------

**Scenario:** Add a new 2D or 3D renderer component to Optiland.

**For 2D (Matplotlib):**

**Step 1:** Create a component class in ``optiland/visualization/`` that accepts an ``Optic``
instance and a Matplotlib ``Axes`` object.
**Step 2:** Implement a ``draw()`` method that adds Matplotlib artists to the axes.
**Step 3:** Integrate with ``OpticViewer`` in ``optiland/visualization/optic_viewer.py`` if the
component should appear in the standard ``lens.draw()`` output.

**For 3D (VTK):**

**Step 1:** Create a component class that generates VTK actors (``vtkActor`` or ``vtkAssembly``).
**Step 2:** Integrate with ``OpticViewer3D`` in ``optiland/visualization/optic_viewer_3d.py``.

**Step 3:** Add tests in ``tests/test_visualization/``.

For step-by-step guidance, see :ref:`extension_recipes`.
