.. _developers_guide_gui:

Optiland GUI
============

This document provides an overview of the Optiland Graphical User Interface (GUI), its components, architecture, and guidelines for developers looking to contribute to its development.

Overview
--------

The Optiland GUI, built with PySide6, provides an interactive way to design, analyze, and manage optical systems using the Optiland backend. It aims to offer a user-friendly experience while exposing the powerful features of the core library. The GUI is designed to be modular, allowing for the addition of new panels and functionalities.

Key Technologies
----------------

*   **PySide6**: The official Python bindings for Qt, used for creating the user interface.
*   **Matplotlib**: Used for 2D plotting in various panels (e.g., 2D system view, analysis plots).
*   **VTK (Visualization Toolkit)**: Used for 3D rendering of optical systems in the ViewerPanel. (Optional, GUI can run without it).

Core Architecture
-----------------

The GUI's architecture revolves around a main window that hosts several dockable panels, each dedicated to a specific aspect of optical design or analysis. The `OptilandConnector` class serves as a crucial intermediary, linking these GUI components to the underlying `Optic` object from the Optiland backend.

Key Components
--------------

Main Window (`main_window.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `MainWindow` class is the entry point and central hub of the GUI. It uses an `ActionManager` to create and manage `QAction` instances, and a `PanelManager` to handle the creation and layout of all dockable panels. Its responsibilities include:

*   **Menu Bar and Toolbars**: Provides standard application menus (File, Edit, View, Help) and quick-action toolbars.
*   **Theme Handling**: Supports light and dark themes, loading stylesheets (`.qss` files) and propagating theme changes to child widgets and plots.
*   **Window Control**: Implements a custom title bar for a modern look and feel, handling minimize, maximize, and close operations.
*   **Settings Persistence**: Uses `QSettings` to save and load window geometry and layout configurations.
*   **Dialogs**: Manages application-level dialogs like "About" and file open/save.

Optiland Connector (`optiland_connector.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `OptilandConnector` is a vital QObject that acts as a bridge between the GUI and the Optiland backend.

*   **`Optic` Object Management**: It holds and manages the instance of the `Optic` class, which represents the current optical system.
*   **Signal Emission**: Emits Qt signals when the optical system is loaded (`opticLoaded`), modified (`opticChanged`), or when specific parts like surface data or count change. GUI panels connect to these signals to refresh their views.
*   **Data Access and Modification**: Provides methods for GUI components to safely access data from the `Optic` object (e.g., `get_surface_data`, `get_column_headers`) and to modify it (e.g., `set_surface_data`, `add_surface`).
*   **Undo/Redo**: Delegates undo and redo operations to an `UndoRedoManager` instance.
*   **File Operations**: Handles loading and saving of optical systems in Optiland's JSON format.

Core Panels
~~~~~~~~~~~~~~

These are the primary QDockWidget or QWidget instances that provide specific functionalities:

*   **Lens Editor (`lens_editor.py`)**:

    *   Displays optical system data in a table format (similar to a lens data editor in commercial software).
    *   Allows users to view and edit surface parameters like radius, thickness, material, conic constant, and semi-diameter.
    *   Interacts with `OptilandConnector` to fetch data and commit changes, triggering updates in other relevant panels.
    *   Supports adding and removing surfaces.

*   **Viewer Panel (`viewer_panel.py`)**:

    *   Provides visual representations of the optical system.
    *   **2D View**: Uses Matplotlib to render a 2D cross-section of the system, including lens elements and ray paths. Includes basic navigation (pan/zoom) and a settings area to control ray display.
    *   **3D View**: Uses VTK (if available) to render a 3D model of the system, allowing for interactive rotation, panning, and zooming.
    *   Both views update in response to changes in the `Optic` object via signals from `OptilandConnector`.

*   **Analysis Panel (`analysis_panel.py`)**:

    *   A comprehensive panel for running various optical analyses.
    *   Users can select an analysis type (e.g., Spot Diagram, Ray Fan, MTF).
    *   Dynamically generates a settings UI based on the selected analysis's parameters.
    *   Executes the analysis using the `OptilandConnector` to get the current `Optic` object.
    *   Displays results, typically as Matplotlib plots, in an embedded canvas.
    *   Supports multiple analysis "pages" (tabs or similar) and cloning of analyses.
    *   Includes features like saving/loading analysis settings.

*   **Optimization Panel (`optimization_panel.py`)**:

    *   Currently a placeholder for future optimization functionalities.
    *   Intended to allow users to define optimization variables, objectives (merit functions), and run optimization routines from the Optiland backend.

*   **System Properties Panel (`system_properties_panel.py`)**:

    *   Manages system-wide settings that are not tied to individual surfaces.
    *   Uses a navigation tree to switch between different property editors (Aperture, Fields, Wavelengths, etc.).
    *   **Aperture Editor**: Configures the system aperture (e.g., Entrance Pupil Diameter, F-number).
    *   **Fields Editor**: Defines field points (e.g., angle or object height) and their vignetting factors.
    *   **Wavelengths Editor**: Manages the wavelengths used for analysis and their weights, including setting the primary wavelength.
    *   Changes made here are applied to the `Optic` object through the `OptilandConnector`.

Key Widgets
~~~~~~~~~~~~~~

*   **Sidebar Widget (`widgets/sidebar.py`)**:

    *   Provides main navigation between different functional areas of the GUI (e.g., Design, Analysis, Scripts).
    *   Can be collapsed to save space, showing only icons.
    *   Emits a `menuSelected` signal when a button is clicked, which the `MainWindow` uses to raise or show relevant panels.

*   **Python Terminal (`widgets/python_terminal.py`)**:

    *   An embedded IPython terminal.
    *   Provides direct access to the `OptilandConnector` instance (via the `connector` variable), allowing advanced users to interact with the optical system programmatically.
    *   Commands executed in the terminal can trigger GUI updates if they modify the `Optic` object.

*   **Command Palette (`widgets/command_palette.py`)**:

    *   A floating panel that allows users to quickly search for and execute GUI commands.
    *   Similar to the command palette found in VS Code, it provides a centralized interface for discovering and running various tools, analyses, and layout operations.
    *   Triggered from the `MainWindow` via the `Ctrl+K` shortcut.

Styling and Resources
~~~~~~~~~~~~~~~~~~~~~~~~

*   **Qt StyleSheets (`.qss`)**: Themes (e.g., `dark_theme.qss`, `light_theme.qss`) are defined using QSS, Qt's CSS-like styling language. These are located in `optiland_gui/resources/styles/`.
*   **Resource Files (`resources.qrc`, `resources_rc.py`)**: Icons and other assets are managed using Qt's resource system. `resources.qrc` is an XML file defining resources, which is compiled into `resources_rc.py` using `pyside6-rcc`. This allows resources to be bundled with the application.
*   **Plot Styling (`gui_plot_utils.py`)**: Contains utility functions to apply consistent Matplotlib styles that match the selected GUI theme.

Running the GUI
---------------

Once Optiland is installed, you can launch the GUI by simply typing the following command in your terminal or console:

.. code-block:: bash

   optiland

This command is a convenient shortcut to the main GUI script. Alternatively, you can run the GUI module directly using Python's `-m` flag, which can be useful for development:

.. code-block:: bash

   python -m optiland_gui.run_gui

Contributing to the GUI
-----------------------

Developing for the Optiland GUI generally involves the following:

1.  **Understanding PySide6**: Familiarity with Qt concepts like signals and slots, layouts, widgets, and the event loop is essential.
2.  **Interacting with `OptilandConnector`**:

    *   When creating a new panel that needs to display or modify optical data, it should take an `OptilandConnector` instance in its constructor.
    *   Connect to relevant signals from the `OptilandConnector` (e.g., `opticChanged`, `surfaceDataChanged`) to update the panel's display when the underlying data changes.
    *   Use the connector's methods to fetch data (e.g., `get_optic()`, `get_surface_data()`) and to apply changes (e.g., `set_surface_data()`, or by directly modifying the `Optic` object obtained from `get_optic()` and then calling `connector.opticChanged.emit()` if the connector doesn't automatically detect the change for undo/redo purposes or specific signal emission).
3.  **Designing UI**:

    *   Use Qt Designer (optional) or create UI elements programmatically.
    *   Employ layouts (QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout) for responsive and well-organized UIs.
    *   Follow existing patterns for styling and theming. New widgets should respect the application's theme.
4.  **Undo/Redo**: For actions that modify the optical system, ensure they are compatible with the `UndoRedoManager`. This usually involves capturing the state of the `Optic` object before a change and adding it to the undo stack via `OptilandConnector._undo_redo_manager.add_state(old_optic_state_dict)`.
5.  **Modularity**: Aim to keep panels self-contained and focused on specific functionalities.

**Example Workflow for Adding a New Panel:**

1.  Create a new Python file for your panel (e.g., `my_new_panel.py`).
2.  Define a QWidget or QDockWidget subclass.
3.  In its `__init__`, accept an `OptilandConnector` instance.
4.  Build the UI for your panel.
5.  Connect to signals from `OptilandConnector` to populate/update your panel.
6.  Implement logic to handle user interactions and, if necessary, modify the `Optic` object via the connector.
7.  In `main_window.py`:

    *   Instantiate your new panel.
    *   Add it as a QDockWidget or integrate it into the UI as appropriate.
    *   Optionally, add menu actions or sidebar buttons to control its visibility or interaction.

By following these guidelines and referring to existing panels as examples, developers can effectively contribute to and extend the Optiland GUI.
