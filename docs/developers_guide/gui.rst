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

    GUI ray sampling lives in ``services/layout_rays.py``. ``layout_pupil``
    returns pupil coordinates and an explicit chief index; the chief is sampled
    at normalized pupil coordinates ``Px = Py = 0`` for each field, using the
    configured aiming and polarization at the primary wavelength. GUI totals
    include the chief. ``RayCountSpinBox`` in ``widgets/ray_count.py`` steps
    through supported totals and normalizes typed values. In particular,
    hexapolar GUI totals count rays, while the helper translates them to the
    ring count expected by the existing Python distribution API. The
    :ref:`GUI quickstart <gui_quickstart>` describes the user-facing counts.

    ``trace_layout_rays`` retains chief identity alongside each physical path,
    including paths terminated by blocking. When only the chief is displayed,
    four additional, undisplayed rim probes size automatic lens outlines.
    Physical apertures and the schematic image marker retain their existing
    precedence; these probes do not alter the prescription or displayed count.

    The 2D presenter retains chief artists so ``style_chief_rays`` can change
    their color, stroke width and drawing order without submitting a calculation
    or changing the view limits. This styling belongs to the System Viewer;
    other consumers of the shared presenter need no chief-ray checkbox. The
    3D preparation includes the chief alongside 24 rim rays per field and creates
    ordinary depth-tested ray actors, with no chief-specific foreground overlay.

*   **Analysis Panel (`analysis_panel.py`)**:

    *   A comprehensive panel for running various optical analyses.
    *   Users can select an analysis type (e.g., Spot Diagram, Ray Fan, MTF).
    *   Dynamically generates a settings UI based on the selected analysis's parameters.
    *   Executes the analysis using the `OptilandConnector` to get the current `Optic` object.
    *   Displays results, typically as Matplotlib plots, in an embedded canvas.
    *   Supports multiple analysis "pages" (tabs or similar) and cloning of analyses.
    *   Includes features like saving/loading analysis settings.

*   **Optimization Panel (`optimization_panel.py`)**:

    *   Provides Variables, Operands, and Optimizer tabs for building an `OptimizationProblem`
        against the current `Optic` instance.
    *   Lets users add/edit variables and operands (with dialogs for adding and auto-generating
        operands), choose an optimizer and its settings, and run/stop the optimization with a
        persistent live-updating merit-function view.

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

*   **Toast Notifications (``widgets/toast.py``)**:

    *   ``MainWindow`` owns a ``ToastManager``, also exposed as
        ``connector.toast_manager``. Call ``notify(message, severity)`` for
        information, success, warning, or error feedback.
    *   Cards can be dismissed by clicking them or their accessible close button.
        Non-error cards expire after seven seconds; errors persist until dismissed
        or evicted from the three-card stack.
    *   Keep these child widgets free of ``WA_TranslucentBackground``. Alongside
        native console/VTK widgets on Windows, this flag creates a layered native
        child window whose mouse hit testing can select the underlying console.

Styling and Resources
~~~~~~~~~~~~~~~~~~~~~~~~

*   **Qt StyleSheets (`.qss`)**: Themes (e.g., `dark_theme.qss`, `light_theme.qss`) are defined using QSS, Qt's CSS-like styling language. These are located in `optiland_gui/resources/styles/`.
*   **Resource Files (`resources.qrc`, `resources_rc.py`)**: Icons and other assets are managed using Qt's resource system. `resources.qrc` is an XML file defining resources, which is compiled into `resources_rc.py` using `pyside6-rcc`. This allows resources to be bundled with the application.
*   **Plot Styling (`gui_plot_utils.py`)**: Contains utility functions to apply consistent Matplotlib styles that match the selected GUI theme.
*   **Tab Placement (``config.DEFAULT_TAB_POSITION``)**: The shared default is
    ``QTabWidget.North``. Main-window and nested terminal dock groups apply it
    to every dock area; viewer, script, optimization, and surface-property tabs
    use the same default. Future settings should configure this shared policy.
    Review orientation-dependent theme borders and the surface-properties
    close-button corner before offering other tab positions.

Running the GUI
---------------

Once Optiland is installed, you can launch the GUI by simply typing the following command in your terminal or console:

.. code-block:: bash

   optiland

This command is a convenient shortcut to the main GUI script. Alternatively, you can run the GUI module directly using Python's `-m` flag, which can be useful for development:

.. code-block:: bash

   python -m optiland_gui.run_gui

Testing Notification Dismissal
------------------------------

Run ``python -m pytest tests/gui/test_toast.py`` for card and close-button clicks,
keyboard activation, timeouts, and stack cleanup. On Windows, run with the native
Qt platform (leave ``QT_QPA_PLATFORM`` unset) to include the native hit-testing
regressions. These cases are skipped on other platforms and with ``offscreen``.

The Windows regressions use ``WindowFromPoint`` to select the actual native
mouse recipient over a notification above a native text editor, then send input
to that window. Sending a test click directly to a label, or selecting it with
``QApplication.widgetAt``, bypasses layered-window hit testing and does not
reproduce clicks passing through to the console.

Shared 2D plot navigation
-------------------------

``widgets.plot_navigation.PlotNavigationToolbar`` owns the shared 2D mouse
gestures for lens layout, sag, analysis and optimization preview canvases.
Right/middle drag uses Matplotlib's frozen press-time pan transform; wheel zoom
uses each axis's scale transform so logarithmic and inverted axes behave
consistently. Ordinary left-click remains available to plot selection handlers.
Explicit toolbar tools use left-drag. A gesture holds the canvas widget lock
and records a completed view in navigation history; focus loss, Escape, scene
replacement and tool changes end active gestures.

Call ``finish_navigation()`` before replacing an existing scene, or ``update()``
when the replacement also requires clearing view history. Do not attach an
additional scroll or default-pan handler to a canvas using this toolbar.
``test_plot_navigation.py`` exercises the shared contract, while
``test_viewer_navigation.py`` checks real lens-worker integration.

The rectangle-zoom glyph combines a dotted selection box and magnifier, with
an explicit accessible name and tooltip. It follows the local foreground
palette and provides native-size pixmaps for standard and HiDPI displays.
Qt owns the finished icon; no Python icon-engine lifetime is exposed to Qt.

.. _gui-atomic-output:

Background saves and atomic publication
---------------------------------------

The GUI output service separates preparing bytes from publishing a completed
file. ``services.output_tasks.prepare_output`` writes an exclusively created
sibling staging file and returns its identity and content digest. After the
coordinator accepts publication, ``publish_output`` checks that receipt and
calls ``os.replace``. Preparation leaves the existing destination untouched;
the replacement publishes the completed file without exposing a partially
copied destination. This contract applies to the background GUI save/export
workflow, not to every path-based core writer.

Why stage beside the destination?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Atomic rename/replacement requires source and destination to be on the same
filesystem. A system temporary directory such as ``/tmp`` may be a separate
mount, an in-memory filesystem, or on a different volume from the design file.
Linux rejects a cross-filesystem rename with ``EXDEV``. Staging beside the
destination avoids that boundary; using ``shutil.move`` as a substitute could
turn the operation into a copy followed by deletion and lose atomicity.

Even if a temporary directory happens to share the destination filesystem,
replacing the destination still changes its directory entry. On Linux this
requires write and search permissions on the relevant directories; write
permission on the existing file alone is not sufficient. Other platforms have
their own ACL, sharing and replacement rules. Choosing a different staging
directory does not bypass the permissions required to publish the replacement.

Writing into the existing file is a different operation. On Linux, an existing
file can be writable even when its directory permits traversal but not creating
or replacing entries. Preparing output in ``/tmp`` and then copying it into
that file can support this case. However, opening it for overwrite can truncate
the old contents, and an interrupted copy, full disk or other write error can
leave only part of the new contents. Finishing serialization before the copy
does not make the copy atomic or provide rollback.

For that reason the GUI does not silently fall back to an in-place write. It
reports the failure and keeps the document available for **Save As** or export
to a writable location. Cancellation may discard preparation, but accepted
publication is non-cancellable. Atomic replacement describes visibility of the
file change; it is not by itself a power-loss durability guarantee. The current
implementation does not explicitly synchronize file contents and the containing
directory to durable storage.

The temporary-file API and its location are separate choices. Python's
``tempfile`` APIs accept a ``dir`` argument, so using a standard temporary-file
helper would not require using the system temporary directory. The current
worker reserves its operation-specific path with exclusive creation and tracks
ownership for subsequent cleanup. Replacement also affects symlinks and file
metadata as described in :doc:`../gui_quickstart`.

See the `Linux rename documentation
<https://man7.org/linux/man-pages/man2/rename.2.html>`_,
`Python os.replace documentation
<https://docs.python.org/3/library/os.html#os.replace>`_ and
`Python temporary-file documentation
<https://docs.python.org/3/library/tempfile.html>`_ for the underlying contracts.

Could QSaveFile replace the staging implementation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, at the file-transaction layer. ``PySide6.QtCore.QSaveFile`` owns temporary
output, remembers write errors, discards uncommitted output during normal
cleanup, and publishes it through ``commit()``. Its default direct-write
fallback is disabled. Keeping ``setDirectWriteFallback(False)`` would preserve
the decision to fail when atomic publication is unavailable. It uses sibling
staging too, so it does not remove filesystem or directory-permission limits.

Qt documents an optional direct-write fallback for editable files in restricted
directories, but that mode sacrifices atomicity and makes ``cancelWriting()``
ineffective. Enabling it would be an explicit product-policy change, not an
implementation-equivalent replacement. See `QSaveFile
<https://doc.qt.io/qt-6/qsavefile.html>`_.

Adopting it still requires integration work:

* File I/O remains synchronous. Create, write and commit the object in a worker,
  rather than moving these operations onto the GUI thread.
* Keep document snapshot ownership, same-destination save ordering, cancellation
  boundaries, and the checks that prevent older saves marking newer edits clean.
  Qt's file transaction does not know those application rules.
* The current protocol finishes preparation and later submits a separate
  publication job using a path, identity and digest. A live ``QSaveFile`` is
  process-local state, not that transferable receipt. Retaining it in a worker
  across the two phases would need explicit lifetime, abort and shutdown rules.
  A worker dying after commit but before acknowledgement still leaves an outcome
  that the coordinator must reconcile.
* Native JSON can be encoded into bytes and written to the Qt device. The
  existing Zemax and CODE V save functions instead take paths and open their
  own files. ``QSaveFile.fileName()`` reports the destination, not a staging path
  to hand to those functions. Direct device output therefore needs an adapter
  or a separate core change exposing encoded output or a writable stream.
  Copying an existing staged export through ``QSaveFile`` is possible but adds
  another temporary file and copy.

``QSaveFile`` is consequently a candidate for simplifying low-level file
handling, not a replacement for the output coordinator. A refactor should
preserve the existing ordering, failure, cancellation and uncertain-outcome
tests, and explicitly compare symlink and metadata behavior before changing
the documented contract. This guide describes that option; the current worker
still uses sibling files and ``os.replace``.

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
