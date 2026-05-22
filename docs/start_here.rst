.. _start_here:

Start Here
==========

**Optiland** is a Python framework for optical design, analysis, and optimization. It covers everything
from basic paraxial layouts to GPU-accelerated differentiable ray tracing. Whether you are tracing your
first ray or training a differentiable lens model, this page will route you to the right starting point.

.. rubric:: Choose Your Path

----

Optics Student / First-Timer
-----------------------------

*Goal: understand Optiland basics, trace rays, and visualize a lens system from scratch.*

You know the fundamentals of optics (lenses, focal lengths, rays) but are new to Optiland or to
programmatic optical design. Start here to build your first lens in Python.

**Recommended path:**

1. :doc:`installation` — install Optiland with ``pip install optiland``
2. `Tutorial 1a — Optiland for Beginners <examples/Tutorial_1a_Optiland_for_Beginners.html>`_ — build and visualize your first lens
3. `Tutorial 1b — Lens Properties <examples/Tutorial_1b_Lens_Properties.html>`_ — paraxial properties and surface data
4. `Tutorial 3a — Common Aberration Analyses <examples/Tutorial_3a_Common_Aberration_Analyses.html>`_ — spot diagrams, ray fans, and wavefront errors

**→** `Start with Tutorial 1a <examples/Tutorial_1a_Optiland_for_Beginners.html>`_

----

Optical Engineer (Practitioner)
---------------------------------

*Goal: get productive fast, import existing designs, and run professional analyses.*

You are migrating from Zemax, CODE V, or OSLO and want to reproduce your existing designs or leverage
Optiland's optimization and tolerancing workflows quickly.

**Recommended path:**

1. :doc:`quickstart` — a complete 5-minute tour from install to optimization
2. :doc:`cheat_sheet` — copy-paste snippets for the 20 most common tasks
3. `Tutorial 9a — Edmund Optics Catalogue <examples/Tutorial_9a_Edmund_Optics_Catalogue.html>`_ — import off-the-shelf catalog lenses
4. `Tutorial 5c — Optimization Case Study (Cooke Triplet) <examples/Tutorial_5c_Optimization_Case_Study.html>`_ — full optimization workflow

**→** `Go to the Quickstart <quickstart.html>`_

----

Computational Researcher
--------------------------

*Goal: use the PyTorch backend for autograd, differentiable optimization, and ML pipelines.*

You are working on differentiable optics, end-to-end training of optical systems, or integrating
Optiland into a PyTorch-based research pipeline.

**Recommended path:**

1. `Tutorial 1f — Differentiable Ray Tracing Hello World <examples/Tutorial_1f_Differentiable_Ray_Tracing_Hello_World.html>`_ — switch to PyTorch and compute gradients
2. `Tutorial 5e — Differentiable Lens Optimization <examples/Tutorial_5e_Differentiable_Optimization.html>`_ — gradient-descent optimization with autograd
3. :ref:`configurable_backend` — backend architecture, device management, and precision control
4. `Tutorial 5b — Advanced Optimization <examples/Tutorial_5b_Advanced_Optimization.html>`_ — multi-operand merit functions and advanced solvers

**→** `Start with Tutorial 1f <examples/Tutorial_1f_Differentiable_Ray_Tracing_Hello_World.html>`_

----

Software Contributor / Extender
---------------------------------

*Goal: add new surface types, analysis classes, or operands; understand the codebase architecture.*

You want to extend Optiland with custom components, integrate it into a larger system, or contribute
a new feature back to the project.

**Recommended path:**

1. :doc:`developers_guide/architecture` — high-level codebase map and key design decisions
2. :doc:`developers_guide/extension_recipes` — step-by-step recipes for the 8 most common extension scenarios
3. `Tutorial 10a — Custom Surface Types <examples/Tutorial_10a_Custom_Surface_Types.html>`_ — add a new geometry class end-to-end
4. `Tutorial 10b — Custom Coating Types <examples/Tutorial_10b_Custom_Coating_Types.html>`_ — add a custom coating interaction model

**→** `Read the Developer's Guide <developers_guide/introduction.html>`_
