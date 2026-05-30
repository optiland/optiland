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
2. :doc:`Tutorial 1a — Optiland for Beginners <examples/Tutorial_1a_Optiland_for_Beginners>` — build and visualize your first lens
3. :doc:`Tutorial 1b — Lens Properties & Prescription <examples/Tutorial_1b_Lens_Properties_and_Prescription>` — paraxial properties and surface data
4. :doc:`Tutorial 2c — Aberration Analyses <examples/Tutorial_2c_Aberration_Analyses>` — spot diagrams, ray fans, and wavefront errors

**→** :doc:`Start with Tutorial 1a <examples/Tutorial_1a_Optiland_for_Beginners>`

----

Optical Engineer (Practitioner)
---------------------------------

*Goal: get productive fast, import existing designs, and run professional analyses.*

You are migrating from Zemax, CODE V, or OSLO and want to reproduce your existing designs or leverage
Optiland's optimization and tolerancing workflows quickly.

**Recommended path:**

1. :doc:`quickstart` — a complete 5-minute tour from install to optimization
2. :doc:`cheat_sheet` — copy-paste snippets for the 20 most common tasks
3. :doc:`Tutorial 4d — Lens Catalogue Integration <examples/Tutorial_4d_Lens_Catalogue_Integration>` — import off-the-shelf catalog lenses
4. :doc:`Tutorial 3d — Optimization Case Study (Cooke Triplet) <examples/Tutorial_3d_Optimization_Case_Study_Cooke_Triplet>` — full optimization workflow

**→** :doc:`Go to the Quickstart <quickstart>`

----

Computational Researcher
--------------------------

*Goal: use the PyTorch backend for autograd, differentiable optimization, and ML pipelines.*

You are working on differentiable optics, end-to-end training of optical systems, or integrating
Optiland into a PyTorch-based research pipeline.

**Recommended path:**

1. :doc:`Tutorial 7a — Differentiable Ray Tracing Hello World <examples/Tutorial_7a_Differentiable_Ray_Tracing_Hello_World>` — switch to PyTorch and compute gradients
2. :doc:`Tutorial 7b — Differentiable Lens Optimization <examples/Tutorial_7b_Differentiable_Lens_Optimization>` — gradient-descent optimization with autograd
3. :ref:`configurable_backend` — backend architecture, device management, and precision control
4. :doc:`Tutorial 3b — Advanced Optimization <examples/Tutorial_3b_Advanced_Optimization>` — multi-operand merit functions and advanced solvers

**→** :doc:`Start with Tutorial 7a <examples/Tutorial_7a_Differentiable_Ray_Tracing_Hello_World>`

----

Software Contributor / Extender
---------------------------------

*Goal: add new surface types, analysis classes, or operands; understand the codebase architecture.*

You want to extend Optiland with custom components, integrate it into a larger system, or contribute
a new feature back to the project.

**Recommended path:**

1. :doc:`developers_guide/architecture` — high-level codebase map and key design decisions
2. :doc:`developers_guide/extension_recipes` — step-by-step recipes for the 8 most common extension scenarios
3. :doc:`Tutorial 8a — Custom Surface Types <examples/Tutorial_8a_Custom_Surface_Types>` — add a new geometry class end-to-end
4. :doc:`Tutorial 8b — Custom Coating Types <examples/Tutorial_8b_Custom_Coating_Types>` — add a custom coating interaction model

**→** :doc:`Read the Developer's Guide <developers_guide/introduction>`
