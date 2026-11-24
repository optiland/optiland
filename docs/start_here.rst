.. _start_here:

.. meta::
   :description: Suggested reading orders through the Optiland documentation for students, optical engineers, computational researchers and contributors.

Start Here
==========

**Optiland** is a Python package for optical design, analysis and
optimization. It covers a lot of ground, from paraxial layouts to
GPU-accelerated differentiable ray tracing, so there is no single right place
to begin. This page suggests a reading order based on what you came here to do.
Pick the path that sounds most like you. They overlap, and nothing stops you
from switching halfway.

.. note::

   Whichever path you pick, it is worth reading :doc:`conventions` early. It is
   one page, and it explains the thing that trips up most newcomers: a
   surface's ``thickness`` and ``material`` describe the space *after* that
   surface.

   Already have a concrete question? :doc:`how_do_i` maps questions such as "how
   do I compute the MTF?" straight to the example that answers them.

.. rubric:: Choose your path

----

.. _optics-student-first-timer:

Optics student or first-time user
---------------------------------

*Goal: understand the basics, trace some rays and see what a lens is doing.*

You know some optics (lenses, focal lengths, rays) but are new to Optiland, or
to doing optical design in code. This path builds one idea at a time, and every
notebook runs on its own.

1. :doc:`installation`: ``pip install optiland`` is usually all you need.
2. :doc:`Tutorial 1a: Optiland for Beginners <examples/Tutorial_1a_Optiland_for_Beginners>`:
   build your first lens and draw it.
3. :doc:`Tutorial 1b: Lens Properties and Prescription <examples/Tutorial_1b_Lens_Properties_and_Prescription>`:
   focal length, f-number, pupils and the surface table.
4. :doc:`Tutorial 2a: Tracing and Analyzing Rays <examples/Tutorial_2a_Tracing_and_Analyzing_Rays>`:
   what a ray trace actually returns.
5. :doc:`Tutorial 2c: Aberration Analyses <examples/Tutorial_2c_Aberration_Analyses>`:
   spot diagrams, ray fans and wavefront errors, and how to read them.
6. :doc:`Tutorial 1e: Design a Doublet, End to End <examples/Tutorial_1e_Design_a_Doublet_End_to_End>`:
   put it all together, from a set of requirements to a finished prescription.
7. :doc:`Tutorial 3a: Simple Optimization <examples/Tutorial_3a_Simple_Optimization>`:
   let the computer improve the design for you.

**→** :doc:`Start with Tutorial 1a <examples/Tutorial_1a_Optiland_for_Beginners>`

----

.. _optical-engineer-practitioner:

Optical engineer
----------------

*Goal: get productive quickly, bring in existing designs and run the analyses
you already know.*

You have used Zemax OpticStudio, CODE V or OSLO and want to know how the same
work looks in Optiland. :doc:`coming_from_other_tools` gives the Optiland name
for each OpticStudio and CODE V term.

1. :doc:`quickstart`: build, trace, analyze, optimize and save in five
   minutes.
2. :doc:`conventions`: how the Optiland lens model maps onto what you are used
   to.
3. :doc:`cheat_sheet`: copy-paste snippets for the most common tasks.
4. :doc:`zemax_import` (or :doc:`oslo_import`): open your existing ``.zmx``
   or ``.len`` files.
5. :doc:`Tutorial 4d: Lens Catalogue Integration <examples/Tutorial_4d_Lens_Catalogue_Integration>`:
   drop in off-the-shelf vendor lenses.
6. :doc:`Tutorial 3d: Optimization Case Study (Cooke Triplet) <examples/Tutorial_3d_Optimization_Case_Study_Cooke_Triplet>`:
   a complete optimization workflow.
7. :doc:`Tutorial 6a: Tolerancing Sensitivity Analysis <examples/Tutorial_6a_Tolerancing_Sensitivity_Analysis>`:
   sensitivity and Monte Carlo tolerancing.

Curious how the numbers compare with commercial tools? :doc:`validation`
puts first-order properties, aberrations and real rays side by side with
OpticStudio and CODE V.

**→** :doc:`Go to the Quickstart <quickstart>`

----

.. _computational-researcher:

Computational researcher
------------------------

*Goal: use the PyTorch backend for autograd, differentiable optimization and
machine-learning pipelines.*

You work on differentiable optics or end-to-end design, or you want to put
Optiland inside a PyTorch training loop.

1. :doc:`installation`: install with the ``torch`` extra, or with a CUDA build
   of PyTorch for GPU work.
2. :doc:`Tutorial 7a: Differentiable Ray Tracing Hello World <examples/Tutorial_7a_Differentiable_Ray_Tracing_Hello_World>`:
   switch to the PyTorch backend and compute your first gradients.
3. :doc:`Tutorial 7b: Differentiable Lens Optimization <examples/Tutorial_7b_Differentiable_Lens_Optimization>`:
   gradient-based lens optimization with autograd.
4. :ref:`configurable_backend`: how the backend works, plus device and
   precision control.
5. :doc:`Machine learning tutorials (chapter 9) <learning_guide>`: surrogates,
   classifiers, GANs and reinforcement learning built on top of Optiland.

**→** :doc:`Start with Tutorial 7a <examples/Tutorial_7a_Differentiable_Ray_Tracing_Hello_World>`

----

.. _software-contributor-extender:

Contributor or extender
-----------------------

*Goal: add a surface type, analysis or operand, or understand how the code
fits together.*

You want to extend Optiland for your own work, or contribute something back
to the project. Contributions of every size are welcome, typo fixes included.

1. :doc:`contributing`: how to set up a development environment and open a
   pull request.
2. :doc:`developers_guide/architecture`: a map of the codebase and the main
   design decisions.
3. :doc:`developers_guide/extension_recipes`: step-by-step recipes for the
   most common extensions.
4. :doc:`Tutorial 8a: Custom Surface Types <examples/Tutorial_8a_Custom_Surface_Types>`:
   add a new geometry class, end to end.
5. :doc:`Tutorial 8b: Custom Coating Types <examples/Tutorial_8b_Custom_Coating_Types>`:
   add a custom coating.

**→** :doc:`Read the Developer's Guide <developers_guide/introduction>`
