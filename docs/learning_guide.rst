.. _learning_guide:
.. _learning_guide_page:

.. meta::
   :description: The Optiland tutorial series: eleven chapters from a first singlet to optimization, tolerancing, differentiable ray tracing, machine learning and non-sequential illumination.

Learning Guide
==============

This is Optiland's main tutorial series. It is organized into chapters that
build on each other, but every notebook is self-contained: you never need to
run an earlier notebook before a later one.

**Where to start.** Chapters 1 to 3 are the core path. Together they cover
how to describe a lens, trace and analyze it, and optimize it, which is enough
to do real design work. Everything after that is an elective: pick the
chapters that match what you are working on. Each chapter lists what you
should know before starting it.

New to Optiland altogether? :doc:`start_here` suggests a reading order for
your background, and :doc:`quickstart` gives you the whole workflow in five
minutes.

.. tip::

   Every tutorial is a Jupyter notebook, and the source files live in
   `docs/examples <https://github.com/optiland/optiland/tree/master/docs/examples>`_
   on GitHub. Download one and run it locally as you read: changing a number
   and re-running a cell is the fastest way to build intuition.


1. Foundational Lens Design :bdg-primary:`core`
-----------------------------------------------

*Prerequisites: some Python and basic optics (what a lens and a focal length
are).*

How to describe a lens in Optiland: surfaces, apertures, fields and
wavelengths, the first-order properties that follow from them, and where
glass data comes from. The chapter ends with a complete doublet design, from
requirements to finished prescription.

.. toctree::
   :maxdepth: 1

   Tutorial 1a: Optiland for Beginners <examples/Tutorial_1a_Optiland_for_Beginners>
   Tutorial 1b: Lens Properties and Prescription <examples/Tutorial_1b_Lens_Properties_and_Prescription>
   Tutorial 1c: Material Database and Catalogs <examples/Tutorial_1c_Material_Database_and_Catalogs>
   Tutorial 1d: Saving and Loading <examples/Tutorial_1d_Saving_and_Loading>
   Tutorial 1e: Design a Doublet, End to End <examples/Tutorial_1e_Design_a_Doublet_End_to_End>
   Tutorial 1f: Paraxial Gaussian Beams and Resonators <examples/Tutorial_1f_Paraxial_Gaussian_Beams>

Interested in propagating a Gaussian beam with diffraction, not only
paraxially? See the :doc:`Gaussian beam propagation example
<examples/gaussian_beam_propagation>`.


2. Real Raytracing & Analysis :bdg-primary:`core`
-------------------------------------------------

*Prerequisites: Tutorial 1a.*

What a real ray trace returns and how to turn it into the analyses optical
designers rely on: spot diagrams, ray fans, aberration coefficients, and, on
the wave-optics side, OPD, PSF and MTF.

.. toctree::
   :maxdepth: 1

   Tutorial 2a: Tracing and Analyzing Rays <examples/Tutorial_2a_Tracing_and_Analyzing_Rays>
   Tutorial 2b: Monte Carlo Raytracing <examples/Tutorial_2b_Monte_Carlo_Raytracing>
   Tutorial 2c: Aberration Analyses <examples/Tutorial_2c_Aberration_Analyses>
   Tutorial 2d: OPD, PSF, and MTF Calculations <examples/Tutorial_2d_OPD_PSF_and_MTF_Calculations>


3. Lens Optimization :bdg-primary:`core`
----------------------------------------

*Prerequisites: chapters 1 and 2.*

Letting the computer improve a design: variables, operands and merit
functions, local and global optimizers, writing your own operands, a full
case study, and optimizing glass choices (both by picking from a catalog and
by treating glass as continuous variables).

.. toctree::
   :maxdepth: 1

   Tutorial 3a: Simple Optimization <examples/Tutorial_3a_Simple_Optimization>
   Tutorial 3b: Advanced Optimization <examples/Tutorial_3b_Advanced_Optimization>
   Tutorial 3c: User-Defined Optimization <examples/Tutorial_3c_User_Defined_Optimization>
   Tutorial 3d: Optimization Case Study: Cooke Triplet <examples/Tutorial_3d_Optimization_Case_Study_Cooke_Triplet>
   Tutorial 3e: Glass Expert Categorical Optimization <examples/Tutorial_3e_Glass_Expert_Categorical_Optimization>
   Tutorial 3f: Constrained Optimization with the Default Merit <examples/Tutorial_3f_Constrained_Default_Merit>
   Tutorial 3g: Model Glass and Continuous Nd/Vd Optimization <examples/Tutorial_3g_Model_Glass_Nd_Vd_Optimization>

The :doc:`optimization_guide` is the companion reference for this chapter.


4. Off-Axis & Complex Systems :bdg-secondary:`elective`
-------------------------------------------------------

*Prerequisites: chapters 1 to 3.*

Systems that are not simple rotationally symmetric lenses: tilted and
decentered elements, aspheres and freeforms, zoom lenses and other
multi-configuration systems, catalog lenses, and two larger worked designs (a
lithographic projection lens and a three-mirror anastigmat).

.. toctree::
   :maxdepth: 1

   Tutorial 4a: Tilts, Decenters, and Asymmetric Systems <examples/Tutorial_4a_Tilts_Decenters_and_Asymmetric_Systems>
   Tutorial 4b: Raytracing Aspheres and Freeforms <examples/Tutorial_4b_Raytracing_Aspheres_and_Freeforms>
   Tutorial 4c: Zoom Lenses and Multi-Configuration <examples/Tutorial_4c_Zoom_Lenses_and_Multi_Configuration>
   Tutorial 4d: Lens Catalogue Integration <examples/Tutorial_4d_Lens_Catalogue_Integration>
   Tutorial 4e: Lithographic Projection System <examples/Tutorial_4e_Lithographic_Projection_System>
   Tutorial 4f: Three-Mirror Anastigmat <examples/Tutorial_4f_Three_Mirror_Anastigmat>


5. Polarization & Coatings :bdg-secondary:`elective`
----------------------------------------------------

*Prerequisites: chapters 1 and 2.*

Light is not just rays with a direction. This chapter covers coatings and
multilayer thin films, polarization ray tracing, and how to design and
tolerance a thin-film stack.

.. toctree::
   :maxdepth: 1

   Tutorial 5a: Coatings and Multilayer Stacks <examples/Tutorial_5a_Coatings_and_Multilayer_Stacks>
   Tutorial 5b: Introduction to Polarization <examples/Tutorial_5b_Introduction_to_Polarization>
   Tutorial 5c: Thin-Film Optimization and Needle Synthesis <examples/Tutorial_5c_Thin_Film_Optimization_and_Needle_Synthesis>
   Tutorial 5d: Advanced Thin-Film Applications <examples/Tutorial_5d_Advanced_Thin_Film_Applications>


6. Tolerancing & Physical Effects :bdg-secondary:`elective`
-----------------------------------------------------------

*Prerequisites: chapters 1 to 3.*

Will the lens still work once it has been built? Sensitivity and Monte Carlo
tolerancing, followed by surface roughness, scattering and extended sources.
Tutorial 6d puts the tolerancing together in a complete tolerance study; the
:doc:`tolerancing_guide` is its reference.

.. toctree::
   :maxdepth: 1

   Tutorial 6a: Tolerancing Sensitivity Analysis <examples/Tutorial_6a_Tolerancing_Sensitivity_Analysis>
   Tutorial 6b: Monte Carlo Tolerancing Analysis <examples/Tutorial_6b_Monte_Carlo_Tolerancing_Analysis>
   Tutorial 6c: Roughness, Scattering, and Extended Sources <examples/Tutorial_6c_Roughness_Scattering_and_Extended_Sources>
   Tutorial 6d: A Complete Tolerance Study <examples/Tutorial_6d_Complete_Tolerance_Study>


7. Differentiable Raytracing :bdg-secondary:`elective`
------------------------------------------------------

*Prerequisites: chapter 1 and Tutorial 3a. Requires PyTorch (*\ ``pip install optiland[torch]``\ *).*

Switch Optiland to its PyTorch backend and every ray trace becomes
differentiable. This chapter computes your first gradients and then builds a
complete gradient-based lens optimization.

.. toctree::
   :maxdepth: 1

   Tutorial 7a: Differentiable Ray Tracing Hello World <examples/Tutorial_7a_Differentiable_Ray_Tracing_Hello_World>
   Tutorial 7b: Differentiable Lens Optimization <examples/Tutorial_7b_Differentiable_Lens_Optimization>


8. Extending Optiland :bdg-secondary:`elective`
-----------------------------------------------

*Prerequisites: chapters 1 to 3 and some comfort with Python classes.*

Optiland is built to be extended. These tutorials add a custom surface
type, a custom coating and a custom optimization algorithm. The
:doc:`developers_guide/extension_recipes` cover more cases in less detail.

.. toctree::
   :maxdepth: 1

   Tutorial 8a: Custom Surface Types <examples/Tutorial_8a_Custom_Surface_Types>
   Tutorial 8b: Custom Coating Types <examples/Tutorial_8b_Custom_Coating_Types>
   Tutorial 8c: Custom Optimization Algorithm <examples/Tutorial_8c_Custom_Optimization_Algorithm>


9. Machine Learning in Optical Design :bdg-secondary:`elective`
---------------------------------------------------------------

*Prerequisites: chapters 1 and 2 and some machine-learning background. Most
notebooks need scikit-learn or PyTorch.*

Optiland as a data generator and a building block for machine learning:
regression and classification on ray-trace data, a neural-network surrogate
for ray tracing, reinforcement learning for lens design, super-resolving
wavefront maps with a GAN, and inferring misalignments from wavefront data.
These are exploratory examples rather than production recipes.

.. toctree::
   :maxdepth: 1

   Tutorial 9a: Predicting Lens Performance (RMS Spot Size) Using Random Forest <examples/Tutorial_9a_Predicting_Lens_Performance_with_Random_Forest>
   Tutorial 9b: Classifying and Predicting Ray Path Failures with Machine Learning <examples/Tutorial_9b_Classifying_Ray_Path_Failures_with_Machine_Learning>
   Tutorial 9c: Building a Deep Learning Neural Network Surrogate for Double Gauss Ray Tracing <examples/Tutorial_9c_Deep_Learning_Surrogate_for_Double_Gauss_Ray_Tracing>
   Tutorial 9d: Optimizing Aspheric Singlet Lenses using Reinforcement Learning <examples/Tutorial_9d_Optimizing_Aspheric_Singlets_via_Reinforcement_Learning>
   Tutorial 9e: Wavefront Map Super-Resolution Using Generative Adversarial Networks (SR-GAN) <examples/Tutorial_9e_Wavefront_Super_Resolution_via_Generative_Adversarial_Networks>
   Tutorial 9f: Predicting Physical Lens Misalignments from Optical Spot Diagrams <examples/Tutorial_9f_Predicting_Physical_Lens_Misalignments_from_Spot_Diagrams>


10. Non-Sequential & Illumination :bdg-secondary:`elective`
-----------------------------------------------------------

*Prerequisites: chapter 1. The differentiable parts need PyTorch.*

Optiland's differentiable **non-sequential (NSQ)** engine handles illumination
design, stray light, ghost analysis and non-imaging optics, where light moves
freely through a 3-D scene instead of following a fixed list of surfaces.
Start with the on-ramp tutorial below, then continue with the 11-notebook
gallery deep dive (sources, components, detectors, scattering, diagnostics,
multi-source illumination, stray light, reflective systems, advanced topics
and differentiable optimization).

.. toctree::
   :maxdepth: 1

   Tutorial 10a: Non-Sequential & Illumination <examples/Tutorial_10a_Non_Sequential_and_Illumination>

* :ref:`NSQ gallery deep dive (notebooks 01-11) <gallery_nonsequential>`
* :ref:`NSQ limitations and roadmap <nsq_limitations_and_roadmap>`


11. Manufacturing & Fabrication Drawings :bdg-secondary:`elective`
------------------------------------------------------------------

*Prerequisites: chapter 1; Tutorial 6a helps with the tolerance annotations.*

Turn a finished design into ISO 10110 fabrication drawings (DXF and PDF) with
per-surface and per-element tolerance annotations, ready to send to an optics
shop.

.. toctree::
   :maxdepth: 1

   Tutorial 11a: ISO 10110 Fabrication Drawings <examples/Tutorial_11a_ISO_10110_Drawings>


Community resources
-------------------

`Computational Optics <https://inspiration-overflow.github.io/computational-optics/>`_
is a free, open-source (MIT-licensed) textbook by community contributor
`goldengrape <https://github.com/goldengrape>`_. It builds up computational
optics from first principles (ray representation, refraction, paraxial
theory, OPD and wavefront error, PSF/OTF/MTF, and a complete Cooke triplet
worked example), using Optiland for validation and worked examples
throughout. It is available in English and Chinese. Its companion code pins a
specific Optiland version, so some API details may differ from the latest
release; see its `Companion Code
<https://inspiration-overflow.github.io/computational-optics/companion-code.html>`_
page for details.

Built something with Optiland that would help others learn? Tell us in
`GitHub Discussions <https://github.com/optiland/optiland/discussions>`_. We
are always happy to link to community material.
