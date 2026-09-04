.. _learning_guide:
.. _learning_guide_page:

Learning Guide
==============

This guide is Optiland's primary learning path. Tutorials are grouped thematically and follow a
progressive arc where concepts from earlier sections inform later ones. Each notebook is
self-contained and runnable; you do not need to execute prior notebooks to run any given one.
New to Optiland? Start with :ref:`start_here` to find the path that fits your goals.


1. Foundational Lens Design
---------------------------

.. toctree::
   :maxdepth: 1

   Tutorial 1a: Optiland for Beginners <examples/Tutorial_1a_Optiland_for_Beginners>
   Tutorial 1b: Lens Properties and Prescription <examples/Tutorial_1b_Lens_Properties_and_Prescription>
   Tutorial 1c: Material Database and Catalogs <examples/Tutorial_1c_Material_Database_and_Catalogs>
   Tutorial 1d: Saving and Loading <examples/Tutorial_1d_Saving_and_Loading>
   Tutorial 1e: Design a Doublet, End to End <examples/Tutorial_1e_Design_a_Doublet_End_to_End>


2. Real Raytracing & Analysis
-----------------------------

.. toctree::
   :maxdepth: 1

   Tutorial 2a: Tracing and Analyzing Rays <examples/Tutorial_2a_Tracing_and_Analyzing_Rays>
   Tutorial 2b: Monte Carlo Raytracing <examples/Tutorial_2b_Monte_Carlo_Raytracing>
   Tutorial 2c: Aberration Analyses <examples/Tutorial_2c_Aberration_Analyses>
   Tutorial 2d: OPD, PSF, and MTF Calculations <examples/Tutorial_2d_OPD_PSF_and_MTF_Calculations>


3. Lens Optimization
--------------------

.. toctree::
   :maxdepth: 1

   Tutorial 3a: Simple Optimization <examples/Tutorial_3a_Simple_Optimization>
   Tutorial 3b: Advanced Optimization <examples/Tutorial_3b_Advanced_Optimization>
   Tutorial 3c: User-Defined Optimization <examples/Tutorial_3c_User_Defined_Optimization>
   Tutorial 3d: Optimization Case Study: Cooke Triplet <examples/Tutorial_3d_Optimization_Case_Study_Cooke_Triplet>
   Tutorial 3e: Glass Expert Categorical Optimization <examples/Tutorial_3e_Glass_Expert_Categorical_Optimization>


4. Off-Axis & Complex Systems
-----------------------------

.. toctree::
   :maxdepth: 1

   Tutorial 4a: Tilts, Decenters, and Asymmetric Systems <examples/Tutorial_4a_Tilts_Decenters_and_Asymmetric_Systems>
   Tutorial 4b: Raytracing Aspheres and Freeforms <examples/Tutorial_4b_Raytracing_Aspheres_and_Freeforms>
   Tutorial 4c: Zoom Lenses and Multi-Configuration <examples/Tutorial_4c_Zoom_Lenses_and_Multi_Configuration>
   Tutorial 4d: Lens Catalogue Integration <examples/Tutorial_4d_Lens_Catalogue_Integration>
   Tutorial 4e: Lithographic Projection System <examples/Tutorial_4e_Lithographic_Projection_System>
   Tutorial 4f: Three-Mirror Anastigmat <examples/Tutorial_4f_Three_Mirror_Anastigmat>


5. Polarization & Coatings
--------------------------

.. toctree::
   :maxdepth: 1

   Tutorial 5a: Coatings and Multilayer Stacks <examples/Tutorial_5a_Coatings_and_Multilayer_Stacks>
   Tutorial 5b: Introduction to Polarization <examples/Tutorial_5b_Introduction_to_Polarization>
   Tutorial 5c: Thin-Film Optimization and Needle Synthesis <examples/Tutorial_5c_Thin_Film_Optimization_and_Needle_Synthesis>
   Tutorial 5d: Advanced Thin-Film Applications <examples/Tutorial_5d_Advanced_Thin_Film_Applications>


6. Tolerancing & Physical Effects
---------------------------------

.. toctree::
   :maxdepth: 1

   Tutorial 6a: Tolerancing Sensitivity Analysis <examples/Tutorial_6a_Tolerancing_Sensitivity_Analysis>
   Tutorial 6b: Monte Carlo Tolerancing Analysis <examples/Tutorial_6b_Monte_Carlo_Tolerancing_Analysis>
   Tutorial 6c: Roughness, Scattering, and Extended Sources <examples/Tutorial_6c_Roughness_Scattering_and_Extended_Sources>


7. Differentiable Raytracing
----------------------------

.. toctree::
   :maxdepth: 1

   Tutorial 7a: Differentiable Ray Tracing Hello World <examples/Tutorial_7a_Differentiable_Ray_Tracing_Hello_World>
   Tutorial 7b: Differentiable Lens Optimization <examples/Tutorial_7b_Differentiable_Lens_Optimization>


8. Extending Optiland
---------------------

.. toctree::
   :maxdepth: 1

   Tutorial 8a: Custom Surface Types <examples/Tutorial_8a_Custom_Surface_Types>
   Tutorial 8b: Custom Coating Types <examples/Tutorial_8b_Custom_Coating_Types>
   Tutorial 8c: Custom Optimization Algorithm <examples/Tutorial_8c_Custom_Optimization_Algorithm>


9. Machine Learning in Optical Design
-------------------------------------

These examples demonstrate how Optiland can be used in conjunction with machine and deep learning to solve complex optical design problems, showing neural network surrogates, classification models, generative adversarial networks (GANs), and reinforcement learning workflows.

.. toctree::
   :maxdepth: 1

   Tutorial 9a: Predicting Lens Performance (RMS Spot Size) Using Random Forest <examples/Tutorial_9a_Predicting_Lens_Performance_with_Random_Forest>
   Tutorial 9b: Classifying and Predicting Ray Path Failures with Machine Learning <examples/Tutorial_9b_Classifying_Ray_Path_Failures_with_Machine_Learning>
   Tutorial 9c: Building a Deep Learning Neural Network Surrogate for Double Gauss Ray Tracing <examples/Tutorial_9c_Deep_Learning_Surrogate_for_Double_Gauss_Ray_Tracing>
   Tutorial 9d: Optimizing Aspheric Singlet Lenses using Reinforcement Learning <examples/Tutorial_9d_Optimizing_Aspheric_Singlets_via_Reinforcement_Learning>
   Tutorial 9e: Wavefront Map Super-Resolution Using Generative Adversarial Networks (SR-GAN) <examples/Tutorial_9e_Wavefront_Super_Resolution_via_Generative_Adversarial_Networks>
   Tutorial 9f: Predicting Physical Lens Misalignments from Optical Spot Diagrams <examples/Tutorial_9f_Predicting_Physical_Lens_Misalignments_from_Spot_Diagrams>


10. Non-Sequential & Illumination
---------------------------------

Optiland's differentiable **non-sequential (NSQ)** engine handles illumination design,
stray-light and ghost analysis, and non-imaging optics - where light propagates freely
through a 3-D scene rather than a fixed surface sequence. Start with the numbered on-ramp
below, then continue into the 11-notebook gallery deep dive (sources, components,
detectors, scattering, diagnostics, multi-source illumination, stray light, reflective
systems, advanced topics, and differentiable optimization).

.. toctree::
   :maxdepth: 1

   Tutorial 10a: Non-Sequential & Illumination <examples/Tutorial_10a_Non_Sequential_and_Illumination>

* :ref:`NSQ gallery deep dive (notebooks 01-11) <gallery_nonsequential>`
* :ref:`NSQ Limitations & Roadmap <nsq_limitations_and_roadmap>`

Community Resources
--------------------

`Computational Optics <https://inspiration-overflow.github.io/computational-optics/>`_ is a free,
open-source (MIT-licensed) textbook by community contributor `goldengrape <https://github.com/goldengrape>`_
that builds up computational optics from first principles (ray representation, refraction, paraxial
theory, OPD/wavefront error, PSF/OTF/MTF, and a complete Cooke Triplet worked example) using Optiland
for validation and worked examples throughout. It is available in English and Chinese. Note that its
companion code pins a specific Optiland version, so API details may drift slightly from the latest
release; see its `Companion Code <https://inspiration-overflow.github.io/computational-optics/companion-code.html>`_
page for details.
