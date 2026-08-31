.. _optiland_docs_home:

.. meta::
   :description: Optiland documentation: open-source optical design, analysis and differentiable ray tracing in Python.

Optiland documentation
======================

.. div:: optiland-hero

   .. rst-class:: lead

   Open-source optical design, analysis and differentiable ray tracing in
   Python. Build, trace and optimize lens and mirror systems with a **NumPy**
   backend for everyday CPU work or a **PyTorch** backend for GPU acceleration
   and automatic differentiation.

   .. button-ref:: start_here
      :ref-type: doc
      :color: primary

      Get started

   .. button-ref:: installation
      :ref-type: doc
      :color: secondary
      :outline:

      Install

   .. button-ref:: gallery/introduction
      :ref-type: doc
      :color: secondary
      :outline:

      Explore examples

   .. button-ref:: api/api_introduction
      :ref-type: doc
      :color: secondary
      :outline:

      API reference

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      .. code-block:: console
         :caption: Install

         pip install optiland

      .. code-block:: python
         :caption: Two lines to a traced, rendered system

         from optiland.samples.objectives import ReverseTelephoto

         lens = ReverseTelephoto()
         lens.draw3D()

   .. grid-item::

      .. image:: images/telephoto.png
         :alt: 3D rendering of a reverse telephoto lens traced with Optiland
         :width: 100%

.. rst-class:: optiland-eyebrow

Find your path

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Student or first-time user
      :link: start_here.html#optics-student-first-timer
      :link-type: url

      Build and visualize your first lens in Python, then learn to read spot
      diagrams, ray fans and wavefront maps.

   .. grid-item-card:: Optical engineer
      :link: start_here.html#optical-engineer-practitioner
      :link-type: url

      Get productive fast: import catalog lenses, reproduce existing designs,
      and run optimization and tolerancing workflows.

   .. grid-item-card:: Computational researcher
      :link: start_here.html#computational-researcher
      :link-type: url

      Use the PyTorch backend for autograd, differentiable optimization and
      end-to-end machine-learning pipelines.

   .. grid-item-card:: Contributor or extender
      :link: start_here.html#software-contributor-extender
      :link-type: url

      Add surface types, coatings, analyses or operands, and understand the
      architecture behind them.

.. rst-class:: optiland-eyebrow

What Optiland does

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Modeling and ray tracing
      :link: functionalities.html#design-tools
      :link-type: url

      Sequential systems with spherical, conic, aspheric, freeform and
      diffractive surfaces; tilts, decenters and fold mirrors; paraxial, real
      and polarization-aware ray tracing.

   .. grid-item-card:: Analysis
      :link: functionalities.html#analysis-tools
      :link-type: url

      Spot diagrams, ray fans, distortion, field curvature, OPD, Zernike
      decomposition, PSF and MTF, encircled energy, image simulation and more.

   .. grid-item-card:: Optimization and tolerancing
      :link: functionalities.html#optimization-and-tolerancing
      :link-type: url

      Local and global optimizers, user-defined operands, Glass Expert
      categorical optimization, sensitivity and Monte Carlo tolerancing.

   .. grid-item-card:: Differentiable optics
      :link: gallery/differentiable_ray_tracing
      :link-type: doc

      A PyTorch backend that makes every trace differentiable: gradients, GPU
      acceleration and integration with deep-learning workflows.

   .. grid-item-card:: Non-sequential illumination
      :link: gallery/nonsequential
      :link-type: doc

      Illumination design, stray-light and ghost analysis with scattering,
      coatings, detectors and differentiable optimization.

   .. grid-item-card:: Extensibility and interoperability
      :link: developers_guide/extension_recipes
      :link-type: doc

      Custom surfaces, coatings, operands and analyses; Zemax, CODE V and
      OSLO import; vendor lens catalogs; a JSON file format; plugin packages.

.. admonition:: Know what you want to do?
   :class: tip

   The :doc:`how_do_i` page is organized by task rather than by feature: find
   your question, follow the link, copy the pattern.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Try Optiland in your browser
      :link: try_it
      :link-type: doc

      Run Optiland in an in-page Python kernel. Nothing to install.

   .. grid-item-card:: Learning Guide
      :link: learning_guide
      :link-type: doc

      The complete tutorial series, from foundational lens design to machine
      learning and non-sequential ray tracing.

.. note::

   You are reading the documentation for Optiland |release|, built
   continuously from the ``master`` branch of
   `optiland/optiland <https://github.com/optiland/optiland>`_.

.. toctree::
   :hidden:

   Get Started <user_guide>
   Learn <learn>
   Capabilities <functionalities>
   API Reference <api/api_introduction>
   Developer Guide <developers_guide/introduction>
   Contributing <contributing>
