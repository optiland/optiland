.. _user_guide:

.. meta::
   :description: Install Optiland, follow the quickstart, and learn the conventions every Optiland model relies on.

Get Started
===========

New here? Welcome. The four steps below take you from an empty Python
environment to a lens you built, traced and optimized yourself. Most people get
through them in well under an hour, and everything after that is optional,
depending on what you want to do.

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: 1. Install
      :link: installation
      :link-type: doc

      ``pip install optiland``. Add PyTorch or the GUI later if you need them.

   .. grid-item-card:: 2. Quickstart
      :link: quickstart
      :link-type: doc

      Five minutes: build a lens, trace it, look at a spot diagram, optimize
      it and save it.

   .. grid-item-card:: 3. Conventions
      :link: conventions
      :link-type: doc

      One short page on how Optiland describes a lens: the "after" rule, sign
      conventions and units. It saves a lot of head-scratching later.

   .. grid-item-card:: 4. Pick a path
      :link: start_here
      :link-type: doc

      Suggested reading orders for students, optical engineers, computational
      researchers and contributors.

.. tip::

   Would rather not install anything yet? :doc:`try_it` runs Optiland in your
   browser.

Pages to keep open
------------------

Once you are up and running, these are the reference pages people come back to
most.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: How Do I …?
      :link: how_do_i
      :link-type: doc

      Organized by question rather than by feature. Find your task and follow
      the link to the notebook that shows how.

   .. grid-item-card:: Cheat Sheet
      :link: cheat_sheet
      :link-type: doc

      Copy-paste snippets for the most common operations.

   .. grid-item-card:: Glossary
      :link: glossary
      :link-type: doc

      Optical and Optiland-specific terms, defined in a sentence or two.

   .. grid-item-card:: Optimization Guide
      :link: optimization_guide
      :link-type: doc

      Choosing an optimizer, running it with ``minimize()``, following and
      cancelling a run, and keeping it fast.

   .. grid-item-card:: Merit Functions and Constraints
      :link: optimization_merit_constraints
      :link-type: doc

      Building the merit function: the default merit, operands, bounds and
      hard constraints.

   .. grid-item-card:: GUI Quickstart
      :link: gui_quickstart
      :link-type: doc

      Prefer clicking to typing? Explore and edit systems in the Optiland
      desktop GUI.

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   quickstart
   conventions
   start_here
   how_do_i
   cheat_sheet
   glossary
   optimization_guide
   optimization_merit_constraints
   gui_quickstart
